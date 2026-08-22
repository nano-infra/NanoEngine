"""Kimi Delta Attention using FlashInfer's public recurrent KDA kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.layers import get_backend
from dlengine.runtime.layers.generic.gated_delta_net import GenericGatedDeltaNet


class SigmoidRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from dlengine.runtime.kernel.triton.generic.k3_output_norm import k3_output_norm

        return k3_output_norm(x, gate, self.weight, self.eps)


class FlashInferKDA(GenericGatedDeltaNet):
    """K3 KDA with attention-TP sharding and the shared linear-state pool.

    The inherited methods implement only causal depthwise convolution and
    state-slot hygiene. The recurrence itself is always FlashInfer KDA; there
    is no GDN/torch fallback.
    """

    def __init__(self, layer_idx: int, state_layer_idx: int, config) -> None:
        nn.Module.__init__(self)
        try:
            from flashinfer import recurrent_kda
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Kimi K3 requires FlashInfer with recurrent_kda support."
            ) from exc
        self._recurrent_kda = recurrent_kda
        try:
            from dlengine.runtime.kernel.triton.fla.kda import chunk_kda
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Kimi K3 prefill requires NanoDeploy's Triton chunk KDA "
                "kernel. No recurrent, GDN, torch, or naive fallback is allowed."
            ) from exc
        self._chunk_kda = chunk_kda
        self.config = config
        self.layer_idx = state_layer_idx
        self.model_layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        linear = config.linear_attn_config
        total_heads = int(linear["num_heads"])
        tp = get_dist_context().attn_tp_world_size
        if total_heads % tp:
            raise ValueError(
                f"K3 KDA heads={total_heads} not divisible by attn_tp={tp}"
            )
        self.num_k_heads = self.num_v_heads = total_heads // tp
        self.head_k_dim = int(linear["head_dim"])
        self.head_v_dim = int(config.v_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.kv_ratio = 1
        self.conv_kernel_size = int(linear["short_conv_kernel_size"])
        self.activation = "silu"
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.lower_bound = float(linear.get("gate_lower_bound", -5.0))
        tp_group = get_dist_context().attn_tp_group
        backend = get_backend()

        self.q_proj = backend.get_column_parallel_linear(
            self.hidden_size,
            total_heads * self.head_k_dim,
            bias=False,
            tp_group=tp_group,
        )
        self.k_proj = backend.get_column_parallel_linear(
            self.hidden_size,
            total_heads * self.head_k_dim,
            bias=False,
            tp_group=tp_group,
        )
        self.v_proj = backend.get_column_parallel_linear(
            self.hidden_size,
            total_heads * self.head_v_dim,
            bias=False,
            tp_group=tp_group,
        )
        self.g_proj = backend.get_column_parallel_linear(
            self.hidden_size,
            total_heads * self.head_k_dim,
            bias=False,
            tp_group=tp_group,
        )
        self.b_proj = backend.get_column_parallel_linear(
            self.hidden_size,
            total_heads,
            bias=False,
            tp_group=tp_group,
        )
        self.f_a_proj = backend.get_replicated_linear(
            self.hidden_size,
            self.head_k_dim,
            bias=False,
        )
        self.f_b_proj = backend.get_column_parallel_linear(
            self.head_k_dim,
            total_heads * self.head_k_dim,
            bias=False,
            tp_group=tp_group,
        )
        self.o_proj = backend.get_row_parallel_linear(
            total_heads * self.head_v_dim,
            self.hidden_size,
            bias=False,
            tp_group=tp_group,
        )

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            # The checkpoint and Blackwell KDA reference keep causal-conv
            # coefficients in fp32 even though projected activations are bf16.
            padding=self.conv_kernel_size - 1,
            dtype=torch.float32,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads * self.head_k_dim, dtype=torch.float32)
        )
        self.o_norm = SigmoidRMSNormGated(self.head_v_dim, float(config.rms_norm_eps))
        self._conv1d_prefill_padded_ws = None
        self.register_buffer("fused_a_beta_weight", None, persistent=False)
        self.register_buffer("fused_qkvg_weight", None, persistent=False)

    def prepare_fused_decode_projections(self) -> None:
        qkvg = torch.cat(
            (
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.g_proj.weight,
            ),
            dim=0,
        ).contiguous()
        offset = 0
        for proj, size in (
            (self.q_proj, self.key_dim),
            (self.k_proj, self.key_dim),
            (self.v_proj, self.value_dim),
            (self.g_proj, self.key_dim),
        ):
            proj.weight.data = qkvg[offset : offset + size]
            offset += size
        self.fused_qkvg_weight = qkvg

        width = self.head_k_dim + self.num_v_heads
        padded = (width + 15) // 16 * 16
        weight = self.f_a_proj.weight.new_zeros((padded, self.hidden_size))
        weight[: self.head_k_dim].copy_(self.f_a_proj.weight)
        weight[self.head_k_dim : width].copy_(self.b_proj.weight)
        self.fused_a_beta_weight = weight
        self.f_a_proj.weight.data = weight[: self.head_k_dim]
        self.b_proj.weight.data = weight[self.head_k_dim : width]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_batch_context()
        total_tokens = hidden_states.shape[0]
        if context.is_prefill:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            gate = self.g_proj(hidden_states)
            beta = self.b_proj(hidden_states)
            forget = self.f_b_proj(self.f_a_proj(hidden_states))
        else:
            if self.fused_a_beta_weight is None or self.fused_qkvg_weight is None:
                raise RuntimeError(
                    "K3 decode projection weights were not fused after loading"
                )
            from dlengine.runtime.kernel.cutedsl_bf16_gemm import cutedsl_bf16_gemm
            from dlengine.runtime.kernel.jit.sgl.tiny_gemm import (
                tiny_k_gemm_bf16,
                tiny_n_gemm_bf16,
            )

            qkvg = cutedsl_bf16_gemm(hidden_states, self.fused_qkvg_weight.detach())
            q, k, v, gate = qkvg.split(
                (self.key_dim, self.key_dim, self.value_dim, self.key_dim), dim=-1
            )
            mixed_qkv = qkvg[:, : self.conv_dim]
            fused_a_beta = tiny_n_gemm_bf16(
                hidden_states, self.fused_a_beta_weight, max_m=8
            )
            beta = fused_a_beta[:, self.head_k_dim : self.head_k_dim + self.num_v_heads]
            forget = tiny_k_gemm_bf16(
                fused_a_beta[:, : self.head_k_dim],
                self.f_b_proj.weight,
                max_m=8,
            )

        if context.is_prefill:
            self._zero_fresh_slots(context)
        if context.is_prefill:
            mixed_qkv = torch.cat((q, k, v), dim=-1)
        if context.is_prefill:
            qkv = self._apply_conv1d(mixed_qkv, context)
        else:
            from dlengine.runtime.kernel.triton.generic.k3_causal_conv import (
                k3_causal_conv_update,
            )

            conv_pool = context.gdn_conv_states[self.layer_idx]
            qkv = k3_causal_conv_update(
                mixed_qkv,
                conv_pool,
                self.conv1d.weight.squeeze(1),
                context.gdn_state_slots_i32[:total_tokens],
            )
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.view(total_tokens, self.num_k_heads, self.head_k_dim)
        k = k.view(total_tokens, self.num_k_heads, self.head_k_dim)
        v = v.view(total_tokens, self.num_v_heads, self.head_v_dim)
        raw_g = forget.view(total_tokens, self.num_v_heads, self.head_k_dim)
        beta_logits = beta.view(total_tokens, self.num_v_heads)

        states = context.gdn_recurrent_states
        slots = context.gdn_state_slots_i32
        if states is None or slots is None:
            raise RuntimeError("K3 KDA requires allocated linear-attention state slots")
        pool = states[self.layer_idx]
        if context.is_prefill:
            beta = beta_logits.float().sigmoid()
            cu = context.cu_seqlens_q.to(torch.int32)
            batch = cu.numel() - 1
            indices = slots[:batch]
            # FlashInfer recurrent_kda is decode-only in this release. K3's
            # safe gate is handled by Triton chunk_kda (the Blackwell CuTe
            # chunk kernel does not support lower_bound yet).
            out = self._chunk_kda(
                q.unsqueeze(0).contiguous(),
                k.unsqueeze(0).contiguous(),
                v.unsqueeze(0).contiguous(),
                raw_g.unsqueeze(0).contiguous(),
                beta.unsqueeze(0).contiguous(),
                initial_state=pool,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                A_log=self.A_log.reshape(-1).float().contiguous(),
                dt_bias=self.dt_bias.reshape(-1).float().contiguous(),
                lower_bound=self.lower_bound,
            )
            out = out.squeeze(0)
        else:
            batch = total_tokens
            indices = slots[:batch]
            from dlengine.runtime.kernel.triton.fla.fused_recurrent import (
                fused_recurrent_kda_packed_decode,
            )

            out = torch.empty(
                batch,
                1,
                self.num_v_heads,
                self.head_v_dim,
                device=q.device,
                dtype=q.dtype,
            )
            fused_recurrent_kda_packed_decode(
                mixed_qkv=qkv,
                a=raw_g.flatten(1).contiguous(),
                b=beta_logits,
                A_log=self.A_log.reshape(-1).float().contiguous(),
                dt_bias=self.dt_bias.reshape(-1).float().contiguous(),
                scale=self.head_k_dim**-0.5,
                initial_state=pool,
                out=out,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                lower_bound=self.lower_bound,
            )
            out = out.squeeze(1)
        out = out.reshape(total_tokens, self.num_v_heads, self.head_v_dim)
        gate = gate.view(total_tokens, self.num_v_heads, self.head_k_dim)
        out = self.o_norm(out, gate).reshape(total_tokens, self.value_dim)
        return self.o_proj(out)

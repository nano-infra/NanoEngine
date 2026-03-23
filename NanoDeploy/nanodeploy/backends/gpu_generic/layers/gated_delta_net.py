"""Generic GatedDeltaNet (Linear Attention) layer implementation.

Provides the linear attention mechanism used in Qwen3.5 MoE.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends import get_backend
from nanodeploy.backends.base_backend import GatedDeltaNetBase, ReplicatedLinearBase
from nanodeploy.context.context import get_context
from nanodeploy.logging import get_logger
from nanodeploy.models.quant_config import QuantizationConfig

logger = get_logger()

# Try to import flash-linear-attention kernels at module level
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False
    logger.warning(
        "flash-linear-attention not installed. GatedDeltaNet will use naive fallback."
    )

# Try to import causal_conv1d for optimized depthwise conv
try:
    from causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
        causal_conv1d_varlen_states,
    )

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU-gated multiplication.

    Applied per-head: weight has shape [head_v_dim].
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size] — the value to normalize
            gate: [..., hidden_size] — gating signal (SiLU applied)
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(input_dtype))
        return x


class GenericGatedDeltaNet(GatedDeltaNetBase):
    """GatedDeltaNet linear attention.

    Uses flash-linear-attention kernels:
    - Prefill: chunk_gated_delta_rule
    - Decode: fused_recurrent_gated_delta_rule
    """

    def __init__(
        self,
        layer_idx: int,
        config,
        quantization_config: QuantizationConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # 4096
        self.num_k_heads = config.linear_num_key_heads  # 16
        self.num_v_heads = config.linear_num_value_heads  # 64
        self.head_k_dim = config.linear_key_head_dim  # 128
        self.head_v_dim = config.linear_value_head_dim  # 128
        self.key_dim = self.num_k_heads * self.head_k_dim  # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 8192
        self.kv_ratio = self.num_v_heads // self.num_k_heads  # 4

        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4
        self.activation = config.hidden_act  # "silu"

        # Conv1d (depthwise, on full QKV concatenation)
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 12288
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Fused QKV projection
        self.in_proj_qkv: ReplicatedLinearBase = get_backend().get_replicated_linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
        )

        # Z gating projection
        self.in_proj_z: ReplicatedLinearBase = get_backend().get_replicated_linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )

        # Output projection (replicated: GDN computes same output on all TP ranks)
        self.out_proj: ReplicatedLinearBase = get_backend().get_replicated_linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )

        # Alpha/Beta projections (NOT quantized, small)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # State parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))

        # Output normalization: RMSNorm with SiLU gating
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # Kernel references
        self._chunk_fn = chunk_gated_delta_rule if _HAS_FLA else None
        self._recurrent_fn = fused_recurrent_gated_delta_rule if _HAS_FLA else None

        # Precompute A
        self._A_exp_cache = None

    def _get_A_exp(self):
        """Cached -exp(A_log)."""
        if self._A_exp_cache is None:
            self._A_exp_cache = -self.A_log.float().exp()
        return self._A_exp_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [total_tokens, hidden_size]
        """
        context = get_context()
        total_tokens = hidden_states.shape[0]

        # 1. Input projections (fused QKV)
        qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        # 2. Causal Conv1d on fused QKV
        qkv = self._apply_conv1d(qkv, context)

        # 3. Split back to Q, K, V (after conv)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # 4. Reshape Q, K, V
        q = q.view(total_tokens, self.num_k_heads, self.head_k_dim)
        k = k.view(total_tokens, self.num_k_heads, self.head_k_dim)
        v = v.view(total_tokens, self.num_v_heads, self.head_v_dim)

        # 5. Compute beta and gating
        beta = b.sigmoid()  # [T, num_v_heads]
        g = self._get_A_exp() * F.softplus(a.float() + self.dt_bias)  # [T, num_v_heads]

        # 6. Expand q, k for GVA
        if self.kv_ratio > 1:
            q = q.repeat_interleave(self.kv_ratio, dim=1)
            k = k.repeat_interleave(self.kv_ratio, dim=1)

        # 7. Apply flash-linear-attention kernel
        scale = self.head_k_dim**-0.5
        core_attn_out = self._apply_gdn(q, k, v, g, beta, scale, context)

        # 8. Apply gated RMSNorm
        z = z.view(total_tokens, self.num_v_heads, self.head_v_dim)
        out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)

        out = self.norm(out, z_flat)

        out = out.view(total_tokens, self.num_v_heads, self.head_v_dim)

        # 9. Output projection
        out = out.reshape(total_tokens, self.value_dim)
        output = self.out_proj(out)
        return output

    def _apply_conv1d(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Apply causal conv1d to concatenated QKV.

        Args:
            qkv: [total_tokens, conv_dim]
        """
        if context.is_prefill:
            if _HAS_CAUSAL_CONV1D:
                return self._conv1d_prefill_fast(qkv, context)
            else:
                return self._conv1d_prefill_naive(qkv, context)
        else:
            return self._conv1d_decode(qkv, context)

    def _conv1d_prefill_fast(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Prefill conv1d using causal_conv1d_fn with seq_idx (single batched kernel)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        conv_weight = self.conv1d.weight.squeeze(1)

        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)

        has_prev_state = (
            context.block_tables is not None and gdn_conv_states is not None
        )

        if has_prev_state:
            # Chunked prefill (chunks 2+): per-sequence with initial_states
            # from the previous chunk's stored conv state.
            #
            # TODO(perf): This per-seq Python loop is a bottleneck for large
            # batches.  causal_conv1d_fn supports seq_idx for batch separation
            # but NOT initial_states + seq_idx together.  Two options:
            #   1. Upstream batched initial_states support to causal_conv1d_fn.
            #   2. Write a custom Triton kernel for causal conv1d with
            #      per-seq initial states (more work, but fully controlled).
            # For now this is acceptable since GDN is used in hybrid attention
            # where prefill batch sizes are typically small.
            qkv_out = torch.empty_like(qkv)
            for i in range(num_seqs):
                start = int(cu_seqlens[i].item())
                end = int(cu_seqlens[i + 1].item())
                if end <= start:
                    continue
                slot = (
                    int(gdn_state_slots[i].item()) if gdn_state_slots is not None else i
                )
                init_state = gdn_conv_states[
                    self.layer_idx, slot : slot + 1, :, 1:
                ].contiguous()
                seq_out = (
                    causal_conv1d_fn(
                        x=qkv[start:end].T.unsqueeze(0),
                        weight=conv_weight,
                        initial_states=init_state,
                        activation=self.activation,
                    )
                    .squeeze(0)
                    .T
                )
                qkv_out[start:end] = seq_out
        else:
            # First chunk or single-chunk: batched with seq_idx
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
            seq_idx = torch.repeat_interleave(
                torch.arange(num_seqs, dtype=torch.int32, device=qkv.device),
                seq_lens,
            ).unsqueeze(0)

            qkv_out = (
                causal_conv1d_fn(
                    x=qkv.T.unsqueeze(0),
                    weight=conv_weight,
                    bias=None,
                    seq_idx=seq_idx,
                    activation=self.activation,
                )
                .squeeze(0)
                .T
            )

        # Store conv states for future chunks/decode — batched extraction
        if gdn_conv_states is not None:
            states = causal_conv1d_varlen_states(
                qkv, cu_seqlens, self.conv_kernel_size - 1
            )
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:num_seqs]
                gdn_conv_states[self.layer_idx, slots, :, 0] = 0
                gdn_conv_states[self.layer_idx, slots, :, 1:] = states
            else:
                gdn_conv_states[self.layer_idx, :num_seqs, :, 0] = 0
                gdn_conv_states[self.layer_idx, :num_seqs, :, 1:] = states

        return qkv_out

    def _conv1d_prefill_naive(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Prefill conv1d using PyTorch (per-sequence, fallback)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        qkv_out = torch.empty_like(qkv)

        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        has_prev_state = (
            context.block_tables is not None and gdn_conv_states is not None
        )

        conv_weight = self.conv1d.weight  # [conv_dim, 1, kernel_size]

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end <= start:
                continue
            seq_qkv = qkv[start:end].unsqueeze(0).transpose(1, 2)  # [1, D, L]

            if has_prev_state:
                slot = gdn_state_slots[i].item() if gdn_state_slots is not None else i
                prev = gdn_conv_states[self.layer_idx, slot, :, 1:].unsqueeze(
                    0
                )  # [1, D, k-1]
                padded = torch.cat([prev, seq_qkv], dim=2)  # [1, D, k-1+L]
                seq_out = F.silu(
                    F.conv1d(padded, conv_weight, groups=self.conv_dim)
                )  # [1, D, L]
            else:
                seq_out = F.silu(self.conv1d(seq_qkv)[:, :, : end - start])

            qkv_out[start:end] = seq_out.squeeze(0).transpose(0, 1)

        # Store conv state
        if gdn_conv_states is not None:
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                if end <= start:
                    continue
                seq_len = end - start
                pad_len = min(seq_len, self.conv_kernel_size - 1)
                slot = gdn_state_slots[i].item() if gdn_state_slots is not None else i
                gdn_conv_states[self.layer_idx, slot, :, :] = 0
                gdn_conv_states[self.layer_idx, slot, :, -pad_len:] = qkv[
                    end - pad_len : end
                ].T

        return qkv_out

    def _conv1d_decode(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Decode conv1d: single token per sequence, update conv state."""
        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        bs = qkv.shape[0]

        if gdn_conv_states is not None and _HAS_CAUSAL_CONV1D:
            conv_weight = self.conv1d.weight.squeeze(1)
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:bs]
                conv_state = gdn_conv_states[self.layer_idx, slots]
            else:
                conv_state = gdn_conv_states[self.layer_idx, :bs]
            qkv_out = causal_conv1d_update(
                qkv,
                conv_state,
                conv_weight,
                bias=None,
                activation=self.activation,
            )
            if gdn_state_slots is not None:
                gdn_conv_states[self.layer_idx, slots] = conv_state
            return qkv_out
        elif gdn_conv_states is not None:
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:bs]
                conv_state = gdn_conv_states[self.layer_idx, slots].clone()
            else:
                conv_state = gdn_conv_states[self.layer_idx, :bs]
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = qkv
            conv_weight = self.conv1d.weight.squeeze(1)
            qkv_out = F.silu((conv_state * conv_weight.unsqueeze(0)).sum(-1))
            if gdn_state_slots is not None:
                gdn_conv_states[self.layer_idx, slots] = conv_state
            return qkv_out
        else:
            return F.silu(qkv)

    def _apply_gdn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        context,
    ) -> torch.Tensor:
        """Apply GatedDeltaNet attention (chunk or recurrent)."""
        if context.is_prefill:
            return self._gdn_prefill(q, k, v, g, beta, scale, context)
        else:
            return self._gdn_decode(q, k, v, g, beta, scale, context)

    def _gdn_prefill(self, q, k, v, g, beta, scale, context) -> torch.Tensor:
        """Prefill: chunk mode."""
        cu_seqlens = context.cu_seqlens_q.long()
        num_seqs = cu_seqlens.shape[0] - 1

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        initial_state = None
        if gdn_recurrent_states is not None:
            if context.block_tables is not None:
                # Chunked prefill (chunks 2+): load state from previous chunk
                if gdn_state_slots is not None:
                    initial_state = gdn_recurrent_states[
                        self.layer_idx, gdn_state_slots[:num_seqs]
                    ]
                else:
                    initial_state = gdn_recurrent_states[self.layer_idx, :num_seqs]
            else:
                initial_state = gdn_recurrent_states.new_zeros(
                    num_seqs, self.num_v_heads, self.head_k_dim, self.head_v_dim
                )

        if self._chunk_fn is not None:
            o, final_state = self._chunk_fn(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            o = o.squeeze(0)
        else:
            o, final_state = self._naive_gdn_prefill(
                q, k, v, g, beta, scale, cu_seqlens, initial_state
            )

        if gdn_recurrent_states is not None and final_state is not None:
            if gdn_state_slots is not None:
                gdn_recurrent_states[self.layer_idx, gdn_state_slots[:num_seqs]] = (
                    final_state
                )
            else:
                gdn_recurrent_states[self.layer_idx, :num_seqs] = final_state

        return o

    def _gdn_decode(self, q, k, v, g, beta, scale, context) -> torch.Tensor:
        """Decode: recurrent mode (single step)."""
        bs = q.shape[0]

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        if gdn_recurrent_states is not None:
            if gdn_state_slots is not None:
                initial_state = gdn_recurrent_states[
                    self.layer_idx, gdn_state_slots[:bs]
                ]  # gather
            else:
                initial_state = gdn_recurrent_states[self.layer_idx, :bs]
        else:
            initial_state = q.new_zeros(
                bs, self.num_v_heads, self.head_k_dim, self.head_v_dim
            )

        q_r = q.unsqueeze(1)
        k_r = k.unsqueeze(1)
        v_r = v.unsqueeze(1)
        g_r = g.unsqueeze(1)
        beta_r = beta.unsqueeze(1)

        if self._recurrent_fn is not None:
            o, final_state = self._recurrent_fn(
                q_r,
                k_r,
                v_r,
                g=g_r,
                beta=beta_r,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o = o.squeeze(1)
            if gdn_recurrent_states is not None and final_state is not None:
                if gdn_state_slots is not None:
                    gdn_recurrent_states[self.layer_idx, gdn_state_slots[:bs]] = (
                        final_state
                    )
                else:
                    gdn_recurrent_states[self.layer_idx, :bs] = final_state
        else:
            o, updated_state = self._naive_gdn_decode(
                q, k, v, g, beta, scale, initial_state
            )
            if gdn_recurrent_states is not None:
                if gdn_state_slots is not None:
                    gdn_recurrent_states[self.layer_idx, gdn_state_slots[:bs]] = (
                        updated_state
                    )
                else:
                    gdn_recurrent_states[self.layer_idx, :bs] = updated_state

        return o

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalization matching FLA's l2norm."""
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def _naive_gdn_prefill(
        self, q, k, v, g, beta, scale, cu_seqlens, initial_state=None
    ):
        """Naive sequential scan for prefill."""
        num_seqs = cu_seqlens.shape[0] - 1
        outputs = []
        final_states = []

        q = self._l2norm(q.float(), dim=-1)
        k = self._l2norm(k.float(), dim=-1)
        q = q * scale

        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()

            if initial_state is not None:
                S = initial_state[i].float().clone()
            else:
                S = torch.zeros(
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    dtype=torch.float32,
                    device=q.device,
                )

            if end <= start:
                final_states.append(S)
                continue
            qi, ki, vi = q[start:end], k[start:end], v[start:end]
            gi, bi = g[start:end], beta[start:end]

            out_seq = []
            for t in range(end - start):
                qt = qi[t]
                kt = ki[t]
                vt = vi[t].float()
                bt = bi[t].float()
                gt = gi[t].float()

                decay = gt.exp().unsqueeze(-1).unsqueeze(-1)
                S = decay * S

                kv = torch.einsum("hk,hv->hkv", kt, vt)
                Sk = torch.einsum("hkv,hk->hv", S, kt)
                correction = torch.einsum("hk,hv->hkv", kt, Sk)
                S = S + bt.unsqueeze(-1).unsqueeze(-1) * (kv - correction)

                out_t = torch.einsum("hk,hkv->hv", qt, S)
                out_seq.append(out_t)

            out_seq = torch.stack(out_seq, dim=0).to(v.dtype)
            outputs.append(out_seq)
            final_states.append(S)

        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = v.new_zeros(0, self.num_v_heads, self.head_v_dim)

        final_state = torch.stack(final_states, dim=0)
        return output, final_state

    def _naive_gdn_decode(self, q, k, v, g, beta, scale, state):
        """Naive recurrent step for decode."""
        q = self._l2norm(q.float(), dim=-1) * scale
        k = self._l2norm(k.float(), dim=-1)
        state_f32 = state.float()

        decay = g.float().exp().unsqueeze(-1).unsqueeze(-1)
        state_f32.mul_(decay)

        kv = torch.einsum("bhk,bhv->bhkv", k, v.float())
        Sk = torch.einsum("bhkv,bhk->bhv", state_f32, k)
        correction = torch.einsum("bhk,bhv->bhkv", k, Sk)
        state_f32.add_(beta.float().unsqueeze(-1).unsqueeze(-1) * (kv - correction))

        out = torch.einsum("bhk,bhkv->bhv", q, state_f32)

        return out.to(v.dtype), state_f32

"""Qwen3.5-MoE model implementation for NanoDeploy.

Supports mixed attention (full_attention GQA + GatedDeltaNet linear_attention)
and sparse MoE with shared expert.

Key design decisions (Phase 1):
  - attention_tp = 1 (no tensor parallelism for now)
  - flash-linear-attention kernels for GatedDeltaNet
  - Fixed-size state buffers for GDN conv & recurrent states
  - Text-only (no vision module)
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nanodeploy.context.context import get_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.attention import Attention
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.logging import get_logger
from nanodeploy.worker.runner_config import get_runner_config
from torch import nn

from ..quant_config import QuantizationConfig

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
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    _HAS_CAUSAL_CONV1D = True
except ImportError:
    _HAS_CAUSAL_CONV1D = False


# ---------------------------------------------------------------------------
# RMSNorm with SiLU gating (for GatedDeltaNet output norm)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Full Attention (GQA with partial RoPE + attention output gate)
# ---------------------------------------------------------------------------
class Qwen3_5MoeFullAttention(nn.Module):
    """Full attention with partial RoPE and output gating.

    Used for ~25% of layers (every 4th layer).
    The q_proj output is doubled: [q, gate] interleaved per-head.
    attn_output = o_proj(attn(q, k, v) * sigmoid(gate))
    """

    def __init__(
        self,
        layer_idx: int,
        config,
        quantization_config: QuantizationConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config
        self.layer_idx = layer_idx

        tp_size = get_dist_context().attn_tp_world_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads  # 32
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 32 for tp=1
        self.total_num_kv_heads = config.num_key_value_heads  # 2
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 2 for tp=1
        self.head_dim = config.head_dim  # 256
        self.q_size = self.num_heads * self.head_dim  # 8192
        self.kv_size = self.num_kv_heads * self.head_dim  # 512
        self.scaling = self.head_dim**-0.5

        # Attention output gate: q_proj outputs 2x (interleaved q and gate per-head)
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        # Partial rotary factor
        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)  # 64
        rope_theta = rope_params.get("rope_theta", 10000000.0)

        # QKV projection (packed)
        # When attn_output_gate=True, q_proj output is doubled (includes gate)
        q_heads_for_proj = self.total_num_heads * (1 + int(self.attn_output_gate))
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            q_heads_for_proj,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quantization_config=quantization_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quantization_config=quantization_config,
        )

        # RoPE: use rotary_dim as head_size for the rotary embedding
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=getattr(config, "max_position_embeddings", 262144),
            base=rope_theta,
            rope_scaling=None,  # rope_type='default' means no special scaling
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.head_dim,
            "GQA",
        )

        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            # q_gate portion: q_size * 2
            # k portion: kv_size
            # v portion: kv_size
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # Split q and gate: interleaved per-head [num_heads, head_dim*2]
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = q_gate.chunk(2, dim=-1)  # each [-1, num_heads, head_dim]
            gate = gate.reshape(-1, self.q_size)  # [-1, q_size]
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Partial RoPE: apply to first rotary_dim, pass through rest
        if self.rotary_dim < self.head_dim:
            q_rot = q[..., : self.rotary_dim].contiguous()
            q_pass = q[..., self.rotary_dim :]
            k_rot = k[..., : self.rotary_dim].contiguous()
            k_pass = k[..., self.rotary_dim :]
            q_rot, k_rot = self.rotary_emb(positions, q_rot, k_rot)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)

        # Apply output gate
        attn_output = o.flatten(1, -1)  # [-1, q_size]
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        output = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# GatedDeltaNet (Linear Attention)
# ---------------------------------------------------------------------------
class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention.

    Uses flash-linear-attention kernels:
    - Prefill: chunk_gated_delta_rule
    - Decode: fused_recurrent_gated_delta_rule

    Weight structure (from checkpoint, prefix = linear_attn):
        in_proj_qkv.weight  [key_dim*2+value_dim, hidden_size]  fp8  (fused QKV)
        in_proj_z.weight    [value_dim, hidden_size]             fp8
        out_proj.weight     [hidden_size, value_dim]             fp8
        in_proj_a.weight    [num_v_heads, hidden_size]           bf16
        in_proj_b.weight    [num_v_heads, hidden_size]           bf16
        conv1d.weight       [conv_dim, 1, kernel_size]           bf16
        A_log               [num_v_heads]                        bf16
        dt_bias             [num_v_heads]                        bf16
        norm.weight         [head_v_dim]                         fp32
    """

    def __init__(
        self,
        layer_idx: int,
        config,
        quantization_config: QuantizationConfig,
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

        # Fused QKV projection (matches checkpoint: in_proj_qkv)
        # Output: [T, key_dim + key_dim + value_dim] = [T, 12288]
        self.in_proj_qkv = ReplicatedLinear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,  # 2048 + 2048 + 8192 = 12288
            bias=False,
            quantization_config=quantization_config,
        )

        # Z gating projection (fp8 quantized)
        self.in_proj_z = ReplicatedLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quantization_config=quantization_config,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quantization_config=quantization_config,
        )

        # Alpha/Beta projections (NOT quantized, small)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # State parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))

        # Output normalization: RMSNorm with SiLU gating (per head_v_dim)
        # NOTE: q_norm/k_norm are NOT used in GatedDeltaNet (checkpoint has none)
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
        qkv = self.in_proj_qkv(hidden_states)  # [T, key_dim*2 + value_dim]

        z = self.in_proj_z(hidden_states)  # [T, value_dim]
        a = self.in_proj_a(hidden_states)  # [T, num_v_heads]
        b = self.in_proj_b(hidden_states)  # [T, num_v_heads]

        # 2. Causal Conv1d on fused QKV
        qkv = self._apply_conv1d(qkv, context)

        # 3. Split back to Q, K, V (after conv)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # 4. Reshape Q, K, V (no q/k norm in GatedDeltaNet, unlike self_attn)
        q = q.view(total_tokens, self.num_k_heads, self.head_k_dim)
        k = k.view(total_tokens, self.num_k_heads, self.head_k_dim)
        v = v.view(total_tokens, self.num_v_heads, self.head_v_dim)

        # 5. Compute beta and gating
        beta = b.sigmoid()  # [T, num_v_heads]
        g = self._get_A_exp() * F.softplus(a.float() + self.dt_bias)  # [T, num_v_heads]

        # 6. Expand q, k for GVA (num_k_heads -> num_v_heads)
        if self.kv_ratio > 1:
            q = q.repeat_interleave(
                self.kv_ratio, dim=1
            )  # [T, num_v_heads, head_k_dim]
            k = k.repeat_interleave(self.kv_ratio, dim=1)

        # 7. Apply flash-linear-attention kernel
        scale = self.head_k_dim**-0.5
        core_attn_out = self._apply_gdn(q, k, v, g, beta, scale, context)

        # 8. Apply gated RMSNorm (per-head)
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
        """Prefill conv1d using causal_conv1d_fn (per-sequence)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        qkv_out = torch.empty_like(qkv)

        conv_weight = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel_size]

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end <= start:
                continue
            seq_qkv = qkv[start:end].unsqueeze(0).transpose(1, 2)  # [1, C, L]
            seq_out = causal_conv1d_fn(
                x=seq_qkv,
                weight=conv_weight,
                bias=None,
                activation=self.activation,
            )
            qkv_out[start:end] = seq_out.squeeze(0).transpose(0, 1)

        # Store conv state for future decode
        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        if gdn_conv_states is not None:
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                if end <= start:
                    continue
                seq_len = end - start
                # Store the last (kernel_size - 1) tokens as conv state
                pad_len = min(seq_len, self.conv_kernel_size - 1)
                gdn_conv_states[self.layer_idx, i, :, :] = 0
                gdn_conv_states[self.layer_idx, i, :, -pad_len:] = qkv[
                    end - pad_len : end
                ].T

        return qkv_out

    def _conv1d_prefill_naive(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Prefill conv1d using PyTorch (per-sequence, fallback)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        qkv_out = torch.empty_like(qkv)

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end <= start:
                continue
            seq_qkv = qkv[start:end].unsqueeze(0).transpose(1, 2)  # [1, C, L]
            seq_out = F.silu(self.conv1d(seq_qkv)[:, :, : end - start])
            qkv_out[start:end] = seq_out.squeeze(0).transpose(0, 1)

        # Store conv state
        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        if gdn_conv_states is not None:
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                if end <= start:
                    continue
                seq_len = end - start
                pad_len = min(seq_len, self.conv_kernel_size - 1)
                gdn_conv_states[self.layer_idx, i, :, :] = 0
                gdn_conv_states[self.layer_idx, i, :, -pad_len:] = qkv[
                    end - pad_len : end
                ].T

        return qkv_out

    def _conv1d_decode(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Decode conv1d: single token per sequence, update conv state."""
        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        bs = qkv.shape[0]

        if gdn_conv_states is not None and _HAS_CAUSAL_CONV1D:
            conv_state = gdn_conv_states[self.layer_idx, :bs]  # [bs, conv_dim, ks]
            conv_weight = self.conv1d.weight.squeeze(1)  # [conv_dim, ks]
            # causal_conv1d_update expects x: [batch, dim], conv_state: [batch, dim, width]
            qkv_out = causal_conv1d_update(
                qkv,  # [bs, conv_dim]
                conv_state,
                conv_weight,
                bias=None,
                activation=self.activation,
            )
            return qkv_out
        elif gdn_conv_states is not None:
            conv_state = gdn_conv_states[self.layer_idx, :bs]  # [bs, conv_dim, ks]
            # Shift state left, add new token
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = qkv
            # Apply conv weights and silu
            conv_weight = self.conv1d.weight.squeeze(1)  # [conv_dim, ks]
            qkv_out = F.silu((conv_state * conv_weight.unsqueeze(0)).sum(-1))
            return qkv_out
        else:
            # No state: fallback (not correct for decode, but won't crash)
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

        # For fresh prefill, always start from zero state.
        # (Warmup/dummy prefills may have written stale data into the buffer.)
        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        initial_state = None
        if gdn_recurrent_states is not None:
            # Zero out the state for sequences being prefilled
            gdn_recurrent_states[self.layer_idx, :num_seqs].zero_()
            initial_state = gdn_recurrent_states[self.layer_idx, :num_seqs]

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
                use_qk_l2norm_in_kernel=True,  # FLA kernel normalizes Q/K internally
            )
            o = o.squeeze(0)
        else:
            o, final_state = self._naive_gdn_prefill(
                q, k, v, g, beta, scale, cu_seqlens
            )

        # Store final recurrent state
        if gdn_recurrent_states is not None and final_state is not None:
            gdn_recurrent_states[self.layer_idx, :num_seqs] = final_state

        return o

    def _gdn_decode(self, q, k, v, g, beta, scale, context) -> torch.Tensor:
        """Decode: recurrent mode (single step)."""
        bs = q.shape[0]

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        if gdn_recurrent_states is not None:
            initial_state = gdn_recurrent_states[self.layer_idx, :bs]
        else:
            initial_state = q.new_zeros(
                bs, self.num_v_heads, self.head_k_dim, self.head_v_dim
            )

        # Reshape for recurrent: [B, 1, H, D]
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
            o = o.squeeze(1)  # [B, H, V]
            # Manually write updated state back to buffer
            if gdn_recurrent_states is not None and final_state is not None:
                gdn_recurrent_states[self.layer_idx, :bs] = final_state
        else:
            # Naive fallback: returns (output, updated_state_f32)
            # We must write the updated state back to the buffer.
            o, updated_state = self._naive_gdn_decode(
                q, k, v, g, beta, scale, initial_state
            )
            if gdn_recurrent_states is not None:
                gdn_recurrent_states[self.layer_idx, :bs] = updated_state

        return o

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalization matching FLA's l2norm (used when use_qk_l2norm_in_kernel=True)."""
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def _naive_gdn_prefill(self, q, k, v, g, beta, scale, cu_seqlens):
        """Naive sequential scan for prefill (correctness fallback).

        Applies L2 normalization to Q/K and scale to Q (matching use_qk_l2norm_in_kernel=True).

        Returns:
            (output, final_states): output [T, H, V], final_states [num_seqs, H, K, V]
        """
        num_seqs = cu_seqlens.shape[0] - 1
        outputs = []
        final_states = []

        # Apply L2 norm to Q/K (matching use_qk_l2norm_in_kernel=True in reference)
        q = self._l2norm(q.float(), dim=-1)
        k = self._l2norm(k.float(), dim=-1)

        # Apply scale to Q (matching reference: query = query * scale)
        q = q * scale

        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()

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

            # Scale is already baked into q, no need to multiply again
            out_seq = torch.stack(out_seq, dim=0).to(v.dtype)
            outputs.append(out_seq)
            final_states.append(S)

        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = v.new_zeros(0, self.num_v_heads, self.head_v_dim)

        final_state = torch.stack(final_states, dim=0)  # [num_seqs, H, K, V]
        return output, final_state

    def _naive_gdn_decode(self, q, k, v, g, beta, scale, state):
        """Naive recurrent step for decode (fallback).

        Applies L2 normalization to Q/K and scale to Q (matching use_qk_l2norm_in_kernel=True).

        Returns:
            (output, updated_state): output [B, H, V], updated_state [B, H, K, V] in float32
        """
        # Apply L2 norm (matching reference)
        q = self._l2norm(q.float(), dim=-1) * scale
        k = self._l2norm(k.float(), dim=-1)
        # IMPORTANT: .float() creates a COPY — the original buffer is NOT modified.
        # We must return the updated state so the caller can write it back.
        state_f32 = state.float()

        decay = g.float().exp().unsqueeze(-1).unsqueeze(-1)
        state_f32.mul_(decay)

        kv = torch.einsum("bhk,bhv->bhkv", k, v.float())
        Sk = torch.einsum("bhkv,bhk->bhv", state_f32, k)
        correction = torch.einsum("bhk,bhv->bhkv", k, Sk)
        state_f32.add_(beta.float().unsqueeze(-1).unsqueeze(-1) * (kv - correction))

        # Scale already baked into q
        out = torch.einsum("bhk,bhkv->bhv", q, state_f32)

        return out.to(v.dtype), state_f32


# ---------------------------------------------------------------------------
# MoE MLP (single expert)
# ---------------------------------------------------------------------------
class Qwen3_5MoeMLP(nn.Module):
    """Single expert MLP (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        gate_up_proj_tensor=None,
        down_proj_tensor=None,
        gate_up_scale_inv_tensor=None,
        down_scale_inv_tensor=None,
        meta: bool = False,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.quantization_config = quantization_config

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            meta=meta,
            weight_tensor=gate_up_proj_tensor,
            scale_tensor=gate_up_scale_inv_tensor,
            quantization_config=quantization_config,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            meta=meta,
            weight_tensor=down_proj_tensor,
            scale_tensor=down_scale_inv_tensor,
            quantization_config=quantization_config,
        )

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Sparse MoE Block with shared expert
# ---------------------------------------------------------------------------
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with TopK routing and shared expert."""

    def __init__(self, config, quantization_config: QuantizationConfig) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size  # 1024
        self.num_experts = config.num_experts  # 512
        self.top_k = config.num_experts_per_tok  # 10

        # Router
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        weight_dtype = quantization_config.dtype or config.dtype

        self.tp_size = get_dist_context().ffn_tp_world_size
        self.tp_group = get_dist_context().ffn_tp_group

        # EP setup
        self.ep_group = get_dist_context().ffn_ep_group
        self.ep_size = get_dist_context().ffn_ep_world_size

        from nanodeploy.layers.distributed_routed_experts import (
            DistributedRoutedExperts,
        )

        self.routed_experts = DistributedRoutedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=self.ep_size,
            tp_size=self.tp_size,
            ep_group=self.ep_group,
            tp_group=self.tp_group,
            n_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            scoring_func="softmax",
            quantization_config=quantization_config,
        )

        # Shared expert
        shared_intermediate = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )
        self.shared_expert = Qwen3_5MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_intermediate,
            hidden_act=config.hidden_act,
            quantization_config=quantization_config,
        )

        # Shared expert gate (sigmoid, scalar per token)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        self.act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        # Shared expert forward
        shared_out = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_out = shared_out * shared_gate

        router_logits = self.gate(hidden_states)

        # Softmax and routing
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        context = get_context()
        is_prefill = context.is_prefill

        final_hidden_states = self.routed_experts(
            hidden_states, selected_experts, routing_weights, is_prefill=is_prefill
        )

        final_hidden_states = final_hidden_states + shared_out
        return final_hidden_states.view(orig_shape)


# ---------------------------------------------------------------------------
# Decoder Layer (mixed attention type)
# ---------------------------------------------------------------------------
class Qwen3_5MoeDecoderLayer(nn.Module):
    """Decoder layer supporting both full_attention and linear_attention."""

    def __init__(
        self,
        config,
        quantization_config: QuantizationConfig,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Determine attention type for this layer
        layer_types = getattr(config, "layer_types", [])
        if layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeFullAttention(
                layer_idx=layer_idx,
                config=config,
                quantization_config=quantization_config,
            )
        else:  # linear_attention
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(
                layer_idx=layer_idx,
                config=config,
                quantization_config=quantization_config,
            )

        # MoE MLP (all layers)
        self.mlp = Qwen3_5MoeSparseMoeBlock(
            config=config, quantization_config=quantization_config
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Qwen3_5MoeModel(nn.Module):
    """Qwen3.5-MoE text model backbone."""

    def __init__(self, config, quantization_config: QuantizationConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, quantization_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


# ---------------------------------------------------------------------------
# Top-level ForConditionalGeneration (text-only)
# ---------------------------------------------------------------------------
class Qwen3_5MoeForConditionalGeneration(nn.Module):
    """Qwen3.5-MoE for conditional generation.

    Checkpoint weight prefix: model.language_model.* → model.*
    The loader strips `language_model.` before parameter lookup.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # The config might have text_config nested (VLM) or be flat
        # After config.py flattening, all text_config attrs are on config
        self.config = config

        quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.quantization_config = quantization_config

        self.model = Qwen3_5MoeModel(config, quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        """Load weights using per-model loader."""
        from .qwen3_5_moe_loader import load_weights

        load_weights(self, weights)

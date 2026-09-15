"""Generic (naive reference) GatedDeltaNet layer implementation.

Provides the linear attention mechanism used in Qwen3.5 MoE. ``GenericGatedDeltaNet``
is the pure-PyTorch **reference** backend (``NaiveRecurrenceMixin``); it owns the
projection setup, causal convolution, state hygiene, output transform, and the
forward/decode orchestration. The FlashInfer and FLA backends subclass it and
swap only the recurrence strategy (by prepending their recurrence mixin), so the
choice of kernel is expressed by composition rather than ``_has_*`` flags.

Components live under ``components/``: convolution (``conv.py``), state hygiene
(``state.py``), recurrence (``recurrence*.py``), gated output (``output.py``).
"""

import torch
import torch.nn.functional as F
from torch import nn

from dlengine.logging import get_logger
from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.layers import get_backend
from dlengine.runtime.layers.base_backend import GatedDeltaNetBase, ReplicatedLinearBase
from dlengine.runtime.models.quant_config import QuantizationConfig

from . import components as gdn
from .components import kernels
from .components.conv import CausalConvMixin
from .components.output import OutputTransformMixin, RMSNormGated
from .components.recurrence import NaiveRecurrenceMixin
from .components.state import StateMixin

# Triton head-repeat helpers (used only in the prefill projection path here).
try:
    from dlengine.runtime.kernel.triton.generic.repeat_heads import (
        can_use_repeat_heads_triton,
        repeat_heads_triton,
    )
except ImportError:
    can_use_repeat_heads_triton = None
    repeat_heads_triton = None

logger = get_logger()

# Backwards-compatible module-level capability aliases. Historically these
# ``_HAS_*`` names lived in this module; they now come from ``gdn.kernels`` but
# are re-exported so existing imports/monkeypatch targets keep working.
_HAS_FLASHINFER_GDN_PREFILL = kernels.HAS_FLASHINFER_GDN_PREFILL
_HAS_FLASHINFER_GDN_PRETRANSPOSE = kernels.HAS_FLASHINFER_GDN_PRETRANSPOSE
_HAS_FLASHINFER_GDN_NONTRANSPOSE = kernels.HAS_FLASHINFER_GDN_NONTRANSPOSE
_HAS_FLA_GDN = kernels.HAS_FLA_GDN
_HAS_CAUSAL_CONV1D = kernels.HAS_CAUSAL_CONV1D


class GenericGatedDeltaNet(
    CausalConvMixin,
    StateMixin,
    NaiveRecurrenceMixin,
    OutputTransformMixin,
    GatedDeltaNetBase,
):
    """GatedDeltaNet linear attention — pure-PyTorch reference backend.

    Uses the naive delta-rule recurrence (``NaiveRecurrenceMixin``); it is the
    portable/correctness backend and the ``ref_fallback_allowed`` target for
    GDN. The FlashInfer / FLA backends subclass this and swap the recurrence
    mixin. State layout is K-last: [N, H, V, K].

    Composed from the ``components/`` mixins: convolution (CausalConvMixin),
    state hygiene (StateMixin), recurrence (NaiveRecurrenceMixin), and output
    transform (OutputTransformMixin). This class owns projection setup and the
    top-level forward/decode orchestration.
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
        self._conv1d_prefill_padded_ws: torch.Tensor | None = None

    def _get_conv1d_prefill_padded_workspace(
        self,
        num_seqs: int,
        dim: int,
        max_seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ws = self._conv1d_prefill_padded_ws
        need_new_ws = (
            ws is None
            or ws.device != device
            or ws.dtype != dtype
            or ws.shape[0] < num_seqs
            or ws.shape[1] < dim
            or ws.shape[2] < max_seqlen
        )
        if need_new_ws:
            self._conv1d_prefill_padded_ws = torch.empty(
                num_seqs,
                dim,
                max_seqlen,
                device=device,
                dtype=dtype,
            )

        return self._conv1d_prefill_padded_ws[:num_seqs, :dim, :max_seqlen]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [total_tokens, hidden_size]
        """
        context = get_batch_context()

        # Lazy verify: 2 tokens/seq processed as two sequential decode passes.
        # Pass 1 processes token_0 (prev_sampled) and saves intermediate state
        # to backup slots; pass 2 processes token_1 (draft) writing final
        # state to active slots.  After verify, rejected seqs can be rolled
        # back by copying backup → active.
        if context.num_tokens_per_seq == 2 and not context.is_prefill:
            return self._lazy_verify_forward(hidden_states, context)

        total_tokens = hidden_states.shape[0]

        # 1. Input projections (fused QKV)
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        scale = self.head_k_dim**-0.5

        if context.is_prefill:
            # Neutralise stale recurrent/conv state left in a recycled slot by a
            # previous request before any prefill path reads it. The GDN state
            # pool is persistent and is not cleared on slot reuse, so without
            # this a fresh sequence batched with a chunked-prefill continuation
            # (block_tables set batch-globally) can pick up the prior occupant's
            # state — harmless zeros on the first run after startup, but garbage
            # on every subsequent run.
            self._zero_fresh_slots(context)
            # 2. Prefill path: chunk conv1d + chunk GDN
            qkv = self._apply_conv1d(qkv, context)
            q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.view(total_tokens, self.num_k_heads, self.head_k_dim).contiguous()
            k = k.view(total_tokens, self.num_k_heads, self.head_k_dim).contiguous()
            v = v.view(total_tokens, self.num_v_heads, self.head_v_dim).contiguous()
            if self.kv_ratio > 1:
                if (
                    repeat_heads_triton is not None
                    and can_use_repeat_heads_triton is not None
                    and can_use_repeat_heads_triton(q, self.kv_ratio)
                    and can_use_repeat_heads_triton(k, self.kv_ratio)
                ):
                    q = repeat_heads_triton(q, self.kv_ratio)
                    k = repeat_heads_triton(k, self.kv_ratio)
                else:
                    q = q.repeat_interleave(self.kv_ratio, dim=1)
                    k = k.repeat_interleave(self.kv_ratio, dim=1)
            beta = b.float().sigmoid()
            A_exp = -self.A_log.float().exp()
            g = A_exp * F.softplus(a.float() + self.dt_bias)
            alpha = g.exp()
            core_attn_out = self._gdn_prefill(q, k, v, g, alpha, beta, scale, context)
        else:
            # 2. Decode path: single-token conv1d + fused GDN decode
            core_attn_out = self._decode_one_step(
                qkv, a, b, total_tokens, scale, context
            )

        return self._apply_output_transform(core_attn_out, z, total_tokens)

    def _lazy_verify_forward(
        self,
        hidden_states: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Forward for lazy verify (num_tokens_per_seq == 2).

        Processes 2 tokens per sequence using two sequential single-token
        decode passes.  Between the passes, the intermediate (1-token) state
        is copied to the backup region of the state pool so that rejected
        sequences can be rolled back cheaply after verification.

        Args:
            hidden_states: [bs*2, hidden_size]  interleaved
                [tok0_seq0, tok1_seq0, tok0_seq1, tok1_seq1, ...]
        """
        total_tokens = hidden_states.shape[0]
        bs = total_tokens // 2

        # ---------- 1. Stateless input projections on ALL tokens ----------
        qkv_all = self.in_proj_qkv(hidden_states)  # [bs*2, conv_dim]
        z_all = self.in_proj_z(hidden_states)  # [bs*2, value_dim]
        a_all = self.in_proj_a(hidden_states)  # [bs*2, num_v_heads]
        b_all = self.in_proj_b(hidden_states)  # [bs*2, num_v_heads]

        # Split into token_0 (even) and token_1 (odd)
        qkv_0 = qkv_all[0::2].contiguous()  # [bs, conv_dim]
        qkv_1 = qkv_all[1::2].contiguous()  # [bs, conv_dim]
        a_0, a_1 = a_all[0::2].contiguous(), a_all[1::2].contiguous()
        b_0, b_1 = b_all[0::2].contiguous(), b_all[1::2].contiguous()

        # ---------- 2. Pass 1: decode token_0 (prev_sampled) ----------
        scale = self.head_k_dim**-0.5
        o0 = self._decode_one_step(qkv_0, a_0, b_0, bs, scale, context)

        # ---------- 3. Snapshot intermediate state → backup slots ----------
        gdn_conv_states = context.gdn_conv_states
        gdn_recurrent_states = context.gdn_recurrent_states
        gdn_state_slots = context.gdn_state_slots
        if gdn_conv_states is not None and gdn_state_slots is not None:
            backup_offset = (gdn_conv_states.shape[1] - 1) // 2
            active_slots = gdn_state_slots[:bs]
            # Clamp so that CUDAGraph dummy slots (= 2*max_bs) map to the
            # dummy slot itself instead of exceeding pool size.
            max_slot = gdn_conv_states.shape[1] - 1
            backup_slots = torch.clamp(active_slots + backup_offset, max=max_slot)
            gdn_conv_states[self.layer_idx, backup_slots] = gdn_conv_states[
                self.layer_idx, active_slots
            ]
            gdn_recurrent_states[self.layer_idx, backup_slots] = gdn_recurrent_states[
                self.layer_idx, active_slots
            ]

        # ---------- 4. Pass 2: decode token_1 (draft) ----------
        o1 = self._decode_one_step(qkv_1, a_1, b_1, bs, scale, context)

        # ---------- 5. Interleave outputs ----------
        core_out = torch.empty(
            total_tokens,
            self.num_v_heads,
            self.head_v_dim,
            dtype=o0.dtype,
            device=o0.device,
        )
        core_out[0::2] = o0
        core_out[1::2] = o1

        # ---------- 6. Gated RMSNorm + output projection ----------
        return self._apply_output_transform(core_out, z_all, total_tokens)

    def _decode_one_step(
        self,
        qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        bs: int,
        scale: float,
        context,
    ) -> torch.Tensor:
        """Conv1d update + split/reshape + GDN decode for a single token per sequence."""
        qkv = self._conv1d_decode(qkv, context)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(bs, self.num_k_heads, self.head_k_dim).contiguous()
        k = k.view(bs, self.num_k_heads, self.head_k_dim).contiguous()
        v = v.view(bs, self.num_v_heads, self.head_v_dim).contiguous()
        if self.kv_ratio > 1:
            q = q.repeat_interleave(self.kv_ratio, dim=1)
            k = k.repeat_interleave(self.kv_ratio, dim=1)
        return self._gdn_decode(q, k, v, a, b, scale, context)


__all__ = ["GenericGatedDeltaNet", "RMSNormGated"]

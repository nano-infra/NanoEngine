"""Generic attention backend.

Provides a FA2-backed implementation for GQA attention that works on
GPUs without Hopper-only kernels (FA3 / flash_mla). Used on sm_80–sm_89
boards (A100, A6000, RTX 4090, RTX 4060 Ti, ...).

MLA is left unimplemented here because it relies on ``flash_mla`` which is
Hopper-only; MLA models should be run on the hopper backend or with a
dedicated MLA fallback (not provided yet).
"""

import torch

try:
    # FA2 (sm_80+): same function names as FA3 but imported from ``flash_attn``.
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _HAS_FA2 = True
except ImportError:
    flash_attn_varlen_func = None  # type: ignore
    flash_attn_with_kvcache = None  # type: ignore
    _HAS_FA2 = False

from dlengine.context_v2.batch import get_batch_context
from dlengine.kernel.triton.generic.kv_store import store_kvcache
from dlengine.kernel.triton.generic.paged_gather import (
    build_paged_gather_indices as _build_paged_gather_indices,
)
from dlengine.layers.base_backend import AttentionBase
from dlengine.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers — mirror the hopper backend (intentional code dup; kept local so
# the generic backend stays standalone).
# ---------------------------------------------------------------------------


def _compute_cached_split(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).long()
    cached_lens = seqlens_k - seqlens_q
    cu_cached = torch.zeros_like(cu_seqlens_k)
    cu_cached[1:] = cached_lens.cumsum(0)
    return cached_lens, cu_cached


def _interleave_cached_fresh(
    cached: torch.Tensor,
    fresh: torch.Tensor,
    cached_lens: torch.Tensor,
    cu_cached: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    # Fully vectorized interleave: per-row destination indices are computed
    # on-device and scattered, avoiding the former Python per-sequence loop with
    # its many ``.item()`` host syncs (one set per sequence, per layer).
    # total_k == total_cached + total_fresh; both are host-known tensor shapes,
    # so the output is allocated without a device->host sync.
    total_cached = cached.shape[0]
    total_fresh = fresh.shape[0]
    ref = cached if cached.numel() > 0 else fresh
    out = ref.new_empty(total_cached + total_fresh, *ref.shape[1:])

    device = cu_seqlens_k.device
    cu_k = cu_seqlens_k.to(torch.int64)
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_c = cu_cached.to(torch.int64)
    clens = cached_lens.to(torch.int64)

    # Cached rows land at cu_k[s] + (j - cu_c[s]) (prefix occupies the head).
    if total_cached > 0:
        idx_c = torch.arange(total_cached, device=device, dtype=torch.int64)
        seq_c = torch.searchsorted(cu_c, idx_c, right=True) - 1
        dest_c = cu_k[seq_c] + (idx_c - cu_c[seq_c])
        out[dest_c] = cached

    # Fresh rows land at cu_k[s] + cached_lens[s] + (j - cu_q[s]) (after prefix).
    if total_fresh > 0:
        idx_f = torch.arange(total_fresh, device=device, dtype=torch.int64)
        seq_f = torch.searchsorted(cu_q, idx_f, right=True) - 1
        dest_f = cu_k[seq_f] + clens[seq_f] + (idx_f - cu_q[seq_f])
        out[dest_f] = fresh

    return out


def _gather_kv_cached_concat(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_fresh: torch.Tensor,
    v_fresh: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached_lens, cu_cached = _compute_cached_split(cu_seqlens_q, cu_seqlens_k)
    total_cached = int(cu_cached[-1].item())

    if total_cached == 0:
        return k_fresh, v_fresh

    cached_indices = _build_paged_gather_indices(
        block_table, cu_cached, block_size, total_k=total_cached
    )
    _, _, num_kv_heads, head_dim = k_cache.shape
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)
    k_cached = k_flat[cached_indices]
    v_cached = v_flat[cached_indices]

    k_out = _interleave_cached_fresh(
        k_cached, k_fresh, cached_lens, cu_cached, cu_seqlens_q, cu_seqlens_k
    )
    v_out = _interleave_cached_fresh(
        v_cached, v_fresh, cached_lens, cu_cached, cu_seqlens_q, cu_seqlens_k
    )
    return k_out, v_out


# ---------------------------------------------------------------------------
# FA2-backed GQA implementation
# ---------------------------------------------------------------------------


class _FA2AttentionImpl:
    """GQA attention impl using FlashAttention-2."""

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        if not _HAS_FA2:
            raise RuntimeError(
                "FlashAttention-2 is required for the generic attention "
                "backend. Install ``flash_attn`` (FA2) or run on a Hopper "
                "GPU with the hopper backend."
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
    ):
        context = get_batch_context()
        if k_cache.numel() and v_cache.numel() and not context.is_dummy:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                num_seqs = context.cu_seqlens_k.shape[0] - 1
                bt = context.block_tables[0, :num_seqs, :]
                k, v = _gather_kv_cached_concat(
                    k_cache,
                    v_cache,
                    k,
                    v,
                    bt,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    k_cache.shape[1],
                )
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )
            return o

        # decode path
        ntps = context.num_tokens_per_seq
        total_tokens, num_head, head_dim = q.shape
        bs = total_tokens // ntps
        context_lens = context.context_lens[0, :bs]
        block_tables = context.block_tables[0, :bs]

        # FA2's ``flash_attn_with_kvcache`` takes the same logical args as
        # FA3 but names the paged-KV table ``block_table`` (FA3: ``page_table``).
        out = flash_attn_with_kvcache(
            q.reshape(bs, ntps, num_head, head_dim),
            k_cache,
            v_cache,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=self.scale,
            causal=ntps > 1,
            # NOTE: ``return_softmax_lse`` intentionally omitted — the LSE is
            # unused here (only ``out`` is consumed) and some flash-attn builds
            # (e.g. the PPU runtime) reject that keyword. Default is False, so
            # upstream FA2 returns just ``out`` too.
        )
        # Defensive: handle builds that still return ``(out, lse)``.
        o = out[0] if isinstance(out, tuple) else out
        if ntps > 1:
            o = o.reshape(total_tokens, num_head, head_dim)
        return o


# ---------------------------------------------------------------------------
# Public layer
# ---------------------------------------------------------------------------


class GenericAttention(AttentionBase):
    """Generic GQA attention layer, FA2-backed."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
        attention_type: str = "GQA",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.attention_type = attention_type
        self.k_cache = self.v_cache = torch.tensor([])

        if attention_type == "GQA":
            self.impl = _FA2AttentionImpl(num_heads, head_dim, scale, num_kv_heads)
        elif attention_type == "MLA":
            raise NotImplementedError(
                "MLA attention requires ``flash_mla`` (Hopper-only). Run MLA "
                "models on the hopper backend, or implement an MLA fallback."
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.impl.forward(
            q, k, v, self.k_cache, self.v_cache, sparse_indices=sparse_indices
        )

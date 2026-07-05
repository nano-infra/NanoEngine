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

try:
    import flashinfer

    _HAS_FLASHINFER = True
except ImportError:
    flashinfer = None  # type: ignore
    _HAS_FLASHINFER = False

from dlengine.context_v2.batch import get_batch_context
from dlengine.context_v2.graph import get_graph_context
from dlengine.kernel.triton.generic.kv_store import store_kvcache
from dlengine.kernel.triton.generic.paged_gather import (
    build_paged_gather_indices as _build_paged_gather_indices,
)
from dlengine.layers.base_backend import AttentionBase
from dlengine.logging import get_logger

logger = get_logger()


_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
_FLASHINFER_DECODE_CACHE: dict[tuple, object] = {}
_FLASHINFER_PREFILL_CACHE: dict[tuple, object] = {}


def _flashinfer_enabled() -> bool:
    import os

    return os.environ.get("DLENGINE_USE_FLASHINFER_DECODE", "1") == "1"


def _flashinfer_prefill_enabled() -> bool:
    import os

    return os.environ.get("DLENGINE_USE_FLASHINFER_PREFILL", "1") == "1"


def _flashinfer_fixed_split_size() -> int:
    import os

    return max(0, int(os.environ.get("DLENGINE_FLASHINFER_FIXED_SPLIT_SIZE", "0")))


def _flashinfer_disable_split_kv() -> bool:
    import os

    return os.environ.get("DLENGINE_FLASHINFER_DISABLE_SPLIT_KV", "1") == "1"


def _has_cached_prefill(context) -> bool:
    return context.max_seqlen_k > context.max_seqlen_q


def _paged_prefill_metadata(
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_seqs: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = (cu_seqlens_k[1 : num_seqs + 1] - cu_seqlens_k[:num_seqs]).to(
        torch.int32
    )
    pages_per_seq = torch.div(
        seq_lens + page_size - 1, page_size, rounding_mode="floor"
    )
    indptr = torch.empty(num_seqs + 1, device=seq_lens.device, dtype=torch.int32)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    max_pages = block_tables.shape[1]
    page_offsets = torch.arange(
        max_pages, device=block_tables.device, dtype=torch.int32
    )
    mask = page_offsets.unsqueeze(0) < pages_per_seq.unsqueeze(1)
    indices = block_tables[:num_seqs, :max_pages][mask].contiguous()

    last_page_len = seq_lens % page_size
    last_page_len = torch.where(
        last_page_len == 0,
        torch.full_like(last_page_len, page_size),
        last_page_len,
    )
    return indptr, indices, last_page_len


def _paged_decode_metadata(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    bs: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = context_lens[:bs].to(torch.int32)
    pages_per_seq = torch.div(
        seq_lens + page_size - 1, page_size, rounding_mode="floor"
    )
    indptr = torch.empty(bs + 1, device=seq_lens.device, dtype=torch.int32)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    max_pages = block_tables.shape[1]
    page_offsets = torch.arange(
        max_pages, device=block_tables.device, dtype=torch.int32
    )
    mask = page_offsets.unsqueeze(0) < pages_per_seq.unsqueeze(1)
    indices = block_tables[:bs, :max_pages][mask].contiguous()

    last_page_len = seq_lens % page_size
    last_page_len = torch.where(
        last_page_len == 0,
        torch.full_like(last_page_len, page_size),
        last_page_len,
    )
    return indptr, indices, last_page_len


def _flashinfer_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    scale: float,
) -> torch.Tensor:
    key = (
        q.device.index,
        q.dtype,
        k_cache.dtype,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        block_tables.data_ptr(),
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
    )
    wrapper = _FLASHINFER_PREFILL_CACHE.get(key)
    if wrapper is None:
        workspace = torch.empty(
            _FLASHINFER_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=q.device,
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(  # type: ignore[union-attr]
            workspace,
            kv_layout="NHD",
        )
        _FLASHINFER_PREFILL_CACHE.clear()
        _FLASHINFER_PREFILL_CACHE[key] = wrapper

    context = get_batch_context()
    plan_key = (key, id(context))
    if getattr(context, "_flashinfer_prefill_plan_key", None) != plan_key:
        indptr, indices, last_page_len = _paged_prefill_metadata(
            block_tables, cu_seqlens_k, num_seqs, page_size
        )
        plan_kwargs = {}
        fixed_split_size = _flashinfer_fixed_split_size()
        if fixed_split_size > 0:
            plan_kwargs["fixed_split_size"] = fixed_split_size
        if _flashinfer_disable_split_kv():
            plan_kwargs["disable_split_kv"] = True
        wrapper.plan(
            cu_seqlens_q,
            indptr,
            indices,
            last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            o_data_type=q.dtype,
            sm_scale=scale,
            **plan_kwargs,
        )
        context._flashinfer_prefill_plan_key = plan_key

    return wrapper.run(q, (k_cache, v_cache))


def _flashinfer_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    bs: int,
    ntps: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    scale: float,
) -> torch.Tensor:
    if ntps != 1:
        raise NotImplementedError(
            "flashinfer decode fast path only supports one token/seq"
        )

    graph_wrapper = get_graph_context().active_flashinfer_decode_wrapper
    if graph_wrapper is not None:
        return graph_wrapper.run(q.reshape(bs, num_heads, head_dim), (k_cache, v_cache))

    context = get_batch_context()
    key = (
        q.device.index,
        q.dtype,
        k_cache.dtype,
        bs,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        block_tables.data_ptr(),
        context_lens.data_ptr(),
    )
    wrapper = _FLASHINFER_DECODE_CACHE.get(key)
    if wrapper is None:
        workspace = torch.empty(
            _FLASHINFER_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=q.device,
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(  # type: ignore[union-attr]
            workspace,
            kv_layout="NHD",
            use_tensor_cores=False,
        )
        _FLASHINFER_DECODE_CACHE.clear()
        _FLASHINFER_DECODE_CACHE[key] = wrapper

    # BatchContext is recreated for every scheduler step, so storing the plan
    # marker here shares one plan across layers without reusing stale metadata
    # across decode steps as context_lens advances.
    plan_key = (key, id(context))
    if getattr(context, "_flashinfer_decode_plan_key", None) != plan_key:
        indptr, indices, last_page_len = _paged_decode_metadata(
            block_tables, context_lens, bs, page_size
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            o_data_type=q.dtype,
            sm_scale=scale,
        )
        context._flashinfer_decode_plan_key = plan_key

    return wrapper.run(q.reshape(bs, num_heads, head_dim), (k_cache, v_cache))


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
                if (
                    _HAS_FLASHINFER
                    and _flashinfer_prefill_enabled()
                    and _has_cached_prefill(context)
                ):
                    return _flashinfer_prefill_paged(
                        q,
                        k_cache,
                        v_cache,
                        bt,
                        context.cu_seqlens_q,
                        context.cu_seqlens_k,
                        num_seqs,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        k_cache.shape[1],
                        self.scale,
                    )
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

        if _HAS_FLASHINFER and _flashinfer_enabled() and ntps == 1:
            return _flashinfer_decode(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                bs,
                ntps,
                num_head,
                self.num_kv_heads,
                head_dim,
                k_cache.shape[1],
                self.scale,
            )

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

"""Generic attention backend.

Provides a FA2-backed implementation for GQA attention that works on
GPUs without Hopper-only kernels (FA3 / flash_mla). Used on sm_80–sm_89
boards (A100, A6000, RTX 4090, RTX 4060 Ti, ...).

MLA is left unimplemented here because it relies on ``flash_mla`` which is
Hopper-only; MLA models should be run on the hopper backend or with a
dedicated MLA fallback (not provided yet).
"""

import torch
import torch.nn.functional as F

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
from dlengine.context_v2.cache.hisparse import get_hisparse_context
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


def _hisparse_swa_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hot_k_cache: torch.Tensor,
    hot_v_cache: torch.Tensor,
    sliding_window: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    write_kv_cache: bool = True,
) -> torch.Tensor | None:
    context = get_batch_context()
    hisparse_ctx = get_hisparse_context()
    if (
        context.is_prefill
        or context.hisparse_slots is None
        or context.hisparse_slot_mapping is None
        or hot_k_cache.numel() == 0
        or hot_v_cache.numel() == 0
        or hisparse_ctx.tokens_per_seq <= 0
    ):
        return None

    ntps = context.num_tokens_per_seq
    if ntps != 1:
        return None

    bs = q.shape[0]
    slots = context.hisparse_slots[:bs].to(torch.int64)
    valid_seq = slots < hisparse_ctx.max_num_seqs

    if write_kv_cache:
        store_kvcache(
            k, v, hot_k_cache, hot_v_cache, context.hisparse_slot_mapping[:bs]
        )

    window = min(int(sliding_window), int(hisparse_ctx.tokens_per_seq))
    if window <= 0:
        return None

    context_lens = context.context_lens[0, :bs].to(torch.int64)
    win_lens = context_lens.clamp(min=1, max=window)
    offsets = torch.arange(window, device=q.device, dtype=torch.int64)
    logical = context_lens.unsqueeze(1) - win_lens.unsqueeze(1) + offsets.unsqueeze(0)
    valid_tok = offsets.unsqueeze(0) < win_lens.unsqueeze(1)
    physical = slots.unsqueeze(1) * hisparse_ctx.tokens_per_seq
    physical = physical + torch.remainder(logical, hisparse_ctx.tokens_per_seq)
    physical = torch.where(valid_seq.unsqueeze(1) & valid_tok, physical, 0)

    k_win = hot_k_cache[physical.reshape(-1)].view(bs, window, num_kv_heads, -1)
    v_win = hot_v_cache[physical.reshape(-1)].view(bs, window, num_kv_heads, -1)
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k_win = k_win.repeat_interleave(repeat, dim=2)
        v_win = v_win.repeat_interleave(repeat, dim=2)

    valid = valid_seq.unsqueeze(1) & valid_tok
    scores = torch.einsum("bhd,bshd->bhs", q.float(), k_win.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e30)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bshd->bhd", probs, v_win.float()).to(q.dtype)
    out = torch.where(valid_seq.view(bs, 1, 1), out, torch.zeros_like(out))
    return out


def _hisparse_promote_swa_suffix(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    hot_k_cache: torch.Tensor,
    hot_v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sliding_window: int,
) -> None:
    context = get_batch_context()
    hisparse_ctx = get_hisparse_context()
    if (
        context.hisparse_slots is None
        or hot_k_cache.numel() == 0
        or hot_v_cache.numel() == 0
        or hisparse_ctx.tokens_per_seq <= 0
        or block_tables is None
        or block_tables.numel() == 0
    ):
        return

    bs = min(context.hisparse_slots.numel(), cu_seqlens_k.numel() - 1)
    if bs <= 0:
        return

    slots = context.hisparse_slots[:bs].to(torch.int64)
    valid_seq = slots < hisparse_ctx.max_num_seqs
    if not bool(valid_seq.any().item()):
        return

    seq_lens = (cu_seqlens_k[1 : bs + 1] - cu_seqlens_k[:bs]).to(torch.int64)
    window = min(int(sliding_window), int(hisparse_ctx.tokens_per_seq))
    win_lens = seq_lens.clamp(min=0, max=window)
    if not bool((win_lens > 0).any().item()):
        return

    offsets = torch.arange(window, device=k_cache.device, dtype=torch.int64)
    logical = seq_lens.unsqueeze(1) - win_lens.unsqueeze(1) + offsets.unsqueeze(0)
    valid_tok = offsets.unsqueeze(0) < win_lens.unsqueeze(1)
    page_size = k_cache.shape[1]
    page_idx = torch.div(logical.clamp(min=0), page_size, rounding_mode="floor")
    page_idx_safe = page_idx.clamp(0, block_tables.shape[1] - 1)
    page_blocks = block_tables[:bs].to(torch.int64).gather(1, page_idx_safe)
    physical = page_blocks * page_size + torch.remainder(
        logical.clamp(min=0), page_size
    )
    valid = valid_seq.unsqueeze(1) & valid_tok
    physical = torch.where(valid, physical, 0)

    k_suffix = k_cache.reshape(-1, *k_cache.shape[2:])[physical.reshape(-1)]
    v_suffix = v_cache.reshape(-1, *v_cache.shape[2:])[physical.reshape(-1)]
    hot = slots.unsqueeze(1) * hisparse_ctx.tokens_per_seq
    hot = hot + torch.remainder(logical.clamp(min=0), hisparse_ctx.tokens_per_seq)
    hot = torch.where(valid, hot, 0).reshape(-1)
    valid_flat = valid.reshape(-1)
    if bool(valid_flat.any().item()):
        hot_k_cache[hot[valid_flat]] = k_suffix[valid_flat]
        hot_v_cache[hot[valid_flat]] = v_suffix[valid_flat]


def _hisparse_prefill_fresh_slot_mapping(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor | None:
    context = get_batch_context()
    hisparse_ctx = get_hisparse_context()
    if context.hisparse_slots is None or hisparse_ctx.tokens_per_seq <= 0:
        return None

    total_q = int(cu_seqlens_q[-1].item())
    if total_q <= 0:
        return None

    idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_k = cu_seqlens_k.to(torch.int64)
    seq_idx = torch.searchsorted(cu_q, idx, right=True) - 1
    q_lens = cu_q[1:] - cu_q[:-1]
    k_lens = cu_k[1:] - cu_k[:-1]
    start_pos = k_lens - q_lens
    logical = start_pos[seq_idx] + (idx - cu_q[seq_idx])
    slots = context.hisparse_slots.to(torch.int64)
    seq_slots = slots[seq_idx]
    hot = seq_slots * hisparse_ctx.tokens_per_seq
    hot = hot + torch.remainder(logical, hisparse_ctx.tokens_per_seq)
    hot = torch.where(seq_slots < hisparse_ctx.max_num_seqs, hot, -1)
    return hot.to(torch.int32)


def _hisparse_store_swa_fresh(
    k: torch.Tensor,
    v: torch.Tensor,
    hot_k_cache: torch.Tensor,
    hot_v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    slot_mapping = _hisparse_prefill_fresh_slot_mapping(cu_seqlens_q, cu_seqlens_k)
    if slot_mapping is not None:
        store_kvcache(k, v, hot_k_cache, hot_v_cache, slot_mapping)


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


def _sdpa_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    outs = []
    num_seqs = cu_seqlens_q.numel() - 1
    for i in range(num_seqs):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        qi = q[q_start:q_end]
        ki = k[k_start:k_end]
        vi = v[k_start:k_end]
        if qi.numel() == 0:
            continue
        if ki.shape[1] != qi.shape[1]:
            repeat = qi.shape[1] // ki.shape[1]
            ki = ki.repeat_interleave(repeat, dim=1)
            vi = vi.repeat_interleave(repeat, dim=1)

        q_len = qi.shape[0]
        k_len = ki.shape[0]
        cached_len = k_len - q_len
        q_pos = torch.arange(q_len, device=q.device, dtype=torch.int64)
        k_pos = torch.arange(k_len, device=q.device, dtype=torch.int64)
        allowed = k_pos.unsqueeze(0) <= (cached_len + q_pos).unsqueeze(1)
        if sliding_window is not None:
            min_k = cached_len + q_pos - int(sliding_window) + 1
            allowed &= k_pos.unsqueeze(0) >= min_k.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            qi.transpose(0, 1).unsqueeze(0),
            ki.transpose(0, 1).unsqueeze(0),
            vi.transpose(0, 1).unsqueeze(0),
            attn_mask=allowed.unsqueeze(0).unsqueeze(0),
            dropout_p=0.0,
            scale=scale,
        )
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _sdpa_fixed_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_tokens_per_seq: int,
    scale: float,
) -> torch.Tensor:
    """Graph-safe SDPA for decode layers that intentionally own no KV cache."""
    total_tokens, num_heads, head_dim = q.shape
    bs = total_tokens // num_tokens_per_seq
    q = q.view(bs, num_tokens_per_seq, num_heads, head_dim).transpose(1, 2)
    k = k.view(bs, num_tokens_per_seq, k.shape[1], head_dim).transpose(1, 2)
    v = v.view(bs, num_tokens_per_seq, v.shape[1], head_dim).transpose(1, 2)
    if k.shape[1] != num_heads:
        repeat = num_heads // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=num_tokens_per_seq > 1,
        scale=scale,
    )
    return out.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)


# ---------------------------------------------------------------------------
# FA2-backed GQA implementation
# ---------------------------------------------------------------------------


class _FA2AttentionImpl:
    """GQA attention impl using FlashAttention-2."""

    def __init__(self, num_heads, head_dim, scale, num_kv_heads, sliding_window=None):
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
        self.sliding_window = sliding_window
        self.has_flashinfer = _HAS_FLASHINFER
        self.use_flashinfer_decode = True
        self.use_flashinfer_prefill = True
        self.use_fa2 = True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
        hot_k_cache: torch.Tensor | None = None,
        hot_v_cache: torch.Tensor | None = None,
        write_kv_cache: bool = True,
    ):
        context = get_batch_context()
        if (
            self.sliding_window is not None
            and hot_k_cache is not None
            and hot_v_cache is not None
        ):
            out = _hisparse_swa_decode(
                q,
                k,
                v,
                hot_k_cache,
                hot_v_cache,
                self.sliding_window,
                self.num_heads,
                self.num_kv_heads,
                self.scale,
                write_kv_cache,
            )
            if out is not None:
                return out

        if (
            write_kv_cache
            and k_cache.numel()
            and v_cache.numel()
            and not context.is_dummy
        ):
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if (
                self.sliding_window is not None
                and k_cache.numel() == 0
                and hot_k_cache is not None
                and hot_v_cache is not None
                and hot_k_cache.numel() > 0
            ):
                _hisparse_store_swa_fresh(
                    k,
                    v,
                    hot_k_cache,
                    hot_v_cache,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                )
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_q,
                    cu_seqlens_k=context.cu_seqlens_q,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=(int(self.sliding_window) - 1, 0),
                )
                return o

            if context.block_tables is not None and k_cache.numel() and v_cache.numel():
                num_seqs = context.cu_seqlens_k.shape[0] - 1
                bt = context.block_tables[0, :num_seqs, :]
                if (
                    self.sliding_window is not None
                    and hot_k_cache is not None
                    and hot_v_cache is not None
                ):
                    _hisparse_promote_swa_suffix(
                        k_cache,
                        v_cache,
                        hot_k_cache,
                        hot_v_cache,
                        bt,
                        context.cu_seqlens_k,
                        self.sliding_window,
                    )
                if (
                    self.has_flashinfer
                    and self.use_flashinfer_prefill
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
            attn_kwargs = {}
            if self.sliding_window is not None:
                attn_kwargs["window_size"] = (int(self.sliding_window) - 1, 0)
            if self.head_dim > 256:
                return _sdpa_varlen_func(
                    q,
                    k,
                    v,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    self.scale,
                    sliding_window=self.sliding_window,
                )
            if not self.use_fa2:
                return _sdpa_varlen_func(
                    q,
                    k,
                    v,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    self.scale,
                    sliding_window=self.sliding_window,
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
                **attn_kwargs,
            )
            return o

        # decode path
        ntps = context.num_tokens_per_seq
        total_tokens, num_head, head_dim = q.shape
        bs = total_tokens // ntps
        context_lens = context.context_lens[0, :bs]
        block_tables = context.block_tables[0, :bs]

        if (
            k_cache.numel()
            and v_cache.numel()
            and self.has_flashinfer
            and self.use_flashinfer_decode
            and _flashinfer_enabled()
            and ntps == 1
        ):
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
        if not (k_cache.numel() and v_cache.numel()):
            return _sdpa_fixed_decode(q, k, v, ntps, self.scale)

        if not self.use_fa2:
            raise RuntimeError(
                "Torch attention does not yet support paged KV decode; select "
                "fa2 or flashinfer for serving."
            )
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
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.attention_type = attention_type
        self.k_cache = self.v_cache = torch.tensor([])
        self.hisparse_k_cache = self.hisparse_v_cache = torch.tensor([])

        if attention_type == "GQA":
            self.impl = _FA2AttentionImpl(
                num_heads, head_dim, scale, num_kv_heads, sliding_window=sliding_window
            )
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
        write_kv_cache: bool = True,
    ) -> torch.Tensor:
        return self.impl.forward(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            sparse_indices=sparse_indices,
            hot_k_cache=self.hisparse_k_cache,
            hot_v_cache=self.hisparse_v_cache,
            write_kv_cache=write_kv_cache,
        )

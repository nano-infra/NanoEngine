import torch

try:
    import flash_mla
except ImportError:
    flash_mla = None

try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

    _FA_KVCACHE_TABLE_ARG = "page_table"
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        _FA_KVCACHE_TABLE_ARG = "block_table"
    except ImportError:
        flash_attn_varlen_func = None  # type: ignore
        flash_attn_with_kvcache = None  # type: ignore
        _FA_KVCACHE_TABLE_ARG = "page_table"

from dlengine.logging import get_logger
from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.context.cache.hca import get_hca_context
from dlengine.runtime.context.cache.hisparse import get_hisparse_context
from dlengine.runtime.context.cache.mla import get_mla_context
from dlengine.runtime.kernel.triton.generic.kv_store import store_kcache, store_kvcache
from dlengine.runtime.kernel.triton.generic.paged_gather import (
    build_paged_gather_indices as _build_paged_gather_indices,
)
from dlengine.runtime.kernel.triton.hopper.fp8_utils import store_kcache_fp8
from dlengine.runtime.layers.base_backend import AttentionBase
from dlengine.runtime.layers.backends.attention.mla_utils import (
    _compute_cached_split,
    _gather_cache_cached_only,
    _gather_kv_cached_concat,
    _interleave_cached_fresh,
    topk_indices_to_physical,
)

logger = get_logger()


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
    return torch.where(valid_seq.view(bs, 1, 1), out, torch.zeros_like(out))


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
    start_pos = cu_k[1:] - q_lens
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


class Fa3AttentionImpl:

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        sliding_window=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

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
            and not get_batch_context().is_dummy
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

            if context.block_tables is not None:
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
        else:  # decode
            ntps = context.num_tokens_per_seq
            total_tokens, num_head, head_dim = q.shape
            bs = total_tokens // ntps
            context_lens = context.context_lens[0, :bs]
            block_tables = context.block_tables[0, :bs]

            kwargs = {
                "cache_seqlens": context_lens,
                _FA_KVCACHE_TABLE_ARG: block_tables,
                "softmax_scale": self.scale,
                "causal": ntps > 1,
                "return_softmax_lse": True,
            }
            o, lse = flash_attn_with_kvcache(
                q.reshape(bs, ntps, num_head, head_dim),
                k_cache,
                v_cache,
                **kwargs,
            )[:2]

            # o: (bs, ntps, H, D) → (total_tokens, H, D)
            if ntps > 1:
                o = o.reshape(total_tokens, num_head, head_dim)

        return o


class FlashMlaAttentionImpl:
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        causal: bool = True,
        nsa_index_topk: int = 0,
    ):
        import flash_mla

        if scale is None:
            scale = 1.0 / (head_size**0.5)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if v_head_size is None:
            v_head_size = head_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_size = v_head_size
        self.causal = causal
        self.nsa_index_topk = nsa_index_topk

        assert num_kv_heads == 1, "MLA requires num kv heads equal to 1"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
        write_kv_cache: bool = True,
    ):

        context = get_batch_context()
        if k_cache.numel() and not get_batch_context().is_dummy:
            slot_mapping = (
                context.hisparse_slot_mapping
                if context.hisparse_slot_mapping is not None
                else context.slot_mapping
            )
            if k_cache.dtype == torch.float8_e4m3fn:
                store_kcache_fp8(k, k_cache, slot_mapping)
            else:
                store_kcache(k, k_cache, slot_mapping)

        if context.is_prefill:
            # NOTE: MLA prefill is handled directly in DeepseekV2Attention.forward
            # using the non-absorbed approach (expanded K/V). This path should not
            # be reached for MLA models.
            raise RuntimeError(
                "FlashMlaAttentionImpl.forward should not be called during prefill. "
                "MLA prefill is handled in DeepseekV2Attention.forward."
            )

        else:  # decode
            ntps = context.num_tokens_per_seq
            total_tokens, num_head, head_dim = q.shape
            bs = total_tokens // ntps
            q = q[: bs * ntps]
            context_lens = context.context_lens[0, :bs]
            block_tables = context.block_tables[0, :bs]

            # FP8 KV cache REQUIRES sparse decode — dense_decode_fwd
            # does not support FP8.  When sparse_indices is None (e.g.
            # during CUDA graph capture warmup), synthesise dummy all-invalid
            # indices so the sparse kernel is still used.
            if (
                k_cache.dtype == torch.float8_e4m3fn
                and sparse_indices is None
                and self.nsa_index_topk > 0
            ):
                sparse_indices = torch.full(
                    (bs * ntps, self.nsa_index_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )

            if sparse_indices is not None and k_cache.dtype == torch.float8_e4m3fn:
                # === Sparse decode (NSA V3.2) ===
                # sparse_indices: (bs * ntps, topk) — physical slot indices
                # Reshape to (bs, ntps, topk) for flash_mla_with_kvcache
                topk = sparse_indices.shape[-1]
                indices_3d = sparse_indices.view(bs, ntps, topk)

                # Use context-managed sparse sched meta (CUDA graph compatible)
                mla_context = get_mla_context()
                sparse_meta = mla_context.sparse_tile_scheduler_metadata
                if sparse_meta is None:
                    sparse_meta, _ = flash_mla.get_mla_metadata()

                o, lse = flash_mla.flash_mla_with_kvcache(
                    q.reshape(bs, ntps, num_head, head_dim),
                    k_cache,
                    None,  # block_table (not needed for sparse)
                    None,  # cache_seqlens (not needed for sparse)
                    self.v_head_size,
                    sparse_meta,
                    None,  # num_splits
                    self.scale,
                    False,  # causal must be False for sparse
                    is_fp8_kvcache=True,  # sparse requires FP8
                    indices=indices_3d,
                )
                # Write back so graph runner can track it
                mla_context.sparse_tile_scheduler_metadata = sparse_meta
            else:
                # === Dense decode (default) ===
                hca_context = get_hca_context()
                if hca_context.tile_scheduler_metadata is not None:
                    # Use precomputed metadata from prepare_decode (CUDA graph compatible)
                    tile_scheduler_metadata = hca_context.tile_scheduler_metadata
                else:
                    # Fallback: create fresh FlashMLASchedMeta (will be initialized on first kernel call)
                    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

                o, lse = flash_mla.flash_mla_with_kvcache(
                    q.reshape(bs, ntps, num_head, head_dim),
                    k_cache,
                    block_tables,
                    context_lens,
                    self.v_head_size,
                    tile_scheduler_metadata,
                    None,  # num_splits (managed internally by FlashMLASchedMeta)
                    self.scale,
                    ntps > 1,  # causal=True when lazy verify
                    is_fp8_kvcache=k_cache.dtype == torch.float8_e4m3fn,
                )

            # o: (bs, ntps, H, v_head_dim) → (q_len, H, v_head_dim)
            o = o.reshape(bs * ntps, o.shape[2], o.shape[3])

        return o


class Fa3Attention(AttentionBase):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        v_head_dim,
        attention_type: str = "MLA",
        nsa_index_topk: int = 0,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.hisparse_k_cache = self.hisparse_v_cache = torch.tensor([])
        self.forward_method = None

        if attention_type == "MLA":
            self.impl = FlashMlaAttentionImpl(
                num_heads,
                head_dim,
                scale,
                num_kv_heads,
                v_head_dim,
                nsa_index_topk=nsa_index_topk,
            )
        elif attention_type == "GQA":
            self.impl = Fa3AttentionImpl(
                num_heads,
                head_dim,
                scale,
                num_kv_heads,
                sliding_window=sliding_window,
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
    ):
        """forward."""
        kwargs = {
            "sparse_indices": sparse_indices,
            "write_kv_cache": write_kv_cache,
        }
        # Hot GQA caches are a FlashAttention/HiSparse detail. Passing them to
        # FlashMlaAttentionImpl breaks MLA graph capture because its forward signature
        # intentionally has no hot-cache arguments.
        if isinstance(self.impl, Fa3AttentionImpl):
            kwargs.update(
                hot_k_cache=self.hisparse_k_cache,
                hot_v_cache=self.hisparse_v_cache,
            )
        return self.impl.forward(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            **kwargs,
        )

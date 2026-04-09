import flash_mla
import torch
from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

from nanodeploy.backends.base_backend import AttentionBase
from nanodeploy.backends.gpu_generic.kernels.kv_store import store_kcache, store_kvcache
from nanodeploy.backends.gpu_generic.kernels.paged_gather import (
    build_paged_gather_indices as _build_paged_gather_indices,
)
from nanodeploy.context.context import get_context
from nanodeploy.logging import get_logger


logger = get_logger()


def _compute_cached_split(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence cached/fresh split for chunked prefill.

    Returns:
        cached_lens:  [num_seqs] — number of previously-cached tokens per sequence
        cu_cached:    [num_seqs + 1] — cumulative cached lengths
    """
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
    """Interleave cached and fresh tensors into ragged K layout.

    Per sequence i the output is [cached_tokens_i, fresh_tokens_i] contiguously.
    """
    num_seqs = cached_lens.shape[0]
    total_k = int(cu_seqlens_k[-1].item())
    ref = cached if cached.numel() > 0 else fresh
    out = ref.new_empty(total_k, *ref.shape[1:])

    for i in range(num_seqs):
        dst = int(cu_seqlens_k[i].item())
        nc = int(cached_lens[i].item())
        cs = int(cu_cached[i].item())
        qs = int(cu_seqlens_q[i].item())
        nf = int(cu_seqlens_q[i + 1].item()) - qs

        if nc > 0:
            out[dst : dst + nc] = cached[cs : cs + nc]
        if nf > 0:
            out[dst + nc : dst + nc + nf] = fresh[qs : qs + nf]

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
    """Gather only previously-cached K/V from paged cache, concat with fresh K/V.

    Avoids redundantly re-reading fresh tokens that were just written to cache.
    Falls back to full gather when there are no cached tokens.
    """
    cached_lens, cu_cached = _compute_cached_split(cu_seqlens_q, cu_seqlens_k)
    total_cached = int(cu_cached[-1].item())

    if total_cached == 0:
        return k_fresh, v_fresh

    cached_indices = _build_paged_gather_indices(block_table, cu_cached, block_size)
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


def _gather_cache_cached_only(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather only previously-cached tokens from a single paged cache.

    Returns:
        gathered:    [total_cached, ...] — cached tokens from paged cache
        cached_lens: [num_seqs] — per-sequence cached counts
        cu_cached:   [num_seqs + 1] — cumulative cached lengths
    """
    cached_lens, cu_cached = _compute_cached_split(cu_seqlens_q, cu_seqlens_k)
    total_cached = int(cu_cached[-1].item())

    if total_cached == 0:
        trailing = cache.shape[2:]
        gathered = cache.new_empty(0, *trailing)
        return gathered, cached_lens, cu_cached

    cached_indices = _build_paged_gather_indices(block_table, cu_cached, block_size)
    trailing = cache.shape[2:]
    flat = cache.reshape(-1, *trailing)
    return flat[cached_indices], cached_lens, cu_cached


class FlashAttentionImpl:

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
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
    ):
        context = get_context()
        if k_cache.numel() and v_cache.numel() and not get_context().is_dummy:
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
        else:  # decode
            ntps = context.num_tokens_per_seq
            total_tokens, num_head, head_dim = q.shape
            bs = total_tokens // ntps
            context_lens = context.context_lens[0, :bs]
            block_tables = context.block_tables[0, :bs]

            o, lse = flash_attn_with_kvcache(
                q.reshape(bs, ntps, num_head, head_dim),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                page_table=block_tables,
                softmax_scale=self.scale,
                causal=ntps > 1,
                return_softmax_lse=True,
            )[:2]

            # o: (bs, ntps, H, D) → (total_tokens, H, D)
            if ntps > 1:
                o = o.reshape(total_tokens, num_head, head_dim)

        return o


class FlashMLAImpl:
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        causal: bool = True,
        **kwargs,
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

        assert num_kv_heads == 1, "MLA requires num kv heads equal to 1"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):

        context = get_context()
        if k_cache.numel() and not get_context().is_dummy:
            store_kcache(k, k_cache, context.slot_mapping)

        if context.is_prefill:
            # NOTE: MLA prefill is handled directly in DeepseekV2Attention.forward
            # using the non-absorbed approach (expanded K/V). This path should not
            # be reached for MLA models.
            raise RuntimeError(
                "FlashMLAImpl.forward should not be called during prefill. "
                "MLA prefill is handled in DeepseekV2Attention.forward."
            )

        else:  # decode
            ntps = context.num_tokens_per_seq
            total_tokens, num_head, head_dim = q.shape
            bs = total_tokens // ntps
            q = q[: bs * ntps]
            context_lens = context.context_lens[0, :bs]
            block_tables = context.block_tables[0, :bs]

            if context.tile_scheduler_metadata is not None:
                # Use precomputed metadata from prepare_decode (CUDA graph compatible)
                tile_scheduler_metadata = context.tile_scheduler_metadata
                # num_splits must have shape (batch_size + 1); slice buffer to match
                num_splits = context.num_splits[: bs + 1]
            else:
                # Fallback: compute inline
                # For seqlen_q > 1 (lazy verify), adjust num_q_heads_per_kv
                tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
                    context_lens,
                    ntps * self.num_heads // self.num_kv_heads,
                    self.num_kv_heads,
                )

            o, lse = flash_mla.flash_mla_with_kvcache(
                q.reshape(bs, ntps, num_head, head_dim),
                k_cache,
                block_tables,
                context_lens,
                self.v_head_size,
                tile_scheduler_metadata,
                num_splits,
                self.scale,
                ntps > 1,  # causal=True when lazy verify
            )

            # o: (bs, ntps, H, v_head_dim) → (q_len, H, v_head_dim)
            o = o.reshape(bs * ntps, o.shape[2], o.shape[3])

        return o


class HopperAttention(AttentionBase):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        v_head_dim,
        attention_type: str = "MLA",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.forward_method = None

        if attention_type == "MLA":
            self.impl = FlashMLAImpl(
                num_heads, head_dim, scale, num_kv_heads, v_head_dim
            )
        elif attention_type == "GQA":
            self.impl = FlashAttentionImpl(
                num_heads,
                head_dim,
                scale,
                num_kv_heads,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """forward."""
        return self.impl.forward(q, k, v, self.k_cache, self.v_cache)

"""FlashMLA (Hopper) dense MLA decode backend.

Wraps ``flash_mla.flash_mla_with_kvcache`` for compressed-KV MLA decode, with
both the dense paged path and the FP8 sparse path (the latter shared with the
DSA family). MLA prefill is handled in ``DeepseekV2Attention.forward`` and must
not reach this backend.
"""

import torch

from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.context.cache.hca import get_hca_context
from dlengine.runtime.context.cache.mla import get_mla_context
from dlengine.runtime.kernel.triton.generic.kv_store import store_kcache
from dlengine.runtime.kernel.triton.hopper.fp8_utils import store_kcache_fp8
from dlengine.runtime.layers.backends.mla.base import MlaAttentionBase


class FlashMlaAttention(MlaAttentionBase):
    """Hopper FlashMLA compressed-KV decode (dense + FP8 sparse)."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
        write_kv_cache: bool = True,
    ):
        import flash_mla

        k_cache = self.k_cache
        context = get_batch_context()

        if k_cache.numel() and not context.is_dummy:
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
            # MLA prefill uses the non-absorbed path in DeepseekV2Attention.
            raise RuntimeError(
                "FlashMlaAttention.forward should not be called during prefill. "
                "MLA prefill is handled in DeepseekV2Attention.forward."
            )

        ntps = context.num_tokens_per_seq
        total_tokens, num_head, head_dim = q.shape
        bs = total_tokens // ntps
        q = q[: bs * ntps]
        context_lens = context.context_lens[0, :bs]
        block_tables = context.block_tables[0, :bs]

        # FP8 KV cache REQUIRES sparse decode — dense_decode_fwd does not
        # support FP8. When sparse_indices is None (e.g. CUDA graph capture
        # warmup), synthesise dummy all-invalid indices so the sparse kernel is
        # still used.
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
            topk = sparse_indices.shape[-1]
            indices_3d = sparse_indices.view(bs, ntps, topk)

            mla_context = get_mla_context()
            sparse_meta = mla_context.sparse_tile_scheduler_metadata
            if sparse_meta is None:
                sparse_meta, _ = flash_mla.get_mla_metadata()

            o, _ = flash_mla.flash_mla_with_kvcache(
                q.reshape(bs, ntps, num_head, head_dim),
                k_cache,
                None,  # block_table (not needed for sparse)
                None,  # cache_seqlens (not needed for sparse)
                self.v_head_dim,
                sparse_meta,
                None,  # num_splits
                self.scale,
                False,  # causal must be False for sparse
                is_fp8_kvcache=True,
                indices=indices_3d,
            )
            mla_context.sparse_tile_scheduler_metadata = sparse_meta
        else:
            # === Dense decode (default) ===
            hca_context = get_hca_context()
            if hca_context.tile_scheduler_metadata is not None:
                tile_scheduler_metadata = hca_context.tile_scheduler_metadata
            else:
                tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

            o, _ = flash_mla.flash_mla_with_kvcache(
                q.reshape(bs, ntps, num_head, head_dim),
                k_cache,
                block_tables,
                context_lens,
                self.v_head_dim,
                tile_scheduler_metadata,
                None,  # num_splits (managed internally)
                self.scale,
                ntps > 1,  # causal=True when lazy verify
                is_fp8_kvcache=k_cache.dtype == torch.float8_e4m3fn,
            )

        return o.reshape(bs * ntps, o.shape[2], o.shape[3])


__all__ = ["FlashMlaAttention"]

"""Cache allocation dispatchers.

The public method names stay on CacheContext for compatibility. Backend
allocation details live in the backend cache modules.
"""

from dlengine.context_v2.cache.csa import (
    allocate_dsv4_compressed_caches,
    allocate_dsv4_compressor_state,
)
from dlengine.context_v2.cache.gdn import allocate_gdn_states, estimate_gdn_state_bytes
from dlengine.context_v2.cache.gqa import allocate_gqa_kvcache
from dlengine.context_v2.cache.hca import allocate_dsv4_kvcache
from dlengine.context_v2.cache.indexer import allocate_indexer_cache
from dlengine.context_v2.cache.mla import allocate_mla_kvcache


class KVCacheAllocatorMixin:
    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks

        if self.mode == "gqa":
            allocate_gqa_kvcache(self)
        elif self.mode == "mla":
            allocate_mla_kvcache(self)
        elif self.mode == "dsv4":
            allocate_dsv4_kvcache(self)
        else:
            raise ValueError(f"Unknown cache mode: {self.mode}")

    def allocate_dsv4_compressed_caches(
        self,
        compress_ratios: list[int],
        max_num_seqs: int,
        max_model_len: int,
        pool_pages_per_ratio: dict[int, int] | None = None,
    ):
        return allocate_dsv4_compressed_caches(
            self,
            compress_ratios,
            max_num_seqs,
            max_model_len,
            pool_pages_per_ratio=pool_pages_per_ratio,
        )

    def allocate_dsv4_compressor_state(
        self,
        compress_ratios: list[int],
        head_dim: int,
        max_num_seqs: int,
    ):
        return allocate_dsv4_compressor_state(
            self,
            compress_ratios,
            head_dim,
            max_num_seqs,
        )

    def allocate_indexer_cache(self, hf_config):
        return allocate_indexer_cache(self, hf_config)

    @staticmethod
    def estimate_gdn_state_bytes(
        hf_config,
        layer_types,
        max_bs: int,
        need_backup: bool = False,
        cache_slots: int = 0,
    ) -> int:
        return estimate_gdn_state_bytes(
            hf_config,
            layer_types,
            max_bs,
            need_backup=need_backup,
            cache_slots=cache_slots,
        )

    def allocate_gdn_states(
        self,
        hf_config,
        layer_types,
        max_bs: int,
        need_backup: bool = False,
        cache_slots: int = 0,
    ):
        return allocate_gdn_states(
            self,
            hf_config,
            layer_types,
            max_bs,
            need_backup=need_backup,
            cache_slots=cache_slots,
        )


__all__ = ["KVCacheAllocatorMixin"]

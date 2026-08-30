"""Cache allocation dispatchers.

The public method names stay on CacheContext for compatibility. Backend
allocation details live in the backend cache modules.
"""

from dlengine.runtime.context.cache._backend import resolve_cache_backend
from dlengine.runtime.context.cache.csa import (
    allocate_dsv4_compressed_caches,
    allocate_dsv4_compressor_state,
)
from dlengine.runtime.context.cache.gdn import (
    allocate_gdn_states,
    estimate_gdn_state_bytes,
)
from dlengine.runtime.context.cache.indexer import allocate_indexer_cache


def allocate_device_tensor(
    context, name: str, shape: tuple[int, ...], dtype, *, zero: bool = False
):
    """Allocate ordinary device storage or PeerAgent-owned Fabric storage."""
    peer_context = getattr(context, "peer_context", None)
    if (
        getattr(context, "peer_fabric_enabled", False)
        and peer_context is not None
        and str(context.device).startswith("cuda")
    ):
        return peer_context.allocate_tensor(name, shape, dtype, zero=zero)

    import torch

    device = torch.device(context.device)
    pin_memory = device.type == "cpu" and torch.cuda.is_available()
    tensor = torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)
    if zero:
        tensor.zero_()
    return tensor


class KVCacheAllocatorMixin:
    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks
        resolve_cache_backend(self.mode).allocate(self)

    def allocate_host_kvcache(self, num_host_kvcache_blocks: int):
        self.num_host_kvcache_blocks = max(0, int(num_host_kvcache_blocks))
        if self.num_host_kvcache_blocks <= 0:
            self.host_kv_cache = None
            return

        backend = resolve_cache_backend(self.mode)
        gpu_kv_cache = self.kv_cache
        local_blocks = self.num_local_kvcache_blocks
        device = self.device
        try:
            self.num_local_kvcache_blocks = self.num_host_kvcache_blocks
            self.device = "cpu"
            backend.allocate(self)
            self.host_kv_cache = self.kv_cache
        finally:
            self.kv_cache = gpu_kv_cache
            self.num_local_kvcache_blocks = local_blocks
            self.device = device

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
        attention_tp: int = 1,
    ) -> int:
        return estimate_gdn_state_bytes(
            hf_config,
            layer_types,
            max_bs,
            need_backup=need_backup,
            cache_slots=cache_slots,
            attention_tp=attention_tp,
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


__all__ = ["KVCacheAllocatorMixin", "allocate_device_tensor"]

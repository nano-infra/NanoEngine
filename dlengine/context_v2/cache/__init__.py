"""Runtime cache context.

CacheContext is initialized once during engine startup and owns cache
configuration plus cache-backed runtime state. Model layers query it for
compute/load/store buffers, while disaggregation code uses it for cache layout
arithmetic and local tensor access.

The implementation is currently split into internal helpers:
- _allocator: local cache buffer allocation.
P2P transfer helpers and RDMA byte-offset math live under ``dlengine.disagg.p2p``.
"""

import dataclasses
from typing import Any, Literal

import torch
import torch.distributed as dist

from dlengine.context_v2.cache import (  # noqa: F401  # noqa: F401  # noqa: F401
    gqa as _gqa_backend,
    hca as _hca_backend,
    hisparse as _hisparse_backend,
    mla as _mla_backend,
)
from dlengine.context_v2.cache._allocator import KVCacheAllocatorMixin
from dlengine.context_v2.cache._registry import (
    CacheBackend,
    get_cache_backend,
    register_cache_backend,
    registered_cache_backends,
)
from dlengine.context_v2.cache.csa import get_csa_context
from dlengine.context_v2.cache.gdn import get_gdn_context, initialize_gdn_cache_state
from dlengine.context_v2.cache.gqa import get_gqa_context
from dlengine.context_v2.cache.hca import get_hca_context
from dlengine.context_v2.cache.hisparse import get_hisparse_context
from dlengine.context_v2.cache.indexer import (
    get_indexer_block_bytes,
    get_indexer_context,
)
from dlengine.context_v2.cache.mla import get_mla_context
from dlengine.logging import get_logger

logger = get_logger("dlengine")


@dataclasses.dataclass(frozen=True)
class MLAHiSparseCapacity:
    """Byte-accurate capacity plan for the MLA HiSparse cache hierarchy."""

    block_size: int
    gpu_cache_budget_bytes: int
    host_cache_budget_bytes: int
    kv_block_bytes: int
    indexer_block_bytes: int
    hot_blocks: int
    hot_tier_bytes: int
    indexer_budget_bytes: int
    indexer_blocks: int
    host_blocks: int
    logical_blocks: int

    @property
    def hot_tokens(self) -> int:
        return self.hot_blocks * self.block_size

    @property
    def indexer_tokens(self) -> int:
        return self.indexer_blocks * self.block_size

    @property
    def host_tokens(self) -> int:
        return self.host_blocks * self.block_size

    @property
    def logical_tokens(self) -> int:
        return self.logical_blocks * self.block_size

    @property
    def buffer_hbm_shortfall_bytes(self) -> int:
        return max(0, self.hot_tier_bytes - self.gpu_cache_budget_bytes)

    @property
    def allocated_indexer_bytes(self) -> int:
        return self.logical_blocks * self.indexer_block_bytes

    @property
    def unused_indexer_budget_bytes(self) -> int:
        return self.indexer_budget_bytes - self.allocated_indexer_bytes

    @property
    def allocated_host_bytes(self) -> int:
        return self.logical_blocks * self.kv_block_bytes

    @property
    def unused_host_budget_bytes(self) -> int:
        return self.host_cache_budget_bytes - self.allocated_host_bytes

    @property
    def limiting_tier(self) -> str:
        if self.indexer_blocks < self.host_blocks:
            return "gpu_indexer"
        if self.host_blocks < self.indexer_blocks:
            return "host_mla"
        return "both"


def plan_mla_hisparse_capacity(
    *,
    gpu_cache_budget: int,
    host_cache_budget: int,
    max_num_seqs: int,
    device_buffer_size: int,
    block_size: int,
    kv_block_bytes: int,
    indexer_block_bytes: int,
) -> MLAHiSparseCapacity:
    """Reserve the hot Buffer first, then size the logical cold tier.

    The Indexer must cover every logical cold token and is resident on the GPU,
    while the MLA KV for those tokens is resident in host memory. Consequently,
    the usable logical capacity is the smaller of the Indexer-backed capacity
    and the host-backed capacity.
    """
    if min(block_size, kv_block_bytes, indexer_block_bytes) <= 0:
        raise ValueError("HiSparse cache block sizes must be positive")
    if max_num_seqs <= 0 or device_buffer_size <= 0:
        raise ValueError("HiSparse sequence count and device buffer must be positive")

    # One extra block per sequence is reserved for the newly generated token.
    hot_blocks_per_seq = (
        device_buffer_size + block_size + block_size - 1
    ) // block_size
    hot_blocks = max_num_seqs * hot_blocks_per_seq
    hot_tier_bytes = hot_blocks * kv_block_bytes

    indexer_budget_bytes = max(0, gpu_cache_budget - hot_tier_bytes)
    indexer_blocks = indexer_budget_bytes // indexer_block_bytes
    host_blocks = max(0, host_cache_budget) // kv_block_bytes

    return MLAHiSparseCapacity(
        block_size=block_size,
        gpu_cache_budget_bytes=max(0, gpu_cache_budget),
        host_cache_budget_bytes=max(0, host_cache_budget),
        kv_block_bytes=kv_block_bytes,
        indexer_block_bytes=indexer_block_bytes,
        hot_blocks=hot_blocks,
        hot_tier_bytes=hot_tier_bytes,
        indexer_budget_bytes=indexer_budget_bytes,
        indexer_blocks=indexer_blocks,
        host_blocks=host_blocks,
        logical_blocks=min(indexer_blocks, host_blocks),
    )


@dataclasses.dataclass
class CacheContext(KVCacheAllocatorMixin):
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_hidden_layers: int
    attention_tp: int
    gpu_memory_utilization: float
    gpu_memory_limit_gb: float | None = None
    host_utilization_per_device: float = 0.0
    # Bytes to reserve out of the utilization budget for state buffers allocated
    # *after* KV sizing (e.g. GDN linear-attention conv/recurrent states).
    reserved_state_bytes: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    mode: Literal["gqa", "mla", "dsv4"] = "gqa"
    num_local_kvcache_blocks = -1
    num_host_kvcache_blocks = 0
    num_remote_kvcache_blocks: dict[str, int] = None
    host_kv_cache: torch.Tensor | None = None

    # used for MLA mode
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    is_fp8_kvcache: bool = False

    # NSA Indexer (V3.2 only)
    index_head_dim: int = 0  # 128 for V3.2, 0 otherwise

    # Control plane: server address and engine ID for disaggregation users.
    ctrl_address: str | None = (
        None  # Control plane server URL (e.g., "http://127.0.0.1:4479")
    )
    ctrl_scope: str | None = None  # Scope for multi-tenant isolation
    engine_id: str | None = None  # Engine ID for agent naming (format: EngineName:rank)
    architecture: str | None = None
    enable_hisparse: bool = False
    max_num_seqs: int = 0
    hisparse_device_buffer_size: int = 0
    # If ctrl_address is provided, engine_id will be fetched from NanoCtrl instead of config

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def _primary_cache_context(self):
        if self.mode == "gqa":
            return get_gqa_context()
        if self.mode == "mla":
            return get_mla_context()
        if self.mode == "dsv4":
            return get_hca_context()
        raise ValueError(f"Unknown cache mode: {self.mode}")

    @property
    def kv_cache(self):
        return self._primary_cache_context().kv_cache

    @kv_cache.setter
    def kv_cache(self, value) -> None:
        self._primary_cache_context().kv_cache = value

    @property
    def gdn_conv_states(self):
        return get_gdn_context().gdn_conv_states

    @gdn_conv_states.setter
    def gdn_conv_states(self, value) -> None:
        get_gdn_context().gdn_conv_states = value

    @property
    def gdn_recurrent_states(self):
        return get_gdn_context().gdn_recurrent_states

    @gdn_recurrent_states.setter
    def gdn_recurrent_states(self, value) -> None:
        get_gdn_context().gdn_recurrent_states = value

    @property
    def gdn_num_slots(self) -> int:
        return get_gdn_context().gdn_num_slots

    @gdn_num_slots.setter
    def gdn_num_slots(self, value: int) -> None:
        get_gdn_context().gdn_num_slots = value

    @property
    def gdn_max_active_slots(self) -> int:
        return get_gdn_context().gdn_max_active_slots

    @gdn_max_active_slots.setter
    def gdn_max_active_slots(self, value: int) -> None:
        get_gdn_context().gdn_max_active_slots = value

    @property
    def indexer_cache(self):
        return get_indexer_context().indexer_cache

    @indexer_cache.setter
    def indexer_cache(self, value) -> None:
        get_indexer_context().indexer_cache = value

    @property
    def dsv4_compress_ratios(self):
        return get_csa_context().dsv4_compress_ratios

    @dsv4_compress_ratios.setter
    def dsv4_compress_ratios(self, value) -> None:
        get_csa_context().dsv4_compress_ratios = value

    @property
    def dsv4_compressed_caches(self):
        return get_csa_context().dsv4_compressed_caches

    @dsv4_compressed_caches.setter
    def dsv4_compressed_caches(self, value) -> None:
        get_csa_context().dsv4_compressed_caches = value

    @property
    def dsv4_compressed_caches_flat(self):
        return get_csa_context().dsv4_compressed_caches_flat

    @dsv4_compressed_caches_flat.setter
    def dsv4_compressed_caches_flat(self, value) -> None:
        get_csa_context().dsv4_compressed_caches_flat = value

    @property
    def dsv4_layers_per_ratio(self):
        return get_csa_context().dsv4_layers_per_ratio

    @dsv4_layers_per_ratio.setter
    def dsv4_layers_per_ratio(self, value) -> None:
        get_csa_context().dsv4_layers_per_ratio = value

    @property
    def dsv4_layer_to_ratio_idx(self):
        return get_csa_context().dsv4_layer_to_ratio_idx

    @dsv4_layer_to_ratio_idx.setter
    def dsv4_layer_to_ratio_idx(self, value) -> None:
        get_csa_context().dsv4_layer_to_ratio_idx = value

    @property
    def dsv4_compressed_pool_config(self):
        return get_csa_context().dsv4_compressed_pool_config

    @dsv4_compressed_pool_config.setter
    def dsv4_compressed_pool_config(self, value) -> None:
        get_csa_context().dsv4_compressed_pool_config = value

    @property
    def dsv4_compressed_dummy_page(self):
        return get_csa_context().dsv4_compressed_dummy_page

    @dsv4_compressed_dummy_page.setter
    def dsv4_compressed_dummy_page(self, value) -> None:
        get_csa_context().dsv4_compressed_dummy_page = value

    @property
    def dsv4_compressor_kv_flat(self):
        return get_csa_context().dsv4_compressor_kv_flat

    @dsv4_compressor_kv_flat.setter
    def dsv4_compressor_kv_flat(self, value) -> None:
        get_csa_context().dsv4_compressor_kv_flat = value

    @property
    def dsv4_compressor_score_flat(self):
        return get_csa_context().dsv4_compressor_score_flat

    @dsv4_compressor_score_flat.setter
    def dsv4_compressor_score_flat(self, value) -> None:
        get_csa_context().dsv4_compressor_score_flat = value

    @property
    def dsv4_compressor_counts_flat(self):
        return get_csa_context().dsv4_compressor_counts_flat

    @dsv4_compressor_counts_flat.setter
    def dsv4_compressor_counts_flat(self, value) -> None:
        get_csa_context().dsv4_compressor_counts_flat = value

    def __post_init__(self):
        free, total = torch.cuda.mem_get_info(self.device)
        real_total = total
        if self.gpu_memory_limit_gb is not None:
            total = min(total, self.gpu_memory_limit_gb * 1024**3)
        used = real_total - free  # real used
        memory_stats = torch.cuda.memory_stats(self.device)
        peak = memory_stats.get("allocated_bytes.all.peak", 0)
        current = memory_stats.get("allocated_bytes.all.current", 0)

        backend = get_cache_backend(self.mode)
        backend.configure(self)
        kv_block_bytes = backend.get_block_bytes(self)
        indexer_block_bytes = get_indexer_block_bytes(self)
        gpu_cache_budget = int(
            total * self.gpu_memory_utilization
            - used
            - peak
            + current
            - self.reserved_state_bytes
        )

        is_mla_hisparse = (
            self.enable_hisparse
            and self.mode == "mla"
            and indexer_block_bytes > 0
            and bool(self.ctrl_address)
        )
        if is_mla_hisparse:
            host_budget_bytes = int(
                max(0.0, float(self.host_utilization_per_device or 0.0))
                * 1024**3
            )
            capacity = plan_mla_hisparse_capacity(
                gpu_cache_budget=gpu_cache_budget,
                host_cache_budget=host_budget_bytes,
                max_num_seqs=self.max_num_seqs,
                device_buffer_size=self.hisparse_device_buffer_size,
                block_size=self.block_size,
                kv_block_bytes=kv_block_bytes,
                indexer_block_bytes=indexer_block_bytes,
            )
            self.num_local_kvcache_blocks = capacity.logical_blocks
            # Do not allocate cold pages which cannot be indexed. Logical,
            # host-cold, and Indexer page counts deliberately stay identical.
            self.num_host_kvcache_blocks = self.num_local_kvcache_blocks
            logger.info(
                "Rank%s MLA HiSparse HBM plan: cache_budget=%.2f GiB, "
                "hot_buffer=%s tokens/%.2f GiB, remaining_for_indexer=%.2f GiB",
                dist.get_rank(),
                capacity.gpu_cache_budget_bytes / 1024**3,
                capacity.hot_tokens,
                capacity.hot_tier_bytes / 1024**3,
                capacity.indexer_budget_bytes / 1024**3,
            )
            logger.info(
                "Rank%s MLA HiSparse capacity ceilings: gpu_indexer=%s tokens, "
                "host_mla=%s tokens/%.2f GiB; selected=%s tokens, limiter=%s",
                dist.get_rank(),
                capacity.indexer_tokens,
                capacity.host_tokens,
                capacity.host_cache_budget_bytes / 1024**3,
                capacity.logical_tokens,
                capacity.limiting_tier,
            )
            logger.info(
                "Rank%s MLA HiSparse unused capacity after min(): "
                "gpu_indexer_hbm=%.2f GiB, host_mla=%.2f GiB",
                dist.get_rank(),
                capacity.unused_indexer_budget_bytes / 1024**3,
                capacity.unused_host_budget_bytes / 1024**3,
            )
            if capacity.buffer_hbm_shortfall_bytes > 0:
                logger.error(
                    "Rank%s MLA HiSparse Buffer exceeds the GPU cache budget "
                    "by %.2f GiB; no HBM remains for the Indexer",
                    dist.get_rank(),
                    capacity.buffer_hbm_shortfall_bytes / 1024**3,
                )
        else:
            block_bytes = kv_block_bytes + indexer_block_bytes
            self.num_local_kvcache_blocks = gpu_cache_budget // block_bytes
            self.num_host_kvcache_blocks, host_budget_bytes = (
                self._compute_host_kvcache_blocks(block_bytes)
            )

        logger.debug(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )
        if self.num_host_kvcache_blocks > 0:
            logger.info(
                "Rank%s host KV cache: %s blocks (%.2f GiB budget)",
                dist.get_rank(),
                self.num_host_kvcache_blocks,
                host_budget_bytes / 1024**3,
            )

        assert self.num_local_kvcache_blocks > 0

        initialize_gdn_cache_state(self)

    def _compute_host_kvcache_blocks(self, block_bytes: int) -> tuple[int, int]:
        budget_gib = max(0.0, float(self.host_utilization_per_device or 0.0))
        if budget_gib <= 0 or block_bytes <= 0:
            return 0, 0
        budget_bytes = int(budget_gib * 1024**3)
        return max(0, budget_bytes // block_bytes), budget_bytes


_CACHE_CONTEXT: CacheContext


def get_cache_context():
    return _CACHE_CONTEXT


def set_cache_context(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_hidden_layers: int,
    attention_tp: int,
    gpu_memory_utilization: float,
    gpu_memory_limit_gb: float | None = None,
    host_utilization_per_device: float = 0.0,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    index_head_dim: int = 0,
    is_fp8_kvcache: bool = False,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mode: Literal["gqa", "mla", "dsv4"] = "gqa",
    ctrl_address: str | None = None,
    ctrl_scope: str | None = None,
    engine_id: str | None = None,
    architecture: str | None = None,
    enable_hisparse: bool = False,
    max_num_seqs: int = 0,
    hisparse_device_buffer_size: int = 0,
    reserved_state_bytes: int = 0,
):
    global _CACHE_CONTEXT
    _CACHE_CONTEXT = CacheContext(
        reserved_state_bytes=reserved_state_bytes,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_head_dim=index_head_dim,
        is_fp8_kvcache=is_fp8_kvcache,
        block_size=block_size,
        num_hidden_layers=num_hidden_layers,
        attention_tp=attention_tp,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_limit_gb=gpu_memory_limit_gb,
        host_utilization_per_device=host_utilization_per_device,
        device=device,
        dtype=dtype,
        mode=mode,
        ctrl_address=ctrl_address,
        ctrl_scope=ctrl_scope,
        engine_id=engine_id,
        architecture=architecture,
        enable_hisparse=enable_hisparse,
        max_num_seqs=max_num_seqs,
        hisparse_device_buffer_size=hisparse_device_buffer_size,
    )
    return _CACHE_CONTEXT

"""Runtime cache context.

CacheContext is initialized once during engine startup and owns cache
configuration plus cache-backed runtime state. Model layers query it for
compute/load/store buffers, while disaggregation code uses it for cache layout
arithmetic and local tensor access.

The implementation is currently split into internal mixins:
- _allocator: local cache buffer allocation.
- _layout: byte-offset, stride, gather/scatter, and layout transformation math.
P2P transfer helpers live under ``dlengine.disagg.p2p`` and are still mixed in
here temporarily to preserve the current public API.
"""

import dataclasses
from typing import Any, Literal

import torch
import torch.distributed as dist

from dlengine.context_v2.cache._allocator import KVCacheAllocatorMixin
from dlengine.context_v2.cache._layout import CacheLayoutMixin
from dlengine.context_v2.cache.gdn import initialize_gdn_cache_state
from dlengine.context_v2.cache.gqa import configure_gqa_cache, get_gqa_block_bytes
from dlengine.context_v2.cache.hca import configure_dsv4_cache, get_dsv4_block_bytes
from dlengine.context_v2.cache.indexer import get_indexer_block_bytes
from dlengine.context_v2.cache.mla import configure_mla_cache, get_mla_block_bytes
from dlengine.context_v2.peer import PeerAgentContext
from dlengine.disagg.p2p.cache_transfer import (
    initialize_migration_state,
    KVMigratorMixin,
    select_peer_device,
)
from dlengine.logging import get_logger

logger = get_logger("dlengine")


@dataclasses.dataclass
class CacheContext(CacheLayoutMixin, KVCacheAllocatorMixin, KVMigratorMixin):
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_hidden_layers: int
    attention_tp: int
    gpu_memory_utilization: float
    gpu_memory_limit_gb: float | None = None
    # Bytes to reserve out of the utilization budget for state buffers allocated
    # *after* KV sizing (e.g. GDN linear-attention conv/recurrent states).
    reserved_state_bytes: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    mode: Literal["gqa", "mla", "dsv4"] = "gqa"
    num_local_kvcache_blocks = -1
    num_remote_kvcache_blocks: dict[str, int] = None
    kv_cache: torch.Tensor = None
    gdn_conv_states: torch.Tensor | None = None

    # DSv4 compressed KV caches (per-layer, separate from SWA paged cache)
    # Shape per layer: [max_num_seqs, max_compressed_tokens, 1, 584] uint8
    dsv4_compressed_caches: dict[int, torch.Tensor] | None = None
    dsv4_compress_ratios: list[int] | None = None  # per-layer compress ratios
    gdn_recurrent_states: torch.Tensor | None = None
    selected_nic: str | None = None
    endpoints: dict[str, dict[int, Any]] = None  # RDMAEndpoint or RDMALazyPeer

    # used for MLA mode
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    is_fp8_kvcache: bool = False

    # NSA Indexer (V3.2 only)
    index_head_dim: int = 0  # 128 for V3.2, 0 otherwise
    indexer_cache: Any = None  # IndexerCache instance, set after allocation

    # Control plane: server address and engine ID for centralized connection
    ctrl_address: str | None = (
        None  # Control plane server URL (e.g., "http://127.0.0.1:4479")
    )
    ctrl_scope: str | None = None  # Scope for multi-tenant isolation
    engine_id: str | None = None  # Engine ID for agent naming (format: EngineName:rank)
    peer_agent_context: PeerAgentContext | None = None
    # If ctrl_address is provided, engine_id will be fetched from NanoCtrl instead of config

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def __post_init__(self):
        free, total = torch.cuda.mem_get_info()
        if self.gpu_memory_limit_gb is not None:
            total = min(total, self.gpu_memory_limit_gb * 1024**3)
        used = torch.cuda.mem_get_info()[1] - free  # real used
        memory_stats = torch.cuda.memory_stats()
        peak = memory_stats["allocated_bytes.all.peak"]
        current = memory_stats["allocated_bytes.all.current"]

        if self.mode == "gqa":
            configure_gqa_cache(self)
            block_bytes = get_gqa_block_bytes(self)
        elif self.mode == "mla":
            configure_mla_cache(self)
            block_bytes = get_mla_block_bytes(self)
        elif self.mode == "dsv4":
            configure_dsv4_cache(self)
            block_bytes = get_dsv4_block_bytes(self)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        block_bytes += get_indexer_block_bytes(self)

        self.num_local_kvcache_blocks = (
            int(
                total * self.gpu_memory_utilization
                - used
                - peak
                + current
                - self.reserved_state_bytes
            )
            // block_bytes
        )

        logger.debug(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )

        assert self.num_local_kvcache_blocks > 0

        self.selected_nic = select_peer_device()
        initialize_migration_state(self)
        initialize_gdn_cache_state(self)


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
        device=device,
        dtype=dtype,
        mode=mode,
        ctrl_address=ctrl_address,
        ctrl_scope=ctrl_scope,
        engine_id=engine_id,
    )
    return _CACHE_CONTEXT

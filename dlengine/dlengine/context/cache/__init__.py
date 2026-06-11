import dataclasses
from typing import Any, Literal

import dlslime
import torch
import torch.distributed as dist
from dlengine.context.cache.allocator import KVCacheAllocatorMixin
from dlengine.context.cache.layout import CacheLayoutMixin
from dlengine.context.cache.migrator import KVMigratorMixin
from dlengine.context.peer_agent import PeerAgentContext
from dlengine.logging import get_logger

logger = get_logger("dlengine")

# FP8 quantization tile size (matches deep_gemm per_token_cast_to_fp8)
_FP8_QUANT_TILE_SIZE = 128

# NSA Indexer FP8 cache quantization block size
INDEXER_QUANT_BLOCK_SIZE = _FP8_QUANT_TILE_SIZE


@dataclasses.dataclass
class CacheContext(CacheLayoutMixin, KVCacheAllocatorMixin, KVMigratorMixin):
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_hidden_layers: int
    attention_tp: int
    gpu_memory_utilization: float
    gpu_memory_limit_gb: float | None = None
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
            assert self.attention_tp <= self.num_kv_heads
        elif self.mode == "mla":
            assert self.attention_tp == 1
            assert self.block_size == 64, "MLA mode only support block_size=64"
            self.num_kv_heads = 1
            self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        elif self.mode == "dsv4":
            assert self.attention_tp == 1
            assert self.block_size % 64 == 0, "DSv4 block_size must be multiple of 64"
            self.num_kv_heads = 1
            self.head_dim = 512  # fixed for DSv4
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # DSv4 FP8 packed format: 584 bytes per token
        _DSV4_BYTES_PER_TOKEN = 584

        if self.mode == "mla" and self.is_fp8_kvcache:
            # FP8 MLA layout per token:
            #   NoPE:  kv_lora_rank bytes (float8_e4m3fn)
            #   Scale: (kv_lora_rank // tile_size) * 4 bytes (float32 per tile)
            #   RoPE:  qk_rope_head_dim * 2 bytes (bfloat16)
            nope_bytes = self.kv_lora_rank
            scale_bytes = (self.kv_lora_rank // _FP8_QUANT_TILE_SIZE) * 4
            rope_bytes = self.qk_rope_head_dim * 2
            self._fp8_head_dim = nope_bytes + scale_bytes + rope_bytes
            block_bytes = (
                self.num_hidden_layers
                * self.block_size
                * 1  # num_kv_heads
                * self._fp8_head_dim
                * 1  # fp8 element size
            )
        elif self.mode == "dsv4":
            self._fp8_head_dim = 0
            # SWA paged cache: [num_layers, num_pages, page_size, 1, 584] uint8
            # Each block = page_size * 584 bytes per layer
            block_bytes = (
                self.num_hidden_layers * self.block_size * _DSV4_BYTES_PER_TOKEN
            )
        else:
            self._fp8_head_dim = 0
            block_bytes = (
                self.num_hidden_layers
                * self.block_size
                * self.num_local_kv_heads
                * self.head_dim
                * self.dtype.itemsize
            )
        if self.mode == "gqa":
            block_bytes *= 2

        # Account for NSA indexer FP8 cache (V3.2 only)
        if self.index_head_dim > 0:
            indexer_bytes_per_token = (
                self.index_head_dim
                + self.index_head_dim // INDEXER_QUANT_BLOCK_SIZE * 4
            )
            block_bytes += (
                self.num_hidden_layers * self.block_size * indexer_bytes_per_token
            )

        self.num_local_kvcache_blocks = (
            int(total * self.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        logger.debug(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )

        assert self.num_local_kvcache_blocks > 0

        available_nics = dlslime.available_nic()
        selected_nic_idx = dist.get_rank() % len(available_nics)
        self.selected_nic = available_nics[selected_nic_idx]
        assert self.selected_nic

        self.endpoints = {}
        self.num_remote_kvcache_blocks = {}
        self.remote_max_num_seqs: dict[str, int] = {}  # engine_id -> max_num_seqs
        self.remote_gdn_num_slots: dict[str, int] = {}  # engine_id -> gdn_num_slots
        # engine_id -> remote engine's attention_tp (for PD + GQA peer mapping).
        self.remote_attention_tp: dict[str, int] = {}
        self.gdn_num_slots: int = 0  # actual dim-1 of gdn tensors
        # DSv4 (S2.5): per-remote-engine pool sizes for stride math.
        # remote_compressed_pool_pages[engine_id][ratio] = num_pages on that engine
        self.remote_compressed_pool_pages: dict[str, dict[int, int]] = {}
        # remote_dsv4_max_slots[engine_id] = max_num_seqs on that engine
        self.remote_dsv4_max_slots: dict[str, int] = {}
        # remote_dsv4_num_layers_per_ratio[engine_id][ratio] = num layers using that ratio
        self.remote_dsv4_num_layers_per_ratio: dict[str, dict[int, int]] = {}
        self._local_mr_handler: int | None = None  # local MR handler for kv_cache
        self._local_gdn_conv_mr_handler: int | None = None
        self._local_gdn_recurrent_mr_handler: int | None = None
        self._local_indexer_mr_handler: int | None = (
            None  # local MR handler for indexer_cache
        )
        # DSv4 (S2.4): per-ratio MR handlers for compressed cache and compressor state.
        self._local_dsv4_compressed_mr_handlers: dict[int, int] = {}
        self._local_dsv4_compressor_kv_mr_handlers: dict[int, int] = {}
        self._local_dsv4_compressor_score_mr_handlers: dict[int, int] = {}
        self._local_dsv4_compressor_counts_mr_handlers: dict[int, int] = {}
        # NOTE: Remote MR handler caching removed from app layer
        # PeerAgent handles MR info caching via pubsub (mr_update events)
        # register_remote_memory_region is idempotent at endpoint layer
        self._engine_info_cache: tuple[float, dict[str, dict]] | None = (
            None  # (timestamp, engine_id -> engine_info_dict)
        )


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
    mode: Literal["gqa", "mla"] = "gqa",
    ctrl_address: str | None = None,
    ctrl_scope: str | None = None,
    engine_id: str | None = None,
):
    global _CACHE_CONTEXT
    _CACHE_CONTEXT = CacheContext(
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

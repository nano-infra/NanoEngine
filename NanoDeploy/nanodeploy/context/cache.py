import concurrent.futures
import dataclasses
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import dlslime
import numpy as np
import torch
import torch.distributed as dist
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.context.peer_agent import PeerAgentContext
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")

# PeerAgent path: buffer ID for kv_cache registration
_KV_CACHE_BUFFER_ID = "kv_cache"

# FP8 quantization tile size (matches deep_gemm per_token_cast_to_fp8)
_FP8_QUANT_TILE_SIZE = 128

# NSA Indexer FP8 cache quantization block size
INDEXER_QUANT_BLOCK_SIZE = _FP8_QUANT_TILE_SIZE

# Cache TTL for engine_info from NanoCtrl (seconds)
# Engine registration rarely changes, cache forever by default (inf means never expire)
# To refresh, call invalidate_engine_info_cache() or restart the engine
_ENGINE_INFO_CACHE_TTL = float("inf")


@dataclasses.dataclass
class CacheContext:
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

    def block_stride(self, block_idx: int):
        if self.is_fp8_kvcache and self.mode == "mla":
            # Physical stride uses (block_size + 1) due to padding row for
            # FlashMLA alignment.  The tensor is allocated with block_size+1
            # rows then sliced to block_size, so the underlying storage pitch
            # is (block_size + 1) * head_dim bytes per block.
            return block_idx * (self.block_size + 1) * 1 * self._fp8_head_dim * 1
        return (
            block_idx
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    def local_layer_stride(self, layer_idx: int, block_idx: int):
        return (
            self.block_stride(self.num_local_kvcache_blocks)
        ) * layer_idx + self.block_stride(block_idx)

    def remote_layer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return (
            self.block_stride(self.num_remote_kvcache_blocks[remote_engine_id])
        ) * layer_idx + self.block_stride(block_idx)

    def local_kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int):
        return self.local_layer_stride(
            self.num_hidden_layers, 0
        ) * kv_idx + self.local_layer_stride(layer_idx, block_idx)

    def remote_kv_stride(
        self, kv_idx: int, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return self.remote_layer_stride(
            self.num_hidden_layers, 0, remote_engine_id
        ) * kv_idx + self.remote_layer_stride(layer_idx, block_idx, remote_engine_id)

    def gdn_conv_stride(self, layer_idx: int, slot_idx: int) -> int:
        if self.gdn_conv_states is None:
            return -1
        return (
            layer_idx * self.gdn_conv_states.stride(0)
            + slot_idx * self.gdn_conv_states.stride(1)
        ) * self.gdn_conv_states.element_size()

    def gdn_recurrent_stride(self, layer_idx: int, slot_idx: int) -> int:
        if self.gdn_recurrent_states is None:
            return -1
        return (
            layer_idx * self.gdn_recurrent_states.stride(0)
            + slot_idx * self.gdn_recurrent_states.stride(1)
        ) * self.gdn_recurrent_states.element_size()

    def remote_gdn_conv_stride(
        self, layer_idx: int, slot_idx: int, remote_engine_id: str
    ) -> int:
        """Compute GDN conv state offset for a REMOTE engine's tensor layout."""
        if self.gdn_conv_states is None:
            return -1
        remote_num_slots = self.remote_gdn_num_slots.get(remote_engine_id, 0)
        if remote_num_slots == 0:
            # Fallback: assume same layout as local
            remote_num_slots = self.gdn_num_slots
        remote_stride0 = remote_num_slots * self.gdn_conv_states.stride(1)
        return (
            layer_idx * remote_stride0 + slot_idx * self.gdn_conv_states.stride(1)
        ) * self.gdn_conv_states.element_size()

    def remote_gdn_recurrent_stride(
        self, layer_idx: int, slot_idx: int, remote_engine_id: str
    ) -> int:
        """Compute GDN recurrent state offset for a REMOTE engine's tensor layout."""
        if self.gdn_recurrent_states is None:
            return -1
        remote_num_slots = self.remote_gdn_num_slots.get(remote_engine_id, 0)
        if remote_num_slots == 0:
            remote_num_slots = self.gdn_num_slots
        remote_stride0 = remote_num_slots * self.gdn_recurrent_states.stride(1)
        return (
            layer_idx * remote_stride0 + slot_idx * self.gdn_recurrent_states.stride(1)
        ) * self.gdn_recurrent_states.element_size()

    def gdn_conv_slot_num_bytes(self) -> int:
        return self.gdn_conv_states.stride(1) * self.gdn_conv_states.element_size()

    def gdn_recurrent_slot_num_bytes(self) -> int:
        return (
            self.gdn_recurrent_states.stride(1)
            * self.gdn_recurrent_states.element_size()
        )

    # ------------------------------------------------------------------
    # NSA Indexer cache stride helpers (PD disaggregation)
    # ------------------------------------------------------------------
    # IndexerCache.buffer shape: (num_layers, num_pages, page_size * bytes_per_token)
    # All offsets are in bytes (uint8 buffer).

    def indexer_page_num_bytes(self) -> int:
        """Bytes per page (one block) in the indexer cache."""
        if self.indexer_cache is None:
            return 0
        return self.indexer_cache.page_size * self.indexer_cache.bytes_per_token

    def local_indexer_stride(self, layer_idx: int, block_idx: int) -> int:
        """Byte offset for (layer, block) in the local indexer buffer."""
        page_bytes = self.indexer_page_num_bytes()
        return (layer_idx * self.num_local_kvcache_blocks + block_idx) * page_bytes

    def remote_indexer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ) -> int:
        """Byte offset for (layer, block) in a remote engine's indexer buffer."""
        page_bytes = self.indexer_page_num_bytes()
        return (
            layer_idx * self.num_remote_kvcache_blocks[remote_engine_id] + block_idx
        ) * page_bytes

    # ------------------------------------------------------------------
    # DSv4 compressed cache + compressor scratch state stride helpers (S2.3)
    # ------------------------------------------------------------------
    # All buffers are flat per ratio:
    #   compressed_caches_flat[ratio]:    [num_layers, num_pages+1, page_size, 1, 584] uint8
    #   compressor_kv_flat[ratio]:        [num_layers, max_slots+1, coeff*ratio, coeff*head_dim] fp32
    #   compressor_score_flat[ratio]:     same shape as kv
    #   compressor_counts_flat[ratio]:    [num_layers, max_slots+1] int32

    def compressed_page_bytes(self, ratio: int) -> int:
        """Bytes per page in the DSv4 compressed cache for a given ratio."""
        cfg = getattr(self, "dsv4_compressed_pool_config", {}).get(ratio)
        if cfg is None:
            return 0
        _num_pages, page_size, _max_blocks = cfg
        _DSV4_BYTES_PER_TOKEN = 584
        return page_size * _DSV4_BYTES_PER_TOKEN

    def local_compressed_stride(
        self, ratio: int, ratio_layer_idx: int, page_idx: int
    ) -> int:
        """Byte offset for (layer-within-ratio, page) in the LOCAL flat compressed cache."""
        page_bytes = self.compressed_page_bytes(ratio)
        cfg = self.dsv4_compressed_pool_config[ratio]
        num_pages = cfg[0] + 1  # include +1 dummy
        return (ratio_layer_idx * num_pages + page_idx) * page_bytes

    def remote_compressed_stride(
        self,
        ratio: int,
        ratio_layer_idx: int,
        page_idx: int,
        remote_engine_id: str,
    ) -> int:
        """Byte offset for (layer-within-ratio, page) on a REMOTE engine."""
        page_bytes = self.compressed_page_bytes(ratio)
        remote_pages = self.remote_compressed_pool_pages.get(remote_engine_id, {}).get(
            ratio
        )
        if remote_pages is None:
            return -1
        return (ratio_layer_idx * (remote_pages + 1) + page_idx) * page_bytes

    def _compressor_state_row_bytes(self, ratio: int, kind: str) -> int:
        """Bytes per (layer, slot) row in the compressor scratch buffers."""
        if kind == "counts":
            return 4  # int32 scalar per slot
        # kv / score: coeff*ratio * coeff*head_dim * fp32
        coeff = 2 if ratio == 4 else 1
        head_dim = 512  # DSv4 fixed
        return coeff * ratio * coeff * head_dim * 4

    def _local_max_slots_plus_dummy(self, ratio: int) -> int:
        """Get the LOCAL max_slots (+1 for dummy) from the compressor state shape."""
        kv = getattr(self, "dsv4_compressor_kv_flat", {}).get(ratio)
        if kv is None:
            return 0
        return kv.shape[1]  # already includes the +1 dummy

    def local_compressor_state_stride(
        self, ratio: int, ratio_layer_idx: int, slot: int, kind: str
    ) -> int:
        """Byte offset for compressor scratch state on the LOCAL engine.

        kind is one of 'kv', 'score', 'counts'.
        """
        row_bytes = self._compressor_state_row_bytes(ratio, kind)
        slots = self._local_max_slots_plus_dummy(ratio)
        return (ratio_layer_idx * slots + slot) * row_bytes

    def remote_compressor_state_stride(
        self,
        ratio: int,
        ratio_layer_idx: int,
        slot: int,
        kind: str,
        remote_engine_id: str,
    ) -> int:
        """Byte offset for compressor scratch state on a REMOTE engine."""
        row_bytes = self._compressor_state_row_bytes(ratio, kind)
        remote_max_slots = self.remote_dsv4_max_slots.get(remote_engine_id, 0)
        if remote_max_slots == 0:
            remote_max_slots = self._local_max_slots_plus_dummy(ratio) - 1
        # +1 for dummy slot
        return (ratio_layer_idx * (remote_max_slots + 1) + slot) * row_bytes

    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks

        if self.mode == "mla" and self.is_fp8_kvcache:
            # FP8 MLA: allocate (block_size+1) rows per block for stride padding,
            # then slice back to block_size.  This ensures the FlashMLA kernel
            # never reads out-of-bounds on the last row.
            kv_cache_padded = torch.empty(
                1,
                self.num_hidden_layers,
                self.num_local_kvcache_blocks,
                self.block_size + 1,
                1,  # num_kv_heads
                self._fp8_head_dim,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            self.kv_cache = kv_cache_padded[:, :, :, : self.block_size, :, :]
        elif self.mode == "dsv4":
            _DSV4_BYTES_PER_TOKEN = 584
            # SWA paged cache: [num_layers, num_pages+1, page_size, 1, 584] uint8
            # Extra +1 page is a "dummy" absorbing invalid writes (graph-safe).
            # flash_mla reads this via sparse indices (MODEL1 code path).
            self.kv_cache = torch.zeros(
                self.num_hidden_layers,
                self.num_local_kvcache_blocks + 1,  # +1 dummy page
                self.block_size,
                1,  # num_kv_heads (always 1)
                _DSV4_BYTES_PER_TOKEN,
                dtype=torch.uint8,
                device=self.device,
            )
            logger.info(
                f"DSv4 SWA cache: {self.kv_cache.shape} (incl dummy page), "
                f"{self.kv_cache.nelement() / 1e9:.2f} GB"
            )
        else:
            kv_count = 2 if self.mode == "gqa" else 1
            self.kv_cache = torch.empty(
                kv_count,
                self.num_hidden_layers,
                self.num_local_kvcache_blocks,
                self.block_size,
                self.num_local_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )

    def allocate_dsv4_compressed_caches(
        self,
        compress_ratios: list[int],
        max_num_seqs: int,
        max_model_len: int,
        pool_pages_per_ratio: dict[int, int] | None = None,
    ):
        """Allocate per-layer compressed KV caches for DSv4 (paged shared pool).

        Each compression ratio gets its own pool of pages, sized either by
        ``pool_pages_per_ratio[ratio]`` (when provided and > 0) or by the
        worst case ``ceil(max_num_seqs * max_compressed / page_size)``.
        Pages are shared across sequences and assigned by the C++
        CompressedBlockManager via per-seq block tables.

        Cache shape per layer per ratio:
            [num_pages + 1, page_size=2, 1, 584] uint8
        page_size=2 ensures the per-page stride (2 * 584) is 16-byte aligned
        for flash_mla's 128-bit vector loads. The +1 dummy page absorbs
        graph-safe invalid writes.
        """
        _DSV4_BYTES_PER_TOKEN = 584
        compressed_page_size = 2
        self.dsv4_compress_ratios = compress_ratios
        # Per-layer compressed cache *view* into per-ratio flat buffer (S2.1).
        # layer_idx -> uint8 tensor [num_pages+1, 2, 1, 584] (a view, not a copy)
        self.dsv4_compressed_caches: dict[int, torch.Tensor] = {}
        # Per-ratio FLAT compressed cache (one buffer for all layers in this
        # ratio).  Shape: [num_layers_for_ratio, num_pages+1, page_size, 1, 584].
        # Single MR registration per ratio for RDMA migration efficiency.
        self.dsv4_compressed_caches_flat: dict[int, torch.Tensor] = {}
        # ratio -> ordered list of model layer_idx that use this ratio
        self.dsv4_layers_per_ratio: dict[int, list[int]] = {}
        # layer_idx -> position of this layer within its ratio's flat tensor
        self.dsv4_layer_to_ratio_idx: dict[int, int] = {}
        # Per-ratio pool config (used by Scheduler.configure_compressed_pools)
        # ratio -> (num_pages, page_size, max_blocks_per_seq)
        self.dsv4_compressed_pool_config: dict[int, tuple[int, int, int]] = {}
        # Per-ratio dummy page id (last index, used to pad block_tables on
        # the Python side for invalid / unused batch positions).
        self.dsv4_compressed_dummy_page: dict[int, int] = {}
        pool_pages_per_ratio = pool_pages_per_ratio or {}

        total_bytes = 0
        # Group layers by ratio to compute pool size once per ratio.
        unique_ratios = sorted({r for r in compress_ratios if r > 0})
        for ratio in unique_ratios:
            max_compressed = (max_model_len // ratio + 63) // 64 * 64
            max_blocks_per_seq = (
                max_compressed + compressed_page_size - 1
            ) // compressed_page_size
            # Worst-case: every seq fills its full reservation at the same time.
            worst_case_pages = max_num_seqs * max_blocks_per_seq
            override = pool_pages_per_ratio.get(ratio, 0)
            num_pages = override if override > 0 else worst_case_pages
            self.dsv4_compressed_pool_config[ratio] = (
                num_pages,
                compressed_page_size,
                max_blocks_per_seq,
            )
            self.dsv4_compressed_dummy_page[ratio] = num_pages  # last index = dummy
            # Collect layer indices using this ratio (in model order)
            layers_for_ratio = [
                i for i, lr in enumerate(compress_ratios) if lr == ratio
            ]
            self.dsv4_layers_per_ratio[ratio] = layers_for_ratio
            n_layers = len(layers_for_ratio)
            # Allocate single flat buffer for this ratio.
            flat = torch.zeros(
                n_layers,
                num_pages + 1,  # +1 dummy page
                compressed_page_size,
                1,
                _DSV4_BYTES_PER_TOKEN,
                dtype=torch.uint8,
                device=self.device,
            )
            self.dsv4_compressed_caches_flat[ratio] = flat
            total_bytes += flat.nelement()
            # Expose per-layer views (existing API the model layer expects).
            for ratio_layer_idx, layer_idx in enumerate(layers_for_ratio):
                self.dsv4_compressed_caches[layer_idx] = flat[ratio_layer_idx]
                self.dsv4_layer_to_ratio_idx[layer_idx] = ratio_layer_idx
        if total_bytes > 0:
            sizes_str = ", ".join(
                f"ratio={r}: {p[0]} pages × {p[1]} tok × "
                f"{len(self.dsv4_layers_per_ratio[r])} layers"
                for r, p in self.dsv4_compressed_pool_config.items()
            )
            logger.info(
                f"DSv4 compressed caches (flat per ratio): "
                f"{len(self.dsv4_compressed_caches)} layer views, "
                f"total {total_bytes / 1e9:.2f} GB ({sizes_str})"
            )

    def allocate_dsv4_compressor_state(
        self,
        compress_ratios: list[int],
        head_dim: int,
        max_num_seqs: int,
    ):
        """Allocate flat per-ratio compressor scratch state buffers (S2.2).

        For RDMA migration we need single contiguous buffers per (ratio, kind)
        spanning all layers using that ratio.  Each per-layer Compressor will
        hold a slice view (`flat[ratio_layer_idx]`) so existing
        `_kv_states[seq_slot]` indexing keeps working.

        Tensor shapes (per ratio):
          dsv4_compressor_kv_flat[ratio]:    [num_layers, max_slots+1, coeff*ratio, coeff*head_dim] fp32
          dsv4_compressor_score_flat[ratio]: same, init to -inf
          dsv4_compressor_counts_flat[ratio]: [num_layers, max_slots+1] int32
        """
        self.dsv4_compressor_kv_flat: dict[int, torch.Tensor] = {}
        self.dsv4_compressor_score_flat: dict[int, torch.Tensor] = {}
        self.dsv4_compressor_counts_flat: dict[int, torch.Tensor] = {}
        if not getattr(self, "dsv4_layers_per_ratio", None):
            return
        total_bytes = 0
        for ratio, layers in self.dsv4_layers_per_ratio.items():
            n_layers = len(layers)
            coeff = 2 if ratio == 4 else 1  # overlap when ratio==4 (matches Compressor)
            kv_buf = torch.zeros(
                n_layers,
                max_num_seqs + 1,  # +1 dummy slot
                coeff * ratio,
                coeff * head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            score_buf = torch.full(
                (n_layers, max_num_seqs + 1, coeff * ratio, coeff * head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            counts_buf = torch.zeros(
                n_layers,
                max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsv4_compressor_kv_flat[ratio] = kv_buf
            self.dsv4_compressor_score_flat[ratio] = score_buf
            self.dsv4_compressor_counts_flat[ratio] = counts_buf
            total_bytes += (
                kv_buf.nelement() * kv_buf.itemsize
                + score_buf.nelement() * score_buf.itemsize
                + counts_buf.nelement() * counts_buf.itemsize
            )
        if total_bytes > 0:
            logger.info(
                f"DSv4 compressor scratch state (flat per ratio): "
                f"{total_bytes / 1e9:.3f} GB"
            )

    def allocate_indexer_cache(self, hf_config):
        """Allocate NSA indexer FP8 cache for DeepSeek V3.2.

        Uses the same num_pages / page_size as the main KV cache so that
        block tables are shared between the main attention and the indexer.
        """
        from nanodeploy.layers.indexer import IndexerCache

        index_head_dim = getattr(hf_config, "index_head_dim", 0)
        if index_head_dim == 0:
            return

        self.indexer_cache = IndexerCache(
            num_layers=self.num_hidden_layers,
            num_pages=self.num_local_kvcache_blocks,
            page_size=self.block_size,
            head_dim=index_head_dim,
            device=self.device,
        )
        total_bytes = sum(b.nelement() for b in self.indexer_cache.buffers)
        logger.debug(
            f"Allocated IndexerCache: {self.num_hidden_layers} layers, "
            f"{self.num_local_kvcache_blocks} pages, page_size={self.block_size}, "
            f"head_dim={index_head_dim}, total={total_bytes / 1e9:.2f} GB"
        )

    def allocate_gdn_states(
        self, hf_config, layer_types, max_bs: int, need_backup: bool = False
    ):
        """Allocate fixed-size GDN state buffers for linear_attention layers.

        When *need_backup* is True (MTP enabled), the layout is:
          - Slots ``0 .. max_bs-1``: **active** slots.
          - Slots ``max_bs .. 2*max_bs-1``: **backup** slots for lazy verify rollback.
          - Slot ``2*max_bs``: reserved **dummy** slot.
          Total: ``max_bs * 2 + 1``.

        When *need_backup* is False, backup slots are omitted:
          - Slots ``0 .. max_bs-1``: active slots.
          - Slot ``max_bs``: dummy slot.
          Total: ``max_bs + 1``.
        """
        num_layers = len(layer_types)
        num_k_heads = getattr(hf_config, "linear_num_key_heads", 0)
        num_v_heads = getattr(hf_config, "linear_num_value_heads", 0)
        head_k_dim = getattr(hf_config, "linear_key_head_dim", 0)
        head_v_dim = getattr(hf_config, "linear_value_head_dim", 0)
        conv_kernel_size = getattr(hf_config, "linear_conv_kernel_dim", 4)
        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim  # q + k + v

        if num_v_heads == 0:
            return

        if need_backup:
            num_slots = max_bs * 2 + 1
        else:
            num_slots = max_bs + 1
        self.gdn_num_slots = num_slots
        self.gdn_max_active_slots = max_bs  # boundary between active/backup

        # Conv state: [num_layers, num_slots, conv_dim, kernel_size]
        self.gdn_conv_states = torch.zeros(
            num_layers,
            num_slots,
            conv_dim,
            conv_kernel_size,
            dtype=torch.bfloat16,
            device=torch.get_default_device(),
        )

        # Recurrent state: [num_layers, num_slots, num_v_heads, head_v_dim, head_k_dim]
        # K-last layout to match flashinfer's gated_delta_rule_decode_pretranspose
        self.gdn_recurrent_states = torch.zeros(
            num_layers,
            num_slots,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.float32,
            device=torch.get_default_device(),
        )

        if need_backup:
            slot_info = (
                f"active_slots=0..{max_bs-1}, backup_slots={max_bs}..{2*max_bs-1}, "
                f"dummy_slot={2*max_bs}"
            )
        else:
            slot_info = f"active_slots=0..{max_bs-1}, dummy_slot={max_bs}"
        logger.debug(
            f"Allocated GDN states: conv={self.gdn_conv_states.shape} "
            f"({self.gdn_conv_states.element_size() * self.gdn_conv_states.nelement() / 1e9:.2f} GB), "
            f"recurrent={self.gdn_recurrent_states.shape} "
            f"({self.gdn_recurrent_states.element_size() * self.gdn_recurrent_states.nelement() / 1e9:.2f} GB), "
            f"{slot_info}"
        )

    def set_peer_agent_context(self, peer_context: PeerAgentContext | None) -> None:
        """Attach the worker-owned PeerAgentContext to cache RDMA users."""
        self.peer_agent_context = peer_context

    def register_peer_agent_memory_regions(self, mode: str = "hybrid") -> None:
        """Register cache-owned RDMA memory regions on the attached PeerAgent.

        Must be called AFTER allocate_kvcache() and allocate_gdn_states() so that
        all tensors exist before registration. In hybrid mode the PeerAgent is
        still alive, but KV cache / GDN MR registration is skipped because
        hybrid mode does not perform P2P KV transfer.
        """
        peer_context = self.peer_agent_context
        if peer_context is None:
            return

        agent_alias = peer_context.alias
        server_url = peer_context.server_url
        peer_agent = peer_context.agent

        try:
            # In hybrid mode we only need the PeerAgent alive (for vision
            # embed RDMA fetch); KV cache / GDN MR registration is not needed.
            if mode == "hybrid":
                logger.info(
                    f"PeerAgent started (hybrid, no KV MR): alias={agent_alias}, "
                    f"server={server_url}"
                )
                return

            # Register KV cache
            kv_size = self.kv_cache.numel() * self.kv_cache.itemsize
            self._local_mr_handler = peer_agent.register_memory_region(
                _KV_CACHE_BUFFER_ID,
                self.kv_cache.data_ptr(),
                int(self.kv_cache.storage_offset()),
                kv_size,
            )
            logger.info(
                f"PeerAgent started: alias={agent_alias}, server={server_url}, "
                f"kv_cache MR handler={self._local_mr_handler}"
            )

            # Register GDN states (if allocated)
            if (
                self.gdn_conv_states is not None
                and self.gdn_recurrent_states is not None
            ):
                conv_size = self.gdn_conv_states.numel() * self.gdn_conv_states.itemsize
                self._local_gdn_conv_mr_handler = peer_agent.register_memory_region(
                    "gdn_conv",
                    self.gdn_conv_states.data_ptr(),
                    int(self.gdn_conv_states.storage_offset()),
                    conv_size,
                )
                recurrent_size = (
                    self.gdn_recurrent_states.numel()
                    * self.gdn_recurrent_states.itemsize
                )
                self._local_gdn_recurrent_mr_handler = (
                    peer_agent.register_memory_region(
                        "gdn_recurrent",
                        self.gdn_recurrent_states.data_ptr(),
                        int(self.gdn_recurrent_states.storage_offset()),
                        recurrent_size,
                    )
                )
                logger.info(
                    f"Registered GDN MRs: conv={self._local_gdn_conv_mr_handler}, "
                    f"recurrent={self._local_gdn_recurrent_mr_handler}"
                )

            # Register IndexerCache (if allocated, V3.2 sparse attention)
            if self.indexer_cache is not None:
                indexer_buf = self.indexer_cache.buffer
                indexer_size = indexer_buf.numel() * indexer_buf.itemsize
                self._local_indexer_mr_handler = peer_agent.register_memory_region(
                    "indexer_cache",
                    indexer_buf.data_ptr(),
                    0,
                    indexer_size,
                )
                logger.info(
                    f"Registered IndexerCache MR: handler={self._local_indexer_mr_handler}"
                )

            # DSv4 (S2.4): register flat per-ratio compressed cache + compressor
            # state buffers — one MR per ratio per kind. Skipped in hybrid mode
            # via the same outer guard that protects KV/GDN registration.
            for ratio, buf in (
                getattr(self, "dsv4_compressed_caches_flat", None) or {}
            ).items():
                handler = peer_agent.register_memory_region(
                    f"dsv4_compressed_r{ratio}",
                    buf.data_ptr(),
                    int(buf.storage_offset()),
                    buf.numel() * buf.itemsize,
                )
                self._local_dsv4_compressed_mr_handlers[ratio] = handler
                logger.info(
                    f"Registered DSv4 compressed cache MR: ratio={ratio}, "
                    f"handler={handler}, size={buf.numel() * buf.itemsize / 1e9:.2f} GB"
                )

            for ratio, buf in (
                getattr(self, "dsv4_compressor_kv_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_kv_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_kv_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            for ratio, buf in (
                getattr(self, "dsv4_compressor_score_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_score_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_score_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            for ratio, buf in (
                getattr(self, "dsv4_compressor_counts_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_counts_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_counts_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            if self._local_dsv4_compressor_kv_mr_handlers:
                logger.info(
                    f"Registered DSv4 compressor scratch MRs: "
                    f"kv={self._local_dsv4_compressor_kv_mr_handlers}, "
                    f"score={self._local_dsv4_compressor_score_mr_handlers}, "
                    f"counts={self._local_dsv4_compressor_counts_mr_handlers}"
                )

        except Exception as e:
            logger.error(f"Failed to register PeerAgent memory regions: {e}")
            raise

    def get_peer_agent_addr(self) -> str | None:
        """Return the local peer agent address for this rank."""
        return (
            None if self.peer_agent_context is None else self.peer_agent_context.alias
        )

    def get_peer_agent_context(self) -> PeerAgentContext:
        """Return the attached worker-owned PeerAgentContext."""
        if self.peer_agent_context is None:
            raise RuntimeError(
                "CacheContext PeerAgentContext is not attached. "
                "Was ModelRunner PeerAgentContext initialized?"
            )
        return self.peer_agent_context

    def ensure_peer_agent_connected(self, peer_alias: str) -> None:
        """Ensure the local PeerAgent is connected to ``peer_alias``."""
        self.get_peer_agent_context().ensure_connected(peer_alias)

    def invalidate_engine_info_cache(self):
        """Invalidate the engine_info cache to force a refresh on next fetch."""
        self._engine_info_cache = None
        logger.info("Invalidated engine_info cache")

    def _fetch_engine_info_from_ctrl(self, engine_ids: set[str]) -> dict[str, dict]:
        """Get engine_info for specified engine_ids (cache + fetch if needed).

        This method handles all caching logic: checks cache, identifies missing IDs,
        fetches only missing ones from NanoCtrl, and updates cache.

        Uses the lightweight /get_entity_info endpoint instead of /list_entities.

        Args:
            engine_ids: Set of engine_ids to get info for.

        Returns:
            dict mapping engine_id to engine_info dict containing:
                - id, role, world_size, num_blocks, host, port, peer_addrs, etc.
        """
        import httpx

        if not engine_ids:
            return {}

        # Check cache and identify missing IDs
        engine_info_map = {}
        missing_ids = engine_ids

        if self._engine_info_cache is not None:
            cached_at, cached = self._engine_info_cache
            if time.time() - cached_at < _ENGINE_INFO_CACHE_TTL:
                # Get cached results
                engine_info_map = {
                    eid: info for eid, info in cached.items() if eid in engine_ids
                }
                missing_ids = engine_ids - cached.keys()

                if not missing_ids:
                    logger.debug(
                        f"All {len(engine_ids)} engines found in cache, no fetch needed"
                    )
                    return engine_info_map
                else:
                    logger.debug(
                        f"Cache hit for {len(engine_info_map)} engines, fetching {len(missing_ids)} missing: {missing_ids}"
                    )

        # Fetch missing engines from NanoCtrl
        if not self.ctrl_address:
            logger.warning("ctrl_address not configured, returning cached results only")
            return engine_info_map

        fetched_map: dict[str, dict] = {}
        url = f"{self.ctrl_address}/get_entity_info"
        scope = self.ctrl_scope or ""

        try:
            # trust_env=False: ignore HTTP(S)_PROXY/ALL_PROXY env vars. NanoCtrl
            # is an internal address; routing it through a cluster proxy makes
            # the request hang (httpx defaults to trust_env=True). This matches
            # dlslime's NanoCtrlClient, which also uses trust_env=False.
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                for engine_id in missing_ids:
                    try:
                        request_payload = {
                            "entity_type": "service",
                            "entity_id": engine_id,
                        }
                        if scope:
                            request_payload["scope"] = scope

                        response = client.post(url, json=request_payload)
                        response.raise_for_status()
                        data = response.json()

                        if data.get("status") == "ok":
                            entity_info = data.get("entity_info") or {}
                            engine_info = dict(entity_info.get("metadata") or {})
                            if engine_info:
                                engine_info.setdefault(
                                    "id",
                                    entity_info.get("entity_id", engine_id),
                                )
                                fetched_map[engine_id] = engine_info
                        else:
                            logger.warning(
                                f"get_entity_info for {engine_id} returned status: {data.get('status')}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching entity_info for {engine_id}: {e}")
                        continue

            # Update cache with newly fetched data
            if fetched_map:
                now = time.time()
                if self._engine_info_cache is not None:
                    cached_data = self._engine_info_cache[1]
                    cached_data.update(fetched_map)
                    self._engine_info_cache = (now, cached_data)
                else:
                    self._engine_info_cache = (now, fetched_map)

                logger.debug(
                    f"Fetched and cached {len(fetched_map)} engine_info: {list(fetched_map.keys())}"
                )

            # Return combined results
            engine_info_map.update(fetched_map)
            return engine_info_map

        except Exception as e:
            logger.error(f"Error fetching engine_info from NanoCtrl: {e}")
            # Return whatever we have from cache
            return engine_info_map

    # ------------------------------------------------------------------
    # Shared migration helpers
    # ------------------------------------------------------------------

    def _ensure_peer_connections(
        self, connection_requests: list[tuple[str, str, int, int, int]]
    ) -> None:
        """Establish connections to remote peers if not already connected.

        Args:
            connection_requests: list of (peer_alias, engine_id, num_kvcache_blocks,
                                          max_num_seqs, gdn_num_slots)
        """
        remote_peers_to_connect: dict[str, str] = {}
        peer_context = self.get_peer_agent_context()
        for (
            peer_alias,
            engine_id,
            num_kvcache_blocks,
            max_num_seqs,
            gdn_num_slots,
        ) in connection_requests:
            if peer_alias and not peer_context.is_connected(peer_alias):
                self.num_remote_kvcache_blocks[engine_id] = num_kvcache_blocks
                self.remote_max_num_seqs[engine_id] = max_num_seqs
                if gdn_num_slots > 0:
                    self.remote_gdn_num_slots[engine_id] = gdn_num_slots
                remote_peers_to_connect[peer_alias] = engine_id

        if not remote_peers_to_connect:
            return

        new_peers = list(remote_peers_to_connect.keys())
        logger.info(f"Batch connecting to {len(new_peers)} peers: {new_peers}")
        connected = peer_context.ensure_many_connected(new_peers)
        logger.info(f"Batch connection completed for {len(connected)} peers")

    def _execute_rdma_reads(
        self,
        assigns: dict[str, dict[str, list[tuple]]],
        gdn_assigns: dict[str, dict[str, list[tuple]]],
        indexer_assigns: dict[str, dict[str, list[tuple]]] | None = None,
        compressed_assigns: dict[str, dict[str, list[tuple]]] | None = None,
        compressor_state_assigns: dict[str, dict[str, list[tuple]]] | None = None,
    ) -> None:
        """Execute batched RDMA reads for KV cache and GDN state migration.

        Args:
            assigns: engine_id -> peer_alias -> list of
                     (peer_alias, kv_idx, layer_idx, remote_block_idx, source_block_idx)
            gdn_assigns: engine_id -> peer_alias -> list of
                         (layer_idx, remote_state_slot, local_state_slot)
            indexer_assigns: engine_id -> peer_alias -> list of
                             (layer_idx, remote_block_idx, source_block_idx)
            compressed_assigns: engine_id -> peer_alias -> list of
                                (ratio, ratio_layer_idx, remote_page_idx, local_page_idx)
            compressor_state_assigns: engine_id -> peer_alias -> list of
                                (ratio, ratio_layer_idx, remote_state_slot, local_state_slot)
        """
        peer_context = self.get_peer_agent_context()
        peer_agent = peer_context.agent
        for engine_id, peer_assigns in assigns.items():
            for peer_alias, assign_batch in peer_assigns.items():
                if not peer_context.is_connected(peer_alias):
                    logger.error(f"Peer {peer_alias} not connected, skipping")
                    continue

                conn = peer_agent.query_connection(peer_alias)
                if conn is None or conn.endpoint is None:
                    logger.error(f"Failed to get endpoint for {peer_alias}")
                    continue
                endpoint = conn.endpoint

                remote_mr_info = peer_agent.get_mr_info(peer_alias, _KV_CACHE_BUFFER_ID)
                if remote_mr_info is None:
                    logger.error(f"Failed to get MR info for {peer_alias}")
                    continue

                remote_mr_handler = peer_agent.get_handle(
                    _KV_CACHE_BUFFER_ID, peer_alias=peer_alias
                )
                logger.debug(
                    f"Remote MR for {peer_alias}: handler={remote_mr_handler}, "
                    f"local_handler={self._local_mr_handler}"
                )

                if self._local_mr_handler is None:
                    logger.error(
                        f"Local MR handler not available for {_KV_CACHE_BUFFER_ID}"
                    )
                    continue
                local_mr_handler = self._local_mr_handler

                # Build KV cache RDMA ops
                rdma_ops: list[tuple] = []
                for op_idx, (
                    _peer_alias,
                    kv_idx,
                    layer_idx,
                    remote_block_idx,
                    source_block_idx,
                ) in enumerate(assign_batch):
                    local_off = self.local_kv_stride(
                        kv_idx, layer_idx, source_block_idx
                    )
                    remote_off = self.remote_kv_stride(
                        kv_idx, layer_idx, remote_block_idx, engine_id
                    )
                    length = self.block_stride(1)

                    if local_mr_handler is None or remote_mr_handler is None:
                        logger.error(
                            f"[Op {op_idx}] Invalid MR handlers: local={local_mr_handler}, remote={remote_mr_handler}"
                        )
                        continue
                    if local_off < 0 or remote_off < 0 or length <= 0:
                        logger.error(
                            f"[Op {op_idx}] Invalid offsets/length: local_off={local_off}, remote_off={remote_off}, length={length}"
                        )
                        continue

                    rdma_ops.append(
                        (
                            local_mr_handler,
                            remote_mr_handler,
                            remote_off,
                            local_off,
                            length,
                        )
                    )

                # Append GDN state RDMA ops
                gdn_batch = gdn_assigns.get(engine_id, {}).get(peer_alias, [])

                if (
                    gdn_batch
                    and self.gdn_conv_states is not None
                    and self.gdn_recurrent_states is not None
                ):
                    # Conv state
                    remote_conv_mr_info = peer_agent.get_mr_info(peer_alias, "gdn_conv")
                    if remote_conv_mr_info:
                        remote_conv_mr = peer_agent.get_handle(
                            "gdn_conv", peer_alias=peer_alias
                        )
                        local_conv_mr = self._local_gdn_conv_mr_handler
                        conv_len = self.gdn_conv_slot_num_bytes()
                        for layer_idx, remote_slot, local_slot in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_conv_mr,
                                    remote_conv_mr,
                                    self.remote_gdn_conv_stride(
                                        layer_idx, remote_slot, engine_id
                                    ),
                                    self.gdn_conv_stride(layer_idx, local_slot),
                                    conv_len,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get gdn_conv MR info for {peer_alias}"
                        )

                    # Recurrent state
                    remote_rec_mr_info = peer_agent.get_mr_info(
                        peer_alias, "gdn_recurrent"
                    )
                    if remote_rec_mr_info:
                        remote_rec_mr = peer_agent.get_handle(
                            "gdn_recurrent", peer_alias=peer_alias
                        )
                        local_rec_mr = self._local_gdn_recurrent_mr_handler
                        rec_len = self.gdn_recurrent_slot_num_bytes()
                        for layer_idx, remote_slot, local_slot in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_rec_mr,
                                    remote_rec_mr,
                                    self.remote_gdn_recurrent_stride(
                                        layer_idx, remote_slot, engine_id
                                    ),
                                    self.gdn_recurrent_stride(layer_idx, local_slot),
                                    rec_len,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get gdn_recurrent MR info for {peer_alias}"
                        )

                # Append DSv4 compressed cache + compressor scratch state RDMA ops.
                comp_batch = (
                    (compressed_assigns or {}).get(engine_id, {}).get(peer_alias, [])
                )
                if comp_batch:
                    # Group by ratio to register the right MR per ratio.
                    by_ratio: dict[int, list[tuple]] = {}
                    for r, rli, rpage, lpage in comp_batch:
                        by_ratio.setdefault(r, []).append((rli, rpage, lpage))
                    for ratio, ops in by_ratio.items():
                        local_handler = self._local_dsv4_compressed_mr_handlers.get(
                            ratio
                        )
                        if local_handler is None:
                            continue
                        remote_info = peer_agent.get_mr_info(
                            peer_alias, f"dsv4_compressed_r{ratio}"
                        )
                        if not remote_info:
                            logger.warning(
                                f"Failed to get DSv4 compressed MR info for {peer_alias}, ratio={ratio}"
                            )
                            continue
                        remote_handler = peer_agent.get_handle(
                            f"dsv4_compressed_r{ratio}", peer_alias=peer_alias
                        )
                        page_bytes = self.compressed_page_bytes(ratio)
                        for rli, rpage, lpage in ops:
                            rdma_ops.append(
                                (
                                    local_handler,
                                    remote_handler,
                                    self.remote_compressed_stride(
                                        ratio, rli, rpage, engine_id
                                    ),
                                    self.local_compressed_stride(ratio, rli, lpage),
                                    page_bytes,
                                )
                            )

                cstate_batch = (
                    (compressor_state_assigns or {})
                    .get(engine_id, {})
                    .get(peer_alias, [])
                )
                if cstate_batch:
                    by_ratio_s: dict[int, list[tuple]] = {}
                    for r, rli, rslot, lslot in cstate_batch:
                        by_ratio_s.setdefault(r, []).append((rli, rslot, lslot))
                    for ratio, ops in by_ratio_s.items():
                        for kind, local_map in (
                            ("kv", self._local_dsv4_compressor_kv_mr_handlers),
                            ("score", self._local_dsv4_compressor_score_mr_handlers),
                            ("counts", self._local_dsv4_compressor_counts_mr_handlers),
                        ):
                            local_handler = local_map.get(ratio)
                            if local_handler is None:
                                continue
                            mr_name = f"dsv4_compressor_{kind}_r{ratio}"
                            remote_info = peer_agent.get_mr_info(peer_alias, mr_name)
                            if not remote_info:
                                logger.warning(
                                    f"Failed to get {mr_name} MR info for {peer_alias}"
                                )
                                continue
                            remote_handler = peer_agent.get_handle(
                                mr_name, peer_alias=peer_alias
                            )
                            row_bytes = self._compressor_state_row_bytes(ratio, kind)
                            for rli, rslot, lslot in ops:
                                rdma_ops.append(
                                    (
                                        local_handler,
                                        remote_handler,
                                        self.remote_compressor_state_stride(
                                            ratio, rli, rslot, kind, engine_id
                                        ),
                                        self.local_compressor_state_stride(
                                            ratio, rli, lslot, kind
                                        ),
                                        row_bytes,
                                    )
                                )

                # Append IndexerCache RDMA ops (V3.2 sparse attention)
                indexer_batch = (
                    (indexer_assigns or {}).get(engine_id, {}).get(peer_alias, [])
                )
                if indexer_batch and self.indexer_cache is not None:
                    remote_indexer_mr_info = peer_agent.get_mr_info(
                        peer_alias, "indexer_cache"
                    )
                    if remote_indexer_mr_info:
                        remote_indexer_mr = peer_agent.get_handle(
                            "indexer_cache", peer_alias=peer_alias
                        )
                        local_indexer_mr = self._local_indexer_mr_handler
                        page_bytes = self.indexer_page_num_bytes()
                        for layer_idx, remote_block, local_block in indexer_batch:
                            rdma_ops.append(
                                (
                                    local_indexer_mr,
                                    remote_indexer_mr,
                                    self.remote_indexer_stride(
                                        layer_idx, remote_block, engine_id
                                    ),
                                    self.local_indexer_stride(layer_idx, local_block),
                                    page_bytes,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get indexer_cache MR info for {peer_alias}"
                        )

                if not rdma_ops:
                    logger.error(f"No valid RDMA ops for {peer_alias}, skipping")
                    continue

                try:
                    slot = endpoint.read(rdma_ops, None)
                    if slot is None:
                        logger.error("endpoint.read returned None")
                        raise RuntimeError("endpoint.read returned None")
                    slot.wait()
                    # GPUDirect RDMA may bypass CUDA stream ordering.
                    # Synchronize to ensure migrated KV data is visible to subsequent kernels.
                    torch.cuda.synchronize()

                    logger.info(
                        f"Completed batch RDMA read from {peer_alias} ({len(rdma_ops)} operations)"
                    )
                except Exception as e:
                    logger.error(
                        f"Batch RDMA read FAILED from {peer_alias}: {len(rdma_ops)} ops, error={e}",
                        exc_info=True,
                    )
                    raise

    def migrate_from_bytes(self, data: bytes):
        """Migrate KV cache using lean MigrateBatchInput protocol (no Sequence objects)."""
        from nanodeploy._cpp import parse_migrate_batch

        views = parse_migrate_batch(data)

        if self.peer_agent_context is None:
            logger.error("migrate_from_bytes called but PeerAgent not initialized")
            return

        # Collect target engine_ids
        target_engine_ids = set()
        for v in views:
            if v.migrate_engine_id:
                target_engine_ids.add(v.migrate_engine_id)

        if not target_engine_ids:
            logger.debug("No target engine_ids found, skipping migration")
            return

        engine_info_map = self._fetch_engine_info_from_ctrl(target_engine_ids)

        # Ensure connections
        connection_requests: list[tuple[str, str, int, int]] = []
        for v in views:
            engine_id = v.migrate_engine_id
            engine_info = engine_info_map.get(engine_id, {})
            remote_max_num_seqs = engine_info.get("max_num_seqs", 0)
            remote_gdn_num_slots = engine_info.get("gdn_num_slots", 0)
            # DSv4 (S2.5): record remote pool sizes for stride math.
            remote_dsv4_pools = engine_info.get("dsv4_compressed_pool_pages", {}) or {}
            # Keys may be strings (JSON) — coerce to int.
            self.remote_compressed_pool_pages[engine_id] = {
                int(r): int(p) for r, p in remote_dsv4_pools.items()
            }
            self.remote_dsv4_max_slots[engine_id] = int(
                engine_info.get("dsv4_max_slots", remote_max_num_seqs)
            )
            remote_layers = engine_info.get("dsv4_num_layers_per_ratio", {}) or {}
            self.remote_dsv4_num_layers_per_ratio[engine_id] = {
                int(r): int(n) for r, n in remote_layers.items()
            }
            for peer_alias in engine_info.get("peer_addrs", []):
                connection_requests.append(
                    (
                        peer_alias,
                        engine_id,
                        v.migrate_num_kvcache_blocks,
                        remote_max_num_seqs,
                        remote_gdn_num_slots,
                    )
                )
        self._ensure_peer_connections(connection_requests)

        # Build assignment list
        assigns = defaultdict(lambda: defaultdict(list))
        gdn_assigns = defaultdict(lambda: defaultdict(list))
        indexer_assigns = defaultdict(lambda: defaultdict(list))
        # DSv4 (S2.6): per-ratio compressed pages + compressor scratch state.
        compressed_assigns = defaultdict(lambda: defaultdict(list))
        compressor_state_assigns = defaultdict(lambda: defaultdict(list))
        sp_idx = get_dist_context().attn_sp_rank

        for v in views:
            engine_id = v.migrate_engine_id
            engine_info = engine_info_map.get(engine_id, {})
            peer_addrs = engine_info.get("peer_addrs", [])
            if not peer_addrs:
                logger.warning(
                    f"Sequence {v.seq_id} has no peer_addrs for engine {engine_id}"
                )
                continue

            if len(v.migrate_block_location) > len(v.active_block_location):
                logger.error(
                    f"Sequence {v.seq_id}: migrate has MORE blocks than active! "
                    f"migrate={len(v.migrate_block_location)}, active={len(v.active_block_location)}"
                )
                continue
            if len(v.migrate_block_location) < len(v.active_block_location):
                # Expected when prompt_tokens % block_size == 0: prefill serializes N blocks
                # for prompt KV, but decode allocates N+1 blocks for (prompt+1) total tokens.
                # zip() below naturally iterates only over the migrate (shorter) side;
                # the extra active block will be filled during the first decode step.
                logger.info(
                    f"Sequence {v.seq_id}: partial migration "
                    f"(migrate={len(v.migrate_block_location)}, active={len(v.active_block_location)})"
                )

            for remote_bl, source_bl in zip(
                v.migrate_block_location, v.active_block_location
            ):
                remote_sp_idx, remote_block_idx = remote_bl
                source_sp_idx, source_block_idx = source_bl

                # Validate block indices
                if (
                    source_block_idx < 0
                    or source_block_idx >= self.num_local_kvcache_blocks
                ):
                    logger.error(
                        f"Sequence {v.seq_id}: source_block_idx {source_block_idx} "
                        f"out of range [0, {self.num_local_kvcache_blocks})"
                    )
                    continue
                remote_max = self.num_remote_kvcache_blocks.get(engine_id, 0)
                if remote_block_idx < 0 or (
                    remote_max > 0 and remote_block_idx >= remote_max
                ):
                    logger.error(
                        f"Sequence {v.seq_id}: remote_block_idx {remote_block_idx} "
                        f"out of range [0, {remote_max})"
                    )
                    continue

                if source_sp_idx != sp_idx:
                    continue

                remote_rank = v.migrate_dp_idx * v.migrate_group_size + remote_sp_idx

                if remote_rank >= len(peer_addrs):
                    logger.error(
                        f"remote_rank {remote_rank} >= len(peer_addrs) {len(peer_addrs)}"
                    )
                    continue
                peer_alias = peer_addrs[remote_rank]

                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        assigns[engine_id][peer_alias].append(
                            (
                                peer_alias,
                                kv_idx,
                                layer_idx,
                                remote_block_idx,
                                source_block_idx,
                            )
                        )

                # Indexer cache assignments (V3.2 sparse attention)
                if self.indexer_cache is not None:
                    for layer_idx in range(self.num_hidden_layers):
                        indexer_assigns[engine_id][peer_alias].append(
                            (layer_idx, remote_block_idx, source_block_idx)
                        )

            # DSv4 (S2.6): per-ratio compressed cache + compressor state migration.
            # migrate_compressed_block_tables[ratio] = list of remote page IDs.
            # active_compressed_block_tables[ratio]  = list of local page IDs allocated by
            #                                          the decode engine's GroupManager.
            if (
                getattr(self, "dsv4_compressed_caches_flat", None)
                and v.migrate_compressed_block_tables
            ):
                remote_rank = v.migrate_dp_idx * v.migrate_group_size + (
                    v.migrate_group_size - 1
                )
                if 0 <= remote_rank < len(peer_addrs):
                    peer_alias = peer_addrs[remote_rank]
                    for (
                        ratio,
                        remote_pages,
                    ) in v.migrate_compressed_block_tables.items():
                        if ratio not in self.dsv4_compressed_caches_flat:
                            continue  # decode engine doesn't have this ratio (mismatch)
                        local_pages = v.active_compressed_block_tables.get(ratio, [])
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        # Pair-wise remote→local page mapping; for each (rli) layer, the
                        # SAME (remote_page, local_page) pairs apply (the table is per-seq,
                        # not per-layer).
                        for ratio_layer_idx in range(n_layers):
                            for rpage, lpage in zip(remote_pages, local_pages):
                                compressed_assigns[engine_id][peer_alias].append(
                                    (ratio, ratio_layer_idx, rpage, lpage)
                                )

            # DSv4 (S2.6): per-ratio compressor scratch state migration.
            if (
                getattr(self, "dsv4_compressor_kv_flat", None)
                and v.migrate_state_slot >= 0
                and v.active_state_slot >= 0
            ):
                remote_rank = v.migrate_dp_idx * v.migrate_group_size + (
                    v.migrate_group_size - 1
                )
                if 0 <= remote_rank < len(peer_addrs):
                    peer_alias = peer_addrs[remote_rank]
                    for ratio in self.dsv4_compressor_kv_flat.keys():
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        for ratio_layer_idx in range(n_layers):
                            compressor_state_assigns[engine_id][peer_alias].append(
                                (
                                    ratio,
                                    ratio_layer_idx,
                                    v.migrate_state_slot,
                                    v.active_state_slot,
                                )
                            )

            # GDN assignments
            if (
                self.gdn_conv_states is not None
                and self.gdn_recurrent_states is not None
            ):
                remote_state_slot = v.migrate_state_slot
                local_state_slot = v.active_state_slot

                if remote_state_slot >= 0 and local_state_slot >= 0:
                    remote_rank = v.migrate_dp_idx * v.migrate_group_size + (
                        v.migrate_group_size - 1
                    )
                    if remote_rank < len(peer_addrs):
                        peer_alias = peer_addrs[remote_rank]
                        num_gdn_layers = self.gdn_recurrent_states.shape[0]
                        for layer_idx in range(num_gdn_layers):
                            gdn_assigns[engine_id][peer_alias].append(
                                (
                                    layer_idx,
                                    remote_state_slot,
                                    local_state_slot,
                                )
                            )

        self._execute_rdma_reads(
            assigns,
            gdn_assigns,
            indexer_assigns,
            compressed_assigns=compressed_assigns,
            compressor_state_assigns=compressor_state_assigns,
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

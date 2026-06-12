"""GPU buffer allocation for the KV cache and auxiliary caches.

Extracted from ``CacheContext`` and mixed back in. Each method allocates
torch tensors and stores them on the shared ``CacheContext`` state via
``self``.
"""

import torch
from dlengine.logging import get_logger

logger = get_logger("dlengine")


class KVCacheAllocatorMixin:
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
        from dlengine.layers.indexer import IndexerCache

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

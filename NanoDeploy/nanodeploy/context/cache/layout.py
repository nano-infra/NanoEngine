"""KV-cache byte-offset / stride math (extracted from ``CacheContext``).

Pure layout arithmetic for the paged KV cache and the auxiliary GDN /
indexer / DSv4-compressed caches. Mixed into :class:`CacheContext`; every
method operates on the shared ``CacheContext`` state via ``self`` and has
no side effects.
"""


class CacheLayoutMixin:
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

"""P2P cache transfer byte-offset / stride math.

``CacheTensorLayout`` is an explicit layout snapshot: it contains only the
metadata needed to translate logical cache coordinates into byte offsets.
``P2PCacheLayout`` builds local/remote snapshots from the active cache context
and peer metadata, then exposes the transfer-facing helper methods.
"""

from dataclasses import dataclass, field

from dlengine.runtime.context.cache.hca import DSV4_BYTES_PER_TOKEN


@dataclass
class CacheTensorLayout:
    num_blocks: int
    block_size: int
    num_local_kv_heads: int
    head_dim: int
    dtype_itemsize: int
    num_hidden_layers: int
    mode: str
    is_fp8_kvcache: bool = False
    fp8_head_dim: int = 0
    raw_fp8_mla_layout: bool = False
    gdn_num_slots: int = 0
    gdn_conv_stride0: int = 0
    gdn_conv_stride1: int = 0
    gdn_conv_element_size: int = 0
    gdn_recurrent_stride0: int = 0
    gdn_recurrent_stride1: int = 0
    gdn_recurrent_element_size: int = 0
    indexer_page_bytes: int = 0
    compressed_pool_pages: dict[int, int] = field(default_factory=dict)
    compressed_page_bytes_by_ratio: dict[int, int] = field(default_factory=dict)
    compressor_slots_plus_dummy_by_ratio: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_cache_context(cls, context) -> "CacheTensorLayout":
        compressed_pool_pages = {}
        compressed_page_bytes_by_ratio = {}
        for ratio, cfg in (
            getattr(context, "dsv4_compressed_pool_config", None) or {}
        ).items():
            num_pages, page_size, _max_blocks = cfg
            compressed_pool_pages[int(ratio)] = int(num_pages)
            compressed_page_bytes_by_ratio[int(ratio)] = (
                int(page_size) * DSV4_BYTES_PER_TOKEN
            )

        compressor_slots_plus_dummy_by_ratio = {}
        for ratio, kv in (
            getattr(context, "dsv4_compressor_kv_flat", None) or {}
        ).items():
            if kv is not None:
                compressor_slots_plus_dummy_by_ratio[int(ratio)] = int(kv.shape[1])

        indexer_page_bytes = 0
        if context.indexer_cache is not None:
            indexer_page_bytes = (
                context.indexer_cache.page_size * context.indexer_cache.bytes_per_token
            )

        gdn_conv_stride1 = 0
        gdn_conv_stride0 = 0
        gdn_conv_element_size = 0
        if context.gdn_conv_states is not None:
            gdn_conv_stride0 = context.gdn_conv_states.stride(0)
            gdn_conv_stride1 = context.gdn_conv_states.stride(1)
            gdn_conv_element_size = context.gdn_conv_states.element_size()

        gdn_recurrent_stride1 = 0
        gdn_recurrent_stride0 = 0
        gdn_recurrent_element_size = 0
        if context.gdn_recurrent_states is not None:
            gdn_recurrent_stride0 = context.gdn_recurrent_states.stride(0)
            gdn_recurrent_stride1 = context.gdn_recurrent_states.stride(1)
            gdn_recurrent_element_size = context.gdn_recurrent_states.element_size()

        return cls(
            num_blocks=context.num_local_kvcache_blocks,
            block_size=context.block_size,
            num_local_kv_heads=context.num_local_kv_heads,
            head_dim=context.head_dim,
            dtype_itemsize=context.dtype.itemsize,
            num_hidden_layers=context.num_hidden_layers,
            mode=context.mode,
            is_fp8_kvcache=context.is_fp8_kvcache,
            fp8_head_dim=getattr(context, "_fp8_head_dim", 0),
            raw_fp8_mla_layout=bool(getattr(context, "raw_fp8_mla_layout", False)),
            gdn_num_slots=context.gdn_num_slots,
            gdn_conv_stride0=gdn_conv_stride0,
            gdn_conv_stride1=gdn_conv_stride1,
            gdn_conv_element_size=gdn_conv_element_size,
            gdn_recurrent_stride0=gdn_recurrent_stride0,
            gdn_recurrent_stride1=gdn_recurrent_stride1,
            gdn_recurrent_element_size=gdn_recurrent_element_size,
            indexer_page_bytes=indexer_page_bytes,
            compressed_pool_pages=compressed_pool_pages,
            compressed_page_bytes_by_ratio=compressed_page_bytes_by_ratio,
            compressor_slots_plus_dummy_by_ratio=compressor_slots_plus_dummy_by_ratio,
        )

    @classmethod
    def from_peer_metadata(
        cls,
        *,
        local_layout: "CacheTensorLayout",
        num_blocks: int,
        gdn_num_slots: int = 0,
        compressed_pool_pages: dict[int, int] | None = None,
        dsv4_max_slots: int = 0,
        num_hidden_layers: int = 0,
    ) -> "CacheTensorLayout":
        compressor_slots_plus_dummy_by_ratio = {}
        for ratio, local_slots in (
            local_layout.compressor_slots_plus_dummy_by_ratio
        ).items():
            remote_slots = dsv4_max_slots if dsv4_max_slots > 0 else local_slots - 1
            compressor_slots_plus_dummy_by_ratio[ratio] = remote_slots + 1

        return cls(
            num_blocks=num_blocks,
            block_size=local_layout.block_size,
            num_local_kv_heads=local_layout.num_local_kv_heads,
            head_dim=local_layout.head_dim,
            dtype_itemsize=local_layout.dtype_itemsize,
            # A PP prefill stage's kv_cache tensor holds only that stage's
            # layers, so its stride math must use the stage-local layer count
            # rather than the (full) local one.
            num_hidden_layers=(num_hidden_layers or local_layout.num_hidden_layers),
            mode=local_layout.mode,
            is_fp8_kvcache=local_layout.is_fp8_kvcache,
            fp8_head_dim=local_layout.fp8_head_dim,
            raw_fp8_mla_layout=local_layout.raw_fp8_mla_layout,
            gdn_num_slots=gdn_num_slots or local_layout.gdn_num_slots,
            gdn_conv_stride0=(
                (gdn_num_slots or local_layout.gdn_num_slots)
                * local_layout.gdn_conv_stride1
            ),
            gdn_conv_stride1=local_layout.gdn_conv_stride1,
            gdn_conv_element_size=local_layout.gdn_conv_element_size,
            gdn_recurrent_stride0=(
                (gdn_num_slots or local_layout.gdn_num_slots)
                * local_layout.gdn_recurrent_stride1
            ),
            gdn_recurrent_stride1=local_layout.gdn_recurrent_stride1,
            gdn_recurrent_element_size=local_layout.gdn_recurrent_element_size,
            indexer_page_bytes=local_layout.indexer_page_bytes,
            compressed_pool_pages=compressed_pool_pages or {},
            compressed_page_bytes_by_ratio=(
                local_layout.compressed_page_bytes_by_ratio.copy()
            ),
            compressor_slots_plus_dummy_by_ratio=compressor_slots_plus_dummy_by_ratio,
        )

    def block_stride(self, block_idx: int) -> int:
        if self.mode == "dsv4":
            return block_idx * self.block_size * DSV4_BYTES_PER_TOKEN
        if self.is_fp8_kvcache and self.mode == "mla":
            rows = self.block_size if self.raw_fp8_mla_layout else self.block_size + 1
            return block_idx * rows * self.fp8_head_dim
        return (
            block_idx
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype_itemsize
        )

    def block_num_bytes(self) -> int:
        """Return payload bytes, excluding any inter-block padding."""
        if self.is_fp8_kvcache and self.mode == "mla":
            return self.block_size * self.fp8_head_dim
        return self.block_stride(1)

    def layer_stride(self, layer_idx: int, block_idx: int) -> int:
        # DSv4 HCA allocates one extra dummy page per layer.
        layer_blocks = self.num_blocks + 1 if self.mode == "dsv4" else self.num_blocks
        return self.block_stride(layer_blocks) * layer_idx + self.block_stride(
            block_idx
        )

    def kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int) -> int:
        if self.mode == "dsv4":
            if kv_idx != 0:
                raise ValueError("DSv4 HCA cache has a single cache plane")
            return self.layer_stride(layer_idx, block_idx)
        return self.layer_stride(
            self.num_hidden_layers, 0
        ) * kv_idx + self.layer_stride(layer_idx, block_idx)

    def gdn_conv_stride(self, layer_idx: int, slot_idx: int) -> int:
        if self.gdn_conv_stride1 == 0:
            return -1
        return (
            layer_idx * self.gdn_conv_stride0 + slot_idx * self.gdn_conv_stride1
        ) * self.gdn_conv_element_size

    def gdn_recurrent_stride(self, layer_idx: int, slot_idx: int) -> int:
        if self.gdn_recurrent_stride1 == 0:
            return -1
        return (
            layer_idx * self.gdn_recurrent_stride0
            + slot_idx * self.gdn_recurrent_stride1
        ) * self.gdn_recurrent_element_size

    def gdn_conv_slot_num_bytes(self) -> int:
        return self.gdn_conv_stride1 * self.gdn_conv_element_size

    def gdn_recurrent_slot_num_bytes(self) -> int:
        return self.gdn_recurrent_stride1 * self.gdn_recurrent_element_size

    def indexer_stride(self, layer_idx: int, block_idx: int) -> int:
        return (layer_idx * self.num_blocks + block_idx) * self.indexer_page_bytes

    def compressed_page_bytes(self, ratio: int) -> int:
        return self.compressed_page_bytes_by_ratio.get(ratio, 0)

    def compressed_stride(self, ratio: int, ratio_layer_idx: int, page_idx: int) -> int:
        page_bytes = self.compressed_page_bytes(ratio)
        num_pages = self.compressed_pool_pages.get(ratio)
        if num_pages is None:
            return -1
        return (ratio_layer_idx * (num_pages + 1) + page_idx) * page_bytes

    def compressor_state_row_bytes(self, ratio: int, kind: str) -> int:
        if kind == "counts":
            return 4
        coeff = 2 if ratio == 4 else 1
        head_dim = 512  # DSv4 fixed
        return coeff * ratio * coeff * head_dim * 4

    def compressor_state_stride(
        self, ratio: int, ratio_layer_idx: int, slot: int, kind: str
    ) -> int:
        row_bytes = self.compressor_state_row_bytes(ratio, kind)
        slots = self.compressor_slots_plus_dummy_by_ratio.get(ratio, 0)
        return (ratio_layer_idx * slots + slot) * row_bytes


class P2PCacheLayout:
    def __init__(self, transfer) -> None:
        self.transfer = transfer

    def local(self) -> CacheTensorLayout:
        return CacheTensorLayout.from_cache_context(self.transfer.cache_context)

    def remote(
        self, remote_engine_id: str, num_hidden_layers: int = 0
    ) -> CacheTensorLayout:
        """Layout snapshot of a remote engine's cache tensors.

        ``num_hidden_layers`` overrides the remote layer count for PP prefill
        engines, whose per-stage kv_cache tensors hold only the stage's own
        layers (0 = same as local, i.e. a non-PP peer).
        """
        local_layout = self.local()
        return CacheTensorLayout.from_peer_metadata(
            local_layout=local_layout,
            num_blocks=self.transfer.num_remote_kvcache_blocks[remote_engine_id],
            gdn_num_slots=self.transfer.remote_gdn_num_slots.get(remote_engine_id, 0),
            compressed_pool_pages=self.transfer.remote_compressed_pool_pages.get(
                remote_engine_id, {}
            ),
            dsv4_max_slots=self.transfer.remote_dsv4_max_slots.get(remote_engine_id, 0),
            num_hidden_layers=num_hidden_layers,
        )

    def block_stride(self, block_idx: int) -> int:
        return self.local().block_stride(block_idx)

    def block_num_bytes(self) -> int:
        return self.local().block_num_bytes()

    def local_kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int) -> int:
        return self.local().kv_stride(kv_idx, layer_idx, block_idx)

    def remote_kv_stride(
        self,
        kv_idx: int,
        layer_idx: int,
        block_idx: int,
        remote_engine_id: str,
        remote_num_layers: int = 0,
    ) -> int:
        return self.remote(remote_engine_id, remote_num_layers).kv_stride(
            kv_idx, layer_idx, block_idx
        )

    def gdn_conv_stride(self, layer_idx: int, slot_idx: int) -> int:
        return self.local().gdn_conv_stride(layer_idx, slot_idx)

    def gdn_recurrent_stride(self, layer_idx: int, slot_idx: int) -> int:
        return self.local().gdn_recurrent_stride(layer_idx, slot_idx)

    def remote_gdn_conv_stride(
        self, layer_idx: int, slot_idx: int, remote_engine_id: str
    ) -> int:
        return self.remote(remote_engine_id).gdn_conv_stride(layer_idx, slot_idx)

    def remote_gdn_recurrent_stride(
        self, layer_idx: int, slot_idx: int, remote_engine_id: str
    ) -> int:
        return self.remote(remote_engine_id).gdn_recurrent_stride(layer_idx, slot_idx)

    def gdn_conv_slot_num_bytes(self) -> int:
        return self.local().gdn_conv_slot_num_bytes()

    def gdn_recurrent_slot_num_bytes(self) -> int:
        return self.local().gdn_recurrent_slot_num_bytes()

    def indexer_page_num_bytes(self) -> int:
        return self.local().indexer_page_bytes

    def local_indexer_stride(self, layer_idx: int, block_idx: int) -> int:
        return self.local().indexer_stride(layer_idx, block_idx)

    def remote_indexer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ) -> int:
        return self.remote(remote_engine_id).indexer_stride(layer_idx, block_idx)

    def compressed_page_bytes(self, ratio: int) -> int:
        return self.local().compressed_page_bytes(ratio)

    def local_compressed_stride(
        self, ratio: int, ratio_layer_idx: int, page_idx: int
    ) -> int:
        return self.local().compressed_stride(ratio, ratio_layer_idx, page_idx)

    def remote_compressed_stride(
        self,
        ratio: int,
        ratio_layer_idx: int,
        page_idx: int,
        remote_engine_id: str,
    ) -> int:
        return self.remote(remote_engine_id).compressed_stride(
            ratio, ratio_layer_idx, page_idx
        )

    def compressor_state_row_bytes(self, ratio: int, kind: str) -> int:
        return self.local().compressor_state_row_bytes(ratio, kind)

    def local_compressor_state_stride(
        self, ratio: int, ratio_layer_idx: int, slot: int, kind: str
    ) -> int:
        return self.local().compressor_state_stride(ratio, ratio_layer_idx, slot, kind)

    def remote_compressor_state_stride(
        self,
        ratio: int,
        ratio_layer_idx: int,
        slot: int,
        kind: str,
        remote_engine_id: str,
    ) -> int:
        return self.remote(remote_engine_id).compressor_state_stride(
            ratio, ratio_layer_idx, slot, kind
        )


# Backward-compatible name for old imports.
CacheLayoutMixin = P2PCacheLayout


__all__ = ["CacheLayoutMixin", "CacheTensorLayout", "P2PCacheLayout"]

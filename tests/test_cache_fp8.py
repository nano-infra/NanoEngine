"""Unit tests for CacheContext FP8 MLA allocation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

# We need to mock distributed + dlslime since CacheContext.__post_init__ calls them.
# Instead, we test allocate_kvcache directly.

from dlengine.runtime.context.cache import CacheContext
from dlengine.runtime.context.cache.mla import get_mla_block_bytes
from dlengine.runtime.disagg.p2p.cache_layout import CacheTensorLayout


class TestCacheContextFP8:

    def _make_cache_context(self, is_fp8: bool, *, raw_fp8: bool = False):
        """Create a CacheContext with minimal init by bypassing __post_init__."""
        ctx = object.__new__(CacheContext)
        ctx.num_kv_heads = 1
        ctx.head_dim = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
        ctx.block_size = 64
        ctx.num_hidden_layers = 4  # small for test
        ctx.attention_tp = 1
        ctx.gpu_memory_utilization = 0.9
        ctx.gpu_memory_limit_gb = None
        ctx.device = "cuda"
        ctx.dtype = torch.bfloat16
        ctx.mode = "mla"
        ctx.kv_lora_rank = 512
        ctx.qk_rope_head_dim = 64
        ctx.is_fp8_kvcache = is_fp8
        ctx.raw_fp8_mla_layout = raw_fp8
        ctx.num_local_kvcache_blocks = -1
        ctx.num_remote_kvcache_blocks = {}
        ctx.kv_cache = None
        ctx.gdn_conv_states = None
        ctx.gdn_recurrent_states = None
        ctx.selected_nic = None
        ctx.endpoints = {}
        ctx.ctrl_address = None
        ctx.ctrl_scope = None
        ctx.engine_id = None
        ctx._fp8_head_dim = 0
        if is_fp8:
            get_mla_block_bytes(ctx)
        return ctx

    @staticmethod
    def _layout(ctx):
        return CacheTensorLayout(
            num_blocks=ctx.num_local_kvcache_blocks,
            block_size=ctx.block_size,
            num_local_kv_heads=ctx.num_local_kv_heads,
            head_dim=ctx.head_dim,
            dtype_itemsize=ctx.dtype.itemsize,
            num_hidden_layers=ctx.num_hidden_layers,
            mode=ctx.mode,
            is_fp8_kvcache=ctx.is_fp8_kvcache,
            fp8_head_dim=ctx._fp8_head_dim,
            raw_fp8_mla_layout=ctx.raw_fp8_mla_layout,
        )

    def test_fp8_allocate_shape(self):
        """FP8 MLA cache should have shape [1, layers, blocks, 64, 1, 656]."""
        ctx = self._make_cache_context(is_fp8=True)
        num_blocks = 10
        ctx.allocate_kvcache(num_blocks)

        assert ctx.kv_cache is not None
        assert ctx.kv_cache.shape == (1, 4, 10, 64, 1, 656)
        assert ctx.kv_cache.dtype == torch.float8_e4m3fn

    def test_fp8_allocate_stride_padding(self):
        """FP8 MLA cache should be sliced from block_size+1 allocation (stride padding)."""
        ctx = self._make_cache_context(is_fp8=True)
        ctx.allocate_kvcache(8)

        # The underlying storage should be larger than the view
        # (block_size+1) rows per block vs block_size visible
        kv = ctx.kv_cache
        assert kv.shape[3] == 64  # visible block_size
        # stride[3] should be 1 * 656 (single row stride)
        # and stride[2] should be (block_size+1) * 1 * 656
        row_stride = kv.stride(3)
        block_stride = kv.stride(2)
        assert (
            block_stride == (64 + 1) * row_stride
        ), f"Block stride {block_stride} != (64+1) * row_stride {row_stride}"

    def test_fp8_memory_savings(self):
        """FP8 MLA should use ~57% of BF16 memory (656 vs 1152 bytes/token)."""
        ctx_bf16 = self._make_cache_context(is_fp8=False)
        ctx_bf16.allocate_kvcache(100)
        bf16_bytes = ctx_bf16.kv_cache.nelement() * ctx_bf16.kv_cache.element_size()

        ctx_fp8 = self._make_cache_context(is_fp8=True)
        ctx_fp8.allocate_kvcache(100)

        fp8_bytes = ctx_fp8.kv_cache.untyped_storage().nbytes()

        ratio = fp8_bytes / bf16_bytes
        # 656 / (576*2) = 0.569, but fp8 has +1 row padding so slightly more
        assert 0.55 < ratio < 0.65, f"FP8/BF16 ratio = {ratio:.3f}, expected ~0.57"

    def test_bf16_allocate_unchanged(self):
        """BF16 MLA allocation should be unaffected by is_fp8_kvcache changes."""
        ctx = self._make_cache_context(is_fp8=False)
        ctx.allocate_kvcache(10)

        assert ctx.kv_cache.shape == (1, 4, 10, 64, 1, 576)
        assert ctx.kv_cache.dtype == torch.bfloat16

    def test_per_layer_slice(self):
        """Each layer's cache should be independently addressable."""
        ctx = self._make_cache_context(is_fp8=True)
        ctx.allocate_kvcache(8)

        for layer_id in range(4):
            layer_cache = ctx.kv_cache[0][layer_id]
            assert layer_cache.shape == (8, 64, 1, 656)

    def test_block_stride_fp8(self):
        """block_stride should return correct byte offset for FP8."""
        ctx = self._make_cache_context(is_fp8=True)
        stride = self._layout(ctx).block_stride(1)
        expected = (64 + 1) * 656  # FP8 blocks include one padding row.
        assert stride == expected, f"block_stride(1) = {stride}, expected {expected}"

    def test_block_stride_bf16(self):
        """block_stride should return correct byte offset for BF16."""
        ctx = self._make_cache_context(is_fp8=False)
        stride = self._layout(ctx).block_stride(1)
        expected = 64 * 1 * 576 * 2  # block_size * kv_heads * head_dim * bf16_size
        assert stride == expected, f"block_stride(1) = {stride}, expected {expected}"

    def test_raw_fp8_allocate_shape_and_stride(self):
        ctx = self._make_cache_context(is_fp8=True, raw_fp8=True)
        ctx.allocate_kvcache(10)

        assert ctx.kv_cache.shape == (1, 4, 10, 64, 1, 576)
        assert ctx.kv_cache.dtype == torch.float8_e4m3fn
        assert ctx.kv_cache.stride(2) == 64 * ctx.kv_cache.stride(3)
        assert self._layout(ctx).block_stride(1) == 64 * 576


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

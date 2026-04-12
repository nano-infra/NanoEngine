"""Unit tests for FP8 store_kcache integration in attention.py."""

import pytest
import torch
from nanodeploy.backends.gpu_generic.kernels.kv_store import store_kcache

from nanodeploy.backends.hopper.kernels.fp8_utils import (
    D_NOPE,
    D_TOTAL,
    dequantize_nope_fp8,
    FP8_BYTES_PER_TOKEN,
    store_kcache_fp8,
    unpack_mla_fp8,
)


class TestStoreFP8Integration:

    def test_store_fp8_writes_correct_data(self):
        """Write BF16 compressed KV via store_kcache_fp8 and verify content."""
        N = 16
        num_blocks = 4
        block_size = 64

        # Source: BF16 compressed KV [N, 1, 576]
        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device="cuda")

        # FP8 paged cache: [num_blocks, block_size, 1, 656] float8_e4m3fn
        # Allocate with block_size+1 padding (same as CacheContext.allocate_kvcache)
        k_cache_padded = torch.zeros(
            num_blocks,
            block_size + 1,
            1,
            FP8_BYTES_PER_TOKEN,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        k_cache = k_cache_padded[:, :block_size, :, :]

        slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")
        store_kcache_fp8(key, k_cache, slot_mapping)

        # Verify: read back and check RoPE is exact, NoPE is close
        for i in range(N):
            slot = slot_mapping[i].item()
            blk = slot // block_size
            off = slot % block_size
            packed = k_cache[blk, off, 0, :].view(torch.uint8).unsqueeze(0)
            fp8_nope, scales, rope = unpack_mla_fp8(packed)

            # RoPE should be bit-exact
            assert torch.equal(rope[0], key[i, 0, D_NOPE:])

            # NoPE should be close
            nope_rec = dequantize_nope_fp8(fp8_nope, scales)
            err = (nope_rec[0].float() - key[i, 0, :D_NOPE].float()).abs().max()
            absmax = key[i, 0, :D_NOPE].float().abs().max()
            assert err / (absmax + 1e-8) < 0.06

    def test_store_bf16_still_works(self):
        """BF16 store_kcache path should be unaffected."""
        N = 8
        num_blocks = 2
        block_size = 64

        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device="cuda")
        k_cache = torch.zeros(
            num_blocks,
            block_size,
            1,
            D_TOTAL,
            dtype=torch.bfloat16,
            device="cuda",
        )
        slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")
        store_kcache(key, k_cache, slot_mapping)

        # Read back — should be exact for BF16
        cache_flat = k_cache.reshape(-1, D_TOTAL)
        for i in range(N):
            assert torch.equal(cache_flat[i], key[i, 0])

    def test_store_fp8_scattered_slots(self):
        """FP8 store with non-contiguous slot mapping."""
        N = 5
        num_blocks = 4
        block_size = 64

        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device="cuda")

        k_cache_padded = torch.zeros(
            num_blocks,
            block_size + 1,
            1,
            FP8_BYTES_PER_TOKEN,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        k_cache = k_cache_padded[:, :block_size, :, :]

        # Scatter to non-contiguous slots
        slot_mapping = torch.tensor(
            [10, 50, 100, 200, 3], dtype=torch.int32, device="cuda"
        )
        store_kcache_fp8(key, k_cache, slot_mapping)

        cache_flat = k_cache  # [num_blocks, block_size, 1, 656]
        for i in range(N):
            slot = slot_mapping[i].item()
            blk = slot // block_size
            off = slot % block_size
            packed = cache_flat[blk, off, 0, :].view(torch.uint8).unsqueeze(0)
            _, _, rope = unpack_mla_fp8(packed)
            assert torch.equal(rope[0], key[i, 0, D_NOPE:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

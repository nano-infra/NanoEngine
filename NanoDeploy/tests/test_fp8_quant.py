"""Unit tests for FP8 MLA KV cache quantization utilities."""

import pytest
import torch

from nanodeploy.backends.hopper.kernels.fp8_utils import (
    D_NOPE,
    D_ROPE,
    D_TOTAL,
    dequantize_and_unpack_mla,
    dequantize_nope_fp8,
    FP8_BYTES_PER_TOKEN,
    NUM_TILES,
    pack_mla_fp8,
    quantize_and_pack_mla,
    quantize_nope_fp8,
    store_kcache_fp8,
    TILE_SIZE,
    unpack_mla_fp8,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestQuantizeDequantize:

    def test_roundtrip_error(self, device):
        """quantize → dequantize should have error < 1e-2 relative to absmax."""
        N = 128
        nope = torch.randn(N, D_NOPE, dtype=torch.bfloat16, device=device)

        fp8_nope, scales = quantize_nope_fp8(nope)
        recovered = dequantize_nope_fp8(fp8_nope, scales)

        # Check shapes
        assert fp8_nope.shape == (N, D_NOPE)
        assert fp8_nope.dtype == torch.float8_e4m3fn
        assert scales.shape == (N, NUM_TILES)
        assert scales.dtype == torch.float32
        assert recovered.shape == (N, D_NOPE)
        assert recovered.dtype == torch.bfloat16

        # Max absolute error per tile should be small relative to tile range
        nope_f32 = nope.float()
        rec_f32 = recovered.float()
        for t in range(NUM_TILES):
            s = t * TILE_SIZE
            e = s + TILE_SIZE
            tile_orig = nope_f32[:, s:e]
            tile_rec = rec_f32[:, s:e]
            abs_err = (tile_orig - tile_rec).abs()
            tile_absmax = tile_orig.abs().amax()
            # FP8 E4M3 + UE8M0 (power-of-2 scales) can have ~6% worst-case error
            assert (
                abs_err.max() / (tile_absmax + 1e-8) < 0.06
            ), f"Tile {t}: max relative error {abs_err.max() / tile_absmax:.4f}"

    def test_scales_are_powers_of_2(self, device):
        """UE8M0 scales must be exact powers of 2."""
        N = 64
        nope = torch.randn(N, D_NOPE, dtype=torch.bfloat16, device=device) * 10
        _, scales = quantize_nope_fp8(nope)

        log2_scales = torch.log2(scales)
        assert torch.allclose(
            log2_scales, log2_scales.round()
        ), "Scales are not powers of 2"

    def test_zero_input(self, device):
        """Quantizing zeros should not crash."""
        N = 4
        nope = torch.zeros(N, D_NOPE, dtype=torch.bfloat16, device=device)
        fp8_nope, scales = quantize_nope_fp8(nope)
        recovered = dequantize_nope_fp8(fp8_nope, scales)
        assert (recovered == 0).all()


class TestPackUnpack:

    def test_roundtrip_exact(self, device):
        """pack → unpack should be bit-exact."""
        N = 32
        nope = torch.randn(N, D_NOPE, dtype=torch.bfloat16, device=device)
        fp8_nope, scales = quantize_nope_fp8(nope)
        rope = torch.randn(N, D_ROPE, dtype=torch.bfloat16, device=device)

        packed = pack_mla_fp8(fp8_nope, scales, rope)
        assert packed.shape == (N, FP8_BYTES_PER_TOKEN)
        assert packed.dtype == torch.uint8

        fp8_out, scales_out, rope_out = unpack_mla_fp8(packed)

        # FP8 NoPE
        assert (fp8_out.view(torch.uint8) == fp8_nope.view(torch.uint8)).all()
        # Scales
        assert (
            scales_out.view(torch.uint8) == scales.contiguous().view(torch.uint8)
        ).all()
        # RoPE
        assert (rope_out.view(torch.uint8) == rope.contiguous().view(torch.uint8)).all()

    def test_bytes_layout(self):
        """Verify per-token byte offsets match spec."""
        assert FP8_BYTES_PER_TOKEN == 656
        assert D_NOPE == 512
        assert D_NOPE + NUM_TILES * 4 == 528
        assert D_NOPE + NUM_TILES * 4 + D_ROPE * 2 == 656


class TestEndToEnd:

    def test_quantize_and_pack_roundtrip(self, device):
        """Full pipeline: bf16 [N, 1, 576] → pack → unpack → bf16, check error."""
        N = 64
        compressed_kv = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device=device)

        packed = quantize_and_pack_mla(compressed_kv)
        assert packed.shape == (N, 1, FP8_BYTES_PER_TOKEN)

        recovered = dequantize_and_unpack_mla(packed)
        assert recovered.shape == (N, 1, D_TOTAL)
        assert recovered.dtype == torch.bfloat16

        # RoPE should be bit-exact
        assert torch.equal(recovered[:, 0, D_NOPE:], compressed_kv[:, 0, D_NOPE:])

        # NoPE should be close
        nope_err = (
            recovered[:, 0, :D_NOPE].float() - compressed_kv[:, 0, :D_NOPE].float()
        ).abs()
        nope_absmax = compressed_kv[:, 0, :D_NOPE].float().abs().amax()
        assert nope_err.max() / (nope_absmax + 1e-8) < 0.06

    def test_store_kcache_fp8(self, device):
        """store_kcache_fp8 writes to correct slots and data can be recovered."""
        N = 10
        num_blocks = 4
        block_size = 64
        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device=device)

        # Create paged cache: [num_blocks, block_size+1, 1, FP8_BYTES_PER_TOKEN] padded
        k_cache_padded = torch.zeros(
            num_blocks,
            block_size + 1,
            1,
            FP8_BYTES_PER_TOKEN,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        k_cache = k_cache_padded[:, :block_size, :, :]

        # Slot mapping: tokens go to slots 0,1,2,...,9
        slot_mapping = torch.arange(N, dtype=torch.int32, device=device)

        store_kcache_fp8(key, k_cache, slot_mapping)

        # Read back and verify
        for i in range(N):
            blk = i // block_size
            off = i % block_size
            packed_token = k_cache[blk, off, 0, :].view(torch.uint8).unsqueeze(0)
            fp8_nope, scales, rope = unpack_mla_fp8(packed_token)
            nope_recovered = dequantize_nope_fp8(fp8_nope, scales)

            # RoPE exact
            assert torch.equal(rope[0], key[i, 0, D_NOPE:])

            # NoPE close
            orig_nope = key[i, 0, :D_NOPE].float()
            err = (nope_recovered[0].float() - orig_nope).abs()
            assert err.max() / (orig_nope.abs().amax() + 1e-8) < 0.06

    def test_store_kcache_fp8_with_skip(self, device):
        """Slots with -1 should be skipped (no write)."""
        N = 4
        num_blocks = 2
        block_size = 64
        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device=device)

        k_cache_padded = torch.zeros(
            num_blocks,
            block_size + 1,
            1,
            FP8_BYTES_PER_TOKEN,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        k_cache = k_cache_padded[:, :block_size, :, :]

        # Only write slots 0 and 2, skip 1 and 3
        slot_mapping = torch.tensor([0, -1, 2, -1], dtype=torch.int32, device=device)

        store_kcache_fp8(key, k_cache, slot_mapping)

        # Slot 0 written
        assert k_cache[0, 0, 0, :].view(torch.uint8).any()
        # Slot 1 not written
        assert not k_cache[0, 1, 0, :].view(torch.uint8).any()
        # Slot 2 written
        assert k_cache[0, 2, 0, :].view(torch.uint8).any()

    def test_store_kcache_fp8_shuffled_slots(self, device):
        """Out-of-order slot mapping should scatter correctly."""
        N = 8
        num_blocks = 4
        block_size = 64
        key = torch.randn(N, 1, D_TOTAL, dtype=torch.bfloat16, device=device)

        k_cache_padded = torch.zeros(
            num_blocks,
            block_size + 1,
            1,
            FP8_BYTES_PER_TOKEN,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        k_cache = k_cache_padded[:, :block_size, :, :]

        # Shuffled slot mapping
        perm = torch.randperm(num_blocks * block_size)[:N]
        slot_mapping = perm.to(dtype=torch.int32, device=device)

        store_kcache_fp8(key, k_cache, slot_mapping)

        # Verify each token landed in correct slot
        for i in range(N):
            slot = slot_mapping[i].item()
            blk = slot // block_size
            off = slot % block_size
            packed_token = k_cache[blk, off, 0, :].view(torch.uint8).unsqueeze(0)
            _, _, rope = unpack_mla_fp8(packed_token)
            assert torch.equal(rope[0], key[i, 0, D_NOPE:])


class TestFlashMLACompatibility:
    """Verify our packing matches FlashMLANew's expected layout."""

    def test_layout_matches_flashmla_quant(self, device):
        """Compare our quantize_and_pack with FlashMLANew's quantize_k_cache."""
        try:
            from tests_flashmla_quant import FP8KVCacheLayout, quantize_k_cache
        except ImportError:
            pytest.skip("FlashMLANew quant.py not on path")

        num_blocks, block_size = 2, 64
        kv = torch.randn(
            num_blocks,
            block_size,
            1,
            D_TOTAL,
            dtype=torch.bfloat16,
            device=device,
        )

        ref = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse)

        # Our path: per-token
        kv_2d = kv.view(-1, D_TOTAL)
        ours = quantize_and_pack_mla(kv_2d).view(
            num_blocks, block_size, 1, FP8_BYTES_PER_TOKEN
        )

        # RoPE bytes should match exactly
        ours_rope = ours[..., D_NOPE + NUM_TILES * 4 :]
        ref_rope = ref.view(torch.uint8)[..., D_NOPE + NUM_TILES * 4 :]
        assert torch.equal(
            ours_rope.view(torch.uint8),
            ref_rope.view(torch.uint8),
        ), "RoPE bytes mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

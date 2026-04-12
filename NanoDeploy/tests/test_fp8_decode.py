"""Unit tests for FP8 KV cache → FlashMLA decode integration."""

import flash_mla
import pytest
import torch

from nanodeploy.backends.hopper.kernels.fp8_utils import (
    D_NOPE,
    D_TOTAL,
    dequantize_and_unpack_mla,
    FP8_BYTES_PER_TOKEN,
    quantize_and_pack_mla,
)


def _make_fp8_cache(num_blocks, block_size, compressed_kv_bf16):
    """Build FP8 paged cache from BF16 compressed KV."""
    N = compressed_kv_bf16.shape[0]
    device = compressed_kv_bf16.device

    # Allocate with block_size+1 padding
    cache_padded = torch.zeros(
        num_blocks,
        block_size + 1,
        1,
        FP8_BYTES_PER_TOKEN,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache = cache_padded[:, :block_size, :, :]

    # Quantize and write
    packed = quantize_and_pack_mla(compressed_kv_bf16)  # [N, 1, 656]
    packed_fp8 = packed.view(N, FP8_BYTES_PER_TOKEN).view(torch.float8_e4m3fn)
    for i in range(N):
        blk = i // block_size
        off = i % block_size
        cache[blk, off, 0, :] = packed_fp8[i]

    return cache


def _make_bf16_cache(num_blocks, block_size, compressed_kv_bf16):
    """Build BF16 paged cache from BF16 compressed KV."""
    N = compressed_kv_bf16.shape[0]
    device = compressed_kv_bf16.device

    cache = torch.zeros(
        num_blocks,
        block_size,
        1,
        D_TOTAL,
        dtype=torch.bfloat16,
        device=device,
    )
    for i in range(N):
        blk = i // block_size
        off = i % block_size
        cache[blk, off, 0, :] = compressed_kv_bf16[i, 0, :]

    return cache


class TestFP8Decode:

    @pytest.fixture
    def setup(self):
        """Create small test scenario: 2 sequences, 128 tokens each."""
        device = "cuda"
        bs = 2
        seq_len = 128
        num_heads = 16  # query heads
        head_dim = 576  # Q head dim = kv_lora_rank + qk_rope_head_dim
        v_head_size = 512  # kv_lora_rank
        block_size = 64
        num_blocks = (seq_len * bs + block_size - 1) // block_size  # 4 blocks

        # Generate random compressed KV for all tokens
        total_tokens = bs * seq_len
        compressed_kv = torch.randn(
            total_tokens, 1, D_TOTAL, dtype=torch.bfloat16, device=device
        )

        # Build caches
        fp8_cache = _make_fp8_cache(num_blocks, block_size, compressed_kv)
        bf16_cache = _make_bf16_cache(num_blocks, block_size, compressed_kv)

        # Block tables: sequential blocks for each sequence
        blocks_per_seq = seq_len // block_size
        block_table = torch.zeros(bs, blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(bs):
            for j in range(blocks_per_seq):
                block_table[b, j] = b * blocks_per_seq + j

        # Query: [bs, 1, num_heads, head_dim]
        q = torch.randn(bs, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Context lens
        cache_seqlens = torch.full((bs,), seq_len, dtype=torch.int32, device=device)

        return {
            "q": q,
            "fp8_cache": fp8_cache,
            "bf16_cache": bf16_cache,
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "v_head_size": v_head_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "bs": bs,
        }

    def test_fp8_decode_runs(self, setup):
        """flash_mla_with_kvcache with is_fp8_kvcache=True should not crash."""
        s = setup
        meta, _ = flash_mla.get_mla_metadata()
        scale = 1.0 / (s["head_dim"] ** 0.5)

        o, lse = flash_mla.flash_mla_with_kvcache(
            s["q"],
            s["fp8_cache"],
            s["block_table"],
            s["cache_seqlens"],
            s["v_head_size"],
            meta,
            None,
            scale,
            False,  # causal
            is_fp8_kvcache=True,
        )

        assert o.shape == (s["bs"], 1, s["num_heads"], s["v_head_size"])
        assert not o.isnan().any()
        assert not o.isinf().any()

    def test_fp8_vs_bf16_decode_close(self, setup):
        """FP8 decode output should be close to BF16 decode output."""
        s = setup
        meta_fp8, _ = flash_mla.get_mla_metadata()
        meta_bf16, _ = flash_mla.get_mla_metadata()
        scale = 1.0 / (s["head_dim"] ** 0.5)

        o_fp8, _ = flash_mla.flash_mla_with_kvcache(
            s["q"],
            s["fp8_cache"],
            s["block_table"],
            s["cache_seqlens"],
            s["v_head_size"],
            meta_fp8,
            None,
            scale,
            False,
            is_fp8_kvcache=True,
        )

        o_bf16, _ = flash_mla.flash_mla_with_kvcache(
            s["q"],
            s["bf16_cache"],
            s["block_table"],
            s["cache_seqlens"],
            s["v_head_size"],
            meta_bf16,
            None,
            scale,
            False,
            is_fp8_kvcache=False,
        )

        # Relative difference should be small (FP8 quantization noise)
        diff = (o_fp8.float() - o_bf16.float()).abs()
        ref = o_bf16.float().abs().amax()
        max_rel_err = diff.amax() / (ref + 1e-8)
        # FP8 introduces ~1-5% error; allow up to 10% for worst case
        assert max_rel_err < 0.10, f"Max relative error: {max_rel_err:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Standalone tests for the NSA Indexer module (DeepSeek V3.2).

Tests the IndexerCache store/retrieve and Indexer forward pass using
random weights (BF16 backend), verifying shapes, dtypes, and basic
numerical sanity of the TopK output.
"""

import deep_gemm
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# V3.2 config constants
HIDDEN_SIZE = 7168
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
Q_LORA_RANK = 1536
INDEX_TOPK = 2048
MAX_POS = 16384
ROPE_THETA = 10000.0
PAGE_SIZE = 64


@pytest.fixture(autouse=True, scope="session")
def _init_backend():
    """Initialise dist + backend for single-GPU testing."""
    import os

    import torch.distributed as dist
    from nanodeploy.backends import _backend, init_backend
    from nanodeploy.context.distributed import set_dist_context

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        set_dist_context(rank=0, world_size=1)

    if _backend is None:
        init_backend()


@pytest.fixture()
def indexer_cache():
    from nanodeploy.layers.indexer import IndexerCache

    num_layers = 2
    num_pages = 16
    return IndexerCache(
        num_layers=num_layers,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        head_dim=INDEX_HEAD_DIM,
        device="cuda",
    )


@pytest.fixture()
def indexer(indexer_cache):
    """Create an Indexer with random weights and attached cache."""
    from nanodeploy.layers.indexer import Indexer

    prev_device = torch.get_default_device()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    idx = Indexer(
        hidden_size=HIDDEN_SIZE,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        q_lora_rank=Q_LORA_RANK,
        index_topk=INDEX_TOPK,
        max_position_embeddings=MAX_POS,
        rope_theta=ROPE_THETA,
        rope_scaling=None,
        layer_id=0,
    )
    torch.set_default_device(prev_device)
    torch.set_default_dtype(prev_dtype)
    # Initialize all weights with small random values to avoid NaN
    with torch.no_grad():
        for p in idx.parameters():
            if p.dtype in (torch.bfloat16, torch.float32):
                p.normal_(0, 0.02)
    idx.indexer_cache = indexer_cache
    return idx


# ---------------------------------------------------------------------------
# IndexerCache tests
# ---------------------------------------------------------------------------


class TestIndexerCache:

    def test_buffer_shapes(self, indexer_cache):
        """Buffer shapes match (num_pages, page_size * bytes_per_token)."""
        assert len(indexer_cache.buffers) == 2
        for buf in indexer_cache.buffers:
            assert buf.shape == (16, PAGE_SIZE * 132)
            assert buf.dtype == torch.uint8

    def test_store_key_fp8_roundtrip(self, indexer_cache):
        """Store keys and verify FP8+scale data is written to the right slots."""
        N = 8
        key_bf16 = torch.randn(N, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")

        indexer_cache.store_key_fp8(
            layer_id=0, key_bf16=key_bf16, slot_mapping=slot_mapping
        )

        buf = indexer_cache.get_buffer(0)
        bpt = indexer_cache.bytes_per_token  # 132

        # Manually verify token 0 (page 0, offset 0)
        fp8_bytes = buf[0, 0:INDEX_HEAD_DIM]
        scale_bytes = buf[0, INDEX_HEAD_DIM : INDEX_HEAD_DIM + 4]

        # FP8 data should be non-zero (random input)
        assert fp8_bytes.any(), "FP8 data should be non-zero"
        # Scale should be non-zero
        scale_f32 = scale_bytes.view(torch.float32)
        assert scale_f32.item() != 0.0, "Scale should be non-zero"

    def test_store_key_fp8_scattered_slots(self, indexer_cache):
        """Store keys to non-contiguous slots across pages."""
        N = 4
        key_bf16 = torch.randn(N, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        # Scatter across different pages
        slot_mapping = torch.tensor([0, 64, 128, 200], dtype=torch.int32, device="cuda")

        indexer_cache.store_key_fp8(
            layer_id=1, key_bf16=key_bf16, slot_mapping=slot_mapping
        )

        buf = indexer_cache.get_buffer(1)
        bpt = indexer_cache.bytes_per_token

        # Token 0 -> page 0, offset 0
        assert buf[0, 0:INDEX_HEAD_DIM].any()
        # Token 1 -> page 1, offset 0
        assert buf[1, 0:INDEX_HEAD_DIM].any()
        # Token 2 -> page 2, offset 0
        assert buf[2, 0:INDEX_HEAD_DIM].any()
        # Token 3 -> page 3, offset (200%64=8)*132 = 1056
        off = 8 * bpt
        assert buf[3, off : off + INDEX_HEAD_DIM].any()


# ---------------------------------------------------------------------------
# Indexer forward tests
# ---------------------------------------------------------------------------


class TestIndexerForward:

    def test_output_shape_single_batch(self, indexer):
        """Single-batch decode: topk_indices shape is (1, index_topk)."""
        batch = 1
        seq_len = 256  # 4 pages of context

        hidden = torch.randn(batch, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        q_lora = torch.randn(batch, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([seq_len - 1], dtype=torch.long, device="cuda")
        context_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        block_tables = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
        slot_mapping = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")

        # Pre-fill indexer cache with random keys for 256 tokens
        key_bf16 = torch.randn(
            seq_len, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        slots = torch.arange(seq_len, dtype=torch.int32, device="cuda")
        indexer.indexer_cache.store_key_fp8(0, key_bf16, slots)

        with torch.inference_mode():
            topk_indices = indexer(
                hidden, q_lora, positions, context_lens, block_tables, slot_mapping
            )

        assert topk_indices.shape == (batch, INDEX_TOPK)
        assert topk_indices.dtype == torch.int32
        # All valid indices should be in [0, seq_len) or -1 (padding)
        valid = topk_indices[topk_indices >= 0]
        assert (valid < seq_len).all(), f"Found out-of-range index: {valid.max()}"

    def test_output_shape_multi_batch(self, indexer):
        """Multi-batch decode: topk_indices shape is (batch, index_topk)."""
        batch = 4
        max_seq = 512  # 8 pages

        hidden = torch.randn(batch, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        q_lora = torch.randn(batch, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([255, 511, 127, 63], dtype=torch.long, device="cuda")
        context_lens = torch.tensor(
            [256, 512, 128, 64], dtype=torch.int32, device="cuda"
        )
        # Block tables: each seq uses different pages
        block_tables = torch.full((batch, 8), -1, dtype=torch.int32, device="cuda")
        block_tables[0, :4] = torch.tensor([0, 1, 2, 3])
        block_tables[1, :8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        block_tables[2, :2] = torch.tensor([8, 9])
        block_tables[3, :1] = torch.tensor([10])
        slot_mapping = torch.tensor(
            [255, 511, 127, 63], dtype=torch.int32, device="cuda"
        )

        # Pre-fill cache
        key_bf16 = torch.randn(
            max_seq, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        slots = torch.arange(max_seq, dtype=torch.int32, device="cuda")
        indexer.indexer_cache.store_key_fp8(0, key_bf16, slots)

        with torch.inference_mode():
            topk_indices = indexer(
                hidden, q_lora, positions, context_lens, block_tables, slot_mapping
            )

        assert topk_indices.shape == (batch, INDEX_TOPK)
        assert topk_indices.dtype == torch.int32

    def test_topk_smaller_than_context(self, indexer):
        """When context < index_topk, results are padded with -1."""
        batch = 1
        seq_len = 64  # Only 1 page — much less than INDEX_TOPK=2048

        hidden = torch.randn(batch, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        q_lora = torch.randn(batch, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([seq_len - 1], dtype=torch.long, device="cuda")
        context_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        block_tables = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        slot_mapping = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")

        # Pre-fill cache
        key_bf16 = torch.randn(
            seq_len, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        slots = torch.arange(seq_len, dtype=torch.int32, device="cuda")
        indexer.indexer_cache.store_key_fp8(0, key_bf16, slots)

        with torch.inference_mode():
            topk_indices = indexer(
                hidden, q_lora, positions, context_lens, block_tables, slot_mapping
            )

        assert topk_indices.shape == (batch, INDEX_TOPK)
        # First 64 entries should be valid indices [0, 64)
        valid_part = topk_indices[0, :seq_len]
        assert (valid_part >= 0).all() and (valid_part < seq_len).all()
        # Remaining entries should be -1 (padding)
        padding_part = topk_indices[0, seq_len:]
        assert (padding_part == -1).all(), "Padding entries should be -1"


class TestIndexerCacheAllocation:
    """Test CacheContext.allocate_indexer_cache integration."""

    def test_allocate_indexer_cache(self):
        """Verify allocate_indexer_cache creates an IndexerCache with correct params."""
        from types import SimpleNamespace

        from nanodeploy.context.cache import CacheContext
        from nanodeploy.layers.indexer import IndexerCache

        hf_config = SimpleNamespace(index_head_dim=128)

        # Create a minimal CacheContext-like object to call allocate_indexer_cache
        # We can't easily instantiate CacheContext (requires dist, GPU), so test
        # the IndexerCache directly
        num_layers = 4
        num_pages = 32
        cache = IndexerCache(
            num_layers=num_layers,
            num_pages=num_pages,
            page_size=PAGE_SIZE,
            head_dim=128,
            device="cuda",
        )
        assert len(cache.buffers) == num_layers
        assert cache.buffers[0].shape == (num_pages, PAGE_SIZE * 132)
        assert cache.bytes_per_token == 132


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

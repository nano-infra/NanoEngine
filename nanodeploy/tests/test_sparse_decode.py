"""Tests for Phase 3: Sparse Decode integration.

Tests:
  - topk_indices_to_physical: logical → physical index conversion
  - Indexer.store_prefill_keys: prefill key storage to IndexerCache
  - FlashMLAImpl sparse decode path (shape/dispatch validation)
"""

import pytest
import torch

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
        os.environ.setdefault("MASTER_PORT", "29502")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        set_dist_context(rank=0, world_size=1)

    if _backend is None:
        init_backend()


@pytest.fixture()
def indexer_cache():
    from nanodeploy.layers.indexer import IndexerCache

    return IndexerCache(
        num_layers=2,
        num_pages=32,
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
    with torch.no_grad():
        for p in idx.parameters():
            if p.dtype in (torch.bfloat16, torch.float32):
                p.normal_(0, 0.02)
    idx.indexer_cache = indexer_cache
    return idx


# ---------------------------------------------------------------------------
# topk_indices_to_physical
# ---------------------------------------------------------------------------


class TestTopkIndicesToPhysical:

    def test_basic_conversion(self):
        """Verify logical→physical mapping matches manual calculation."""
        from nanodeploy.backends.hopper.layers.attention import topk_indices_to_physical

        block_size = 64
        # block_table: batch=1, 4 blocks with physical IDs [5, 10, 2, 8]
        block_table = torch.tensor([[5, 10, 2, 8]], dtype=torch.int32, device="cuda")
        # Logical indices: token 0 (block 0, off 0), token 65 (block 1, off 1),
        #                  token 192 (block 3, off 0), token 130 (block 2, off 2)
        topk_indices = torch.tensor(
            [[0, 65, 192, 130]], dtype=torch.int32, device="cuda"
        )

        result = topk_indices_to_physical(topk_indices, block_table, block_size)

        expected = torch.tensor(
            [
                [
                    5 * 64 + 0,  # logical 0   → block 0 (phys 5), offset 0
                    10 * 64 + 1,  # logical 65  → block 1 (phys 10), offset 1
                    8 * 64 + 0,  # logical 192 → block 3 (phys 8), offset 0
                    2 * 64 + 2,  # logical 130 → block 2 (phys 2), offset 2
                ]
            ],
            dtype=torch.int32,
            device="cuda",
        )
        assert torch.equal(result, expected)

    def test_negative_padding(self):
        """Padding entries (-1) are preserved as -1."""
        from nanodeploy.backends.hopper.layers.attention import topk_indices_to_physical

        block_table = torch.tensor([[3, 7]], dtype=torch.int32, device="cuda")
        topk_indices = torch.tensor(
            [[10, -1, 70, -1]], dtype=torch.int32, device="cuda"
        )
        result = topk_indices_to_physical(topk_indices, block_table, block_size=64)

        assert (
            result[0, 0].item() == 3 * 64 + 10
        )  # logical 10 → block 0 (phys 3), offset 10
        assert result[0, 1].item() == -1  # padding → preserved as -1
        assert (
            result[0, 2].item() == 7 * 64 + 6
        )  # logical 70 → block 1 (phys 7), offset 6
        assert result[0, 3].item() == -1  # padding → preserved as -1

    def test_multi_batch(self):
        """Multi-batch conversion works correctly."""
        from nanodeploy.backends.hopper.layers.attention import topk_indices_to_physical

        block_table = torch.tensor([[1, 2], [5, 6]], dtype=torch.int32, device="cuda")
        topk_indices = torch.tensor(
            [[0, 64], [63, 127]], dtype=torch.int32, device="cuda"
        )
        result = topk_indices_to_physical(topk_indices, block_table, block_size=64)

        # Batch 0: logical 0 → phys 1*64+0=64, logical 64 → phys 2*64+0=128
        assert result[0, 0].item() == 64
        assert result[0, 1].item() == 128
        # Batch 1: logical 63 → phys 5*64+63=383, logical 127 → phys 6*64+63=447
        assert result[1, 0].item() == 383
        assert result[1, 1].item() == 447


# ---------------------------------------------------------------------------
# Indexer.store_prefill_keys
# ---------------------------------------------------------------------------


class TestStorePrefillKeys:

    def test_stores_keys_to_cache(self, indexer):
        """store_prefill_keys writes non-zero FP8 data to the indexer cache."""
        seq_len = 128  # 2 pages
        hidden = torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(seq_len, dtype=torch.long, device="cuda")
        slot_mapping = torch.arange(seq_len, dtype=torch.int32, device="cuda")

        with torch.inference_mode():
            indexer.store_prefill_keys(hidden, positions, slot_mapping)

        buf = indexer.indexer_cache.get_buffer(0)
        # Page 0 and page 1 should have non-zero data
        assert buf[0].any(), "Page 0 should have data"
        assert buf[1].any(), "Page 1 should have data"

    def test_key_shape_matches_direct(self, indexer):
        """_compute_key returns (num_tokens, head_dim) BF16."""
        N = 4
        hidden = torch.randn(N, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(N, dtype=torch.long, device="cuda")

        with torch.inference_mode():
            key = indexer._compute_key(hidden, positions)

        assert key.shape == (N, INDEX_HEAD_DIM)
        assert key.dtype == torch.bfloat16
        assert not key.isnan().any()

    def test_prefill_then_decode_indexer(self, indexer):
        """After prefill stores keys, decode indexer forward succeeds."""
        seq_len = 256  # 4 pages
        batch = 1

        # Prefill: store keys
        hidden_pf = torch.randn(
            seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        positions_pf = torch.arange(seq_len, dtype=torch.long, device="cuda")
        slots_pf = torch.arange(seq_len, dtype=torch.int32, device="cuda")

        with torch.inference_mode():
            indexer.store_prefill_keys(hidden_pf, positions_pf, slots_pf)

        # Decode: run indexer forward (one new token at position seq_len)
        hidden_dec = torch.randn(
            batch, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        q_lora_dec = torch.randn(
            batch, Q_LORA_RANK, dtype=torch.bfloat16, device="cuda"
        )
        positions_dec = torch.tensor([seq_len], dtype=torch.long, device="cuda")
        context_lens = torch.tensor([seq_len + 1], dtype=torch.int32, device="cuda")
        # 5 pages total: 4 full + 1 with the new token
        block_tables = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32, device="cuda")
        slot_mapping_dec = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        with torch.inference_mode():
            topk_indices = indexer(
                hidden_dec,
                q_lora_dec,
                positions_dec,
                context_lens,
                block_tables,
                slot_mapping_dec,
            )

        assert topk_indices.shape == (batch, INDEX_TOPK)
        assert topk_indices.dtype == torch.int32
        # Valid indices should be in [0, seq_len+1)
        valid = topk_indices[topk_indices >= 0]
        assert (valid < seq_len + 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

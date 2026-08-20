from types import SimpleNamespace
from unittest.mock import patch

import torch

import nanodeploy.layers.attention as attention
import nanodeploy.worker.mla_metadata as mla_metadata


class _FakeDecodeFlashMLA:
    metadata_calls = 0
    forward_calls = 0

    @classmethod
    def get_mla_metadata(cls, *_args):
        cls.metadata_calls += 1
        raise AssertionError("attention layers must reuse prepared decode metadata")

    @classmethod
    def flash_mla_with_kvcache(
        cls,
        q: torch.Tensor,
        _k_cache: torch.Tensor,
        _block_tables: torch.Tensor,
        _context_lens: torch.Tensor,
        v_head_size: int,
        *_args,
    ):
        cls.forward_calls += 1
        return torch.zeros((*q.shape[:-1], v_head_size)), torch.empty(0)


def test_flash_mla_decode_layers_reuse_prepared_metadata():
    context = SimpleNamespace(
        is_prefill=False,
        is_dummy=False,
        use_sp_a2a=False,
        attention_compute_bs=2,
        context_lens_for_attn=torch.tensor([9, 17], dtype=torch.int32),
        block_tables=torch.tensor([[1], [2]], dtype=torch.int32),
        tile_scheduler_metadata=torch.ones((3, 8), dtype=torch.int32),
        num_splits=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    _FakeDecodeFlashMLA.metadata_calls = 0
    _FakeDecodeFlashMLA.forward_calls = 0

    with (
        patch.object(attention, "get_context", return_value=context),
        patch.object(
            attention,
            "get_dist_context",
            return_value=SimpleNamespace(attn_sp_rank=0, attn_sp_world_size=1),
        ),
        patch.object(attention, "flash_mla", _FakeDecodeFlashMLA),
    ):
        implementation = attention.FlashMLAImpl(
            num_heads=2,
            head_size=4,
            num_kv_heads=1,
            v_head_size=3,
        )
        q = torch.randn(2, 2, 4)
        empty_cache = torch.empty(0)
        for _ in range(2):
            output = implementation.forward(
                q,
                torch.empty(0),
                torch.empty(0),
                empty_cache,
                empty_cache,
            )

    assert output.shape == (2, 2, 3)
    assert _FakeDecodeFlashMLA.metadata_calls == 0
    assert _FakeDecodeFlashMLA.forward_calls == 2


def test_prepare_decode_mla_metadata_uses_stable_graph_buffer(monkeypatch):
    calls = []

    def fake_get_mla_metadata(context_lens, tokens_per_k_head, num_kv_heads):
        calls.append((context_lens.clone(), tokens_per_k_head, num_kv_heads))
        batch_size = context_lens.numel()
        return (
            torch.arange(24, dtype=torch.int32).reshape(3, 8),
            torch.arange(batch_size + 1, dtype=torch.int32),
        )

    monkeypatch.setattr(
        mla_metadata.flash_mla,
        "get_mla_metadata",
        fake_get_mla_metadata,
    )
    hf_config = SimpleNamespace(
        num_attention_heads=128,
        num_key_value_heads=1,
    )
    context_lens = torch.tensor([11, 23, 37], dtype=torch.int32)
    tile_buffer = torch.full((8, 8), -1, dtype=torch.int32)
    splits_buffer = torch.full((17,), -1, dtype=torch.int32)

    tile_metadata, num_splits = mla_metadata.prepare_decode_mla_metadata(
        hf_config,
        context_lens,
        tile_buffer,
        splits_buffer,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0][0], context_lens)
    assert calls[0][1:] == (128, 1)
    assert tile_metadata.shape == (3, 8)
    assert num_splits.shape == (4,)
    assert tile_metadata.data_ptr() == tile_buffer.data_ptr()
    assert num_splits.data_ptr() == splits_buffer.data_ptr()
    assert torch.equal(
        tile_metadata,
        torch.arange(24, dtype=torch.int32).reshape(3, 8),
    )
    assert torch.equal(num_splits, torch.arange(4, dtype=torch.int32))
    assert torch.all(tile_buffer[3:] == -1)
    assert torch.all(splits_buffer[4:] == -1)

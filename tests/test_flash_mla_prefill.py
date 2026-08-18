from types import SimpleNamespace
from unittest.mock import patch

import torch

import nanodeploy.layers.attention as attention


class _FakeFlashMLA:

    calls: list[tuple] = []

    @classmethod
    def get_mla_metadata(
        cls,
        context_lens: torch.Tensor,
        tokens_per_k_head: int,
        num_kv_heads: int,
    ):
        cls.calls.append(
            (
                "metadata",
                tuple(context_lens.tolist()),
                tokens_per_k_head,
                num_kv_heads,
            )
        )
        return (
            torch.empty((1, 1), dtype=torch.int32),
            torch.empty((context_lens.numel() + 1,), dtype=torch.int32),
        )

    @classmethod
    def flash_mla_with_kvcache(
        cls,
        q: torch.Tensor,
        _k_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        v_head_size: int,
        *_args,
    ):
        cls.calls.append(
            (
                "forward",
                tuple(q.shape),
                tuple(block_tables.shape),
                tuple(context_lens.tolist()),
            )
        )
        output = torch.zeros((*q.shape[:-1], v_head_size), dtype=q.dtype)
        return output, torch.empty(0)


def _prefill_context(*, is_dummy: bool):
    return SimpleNamespace(
        is_prefill=True,
        is_dummy=is_dummy,
        block_tables=(
            None
            if is_dummy
            else torch.tensor([[3], [5]], dtype=torch.int32)
        ),
        prefill_cu_seqlens_q_host=(0, 2, 3),
        cu_seqlens_k=torch.tensor([0, 2, 3], dtype=torch.int32),
        slot_mapping=torch.tensor([192, 193, 320], dtype=torch.int32),
        use_sp_a2a=False,
    )


def test_flash_mla_prefill_groups_equal_query_lengths_and_uses_paged_cache():
    context = _prefill_context(is_dummy=False)
    store_calls: list[int] = []
    _FakeFlashMLA.calls.clear()

    with (
        patch.object(attention, "get_context", return_value=context),
        patch.object(
            attention,
            "get_dist_context",
            return_value=SimpleNamespace(attn_sp_rank=0, attn_sp_world_size=1),
        ),
        patch.object(
            attention,
            "store_kcache",
            side_effect=lambda *_args: store_calls.append(_args[-1].numel()),
        ),
        patch.object(attention, "flash_mla", _FakeFlashMLA),
    ):
        implementation = attention.FlashMLAImpl(
            2,
            4,
            num_kv_heads=1,
            v_head_size=3,
        )
        q = torch.randn(3, 2, 4)
        k = torch.randn(3, 1, 4)
        v = torch.randn(3, 1, 3)
        output = implementation.forward(
            q,
            k,
            v,
            torch.empty(8, 64, 1, 4),
            torch.empty(0),
        )

    assert output.shape == (3, 2, 3)
    assert store_calls == [3]
    assert _FakeFlashMLA.calls == [
        ("metadata", (2,), 4, 1),
        ("forward", (1, 2, 2, 4), (1, 1), (2,)),
        ("metadata", (1,), 2, 1),
        ("forward", (1, 1, 2, 4), (1, 1), (1,)),
    ]


def test_flash_mla_prefill_dummy_skips_kv_cache_and_attention_kernel():
    context = _prefill_context(is_dummy=True)
    _FakeFlashMLA.calls.clear()

    with (
        patch.object(attention, "get_context", return_value=context),
        patch.object(
            attention,
            "get_dist_context",
            return_value=SimpleNamespace(attn_sp_rank=0, attn_sp_world_size=1),
        ),
        patch.object(
            attention,
            "store_kcache",
            side_effect=AssertionError("dummy must not write KV cache"),
        ),
        patch.object(attention, "flash_mla", _FakeFlashMLA),
    ):
        implementation = attention.FlashMLAImpl(
            2,
            4,
            num_kv_heads=1,
            v_head_size=3,
        )
        output = implementation.forward(
            torch.randn(1, 2, 4),
            torch.randn(1, 1, 4),
            torch.randn(1, 1, 3),
            torch.empty(8, 64, 1, 4),
            torch.empty(0),
        )

    assert output.shape == (1, 2, 3)
    assert torch.count_nonzero(output).item() == 0
    assert _FakeFlashMLA.calls == []

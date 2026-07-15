from types import SimpleNamespace

import pytest
import torch

from dlengine.models.deepseek_v2.deepseek_v2 import (
    _IndexerTopKState,
    _get_indexer_mode,
)
from dlengine.layers.indexer import _expand_decode_context_lens


GLM52_INDEXER_TYPES = [
    "full" if layer_idx < 3 or (layer_idx - 2) % 4 == 0 else "shared"
    for layer_idx in range(78)
]


def test_glm52_explicit_shared_indexer_schedule():
    config = SimpleNamespace(
        num_hidden_layers=78,
        index_topk=2048,
        indexer_types=GLM52_INDEXER_TYPES,
    )

    modes = [_get_indexer_mode(config, layer_idx) for layer_idx in range(78)]

    assert modes.count("full") == 21
    assert modes.count("shared") == 57
    assert [idx for idx, mode in enumerate(modes) if mode == "full"] == [
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
    ]


def test_glm52_frequency_schedule_matches_explicit_schedule():
    config = SimpleNamespace(
        num_hidden_layers=78,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )

    assert [
        _get_indexer_mode(config, layer_idx) for layer_idx in range(78)
    ] == GLM52_INDEXER_TYPES


def test_mtp_layer_always_constructs_full_indexer():
    config = SimpleNamespace(
        num_hidden_layers=78,
        index_topk=2048,
        indexer_types=GLM52_INDEXER_TYPES,
    )

    assert _get_indexer_mode(config, 78) == "full"


def test_shared_indexer_state_reuses_exact_logical_and_physical_indices():
    logical = torch.tensor([[7, 3], [11, 5]], dtype=torch.int32)
    physical = torch.tensor([[71, 31], [111, 51]], dtype=torch.int32)
    state = _IndexerTopKState()

    state.publish(2, logical, physical)
    reused_logical, reused_physical = state.require(
        layer_idx=3,
        num_tokens=2,
        topk=2,
        require_physical=True,
    )

    assert reused_logical is logical
    assert reused_physical is physical
    assert state.source_layer == 2


def test_shared_indexer_state_rejects_missing_or_stale_topk():
    state = _IndexerTopKState()
    with pytest.raises(RuntimeError, match="no TopK"):
        state.require(3, num_tokens=2, topk=2, require_physical=False)

    state.publish(2, torch.zeros((1, 2), dtype=torch.int32))
    with pytest.raises(RuntimeError, match="stale TopK shape"):
        state.require(3, num_tokens=2, topk=2, require_physical=False)
    with pytest.raises(RuntimeError, match="invalid physical TopK"):
        state.require(3, num_tokens=1, topk=2, require_physical=True)


def test_invalid_explicit_schedule_fails_fast():
    short_config = SimpleNamespace(
        num_hidden_layers=2,
        index_topk=2048,
        indexer_types=["full"],
    )
    with pytest.raises(ValueError, match="one entry per backbone layer"):
        _get_indexer_mode(short_config, 0)

    invalid_config = SimpleNamespace(
        num_hidden_layers=1,
        index_topk=2048,
        indexer_types=["unknown"],
    )
    with pytest.raises(ValueError, match="Unsupported indexer_types"):
        _get_indexer_mode(invalid_config, 0)


def test_multi_token_decode_context_lens_match_deep_gemm_query_layout():
    final_lens = torch.tensor([4098, 513], dtype=torch.int64)

    expanded = _expand_decode_context_lens(final_lens, next_n=2)

    assert expanded.dtype == torch.int32
    assert torch.equal(
        expanded,
        torch.tensor([[4097, 4098], [512, 513]], dtype=torch.int32),
    )


def test_dummy_decode_context_lens_do_not_underflow():
    expanded = _expand_decode_context_lens(
        torch.tensor([1], dtype=torch.int32), next_n=8
    )

    assert expanded.shape == (1, 8)
    assert torch.equal(expanded, torch.ones_like(expanded))

from types import SimpleNamespace

import dlengine.models.pp_utils as pp_utils
import pytest
from torch import nn


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (0, 4)),
        (1, (4, 7)),
        (2, (7, 10)),
    ],
)
def test_get_pp_layer_range_balances_remainder(monkeypatch, rank, expected):
    context = SimpleNamespace(pp_world_size=3, pp_rank=rank)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)

    assert pp_utils.get_pp_layer_range(10) == expected


def test_make_pp_layers_preserves_global_indices(monkeypatch):
    context = SimpleNamespace(pp_world_size=3, pp_rank=1)
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)

    start, end, layers = pp_utils.make_pp_layers(10, lambda _: nn.Identity())

    assert (start, end) == (4, 7)
    assert len(layers) == 10
    assert all(isinstance(layers[idx], nn.Identity) for idx in range(start, end))
    assert all(
        isinstance(layer, pp_utils.PPMissingLayer)
        for idx, layer in enumerate(layers)
        if idx < start or idx >= end
    )


@pytest.mark.parametrize(
    ("is_first", "is_last", "weight_name", "expected"),
    [
        (True, False, "model.embed_tokens.weight", True),
        (False, False, "model.embed_tokens.weight", False),
        (False, True, "model.norm.weight", True),
        (False, False, "model.norm.weight", False),
        (False, True, "lm_head.weight", True),
        (False, False, "lm_head.weight", False),
        (False, False, "model.layers.4.mlp.weight", True),
        (False, False, "model.layers.7.mlp.weight", False),
    ],
)
def test_pp_weight_ownership(monkeypatch, is_first, is_last, weight_name, expected):
    context = SimpleNamespace(
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )
    monkeypatch.setattr(pp_utils, "get_dist_context", lambda: context)

    assert pp_utils.pp_weight_belongs_to_stage(weight_name, 4, 7) is expected


def _gemma_config():
    return SimpleNamespace(
        num_hidden_layers=12,
        num_kv_shared_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 6,
    )


def test_gemma4_pp_keeps_shared_kv_sources_on_last_stage(monkeypatch):
    config = _gemma_config()
    assert pp_utils.get_gemma4_shared_kv_source_start(config) == 6

    ranges = []
    for rank in range(4):
        monkeypatch.setattr(
            pp_utils,
            "get_dist_context",
            lambda rank=rank: SimpleNamespace(pp_world_size=4, pp_rank=rank),
        )
        ranges.append(pp_utils.get_gemma4_pp_layer_range(config))

    assert ranges == [(0, 2), (2, 4), (4, 6), (6, 12)]


def test_gemma4_without_shared_kv_uses_balanced_split(monkeypatch):
    config = _gemma_config()
    config.num_kv_shared_layers = 0
    monkeypatch.setattr(
        pp_utils,
        "get_dist_context",
        lambda: SimpleNamespace(pp_world_size=4, pp_rank=2),
    )

    assert pp_utils.get_gemma4_pp_layer_range(config) == (6, 9)

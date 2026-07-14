from types import SimpleNamespace

import pytest
from dlengine.models import pp_utils
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

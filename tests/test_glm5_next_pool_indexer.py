import torch

from dlengine.runtime.layers.backends.dsa.indexer import (
    append_pool_tail,
    pool_indexer_states,
    pool_indexer_topk,
)


def test_glm_pool_indexer_starts_at_first_valid_token_and_keeps_tail():
    # The first two entries model padding in the first physical page.  The
    # reference starts pooling at token 2, not at physical slot 0.
    d, k = 2, 4
    keys = torch.arange(8 * d, dtype=torch.float32).view(1, 8, d)
    gates = torch.zeros_like(keys)
    valid = torch.tensor([[[0], [0], [1], [1], [1], [1], [1], [1]]], dtype=torch.float32)
    packed = torch.cat([keys, gates, valid], dim=-1)
    pool_keys, pool_indices, pool_valid = pool_indexer_states(
        packed, k, torch.zeros(k, d)
    )
    assert torch.equal(pool_indices, torch.tensor([[[2, 3, 4, 5]]]))
    assert torch.equal(pool_valid, torch.tensor([[True]]))
    expected_pool = keys[0, 2:6].mean(0)
    assert torch.allclose(pool_keys[0, 0], expected_pool)

    visible = torch.tensor(
        [[[0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]]],
        dtype=torch.bool,
    )
    tail = append_pool_tail(
        torch.tensor([[[2, 3, 4, 5], [2, 3, 4, 5]]]),
        visible,
        valid.bool().squeeze(-1),
        k,
    )
    assert torch.equal(tail[0, 0], torch.tensor([2, 3, 4, 5, 6, 7, -1]))


def test_glm_pool_indexer_topk_matches_reference_shape_and_causality():
    # Two complete pools, with the second pool visible only to the second query.
    d, heads, k = 2, 2, 2
    keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    gates = torch.zeros_like(keys)
    valid = torch.ones(1, 4, 1)
    packed = torch.cat([keys, gates, valid], dim=-1)
    query = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]
    )
    weights = torch.ones(1, 2, heads)
    visible = torch.tensor(
        [[[1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.bool
    )
    out = pool_indexer_topk(
        query, weights, packed, visible, index_topk=2, index_kpool=k,
        index_kpool_compress_ape=torch.zeros(k, d), always_select_tail=True,
    )
    # Width is index_topk + (k - 1); first row cannot see pool 1, second row
    # selects the second pool and has no incomplete tail.
    assert out.shape == (1, 2, 3)
    assert torch.equal(out[0, 0], torch.tensor([0, 1, -1]))
    assert torch.equal(out[0, 1], torch.tensor([2, 3, -1]))


def test_glm_pool_parameter_shapes_are_config_driven():
    # The shipped checkpoint uses index_kpool=4; the reference class default
    # is 16, so the runtime must not bake either value into the parameter shape.
    packed = torch.zeros(1, 4, 2 * 3 + 1)
    packed[..., -1] = 1
    ape = torch.zeros(2, 3)
    pools, indices, valid = pool_indexer_states(packed, 2, ape)
    assert pools.shape == (1, 2, 3)
    assert indices.shape == (1, 2, 2)
    assert valid.shape == (1, 2)

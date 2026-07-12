from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Indexer transform kernels require CUDA"
)


def test_indexer_layer_norm_bf16_matches_torch():
    from dlengine.kernel.triton.generic.indexer_transform import indexer_layer_norm_bf16

    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    bias = torch.randn(128, dtype=torch.float32, device="cuda")
    expected = F.layer_norm(x.float(), (128,), weight, bias, 1e-5).to(torch.bfloat16)

    actual = indexer_layer_norm_bf16(x, weight, bias, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_indexer_qk_rope_inplace_matches_torch():
    from dlengine.kernel.triton.generic.indexer_transform import indexer_qk_rope_inplace
    from dlengine.layers.rotary_embedding import RotaryEmbedding

    torch.manual_seed(1)
    tokens, heads, head_dim, rope_dim = 4, 8, 128, 64
    query = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(tokens, head_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([0, 1, 17, 255], dtype=torch.int64, device="cuda")
    rope = RotaryEmbedding(rope_dim, rope_dim, 512, 10000).cuda()

    expected_query = query.clone()
    expected_key = key.clone()
    q_half = (
        expected_query[..., :rope_dim]
        .unflatten(-1, (-1, 2))
        .transpose(-1, -2)
        .contiguous()
        .flatten(-2)
    )
    k_half = (
        expected_key[..., :rope_dim]
        .unsqueeze(1)
        .unflatten(-1, (-1, 2))
        .transpose(-1, -2)
        .contiguous()
        .flatten(-2)
    )
    q_half, k_half = rope(positions, q_half, k_half)
    expected_query[..., :rope_dim] = q_half
    expected_key[..., :rope_dim] = k_half.squeeze(1)

    indexer_qk_rope_inplace(
        query,
        key,
        positions,
        rope.cos_sin_cache,
        rope_dim,
    )
    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)

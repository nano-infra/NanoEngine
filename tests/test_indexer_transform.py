from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Indexer transform kernels require CUDA"
)


def test_indexer_layer_norm_bf16_matches_torch():
    from dlengine.runtime.kernel.triton.generic.indexer_transform import (
        indexer_layer_norm_bf16,
    )

    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    bias = torch.randn(128, dtype=torch.float32, device="cuda")
    expected = F.layer_norm(x.float(), (128,), weight, bias, 1e-5).to(torch.bfloat16)

    actual = indexer_layer_norm_bf16(x, weight, bias, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_indexer_qk_rope_inplace_matches_torch():
    from dlengine.runtime.kernel.triton.generic.indexer_transform import (
        indexer_qk_rope_inplace,
    )
    from dlengine.runtime.layers.rotary_embedding import RotaryEmbedding

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


def test_indexer_q_rope_hadamard_quant_matches_reference():
    pytest.importorskip("tvm_ffi")

    from dlengine.runtime.kernel.jit.sgl.deepseek_v4 import (
        indexer_q_rope_hadamard_quant,
    )
    from dlengine.runtime.layers.rotary_embedding import RotaryEmbedding
    from fast_hadamard_transform import hadamard_transform

    torch.manual_seed(2)
    tokens, heads, head_dim, rope_dim = 4, 8, 128, 64
    query = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(tokens, heads, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([0, 3, 29, 511], dtype=torch.int64, device="cuda")
    rope = RotaryEmbedding(rope_dim, rope_dim, 1024, 10000).cuda()
    weight_scale = 0.03125

    reference = query.clone()
    q_half = (
        reference[..., :rope_dim]
        .unflatten(-1, (-1, 2))
        .transpose(-1, -2)
        .contiguous()
        .flatten(-2)
    )
    q_half, _ = rope(positions, q_half, q_half)
    reference[..., :rope_dim] = q_half
    reference = hadamard_transform(reference.contiguous(), scale=head_dim**-0.5)
    flat = reference.view(-1, head_dim)
    amax = flat.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    expected_fp8 = (flat.float() / scale).to(torch.float8_e4m3fn).view_as(query)
    expected_weights = (
        gate.float().unsqueeze(-1) * weight_scale * scale.view(tokens, heads, 1)
    )

    actual_fp8, actual_weights = indexer_q_rope_hadamard_quant(
        query,
        gate,
        weight_scale,
        rope.cos_sin_cache,
        positions,
    )
    assert torch.equal(actual_fp8.view(torch.uint8), expected_fp8.view(torch.uint8))
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_indexer_k_transform_store_matches_reference():
    from dlengine.runtime.kernel.triton.generic.indexer_transform import (
        indexer_k_transform_store_fp8,
    )
    from dlengine.runtime.layers.rotary_embedding import RotaryEmbedding
    from fast_hadamard_transform import hadamard_transform

    torch.manual_seed(3)
    tokens, head_dim, rope_dim, page_size = 4, 128, 64, 64
    key = torch.randn(tokens, head_dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(head_dim, dtype=torch.float32, device="cuda")
    bias = torch.randn(head_dim, dtype=torch.float32, device="cuda")
    positions = torch.tensor([0, 5, 31, 511], dtype=torch.int64, device="cuda")
    slots = torch.tensor([1, 65, 7, 130], dtype=torch.int32, device="cuda")
    rope = RotaryEmbedding(rope_dim, rope_dim, 1024, 10000).cuda()
    cache = torch.zeros(3, page_size * 132, dtype=torch.uint8, device="cuda")

    expected = F.layer_norm(key.float(), (head_dim,), weight, bias, 1e-5).to(
        torch.bfloat16
    )
    half = (
        expected[:, :rope_dim]
        .unsqueeze(1)
        .unflatten(-1, (-1, 2))
        .transpose(-1, -2)
        .contiguous()
        .flatten(-2)
    )
    _, half = rope(positions, half, half)
    expected[:, :rope_dim] = half.squeeze(1)
    expected = hadamard_transform(expected.contiguous(), scale=head_dim**-0.5)
    scale = torch.exp2(
        torch.ceil(
            torch.log2(
                expected.abs().float().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0
            )
        )
    )
    expected_fp8 = (expected.float() / scale).to(torch.float8_e4m3fn)

    indexer_k_transform_store_fp8(
        key,
        weight,
        bias,
        1e-5,
        positions,
        rope.cos_sin_cache,
        cache,
        slots,
        page_size,
    )
    for token, slot in enumerate(slots.tolist()):
        page, offset = divmod(slot, page_size)
        base = page * page_size * 132
        fp8_offset = base + offset * head_dim
        scale_offset = base + page_size * head_dim + offset * 4
        assert torch.equal(
            cache.view(-1)[fp8_offset : fp8_offset + head_dim],
            expected_fp8[token].view(torch.uint8),
        )
        assert torch.equal(
            cache.view(-1)[scale_offset : scale_offset + 4],
            scale[token].view(torch.uint8),
        )

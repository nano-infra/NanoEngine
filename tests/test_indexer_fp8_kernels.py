from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Indexer FP8 kernels require CUDA"
)


def _reference_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def test_indexer_query_quant_matches_reference():
    from dlengine.kernel.triton.hopper.block_gemm_fp8 import quant_fp8

    torch.manual_seed(0)
    query = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    query[0].zero_()

    actual_fp8, actual_scale = quant_fp8(
        query,
        128,
        round_ue8m0=True,
        min_absmax=1e-4,
    )
    expected_fp8, expected_scale = _reference_quant(query)

    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    assert torch.equal(actual_fp8.view(torch.uint8), expected_fp8.view(torch.uint8))


def test_indexer_key_store_matches_split_page_layout():
    from dlengine.kernel.triton.generic.fp8_ue8m0_quant import (
        store_indexer_key_fp8_fused,
    )

    torch.manual_seed(1)
    page_size = 64
    head_dim = 128
    bytes_per_token = head_dim + 4
    num_pages = 3
    keys = torch.randn(3, head_dim, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([5, 70, 130], dtype=torch.int32, device="cuda")
    cache = torch.zeros(
        num_pages,
        page_size * bytes_per_token,
        dtype=torch.uint8,
        device="cuda",
    )

    store_indexer_key_fp8_fused(keys, cache, slots, page_size)

    expected = torch.zeros_like(cache)
    expected_flat = expected.view(-1)
    expected_fp8, expected_scale = _reference_quant(keys)
    for token, slot_tensor in enumerate(slots):
        slot = int(slot_tensor)
        page = slot // page_size
        offset = slot % page_size
        page_base = page * page_size * bytes_per_token
        fp8_offset = page_base + offset * head_dim
        scale_offset = page_base + page_size * head_dim + offset * 4
        expected_flat[fp8_offset : fp8_offset + head_dim] = expected_fp8[token].view(
            torch.uint8
        )
        expected_flat[scale_offset : scale_offset + 4] = expected_scale[token].view(
            torch.uint8
        )

    assert torch.equal(cache, expected)

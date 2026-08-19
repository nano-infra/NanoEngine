import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_dense_causal_prefill_matches_torch():
    import flash_mla

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    query_len = context_len = 15
    num_heads = 128
    num_kv_heads = 1
    head_dim = 576
    value_dim = 512
    block_size = 64
    scale = 1 / math.sqrt(192)

    query = (
        torch.randn(
            1,
            query_len,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    keys = (
        torch.randn(
            context_len,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    key_cache = torch.full(
        (1, block_size, num_kv_heads, head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache[0, :context_len].copy_(keys)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    context_lens = torch.tensor(
        [context_len], dtype=torch.int32, device=device
    )
    metadata, num_splits = flash_mla.get_mla_metadata(
        context_lens,
        query_len * num_heads // num_kv_heads,
        num_kv_heads,
    )

    actual, _ = flash_mla.flash_mla_with_kvcache(
        query,
        key_cache,
        block_table,
        context_lens,
        value_dim,
        metadata,
        num_splits,
        scale,
        True,
    )

    query_by_head = query[0].float().transpose(0, 1)
    key = keys[:, 0].float()
    scores = torch.matmul(query_by_head, key.transpose(0, 1)) * scale
    causal_mask = torch.ones(
        query_len, context_len, dtype=torch.bool, device=device
    ).triu(1)
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.matmul(probabilities, key[:, :value_dim])
    expected = expected.transpose(0, 1).unsqueeze(0)

    torch.testing.assert_close(
        actual.float(), expected, rtol=0.025, atol=0.002
    )

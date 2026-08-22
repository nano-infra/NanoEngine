"""GPU correctness tests for NSA/DSA prefill index selection."""

import types

import pytest
import torch
import torch.nn as nn


class _FixedWeights(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value


@pytest.mark.parametrize("interleaved", [False, True])
def test_indexer_key_rope_layout_follows_model_config(monkeypatch, interleaved):
    import dlengine.runtime.layers.indexer as indexer_module
    from dlengine.runtime.layers.indexer import _interleaved_to_half, Indexer

    class CaptureRope(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_key = None

        def forward(self, positions, query, key):
            self.seen_key = key.clone()
            return query, key

    indexer = Indexer.__new__(Indexer)
    nn.Module.__init__(indexer)
    indexer.rope_head_dim = 4
    indexer.indexer_rope_interleave = interleaved
    indexer.wk = nn.Identity()
    indexer.k_norm = nn.Identity()
    indexer.rotary_emb = CaptureRope()
    monkeypatch.setattr(indexer_module, "_hadamard_rotate", lambda value: value)

    hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    actual = indexer._compute_key(hidden_states, torch.tensor([0]))
    expected_rope = hidden_states[:, :4].unsqueeze(1)
    if interleaved:
        expected_rope = _interleaved_to_half(expected_rope)

    assert torch.equal(indexer.rotary_emb.seen_key, expected_rope)
    assert torch.equal(actual[:, :4], expected_rope.squeeze(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_weighted_relu_mqa_scores_matches_dense_reference():
    from dlengine.runtime.layers.indexer import _weighted_relu_mqa_scores

    torch.manual_seed(17)
    query = torch.randn(7, 6, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(7, 6, device="cuda", dtype=torch.float32)
    key = torch.randn(11, 128, device="cuda", dtype=torch.bfloat16)

    actual = _weighted_relu_mqa_scores(query, weights, key, head_chunk=2)
    per_head = torch.einsum("qhd,kd->qhk", query.float(), key.float())
    expected = (per_head.relu() * weights[:, :, None]).sum(dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_weighted_relu_cannot_be_replaced_by_folded_query():
    from dlengine.runtime.layers.indexer import _weighted_relu_mqa_scores

    query = torch.tensor([[[2.0], [-1.0]]], device="cuda")
    weights = torch.tensor([[1.0, 3.0]], device="cuda")
    key = torch.tensor([[10.0], [-1.0]], device="cuda")

    correct = _weighted_relu_mqa_scores(query, weights, key, head_chunk=1)
    folded = torch.einsum("qhd,qh->qd", query, weights) @ key.T

    assert correct.argmax(dim=-1).item() == 0
    assert folded.argmax(dim=-1).item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_aware_prefill_topk_matches_ragged_dense_reference():
    from dlengine.runtime.layers.indexer import Indexer, IndexerCache

    torch.manual_seed(23)
    device = torch.device("cuda")
    head_dim = 128
    num_heads = 4
    page_size = 4
    topk = 3

    indexer = Indexer.__new__(Indexer)
    nn.Module.__init__(indexer)
    indexer.n_heads = num_heads
    indexer.head_dim = head_dim
    indexer.index_topk = topk
    indexer.layer_id = 0
    indexer.indexer_cache = IndexerCache(
        num_layers=1,
        num_pages=6,
        page_size=page_size,
        head_dim=head_dim,
        device="cuda",
    )

    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    cu_cached = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cached_lens = torch.tensor([3, 2], dtype=torch.int64, device=device)
    block_table = torch.tensor([[3, 1], [4, 0]], dtype=torch.int32, device=device)

    cached_source = torch.randn(5, head_dim, dtype=torch.bfloat16, device=device)
    cached_slots = torch.tensor([12, 13, 14, 16, 17], dtype=torch.int32, device=device)
    indexer.indexer_cache.store_key_fp8(0, cached_source, cached_slots)
    cached_keys = indexer._gather_cached_prefix_keys(
        block_table, cached_lens, cu_cached, dtype=torch.float32
    )

    query = torch.randn(5, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    fresh_keys = torch.randn(5, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(5, num_heads, dtype=torch.bfloat16, device=device)
    indexer.weights_proj = _FixedWeights(weights)

    def fake_compute_q_k(self, q_lora, hidden_states, positions):
        return query, fresh_keys

    indexer._compute_q_k = types.MethodType(fake_compute_q_k, indexer)
    dummy = torch.empty(5, 1, dtype=torch.bfloat16, device=device)
    actual = indexer.compute_prefill_topk_cache_aware(
        dummy,
        dummy,
        torch.arange(5, device=device),
        cu_q,
        cu_k,
        block_table,
        query_chunk=2,
    )

    expected = torch.full_like(actual, -1)
    scaled_weights = weights.float() * (num_heads**-0.5)
    cached_offset = 0
    for seq_id, (q_len, cached_len) in enumerate(((2, 3), (3, 2))):
        q_start = int(cu_q[seq_id].item())
        k_start = int(cu_k[seq_id].item())
        seq_keys = torch.cat(
            [
                cached_keys[cached_offset : cached_offset + cached_len],
                fresh_keys[q_start : q_start + q_len].float(),
            ]
        )
        for local_q in range(q_len):
            per_head = torch.einsum(
                "hd,kd->hk", query[q_start + local_q].float(), seq_keys
            )
            scores = (per_head.relu() * scaled_weights[q_start + local_q, :, None]).sum(
                dim=0
            )
            visible = cached_len + local_q + 1
            k = min(topk, visible)
            selected = scores[:visible].topk(k).indices.to(torch.int32)
            expected[q_start + local_q, :k] = selected + k_start
        cached_offset += cached_len

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_prefill_topk_uses_per_query_causal_context(monkeypatch):
    import dlengine.runtime.layers.indexer as indexer_module
    from dlengine.runtime.layers.indexer import Indexer, IndexerCache

    device = torch.device("cuda")
    indexer = Indexer.__new__(Indexer)
    nn.Module.__init__(indexer)
    indexer.n_heads = 2
    indexer.head_dim = 128
    indexer.index_topk = 3
    indexer.layer_id = 0
    indexer.softmax_scale = 1.0
    indexer.indexer_cache = IndexerCache(
        num_layers=1, num_pages=6, page_size=4, head_dim=128, device="cuda"
    )

    query = torch.ones(5, 2, 128, dtype=torch.bfloat16, device=device)

    def fake_compute_q_k(self, q_lora, hidden_states, positions, **kwargs):
        return query, torch.empty(0, device=device)

    indexer._compute_q_k = types.MethodType(fake_compute_q_k, indexer)
    indexer._compute_gate_weights = types.MethodType(
        lambda self, hidden_states, q_scale: torch.ones(
            hidden_states.shape[0], self.n_heads, device=device
        ),
        indexer,
    )
    indexer.build_schedule_metadata = types.MethodType(
        lambda self, context_lens: torch.empty(0, device=device),
        indexer,
    )

    monkeypatch.setattr(indexer_module, "fused_kernels_enabled", lambda: False)
    monkeypatch.setattr(
        indexer_module,
        "quant_fp8",
        lambda value, *args, **kwargs: (
            value,
            torch.ones(value.shape[0], 1, device=device),
        ),
    )
    seen_context_lens = []

    def fake_paged_logits(
        tiled_q,
        kv_cache,
        weights,
        context_lens,
        page_tables,
        schedule,
        max_context_len,
        **kwargs,
    ):
        seen_context_lens.append(context_lens.flatten().tolist())
        return (
            torch.arange(max_context_len, dtype=torch.float32, device=device)
            .expand(tiled_q.shape[0], -1)
            .clone()
        )

    monkeypatch.setattr(
        indexer_module.deep_gemm, "fp8_paged_mqa_logits", fake_paged_logits
    )

    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    block_table = torch.tensor([[3, 1], [4, 0]], dtype=torch.int32, device=device)
    dummy = torch.empty(5, 1, dtype=torch.bfloat16, device=device)
    actual = indexer.compute_prefill_topk_paged(
        dummy,
        dummy,
        torch.arange(5, device=device),
        cu_q,
        cu_k,
        block_table,
        query_chunk=2,
    )

    expected = torch.tensor(
        [[3, 2, 1], [4, 3, 2], [7, 6, 5], [8, 7, 6], [9, 8, 7]],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(actual, expected)
    assert seen_context_lens == [[4, 5], [3, 4], [5]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_prefill_long_context_preserves_exact_topk(monkeypatch):
    """Paged scoring must preserve exact causal TopK beyond 64K."""
    import dlengine.runtime.layers.indexer as indexer_module
    from dlengine.runtime.layers.indexer import Indexer, IndexerCache

    device = torch.device("cuda")
    indexer = Indexer.__new__(Indexer)
    nn.Module.__init__(indexer)
    indexer.n_heads = 2
    indexer.head_dim = 128
    indexer.index_topk = 512
    indexer.layer_id = 0
    indexer.softmax_scale = 1.0
    indexer.indexer_cache = IndexerCache(
        num_layers=1, num_pages=1, page_size=64, head_dim=128, device="cuda"
    )

    query = torch.ones(2, 2, 128, dtype=torch.bfloat16, device=device)
    indexer._compute_q_k = types.MethodType(
        lambda self, *args, **kwargs: (query, torch.empty(0, device=device)),
        indexer,
    )
    indexer._compute_gate_weights = types.MethodType(
        lambda self, hidden_states, q_scale: torch.ones(
            hidden_states.shape[0], self.n_heads, device=device
        ),
        indexer,
    )
    indexer.build_schedule_metadata = types.MethodType(
        lambda self, context_lens: torch.empty(0, device=device), indexer
    )
    monkeypatch.setattr(indexer_module, "fused_kernels_enabled", lambda: False)
    monkeypatch.setattr(
        indexer_module,
        "quant_fp8",
        lambda value, *args, **kwargs: (
            value,
            torch.ones(value.shape[0], 1, device=device),
        ),
    )
    monkeypatch.setattr(
        indexer_module.deep_gemm,
        "fp8_paged_mqa_logits",
        lambda tiled_q, kv_cache, weights, context_lens, page_tables, schedule, max_context_len, **kwargs: torch.arange(
            max_context_len, dtype=torch.float32, device=device
        )
        .expand(tiled_q.shape[0], -1)
        .clone(),
    )

    cu_q = torch.tensor([0, 2], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 65538], dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1025), dtype=torch.int32, device=device)
    dummy = torch.empty(2, 1, dtype=torch.bfloat16, device=device)
    actual = indexer.compute_prefill_topk_paged(
        dummy,
        dummy,
        torch.arange(2, device=device),
        cu_q,
        cu_k,
        block_table,
        query_chunk=2,
    )

    assert actual.shape == (2, 512)
    assert actual[0].max().item() == 65536
    assert actual[0].min().item() == 65536 - 511
    assert actual[1].max().item() == 65537
    assert actual[1].min().item() == 65537 - 511

import dlengine.runtime.layers as layers
import pytest
import torch
from dlengine.runtime.context.batch import reset_batch_context, set_batch_context
from dlengine.runtime.layers.backends.attention import fa4 as attention
from dlengine.runtime.layers.backends.mla import trtllm as mla_trtllm


def teardown_function():
    reset_batch_context()
    layers.reset_backend()
    attention._trtllm_workspace = None
    mla_trtllm._trtllm_workspace = None


def test_blackwell_auto_selects_blackwell_backend(monkeypatch):
    monkeypatch.delenv("NANO_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    layers.init_backend(quant_config=object())

    assert layers.get_backend().hardware_backend == "blackwell"


def test_blackwell_decode_uses_trtllm_paged_kv(monkeypatch):
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(1)
        return kwargs["out"]

    monkeypatch.setattr(attention, "_trtllm_decode_func", fake_decode)
    attention._trtllm_workspace = torch.zeros(1, dtype=torch.uint8)
    impl = attention.Fa4AttentionImpl(8, 128, 0.125, 2)
    q = torch.zeros(2, 8, 128)
    k = torch.empty(2, 2, 128)
    v = torch.empty_like(k)
    k_cache = torch.zeros(2, 64, 2, 128)
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.tensor([[[0], [1]]], dtype=torch.int32)
    context_lens = torch.tensor([[4, 7]], dtype=torch.int32)
    set_batch_context(
        is_prefill=False,
        context_lens=context_lens,
        block_tables=page_table,
        num_tokens_per_seq=1,
    )

    output = impl.forward(q, k, v, k_cache, v_cache, write_kv_cache=False)

    assert output.shape == q.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["query"].shape == (2, 8, 128)
    assert call["kv_cache"][0].shape == (2, 2, 64, 128)
    assert call["block_tables"].data_ptr() == page_table[0].data_ptr()
    assert call["seq_lens"].data_ptr() == context_lens[0].data_ptr()
    assert call["max_seq_len"] == 64
    assert call["backend"] == "auto"
    assert call["out"].data_ptr() == output.data_ptr()


def test_blackwell_trtllm_missing_fails_without_fallback(monkeypatch):
    import_error = ImportError("flashinfer TRTLLM-GEN is unavailable")
    monkeypatch.setattr(attention, "_trtllm_decode_func", None)
    monkeypatch.setattr(attention, "_TRTLLM_IMPORT_ERROR", import_error)

    try:
        attention.Fa4AttentionImpl(8, 128, 0.125, 2)
    except RuntimeError as error:
        assert "No naive or SDPA fallback" in str(error)
        assert error.__cause__ is import_error
    else:
        raise AssertionError("missing TRT-LLM kernel must fail during construction")


def test_blackwell_fa4_missing_fails_without_fallback(monkeypatch):
    import_error = ImportError("flash_attn.cute is unavailable")
    monkeypatch.setattr(attention, "_fa4_varlen_func", None)
    monkeypatch.setattr(attention, "_FA4_IMPORT_ERROR", import_error)

    try:
        attention.Fa4AttentionImpl(8, 128, 0.125, 2)
    except RuntimeError as error:
        assert "No naive or SDPA fallback" in str(error)
        assert error.__cause__ is import_error
    else:
        raise AssertionError("missing FA4 must fail during attention construction")


def test_blackwell_prefill_packs_strided_qkv_views(monkeypatch):
    calls = []

    def fake_fa4(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "_fa4_varlen_func", fake_fa4)
    impl = attention.Fa4AttentionImpl(8, 128, 0.125, 2)
    q = torch.empty(3, 8, 129)[..., :128]
    k = torch.empty(3, 2, 129)[..., :128]
    v = torch.empty(3, 2, 129)[..., :128]
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    set_batch_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_k=3,
    )

    output = impl.forward(
        q,
        k,
        v,
        torch.empty(0),
        torch.empty(0),
        write_kv_cache=False,
    )

    assert output.shape == q.shape
    assert len(calls) == 1
    assert all(tensor.is_contiguous() for tensor in calls[0][:3])


def _make_blackwell_mla(
    monkeypatch, calls, *, num_heads=8, v_head_dim=512, mla_kv_lora_rank=None
):
    def fake_decode(**kwargs):
        calls.append(kwargs)
        assert kwargs["out"].shape == (
            *kwargs["query"].shape[:-1],
            kwargs["kv_lora_rank"],
        )
        kwargs["out"].fill_(1)
        return kwargs["out"]

    monkeypatch.setattr(mla_trtllm, "_trtllm_mla_decode_func", fake_decode)
    monkeypatch.setattr(
        mla_trtllm,
        "_get_trtllm_workspace",
        lambda device: torch.zeros(1, dtype=torch.uint8, device=device),
    )
    impl = mla_trtllm.TrtllmMlaAttention(
        num_heads=num_heads,
        head_dim=576,
        scale=0.125,
        num_kv_heads=1,
        v_head_dim=v_head_dim,
        mla_qk_nope_head_dim=128,
        mla_kv_lora_rank=mla_kv_lora_rank,
    )
    impl.k_cache = torch.zeros(4, 64, 1, 576, dtype=torch.float8_e4m3fn)
    return impl


@pytest.mark.parametrize("v_head_dim", [128, 256])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens_per_seq", [1, 3])
def test_blackwell_mla_returns_compressed_values(
    monkeypatch, v_head_dim, cache_dtype, tokens_per_seq
):
    calls = []
    # Kimi-K3 at attention_tp=8 has 12 local heads, rank 512 and value width 128.
    impl = _make_blackwell_mla(
        monkeypatch, calls, num_heads=12, v_head_dim=v_head_dim, mla_kv_lora_rank=512
    )
    impl.k_cache = impl.k_cache.to(cache_dtype)
    batch_size = 16
    num_tokens = batch_size * tokens_per_seq
    q = torch.zeros(num_tokens, 12, 576, dtype=torch.bfloat16)
    k = torch.zeros(num_tokens, 1, 576, dtype=torch.bfloat16)
    set_batch_context(
        is_prefill=False,
        context_lens=torch.full((1, batch_size), 7, dtype=torch.int32),
        block_tables=torch.zeros(1, batch_size, 1, dtype=torch.int32),
        num_tokens_per_seq=tokens_per_seq,
    )

    output = impl.forward(q, k, torch.empty(0), write_kv_cache=False)

    assert output.shape == (num_tokens, 12, 512)
    assert output.dtype == torch.bfloat16
    assert output.data_ptr() == calls[0]["out"].data_ptr()
    # W_UV is applied by the model after attention; it consumes the KV rank.
    value_weight = torch.ones(12, 512, v_head_dim, dtype=output.dtype)
    projected = torch.einsum("thr,hrv->thv", output, value_weight)
    torch.testing.assert_close(
        projected, torch.full((num_tokens, 12, v_head_dim), 512, dtype=output.dtype)
    )


def test_blackwell_raw_fp8_mla_uses_page_size_axis(monkeypatch):
    calls = []
    impl = _make_blackwell_mla(monkeypatch, calls)
    q = torch.zeros(2, 8, 576, dtype=torch.bfloat16)
    k = torch.zeros(2, 1, 576, dtype=torch.bfloat16)
    page_table = torch.tensor([[[0], [1]]], dtype=torch.int32)
    context_lens = torch.tensor([[4, 7]], dtype=torch.int32)
    set_batch_context(
        is_prefill=False,
        context_lens=context_lens,
        block_tables=page_table,
        num_tokens_per_seq=1,
    )

    output = impl.forward(q, k, torch.empty(0), write_kv_cache=False)

    assert output.shape == (2, 8, 512)
    call = calls[0]
    assert call["query"].dtype == torch.float8_e4m3fn
    assert call["kv_cache"].shape == (4, 1, 64, 576)
    assert call["block_tables"].shape == (2, 2)
    assert call["sparse_mla_top_k"] == 0
    assert call["max_seq_len"] == 64
    assert call["backend"] == "trtllm-gen"


def test_blackwell_raw_fp8_sparse_mla_uses_sparse_page_table(monkeypatch):
    calls = []
    impl = _make_blackwell_mla(monkeypatch, calls)
    q = torch.zeros(2, 8, 576, dtype=torch.bfloat16)
    k = torch.zeros(2, 1, 576, dtype=torch.bfloat16)
    sparse_indices = torch.tensor([[3, 1, 0], [2, 1, 0]], dtype=torch.int64)
    set_batch_context(
        is_prefill=False,
        context_lens=torch.tensor([[4, 7]], dtype=torch.int32),
        block_tables=torch.tensor([[[0], [1]]], dtype=torch.int32),
        num_tokens_per_seq=1,
    )

    impl.forward(
        q,
        k,
        torch.empty(0),
        sparse_indices=sparse_indices,
        write_kv_cache=False,
    )

    call = calls[0]
    assert call["block_tables"].shape == (2, 1, 3)
    assert call["block_tables"].dtype == torch.int32
    assert call["sparse_mla_top_k"] == 3
    assert call["max_seq_len"] == 7


def test_blackwell_raw_fp8_sparse_mla_preserves_multi_token_rows(monkeypatch):
    calls = []
    impl = _make_blackwell_mla(monkeypatch, calls)
    batch_size = 2
    tokens_per_seq = 3
    q = torch.zeros(batch_size * tokens_per_seq, 8, 576, dtype=torch.bfloat16)
    k = torch.zeros(batch_size * tokens_per_seq, 1, 576, dtype=torch.bfloat16)
    sparse_indices = torch.arange(24, dtype=torch.int64).reshape(6, 4)
    set_batch_context(
        is_prefill=False,
        context_lens=torch.tensor([[7, 9]], dtype=torch.int32),
        block_tables=torch.tensor([[[0], [1]]], dtype=torch.int32),
        num_tokens_per_seq=tokens_per_seq,
    )

    impl.forward(
        q,
        k,
        torch.empty(0),
        sparse_indices=sparse_indices,
        write_kv_cache=False,
    )

    call = calls[0]
    assert call["block_tables"].shape == (batch_size, tokens_per_seq, 4)
    assert torch.equal(
        call["block_tables"], sparse_indices.to(torch.int32).reshape(2, 3, 4)
    )
    assert call["sparse_mla_top_k"] == 4


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell CUDA",
)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_kimi_k3_mla_decode_matches_reference_and_cuda_graph(cache_dtype):
    if mla_trtllm._trtllm_mla_decode_func is None:
        pytest.skip("FlashInfer TRTLLM-GEN MLA is unavailable")
    torch.manual_seed(42)
    batch_size, num_heads, page_size = 16, 12, 64
    impl = mla_trtllm.TrtllmMlaAttention(
        num_heads=num_heads,
        head_dim=576,
        scale=192**-0.5,
        num_kv_heads=1,
        v_head_dim=128,
        mla_qk_nope_head_dim=128,
        mla_kv_lora_rank=512,
    )
    impl.k_cache = (
        torch.randn(batch_size * 2, page_size, 1, 576, device="cuda") * 0.5
    ).to(cache_dtype)
    q = torch.randn(batch_size, num_heads, 576, device="cuda").bfloat16()
    k = torch.empty(batch_size, 1, 576, dtype=q.dtype, device=q.device)
    seq_lens = torch.tensor(
        [1, 2, 7, 16, 31, 32, 63, 64, 65, 70, 80, 95, 96, 110, 127, 128],
        dtype=torch.int32,
        device=q.device,
    )
    set_batch_context(
        is_prefill=False,
        context_lens=seq_lens.unsqueeze(0),
        block_tables=torch.arange(
            batch_size * 2, dtype=torch.int32, device=q.device
        ).reshape(1, batch_size, 2),
        num_tokens_per_seq=1,
    )

    query = q.to(cache_dtype).float()
    cache = impl.k_cache.float().reshape(batch_size, page_size * 2, 576)
    scores = torch.einsum("bhd,btd->bht", query, cache) * impl.scale
    mask = torch.arange(page_size * 2, device=q.device)[None, :] >= seq_lens[:, None]
    scores.masked_fill_(mask[:, None, :], float("-inf"))
    expected = torch.einsum(
        "bht,btr->bhr", scores.softmax(dim=-1), cache[..., :512]
    ).bfloat16()

    # FP8 attention has additional intermediate rounding beyond quantizing Q/KV.
    # Bound the aggregate error as well as elementwise error near zero.
    fp8_cache = cache_dtype == torch.float8_e4m3fn
    rtol, atol = (5e-2, 4e-2) if fp8_cache else (2e-2, 1e-2)
    actual = impl.forward(q, k, torch.empty(0), write_kv_cache=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error.item() < (3e-2 if fp8_cache else 5e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = impl.forward(q, k, torch.empty(0), write_kv_cache=False)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mla_dummy_decode_has_positive_sequence_bound(monkeypatch, cache_dtype):
    calls = []
    impl = _make_blackwell_mla(monkeypatch, calls, v_head_dim=128, mla_kv_lora_rank=512)
    impl.k_cache = impl.k_cache.to(cache_dtype)
    set_batch_context(
        is_prefill=False,
        is_dummy=True,
        context_lens=torch.ones(1, 1, dtype=torch.int32),
        block_tables=torch.empty(1, 0, 0, dtype=torch.int32),
    )
    q = torch.zeros(1, 8, 576, dtype=torch.bfloat16)
    output = impl(q, torch.empty(1, 1, 576), torch.empty(0))
    assert output.shape == (1, 8, 512)
    assert calls[0]["max_seq_len"] == 64
    assert calls[0]["block_tables"].shape == (1, 2)

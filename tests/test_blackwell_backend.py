import dlengine.layers as layers
import torch
from dlengine.context.batch import reset_batch_context, set_batch_context
from dlengine.layers.blackwell import attention, BlackwellBackendFactory


def teardown_function():
    reset_batch_context()
    layers.reset_backend()
    attention._trtllm_workspace = None


def test_blackwell_auto_selects_blackwell_backend(monkeypatch):
    monkeypatch.delenv("NANO_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    layers.init_backend(quant_config=object())

    assert isinstance(layers.get_backend(), BlackwellBackendFactory)


def test_blackwell_decode_uses_trtllm_paged_kv(monkeypatch):
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(1)
        return kwargs["out"]

    monkeypatch.setattr(attention, "_trtllm_decode_func", fake_decode)
    attention._trtllm_workspace = torch.zeros(1, dtype=torch.uint8)
    impl = attention.BlackwellAttentionImpl(8, 128, 0.125, 2)
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
        attention.BlackwellAttentionImpl(8, 128, 0.125, 2)
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
        attention.BlackwellAttentionImpl(8, 128, 0.125, 2)
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
    impl = attention.BlackwellAttentionImpl(8, 128, 0.125, 2)
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

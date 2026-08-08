import importlib
from types import SimpleNamespace

import pytest
import torch

from nanodeploy.kernels import deep_gemm_backend
from nanodeploy.layers import token_dispatcher
from nanodeploy.layers.deepep_moe import _allocate_column_major_scales


class _Hook:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


class _LowLatencyBuffer:
    def __init__(self):
        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.dispatch_hook = _Hook()
        self.combine_hook = _Hook()

    def low_latency_dispatch(self, hidden, ids, max_tokens, experts, **kwargs):
        self.dispatch_kwargs = (hidden, ids, max_tokens, experts, kwargs)
        packed = (
            torch.empty(2, 8, 16),
            torch.empty_strided((2, 8, 2), (16, 1, 8)),
        )
        return packed, torch.tensor([2, 1], dtype=torch.int32), ("h",), None, self.dispatch_hook

    def low_latency_combine(self, hidden, ids, weights, handle, **kwargs):
        self.combine_kwargs = (hidden, ids, weights, handle, kwargs)
        return torch.empty(1, 16), None, self.combine_hook


class _NormalBuffer:
    def __init__(self):
        self.layout_kwargs = None
        self.dispatch_kwargs = None
        self.combine_kwargs = None

    def get_dispatch_layout(self, ids, experts, **kwargs):
        self.layout_kwargs = (ids, experts, kwargs)
        return "rank", "rdma", "expert", "mask", "layout-event"

    def dispatch(self, x, **kwargs):
        self.dispatch_kwargs = (x, kwargs)
        return x, torch.tensor([[0, -1]]), torch.tensor([[1.0, 0.0]]), [128, 0], ("normal",), None

    def combine(self, hidden, handle, **kwargs):
        self.combine_kwargs = (hidden, handle, kwargs)
        return hidden, None, None


def _context(buffer):
    return SimpleNamespace(
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        ep_size=2,
        max_tokens_per_rank=4,
        topk_idx_dtype=torch.int64,
        get_buffer=lambda: buffer,
        prepare_low_latency=lambda: None,
        mark_normal=lambda: None,
    )


def test_low_latency_dispatch_and_combine_use_target_native_contract(monkeypatch):
    buffer = _LowLatencyBuffer()
    monkeypatch.setattr(token_dispatcher, "get_ep_context", lambda: _context(buffer))
    dispatcher = token_dispatcher.DeepEPTokenDispatcherLowLatency(
        group=object(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
    )
    hidden = torch.empty(1, 16)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.bfloat16)

    packed, converted_ids, converted_weights, masked_m, expected_m = (
        dispatcher.dispatch(hidden, ids, weights)
    )

    assert converted_ids.dtype == torch.int64
    assert converted_weights.dtype == torch.float32
    assert masked_m.dtype == torch.int32
    assert expected_m == 1
    assert packed[1].stride() == (16, 1, 8)
    assert buffer.dispatch_hook.calls == 1
    _, sent_ids, max_tokens, experts, kwargs = buffer.dispatch_kwargs
    assert sent_ids.dtype == torch.int64
    assert (max_tokens, experts) == (4, 4)
    assert kwargs == {
        "use_fp8": True,
        "round_scale": False,
        "use_ue8m0": False,
        "async_finish": False,
        "return_recv_hook": True,
    }

    output = dispatcher.combine(
        torch.empty(2, 8, 16), converted_ids, converted_weights
    )
    assert output.shape == (1, 16)
    assert buffer.combine_hook.calls == 1
    _, sent_ids, sent_weights, handle, kwargs = buffer.combine_kwargs
    assert sent_ids.dtype == torch.int64
    assert sent_weights.dtype == torch.float32
    assert handle == ("h",)
    assert kwargs == {
        "async_finish": False,
        "zero_copy": False,
        "return_recv_hook": True,
    }
    with pytest.raises(RuntimeError, match="without dispatch"):
        dispatcher.combine(torch.empty(2, 8, 16), converted_ids, converted_weights)


def test_low_latency_rejects_overwriting_inflight_handle(monkeypatch):
    buffer = _LowLatencyBuffer()
    monkeypatch.setattr(token_dispatcher, "get_ep_context", lambda: _context(buffer))
    dispatcher = token_dispatcher.DeepEPTokenDispatcherLowLatency(
        group=object(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
    )
    dispatcher.dispatch(
        torch.empty(1, 16),
        torch.tensor([[0, 1]]),
        torch.tensor([[0.5, 0.5]]),
    )
    with pytest.raises(RuntimeError, match="in-flight"):
        dispatcher.dispatch(
            torch.empty(1, 16),
            torch.tensor([[0, 1]]),
            torch.tensor([[0.5, 0.5]]),
        )


def test_normal_dispatch_uses_layout_and_single_handle(monkeypatch):
    buffer = _NormalBuffer()
    context = _context(buffer)
    normal_marks = []
    context.mark_normal = lambda: normal_marks.append(True)
    monkeypatch.setattr(token_dispatcher, "get_ep_context", lambda: context)
    dispatcher = token_dispatcher.DeepEPTokenDispatcherNormal(
        group=object(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
        expert_alignment=128,
    )
    x = (torch.empty(1, 16), torch.empty(1, 2))
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.bfloat16)

    recv_x, recv_ids, recv_weights, counts = dispatcher.dispatch(
        x, ids, weights
    )
    assert recv_x is x
    assert recv_ids.shape == recv_weights.shape == (1, 2)
    assert counts == [128, 0]
    assert normal_marks == [True]
    layout_ids, experts, layout_kwargs = buffer.layout_kwargs
    assert layout_ids.dtype == torch.int64
    assert experts == 4
    assert layout_kwargs["async_finish"] is False
    _, dispatch_kwargs = buffer.dispatch_kwargs
    assert dispatch_kwargs["topk_weights"].dtype == torch.float32
    assert dispatch_kwargs["expert_alignment"] == 128
    with pytest.raises(RuntimeError, match="in-flight"):
        dispatcher.dispatch(x, ids, weights)

    output = dispatcher.combine(torch.empty(1, 16))
    assert output.shape == (1, 16)
    _, handle, combine_kwargs = buffer.combine_kwargs
    assert handle == ("normal",)
    assert combine_kwargs["async_finish"] is False
    with pytest.raises(RuntimeError, match="without dispatch"):
        dispatcher.combine(torch.empty(1, 16))


def test_low_latency_down_scales_are_column_major():
    scales = _allocate_column_major_scales(24, 4096, 16, torch.device("cpu"))
    assert scales.shape == (24, 4096, 16)
    assert scales.stride() == (4096 * 16, 1, 4096)
    assert not scales.is_contiguous()


def test_deep_gemm_adapter_uses_only_target_symbols_and_natural_scales(
    monkeypatch, tmp_path
):
    calls = []

    def record(name, result=None):
        def function(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return function

    fake = SimpleNamespace(
        fp8_gemm_nt=record("dense"),
        m_grouped_fp8_gemm_nt_contiguous=record("contiguous"),
        m_grouped_fp8_gemm_nt_masked=record("masked"),
        get_mk_alignment_for_contiguous_layout=record("alignment", 128),
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)
    deep_gemm_backend._reset_for_testing()

    assert deep_gemm_backend.get_m_alignment_for_contiguous_layout() == 128
    deep_gemm_backend.fp8_gemm_nt(1, 2, 3)
    deep_gemm_backend.m_grouped_fp8_gemm_nt_contiguous(1, 2, 3, 4)
    deep_gemm_backend.m_grouped_fp8_gemm_nt_masked(1, 2, 3, 4, 5)

    assert [call[0] for call in calls] == [
        "alignment", "dense", "contiguous", "masked"
    ]
    for _, _, kwargs in calls[1:]:
        assert kwargs["disable_ue8m0_cast"] is True
    assert str(tmp_path) in deep_gemm_backend.os.environ["DG_JIT_CACHE_DIR"]
    deep_gemm_backend._reset_for_testing()

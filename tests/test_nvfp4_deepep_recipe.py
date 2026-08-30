from types import SimpleNamespace
from unittest.mock import Mock

import torch
from dlengine.runtime.layers.backends.nvfp4.experts import ModelOptNvFp4Experts
from dlengine.runtime.layers.token_dispatcher import DeepEPTokenDispatcherLowLatency


def _experts():
    experts = ModelOptNvFp4Experts.__new__(ModelOptNvFp4Experts)
    experts.ep_group = object()
    experts.ep_size = 4
    experts.num_experts = 8
    experts.num_local_experts = 2
    experts.hidden_size = 4
    experts.top_k = 2
    experts.gate_up_proj = SimpleNamespace(device=torch.device("cpu"))
    experts.input1_quant = torch.tensor([2.0, 2.0])
    return experts


def test_low_latency_dispatch_forwards_nvfp4_quantization_options():
    scale = torch.tensor([2.0])
    event = Mock()
    buffer = Mock()
    buffer.group_size = 4
    buffer.low_latency_dispatch.return_value = (
        (torch.empty(1, 4), torch.empty(1, 1)),
        torch.tensor([1]),
        "handle",
        event,
        None,
    )
    dispatcher = DeepEPTokenDispatcherLowLatency.__new__(
        DeepEPTokenDispatcherLowLatency
    )
    dispatcher.num_experts = 8
    dispatcher.num_max_dispatch_tokens_per_rank = 16
    dispatcher.return_recv_hook = False
    dispatcher.buffer_low_latency = buffer

    dispatcher.dispatch(
        torch.empty(2, 4),
        torch.tensor([[0, 1], [2, 3]]),
        torch.ones(2, 2),
        use_fp8=False,
        use_nvfp4=True,
        x_global_scale=scale,
    )

    kwargs = buffer.low_latency_dispatch.call_args.kwargs
    assert kwargs["use_fp8"] is False
    assert kwargs["use_nvfp4"] is True
    assert kwargs["x_global_scale"] is scale
    event.current_stream_wait.assert_called_once_with()


def test_low_latency_dispatch_keeps_legacy_kwargs_by_default():
    event = Mock()
    buffer = Mock()
    buffer.group_size = 1
    buffer.low_latency_dispatch.return_value = (
        torch.empty(1, 4),
        torch.tensor([1]),
        "handle",
        event,
        None,
    )
    dispatcher = DeepEPTokenDispatcherLowLatency.__new__(
        DeepEPTokenDispatcherLowLatency
    )
    dispatcher.num_experts = 2
    dispatcher.num_max_dispatch_tokens_per_rank = 4
    dispatcher.return_recv_hook = False
    dispatcher.buffer_low_latency = buffer

    dispatcher.dispatch(
        torch.empty(1, 4), torch.tensor([[0]]), torch.ones(1, 1), use_fp8=False
    )

    kwargs = buffer.low_latency_dispatch.call_args.kwargs
    assert "use_nvfp4" not in kwargs
    assert "x_global_scale" not in kwargs


def test_decode_uses_low_latency_nvfp4_without_double_scaling(monkeypatch):
    import dlengine.runtime.context.expert as expert_context
    import dlengine.runtime.layers.token_dispatcher as token_dispatcher

    context = Mock()
    monkeypatch.setattr(
        expert_context.ExpertContext, "get_instance", Mock(return_value=context)
    )
    dispatcher = Mock()
    recv_hidden = torch.empty(2, 4)
    recv_scale = torch.empty(2, 1)
    recv_ids = torch.tensor([[0, 1]])
    recv_weights = torch.ones(1, 2)
    dispatcher.dispatch.return_value = (
        (recv_hidden, recv_scale),
        recv_ids,
        recv_weights,
        torch.tensor([1, 1]),
        1,
    )
    dispatcher.combine.return_value = torch.empty(1, 4)
    monkeypatch.setattr(
        token_dispatcher,
        "DeepEPTokenDispatcherLowLatency",
        Mock(return_value=dispatcher),
    )
    experts = _experts()
    experts._run_masked_experts = Mock(return_value=torch.empty(2, 4))

    experts._compute_decode_ep(
        torch.empty(1, 4), torch.tensor([[0, 1]]), torch.ones(1, 2)
    )

    context.transition_to_low_latency.assert_called_once_with()
    kwargs = dispatcher.dispatch.call_args.kwargs
    assert kwargs["use_fp8"] is False
    assert kwargs["use_nvfp4"] is True
    assert kwargs["x_global_scale"] is experts.input1_quant
    assert experts._run_masked_experts.call_args.args[2] is None


def test_prefill_uses_normal_dispatch_and_local_padding(monkeypatch):
    import dlengine.runtime.context.expert as expert_context
    import dlengine.runtime.layers.local_dispatch as local_dispatch
    import dlengine.runtime.layers.token_dispatcher as token_dispatcher

    context = Mock()
    monkeypatch.setattr(
        expert_context.ExpertContext, "get_instance", Mock(return_value=context)
    )
    dispatcher = Mock()
    recv_x = torch.empty(100, 4)
    recv_ids = torch.zeros(100, 2, dtype=torch.int64)
    recv_weights = torch.ones(100, 2)
    dispatcher.dispatch.return_value = (
        recv_x,
        recv_ids,
        recv_weights,
        None,
        None,
        None,
    )
    dispatcher.combine.return_value = torch.empty(3, 4)
    normal_cls = Mock(return_value=dispatcher)
    monkeypatch.setattr(token_dispatcher, "DeepEPTokenDispatcherNormal", normal_cls)

    padded = torch.empty(2, 128, 4)
    masked_m = torch.tensor([3, 3], dtype=torch.int32)
    local = Mock()
    local.dispatch.return_value = (padded, masked_m, 3)
    local.combine.return_value = recv_x
    local_cls = Mock(return_value=local)
    monkeypatch.setattr(local_dispatch, "LocalPaddedDispatcher", local_cls)

    experts = _experts()
    # Capacity must cover the maximally skewed routing case: T * top_k.
    expected_max_m = recv_x.shape[0] * experts.top_k
    experts._run_masked_experts = Mock(return_value=padded)
    experts._compute_prefill_ep(
        torch.empty(2, 4), torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 2)
    )

    context.transition_to_normal.assert_called_once_with()
    assert normal_cls.call_args.kwargs["expert_alignment"] == 1
    experts._run_masked_experts.assert_called_once_with(
        (padded, None), masked_m, experts.input1_quant
    )
    assert local_cls.call_args.kwargs["max_m"] == expected_max_m
    dispatcher.combine.assert_called_once_with(recv_x)

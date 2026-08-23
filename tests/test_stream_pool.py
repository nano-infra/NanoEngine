from unittest.mock import Mock

import pytest
import torch
from dlengine.runtime.stream_pool import (
    _clear_cuda_stream_pool_for_test,
    get_cuda_stream,
)


@pytest.fixture(autouse=True)
def clear_stream_pool():
    _clear_cuda_stream_pool_for_test()
    yield
    _clear_cuda_stream_pool_for_test()


def test_reuses_role_on_same_device(monkeypatch):
    stream_factory = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)

    first = get_cuda_stream("shared_expert", 0)
    second = get_cuda_stream("shared_expert", torch.device("cuda:0"))

    assert first is second
    stream_factory.assert_called_once_with(device=0, priority=0)


def test_separates_roles_devices_and_priorities(monkeypatch):
    stream_factory = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)

    streams = {
        get_cuda_stream("attention_q", 0),
        get_cuda_stream("attention_kv", 0),
        get_cuda_stream("attention_q", 1),
        get_cuda_stream("attention_q", 0, priority=-1),
    }

    assert len(streams) == 4
    assert stream_factory.call_count == 4


def test_rejects_empty_role_and_cpu_device():
    with pytest.raises(ValueError, match="non-empty"):
        get_cuda_stream("", 0)
    with pytest.raises(ValueError, match="non-CUDA"):
        get_cuda_stream("shared_expert", torch.device("cpu"))

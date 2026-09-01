from unittest.mock import Mock

import pytest
from dlengine.executor import dlslime_executor
from dlengine.executor.dlslime_executor import _select_driver_nic, DLSLimeExecutor


def test_unfiltered_multi_nic_selection_warns(monkeypatch):
    monkeypatch.delenv("SLIME_VISIBLE_DEVICES", raising=False)
    warning = Mock()
    monkeypatch.setattr(dlslime_executor.logger, "warning", warning)

    assert _select_driver_nic(["mlx5_0", "mlx5_1"]) == "mlx5_0"

    warning.assert_called_once()
    assert "SLIME_VISIBLE_DEVICES is not set" in warning.call_args.args[0]


def test_cuda_init_proxy_failure_has_focused_diagnostic(monkeypatch):
    monkeypatch.delenv("SLIME_VISIBLE_DEVICES", raising=False)
    executor = object.__new__(DLSLimeExecutor)
    executor._driver_agent = object()
    executor._proxy_factory = Mock(
        side_effect=RuntimeError("CUDA error: initialization error")
    )

    with pytest.raises(RuntimeError) as caught:
        executor._create_proxies(["worker:0"])

    message = str(caught.value)
    assert "pinned RPC buffers" in message
    assert "multiprocessing 'fork'" in message
    assert "before RDMA memory-region registration" in message
    assert "SLIME_VISIBLE_DEVICES='<unset>' only filters RDMA NIC" in message


def test_non_cuda_proxy_failure_is_not_wrapped():
    executor = object.__new__(DLSLimeExecutor)
    executor._driver_agent = object()
    original = ValueError("connection refused")
    executor._proxy_factory = Mock(side_effect=original)

    with pytest.raises(ValueError) as caught:
        executor._create_proxies(["worker:0"])

    assert caught.value is original

import importlib
import sys
import time
from types import ModuleType, SimpleNamespace


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _Endpoint:
    def __init__(self):
        self.sent = None

    def send_seqs(self, sequences, is_prefill):
        self.sent = (sequences, is_prefill)


def _load_ray_executor(monkeypatch):
    model_runner_module = ModuleType("nanodeploy.worker.model_runner")
    model_runner_module.ModelRunner = type("ModelRunner", (), {})
    monkeypatch.setitem(
        sys.modules,
        "nanodeploy.worker.model_runner",
        model_runner_module,
    )
    monkeypatch.delitem(
        sys.modules, "nanodeploy.engine.ray_executor", raising=False
    )
    ray_executor_module = importlib.import_module(
        "nanodeploy.engine.ray_executor"
    )
    return ray_executor_module.RayExecutor


def test_ray_executor_collects_worker_quantum_diagnostics(monkeypatch):
    RayExecutor = _load_ray_executor(monkeypatch)

    class Recorder:
        def __init__(self):
            self.samples = []

        def record(self, sample):
            self.samples.append(sample)

    worker_diagnostic = {
        "global_rank": 0,
        "recv_seqs_ms": 1.0,
        "prepare_update_host_ms": 2.0,
        "forward_host_ms": 3.0,
        "gpu_loop_ms": 4.0,
        "loop_host_ms": 5.0,
        "token_materialize_ms": 0.5,
        "worker_body_ms": 6.0,
        "worker_total_ms": 7.0,
    }
    run = _RemoteMethod(([[11, 12]], time.time(), worker_diagnostic))
    worker = SimpleNamespace(run=run)
    executor = object.__new__(RayExecutor)
    executor.config = SimpleNamespace(
        use_dlslime_rpc=True,
        hierarchical_quantum_diagnostics=True,
    )
    executor.workers = [worker]
    executor.endpoint = _Endpoint()
    executor._execution_boundary = Recorder()
    executor.last_quantum_diagnostic = None
    monkeypatch.setattr(
        "nanodeploy.engine.ray_executor.ray.get",
        lambda refs, timeout: refs,
    )

    output = executor.run([[object()]], is_prefill=False, timeout=1.0)

    assert output == [[[11, 12]]]
    assert executor.endpoint.sent is not None
    assert run.calls[0][1]["hierarchical_quantum_diagnostics"] is True
    diagnostic = executor.last_quantum_diagnostic
    assert diagnostic is not None
    assert diagnostic["critical_worker_global_rank"] == 0
    assert diagnostic["gpu_loop_ms_max"] == 4.0
    assert diagnostic["worker_rank_timings"] == (worker_diagnostic,)
    assert diagnostic["result_unpack_ms"] >= 0.0


def test_ray_executor_keeps_diagnostics_off_envelope(monkeypatch):
    RayExecutor = _load_ray_executor(monkeypatch)

    class Recorder:
        def record(self, _sample):
            return None

    run = _RemoteMethod(([[21, 22]], time.time()))
    executor = object.__new__(RayExecutor)
    executor.config = SimpleNamespace(
        use_dlslime_rpc=True,
        hierarchical_quantum_diagnostics=False,
    )
    executor.workers = [SimpleNamespace(run=run)]
    executor.endpoint = _Endpoint()
    executor._execution_boundary = Recorder()
    executor.last_quantum_diagnostic = None
    monkeypatch.setattr(
        "nanodeploy.engine.ray_executor.ray.get",
        lambda refs, timeout: refs,
    )

    output = executor.run([[object()]], is_prefill=False, timeout=1.0)

    assert output == [[[21, 22]]]
    assert run.calls[0][1]["hierarchical_quantum_diagnostics"] is False
    assert executor.last_quantum_diagnostic is None

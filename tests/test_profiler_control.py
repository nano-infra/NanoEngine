import asyncio
import json
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import pytest
from dlengine.engine.llm_engine import LLMEngine
from dlengine.server.engine_server import (
    _ACTION_START_PROFILER,
    _ACTION_STOP_PROFILER,
    BackendService,
)
from dlengine.server.zmq_engine_client import ZmqEngineWorker
from dlengine.worker.model_runner import ModelRunner


class _FakeProfiler:
    def __init__(self, trace_file: Path):
        self.trace_file = trace_file
        self.started = 0
        self.stepped = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def step(self):
        self.stepped += 1

    def stop(self):
        self.stopped += 1
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.trace_file.write_text("{}", encoding="utf-8")


def _bare_runner(tmp_path: Path):
    runner_cls = ModelRunner.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner.config = SimpleNamespace(profiler_dir=str(tmp_path))
    runner.engine_id = "test-engine"
    runner.rank = 0
    runner.run_count = 12
    runner.profiler = None
    runner.profiler_active = False
    runner.profiler_mode = None
    runner.profiler_trace_dir = None
    runner.profiler_start_step = 34
    runner.profiler_end_step = 50
    runner.profiler_forward_per_step = 2
    return runner


def test_model_runner_manual_profiler_session(tmp_path, monkeypatch):
    runner = _bare_runner(tmp_path)
    fake = _FakeProfiler(tmp_path / "baseline" / "worker.pt.trace.json")
    monkeypatch.setattr(runner, "_create_profiler", lambda _trace_dir: fake)

    started = runner.start_profiler("baseline")
    duplicate = runner.start_profiler("ignored")
    runner._advance_profiler()
    stopped = runner.stop_profiler()
    repeated_stop = runner.stop_profiler()

    assert started["status"] == "started"
    assert duplicate["status"] == "already_running"
    assert fake.started == 1
    assert fake.stepped == 1
    assert fake.stopped == 1
    assert stopped["status"] == "stopped"
    assert stopped["trace_files"] == [str(fake.trace_file)]
    assert repeated_stop["status"] == "not_running"


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    def collective_rpc(self, method, args=None):
        self.calls.append((method, args))
        return [{"rank": 0, "status": "started" if args else "stopped"}]


def test_llm_engine_validates_trace_name():
    engine = object.__new__(LLMEngine)
    engine.executor = _FakeExecutor()

    result = engine.start_profiler("baseline-01")
    stopped = engine.stop_profiler()

    assert result["trace_name"] == "baseline-01"
    assert result["status"] == "started"
    assert stopped["status"] == "stopped"
    assert stopped["workers"][0]["status"] == "stopped"
    assert engine.executor.calls == [
        ("start_profiler", ("baseline-01",)),
        ("stop_profiler", None),
    ]
    with pytest.raises(ValueError, match="trace_name"):
        engine.start_profiler("../escape")


class _FakeEngine:
    def start_profiler(self, trace_name):
        return {"trace_name": trace_name, "workers": [{"status": "started"}]}

    def stop_profiler(self):
        return {"workers": [{"status": "stopped"}]}


def test_backend_profiler_control_round_trip():
    results = SimpleQueue()
    service = BackendService(_FakeEngine(), results)

    service._handle_profiler_control(
        _ACTION_START_PROFILER,
        json.dumps({"trace_name": "baseline"}).encode(),
    )
    service._handle_profiler_control(_ACTION_STOP_PROFILER, b"")

    start_action, start_payload = results.get()
    stop_action, stop_payload = results.get()
    assert start_action == _ACTION_START_PROFILER
    assert json.loads(start_payload)["trace_name"] == "baseline"
    assert stop_action == _ACTION_STOP_PROFILER
    assert json.loads(stop_payload)["workers"][0]["status"] == "stopped"


def test_zmq_client_profiler_control_response():
    async def run_test():
        worker = ZmqEngineWorker("ipc:///unused")
        worker._socket = object()
        worker._outbox = asyncio.Queue()

        task = asyncio.create_task(worker.start_profiler("baseline"))
        action, payload = await worker._outbox.get()
        assert action == _ACTION_START_PROFILER
        assert json.loads(payload)["trace_name"] == "baseline"

        worker._handle_profiler_response(b'{"ok": true, "trace_name": "baseline"}')
        assert await task == {"ok": True, "trace_name": "baseline"}

    asyncio.run(run_test())

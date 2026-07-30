from __future__ import annotations

import importlib
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

from nanodeploy.engine.decode_coordinator import DecodeCoordinatorState
from nanodeploy.engine.execution_boundary import ExecutionBoundaryRecorder
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AbortResult,
    FirstScheduleEvent,
    FirstTokenEvent,
    FrontendEventBatch,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    validate_execution_trace_set,
)
from nanodeploy.engine.local_executor import LocalExecutor
from nanodeploy.engine.local_engine import LocalEngineCore
from nanodeploy.engine.topology import EngineTopology
from nanodeploy.router.request_router import RequestOwner, RequestRouter


@dataclass
class FakeEngine:
    engine_id: int
    add_reasons: list[str | None] = field(default_factory=list)
    commands: list[AddCommand] = field(default_factory=list)
    aborts: list[int] = field(default_factory=list)

    def add(self, command: AddCommand) -> AddResult:
        self.commands.append(command)
        reason = self.add_reasons.pop(0) if self.add_reasons else None
        return AddResult(
            request_id=command.request_id,
            accepted=reason is None,
            engine_id=self.engine_id,
            reason=reason,
        )

    def abort(self, request_id: int) -> AbortResult:
        self.aborts.append(request_id)
        return AbortResult(request_id=request_id, status="aborted")

    def load(self) -> LoadSnapshot:
        return LoadSnapshot(
            engine_id=self.engine_id,
            ready=True,
            waiting=0,
            running=0,
            free_blocks_min=10,
            wave_id=1,
            quantum_id=2,
        )


def route(router: RequestRouter, request_id: int) -> AddResult:
    return router.add(
        request_id=request_id,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )


def test_execution_boundary_recorder_reset_and_snapshot():
    recorder = ExecutionBoundaryRecorder()
    recorder.record({"phase_ms": 2.0, "signed_gap_ms": -1.0})
    recorder.record({"phase_ms": 4.0, "signed_gap_ms": 3.0})

    assert recorder.snapshot() == {
        "sample_count": 2,
        "phase_ms_total": 6.0,
        "phase_ms_mean": 3.0,
        "phase_ms_min": 2.0,
        "phase_ms_max": 4.0,
        "signed_gap_ms_total": 2.0,
        "signed_gap_ms_mean": 1.0,
        "signed_gap_ms_min": -1.0,
        "signed_gap_ms_max": 3.0,
    }

    recorder.reset()
    assert recorder.snapshot() == {"sample_count": 0}


def load_snapshot(engine_id: int, free_blocks_min: int) -> LoadSnapshot:
    return LoadSnapshot(
        engine_id=engine_id,
        ready=True,
        waiting=0,
        running=0,
        free_blocks_min=free_blocks_min,
        wave_id=1,
        quantum_id=2,
    )


def test_router_round_robin_queue_full_retry_and_sticky_owner():
    engines = {
        0: FakeEngine(0, add_reasons=["queue_full"]),
        1: FakeEngine(1),
    }
    wakeups = []

    def wake(engine_id: int, wave_id: int) -> int:
        wakeups.append((engine_id, wave_id))
        return wave_id + 1

    router = RequestRouter(engines, wakeup=wake)
    result = route(router, 10)

    assert result.accepted and result.engine_id == 1
    assert [command.request_id for command in engines[0].commands] == [10]
    assert [command.request_id for command in engines[1].commands] == [10]
    assert router.owner(10).state == OwnerState.OWNED
    assert router.owner(10).engine_id == 1
    assert wakeups == [(1, 0)]

    assert router.abort(10).status == "aborted"
    assert engines[1].aborts == [10]
    router.finish(FinishEvent(10, 16, "ABORTED", 1))
    assert router.is_idle
    assert router.abort(10).status == "already_terminal"


def test_router_rejects_duplicate_and_wrong_owner_terminal_event():
    engines = {0: FakeEngine(0), 1: FakeEngine(1)}
    router = RequestRouter(engines)
    assert route(router, 20).accepted
    assert route(router, 20).reason == "duplicate_request_id"

    with pytest.raises(RuntimeError, match="owner mismatch"):
        router.finish(FinishEvent(20, 16, "FINISHED", 1))
    router.finish(FinishEvent(20, 16, "FINISHED", 0))
    with pytest.raises(RuntimeError, match="duplicate terminal"):
        router.finish(FinishEvent(20, 16, "FINISHED", 0))


def test_router_round_robin_advances_once_per_request():
    engines = {0: FakeEngine(0), 1: FakeEngine(1)}
    router = RequestRouter(engines, router_policy="round_robin")

    assert route(router, 1).engine_id == 0
    assert route(router, 2).engine_id == 1
    assert route(router, 3).engine_id == 0


def test_router_rejects_invalid_load_policy_configuration():
    engines = {0: FakeEngine(0)}
    assert RequestRouter(engines).router_policy == "least_batch"
    with pytest.raises(ValueError, match="unsupported router policy"):
        RequestRouter(engines, router_policy="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive kvcache_block_size"):
        RequestRouter(engines, router_policy="least_cache")


@dataclass
class FakeAsyncEngine(FakeEngine):
    enqueue_reasons: list[str | None] = field(default_factory=list)
    handles: list[dict] = field(default_factory=list)
    admission_version: int = 0

    def enqueue_async(self, command: AddCommand):
        reason = (
            self.enqueue_reasons.pop(0)
            if self.enqueue_reasons
            else None
        )
        handle = {
            "ready": False,
            "ack": IngressAck(
                request_id=command.request_id,
                engine_id=self.engine_id,
                enqueued=reason is None,
                reason=reason,
            ),
        }
        self.commands.append(command)
        self.handles.append(handle)
        return handle

    def admit_async(self, command: AddCommand):
        handle = self.enqueue_async(command)
        if handle["ack"].enqueued:
            self.admission_version += 1
            handle["ack"] = IngressAck(
                request_id=command.request_id,
                engine_id=self.engine_id,
                enqueued=True,
                admission_version=self.admission_version,
            )
        return handle

    def admit_batch_async(self, commands: tuple[AddCommand, ...]):
        acks = []
        for command in commands:
            reason = (
                self.enqueue_reasons.pop(0)
                if self.enqueue_reasons
                else None
            )
            self.commands.append(command)
            if reason is None:
                self.admission_version += 1
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=reason is None,
                    reason=reason,
                    admission_version=(
                        self.admission_version
                        if reason is None
                        else None
                    ),
                )
            )
        handle = {"ready": False, "acks": tuple(acks)}
        self.handles.append(handle)
        return handle

    def poll_enqueue(self, handle):
        if not handle["ready"]:
            return False, None
        return True, handle["ack"]

    def poll_admission_batch(self, handle):
        if not handle["ready"]:
            return False, None
        return True, handle["acks"]


def test_router_least_batch_drains_global_pending_with_tentative_counts():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(engines, router_policy="least_batch")

    for request_id in (1, 2, 3, 4):
        router.submit_async(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert engines[0].commands == []
    assert engines[1].commands == []
    assert router.pending_global_count == 4
    assert all(
        router.owner(request_id).state == OwnerState.PENDING_GLOBAL
        for request_id in (1, 2, 3, 4)
    )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1, 3]
    assert [command.request_id for command in engines[1].commands] == [2, 4]
    assert all(
        router.owner(request_id).state == OwnerState.PENDING_INGRESS
        for request_id in (1, 2, 3, 4)
    )


def test_router_least_batch_uses_live_running_plus_tentative_admissions():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(engines, router_policy="least_batch")
    router.record_loads(
        (
            LoadSnapshot(0, True, 0, 3, 10, 1, 2),
            LoadSnapshot(1, True, 0, 1, 10, 1, 2),
        )
    )

    for request_id in (1, 2, 3):
        router.submit_async(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )
    assert router.poll_ingress_acks() == ()

    # Start from running=[3, 1], then account for each tentative admission:
    # request 1 -> DP1, request 2 -> DP1, request 3 -> DP0 after the
    # tentative charges bring both projected batches to three.
    assert [command.request_id for command in engines[0].commands] == [3]
    assert [command.request_id for command in engines[1].commands] == [1, 2]


def test_router_polls_at_most_one_admission_batch_per_dp():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    poll_calls = []

    def poll_batches(handles):
        poll_calls.append(dict(handles))
        return {
            engine_id: handle["acks"]
            for engine_id, handle in handles.items()
            if handle["ready"]
        }

    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_batch_size=2,
        poll_admission_batches=poll_batches,
    )
    for request_id in range(5):
        router.submit_async(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert router.poll_ingress_acks() == ()
    assert len(poll_calls) == 1
    assert set(poll_calls[0]) == {0, 1}
    assert all(
        len(handle["acks"]) == 2
        for handle in poll_calls[0].values()
    )
    metrics = router.admission_metrics()
    assert metrics["pending_rpc"] == 2
    assert metrics["pending_rpc_requests"] == 4
    assert metrics["global_pending"] == 1


def test_local_engine_drains_frontend_events_in_one_batch():
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.engine_id = 0
    engine._failure = None
    engine._loop_thread = SimpleNamespace(is_alive=lambda: True)
    engine._events_lock = threading.Lock()
    engine._load_lock = threading.Lock()
    engine._add_result_events = deque((AddResultEvent(1, 0, True),))
    engine._first_schedule_events = deque(
        (FirstScheduleEvent(1, 0, 3.0),)
    )
    engine._first_token_events = deque((FirstTokenEvent(1, 0, 1),))
    engine._terminal_events = deque(
        (FinishEvent(1, 16, "FINISHED", 0),)
    )
    engine._cached_load_snapshot = load_snapshot(0, 9)

    batch = engine.drain_frontend_events()

    assert isinstance(batch, FrontendEventBatch)
    assert batch.load.free_blocks_min == 9
    assert batch.add_results[0].request_id == 1
    assert batch.first_schedule_events[0].request_id == 1
    assert batch.first_token_events[0].request_id == 1
    assert batch.finish_events[0].request_id == 1
    assert engine._add_result_events == deque()
    assert engine._first_schedule_events == deque()
    assert engine._first_token_events == deque()
    assert engine._terminal_events == deque()


def test_hierarchical_engine_is_not_finished_with_buffered_finish_events(
    monkeypatch,
):
    lightweight_imports = {
        "nanodeploy.engine.deployment_manager": "DeploymentManager",
        "nanodeploy.engine.ray_executor": "RayExecutor",
        "nanodeploy.engine.scheduler": "Scheduler",
        "nanodeploy.engine.sequence": "Sequence",
    }
    for module_name, symbol in lightweight_imports.items():
        module = ModuleType(module_name)
        setattr(module, symbol, type(symbol, (), {}))
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delitem(
        sys.modules, "nanodeploy.engine.llm_engine", raising=False
    )
    llm_engine_module = importlib.import_module(
        "nanodeploy.engine.llm_engine"
    )
    LLMEngine = llm_engine_module.LLMEngine
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(scheduler_arch="hierarchical")
    engine.router = SimpleNamespace(is_idle=True)
    engine._frontend_finish_events = deque(
        (FinishEvent(1, 16, "FINISHED", 0),)
    )

    assert not engine.is_finished()
    engine._frontend_finish_events.clear()
    assert engine.is_finished()


def test_router_buffers_terminal_until_async_owner_commit():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(engines, router_policy="least_batch")
    router.submit_async(
        request_id=5,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert router.owner(5) == RequestOwner(
        OwnerState.PENDING_INGRESS, 0
    )
    event = FinishEvent(
        5,
        16,
        "FINISHED",
        0,
        first_forward_to_terminal_ms=750.0,
    )
    assert router.record_finish_events((event,)) == ()

    engines[0].handles[0]["ready"] = True
    ack = router.poll_ingress_acks()[0]
    assert ack.enqueued
    assert ack.router_pending_ms is not None
    assert ack.admission_rpc_ms is not None
    assert router.owner(5) == RequestOwner(OwnerState.PENDING_ADD, 0)
    assert router.record_finish_events(()) == ()

    router.record_add_results((AddResultEvent(5, 0, True),))
    terminal_events = router.record_finish_events(())
    assert len(terminal_events) == 1
    assert terminal_events[0].request_id == 5
    assert terminal_events[0].first_forward_to_terminal_ms == 750.0
    assert router.terminal_event(5) == terminal_events[0]
    assert router.is_idle


def test_router_retries_globally_after_all_dps_defer_admission(
    monkeypatch,
):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "nanodeploy.router.request_router.perf_counter",
        lambda: clock["now"],
    )
    engines = {
        0: FakeAsyncEngine(
            0, enqueue_reasons=["admission_deferred"]
        ),
        1: FakeAsyncEngine(
            1, enqueue_reasons=["admission_deferred"]
        ),
    }
    router = RequestRouter(engines, router_policy="least_batch")
    router.record_loads(
        (load_snapshot(0, 10), load_snapshot(1, 10))
    )
    router.submit_async(
        request_id=9,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    engines[0].handles[0]["ready"] = True
    assert router.poll_ingress_acks() == ()
    clock["now"] = 10.0
    engines[1].handles[0]["ready"] = True
    assert router.poll_ingress_acks() == ()
    assert router.owner(9) == RequestOwner(OwnerState.PENDING_GLOBAL)
    assert router.pending_global_count == 1

    # The unchanged load generation cannot spin on the same infeasible head.
    assert router.poll_ingress_acks() == ()
    assert len(engines[0].commands) == 1
    assert len(engines[1].commands) == 1

    clock["now"] = 35.0
    router.record_loads(
        (
            LoadSnapshot(
                0,
                True,
                0,
                0,
                11,
                1,
                3,
                capacity_epoch=1,
            ),
            LoadSnapshot(
                1,
                True,
                0,
                0,
                11,
                1,
                3,
                capacity_epoch=1,
            ),
        )
    )
    assert router.poll_ingress_acks() == ()
    assert len(engines[0].commands) == 2
    engines[0].handles[1]["ready"] = True
    ack = router.poll_ingress_acks()[0]
    assert ack.enqueued
    assert ack.router_pending_ms == 25_000.0
    assert ack.admission_rpc_ms == 10_000.0
    schedule_events = router.record_first_schedule_events(
        (FirstScheduleEvent(9, 0, local_scheduler_queue_ms=7.0),)
    )
    assert len(schedule_events) == 1
    assert schedule_events[0].global_capacity_queue_ms == 25_000.0
    assert schedule_events[0].local_scheduler_queue_ms == 7.0
    assert schedule_events[0].first_schedule_latency_ms == 25_007.0
    router.record_add_results((AddResultEvent(9, 0, True),))
    terminal = router.finish(
        FinishEvent(
            9,
            16,
            "FINISHED",
            0,
            first_forward_to_terminal_ms=1_000.0,
        )
    )
    assert terminal.first_forward_to_terminal_ms == 1_000.0
    assert terminal.global_capacity_queue_ms == 25_000.0
    admission_metrics = router.admission_metrics()
    assert admission_metrics["global_retries"] == 2
    assert admission_metrics["fallbacks"] == 0
    assert admission_metrics["per_engine"]["0"]["attempts"] == 2
    assert admission_metrics["per_engine"]["0"]["deferred"] == 1
    assert admission_metrics["per_engine"]["1"]["attempts"] == 1
    assert admission_metrics["per_engine"]["1"]["deferred"] == 1


def test_router_least_cache_uses_padded_request_blocks_optimistically():
    engines = {0: FakeEngine(0), 1: FakeEngine(1)}
    router = RequestRouter(
        engines,
        router_policy="least_cache",
        kvcache_block_size=4,
    )
    router.record_loads(
        (load_snapshot(0, 20), load_snapshot(1, 12))
    )

    # prompt=2, padded completion=32, so each request is charged
    # ceil((2 + 32) / 4) = 9 blocks. After request 1, engine 0 has an
    # optimistic 11 blocks and request 2 therefore goes to engine 1.
    first = router.add(
        request_id=1,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )
    router.record_loads(
        (load_snapshot(0, 20), load_snapshot(1, 12))
    )
    second = router.add(
        request_id=2,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )

    assert first.engine_id == 0
    assert second.engine_id == 1

    # A fresh load report replaces optimistic deductions.
    router.record_loads(
        (load_snapshot(0, 30), load_snapshot(1, 10))
    )
    third = router.add(
        request_id=3,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )
    assert third.engine_id == 0


def test_router_least_cache_refunds_queue_full_before_fallback():
    engines = {
        0: FakeAsyncEngine(0, enqueue_reasons=["queue_full"]),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_cache",
        kvcache_block_size=4,
    )
    router.record_loads(
        (load_snapshot(0, 20), load_snapshot(1, 12))
    )

    router.submit_async(
        request_id=1,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )
    engines[0].handles[0]["ready"] = True
    assert router.poll_ingress_acks() == ()
    assert router.owner(1).engine_id == 1

    # The failed engine-0 estimate was refunded while engine 1 carries the
    # fallback charge, so the next request selects engine 0.
    router.submit_async(
        request_id=2,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )
    assert [command.request_id for command in engines[0].commands] == [1, 2]
    assert [command.request_id for command in engines[1].commands] == [1]


def test_router_async_ingress_fallback_and_add_state_transition():
    engines = {
        0: FakeAsyncEngine(0, enqueue_reasons=["queue_full"]),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(engines)

    router.submit_async(
        request_id=40,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.pending_ingress_count == 1
    assert router.owner(40).state == OwnerState.PENDING_GLOBAL
    assert router.poll_ingress_acks() == ()
    assert router.owner(40).engine_id == 0

    engines[0].handles[0]["ready"] = True
    assert router.poll_ingress_acks() == ()
    assert router.owner(40).engine_id == 1

    engines[1].handles[0]["ready"] = True
    ack = router.poll_ingress_acks()
    assert len(ack) == 1 and ack[0].enqueued
    assert router.owner(40).state == OwnerState.PENDING_ADD
    assert router.pending_add_count == 1
    router.record_loads(
        (
            LoadSnapshot(0, True, 0, 0, 10, 1, 2),
            LoadSnapshot(
                1,
                True,
                0,
                1,
                10,
                1,
                2,
                admission_version=1,
            ),
        )
    )
    assert (
        router.admission_metrics()["per_engine"]["1"][
            "tentative_admissions"
        ]
        == 0
    )

    result = AddResultEvent(40, 1, True)
    assert router.record_add_results((result,)) == (result,)
    assert router.owner(40) == RequestOwner(OwnerState.OWNED, 1)


def test_router_buffers_add_result_observed_before_ingress_ack():
    engine = FakeAsyncEngine(0)
    router = RequestRouter({0: engine})
    router.submit_async(
        request_id=41,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()

    event = AddResultEvent(41, 0, True)
    assert router.record_add_results((event,)) == ()
    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued
    assert router.record_add_results(()) == (event,)
    assert router.owner(41).state == OwnerState.OWNED


def test_local_engine_ingress_is_nonblocking_and_reserves_lifecycle_capacity(
    monkeypatch,
):
    fake_now = 0.0

    def fake_perf_counter():
        nonlocal fake_now
        fake_now += 1.0
        return fake_now

    # A zero time budget must keep draining up to the request-count cap even
    # when the synthetic clock advances far beyond the former 10 ms limit.
    monkeypatch.setattr(
        "nanodeploy.engine.local_engine.perf_counter",
        fake_perf_counter,
    )

    class FakeScheduler:
        def __init__(self):
            self.commands = []

        def add(self, command):
            self.commands.append(command)
            return AddResult(command.request_id, True, engine_id=0)

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        attention_dp=1,
        hierarchical_queue_capacity=2,
        max_ingress_batch_requests=256,
        max_ingress_drain_ms=0.0,
    )
    engine.engine_id = 0
    engine.scheduler = FakeScheduler()
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = set()
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._admission_version = 0
    engine._capacity_epoch = 0
    engine._events_lock = threading.Lock()
    engine._add_result_events = deque()
    engine._first_token_events = deque()
    engine._terminal_events = deque()
    engine._state_cv = threading.Condition()
    engine._wave_running = False
    engine._wave_id = 0
    engine._quantum_id = 0
    engine._coordinator = None
    engine._ingress_queue_delay_ms_total = 0.0
    engine._scheduler_add_ms_total = 0.0

    commands = [
        AddCommand(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=0,
        )
        for request_id in (50, 51, 52)
    ]
    assert engine.enqueue_add(commands[0]).enqueued
    assert engine.enqueue_add(commands[1]).enqueued
    assert not engine.enqueue_add(commands[0]).enqueued
    full = engine.enqueue_add(commands[2])
    assert not full.enqueued and full.reason == "queue_full"
    assert engine.scheduler.commands == []

    engine._drain_ingress()
    assert engine.scheduler.commands == commands[:2]
    assert [event.request_id for event in engine.drain_add_results()] == [
        50,
        51,
    ]
    assert engine._reserved_slots == 2

    engine._publish_events((FinishEvent(50, 16, "FINISHED", 0),))
    assert engine._reserved_slots == 1
    assert engine._capacity_epoch == 1
    assert engine.enqueue_add(commands[2]).enqueued
    assert engine.submit_abort(52).status == "abort_pending"
    engine._drain_ingress()
    assert engine.drain_add_results()[0].accepted
    assert engine.drain_events() == (
        FinishEvent(50, 16, "FINISHED", 0),
        FinishEvent(52, 0, "ABORTED", 0),
    )
    assert engine._reserved_slots == 1


def test_local_engine_central_admission_commits_only_after_local_plan():
    class FakeScheduler:
        def __init__(self):
            self.batches = []

        def try_admit_batch(self, commands):
            self.batches.append(
                tuple(command.request_id for command in commands)
            )
            return tuple(
                {
                    60: AddResult(60, True, engine_id=0),
                    61: AddResult(
                        61,
                        False,
                        engine_id=0,
                        reason="admission_deferred",
                    ),
                }[command.request_id]
                for command in commands
            )

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(attention_dp=1)
    engine.engine_id = 0
    engine.scheduler = FakeScheduler()
    engine._failure = None
    engine._coordinator = None
    engine._command_count = 0
    engine._command_queue_delay_ms_total = 0.0
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = {60, 61}
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._admission_version = 0
    engine._capacity_epoch = 0
    engine._events_lock = threading.Lock()
    engine._add_result_events = deque()
    engine._terminal_events = deque()
    engine._state_cv = threading.Condition()
    engine._wave_running = False
    engine._wave_id = 0
    engine._quantum_id = 0

    def loop_command(request_id):
        return SimpleNamespace(
            kind="admit",
            payload=AddCommand(
                request_id=request_id,
                prompt_token_ids=(1, 2),
                max_tokens=16,
                temperature=0.1,
                ignore_eos=True,
                wave_id=0,
            ),
            enqueued_at=0.0,
            completed=threading.Event(),
            result=None,
            error=None,
        )

    accepted = loop_command(60)
    deferred = loop_command(61)
    engine._complete_admission_commands((accepted, deferred))
    assert engine.scheduler.batches == [(60, 61)]
    assert accepted.error is None
    assert accepted.result.request_id == 60
    assert accepted.result.engine_id == 0
    assert accepted.result.enqueued
    assert accepted.result.admission_version == 1
    assert accepted.result.local_command_queue_ms >= 0.0
    assert accepted.result.local_admission_ms >= 0.0
    assert engine._reserved_request_ids == {60}
    assert engine._reserved_slots == 1
    assert engine.drain_add_results() == (
        AddResultEvent(60, 0, True),
    )
    assert engine._wave_running

    assert deferred.error is None
    assert deferred.result.request_id == 61
    assert deferred.result.engine_id == 0
    assert not deferred.result.enqueued
    assert deferred.result.reason == "admission_deferred"
    assert deferred.result.local_command_queue_ms >= 0.0
    assert deferred.result.local_admission_ms >= 0.0
    assert engine._reserved_request_ids == {60}
    assert engine._reserved_slots == 1
    assert 61 not in engine._admission_pending_ids


def test_decode_coordinator_ready_wave_and_racing_wakeup():
    state = DecodeCoordinatorState(expected_engines=2)
    state.register(0, "same")
    state.register(1, "same")
    state.mark_ready(0)
    state.mark_ready(1)
    assert state.status().ready

    start = state.first_request(target_engine_id=1, observed_wave_id=0)
    assert start.wave_id == 1
    assert state.status().running

    # A request racing with the final consensus cannot be lost. It schedules a
    # follow-up wave if WAVE_COMPLETE wins the race.
    assert state.first_request(0, observed_wave_id=0) is None
    assert state.status().pending_wakeup
    next_start = state.wave_complete(1)
    assert next_start.wave_id == 2
    assert state.status().running

    assert state.wave_complete(2) is None
    assert not state.status().running
    with pytest.raises(RuntimeError, match="while paused"):
        state.wave_complete(2)


def test_decode_coordinator_rejects_fingerprint_mismatch():
    state = DecodeCoordinatorState(expected_engines=2)
    state.register(0, "a")
    state.register(1, "b")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        state.mark_ready(0)


def test_decode_coordinator_registration_is_idempotent_after_ready():
    state = DecodeCoordinatorState(expected_engines=2)
    state.register(0, "same")
    state.register(1, "same")
    state.mark_ready(0)
    state.mark_ready(1)

    state.register(0, "same")

    assert state.status().ready


def test_router_rejects_mismatched_protocol_ids_and_duplicate_loads():
    class BadEngine(FakeEngine):
        def add(self, command: AddCommand) -> AddResult:
            return AddResult(
                request_id=command.request_id + 1,
                accepted=True,
                engine_id=self.engine_id,
            )

    router = RequestRouter({0: BadEngine(0)})
    with pytest.raises(RuntimeError, match="inconsistent request id"):
        route(router, 30)
    assert router.owner(30) is None

    good_router = RequestRouter({0: FakeEngine(0)})
    snapshot = FakeEngine(0).load()
    with pytest.raises(ValueError, match="duplicate engine ids"):
        good_router.record_loads((snapshot, snapshot))


def _trace(global_rank: int, wave_id: int, quantum_id: int) -> dict:
    return {
        "global_rank": global_rank,
        "wave_id": wave_id,
        "quantum_id": quantum_id,
        "forward_count": 16,
        "batch_kind": "all_control_dummy",
        "real_batch_size": 0,
        "control_dummy_count": 1,
        "forwards": tuple(
            {
                "inner_loop_idx": inner_loop_idx,
                "use_sp_a2a": False,
                "forward_begin": float(inner_loop_idx),
                "forward_end": float(inner_loop_idx + 1),
            }
            for inner_loop_idx in range(16)
        ),
    }


def test_execution_trace_validation_requires_identical_global_steps():
    traces = [
        _trace(rank, wave_id, quantum_id)
        for rank in (0, 1)
        for wave_id, quantum_id in ((1, 0), (1, 1), (2, 0))
    ]
    assert validate_execution_trace_set(traces, (0, 1)) == (
        (1, 0),
        (1, 1),
        (2, 0),
    )

    with pytest.raises(ValueError, match="different wave/quantum"):
        validate_execution_trace_set(traces[:-1], (0, 1))


@pytest.mark.parametrize("result_fastpath", [False, True])
@pytest.mark.parametrize("quantum_diagnostics", [False, True])
@pytest.mark.parametrize("execution_trace", [False, True])
def test_local_executor_uses_keyword_only_nested_actor_calls(
    monkeypatch,
    result_fastpath,
    quantum_diagnostics,
    execution_trace,
):
    class FakeEndpoint:
        def __init__(self, *_args, **_kwargs):
            self.connected = None
            self.sent = None

        def init_server_endpoint(self):
            return ("server",)

        def connect(self, client_info):
            self.connected = client_info

        def send_seqs(self, sequences, *, is_prefill):
            self.sent = (sequences, is_prefill)

    class RemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, **kwargs):
            self.calls.append(kwargs)
            return self.result

    class FakeWorker:
        def __init__(self):
            self.init_rpc_endpoint = RemoteMethod(("client",))
            result = (
                [
                    list(range(16)),
                    [0] * 16,
                ],
                1.0,
            )
            if execution_trace:
                result += (
                    {
                        "wave_id": 1,
                        "quantum_id": 0,
                        "global_rank": 0,
                        "forward_count": 16,
                        "real_batch_size": 1,
                        "control_dummy_count": 1,
                        "batch_kind": "real_or_mixed",
                        "forwards": tuple(
                            {
                                "inner_loop_idx": inner_loop_idx,
                                "use_sp_a2a": False,
                            }
                            for inner_loop_idx in range(16)
                        ),
                    },
                )
            if quantum_diagnostics:
                result += (
                    {
                        "global_rank": 0,
                        "recv_seqs_ms": 1.0,
                        "prepare_update_host_ms": 2.0,
                        "forward_host_ms": 3.0,
                        "gpu_loop_ms": 4.0,
                        "loop_host_ms": 5.0,
                        "token_materialize_ms": 0.5,
                        "worker_body_ms": 6.0,
                        "worker_total_ms": 7.0,
                    },
                )
            self.run = RemoteMethod(result)

    class FakeSequence:
        def __init__(self, seq_id):
            self.seq_id = seq_id

        @staticmethod
        def block_ctx():
            return SimpleNamespace(master_sp_idx=0)

    class FakeBatch:
        engine_id = 0
        wave_id = 1
        quantum_id = 0
        engine_has_real = True
        real_sequence = FakeSequence(7)
        control_dummy = FakeSequence(-1)
        per_rank_sequences = {0: [real_sequence, control_dummy]}

        @staticmethod
        def is_control_dummy(sequence):
            return sequence.seq_id == -1

        @staticmethod
        def expected_request_ids(_global_rank):
            return (7,)

    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.RPCServerEndpoint",
        FakeEndpoint,
    )
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.ray.get",
        lambda refs, timeout: refs,
    )
    config = SimpleNamespace(
        optimize_decode_block_table=True,
        hierarchical_execution_trace=execution_trace,
        hierarchical_quantum_diagnostics=quantum_diagnostics,
        hierarchical_result_fastpath=result_fastpath,
    )
    topology = EngineTopology(
        engine_id=0,
        global_dp_idx=0,
        global_ranks=(0,),
        attention_sp=1,
        attention_tp=1,
    )
    worker = FakeWorker()
    executor = LocalExecutor(config, topology, [worker])

    executor.initialize_endpoint(timeout=1.0)
    results = executor.run(FakeBatch(), timeout=1.0)

    assert worker.init_rpc_endpoint.calls == [
        {"server_info": ("server",)}
    ]
    assert len(worker.run.calls) == 1
    run_call = dict(worker.run.calls[0])
    assert run_call.pop("send_timestamp") > 0
    trace_context = run_call.pop("hierarchical_trace")
    if execution_trace:
        assert trace_context == {
            "wave_id": 1,
            "quantum_id": 0,
            "global_rank": 0,
            "real_batch_size": 1,
            "control_dummy_count": 1,
            "batch_kind": "real_or_mixed",
        }
    else:
        assert trace_context is None
    assert run_call == {
        "dp_seqs": [],
        "is_prefill": False,
        "enable_rpc": True,
        "hierarchical_quantum_diagnostics": quantum_diagnostics,
    }
    assert len(results) == 1
    assert results[0].mastered_request_ids == (7,)
    assert results[0].sampled_token_ids == (tuple(range(16)),)
    assert executor.result_rebuild_sample_count == 1
    assert executor.result_rebuild_latency_ms_total >= 0
    boundary = executor.execution_boundary_metrics()
    assert boundary["sample_count"] == 1
    assert boundary["ray_get_latency_ms_mean"] >= 0
    if quantum_diagnostics:
        assert executor.last_quantum_diagnostic is not None
        assert (
            executor.last_quantum_diagnostic[
                "critical_worker_global_rank"
            ]
            == 0
        )
        assert executor.last_quantum_diagnostic["gpu_loop_ms_max"] == 4.0
        assert executor.last_quantum_diagnostic[
            "worker_rank_timings"
        ][0]["worker_total_ms"] == 7.0
    else:
        assert executor.last_quantum_diagnostic is None
    if execution_trace:
        assert len(executor.last_execution_traces) == 1
    else:
        assert executor.last_execution_traces == ()

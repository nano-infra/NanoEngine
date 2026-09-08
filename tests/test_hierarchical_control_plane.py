from __future__ import annotations

import importlib
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed as dist

from nanodeploy._cpp import Sequence
from nanodeploy.engine.decode_coordinator import DecodeCoordinatorState
from nanodeploy.engine.deployment_manager import (
    DeploymentManager,
    _FrontendEventFlight,
)
from nanodeploy.engine.execution_boundary import ExecutionBoundaryRecorder
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    FirstScheduleEvent,
    FirstTokenEvent,
    FrontendEventBatch,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    RankLoad,
    validate_execution_trace_set,
)
from nanodeploy.engine.local_executor import LocalExecutor
from nanodeploy.engine.local_engine import (
    _IngressAdd,
    LocalEngineCore,
)
from nanodeploy.engine.topology import EngineTopology
from nanodeploy.engine.worker_transport import DecodeSuccess
from nanodeploy.router.request_router import RequestOwner, RequestRouter
from nanodeploy.router.admission_planner import AdmissionPlannerConfig


def test_hierarchical_runtime_env_propagates_actor_settings(monkeypatch):
    expected = {
        "SLIME_VISIBLE_DEVICES": (
            "mlx5_0,mlx5_1,mlx5_2,mlx5_3,"
            "mlx5_4,mlx5_5,mlx5_6,mlx5_7"
        ),
        "SLIME_GID_INDEX": "3",
        "SLIME_QP_NUM": "4",
        "NANODEPLOY_LOG_LEVEL": "WARNING",
    }
    for env_name, value in expected.items():
        monkeypatch.setenv(env_name, value)

    manager = object.__new__(DeploymentManager)
    manager.config = SimpleNamespace(ffn_ep=1, max_num_seqs=8)

    assert manager._runtime_env() == {"env_vars": expected}


def test_hierarchical_runtime_env_omits_unset_actor_settings(monkeypatch):
    for env_name in (
        "SLIME_VISIBLE_DEVICES",
        "SLIME_GID_INDEX",
        "SLIME_QP_NUM",
        "NANODEPLOY_LOG_LEVEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    manager = object.__new__(DeploymentManager)
    manager.config = SimpleNamespace(ffn_ep=1, max_num_seqs=8)

    assert manager._runtime_env() == {"env_vars": {}}


def test_frontend_event_poll_consumes_only_ready_engine_and_rearms(
    monkeypatch,
):
    refs = {0: [object(), object()], 1: [object()]}

    class RemoteDrain:
        def __init__(self, engine_id):
            self.engine_id = engine_id

        def remote(self):
            return refs[self.engine_id].pop(0)

    manager = object.__new__(DeploymentManager)
    manager.engines = {
        engine_id: SimpleNamespace(
            drain_frontend_events=RemoteDrain(engine_id)
        )
        for engine_id in (0, 1)
    }
    manager.config = SimpleNamespace(quantum_timeout_s=10.0)
    manager._frontend_event_flights = {}
    manager.close = lambda: pytest.fail("healthy event poll closed deployment")
    clock = {"now": 1.0}
    monkeypatch.setattr(
        "nanodeploy.engine.deployment_manager.monotonic",
        lambda: clock["now"],
    )

    first_ref = refs[0][0]
    second_ref = refs[1][0]
    batches_by_ref = {
        first_ref: FrontendEventBatch(
            engine_id=0,
            load=load_snapshot(0, 1),
        )
    }

    def fake_wait(pending_refs, *, num_returns, timeout):
        assert pending_refs == [first_ref, second_ref]
        assert num_returns == 2
        assert timeout == 0
        return [first_ref], [second_ref]

    monkeypatch.setattr(
        "nanodeploy.engine.deployment_manager.ray.wait", fake_wait
    )
    monkeypatch.setattr(
        "nanodeploy.engine.deployment_manager.ray.get",
        lambda ready_refs: [batches_by_ref[ref] for ref in ready_refs],
    )

    batches = manager.poll_ready_frontend_events()

    assert [batch.engine_id for batch in batches] == [0]
    assert manager._frontend_event_flights[1].ref is second_ref
    assert manager._frontend_event_flights[0].ref is not first_ref
    assert manager.frontend_event_flight_metrics() == {
        "pending": 2,
        "oldest_age_ms": 0.0,
    }


def test_frontend_event_poll_timeout_is_fail_stop(monkeypatch):
    manager = object.__new__(DeploymentManager)
    manager.engines = {0: object()}
    manager.config = SimpleNamespace(quantum_timeout_s=1.0)
    manager._frontend_event_flights = {
        0: _FrontendEventFlight(ref=object(), started_at=1.0)
    }
    closed = []
    manager.close = lambda: closed.append(True)
    monkeypatch.setattr(
        "nanodeploy.engine.deployment_manager.monotonic", lambda: 2.1
    )
    monkeypatch.setattr(
        "nanodeploy.engine.deployment_manager.ray.wait",
        lambda *_args, **_kwargs: ([], []),
    )

    with pytest.raises(TimeoutError, match=r"engines \[0\]"):
        manager.poll_ready_frontend_events()

    assert closed == [True]


def test_finish_event_excludes_unused_final_quantum_decode_slots():
    event = FinishEvent(
        request_id=1,
        generated_count=17,
        status="FINISHED",
        engine_id=0,
        first_forward_to_terminal_ms=1_000.0,
        final_quantum_execute_ms=160.0,
    )

    assert event.final_quantum_real_tokens == 1
    assert event.final_quantum_unused_decode_ms == 150.0
    assert event.first_forward_to_terminal_real_token_ms == 850.0

    full_quantum = FinishEvent(
        request_id=2,
        generated_count=32,
        status="FINISHED",
        engine_id=0,
        first_forward_to_terminal_ms=2_000.0,
    )
    assert full_quantum.final_quantum_real_tokens == 16
    assert full_quantum.final_quantum_unused_decode_ms == 0.0
    assert (
        full_quantum.first_forward_to_terminal_real_token_ms
        == 2_000.0
    )


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

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        self.aborts.append(request_id)
        return AbortResult(request_id=request_id, status="aborted")

    def clear_ingress_abort(self, request_id: int) -> None:
        return None

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


def add_request(
    router: RequestRouter,
    *,
    request_id: int,
    prompt_token_ids,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
) -> AddResult:
    prompt = tuple(prompt_token_ids)
    return router.add(
        request_id=request_id,
        prompt_len=len(prompt),
        num_tokens=len(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        ignore_eos=ignore_eos,
        sequence_payload=b"opaque-test-payload",
    )


def submit_request(
    router: RequestRouter,
    *,
    request_id: int,
    prompt_token_ids,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    sequence_payload: bytes = b"opaque-test-payload",
) -> int:
    prompt = tuple(prompt_token_ids)
    return router.submit_async(
        request_id=request_id,
        prompt_len=len(prompt),
        num_tokens=len(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        ignore_eos=ignore_eos,
        sequence_payload=sequence_payload,
    )


def local_add_pair(
    request_id: int,
    prompt_token_ids,
    *,
    max_tokens: int = 16,
    temperature: float = 0.1,
    ignore_eos: bool = True,
    wave_id: int = 0,
) -> tuple[AddCommand, Sequence]:
    prompt = list(prompt_token_ids)
    sequence = Sequence(
        prompt, temperature, max_tokens, ignore_eos
    )
    sequence.seq_id = request_id
    return (
        AddCommand(
            request_id=request_id,
            prompt_len=len(prompt),
            num_tokens=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            wave_id=wave_id,
            sequence_payload=b"opaque-test-payload",
        ),
        sequence,
    )


def route(router: RequestRouter, request_id: int) -> AddResult:
    return add_request(router,
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


def planner_config(
    *, attention_sp: int = 1, queue_capacity: int = 4096
) -> AdmissionPlannerConfig:
    return AdmissionPlannerConfig(
        attention_sp=attention_sp,
        kvcache_block_size=4,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        max_num_recv_seqs=32,
        reserved_blocks_per_req=1.0,
        segment_size=64,
        queue_capacity=queue_capacity,
    )


def load_snapshot(
    engine_id: int,
    free_blocks_min: int,
    *,
    waiting: int = 0,
    running: int = 0,
    pending_ingress: int = 0,
    quantum_id: int = 2,
    admission_version: int = 0,
    ingress_version: int = 0,
    capacity_epoch: int = 0,
    reserved_slots: int | None = None,
    active_dispatched_tokens: int = 0,
) -> LoadSnapshot:
    return LoadSnapshot(
        engine_id=engine_id,
        ready=True,
        waiting=waiting,
        running=running,
        free_blocks_min=free_blocks_min,
        wave_id=1,
        quantum_id=quantum_id,
        admission_version=admission_version,
        ingress_version=ingress_version,
        capacity_epoch=capacity_epoch,
        pending_ingress=pending_ingress,
        reserved_slots=(
            running + waiting + pending_ingress
            if reserved_slots is None
            else reserved_slots
        ),
        rank_loads=(
            RankLoad(
                global_rank=engine_id,
                sp_idx=0,
                tp_idx=0,
                master_batch_size=running,
                active_master_requests=running,
                free_blocks=free_blocks_min,
                total_blocks=max(256, free_blocks_min + 1),
                master_assignments=running,
                mastered_decode_tokens=0,
                active_dispatched_tokens=active_dispatched_tokens,
                control_dummy_blocks=1,
            ),
        ),
    )


def multi_rank_load_snapshot(
    engine_id: int,
    *,
    free_blocks: tuple[int, ...],
    master_counts: tuple[int, ...],
    receiver_counts: tuple[int, ...],
) -> LoadSnapshot:
    running = sum(master_counts)
    return LoadSnapshot(
        engine_id=engine_id,
        ready=True,
        waiting=0,
        running=running,
        free_blocks_min=min(free_blocks),
        wave_id=1,
        quantum_id=2,
        reserved_slots=running,
        rank_loads=tuple(
            RankLoad(
                global_rank=engine_id * len(free_blocks) + sp_idx,
                sp_idx=sp_idx,
                tp_idx=0,
                master_batch_size=master_counts[sp_idx],
                active_master_requests=master_counts[sp_idx],
                active_receiver_requests=receiver_counts[sp_idx],
                free_blocks=rank_free_blocks,
                total_blocks=256,
                master_assignments=master_counts[sp_idx],
                mastered_decode_tokens=0,
                control_dummy_blocks=1,
            )
            for sp_idx, rank_free_blocks in enumerate(free_blocks)
        ),
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
    planned_batches: list[tuple[AdmissionReservation, ...]] = field(
        default_factory=list
    )
    admission_version: int = 0
    ingress_version: int = 0

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
            self.ingress_version += 1
            handle["ack"] = IngressAck(
                request_id=command.request_id,
                engine_id=self.engine_id,
                enqueued=True,
                ingress_version=self.ingress_version,
            )
        return handle

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ):
        self.planned_batches.append(reservations)
        acks = []
        for command in commands:
            reason = (
                self.enqueue_reasons.pop(0)
                if self.enqueue_reasons
                else None
            )
            self.commands.append(command)
            if reason is None:
                self.ingress_version += 1
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=reason is None,
                    reason=reason,
                    ingress_version=(
                        self.ingress_version
                        if reason is None
                        else None
                    ),
                )
            )
        handle = {"ready": False, "acks": tuple(acks)}
        self.handles.append(handle)
        return handle

    def enqueue_batch_async(self, commands: tuple[AddCommand, ...]):
        acks = []
        for command in commands:
            reason = (
                self.enqueue_reasons.pop(0)
                if self.enqueue_reasons
                else None
            )
            self.commands.append(command)
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=reason is None,
                    reason=reason,
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
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )

    for request_id in (1, 2, 3, 4):
        submit_request(router,
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


def test_router_least_batch_balances_count_not_prompt_work():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )

    for request_id, prompt_len in enumerate((64, 1, 64, 1), start=1):
        submit_request(
            router,
            request_id=request_id,
            prompt_token_ids=tuple(range(prompt_len)),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1, 3]
    assert [command.request_id for command in engines[1].commands] == [2, 4]
    assert [
        sum(command.num_tokens for command in engine.commands)
        for engine in engines.values()
    ] == [128, 2]


def test_router_least_projected_load_balances_prompt_work_and_batch():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_projected_load",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )

    for request_id, prompt_len in enumerate((64, 1, 64, 1), start=1):
        submit_request(
            router,
            request_id=request_id,
            prompt_token_ids=tuple(range(prompt_len)),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1, 4]
    assert [command.request_id for command in engines[1].commands] == [2, 3]
    assert [
        sum(command.num_tokens for command in engine.commands)
        for engine in engines.values()
    ] == [65, 65]


def test_router_least_projected_load_uses_live_context():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_projected_load",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (
            load_snapshot(
                0,
                100,
                running=1,
                active_dispatched_tokens=100,
            ),
            load_snapshot(
                1,
                100,
                running=1,
                active_dispatched_tokens=10,
            ),
        )
    )

    submit_request(
        router,
        request_id=1,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert engines[0].commands == []
    assert [command.request_id for command in engines[1].commands] == [1]
    metrics = router.admission_metrics()["per_engine"]
    assert metrics["0"]["projected_attention_work"] == 100
    assert metrics["1"]["projected_attention_work"] == 11


def test_router_least_projected_load_does_not_make_prompt_strictly_primary():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_projected_load",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (
            load_snapshot(
                0,
                100,
                running=1,
                active_dispatched_tokens=100,
            ),
            load_snapshot(
                1,
                100,
                running=10,
                active_dispatched_tokens=90,
            ),
        )
    )

    submit_request(
        router,
        request_id=1,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1]
    assert engines[1].commands == []


def test_router_least_projected_load_score_ignores_max_tokens():
    def route_with(max_tokens: tuple[int, ...]) -> dict[int, list[int]]:
        engines = {
            0: FakeAsyncEngine(0),
            1: FakeAsyncEngine(1),
        }
        router = RequestRouter(
            engines,
            router_policy="least_projected_load",
            admission_planner_config=planner_config(),
        )
        router.record_loads(
            (load_snapshot(0, 100), load_snapshot(1, 100))
        )
        for request_id, (prompt_len, output_limit) in enumerate(
            zip((64, 1, 64, 1), max_tokens, strict=True), start=1
        ):
            submit_request(
                router,
                request_id=request_id,
                prompt_token_ids=tuple(range(prompt_len)),
                max_tokens=output_limit,
                temperature=0.1,
                ignore_eos=True,
            )
        assert router.poll_ingress_acks() == ()
        return {
            engine_id: [
                command.request_id for command in engine.commands
            ]
            for engine_id, engine in engines.items()
        }

    expected = {0: [1, 4], 1: [2, 3]}
    assert route_with((16, 16, 16, 16)) == expected
    assert route_with((1, 64, 2, 32)) == expected


def test_router_least_projected_load_matches_count_for_equal_prompts():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_projected_load",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )
    for request_id in range(1, 5):
        submit_request(
            router,
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1, 3]
    assert [command.request_id for command in engines[1].commands] == [2, 4]


def test_router_least_projected_load_reconciles_tentative_work_to_snapshot():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_projected_load",
        admission_planner_config=planner_config(),
    )
    router.record_loads((load_snapshot(0, 100),))
    submit_request(
        router,
        request_id=1,
        prompt_token_ids=tuple(range(64)),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    metrics = router.admission_metrics()["per_engine"]["0"]
    assert metrics["tentative_admissions"] == 1
    assert metrics["projected_attention_work"] == 64

    engine.handles[0]["ready"] = True
    ack = router.poll_ingress_acks()
    assert len(ack) == 1 and ack[0].enqueued
    router.record_add_results(
        (AddResultEvent(1, 0, True, admission_version=1),)
    )
    router.record_loads(
        (
            load_snapshot(
                0,
                84,
                running=1,
                admission_version=1,
                ingress_version=1,
                reserved_slots=1,
                active_dispatched_tokens=64,
            ),
        )
    )

    metrics = router.admission_metrics()["per_engine"]["0"]
    assert metrics["tentative_admissions"] == 0
    assert metrics["attention_work_snapshot"] == 64
    assert metrics["projected_attention_work"] == 64


def test_router_least_batch_does_not_double_count_staged_snapshot_slot():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=planner_config(queue_capacity=2),
    )
    router.record_loads((load_snapshot(0, 100),))
    submit_request(
        router,
        request_id=1,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    engine.handles[0]["ready"] = True
    receipt = router.poll_ingress_acks()
    assert len(receipt) == 1
    assert receipt[0].ingress_version == 1
    assert router.owner(1) == RequestOwner(OwnerState.PENDING_ADD, 0)

    router.record_loads(
        (
            load_snapshot(
                0,
                100,
                pending_ingress=1,
                ingress_version=1,
                reserved_slots=1,
            ),
        )
    )
    submit_request(
        router,
        request_id=2,
        prompt_token_ids=(2,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engine.commands] == [1, 2]


def test_router_pending_add_abort_terminal_releases_staged_owner():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads((load_snapshot(0, 100),))
    submit_request(
        router,
        request_id=3,
        prompt_token_ids=(3,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()
    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued
    assert router.owner(3) == RequestOwner(OwnerState.PENDING_ADD, 0)

    event = FinishEvent(3, 0, "ABORTED", 0)
    assert router.record_finish_events((event,)) == (event,)
    assert router.owner(3) is None
    assert router.terminal_event(3) == event
    assert router.is_idle


def test_router_least_batch_uses_live_running_plus_tentative_admissions():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (
            load_snapshot(0, 100, running=3),
            load_snapshot(1, 100, running=1),
        )
    )

    for request_id in (1, 2, 3):
        submit_request(router,
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
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )
    for request_id in range(5):
        submit_request(router,
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


def test_router_capacity_planner_does_not_rpc_past_unfit_fifo_head():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 4), load_snapshot(1, 4))
    )
    submit_request(router,
        request_id=1,
        prompt_token_ids=tuple(range(16)),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    submit_request(router,
        request_id=2,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert engines[0].commands == []
    assert engines[1].commands == []
    assert router.pending_global_count == 2
    assert router.admission_metrics()["global_retries"] == 0

    router.record_loads(
        (
            load_snapshot(0, 6, quantum_id=3, capacity_epoch=1),
            load_snapshot(1, 6, quantum_id=3, capacity_epoch=1),
        )
    )
    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engines[0].commands] == [1]
    assert [command.request_id for command in engines[1].commands] == [2]
    assert engines[0].planned_batches[0][0].request_id == 1
    assert sum(
        engines[0].planned_batches[0][0].dispatched_tokens
    ) == 16
    assert engines[1].planned_batches[0][0].request_id == 2


def test_router_capacity_planner_reserves_each_planned_batch_request():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    # Each one-token prompt needs one KV block plus one reserved block.
    # Earlier requests in this frontend batch are already reflected in the
    # shadow's master count, so four free blocks fit both requests exactly.
    router.record_loads((load_snapshot(0, 4),))
    for request_id in (1, 2):
        submit_request(router,
            request_id=request_id,
            prompt_token_ids=(request_id,),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
        )

    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engine.commands] == [1, 2]
    assert router.owner(2) == RequestOwner(OwnerState.PENDING_INGRESS, 0)
    assert [
        reservation.request_id
        for reservation in engine.planned_batches[0]
    ] == [1, 2]
    assert router.admission_metrics()["global_retries"] == 0


def test_router_capacity_planner_honors_dp_aggregated_receiver_limits():
    engines = {
        0: FakeAsyncEngine(0),
        1: FakeAsyncEngine(1),
    }
    config = planner_config(attention_sp=2)
    config = replace(config, max_num_recv_seqs=1)
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=config,
    )
    router.record_loads(
        (
            multi_rank_load_snapshot(
                0,
                free_blocks=(100, 100),
                master_counts=(0, 1),
                receiver_counts=(0, 1),
            ),
            multi_rank_load_snapshot(
                1,
                free_blocks=(100, 100),
                master_counts=(0, 1),
                receiver_counts=(0, 0),
            ),
        )
    )
    submit_request(router,
        request_id=1,
        prompt_token_ids=tuple(range(65)),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )

    assert router.poll_ingress_acks() == ()
    assert engines[0].commands == []
    assert [command.request_id for command in engines[1].commands] == [1]
    assert router.admission_metrics()["global_retries"] == 0


def test_local_engine_consensus_uses_one_collective_for_global_or(monkeypatch):
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(attention_dp=2)
    engine._wave_id = 7
    engine._quantum_id = 11
    calls = []
    clock = iter((10.0, 10.002))
    monkeypatch.setattr(
        "nanodeploy.engine.local_engine.perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(
        "nanodeploy.engine.local_engine.wall_time_ns",
        lambda: 1_003_000_000,
    )

    class FakeWork:
        def __init__(self, tensor):
            self.tensor = tensor

        def wait(self):
            calls.append("wait")
            self.tensor.copy_(
                self.tensor.new_tensor(
                    [
                        7,
                        -7,
                        11,
                        -11,
                        1,
                        0,
                        1_003_000_000,
                        -1_000_000_000,
                    ]
                )
            )

    def fake_all_reduce(tensor, *, op, async_op):
        calls.append((tensor.tolist(), op, async_op))
        return FakeWork(tensor)

    monkeypatch.setattr(
        "nanodeploy.engine.local_engine.dist.all_reduce", fake_all_reduce
    )

    pending = engine._start_consensus(
        local_unfinished=False,
        wave_id=7,
        quantum_id=11,
    )
    assert calls == [
        (
            [
                7,
                -7,
                11,
                -11,
                0,
                0,
                1_003_000_000,
                -1_003_000_000,
            ],
            dist.ReduceOp.MAX,
            True,
        )
    ]

    assert engine._finish_consensus(pending)
    assert calls[-1] == "wait"
    assert pending.leader_arrival_skew_ms == 3.0
    assert pending.leader_rendezvous_ms == pytest.approx(2.0)
    assert pending.late_participant_collective_ms == pytest.approx(2.0)


def test_local_engine_consensus_detects_mismatched_ids_with_one_collective(
    monkeypatch,
):
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(attention_dp=2)
    engine._wave_id = 7
    engine._quantum_id = 11
    calls = []

    class FakeWork:
        def __init__(self, tensor):
            self.tensor = tensor

        def wait(self):
            calls.append("wait")
            self.tensor.copy_(
                self.tensor.new_tensor(
                    [8, -7, 12, -11, 1, 0, 2_000, -1_000]
                )
            )

    def fake_all_reduce(tensor, *, op, async_op):
        assert op == dist.ReduceOp.MAX
        assert async_op
        calls.append("start")
        return FakeWork(tensor)

    monkeypatch.setattr(
        "nanodeploy.engine.local_engine.dist.all_reduce", fake_all_reduce
    )

    with pytest.raises(
        RuntimeError,
        match=r"min=\[7, 11, 0\], max=\[8, 12, 1\]",
    ):
        engine._finish_consensus(
            engine._start_consensus(
                local_unfinished=False,
                wave_id=7,
                quantum_id=11,
            )
        )
    assert calls == ["start", "wait"]


def test_local_engine_overlaps_consensus_with_plan_and_waits_before_execute():
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    order = []
    frozen_load = load_snapshot(0, 9, quantum_id=11)
    post_load = replace(frozen_load, quantum_id=12)
    batch = SimpleNamespace(
        frozen_load_snapshot=frozen_load,
        engine_has_real=True,
    )

    class FakeScheduler:
        last_itl_token_slots = 0

        @staticmethod
        def admit():
            order.append("admit")

        @staticmethod
        def is_finished():
            return False

        @staticmethod
        def plan_decode(*, wave_id, quantum_id):
            assert (wave_id, quantum_id) == (1, 11)
            order.append("plan")
            return batch

        @staticmethod
        def mark_first_forward_started(planned_batch):
            assert planned_batch is batch
            order.append("mark")
            return ()

        @staticmethod
        def postprocess(planned_batch, results, *, execute_latency_ms):
            assert planned_batch is batch
            assert results == []
            assert execute_latency_ms >= 0
            order.append("postprocess")
            engine._stop = True
            return ()

        @staticmethod
        def drain_first_token_events():
            return ()

    class FakeExecutor:
        last_quantum_diagnostic = {
            "engine_id": 0,
            "wave_id": 1,
            "quantum_id": 11,
        }

        @staticmethod
        def activate_worker_transport(timeout):
            assert timeout == 1.0
            order.append("activate")

        @staticmethod
        def run(planned_batch, *, timeout):
            assert planned_batch is batch
            assert timeout == 2.0
            order.append("execute")
            return []

        @staticmethod
        def shutdown_worker_transport(*, timeout, failed):
            assert (timeout, failed) == (2.0, False)
            order.append("shutdown")
            return ()

    def refresh(snapshot=None):
        if snapshot is None:
            order.append("refresh_post")
            return post_load
        assert snapshot is frozen_load
        order.append("refresh_frozen")
        return snapshot

    engine.config = SimpleNamespace(
        attention_dp=2,
        startup_timeout_s=1.0,
        quantum_timeout_s=2.0,
    )
    engine.engine_id = 0
    engine.topology = SimpleNamespace(global_dp_idx=1)
    engine.scheduler = FakeScheduler()
    engine.executor = FakeExecutor()
    engine._worker_transport_ready = threading.Event()
    engine._failure = None
    engine._stop = False
    engine._wave_running = True
    engine._wave_id = 1
    engine._quantum_id = 11
    engine._state_cv = threading.Condition()
    engine._abort_commands = queue.Queue()
    engine._normal_commands = queue.Queue()
    engine._drain_ingress = lambda: None
    engine._refresh_cached_load = refresh
    engine._start_consensus = lambda *_args, **_kwargs: (
        order.append("consensus_start")
        or SimpleNamespace(
            work=object(),
            arrival_unix_ns=1_000_000_000,
            leader_arrival_skew_ms=0.0,
            leader_rendezvous_ms=0.0,
            late_participant_collective_ms=None,
        )
    )
    engine._finish_consensus = lambda _pending: (
        order.append("consensus_wait") or True
    )
    engine._publish_first_schedule_events = lambda _events: None
    engine._publish_first_token_events = lambda _events: None
    engine._publish_events = lambda _events: None
    engine._admission_latency_ms_total = 0.0
    engine._schedule_latency_ms_total = 0.0
    engine._coordination_latency_ms_total = 0.0
    engine._execute_latency_ms_total = 0.0
    engine._postprocess_latency_ms_total = 0.0
    engine._decode_quantum_count = 0
    engine._decode_itl_samples = []
    engine._decode_itl_ms_weighted_total = 0.0
    engine._decode_itl_token_count = 0
    engine._quantum_diagnostics_enabled = True
    engine._quantum_diagnostics = []
    engine._quantum_diagnostics_lock = threading.Lock()

    engine._event_loop()

    assert order == [
        "activate",
        "admit",
        "consensus_start",
        "plan",
        "refresh_frozen",
        "consensus_wait",
        "mark",
        "execute",
        "postprocess",
        "refresh_post",
        "shutdown",
    ]
    assert engine._failure is None
    assert engine._decode_quantum_count == 1
    assert len(engine._quantum_diagnostics) == 1
    sample = engine._quantum_diagnostics[0]
    assert sample["schema_version"] == 3
    assert sample["ingress_drain_ms"] >= 0.0
    assert sample["consensus_exposed_wait_ms"] >= 0.0
    assert sample["leader_arrival_skew_ms"] == 0.0
    assert sample["late_participant_collective_ms"] is None


def test_local_engine_drains_frontend_events_in_one_batch():
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.engine_id = 0
    engine._failure = None
    engine.executor = SimpleNamespace(check_worker_liveness=lambda: None)
    engine._frontend_server = SimpleNamespace(raise_if_failed=lambda: None)
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


def test_local_engine_transport_startup_failure_releases_ready_barrier():
    class FakeExecutor:
        shutdown_call = None

        @staticmethod
        def activate_worker_transport(_timeout):
            raise RuntimeError("handshake failed")

        def shutdown_worker_transport(self, *, timeout, failed):
            self.shutdown_call = (timeout, failed)
            return ()

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.engine_id = 0
    engine.config = SimpleNamespace(
        startup_timeout_s=3.0,
        quantum_timeout_s=4.0,
    )
    engine.executor = FakeExecutor()
    engine._worker_transport_ready = threading.Event()
    engine._failure = None
    engine._stop = False
    engine._state_cv = threading.Condition()
    engine._abort_commands = queue.Queue()
    engine._normal_commands = queue.Queue()

    engine._event_loop()

    assert engine._worker_transport_ready.is_set()
    assert engine._failure == "RuntimeError: handshake failed"
    assert engine._stop
    assert engine.executor.shutdown_call == (4.0, True)


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
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )
    submit_request(router,
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

    router.record_add_results(
        (AddResultEvent(5, 0, True, admission_version=1),)
    )
    terminal_events = router.record_finish_events(())
    assert len(terminal_events) == 1
    assert terminal_events[0].request_id == 5
    assert terminal_events[0].first_forward_to_terminal_ms == 750.0
    assert router.terminal_event(5) == terminal_events[0]
    assert router.is_idle


def test_router_least_batch_buffers_lifecycle_before_staged_receipt():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads((load_snapshot(0, 10),))
    submit_request(router,
        request_id=6,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()

    add_event = AddResultEvent(6, 0, True, admission_version=1)
    finish_event = FinishEvent(
        6,
        16,
        "FINISHED",
        0,
        first_forward_to_terminal_ms=500.0,
    )
    assert router.record_add_results((add_event,)) == ()
    assert router.record_finish_events((finish_event,)) == ()

    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued
    assert router.record_add_results(()) == (add_event,)
    assert router.owner(6) == RequestOwner(OwnerState.OWNED, 0)
    assert router.record_finish_events(()) == (finish_event,)
    assert router.terminal_event(6) == finish_event
    assert router.is_idle


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
    first = add_request(router,
        request_id=1,
        prompt_token_ids=(1, 2),
        max_tokens=17,
        temperature=0.1,
        ignore_eos=True,
    )
    router.record_loads(
        (load_snapshot(0, 20), load_snapshot(1, 12))
    )
    second = add_request(router,
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
    third = add_request(router,
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

    submit_request(router,
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
    submit_request(router,
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
    router = RequestRouter(
        engines,
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 100), load_snapshot(1, 100))
    )

    payload = bytes(bytearray(b"one-serialized-Sequence"))
    submit_request(router,
        request_id=40,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
        sequence_payload=payload,
    )
    assert router.pending_ingress_count == 1
    assert router.owner(40).state == OwnerState.PENDING_GLOBAL
    assert router.poll_ingress_acks() == ()
    assert router.owner(40).engine_id == 0

    engines[0].handles[0]["ready"] = True
    assert router.poll_ingress_acks() == ()
    assert router.owner(40).engine_id == 1
    assert engines[0].commands[0].sequence_payload is payload
    assert engines[1].commands[0].sequence_payload is payload

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
        == 1
    )

    result = AddResultEvent(40, 1, True, admission_version=1)
    assert router.record_add_results((result,)) == (result,)
    assert router.owner(40) == RequestOwner(OwnerState.OWNED, 1)
    assert (
        router.admission_metrics()["per_engine"]["1"][
            "tentative_admissions"
        ]
        == 0
    )


def test_router_buffers_add_result_observed_before_ingress_ack():
    engine = FakeAsyncEngine(0)
    router = RequestRouter(
        {0: engine},
        admission_planner_config=planner_config(),
    )
    router.record_loads((load_snapshot(0, 100),))
    submit_request(router,
        request_id=41,
        prompt_token_ids=(1, 2),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()

    event = AddResultEvent(41, 0, True, admission_version=1)
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

        def add(self, command, sequence):
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
    engine._ingress_head = None
    engine._ingress_replay = deque()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = set()
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._ingress_version = 0
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
    engine._local_transient_retries = 0
    engine._staged_ingress_depth_max = 0
    engine._planned_commit_attempts = 0

    command_sequences = [
        local_add_pair(request_id, (1, 2))
        for request_id in (50, 51, 52)
    ]
    commands = [item[0] for item in command_sequences]
    sequences = [item[1] for item in command_sequences]
    assert engine.enqueue_add(commands[0], sequences[0]).enqueued
    assert engine.enqueue_add(commands[1], sequences[1]).enqueued
    assert not engine.enqueue_add(commands[0], sequences[0]).enqueued
    full = engine.enqueue_add(commands[2], sequences[2])
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
    assert engine.enqueue_add(commands[2], sequences[2]).enqueued
    assert engine.submit_abort(52).status == "abort_pending"
    engine._drain_ingress()
    assert engine.drain_add_results() == ()
    assert engine.drain_events() == (
        FinishEvent(50, 16, "FINISHED", 0),
        FinishEvent(52, 0, "ABORTED", 0),
    )
    assert engine._reserved_slots == 1


def test_local_engine_planned_stage_uses_one_lock_scope_and_one_wakeup():
    class RemoteRecorder:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        attention_dp=2,
        attention_sp=1,
        hierarchical_queue_capacity=8,
    )
    engine.engine_id = 0
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = set()
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._ingress_version = 0
    engine._staged_ingress_depth_max = 0
    engine._state_cv = threading.Condition()
    engine._wave_id = 7
    first_request = RemoteRecorder()
    engine._coordinator = SimpleNamespace(first_request=first_request)
    command_sequences = tuple(
        local_add_pair(request_id, (1,), wave_id=7)
        for request_id in (1, 2, 3)
    )
    commands = tuple(item[0] for item in command_sequences)
    sequences = tuple(item[1] for item in command_sequences)
    reservations = tuple(
        AdmissionReservation(
            request_id=command.request_id,
            engine_id=0,
            master_sp_idx=0,
            dispatched_tokens=(command.num_tokens,),
        )
        for command in commands
    )

    acks = engine.admit_add_batch(commands, reservations, sequences)

    assert [ack.request_id for ack in acks] == [1, 2, 3]
    assert all(ack.enqueued for ack in acks)
    assert [ack.ingress_version for ack in acks] == [1, 2, 3]
    assert all(ack.admission_version is None for ack in acks)
    assert engine._reserved_slots == 3
    assert engine._ingress_adds.qsize() == 3
    assert first_request.calls == [(0, 7)]

    invalid_command, invalid_sequence = local_add_pair(4, (4,), wave_id=7)
    invalid = engine.admit_add_batch(
        (invalid_command,),
        (AdmissionReservation(4, 0, 1, (1,)),),
        (invalid_sequence,),
    )[0]
    assert not invalid.enqueued
    assert invalid.reason == "invalid_admission_reservation"
    assert engine._ingress_version == 3
    assert first_request.calls == [(0, 7)]


def test_local_engine_transient_planned_head_retries_without_reordering():
    class FakeScheduler:
        def __init__(self):
            self.calls = []

        def commit_planned_batch(self, commands, reservations, sequences):
            request_ids = tuple(command.request_id for command in commands)
            self.calls.append(request_ids)
            if len(self.calls) == 1:
                return tuple(
                    AddResult(
                        request_id,
                        False,
                        engine_id=0,
                        reason="admission_state_mismatch",
                    )
                    for request_id in request_ids
                )
            return tuple(
                AddResult(request_id, True, engine_id=0)
                for request_id in request_ids
            )

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        max_ingress_batch_requests=8,
        max_ingress_drain_ms=0.0,
    )
    engine.engine_id = 0
    engine.scheduler = FakeScheduler()
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_head = None
    engine._ingress_replay = deque()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = {1, 2}
    engine._ingress_pending_ids = {1, 2}
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 2
    engine._admission_version = 0
    engine._capacity_epoch = 0
    engine._events_lock = threading.Lock()
    engine._add_result_events = deque()
    engine._terminal_events = deque()
    engine._ingress_queue_delay_ms_total = 0.0
    engine._scheduler_add_ms_total = 0.0
    engine._local_transient_retries = 0
    engine._planned_commit_attempts = 0
    for request_id in (1, 2):
        command, sequence = local_add_pair(request_id, (request_id,))
        engine._ingress_adds.put_nowait(
            _IngressAdd(
                command,
                sequence,
                reservation=AdmissionReservation(
                    request_id=request_id,
                    engine_id=0,
                    master_sp_idx=0,
                    dispatched_tokens=(1,),
                ),
                ingress_version=request_id,
            )
        )

    engine._drain_ingress()
    assert engine.scheduler.calls == [(1, 2)]
    assert engine._ingress_head.command.request_id == 1
    assert engine._ingress_adds.qsize() == 0
    assert engine.drain_add_results() == ()

    engine._drain_ingress()
    assert engine.scheduler.calls == [(1, 2), (1, 2)]
    assert engine._ingress_head is None
    assert [event.request_id for event in engine.drain_add_results()] == [1, 2]
    assert engine._local_transient_retries == 1
    assert engine._planned_commit_attempts == 4


def test_local_engine_abort_during_planned_commit_wins_ownership_handoff():
    class FakeScheduler:
        def __init__(self):
            self._events = ()

        def commit_planned_batch(self, commands, reservations, sequences):
            request_id = commands[0].request_id
            with engine._ingress_lock:
                engine._cancelled_ingress_ids.add(request_id)
            return (AddResult(request_id, True, engine_id=0),)

        def abort(self, request_id):
            self._events = (FinishEvent(request_id, 0, "ABORTED", 0),)
            return AbortResult(request_id, "aborted")

        def drain_terminal_events(self):
            events = self._events
            self._events = ()
            return events

    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        max_ingress_batch_requests=8,
        max_ingress_drain_ms=0.0,
    )
    engine.engine_id = 0
    engine.scheduler = FakeScheduler()
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_head = None
    engine._ingress_replay = deque()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = {1}
    engine._ingress_pending_ids = {1}
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 1
    engine._admission_version = 0
    engine._capacity_epoch = 0
    engine._events_lock = threading.Lock()
    engine._add_result_events = deque()
    engine._terminal_events = deque()
    engine._ingress_queue_delay_ms_total = 0.0
    engine._scheduler_add_ms_total = 0.0
    engine._local_transient_retries = 0
    engine._planned_commit_attempts = 0
    command, sequence = local_add_pair(1, (1,))
    engine._ingress_adds.put_nowait(
        _IngressAdd(
            command,
            sequence,
            reservation=AdmissionReservation(1, 0, 0, (1,)),
            ingress_version=1,
        )
    )

    engine._drain_ingress()

    assert engine.drain_add_results() == ()
    assert engine.drain_events() == (FinishEvent(1, 0, "ABORTED", 0),)
    assert engine._reserved_slots == 0
    assert engine._cancelled_ingress_ids == set()
    assert engine._admission_version == 1


def test_local_engine_future_ingress_abort_tombstone_wins_enqueue_race():
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        attention_dp=1,
        hierarchical_queue_capacity=8,
    )
    engine.engine_id = 0
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = set()
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._ingress_version = 0
    engine._staged_ingress_depth_max = 0
    engine._state_cv = threading.Condition()
    engine._wave_id = 0
    engine._wave_running = False
    engine._submit = lambda *_args, **_kwargs: AbortResult(
        request_id=77, status="not_found"
    )
    command, sequence = local_add_pair(77, (1,))

    abort = engine.submit_abort(77, allow_future_ingress=True)
    ack = engine.enqueue_add_batch((command,), (sequence,))[0]

    assert abort.status == "abort_pending"
    assert not ack.enqueued and ack.reason == "aborted"
    assert engine._ingress_adds.empty()
    assert engine._reserved_slots == 0
    assert engine._cancelled_ingress_ids == set()


def test_router_least_batch_reconciles_abort_after_staged_receipt():
    @dataclass
    class AbortRaceEngine(FakeAsyncEngine):
        abort_modes: list[bool] = field(default_factory=list)

        def abort(
            self,
            request_id: int,
            *,
            allow_future_ingress: bool = False,
        ) -> AbortResult:
            self.abort_modes.append(allow_future_ingress)
            return AbortResult(
                request_id=request_id,
                status=("abort_pending" if allow_future_ingress else "aborted"),
            )

    engine = AbortRaceEngine(0)
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads((load_snapshot(0, 10),))
    submit_request(router,
        request_id=1,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()

    assert router.abort(1).status == "abort_pending"
    assert engine.abort_modes == [True]
    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued
    assert engine.abort_modes == [True, False]
    assert router.owner(1) == RequestOwner(OwnerState.PENDING_ADD, 0)


def test_router_least_batch_does_not_retry_aborted_queue_full_receipt():
    @dataclass
    class AbortRaceEngine(FakeAsyncEngine):
        cleared_ingress_aborts: list[int] = field(default_factory=list)

        def abort(
            self,
            request_id: int,
            *,
            allow_future_ingress: bool = False,
        ) -> AbortResult:
            return AbortResult(
                request_id=request_id,
                status=("abort_pending" if allow_future_ingress else "aborted"),
            )

        def clear_ingress_abort(self, request_id: int) -> None:
            self.cleared_ingress_aborts.append(request_id)

    engines = {
        0: AbortRaceEngine(0, enqueue_reasons=["queue_full"]),
        1: AbortRaceEngine(1),
    }
    router = RequestRouter(
        engines,
        router_policy="least_batch",
        admission_planner_config=planner_config(),
    )
    router.record_loads(
        (load_snapshot(0, 10), load_snapshot(1, 9))
    )
    submit_request(router,
        request_id=1,
        prompt_token_ids=(1,),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
    )
    assert router.poll_ingress_acks() == ()
    assert router.abort(1).status == "abort_pending"
    engines[0].handles[0]["ready"] = True

    ack = router.poll_ingress_acks()

    assert len(ack) == 1 and ack[0].reason == "aborted"
    assert engines[1].commands == []
    assert engines[0].cleared_ingress_aborts == [1]
    assert router.owner(1) is None
    assert router.terminal_event(1).status == "ABORTED"
    assert router.abort(1).status == "already_terminal"


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
                    list(range(100, 116)),
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
                        "real_batch_size": 2,
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
        second_real_sequence = FakeSequence(9)
        per_rank_sequences = {
            0: [real_sequence, control_dummy, second_real_sequence]
        }
        frozen_mastered_sequences = {
            0: (real_sequence, control_dummy, second_real_sequence)
        }
        frozen_real_row_indices = {0: (0, 2)}

        @staticmethod
        def is_control_dummy(sequence):
            return sequence.seq_id == -1

        @staticmethod
        def expected_request_ids(_global_rank):
            return (7, 9)

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
            "real_batch_size": 2,
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
    assert results[0].mastered_request_ids == (7, 9)
    assert results[0].sampled_token_ids == (
        tuple(range(16)),
        tuple(range(100, 116)),
    )
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


def test_local_executor_zmq_branch_avoids_per_quantum_ray_calls(monkeypatch):
    class FakeEndpoint:
        def __init__(self, *_args, **_kwargs):
            self.sent = None

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
            self.configure_zmq_worker_transport = RemoteMethod(None)
            self.run_zmq_loop = RemoteMethod("persistent-loop")

    class FakeZmqServer:
        instance = None

        def __init__(self):
            self.deployment_epoch = "epoch"
            self.commands = None
            self.activated = None
            self.closed = None
            type(self).instance = self

        @classmethod
        def create_ipc(cls, **_kwargs):
            return cls()

        def worker_config(self, global_rank, **_kwargs):
            return SimpleNamespace(global_rank=global_rank)

        def activate(self, timeout, *, liveness_check):
            self.activated = (timeout, liveness_check)

        def send_decode_commands(self, commands, *, deadline):
            self.commands = (dict(commands), deadline)

        @staticmethod
        def receive_decode_results(*, deadline):
            assert deadline > 0
            return (
                DecodeSuccess(
                    protocol_version=1,
                    deployment_epoch="epoch",
                    engine_id=0,
                    global_rank=0,
                    wave_id=1,
                    quantum_id=0,
                    token_rows=[list(range(16)), [0] * 16],
                    worker_end_time=time.time(),
                ),
            )

        def mark_failed(self):
            raise AssertionError("normal ZMQ run must not fail")

        def close(self, *, graceful, timeout_s):
            self.closed = (graceful, timeout_s)
            return ()

        def cleanup_unbound(self):
            raise AssertionError("normal ZMQ startup must not clean early")

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
        frozen_mastered_sequences = {
            0: (real_sequence, control_dummy)
        }
        frozen_real_row_indices = {0: (0,)}

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
        "nanodeploy.engine.local_executor.ZmqWorkerServer",
        FakeZmqServer,
    )
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.ray.get",
        lambda refs, timeout: refs,
    )
    config = SimpleNamespace(
        optimize_decode_block_table=True,
        hierarchical_execution_trace=False,
        hierarchical_quantum_diagnostics=False,
        hierarchical_result_fastpath=True,
        hierarchical_worker_transport="zmq",
        quantum_timeout_s=2.0,
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

    executor.prepare_worker_transport(timeout=3.0)
    executor.activate_worker_transport(timeout=3.0)
    results = executor.run(FakeBatch(), timeout=2.0)
    shutdown_errors = executor.shutdown_worker_transport(
        timeout=2.0, failed=False
    )

    assert shutdown_errors == ()
    assert worker.configure_zmq_worker_transport.calls == [
        {"transport_config": SimpleNamespace(global_rank=0)}
    ]
    assert worker.run_zmq_loop.calls == [{}]
    assert results[0].mastered_request_ids == (7,)
    assert results[0].sampled_token_ids == (tuple(range(16)),)
    server = FakeZmqServer.instance
    assert server is not None
    commands, _deadline = server.commands
    assert tuple(commands) == (0,)
    assert commands[0].wave_id == 1
    assert commands[0].quantum_id == 0
    assert executor.endpoint.sent[1] is False
    boundary = executor.execution_boundary_metrics()
    assert boundary["worker_result_wait_latency_ms_mean"] >= 0
    assert "ray_get_latency_ms_mean" not in boundary
    assert executor.ray_get_latency_ms_total == 0
    assert executor.worker_result_wait_latency_ms_total >= 0

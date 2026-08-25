from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

from nanodeploy._cpp import Sequence
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    RankLoad,
    ResourceReleaseEvent,
    TokenCommitEvent,
)
from nanodeploy.engine.local_engine import LocalEngineCore, _IngressAdd
from nanodeploy.router.admission_planner import AdmissionPlannerConfig
from nanodeploy.router.request_router import RequestOwner, RequestRouter


def _planner_config(*, queue_capacity: int = 8) -> AdmissionPlannerConfig:
    return AdmissionPlannerConfig(
        attention_sp=1,
        kvcache_block_size=4,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        max_num_recv_seqs=32,
        reserved_blocks_per_req=1.0,
        segment_size=64,
        queue_capacity=queue_capacity,
    )


def _load(
    *,
    ingress_version: int = 0,
    admission_version: int = 0,
    reserved_slots: int = 0,
) -> LoadSnapshot:
    return LoadSnapshot(
        engine_id=0,
        ready=True,
        waiting=0,
        running=0,
        free_blocks_min=100,
        wave_id=1,
        quantum_id=1,
        ingress_version=ingress_version,
        admission_version=admission_version,
        reserved_slots=reserved_slots,
        rank_loads=(
            RankLoad(
                global_rank=0,
                sp_idx=0,
                tp_idx=0,
                master_batch_size=0,
                active_master_requests=0,
                free_blocks=100,
                total_blocks=128,
                master_assignments=0,
                mastered_decode_tokens=0,
                control_dummy_blocks=1,
            ),
        ),
    )


def _submit(router: RequestRouter, request_id: int) -> None:
    router.submit_async(
        request_id=request_id,
        prompt_len=1,
        num_tokens=1,
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
        sequence_payload=b"payload",
    )


@dataclass
class _AsyncEngine:
    engine_id: int = 0
    commands: list[AddCommand] = field(default_factory=list)
    handles: list[dict] = field(default_factory=list)
    ingress_version: int = 0

    def admit_batch_async(self, commands, reservations):
        self.commands.extend(commands)
        acks = []
        for command in commands:
            self.ingress_version += 1
            acks.append(
                IngressAck(
                    command.request_id,
                    self.engine_id,
                    True,
                    ingress_version=self.ingress_version,
                )
            )
        handle = {"ready": False, "acks": tuple(acks)}
        self.handles.append(handle)
        return handle

    def poll_admission_batch(self, handle):
        if not handle["ready"]:
            return False, None
        return True, handle["acks"]

    def abort(self, request_id, *, allow_future_ingress=False):
        return AbortResult(request_id, "abort_pending")

    def clear_ingress_abort(self, request_id):
        return None


def test_snapshot_ingress_watermark_prevents_staged_slot_double_charge():
    engine = _AsyncEngine()
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=_planner_config(queue_capacity=2),
    )
    router.record_loads((_load(),))
    _submit(router, 1)
    assert router.poll_ingress_acks() == ()
    engine.handles[0]["ready"] = True

    receipt = router.poll_ingress_acks()

    assert receipt[0].ingress_version == 1
    assert receipt[0].admission_version is None
    assert router.owner(1) == RequestOwner(OwnerState.PENDING_ADD, 0)
    router.record_loads((_load(ingress_version=1, reserved_slots=1),))
    _submit(router, 2)
    assert router.poll_ingress_acks() == ()
    assert [command.request_id for command in engine.commands] == [1, 2]


def test_pending_add_can_finish_aborted_without_successful_add_result():
    engine = _AsyncEngine()
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=_planner_config(),
    )
    router.record_loads((_load(),))
    _submit(router, 1)
    assert router.poll_ingress_acks() == ()
    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued

    terminal = FinishEvent(1, 0, "ABORTED", 0)

    assert router.record_finish_events((terminal,)) == (terminal,)
    assert router.owner(1) == RequestOwner(
        OwnerState.TERMINAL_DRAINING, 0
    )
    assert router.terminal_event(1) == terminal
    release = ResourceReleaseEvent(1, 0, 0, 0, -1)
    assert router.record_resource_release_events((release,)) == (release,)
    assert router.owner(1) is None


def test_token_commit_waits_for_authoritative_owner_commit():
    engine = _AsyncEngine()
    router = RequestRouter(
        {0: engine},
        router_policy="least_batch",
        admission_planner_config=_planner_config(),
    )
    router.record_loads((_load(),))
    _submit(router, 1)
    assert router.poll_ingress_acks() == ()
    token_event = TokenCommitEvent(
        request_id=1,
        engine_id=0,
        generation_epoch=0,
        wave_id=1,
        quantum_id=0,
        output_offset=1,
        token_ids=(7,),
    )

    assert router.record_token_commit_events((token_event,)) == ()
    engine.handles[0]["ready"] = True
    assert router.poll_ingress_acks()[0].enqueued
    assert router.record_token_commit_events(()) == ()
    assert router.record_add_results(
        (AddResultEvent(1, 0, True, admission_version=1),)
    )
    assert router.record_token_commit_events(()) == (token_event,)


def _command_and_sequence(request_id: int) -> tuple[AddCommand, Sequence]:
    sequence = Sequence([request_id], 0.1, 16, True)
    sequence.seq_id = request_id
    return (
        AddCommand(
            request_id=request_id,
            prompt_len=1,
            num_tokens=1,
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
            sequence_payload=b"payload",
        ),
        sequence,
    )


def _local_engine_for_drain(scheduler, request_ids=(1, 2)):
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        max_ingress_batch_requests=8,
        max_ingress_drain_ms=0.0,
    )
    engine.engine_id = 0
    engine.scheduler = scheduler
    engine._failure = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_head = None
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set(request_ids)
    engine._ingress_pending_ids = set(request_ids)
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = len(request_ids)
    engine._admission_version = 0
    engine._capacity_epoch = 0
    engine._events_lock = threading.Lock()
    engine._add_result_events = deque()
    engine._terminal_events = deque()
    engine._resource_release_events = deque()
    engine._ingress_queue_delay_ms_total = 0.0
    engine._scheduler_add_ms_total = 0.0
    engine._local_transient_retries = 0
    engine._planned_commit_attempts = 0
    for request_id in request_ids:
        command, sequence = _command_and_sequence(request_id)
        engine._ingress_adds.put_nowait(
            _IngressAdd(
                command,
                sequence,
                reservation=AdmissionReservation(
                    request_id, 0, 0, (1,)
                ),
                ingress_version=request_id,
            )
        )
    return engine


def test_transient_planned_head_is_sticky_and_retried_locally():
    class Scheduler:
        def __init__(self):
            self.calls = []

        def commit_planned_batch(self, commands, reservations, sequences):
            request_id = commands[0].request_id
            self.calls.append(request_id)
            if self.calls == [1]:
                return (
                    AddResult(
                        request_id,
                        False,
                        0,
                        "admission_state_mismatch",
                    ),
                )
            return (AddResult(request_id, True, 0),)

    scheduler = Scheduler()
    engine = _local_engine_for_drain(scheduler)

    engine._drain_ingress()
    assert scheduler.calls == [1]
    assert engine._ingress_head.command.request_id == 1
    assert engine.drain_add_results() == ()

    engine._drain_ingress()
    assert scheduler.calls == [1, 1, 2]
    assert [event.request_id for event in engine.drain_add_results()] == [1, 2]
    assert engine._local_transient_retries == 1
    assert engine._planned_commit_attempts == 3


def test_abort_arriving_during_commit_wins_atomic_handoff():
    class Scheduler:
        def __init__(self):
            self.events = ()

        def commit_planned_batch(self, commands, reservations, sequences):
            request_id = commands[0].request_id
            with engine._ingress_lock:
                engine._cancelled_ingress_ids.add(request_id)
            return (AddResult(request_id, True, 0),)

        def abort(self, request_id):
            self.events = (FinishEvent(request_id, 0, "ABORTED", 0),)
            return AbortResult(request_id, "aborted")

        def drain_terminal_events(self):
            events = self.events
            self.events = ()
            return events

        @staticmethod
        def drain_resource_release_events():
            return (ResourceReleaseEvent(1, 0, 0, 0, -1),)

    scheduler = Scheduler()
    engine = _local_engine_for_drain(scheduler, request_ids=(1,))

    engine._drain_ingress()

    assert engine.drain_add_results() == ()
    assert engine.drain_events() == (FinishEvent(1, 0, "ABORTED", 0),)
    assert engine._reserved_slots == 0
    # Releasing the staged lifecycle reservation advances the authoritative
    # LocalEngine state generation even though no commit event was published.
    assert engine._admission_version == 1

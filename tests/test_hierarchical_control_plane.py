from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nanodeploy.engine.decode_coordinator import DecodeCoordinatorState
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AbortResult,
    FinishEvent,
    LoadSnapshot,
    OwnerState,
    validate_execution_trace_set,
)
from nanodeploy.router.request_router import RequestRouter


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
    router = RequestRouter(engines)

    assert route(router, 1).engine_id == 0
    assert route(router, 2).engine_id == 1
    assert route(router, 3).engine_id == 0


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

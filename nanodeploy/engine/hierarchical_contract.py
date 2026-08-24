from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


HIERARCHICAL_LOOP_COUNT = 16
CONTROL_DUMMY_SCHEMA_VERSION = 2
UINT64_MAX = (1 << 64) - 1


def round_up(value: int, quantum: int = HIERARCHICAL_LOOP_COUNT) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    return ((value + quantum - 1) // quantum) * quantum


class RequestState(str, Enum):
    PENDING_ADD = "PENDING_ADD"
    WAITING_ADMISSION = "WAITING_ADMISSION"
    RUNNING_DECODE = "RUNNING_DECODE"
    ABORT_PENDING = "ABORT_PENDING"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RequestState.FINISHED,
            RequestState.ABORTED,
            RequestState.REJECTED,
        }


class OwnerState(str, Enum):
    PENDING_GLOBAL = "PENDING_GLOBAL"
    PENDING_INGRESS = "PENDING_INGRESS"
    PENDING_ADD = "PENDING_ADD"
    OWNED = "OWNED"
    # Compatibility alias for callers that only distinguished pending/owned.
    PENDING_OWNER = "PENDING_INGRESS"


@dataclass(frozen=True, slots=True)
class RequestValidation:
    original_prompt_len: int
    internal_prompt_len: int
    padded_completion_len: int
    total_capacity_len: int


def validate_add_request(
    *,
    request_id: int,
    prompt_len: int,
    max_tokens: int,
    ignore_eos: bool,
    max_model_len: int,
    vocab_size: int,
) -> RequestValidation:
    if not 0 <= request_id <= UINT64_MAX:
        raise ValueError("request_id must fit uint64")
    if prompt_len < 1:
        raise ValueError("prompt_len must be positive")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not ignore_eos:
        raise ValueError("hierarchical scheduler requires ignore_eos=True")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    padded_completion_len = round_up(max_tokens)
    total_capacity_len = prompt_len + 1 + padded_completion_len
    if total_capacity_len > max_model_len:
        raise ValueError(
            "request exceeds hierarchical padded model length: "
            f"prompt={prompt_len} + bootstrap=1 + "
            f"padded_completion={padded_completion_len} > "
            f"max_model_len={max_model_len}"
        )
    return RequestValidation(
        original_prompt_len=prompt_len,
        internal_prompt_len=prompt_len + 1,
        padded_completion_len=padded_completion_len,
        total_capacity_len=total_capacity_len,
    )


@dataclass(frozen=True, slots=True)
class AddCommand:
    request_id: int
    prompt_len: int
    num_tokens: int
    max_tokens: int
    temperature: float
    ignore_eos: bool
    wave_id: int
    sequence_payload: bytes = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AdmissionReservation:
    """One LB-selected LocalEngine SP placement to validate and commit."""

    request_id: int
    engine_id: int
    master_sp_idx: int
    dispatched_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AddResult:
    request_id: int
    accepted: bool
    engine_id: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class IngressAck:
    request_id: int
    engine_id: int
    enqueued: bool
    reason: str | None = None
    # Monotonic LocalEngine admission commit version. A successful
    # centralized-admission ACK carries the version that will be visible in
    # subsequent cached load snapshots.
    admission_version: int | None = None
    # Capacity epoch observed by the LocalEngine planner. Transient admission
    # failures remain blocked until a later snapshot advances past this epoch.
    capacity_epoch: int | None = None
    # Frontend monotonic-clock intervals. router_pending_ms covers time in
    # RequestRouter before control RPC attempts. admission_rpc_ms retains its
    # historical name: for centralized least_batch it ends at authoritative
    # admission, while least_batch_v2 ends at the fast ingress receipt.
    router_pending_ms: float | None = None
    admission_rpc_ms: float | None = None
    # LocalEngine monotonic-clock intervals for the final admission attempt.
    # These are safe across nodes because each duration is computed entirely
    # inside the destination actor.
    local_command_queue_ms: float | None = None
    local_admission_ms: float | None = None


@dataclass(frozen=True, slots=True)
class AddResultEvent:
    request_id: int
    engine_id: int
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FirstTokenEvent:
    request_id: int
    engine_id: int
    generated_count: int


@dataclass(frozen=True, slots=True)
class FirstScheduleEvent:
    """First entry into executor.run, excluding control-plane pickup delay."""

    request_id: int
    engine_id: int
    local_scheduler_queue_ms: float
    global_capacity_queue_ms: float = 0.0

    @property
    def first_schedule_latency_ms(self) -> float:
        return (
            self.local_scheduler_queue_ms
            + self.global_capacity_queue_ms
        )


@dataclass(frozen=True, slots=True)
class AbortResult:
    request_id: int
    status: str


@dataclass(frozen=True, slots=True)
class FinishEvent:
    request_id: int
    generated_count: int
    status: str
    engine_id: int
    first_forward_to_terminal_ms: float | None = None
    global_capacity_queue_ms: float = 0.0
    final_quantum_execute_ms: float | None = None

    @property
    def final_quantum_real_tokens(self) -> int:
        if self.generated_count <= 0:
            return 0
        remainder = self.generated_count % HIERARCHICAL_LOOP_COUNT
        return remainder or HIERARCHICAL_LOOP_COUNT

    @property
    def final_quantum_unused_decode_ms(self) -> float | None:
        if self.status != "FINISHED" or self.generated_count <= 0:
            return None
        unused_tokens = (
            HIERARCHICAL_LOOP_COUNT - self.final_quantum_real_tokens
        )
        if unused_tokens == 0:
            return 0.0
        if self.final_quantum_execute_ms is None:
            return None
        return (
            max(0.0, self.final_quantum_execute_ms)
            * unused_tokens
            / HIERARCHICAL_LOOP_COUNT
        )

    @property
    def first_forward_to_terminal_real_token_ms(self) -> float | None:
        if self.first_forward_to_terminal_ms is None:
            return None
        unused_decode_ms = self.final_quantum_unused_decode_ms
        if unused_decode_ms is None:
            return None
        return max(
            0.0,
            self.first_forward_to_terminal_ms - unused_decode_ms,
        )


@dataclass(frozen=True, slots=True)
class RankLoad:
    """Latest and cumulative load for one runtime SP rank."""

    global_rank: int
    sp_idx: int
    tp_idx: int
    master_batch_size: int
    active_master_requests: int
    free_blocks: int
    total_blocks: int
    master_assignments: int
    mastered_decode_tokens: int
    # Admission-planner state. These values describe the currently running
    # placements, not just the most recent decode batch.
    active_receiver_requests: int = 0
    active_dispatched_tokens: int = 0
    control_dummy_blocks: int = 0


@dataclass(frozen=True, slots=True)
class DecodeITLSample:
    """One lightweight, token-weighted hierarchical decode observation."""

    engine_id: int
    wave_id: int
    quantum_id: int
    itl_ms: float
    token_count: int


@dataclass(frozen=True, slots=True)
class LoadSnapshot:
    engine_id: int
    ready: bool
    waiting: int
    running: int
    free_blocks_min: int
    wave_id: int
    quantum_id: int
    admission_version: int = 0
    # Monotonic LocalEngine capacity generation. Unlike admission_version,
    # this only advances when a lifecycle reservation is released, so a
    # deferred global admission batch is not retried against unchanged
    # capacity.
    capacity_epoch: int = 0
    useful_real_batch_size: int = 0
    control_dummy_count: int = 0
    all_dummy_engine_quantums: int = 0
    useful_decode_tokens: int = 0
    raw_token_slots: int = 0
    control_dummy_slots: int = 0
    total_rank_forwards: int = 0
    all_dummy_rank_forwards: int = 0
    preemption_count: int = 0
    command_count: int = 0
    command_queue_delay_ms_total: float = 0.0
    decode_quantum_count: int = 0
    admission_latency_ms_total: float = 0.0
    schedule_latency_ms_total: float = 0.0
    coordination_latency_ms_total: float = 0.0
    execute_latency_ms_total: float = 0.0
    ray_get_latency_ms_total: float = 0.0
    ray_get_latency_ms_max: float = 0.0
    worker_result_wait_latency_ms_total: float = 0.0
    worker_result_wait_latency_ms_max: float = 0.0
    result_rebuild_latency_ms_total: float = 0.0
    result_rebuild_latency_ms_max: float = 0.0
    result_rebuild_sample_count: int = 0
    result_index_latency_ms_total: float = 0.0
    result_validate_latency_ms_total: float = 0.0
    result_pack_latency_ms_total: float = 0.0
    postprocess_latency_ms_total: float = 0.0
    pending_ingress: int = 0
    pending_add_results: int = 0
    reserved_slots: int = 0
    ingress_queue_delay_ms_total: float = 0.0
    scheduler_add_ms_total: float = 0.0
    decode_itl_ms_weighted_total: float = 0.0
    decode_itl_token_count: int = 0
    decode_itl_sample_count: int = 0
    rank_loads: tuple[RankLoad, ...] = ()

    @property
    def dummy_rank_forward_ratio(self) -> float:
        if self.total_rank_forwards == 0:
            return 0.0
        return self.all_dummy_rank_forwards / self.total_rank_forwards

    @property
    def dummy_slot_ratio(self) -> float:
        if self.raw_token_slots == 0:
            return 0.0
        return self.control_dummy_slots / self.raw_token_slots


@dataclass(frozen=True, slots=True)
class FrontendEventBatch:
    """One consolidated LocalEngine-to-frontend control-plane response."""

    engine_id: int
    load: LoadSnapshot
    add_results: tuple[AddResultEvent, ...] = ()
    first_schedule_events: tuple[FirstScheduleEvent, ...] = ()
    first_token_events: tuple[FirstTokenEvent, ...] = ()
    finish_events: tuple[FinishEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class EngineReady:
    engine_id: int
    global_ranks: tuple[int, ...]
    config_fingerprint: str
    node_id: str
    worker_node_ids: tuple[str, ...]
    frontend_address: str
    frontend_epoch: str


@dataclass(frozen=True, slots=True)
class StartWave:
    wave_id: int


@dataclass(frozen=True, slots=True)
class CoordinatorStatus:
    wave_id: int
    running: bool
    ready: bool
    pending_wakeup: bool


@dataclass(frozen=True, slots=True)
class WorkerDecodeResult:
    wave_id: int
    quantum_id: int
    global_rank: int
    forward_count: int
    mastered_request_ids: tuple[int, ...]
    sampled_token_ids: tuple[tuple[int, ...], ...]

    def validate_quantum(self, wave_id: int, quantum_id: int) -> None:
        if (self.wave_id, self.quantum_id) != (wave_id, quantum_id):
            raise ValueError(
                "worker result step mismatch: "
                f"expected ({wave_id}, {quantum_id}), got "
                f"({self.wave_id}, {self.quantum_id})"
            )
        if self.forward_count != HIERARCHICAL_LOOP_COUNT:
            raise ValueError(
                "hierarchical worker must execute exactly "
                f"{HIERARCHICAL_LOOP_COUNT} forwards, got {self.forward_count}"
            )
        if len(self.mastered_request_ids) != len(self.sampled_token_ids):
            raise ValueError(
                "mastered_request_ids and sampled_token_ids length mismatch"
            )
        if any(
            len(tokens) != HIERARCHICAL_LOOP_COUNT
            for tokens in self.sampled_token_ids
        ):
            raise ValueError(
                "each mastered request must return exactly "
                f"{HIERARCHICAL_LOOP_COUNT} sampled token ids"
            )


@dataclass(slots=True)
class LocalDecodeBatch:
    wave_id: int
    quantum_id: int
    engine_id: int
    engine_has_real: bool
    per_rank_sequences: dict[int, list[Any]]
    request_master_global_rank: dict[int, int]
    frozen_request_order: dict[int, tuple[int, ...]]
    control_dummy_ids: frozenset[int]
    _all_sequences: list[Any] = field(repr=False, default_factory=list)
    _control_dummy_object_ids: frozenset[int] = field(
        repr=False, default_factory=frozenset
    )

    def expected_request_ids(self, global_rank: int) -> tuple[int, ...]:
        return self.frozen_request_order.get(global_rank, ())

    def is_control_dummy(self, sequence: Any) -> bool:
        return id(sequence) in self._control_dummy_object_ids

    def validate_worker_results(
        self, results: Iterable[WorkerDecodeResult]
    ) -> dict[int, WorkerDecodeResult]:
        by_rank: dict[int, WorkerDecodeResult] = {}
        for result in results:
            result.validate_quantum(self.wave_id, self.quantum_id)
            if result.global_rank in by_rank:
                raise ValueError(
                    f"duplicate worker result for global rank {result.global_rank}"
                )
            expected = self.expected_request_ids(result.global_rank)
            if result.mastered_request_ids != expected:
                raise ValueError(
                    f"worker rank {result.global_rank} request order mismatch: "
                    f"expected {expected}, got {result.mastered_request_ids}"
                )
            by_rank[result.global_rank] = result

        expected_ranks = set(self.per_rank_sequences)
        missing = expected_ranks.difference(by_rank)
        extra = set(by_rank).difference(expected_ranks)
        if missing or extra:
            raise ValueError(
                f"worker result rank mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return by_rank


def validate_execution_trace_set(
    traces: Iterable[Mapping[str, Any]],
    expected_global_ranks: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    """Validate a completed integration trace across all global GPU ranks."""

    expected_ranks = tuple(expected_global_ranks)
    if not expected_ranks or len(set(expected_ranks)) != len(expected_ranks):
        raise ValueError("expected_global_ranks must be non-empty and unique")
    by_rank: dict[int, list[tuple[int, int]]] = {
        rank: [] for rank in expected_ranks
    }
    seen_steps: dict[int, set[tuple[int, int]]] = {
        rank: set() for rank in expected_ranks
    }

    for trace in traces:
        global_rank = trace.get("global_rank")
        if global_rank not in by_rank:
            raise ValueError(f"trace contains unknown global rank {global_rank}")
        wave_id = trace.get("wave_id")
        quantum_id = trace.get("quantum_id")
        if (
            not isinstance(wave_id, int)
            or wave_id <= 0
            or not isinstance(quantum_id, int)
            or quantum_id < 0
        ):
            raise ValueError("trace contains an invalid wave/quantum identity")
        step = (wave_id, quantum_id)
        if step in seen_steps[global_rank]:
            raise ValueError(
                f"trace contains duplicate step {step} for rank {global_rank}"
            )
        if trace.get("forward_count") != HIERARCHICAL_LOOP_COUNT:
            raise ValueError(
                f"rank {global_rank} step {step} did not execute "
                f"{HIERARCHICAL_LOOP_COUNT} forwards"
            )
        if trace.get("batch_kind") not in {
            "real_or_mixed",
            "all_control_dummy",
        }:
            raise ValueError("trace contains an invalid batch_kind")
        if any(
            not isinstance(trace.get(name), int) or trace[name] < 0
            for name in ("real_batch_size", "control_dummy_count")
        ):
            raise ValueError("trace contains an invalid batch size")

        forwards = tuple(trace.get("forwards", ()))
        inner_loops = tuple(item.get("inner_loop_idx") for item in forwards)
        if inner_loops != tuple(range(HIERARCHICAL_LOOP_COUNT)):
            raise ValueError(
                f"rank {global_rank} step {step} has invalid inner-loop order"
            )
        if any(
            item.get("forward_begin") is None
            or item.get("forward_end") is None
            or item["forward_end"] < item["forward_begin"]
            for item in forwards
        ):
            raise ValueError("trace contains an invalid forward interval")

        seen_steps[global_rank].add(step)
        by_rank[global_rank].append(step)

    reference = tuple(by_rank[expected_ranks[0]])
    if not reference:
        raise ValueError("execution trace set is empty")
    if reference != tuple(sorted(reference)):
        raise ValueError("execution trace steps are not monotonic")
    for rank in expected_ranks[1:]:
        if tuple(by_rank[rank]) != reference:
            raise ValueError(
                "global ranks executed different wave/quantum sequences: "
                f"rank={rank}, expected={reference}, got={tuple(by_rank[rank])}"
            )
    return reference

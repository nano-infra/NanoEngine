#!/usr/bin/env python3
"""Profile the production hierarchical scheduler control path on CPU.

The legacy ``profile_scheduler_scalability.py`` measures one deployment-wide
C++ ``Scheduler``.  Hierarchical serving has a different ownership boundary:
one frontend ``RequestRouter`` plans admission, then independent
``LocalScheduler`` instances own decode scheduling.  This profiler therefore
reports the full control-plane path, scheduler-only local admission/decode
cost on a matching boundary, and a modelled distributed critical path
separately.

The workload matches the legacy profiler: one logical node contains eight
logical GPUs and ``batch_size_per_gpu`` produces the same total request count.
SP8 scenarios use one LocalScheduler per logical node with ``8 * BS/GPU``
requests; the no-SP scenario uses eight SP1 LocalSchedulers per logical node
with ``BS/GPU`` requests each.

No Ray actor, worker, RDMA endpoint, CUDA context, or GPU kernel is created.
LocalSchedulers are executed serially by this harness.  Metrics named
``modelled_parallel_*`` use the maximum local cost as the ideal distributed
critical path; they are not measured multi-host wall-clock latency.  Logical
node counts outside NanoDeploy's deployable 1/2/4-node topology are explicitly
labelled as an independent-LocalScheduler replica model.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanodeploy.config import Config, DEEPSEEK_V3_BUCKET_POLICY  # noqa: E402
from nanodeploy.engine.hierarchical_contract import (  # noqa: E402
    AddCommand,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    HIERARCHICAL_LOOP_COUNT,
    IngressAck,
    OwnerState,
    WorkerDecodeResult,
)
from nanodeploy.engine.local_scheduler import LocalScheduler  # noqa: E402
from nanodeploy.engine.sequence import Sequence as NanoDeploySequence  # noqa: E402
from nanodeploy.engine.topology import EngineTopology  # noqa: E402
from nanodeploy.router.admission_planner import (  # noqa: E402
    AdmissionPlannerConfig,
)
from nanodeploy.router.request_router import RequestRouter  # noqa: E402
from nanodeploy.sampling_params import SamplingParams  # noqa: E402
from scripts.decentralized_scalability.common import (  # noqa: E402
    base_metadata,
    summarize,
    write_results,
)
from scripts.scheduler_overhead.profile_scheduler_scalability import (  # noqa: E402
    DEFAULT_SCENARIOS,
    GPUS_PER_LOGICAL_NODE,
    SCENARIOS,
    SP8_DEGREE,
    Scenario,
    _bucket_sp_degree,
    _long_request_indices,
    _native_stderr,
    _parse_positive_ints,
    _parse_scenarios,
)


DEFAULT_MODEL = Path(
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V3/snapshots/"
    "e815299b0bcbac849fa540c768ef21845365c9eb"
)
SUPPORTED_PHYSICAL_NODE_COUNTS = frozenset({1, 2, 4})
PHASE_NAMES = (
    "admit_ms",
    "pre_load_ms",
    "plan_decode_ms",
    "mark_first_schedule_ms",
    "postprocess_ms",
    "post_load_ms",
    "scheduler_cpu_ms",
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class HierarchicalProfileCase:
    scenario: Scenario
    logical_nodes: int
    batch_size_per_gpu: int
    short_context_len: int
    long_context_len: int
    block_size: int
    seed: int

    def __post_init__(self) -> None:
        positive = (
            self.logical_nodes,
            self.batch_size_per_gpu,
            self.short_context_len,
            self.long_context_len,
            self.block_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("hierarchical profile dimensions must be positive")
        if self.long_context_len <= self.short_context_len:
            raise ValueError("long context must exceed short context")

    @property
    def logical_gpus(self) -> int:
        return self.logical_nodes * GPUS_PER_LOGICAL_NODE

    @property
    def attention_sp(self) -> int:
        return SP8_DEGREE if self.scenario.use_sp8_topology else 1

    @property
    def local_scheduler_count(self) -> int:
        if self.scenario.use_sp8_topology:
            return self.logical_nodes
        return self.logical_gpus

    @property
    def total_requests(self) -> int:
        return self.logical_gpus * self.batch_size_per_gpu

    @property
    def requests_per_local_scheduler(self) -> int:
        requests, remainder = divmod(
            self.total_requests, self.local_scheduler_count
        )
        if remainder:
            raise RuntimeError("profile requests do not divide across schedulers")
        return requests

    @property
    def expected_sp8_requests(self) -> int:
        return round(self.total_requests * self.scenario.sp8_ratio)

    @property
    def topology_scope(self) -> str:
        if self.logical_nodes in SUPPORTED_PHYSICAL_NODE_COUNTS:
            return "complete_production_topology_cpu_model"
        return "logical_independent_local_scheduler_replica_model"

    @property
    def deployment_topology_supported(self) -> bool:
        return self.logical_nodes in SUPPORTED_PHYSICAL_NODE_COUNTS

    def dynamic_long_indices(self) -> set[int]:
        if not self.scenario.uses_dynamic_policy:
            return set()
        return _long_request_indices(
            self.total_requests,
            self.expected_sp8_requests,
            self.seed,
        )

    def long_requests_per_local_scheduler(self) -> tuple[int, ...]:
        counts = [0] * self.local_scheduler_count
        for request_id in self.dynamic_long_indices():
            counts[request_id % self.local_scheduler_count] += 1
        return tuple(counts)


@dataclass(frozen=True, slots=True)
class _ImmediateFlight:
    acks: tuple[IngressAck, ...]


class _RecordingAdmissionTransport:
    """Immediately-ready transport that retains Router-selected placements."""

    def __init__(self, engine_id: int) -> None:
        self.engine_id = engine_id
        self.ingress_version = 0
        self.planned_batches: list[
            tuple[tuple[AddCommand, ...], tuple[AdmissionReservation, ...]]
        ] = []

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> _ImmediateFlight:
        if len(commands) != len(reservations):
            raise AssertionError("command/reservation count mismatch")
        self.planned_batches.append((commands, reservations))
        acks = []
        for command, reservation in zip(commands, reservations, strict=True):
            if (
                reservation.request_id != command.request_id
                or reservation.engine_id != self.engine_id
            ):
                raise AssertionError("Router emitted an invalid reservation")
            self.ingress_version += 1
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=True,
                    ingress_version=self.ingress_version,
                )
            )
        return _ImmediateFlight(tuple(acks))

    def poll_admission_batch(
        self, flight: _ImmediateFlight
    ) -> tuple[bool, tuple[IngressAck, ...]]:
        return True, flight.acks

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        del allow_future_ingress
        return AbortResult(request_id=request_id, status="abort_pending")

    def clear_ingress_abort(self, request_id: int) -> None:
        del request_id


@dataclass(slots=True)
class _AdmissionState:
    schedulers: list[LocalScheduler]
    router: RequestRouter
    request_ids_by_engine: tuple[tuple[int, ...], ...]
    actual_sp1_requests: int
    actual_sp8_requests: int


def _profile_max_tokens(warmup_iterations: int, measured_iterations: int) -> int:
    return (
        warmup_iterations + measured_iterations + 2
    ) * HIERARCHICAL_LOOP_COUNT


def scheduler_kv_blocks(
    case: HierarchicalProfileCase,
    *,
    max_tokens: int,
) -> int:
    """Conservative per-rank capacity for one LocalScheduler replica."""

    local_requests = case.requests_per_local_scheduler
    block_size = case.block_size
    if case.attention_sp == 1:
        prompt_blocks = local_requests * _ceil_div(
            case.short_context_len, block_size
        )
    elif case.scenario.fixed_sp_size == SP8_DEGREE:
        prompt_blocks = local_requests * _ceil_div(
            _ceil_div(case.short_context_len, SP8_DEGREE),
            block_size,
        )
    else:
        short_blocks = _ceil_div(case.short_context_len, block_size)
        long_rank_blocks = _ceil_div(
            _ceil_div(case.long_context_len, SP8_DEGREE),
            block_size,
        )
        prompt_blocks = max(
            (
                long_count * long_rank_blocks
                + _ceil_div(local_requests - long_count, SP8_DEGREE)
                * short_blocks
            )
            for long_count in case.long_requests_per_local_scheduler()
        )

    mastered_requests_per_rank = _ceil_div(
        local_requests, case.attention_sp
    )
    completion_blocks = mastered_requests_per_rank * _ceil_div(
        max_tokens + 1, block_size
    )
    return (
        _ceil_div(prompt_blocks * 11, 10)
        + completion_blocks
        + local_requests
        + 128
    )


def _config_attention_dp(case: HierarchicalProfileCase) -> int:
    if case.deployment_topology_supported:
        return case.local_scheduler_count
    return 1 if case.attention_sp == SP8_DEGREE else GPUS_PER_LOGICAL_NODE


def _new_config(
    case: HierarchicalProfileCase,
    *,
    model: str,
    max_tokens: int,
    admission_batch_size: int,
) -> Config:
    config_attention_dp = _config_attention_dp(case)
    config_world_size = config_attention_dp * case.attention_sp
    dynamic = case.scenario.uses_dynamic_policy
    max_model_len = case.long_context_len + max_tokens + 1
    local_requests = case.requests_per_local_scheduler
    return Config(
        model=model,
        scheduler_arch="hierarchical",
        mode="decode",
        dummy_prefill=True,
        attention_dp=config_attention_dp,
        attention_sp=case.attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=config_world_size,
        ffn_tp=1,
        kvcache_block_size=case.block_size,
        num_kvcache_blocks=scheduler_kv_blocks(
            case, max_tokens=max_tokens
        ),
        max_model_len=max_model_len,
        max_num_batched_tokens=max(
            max_model_len,
            local_requests * case.long_context_len + 1,
        ),
        max_num_seqs=local_requests + case.attention_sp,
        max_num_recv_seqs=local_requests + case.attention_sp,
        hierarchical_queue_capacity=local_requests + 1,
        max_ingress_batch_requests=admission_batch_size,
        fixed_sp_size=case.scenario.fixed_sp_size,
        segment_size=case.block_size,
        reserved_blocks_per_req=1.0,
        dynamic_sp_size_strategy="bucket" if dynamic else "legacy",
        dynamic_sp_bucket_policy=(
            DEEPSEEK_V3_BUCKET_POLICY if dynamic else ""
        ),
        enable_non_uniform_split=True,
        sp_master_selector="LeastBatch",
        routing_strategy="LeastBatch",
        router_policy="least_batch",
        engine_id="hierarchical-scheduler-scalability",
    )


def _engine_topologies(
    case: HierarchicalProfileCase,
    config: Config,
) -> tuple[EngineTopology, ...]:
    if case.deployment_topology_supported:
        return config.hierarchical_topology.engines
    return tuple(
        EngineTopology(
            engine_id=engine_id,
            # Each object is an independent production LocalScheduler replica.
            # global_dp_idx=0 keeps it within the supported template Config.
            global_dp_idx=0,
            global_ranks=tuple(
                range(
                    engine_id * case.attention_sp,
                    (engine_id + 1) * case.attention_sp,
                )
            ),
            attention_sp=case.attention_sp,
            attention_tp=1,
        )
        for engine_id in range(case.local_scheduler_count)
    )


def _profile_sequence(
    *,
    request_id: int,
    context_len: int,
    max_tokens: int,
    vocab_size: int,
) -> NanoDeploySequence:
    token_id = request_id % max(1, vocab_size - 1) + 1
    sequence = NanoDeploySequence(
        [token_id] * context_len,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )
    sequence.seq_id = request_id
    return sequence


def _build_commands_and_sequences(
    case: HierarchicalProfileCase,
    *,
    max_tokens: int,
    vocab_size: int,
) -> tuple[tuple[AddCommand, ...], dict[int, NanoDeploySequence]]:
    long_indices = case.dynamic_long_indices()
    commands = []
    sequences: dict[int, NanoDeploySequence] = {}
    for request_id in range(case.total_requests):
        context_len = (
            case.long_context_len
            if request_id in long_indices
            else case.short_context_len
        )
        sequence = _profile_sequence(
            request_id=request_id,
            context_len=context_len,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        )
        sequences[request_id] = sequence
        commands.append(
            AddCommand(
                request_id=request_id,
                prompt_len=context_len,
                num_tokens=context_len,
                max_tokens=max_tokens,
                temperature=0.1,
                ignore_eos=True,
                wave_id=1,
                sequence_payload=b"excluded-from-scheduler-profile",
            )
        )
    return tuple(commands), sequences


def _route_admission(
    router: RequestRouter,
    commands: Sequence[AddCommand],
) -> tuple[IngressAck, ...]:
    for command in commands:
        router.submit_async(
            request_id=command.request_id,
            prompt_len=command.prompt_len,
            num_tokens=command.num_tokens,
            max_tokens=command.max_tokens,
            temperature=command.temperature,
            ignore_eos=command.ignore_eos,
            sequence_payload=command.sequence_payload,
        )

    observed: list[IngressAck] = []
    max_polls = len(commands) + 2
    for _ in range(max_polls):
        observed.extend(router.poll_ingress_acks())
        if len(observed) == len(commands):
            break
    if len(observed) != len(commands):
        raise RuntimeError(
            f"Router stopped with {len(observed)}/{len(commands)} receipts"
        )
    if not all(ack.enqueued for ack in observed):
        raise RuntimeError("Router rejected hierarchical profile admission")
    return tuple(observed)


def _reservation_sp_degree(reservation: AdmissionReservation) -> int:
    return sum(token_count > 0 for token_count in reservation.dispatched_tokens)


def _run_admission_once(
    case: HierarchicalProfileCase,
    *,
    model: str,
    max_tokens: int,
    admission_batch_size: int,
    suppress_native_setup_logs: bool,
) -> tuple[_AdmissionState, dict[str, float]]:
    setup_begin = time.perf_counter()
    config = _new_config(
        case,
        model=model,
        max_tokens=max_tokens,
        admission_batch_size=admission_batch_size,
    )
    with _native_stderr(suppress_native_setup_logs):
        schedulers = [
            LocalScheduler(config, topology)
            for topology in _engine_topologies(case, config)
        ]
    commands, sequences = _build_commands_and_sequences(
        case,
        max_tokens=max_tokens,
        vocab_size=config.hf_config.vocab_size,
    )
    setup_ms_excluded = (time.perf_counter() - setup_begin) * 1000.0

    snapshot_samples = []
    snapshots = []
    for scheduler in schedulers:
        begin = time.perf_counter()
        snapshots.append(scheduler.load_snapshot(wave_id=1, quantum_id=0))
        snapshot_samples.append((time.perf_counter() - begin) * 1000.0)

    transports = {
        engine_id: _RecordingAdmissionTransport(engine_id)
        for engine_id in range(case.local_scheduler_count)
    }
    router = RequestRouter(
        transports,
        router_policy="least_batch",
        admission_batch_size=admission_batch_size,
        admission_planner_config=AdmissionPlannerConfig.from_config(config),
        initial_wave_id=1,
    )
    begin = time.perf_counter()
    router.record_loads(snapshots)
    router_record_loads_ms = (time.perf_counter() - begin) * 1000.0

    begin = time.perf_counter()
    _route_admission(router, commands)
    router_plan_receipt_ms = (time.perf_counter() - begin) * 1000.0

    commit_samples = []
    add_result_events = []
    request_ids_by_engine: list[tuple[int, ...]] = []
    actual_sp1 = 0
    actual_sp8 = 0
    # Queue insertion is setup, matching the centralized profiler's
    # ``Scheduler.add`` boundary. The timed section below measures only the
    # planned LocalScheduler admission transition.
    for scheduler, transport in zip(
        schedulers, transports.values(), strict=True
    ):
        for batch_commands, _reservations in transport.planned_batches:
            batch_sequences = tuple(
                sequences[command.request_id] for command in batch_commands
            )
            staged = scheduler.stage_batch(batch_commands, batch_sequences)
            if not all(result.accepted for result in staged):
                raise RuntimeError(
                    "LocalScheduler rejected a staged profiling admission"
                )

    for engine_id, (scheduler, transport) in enumerate(
        zip(schedulers, transports.values(), strict=True)
    ):
        engine_request_ids = []
        admission_version = 0
        engine_commit_ms = 0.0
        for batch_commands, reservations in transport.planned_batches:
            batch_sequences = tuple(
                sequences[command.request_id] for command in batch_commands
            )
            begin = time.perf_counter()
            results = scheduler.commit_planned_batch(
                batch_commands,
                reservations,
                batch_sequences,
                pre_staged=True,
            )
            engine_commit_ms += (time.perf_counter() - begin) * 1000.0
            if not all(result.accepted for result in results):
                raise RuntimeError(
                    f"LocalScheduler {engine_id} rejected planned admission"
                )
            for command, reservation in zip(
                batch_commands, reservations, strict=True
            ):
                degree = _reservation_sp_degree(reservation)
                if degree == 1:
                    actual_sp1 += 1
                elif degree == SP8_DEGREE:
                    actual_sp8 += 1
                else:
                    raise RuntimeError(
                        f"unexpected hierarchical SP degree {degree}"
                    )
                admission_version += 1
                engine_request_ids.append(command.request_id)
                add_result_events.append(
                    AddResultEvent(
                        request_id=command.request_id,
                        engine_id=engine_id,
                        accepted=True,
                        admission_version=admission_version,
                    )
                )
        commit_samples.append(engine_commit_ms)
        request_ids_by_engine.append(tuple(engine_request_ids))
        transport.planned_batches.clear()

    begin = time.perf_counter()
    committed_events = router.record_add_results(add_result_events)
    router_add_result_ms = (time.perf_counter() - begin) * 1000.0
    if len(committed_events) != case.total_requests:
        raise RuntimeError("Router did not commit every hierarchical ADD result")

    counts = tuple(len(ids) for ids in request_ids_by_engine)
    if counts != (case.requests_per_local_scheduler,) * len(counts):
        raise RuntimeError(f"Router produced an imbalanced workload: {counts}")
    if router.pending_ingress_count or router.pending_add_count:
        raise RuntimeError("Router retained pending work after admission commit")
    if any(
        router.owner(request_id) is None
        or router.owner(request_id).state is not OwnerState.OWNED
        for request_id in range(case.total_requests)
    ):
        raise RuntimeError("Router ownership did not reach OWNED")
    if actual_sp8 != case.expected_sp8_requests:
        raise RuntimeError(
            f"admitted {actual_sp8} SP8 requests, expected "
            f"{case.expected_sp8_requests}"
        )
    expected_sp1 = case.total_requests - case.expected_sp8_requests
    if actual_sp1 != expected_sp1:
        raise RuntimeError(
            f"admitted {actual_sp1} SP1 requests, expected {expected_sp1}"
        )

    local_snapshot_sum_ms = sum(snapshot_samples)
    local_snapshot_critical_ms = max(snapshot_samples)
    local_commit_sum_ms = sum(commit_samples)
    local_commit_critical_ms = max(commit_samples)
    modelled_admission_critical_ms = (
        local_snapshot_critical_ms
        + router_record_loads_ms
        + router_plan_receipt_ms
        + local_commit_critical_ms
        + router_add_result_ms
    )
    metrics = {
        "setup_ms_excluded": setup_ms_excluded,
        "local_snapshot_sum_ms": local_snapshot_sum_ms,
        "local_snapshot_critical_ms": local_snapshot_critical_ms,
        "router_record_loads_ms": router_record_loads_ms,
        "router_plan_receipt_ms": router_plan_receipt_ms,
        "local_commit_sum_ms": local_commit_sum_ms,
        "local_commit_critical_ms": local_commit_critical_ms,
        "router_add_result_ms": router_add_result_ms,
        # Primary fair-boundary admission metric. Router planning and receipt
        # handling remain available above as control-plane diagnostics.
        "scheduler_admission_critical_ms": local_commit_critical_ms,
        "modelled_admission_critical_ms": modelled_admission_critical_ms,
    }
    return (
        _AdmissionState(
            schedulers=schedulers,
            router=router,
            request_ids_by_engine=tuple(request_ids_by_engine),
            actual_sp1_requests=actual_sp1,
            actual_sp8_requests=actual_sp8,
        ),
        metrics,
    )


def build_worker_results(
    batch: Any,
    *,
    token_base: int,
) -> list[WorkerDecodeResult]:
    return [
        WorkerDecodeResult(
            wave_id=batch.wave_id,
            quantum_id=batch.quantum_id,
            global_rank=global_rank,
            forward_count=HIERARCHICAL_LOOP_COUNT,
            mastered_request_ids=batch.expected_request_ids(global_rank),
            sampled_token_ids=tuple(
                tuple(
                    token_base + offset
                    for offset in range(HIERARCHICAL_LOOP_COUNT)
                )
                for _ in batch.expected_request_ids(global_rank)
            ),
        )
        for global_rank in batch.per_rank_sequences
    ]


def _run_local_quantum(
    scheduler: LocalScheduler,
    *,
    quantum_id: int,
) -> dict[str, float]:
    # Keep the primary decode metric on the same native boundary as the
    # centralized profiler: one call to C++ Scheduler.schedule().  Contract
    # bookkeeping remains necessary for state progression, but it is measured
    # separately after the native call and is not charged to scheduler_cpu_ms.
    begin = time.perf_counter()
    scheduler.load_snapshot(wave_id=1, quantum_id=quantum_id)
    pre_load_ms = (time.perf_counter() - begin) * 1000.0

    # The centralized profiler disables cyclic GC around its measured decode
    # call. Mirror that runtime condition for the native decentralized
    # boundary, then restore it before Python contract bookkeeping starts.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        begin = time.perf_counter()
        schedule_result = scheduler.cpp_scheduler.schedule()
        native_schedule_ms = (time.perf_counter() - begin) * 1000.0
    finally:
        if gc_was_enabled:
            gc.enable()
    if schedule_result.is_prefill:
        raise RuntimeError("steady-state scheduler unexpectedly admitted work")

    # Build the hierarchical contract view outside the native scheduler timer.
    batch = scheduler.build_decode_batch_from_schedule_result(
        schedule_result,
        wave_id=1,
        quantum_id=quantum_id,
    )
    if not batch.engine_has_real:
        raise RuntimeError("hierarchical profiler produced an all-dummy batch")

    begin = time.perf_counter()
    scheduler.mark_first_forward_started(batch)
    mark_first_schedule_ms = (time.perf_counter() - begin) * 1000.0

    worker_results = build_worker_results(
        batch,
        token_base=100 + quantum_id * HIERARCHICAL_LOOP_COUNT,
    )
    begin = time.perf_counter()
    events = scheduler.postprocess(batch, worker_results)
    postprocess_ms = (time.perf_counter() - begin) * 1000.0
    if events:
        raise RuntimeError("hierarchical profiler workload finished too early")

    begin = time.perf_counter()
    scheduler.load_snapshot(wave_id=1, quantum_id=quantum_id + 1)
    post_load_ms = (time.perf_counter() - begin) * 1000.0
    # This is deliberately the same scheduler-only boundary as the
    # centralized profiler.  The remaining phases are retained as diagnostic
    # control-plane bookkeeping, not folded into the primary decode metric.
    scheduler_cpu_ms = native_schedule_ms
    contract_overhead_ms = (
        pre_load_ms
        + mark_first_schedule_ms
        + postprocess_ms
        + post_load_ms
    )
    return {
        "admit_ms": 0.0,
        "pre_load_ms": pre_load_ms,
        "plan_decode_ms": native_schedule_ms,
        "native_schedule_ms": native_schedule_ms,
        "mark_first_schedule_ms": mark_first_schedule_ms,
        "postprocess_ms": postprocess_ms,
        "post_load_ms": post_load_ms,
        "contract_overhead_ms": contract_overhead_ms,
        "scheduler_cpu_ms": scheduler_cpu_ms,
    }


def _profile_decode(
    case: HierarchicalProfileCase,
    state: _AdmissionState,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    per_engine_samples: list[list[dict[str, float]]] = []
    per_engine_profiles = []
    expected_completed = (
        warmup_iterations + measured_iterations
    ) * HIERARCHICAL_LOOP_COUNT
    for engine_id, (scheduler, request_ids) in enumerate(
        zip(
            state.schedulers,
            state.request_ids_by_engine,
            strict=True,
        )
    ):
        for quantum_id in range(warmup_iterations):
            _run_local_quantum(scheduler, quantum_id=quantum_id)
        samples = [
            _run_local_quantum(
                scheduler,
                quantum_id=warmup_iterations + iteration,
            )
            for iteration in range(measured_iterations)
        ]
        final_load = scheduler.load_snapshot(
            wave_id=1,
            quantum_id=warmup_iterations + measured_iterations,
        )
        completed = tuple(
            scheduler._records[request_id].sequence.num_completed_tokens
            for request_id in request_ids
        )
        correctness = {
            "all_requests_remain_live": (
                not scheduler.is_finished()
                and len(scheduler._records) == len(request_ids)
            ),
            "all_requests_advanced_equally": completed
            == (expected_completed,) * len(request_ids),
            "rank_count_matches_attention_sp": len(final_load.rank_loads)
            == case.attention_sp,
            "no_preemption": final_load.preemption_count == 0,
        }
        if not all(correctness.values()):
            raise AssertionError(
                f"LocalScheduler {engine_id} invariant failed: {correctness}"
            )
        per_engine_samples.append(samples)
        per_engine_profiles.append(
            {
                "engine_id": engine_id,
                "request_count": len(request_ids),
                "phase_mean_ms": {
                    phase: sum(sample[phase] for sample in samples)
                    / measured_iterations
                    for phase in PHASE_NAMES
                },
                "correctness": correctness,
            }
        )

    phase_stats = {
        phase: summarize(
            sample[phase]
            for engine_samples in per_engine_samples
            for sample in engine_samples
        )
        for phase in PHASE_NAMES
    }
    per_engine_phase_mean_stats = {
        phase: summarize(
            profile["phase_mean_ms"][phase]
            for profile in per_engine_profiles
        )
        for phase in PHASE_NAMES
    }
    modelled_parallel_samples = [
        max(
            engine_samples[iteration]["scheduler_cpu_ms"]
            for engine_samples in per_engine_samples
        )
        for iteration in range(measured_iterations)
    ]
    aggregate_cpu_samples = [
        sum(
            engine_samples[iteration]["scheduler_cpu_ms"]
            for engine_samples in per_engine_samples
        )
        for iteration in range(measured_iterations)
    ]
    modelled_parallel = summarize(modelled_parallel_samples)
    aggregate_cpu = summarize(aggregate_cpu_samples)
    global_quantums_per_second = 1000.0 / modelled_parallel["mean"]
    return {
        "local_phase_ms": phase_stats,
        "per_engine_phase_mean_ms": per_engine_phase_mean_stats,
        "modelled_parallel_quantum_ms": modelled_parallel,
        "aggregate_local_cpu_ms_per_global_quantum": aggregate_cpu,
        "modelled_global_quantums_per_second": global_quantums_per_second,
        "modelled_aggregate_local_quantums_per_second": (
            case.local_scheduler_count * global_quantums_per_second
        ),
        "modelled_request_schedule_decisions_per_second": (
            case.total_requests * global_quantums_per_second
        ),
        "per_engine_profiles": per_engine_profiles,
    }


def run_case(
    case: HierarchicalProfileCase,
    *,
    model: str,
    warmup_iterations: int,
    measured_iterations: int,
    admission_iterations: int,
    admission_batch_size: int,
    suppress_native_setup_logs: bool = True,
) -> dict[str, Any]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if min(
        measured_iterations,
        admission_iterations,
        admission_batch_size,
    ) <= 0:
        raise ValueError("iterations and admission batch size must be positive")

    max_tokens = _profile_max_tokens(warmup_iterations, measured_iterations)
    admission_samples = []
    retained_state: _AdmissionState | None = None
    for admission_index in range(admission_iterations):
        state, metrics = _run_admission_once(
            case,
            model=model,
            max_tokens=max_tokens,
            admission_batch_size=admission_batch_size,
            suppress_native_setup_logs=suppress_native_setup_logs,
        )
        admission_samples.append(metrics)
        if admission_index == admission_iterations - 1:
            retained_state = state
        else:
            del state
            gc.collect()
    if retained_state is None:
        raise RuntimeError("admission profile did not retain a scheduler state")

    decode = _profile_decode(
        case,
        retained_state,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    admission_stats = {
        metric: summarize(sample[metric] for sample in admission_samples)
        for metric in admission_samples[0]
    }
    return {
        "scenario": case.scenario.name,
        "logical_nodes": case.logical_nodes,
        "logical_gpus": case.logical_gpus,
        "local_scheduler_count": case.local_scheduler_count,
        "attention_sp": case.attention_sp,
        "configured_attention_dp": _config_attention_dp(case),
        "deployment_topology_supported": case.deployment_topology_supported,
        "topology_scope": case.topology_scope,
        "batch_size_per_gpu": case.batch_size_per_gpu,
        "requests_per_local_scheduler": case.requests_per_local_scheduler,
        "total_requests": case.total_requests,
        "expected_sp1_requests": (
            case.total_requests - case.expected_sp8_requests
        ),
        "expected_sp8_requests": case.expected_sp8_requests,
        "actual_sp1_requests": retained_state.actual_sp1_requests,
        "actual_sp8_requests": retained_state.actual_sp8_requests,
        "short_context_len": case.short_context_len,
        "long_context_len": case.long_context_len,
        "dynamic_sp_bucket_policy": (
            DEEPSEEK_V3_BUCKET_POLICY
            if case.scenario.uses_dynamic_policy
            else ""
        ),
        "block_size": case.block_size,
        "loop_count": HIERARCHICAL_LOOP_COUNT,
        "admission_batch_size": admission_batch_size,
        "admission_iterations": admission_iterations,
        "admission_ms": admission_stats,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        **decode,
        "weak_scaling_efficiency_vs_smallest_node": None,
    }


def _annotate_weak_scaling(records: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["scenario"], record["batch_size_per_gpu"])
        baseline = baselines.get(key)
        if baseline is None or record["logical_nodes"] < baseline["logical_nodes"]:
            baselines[key] = record
    for record in records:
        baseline = baselines[(record["scenario"], record["batch_size_per_gpu"])]
        expected_scale = (
            record["local_scheduler_count"]
            / baseline["local_scheduler_count"]
        )
        actual_scale = (
            record["modelled_aggregate_local_quantums_per_second"]
            / baseline["modelled_aggregate_local_quantums_per_second"]
        )
        record["weak_scaling_efficiency_vs_smallest_node"] = (
            actual_scale / expected_scale
        )


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        REPO_ROOT
        / "bench_logs"
        / "scheduler_overhead"
        / "hierarchical"
        / timestamp
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Local model snapshot used to construct production Config.",
    )
    parser.add_argument(
        "--logical-nodes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("4,8,16,32"),
        help="Comma-separated logical 8-GPU node counts.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128"),
        help="Comma-separated active master-request counts per logical GPU.",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=DEFAULT_SCENARIOS,
        help=f"Comma-separated scenarios (default: {','.join(DEFAULT_SCENARIOS)}).",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--admission-iterations",
        type=int,
        default=3,
        help="Fresh hierarchical bulk-admission repetitions per case.",
    )
    parser.add_argument(
        "--admission-batch-size",
        type=int,
        default=256,
        help="Maximum RequestRouter planned-admission batch per engine.",
    )
    parser.add_argument("--short-context-len", type=int, default=1_024)
    parser.add_argument("--long-context-len", type=int, default=428_033)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--loop-count",
        type=int,
        default=HIERARCHICAL_LOOP_COUNT,
        help="Must remain 16 for the production hierarchical contract.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-native-setup-logs", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not (Path(args.model) / "config.json").is_file():
        parser.error(f"model config not found below --model {args.model!r}")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if min(
        args.iterations,
        args.admission_iterations,
        args.admission_batch_size,
        args.block_size,
    ) <= 0:
        parser.error("iterations, admission batch size, and block size must be positive")
    if args.loop_count != HIERARCHICAL_LOOP_COUNT:
        parser.error(
            "hierarchical scheduler requires --loop-count "
            f"{HIERARCHICAL_LOOP_COUNT}"
        )
    if args.short_context_len < SP8_DEGREE:
        parser.error(f"--short-context-len must be at least {SP8_DEGREE}")
    if args.long_context_len <= args.short_context_len:
        parser.error("--long-context-len must exceed --short-context-len")
    if _bucket_sp_degree(
        DEEPSEEK_V3_BUCKET_POLICY, args.short_context_len
    ) != 1:
        parser.error("--short-context-len must select the DeepSeek-V3 SP1 bucket")
    if _bucket_sp_degree(
        DEEPSEEK_V3_BUCKET_POLICY, args.long_context_len
    ) != SP8_DEGREE:
        parser.error("--long-context-len must select the DeepSeek-V3 SP8 bucket")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    output_dir = args.output_dir or _default_output_dir()

    records = []
    total_cases = (
        len(args.logical_nodes)
        * len(args.batch_sizes)
        * len(args.scenarios)
    )
    case_index = 0
    for logical_nodes in args.logical_nodes:
        for batch_size in args.batch_sizes:
            for scenario_name in args.scenarios:
                case_index += 1
                case = HierarchicalProfileCase(
                    scenario=SCENARIOS[scenario_name],
                    logical_nodes=logical_nodes,
                    batch_size_per_gpu=batch_size,
                    short_context_len=args.short_context_len,
                    long_context_len=args.long_context_len,
                    block_size=args.block_size,
                    seed=args.seed,
                )
                print(
                    f"[{case_index}/{total_cases}] scenario={scenario_name} "
                    f"nodes={logical_nodes} gpus={case.logical_gpus} "
                    f"local_schedulers={case.local_scheduler_count} "
                    f"bs_per_gpu={batch_size} requests={case.total_requests}",
                    flush=True,
                )
                record = run_case(
                    case,
                    model=args.model,
                    warmup_iterations=args.warmup_iterations,
                    measured_iterations=args.iterations,
                    admission_iterations=args.admission_iterations,
                    admission_batch_size=args.admission_batch_size,
                    suppress_native_setup_logs=not args.show_native_setup_logs,
                )
                records.append(record)
                admission = record["admission_ms"][
                    "scheduler_admission_critical_ms"
                ]["mean"]
                control_plane_admission = record["admission_ms"][
                    "modelled_admission_critical_ms"
                ]["mean"]
                decode = record["modelled_parallel_quantum_ms"]["mean"]
                print(
                    f"  admission scheduler-critical={admission:.3f} ms "
                    f"(control-plane={control_plane_admission:.3f} ms) | "
                    f"decode modelled-critical={decode:.3f} ms",
                    flush=True,
                )
                gc.collect()

    _annotate_weak_scaling(records)
    metadata = base_metadata(
        "nanodeploy-hierarchical-cpu-scheduler-scalability"
    )
    metadata.update(
        {
            "repo_root": str(REPO_ROOT),
            "model_config_path": str(Path(args.model) / "config.json"),
            "gpus_per_logical_node": GPUS_PER_LOGICAL_NODE,
            "implementation": {
                "admission_commit": (
                    "LocalScheduler.commit_planned_sequences (native C++)"
                ),
                "decode_scheduler": (
                    "LocalScheduler.cpp_scheduler.schedule (native C++)"
                ),
                "python_contract_bookkeeping": "diagnostic-only",
            },
            "timed_scope": (
                "primary scheduler-only admission is LocalScheduler planned "
                "commit after queue insertion; primary decode is native C++ "
                "Scheduler.schedule(); Router planning/receipt, snapshots, "
                "first-forward marking, and postprocess remain separate "
                "diagnostic phases"
            ),
            "excluded_scope": (
                "Config/LocalScheduler construction, Sequence construction, "
                "fake worker-result construction, real transport, Ray, RDMA, "
                "ModelRunner, CUDA, and GPU kernels"
            ),
            "critical_path_definition": (
                "LocalSchedulers run serially in this CPU harness. Modelled "
                "parallel scheduler-only metrics take the maximum local CPU "
                "cost for matching logical iterations and exclude network "
                "synchronization; modelled_admission_critical_ms retains the "
                "full Router control-plane path."
            ),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    json_path, csv_path = write_results(
        output_dir,
        stem="hierarchical_scheduler_overhead",
        metadata=metadata,
        records=records,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

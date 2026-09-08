#!/usr/bin/env python3
"""Profile mixed admission plus decode scheduler overhead.

This benchmark keeps a decode cohort in a steady state and injects an
admission cohort while those requests are running.  The default ratio preset
keeps ``BS/GPU=128`` and the two dynamic SP policies intentionally use
different admission:decode ratios:

* ``dynamic_sp8_1pct``: approximately 3:100;
* ``dynamic_sp8_5pct``: approximately 1:100.

The ratio is defined against the decode cohort, not against the combined
request count.  For SP8, one logical node has 1,024 decode requests, so the
default admission cohorts contain 31 and 10 requests respectively.  The
same per-local-scheduler cohort is used at 1, 2, 4, and 32 logical nodes.

The ``fig9_dpsk`` preset uses the Fig9 workload points instead: the decode
BS/GPU is taken from the measured point and the admission cohort is the
number of requests arriving during ``16 * 100 ms = 1.6 s`` per-GPU lifetime.
Its default topology is 4 and 32 logical nodes (32 and 256 GPUs).

Only the scheduler CPU boundaries are primary metrics.  Queue construction,
Router planning/receipts, hierarchical contract bookkeeping, transport, Ray,
CUDA, and model execution are excluded from those metrics and retained as
diagnostic fields where applicable.  Logical 32-node values are explicitly a
model of independent LocalScheduler replicas, as in the existing scalability
profiler; no 32-node deployment is claimed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanodeploy.engine.hierarchical_contract import (  # noqa: E402
    AddResultEvent,
    HIERARCHICAL_LOOP_COUNT,
)
from nanodeploy.engine.local_scheduler import LocalScheduler  # noqa: E402
from nanodeploy.router.admission_planner import AdmissionPlannerConfig  # noqa: E402
from nanodeploy.router.request_router import RequestRouter  # noqa: E402
from scripts.scheduler_overhead.profile_hierarchical_scheduler_scalability import (  # noqa: E402
    DEFAULT_MODEL,
    _RecordingAdmissionTransport,
    _build_commands_and_sequences,
    _engine_topologies,
    _new_config,
    _native_stderr,
    _reservation_sp_degree,
    _route_admission,
    _run_local_quantum,
)
from scripts.scheduler_overhead.profile_scheduler_scalability import (  # noqa: E402
    GPUS_PER_LOGICAL_NODE,
    ProfileCase,
    SCENARIOS,
    SP8_DEGREE,
    _advance_decode_state,
    _long_request_indices,
    _new_scheduler,
    _native_stderr as _central_native_stderr,
    _profile_sequence as _central_profile_sequence,
    _sp_histogram,
)
from scripts.decentralized_scalability.common import summarize  # noqa: E402


BS_PER_GPU = 128
MIXED_LOGICAL_NODES = (1, 2, 4, 32)
FIG9_LOGICAL_NODES = (4, 32)
MIXED_SCENARIOS = ("dynamic_sp8_1pct", "dynamic_sp8_5pct")
ADMISSION_RATIO_NUMERATOR = {
    "dynamic_sp8_1pct": 3,
    "dynamic_sp8_5pct": 1,
}
ADMISSION_RATIO_DENOMINATOR = 100
SHORT_CONTEXT_LEN = 1_024
LONG_CONTEXT_LEN = 428_033
BLOCK_SIZE = 64
LOOP_COUNT = HIERARCHICAL_LOOP_COUNT
FIG9_PRESET_NAME = "fig9_dpsk"
FIG9_RATE_REFERENCE_GPUS = 32
FIG9_TARGET_TPOT_MS = 100.0
FIG9_ADMISSION_HORIZON_S = LOOP_COUNT * FIG9_TARGET_TPOT_MS / 1_000.0


@dataclass(frozen=True, slots=True)
class Fig9PresetPoint:
    """One measured Fig9 rate/BS point used by the mixed profiler."""

    name: str
    scenario_name: str
    global_rate_32gpu_rps: float
    batch_size_per_gpu: int


FIG9_PRESET_POINTS = (
    Fig9PresetPoint("fig9_issue1_r80", "dynamic_sp8_1pct", 80.0, 135),
    Fig9PresetPoint("fig9_issue1_r90", "dynamic_sp8_1pct", 90.0, 173),
    Fig9PresetPoint("fig9_issue5_r40", "dynamic_sp8_5pct", 40.0, 56),
    Fig9PresetPoint("fig9_issue5_r45", "dynamic_sp8_5pct", 45.0, 58),
    Fig9PresetPoint("fig9_issue5_r50", "dynamic_sp8_5pct", 50.0, 59),
)


@dataclass(frozen=True, slots=True)
class MixedProfileCase:
    """One mixed workload with a fixed decode cohort and injected admissions."""

    scenario_name: str
    logical_nodes: int
    batch_size_per_gpu: int = BS_PER_GPU
    short_context_len: int = SHORT_CONTEXT_LEN
    long_context_len: int = LONG_CONTEXT_LEN
    block_size: int = BLOCK_SIZE
    seed: int = 0
    preset_name: str = "ratio_bs128"
    preset_point_name: str | None = None
    global_rate_32gpu_rps: float | None = None
    admission_horizon_s: float | None = None

    def __post_init__(self) -> None:
        if self.scenario_name not in MIXED_SCENARIOS:
            raise ValueError(
                f"mixed benchmark supports only {MIXED_SCENARIOS}; "
                f"got {self.scenario_name!r}"
            )
        if self.logical_nodes not in MIXED_LOGICAL_NODES:
            raise ValueError(
                f"mixed benchmark supports logical nodes "
                f"{MIXED_LOGICAL_NODES}; got {self.logical_nodes}"
            )
        if self.batch_size_per_gpu <= 0:
            raise ValueError("batch_size_per_gpu must be positive")
        if self.short_context_len < SP8_DEGREE:
            raise ValueError("short context must be at least SP8 degree")
        if self.long_context_len <= self.short_context_len:
            raise ValueError("long context must exceed short context")
        if self.preset_name == FIG9_PRESET_NAME:
            if self.global_rate_32gpu_rps is None:
                raise ValueError("Fig9 preset requires global_rate_32gpu_rps")
            if self.admission_horizon_s is None or self.admission_horizon_s <= 0:
                raise ValueError(
                    "Fig9 preset requires a positive admission_horizon_s"
                )
        elif self.global_rate_32gpu_rps is not None or self.admission_horizon_s is not None:
            raise ValueError(
                "rate and admission horizon fields are only valid for the Fig9 preset"
            )
        if self.is_fig9_preset and not self.preset_point_name:
            raise ValueError("Fig9 preset requires preset_point_name")

    @property
    def scenario(self):
        return SCENARIOS[self.scenario_name]

    @property
    def logical_gpus(self) -> int:
        return self.logical_nodes * GPUS_PER_LOGICAL_NODE

    @property
    def local_scheduler_count(self) -> int:
        # Both requested policies use the SP8 topology: one LocalScheduler per
        # logical node, with eight SP ranks in each scheduler.
        return self.logical_nodes

    @property
    def attention_sp(self) -> int:
        return SP8_DEGREE

    @property
    def decode_requests(self) -> int:
        return self.logical_gpus * self.batch_size_per_gpu

    @property
    def decode_requests_per_local_scheduler(self) -> int:
        return self.decode_requests // self.local_scheduler_count

    @property
    def admission_ratio_numerator(self) -> int:
        return ADMISSION_RATIO_NUMERATOR[self.scenario_name]

    @property
    def is_fig9_preset(self) -> bool:
        return self.preset_name == FIG9_PRESET_NAME

    @property
    def admission_rate_per_gpu_rps(self) -> float | None:
        if not self.is_fig9_preset:
            return None
        return self.global_rate_32gpu_rps / FIG9_RATE_REFERENCE_GPUS

    @property
    def admission_requests_per_gpu(self) -> float | None:
        if not self.is_fig9_preset:
            return None
        return self.admission_rate_per_gpu_rps * self.admission_horizon_s

    @property
    def effective_global_admission_rate_rps(self) -> float | None:
        if not self.is_fig9_preset:
            return None
        return self.admission_rate_per_gpu_rps * self.logical_gpus

    @property
    def admission_requests_per_local_scheduler(self) -> int:
        if self.is_fig9_preset:
            # Aggregate at the SP8 scheduler boundary.  This avoids rounding
            # fractional per-GPU cohorts (for example 4.5 requests/GPU) and
            # is exact for the Fig9 points at the supported 4/32 node sizes.
            raw = (
                self.admission_rate_per_gpu_rps
                * GPUS_PER_LOGICAL_NODE
                * self.admission_horizon_s
            )
            return max(1, round(raw))
        # Round per local scheduler rather than globally.  This keeps every
        # local scheduler's capacity integral at all four node counts.
        return max(
            1,
            round(
                self.decode_requests_per_local_scheduler
                * self.admission_ratio_numerator
                / ADMISSION_RATIO_DENOMINATOR
            ),
        )

    @property
    def admission_requests(self) -> int:
        return (
            self.local_scheduler_count
            * self.admission_requests_per_local_scheduler
        )

    @property
    def total_requests(self) -> int:
        return self.decode_requests + self.admission_requests

    @property
    def requests_per_local_scheduler(self) -> int:
        return (
            self.decode_requests_per_local_scheduler
            + self.admission_requests_per_local_scheduler
        )

    @property
    def admission_to_decode_ratio(self) -> float:
        return self.admission_requests / self.decode_requests

    @property
    def admission_to_decode_ratio_target(self) -> str:
        if self.is_fig9_preset:
            return (
                f"{self.global_rate_32gpu_rps:g} req/s@32GPU, "
                f"horizon={self.admission_horizon_s:g}s"
            )
        return (
            f"{self.admission_ratio_numerator}:"
            f"{ADMISSION_RATIO_DENOMINATOR}"
        )

    @property
    def expected_sp8_requests(self) -> int:
        return round(self.total_requests * self.scenario.sp8_ratio)

    @property
    def expected_sp1_requests(self) -> int:
        return self.total_requests - self.expected_sp8_requests

    @property
    def topology_scope(self) -> str:
        if self.logical_nodes in (1, 2, 4):
            return "complete_production_topology_cpu_model"
        return "logical_independent_local_scheduler_replica_model"

    @property
    def deployment_topology_supported(self) -> bool:
        return self.logical_nodes in (1, 2, 4)

    def dynamic_long_indices(self) -> set[int]:
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

    def central_case(self) -> ProfileCase:
        # ProfileCase retains BS/GPU=128 as the public label while its
        # override lets the central scheduler reserve the injected cohort.
        return ProfileCase(
            scenario=self.scenario,
            logical_nodes=self.logical_nodes,
            batch_size_per_gpu=self.batch_size_per_gpu,
            short_context_len=self.short_context_len,
            long_context_len=self.long_context_len,
            block_size=self.block_size,
            loop_count=LOOP_COUNT,
            seed=self.seed,
            request_count_override=self.total_requests,
        )


def build_case_matrix(
    logical_nodes: Sequence[int] = MIXED_LOGICAL_NODES,
    scenarios: Sequence[str] = MIXED_SCENARIOS,
    batch_size_per_gpu: int = BS_PER_GPU,
) -> list[MixedProfileCase]:
    if batch_size_per_gpu != BS_PER_GPU:
        raise ValueError(
            f"mixed comparison is fixed at BS/GPU={BS_PER_GPU}; "
            f"got {batch_size_per_gpu}"
        )
    unknown_nodes = sorted(set(logical_nodes) - set(MIXED_LOGICAL_NODES))
    if unknown_nodes:
        raise ValueError(f"unsupported mixed logical nodes: {unknown_nodes}")
    if any(nodes <= 0 for nodes in logical_nodes):
        raise ValueError("logical node counts must be positive")
    if len(set(logical_nodes)) != len(logical_nodes):
        raise ValueError("logical node counts must not contain duplicates")
    unknown_scenarios = sorted(set(scenarios) - set(MIXED_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(f"unsupported mixed scenarios: {unknown_scenarios}")
    if not logical_nodes or not scenarios:
        raise ValueError("at least one node and scenario are required")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("mixed scenarios must not contain duplicates")
    return [
        MixedProfileCase(
            scenario_name=scenario,
            logical_nodes=nodes,
            batch_size_per_gpu=batch_size_per_gpu,
        )
        for nodes in logical_nodes
        for scenario in scenarios
    ]


def build_fig9_case_matrix(
    logical_nodes: Sequence[int] = FIG9_LOGICAL_NODES,
    scenarios: Sequence[str] = MIXED_SCENARIOS,
    *,
    admission_horizon_s: float = FIG9_ADMISSION_HORIZON_S,
    seed: int = 0,
) -> list[MixedProfileCase]:
    """Build the Fig9 rate/BS matrix for the requested logical nodes.

    ``global_rate_32gpu_rps`` is the source Fig9 rate.  The rate is normalized
    by 32 GPUs and then scaled with the number of GPUs in the requested
    topology.  Admission is one aggregate cohort covering the configured
    horizon, rather than a stream of per-second events.
    """

    if admission_horizon_s <= 0:
        raise ValueError("admission_horizon_s must be positive")
    unknown_nodes = sorted(set(logical_nodes) - set(MIXED_LOGICAL_NODES))
    if unknown_nodes:
        raise ValueError(f"unsupported mixed logical nodes: {unknown_nodes}")
    if not logical_nodes:
        raise ValueError("at least one logical node is required")
    if len(set(logical_nodes)) != len(logical_nodes):
        raise ValueError("logical node counts must not contain duplicates")
    unknown_scenarios = sorted(set(scenarios) - set(MIXED_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(f"unsupported mixed scenarios: {unknown_scenarios}")
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("mixed scenarios must not contain duplicates")
    points = [
        point for point in FIG9_PRESET_POINTS if point.scenario_name in scenarios
    ]
    return [
        MixedProfileCase(
            scenario_name=point.scenario_name,
            logical_nodes=nodes,
            batch_size_per_gpu=point.batch_size_per_gpu,
            seed=seed,
            preset_name=FIG9_PRESET_NAME,
            preset_point_name=point.name,
            global_rate_32gpu_rps=point.global_rate_32gpu_rps,
            admission_horizon_s=admission_horizon_s,
        )
        for nodes in logical_nodes
        for point in points
    ]


def _validate_sp_histogram(
    case: MixedProfileCase,
    schedule_result: object,
) -> dict[int, int]:
    histogram = _sp_histogram(schedule_result)
    scheduled = sum(histogram.values())
    if scheduled != case.total_requests:
        raise RuntimeError(
            f"scheduled {scheduled} requests, expected {case.total_requests}"
        )
    if histogram.get(SP8_DEGREE, 0) != case.expected_sp8_requests:
        raise RuntimeError(
            f"scheduled {histogram.get(SP8_DEGREE, 0)} SP8 requests, "
            f"expected {case.expected_sp8_requests}"
        )
    if histogram.get(1, 0) != case.expected_sp1_requests:
        raise RuntimeError(
            f"scheduled {histogram.get(1, 0)} SP1 requests, "
            f"expected {case.expected_sp1_requests}"
        )
    if set(histogram) - {1, SP8_DEGREE}:
        raise RuntimeError(f"unexpected SP degrees: {sorted(histogram)}")
    return histogram


def _run_central_admission_once(
    case: MixedProfileCase,
    *,
    warmup_iterations: int,
    measured_iterations: int,
    suppress_native_setup_logs: bool,
) -> tuple[object, dict[str, Any]]:
    central_case = case.central_case()
    profiled_decode_iterations = 2 + warmup_iterations + measured_iterations
    setup_begin = time.perf_counter_ns()
    with _central_native_stderr(suppress_native_setup_logs):
        scheduler = _new_scheduler(
            central_case,
            profiled_decode_iterations=profiled_decode_iterations,
        )
    long_indices = case.dynamic_long_indices()
    sequences = {
        request_id: _central_profile_sequence(
            context_len=(
                case.long_context_len
                if request_id in long_indices
                else case.short_context_len
            ),
            request_index=request_id,
            loop_count=LOOP_COUNT,
        )
        for request_id in range(case.total_requests)
    }
    for request_id in range(case.decode_requests):
        scheduler.add(sequences[request_id])
    initial = scheduler.schedule()
    initial_count = sum(len(per_dp) for per_dp in initial.dp_seqs)
    if not initial.is_prefill or initial_count != case.decode_requests:
        raise RuntimeError(
            "central mixed setup did not admit exactly the decode cohort: "
            f"prefill={initial.is_prefill}, count={initial_count}"
        )
    del initial
    # Put the baseline into a real decode state before injecting admissions.
    baseline_decode = scheduler.schedule()
    if baseline_decode.is_prefill:
        raise RuntimeError("central baseline unexpectedly remained in admission")
    _advance_decode_state(baseline_decode, LOOP_COUNT)
    del baseline_decode
    for request_id in range(case.decode_requests, case.total_requests):
        scheduler.add(sequences[request_id])
    setup_ms = (time.perf_counter_ns() - setup_begin) / 1_000_000.0

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        begin = time.perf_counter_ns()
        admission_result = scheduler.schedule()
        admission_ms = (time.perf_counter_ns() - begin) / 1_000_000.0
    finally:
        if gc_was_enabled:
            gc.enable()
    admitted = sum(len(per_dp) for per_dp in admission_result.dp_seqs)
    if not admission_result.is_prefill or admitted != case.admission_requests:
        raise RuntimeError(
            "central mixed admission returned an unexpected batch: "
            f"prefill={admission_result.is_prefill}, count={admitted}"
        )
    del admission_result
    return scheduler, {
        "setup_ms_excluded": setup_ms,
        "admission_scheduler_ms": admission_ms,
    }


def _profile_central_decode(
    case: MixedProfileCase,
    scheduler: object,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup_iterations):
        result = scheduler.schedule()
        if result.is_prefill:
            raise RuntimeError("central mixed warmup unexpectedly admitted work")
        _validate_sp_histogram(case, result)
        _advance_decode_state(result, LOOP_COUNT)
        del result
    samples_ms = []
    final_histogram = None
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for _ in range(measured_iterations):
            begin = time.perf_counter_ns()
            result = scheduler.schedule()
            elapsed = (time.perf_counter_ns() - begin) / 1_000_000.0
            if result.is_prefill:
                raise RuntimeError("central mixed decode returned admission")
            final_histogram = _validate_sp_histogram(case, result)
            samples_ms.append(elapsed)
            _advance_decode_state(result, LOOP_COUNT)
            del result
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "decode_scheduler_ms": summarize(samples_ms),
        "actual_sp1_requests": (final_histogram or {}).get(1, 0),
        "actual_sp8_requests": (final_histogram or {}).get(SP8_DEGREE, 0),
    }


def _commit_hierarchical_batches(
    schedulers: Sequence[LocalScheduler],
    transports: dict[int, _RecordingAdmissionTransport],
    sequences_by_id: dict[int, Any],
    *,
    admission_versions: list[int],
    timed: bool,
) -> tuple[list[AddResultEvent], list[float], int, int]:
    """Stage a routed batch, commit it locally, and return receipt events."""

    for scheduler, transport in zip(
        schedulers, transports.values(), strict=True
    ):
        for commands, _reservations in transport.planned_batches:
            sequences = tuple(
                sequences_by_id[command.request_id]
                for command in commands
            )
            results = scheduler.stage_batch(commands, sequences)
            if not all(result.accepted for result in results):
                raise RuntimeError("LocalScheduler rejected staged mixed admission")

    commit_samples: list[float] = []
    events: list[AddResultEvent] = []
    actual_sp1 = 0
    actual_sp8 = 0
    for engine_id, (scheduler, transport) in enumerate(
        zip(schedulers, transports.values(), strict=True)
    ):
        elapsed_total = 0.0
        for commands, reservations in transport.planned_batches:
            sequences = tuple(
                sequences_by_id[command.request_id]
                for command in commands
            )
            begin = time.perf_counter_ns() if timed else 0
            results = scheduler.commit_planned_batch(
                commands,
                reservations,
                sequences,
                pre_staged=True,
            )
            if timed:
                elapsed_total += (time.perf_counter_ns() - begin) / 1_000_000.0
            if not all(result.accepted for result in results):
                raise RuntimeError(
                    f"LocalScheduler {engine_id} rejected mixed admission"
                )
            for command, reservation in zip(commands, reservations, strict=True):
                degree = _reservation_sp_degree(reservation)
                if degree == 1:
                    actual_sp1 += 1
                elif degree == SP8_DEGREE:
                    actual_sp8 += 1
                else:
                    raise RuntimeError(f"unexpected SP degree {degree}")
                admission_versions[engine_id] += 1
                events.append(
                    AddResultEvent(
                        request_id=command.request_id,
                        engine_id=engine_id,
                        accepted=True,
                        admission_version=admission_versions[engine_id],
                    )
                )
        commit_samples.append(elapsed_total)
        transport.planned_batches.clear()
    return events, commit_samples, actual_sp1, actual_sp8


def _run_hierarchical_admission_once(
    case: MixedProfileCase,
    *,
    model: str,
    warmup_iterations: int,
    measured_iterations: int,
    admission_batch_size: int,
    suppress_native_setup_logs: bool,
) -> tuple[list[LocalScheduler], dict[str, Any]]:
    max_tokens = (
        2 + warmup_iterations + measured_iterations
    ) * HIERARCHICAL_LOOP_COUNT
    setup_begin = time.perf_counter()
    # _new_config/_engine_topologies intentionally accept the same case
    # protocol as HierarchicalProfileCase; this mixed case supplies the
    # larger per-local request capacity while retaining BS/GPU=128 metadata.
    config = _new_config(  # type: ignore[arg-type]
        case,  # type: ignore[arg-type]
        model=model,
        max_tokens=max_tokens,
        admission_batch_size=admission_batch_size,
    )
    with _native_stderr(suppress_native_setup_logs):
        schedulers = [
            LocalScheduler(config, topology)
            for topology in _engine_topologies(case, config)  # type: ignore[arg-type]
        ]
    if any(
        not hasattr(scheduler.cpp_scheduler, "commit_planned_sequences")
        for scheduler in schedulers
    ):
        raise RuntimeError(
            "mixed benchmark requires the rebuilt native bulk admission "
            "entry point commit_planned_sequences"
        )
    commands, sequences = _build_commands_and_sequences(  # type: ignore[arg-type]
        case, max_tokens=max_tokens, vocab_size=config.hf_config.vocab_size
    )
    setup_ms = (time.perf_counter() - setup_begin) * 1000.0
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
    snapshots = [
        scheduler.load_snapshot(wave_id=1, quantum_id=0)
        for scheduler in schedulers
    ]
    router_load_ms_begin = time.perf_counter()
    router.record_loads(snapshots)
    router_load_ms = (time.perf_counter() - router_load_ms_begin) * 1000.0
    versions = [0] * case.local_scheduler_count

    baseline_commands = commands[: case.decode_requests]
    pending_commands = commands[case.decode_requests :]
    route_begin = time.perf_counter()
    _route_admission(router, baseline_commands)
    baseline_route_ms = (time.perf_counter() - route_begin) * 1000.0
    baseline_events, _, baseline_sp1, baseline_sp8 = _commit_hierarchical_batches(
        schedulers,
        transports,
        sequences,
        admission_versions=versions,
        timed=False,
    )
    router.record_add_results(baseline_events)
    if len(baseline_events) != case.decode_requests:
        raise RuntimeError("Router did not commit the full decode baseline")

    # Advance the baseline once so the injected cohort is admitted alongside
    # requests that are genuinely in decode, not merely prefill-complete.
    for scheduler in schedulers:
        _run_local_quantum(scheduler, quantum_id=0)

    updated_load_begin = time.perf_counter()
    router.record_loads(
        [
            replace(
                scheduler.load_snapshot(wave_id=1, quantum_id=1),
                admission_version=versions[engine_id],
                ingress_version=transports[engine_id].ingress_version,
            )
            for engine_id, scheduler in enumerate(schedulers)
        ]
    )
    updated_router_load_ms = (
        time.perf_counter() - updated_load_begin
    ) * 1000.0
    route_begin = time.perf_counter()
    try:
        _route_admission(router, pending_commands)
    except RuntimeError as exc:
        raise RuntimeError(
            f"mixed pending admission planning failed: {exc}; "
            f"router_metrics={router.admission_metrics()}"
        ) from exc
    pending_route_ms = (time.perf_counter() - route_begin) * 1000.0

    pending_events, commit_samples, pending_sp1, pending_sp8 = (
        _commit_hierarchical_batches(
            schedulers,
            transports,
            sequences,
            admission_versions=versions,
            timed=True,
        )
    )
    pending_commit_sum_ms = sum(commit_samples)
    add_result_begin = time.perf_counter()
    router.record_add_results(pending_events)
    router_add_result_ms = (time.perf_counter() - add_result_begin) * 1000.0
    if len(pending_events) != case.admission_requests:
        raise RuntimeError("Router did not commit the full mixed admission cohort")
    if router.pending_ingress_count or router.pending_add_count:
        raise RuntimeError("Router retained mixed admission work")
    if router.active_count != case.total_requests:
        raise RuntimeError(
            f"Router active count {router.active_count} != {case.total_requests}"
        )
    if baseline_sp8 + pending_sp8 != case.expected_sp8_requests:
        raise RuntimeError("mixed hierarchical SP8 placement count mismatch")
    if baseline_sp1 + pending_sp1 != case.expected_sp1_requests:
        raise RuntimeError("mixed hierarchical SP1 placement count mismatch")
    return schedulers, {
        "setup_ms_excluded": setup_ms,
        "router_load_ms": router_load_ms,
        "updated_router_load_ms": updated_router_load_ms,
        "baseline_route_ms_excluded": baseline_route_ms,
        "pending_route_ms": pending_route_ms,
        "pending_commit_sum_ms": pending_commit_sum_ms,
        "pending_commit_critical_ms": max(commit_samples, default=0.0),
        "router_add_result_ms": router_add_result_ms,
        "full_control_plane_ms": (
            updated_router_load_ms
            + pending_route_ms
            + pending_commit_sum_ms
            + router_add_result_ms
        ),
        "actual_sp1_requests": baseline_sp1 + pending_sp1,
        "actual_sp8_requests": baseline_sp8 + pending_sp8,
    }


def _profile_hierarchical_decode(
    case: MixedProfileCase,
    schedulers: Sequence[LocalScheduler],
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    per_engine_samples: list[list[float]] = []
    for scheduler in schedulers:
        for iteration in range(warmup_iterations):
            _run_local_quantum(scheduler, quantum_id=1 + iteration)
        samples = [
            _run_local_quantum(
                scheduler,
                quantum_id=1 + warmup_iterations + iteration,
            )["scheduler_cpu_ms"]
            for iteration in range(measured_iterations)
        ]
        snapshot = scheduler.load_snapshot(
            wave_id=1,
            quantum_id=1 + warmup_iterations + measured_iterations,
        )
        if scheduler.is_finished() or snapshot.preemption_count != 0:
            raise RuntimeError("hierarchical mixed decode state did not remain live")
        if len(scheduler._records) != case.requests_per_local_scheduler:
            raise RuntimeError("hierarchical mixed decode request count changed")
        per_engine_samples.append(samples)

    parallel_samples = [
        max(
            engine_samples[iteration]
            for engine_samples in per_engine_samples
        )
        for iteration in range(measured_iterations)
    ]
    aggregate_samples = [
        sum(engine_samples[iteration] for engine_samples in per_engine_samples)
        for iteration in range(measured_iterations)
    ]
    return {
        "decode_scheduler_ms": summarize(parallel_samples),
        "decode_aggregate_local_cpu_ms": summarize(aggregate_samples),
        "per_engine_decode_scheduler_ms": [
            summarize(samples) for samples in per_engine_samples
        ],
        "actual_sp1_requests": case.expected_sp1_requests,
        "actual_sp8_requests": case.expected_sp8_requests,
    }


def _mean_summary(samples: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(sample[key]) for sample in samples]
    return summarize(values)


def run_case(
    case: MixedProfileCase,
    *,
    model: str = str(DEFAULT_MODEL),
    warmup_iterations: int = 3,
    measured_iterations: int = 10,
    admission_iterations: int = 1,
    admission_batch_size: int = 256,
    suppress_native_setup_logs: bool = True,
) -> dict[str, Any]:
    """Run one central/decentralized mixed case without external resources."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measured_iterations <= 0 or admission_iterations <= 0:
        raise ValueError("measured/admission iterations must be positive")
    if admission_batch_size <= 0:
        raise ValueError("admission_batch_size must be positive")

    central_admission_samples: list[dict[str, Any]] = []
    retained_central = None
    for index in range(admission_iterations):
        scheduler, metrics = _run_central_admission_once(
            case,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            suppress_native_setup_logs=suppress_native_setup_logs,
        )
        central_admission_samples.append(metrics)
        if index == admission_iterations - 1:
            retained_central = scheduler
        else:
            del scheduler
            gc.collect()
    if retained_central is None:
        raise RuntimeError("central mixed admission did not retain a scheduler")
    central_decode = _profile_central_decode(
        case,
        retained_central,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )

    decentralized_admission_samples: list[dict[str, Any]] = []
    retained_decentralized = None
    for index in range(admission_iterations):
        schedulers, metrics = _run_hierarchical_admission_once(
            case,
            model=model,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            admission_batch_size=admission_batch_size,
            suppress_native_setup_logs=suppress_native_setup_logs,
        )
        decentralized_admission_samples.append(metrics)
        if index == admission_iterations - 1:
            retained_decentralized = schedulers
        else:
            del schedulers
            gc.collect()
    if retained_decentralized is None:
        raise RuntimeError(
            "decentralized mixed admission did not retain schedulers"
        )
    decentralized_decode = _profile_hierarchical_decode(
        case,
        retained_decentralized,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )

    central_admission = _mean_summary(
        central_admission_samples, "admission_scheduler_ms"
    )
    decentralized_admission = {
        "scheduler_critical_ms": _mean_summary(
            decentralized_admission_samples, "pending_commit_critical_ms"
        ),
        "sum_ms": _mean_summary(
            decentralized_admission_samples, "pending_commit_sum_ms"
        ),
        "router_plan_receipt_ms": _mean_summary(
            decentralized_admission_samples, "pending_route_ms"
        ),
        "router_load_ms": _mean_summary(
            decentralized_admission_samples, "router_load_ms"
        ),
        "updated_router_load_ms": _mean_summary(
            decentralized_admission_samples, "updated_router_load_ms"
        ),
        "router_add_result_ms": _mean_summary(
            decentralized_admission_samples, "router_add_result_ms"
        ),
        "full_control_plane_ms": _mean_summary(
            decentralized_admission_samples, "full_control_plane_ms"
        ),
    }
    central_passed = (
        central_decode["actual_sp1_requests"] == case.expected_sp1_requests
        and central_decode["actual_sp8_requests"] == case.expected_sp8_requests
    )
    decentralized_passed = (
        all(
            scheduler.load_snapshot(wave_id=1, quantum_id=0).preemption_count == 0
            for scheduler in retained_decentralized
        )
        and decentralized_decode["actual_sp1_requests"] == case.expected_sp1_requests
        and decentralized_decode["actual_sp8_requests"] == case.expected_sp8_requests
    )
    central_decode_mean = float(central_decode["decode_scheduler_ms"]["mean"])
    decentral_decode_mean = float(
        decentralized_decode["decode_scheduler_ms"]["mean"]
    )
    central_admission_mean = float(central_admission["mean"])
    decentral_admission_mean = float(
        decentralized_admission["scheduler_critical_ms"]["mean"]
    )
    return {
        "logical_nodes": case.logical_nodes,
        "logical_gpus": case.logical_gpus,
        "scenario": case.scenario_name,
        "preset": case.preset_name,
        "preset_point": case.preset_point_name,
        "batch_size_per_gpu": case.batch_size_per_gpu,
        "decode_requests": case.decode_requests,
        "admission_requests": case.admission_requests,
        "total_requests": case.total_requests,
        "admission_to_decode_ratio": case.admission_to_decode_ratio,
        "admission_to_decode_ratio_target": case.admission_to_decode_ratio_target,
        "global_rate_32gpu_rps": case.global_rate_32gpu_rps,
        "admission_rate_per_gpu_rps": case.admission_rate_per_gpu_rps,
        "admission_requests_per_gpu": case.admission_requests_per_gpu,
        "admission_horizon_s": case.admission_horizon_s,
        "effective_global_admission_rate_rps": (
            case.effective_global_admission_rate_rps
        ),
        "admission_requests_per_local_scheduler": (
            case.admission_requests_per_local_scheduler
        ),
        "expected_sp1_requests": case.expected_sp1_requests,
        "expected_sp8_requests": case.expected_sp8_requests,
        "topology_scope": case.topology_scope,
        "deployment_topology_supported": case.deployment_topology_supported,
        "loop_count": LOOP_COUNT,
        "centralized": {
            "admission_scheduler_ms": central_admission,
            "decode_scheduler_ms": central_decode["decode_scheduler_ms"],
            "actual_sp1_requests": central_decode["actual_sp1_requests"],
            "actual_sp8_requests": central_decode["actual_sp8_requests"],
            "case_passed": central_passed,
        },
        "decentralized": {
            "admission": decentralized_admission,
            "decode_scheduler_ms": decentralized_decode["decode_scheduler_ms"],
            "decode_aggregate_local_cpu_ms": decentralized_decode[
                "decode_aggregate_local_cpu_ms"
            ],
            "actual_sp1_requests": decentralized_decode["actual_sp1_requests"],
            "actual_sp8_requests": decentralized_decode["actual_sp8_requests"],
            "case_passed": decentralized_passed,
        },
        "admission_decentralized_over_centralized_ratio": (
            decentral_admission_mean / central_admission_mean
        ),
        "decode_decentralized_over_centralized_ratio": (
            decentral_decode_mean / central_decode_mean
        ),
        "centralized_raw_admission_samples": central_admission_samples,
        "decentralized_raw_admission_samples": decentralized_admission_samples,
        "centralized_decode_mean_ms_per_step": central_decode_mean / LOOP_COUNT,
        "decentralized_decode_mean_ms_per_step": decentral_decode_mean / LOOP_COUNT,
    }


def _metadata(
    args: argparse.Namespace,
    *,
    case_count: int,
) -> dict[str, Any]:
    return {
        "benchmark": "nanodeploy-bs128-mixed-admission-decode",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "repo_root": str(REPO_ROOT),
        "preset": args.preset,
        "gpus_per_logical_node": GPUS_PER_LOGICAL_NODE,
        "batch_size_per_gpu": (
            BS_PER_GPU if args.preset == "ratio_bs128" else "Fig9 point-specific"
        ),
        "logical_nodes": list(args.logical_nodes),
        "scenarios": list(args.scenarios),
        "case_count": case_count,
        "result_cell_count": case_count * 4,
        "admission_ratio_definition": (
            "ratio preset: new admission requests / requests already in decode; "
            "Fig9 preset: rate_32gpu / 32 * admission_horizon_s per GPU"
        ),
        "admission_ratio_targets": {
            scenario: f"{numerator}:{ADMISSION_RATIO_DENOMINATOR}"
            for scenario, numerator in ADMISSION_RATIO_NUMERATOR.items()
        },
        "fig9_definition": {
            "rate_reference_gpus": FIG9_RATE_REFERENCE_GPUS,
            "target_tpot_ms": FIG9_TARGET_TPOT_MS,
            "loop_count": LOOP_COUNT,
            "default_admission_horizon_s": FIG9_ADMISSION_HORIZON_S,
            "scaling": "fixed admission rate per GPU; global rate scales with logical GPUs",
            "points": [
                {
                    "name": point.name,
                    "scenario": point.scenario_name,
                    "global_rate_32gpu_rps": point.global_rate_32gpu_rps,
                    "batch_size_per_gpu": point.batch_size_per_gpu,
                }
                for point in FIG9_PRESET_POINTS
            ],
        },
        "implementation": {
            "centralized_admission": "native C++ Scheduler.schedule() with running baseline",
            "centralized_decode": "native C++ Scheduler.schedule()",
            "decentralized_admission": (
                "LocalScheduler.commit_planned_batch: Python placement validation "
                "plus native C++ bulk state commit"
            ),
            "decentralized_decode": "LocalScheduler.cpp_scheduler.schedule() (native C++)",
            "python_contract_bookkeeping": "diagnostic/setup only",
        },
        "timed_scope": {
            "centralized_admission": (
                "pending queue insertion excluded; one native Scheduler.schedule() "
                "while decode baseline is running"
            ),
            "decentralized_admission": (
                "pending stage/Router planning excluded; max local planned commit "
                "(placement validation plus native bulk state commit) while decode "
                "baseline is running"
            ),
            "decode": (
                "one native Scheduler.schedule() per local scheduler; decentralized "
                "primary value is ideal max(local) critical path"
            ),
        },
        "excluded_scope": (
            "scheduler/Sequence construction, baseline setup, queue insertion, "
            "Router planning/receipts, snapshots, contract bookkeeping, transport, "
            "Ray/RDMA/ZMQ, ModelRunner, CUDA, and GPU kernels"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def _write_results(
    output_dir: Path,
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    if not records:
        raise ValueError("cannot write an empty mixed benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mixed_admission_decode.json"
    csv_path = output_dir / "mixed_admission_decode.csv"
    html_path = output_dir / "report.html"
    readme_path = output_dir / "README.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "records": list(records)},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    rows = []
    for record in records:
        rows.append(
            {
                "logical_nodes": record["logical_nodes"],
                "logical_gpus": record["logical_gpus"],
                "preset": record["preset"],
                "preset_point": record["preset_point"],
                "scenario": record["scenario"],
                "batch_size_per_gpu": record["batch_size_per_gpu"],
                "global_rate_32gpu_rps": record["global_rate_32gpu_rps"],
                "admission_rate_per_gpu_rps": record[
                    "admission_rate_per_gpu_rps"
                ],
                "admission_requests_per_gpu": record[
                    "admission_requests_per_gpu"
                ],
                "admission_horizon_s": record["admission_horizon_s"],
                "effective_global_admission_rate_rps": record[
                    "effective_global_admission_rate_rps"
                ],
                "decode_requests": record["decode_requests"],
                "admission_requests": record["admission_requests"],
                "admission_to_decode_ratio": record["admission_to_decode_ratio"],
                "target_ratio": record["admission_to_decode_ratio_target"],
                "centralized_admission_mean_ms": record[
                    "centralized"
                ]["admission_scheduler_ms"]["mean"],
                "decentralized_admission_mean_ms": record[
                    "decentralized"
                ]["admission"]["scheduler_critical_ms"]["mean"],
                "centralized_decode_mean_ms": record["centralized"][
                    "decode_scheduler_ms"
                ]["mean"],
                "decentralized_decode_mean_ms": record["decentralized"][
                    "decode_scheduler_ms"
                ]["mean"],
                "centralized_decode_p99_ms": record["centralized"][
                    "decode_scheduler_ms"
                ]["p99"],
                "decentralized_decode_p99_ms": record["decentralized"][
                    "decode_scheduler_ms"
                ]["p99"],
                "admission_ratio_decentralized_over_centralized": record[
                    "admission_decentralized_over_centralized_ratio"
                ],
                "decode_ratio_decentralized_over_centralized": record[
                    "decode_decentralized_over_centralized_ratio"
                ],
                "centralized_passed": record["centralized"]["case_passed"],
                "decentralized_passed": record["decentralized"]["case_passed"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    html_path.write_text(_render_html(metadata, records), encoding="utf-8")
    passed_count = sum(
        bool(record["centralized"]["case_passed"] and record["decentralized"]["case_passed"])
        for record in records
    )
    if records[0]["preset"] == FIG9_PRESET_NAME:
        readme = (
            "# Fig9 mixed admission + decode\n\n"
            f"完成 case：{passed_count}/{len(records)}。\n\n"
            "Admission cohort = `rate_32gpu / 32 * admission_horizon_s` per GPU，"
            f"其中 `loop_count={LOOP_COUNT}`、目标 TPOT=100 ms、"
            f"默认 horizon={FIG9_ADMISSION_HORIZON_S:g} s。\n"
            "Rate 按每 GPU 固定，32 node 的全局 rate 随 GPU 数量线性放大。\n\n"
        )
    else:
        readme = (
            "# BS/GPU=128 mixed admission + decode\n\n"
            f"完成 case：{passed_count}/{len(records)}。\n\n"
            "比例定义为新 admission 请求 / 已在 decode 请求；\n"
            "dynamic_sp8_1pct 目标约 3:100，dynamic_sp8_5pct 目标约 1:100。\n\n"
        )
    readme += (
        "主指标是对称 scheduler-only 边界；Router、transport、Ray/RDMA、"
        "Python contract bookkeeping 和 GPU/model execution 不计入主指标。\n\n"
        "- `report.html`: 可读主报告\n"
        "- `mixed_admission_decode.csv`: 扁平对照表\n"
        "- `mixed_admission_decode.json`: 完整原始记录和诊断字段\n"
    )
    readme_path.write_text(readme, encoding="utf-8")
    return json_path, csv_path, html_path, readme_path


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, int)):
        return f"{value:.3f}"
    return html.escape(str(value))


def _render_html(
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> str:
    rows = []
    passed = 0
    for record in records:
        central = record["centralized"]
        decentral = record["decentralized"]
        ok = bool(central["case_passed"] and decentral["case_passed"])
        passed += ok
        rows.append(
            "<tr>"
            f"<td>{record['logical_nodes']}</td>"
            f"<td>{record['logical_gpus']}</td>"
            f"<td>{html.escape(record['preset_point'] or record['preset'])}</td>"
            f"<td>{html.escape(record['scenario'])}</td>"
            f"<td>{_fmt(record['batch_size_per_gpu'])}</td>"
            f"<td>{_fmt(record['global_rate_32gpu_rps'])}</td>"
            f"<td>{record['decode_requests']}</td>"
            f"<td>{record['admission_requests']} "
            f"({record['admission_to_decode_ratio_target']}; "
            f"actual {_fmt(record['admission_to_decode_ratio'])})</td>"
            f"<td>{_fmt(central['admission_scheduler_ms']['mean'])} / "
            f"{_fmt(decentral['admission']['scheduler_critical_ms']['mean'])}</td>"
            f"<td>{_fmt(central['decode_scheduler_ms']['mean'])} / "
            f"{_fmt(decentral['decode_scheduler_ms']['mean'])}</td>"
            f"<td>{_fmt(central['decode_scheduler_ms']['p99'])} / "
            f"{_fmt(decentral['decode_scheduler_ms']['p99'])}</td>"
            f"<td>{'PASS' if ok else 'FAIL'}</td>"
            "</tr>"
        )
    is_fig9 = records[0]["preset"] == FIG9_PRESET_NAME
    title = "Fig9 混合 admission + decode" if is_fig9 else "BS/GPU=128 混合 admission + decode"
    subtitle = (
        "Fig9 rate/BS points · 4/32 logical nodes · 16 loops · target TPOT 100 ms"
        if is_fig9
        else "dynamic_sp8_1pct / dynamic_sp8_5pct · 1/2/4/32 logical nodes"
    )
    admission_definition = (
        "Fig9 admission = rate_32gpu / 32 × 1.6 s per GPU; rate is scaled per GPU."
        if is_fig9
        else "Admission is defined as new requests / requests already in decode."
    )
    metadata_json = html.escape(json.dumps(metadata, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{margin:0;background:#f4f7f9;color:#17212b;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:28px 20px 52px}}
.hero,.card{{background:#fff;border:1px solid #d7e0e6;border-radius:12px;padding:18px 20px;margin-bottom:14px}}
.hero{{border-left:5px solid #075985}}h1{{margin:0 0 8px;font-size:28px}}
.muted{{color:#5c6b76}}.ok{{color:#087f5b;font-weight:700}}
table{{width:100%;border-collapse:collapse;margin:10px 0 15px}}
th,td{{border:1px solid #d7e0e6;padding:8px;text-align:left;vertical-align:top}}
th{{background:#edf3f6}}code{{background:#eef3f6;padding:1px 4px;border-radius:4px}}
pre{{background:#15232d;color:#e8f1f5;padding:12px;border-radius:8px;overflow:auto;font-size:12px}}
</style></head><body><main>
<section class="hero"><h1>{title}</h1>
<p class="muted">中心化 vs 去中心化 · {subtitle}</p>
<p class="ok">完整 case：{passed}/{len(records)} 通过。</p>
<p>每个 case 先让 decode cohort 运行一个 quantum，再注入 admission cohort。主表 admission 是中心化/去中心化对称的 scheduler-only critical path；decode 是原生 C++ Scheduler.schedule()，去中心化取 max(local) 的理想并行 critical path。JSON 另保留去中心化 full control-plane（Router load/plan/receipt + local commit）诊断。</p>
<p>{admission_definition} 32 节点是独立 LocalScheduler replica model，不是物理 32 节点 wall-clock。</p></section>
<section class="card"><h2>主结果（中心化 / 去中心化）</h2>
<table><tr><th>Nodes</th><th>GPUs</th><th>Preset point</th><th>策略</th><th>BS/GPU</th><th>Rate@32GPU</th><th>Decode cohort</th><th>Admission cohort</th><th>Admission Mean ms</th><th>Decode Mean ms</th><th>Decode P99 ms</th><th>状态</th></tr>
{''.join(rows)}</table></section>
<section class="card"><h2>测量边界</h2>
<ul><li>Admission：中心化计一个 native <code>Scheduler.schedule()</code>；去中心化计各 LocalScheduler planned commit（Python placement validation + native bulk state commit）的最大值。新请求入队、Router 规划/receipt、快照和 Python contract bookkeeping 均不计入主指标。</li>
<li>Decode：两边都计一个 native <code>Scheduler.schedule()</code>；去中心化报告 max(local) critical path，并保留 aggregate local CPU 到 JSON。</li>
<li>场景构造和 baseline admission/decode quantum 是 setup，单独标记为 excluded。</li></ul></section>
<section class="card"><h2>Metadata</h2><pre>{metadata_json}</pre></section>
</main></body></html>"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--preset",
        choices=("ratio_bs128", FIG9_PRESET_NAME),
        default="ratio_bs128",
        help="Workload preset (default: ratio_bs128; Fig9 uses 4/32 nodes).",
    )
    parser.add_argument(
        "--logical-nodes",
        type=lambda value: tuple(int(item.strip()) for item in value.split(",")),
        default=None,
        help="Comma-separated logical node counts (preset-dependent default).",
    )
    parser.add_argument(
        "--scenarios",
        type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()),
        default=None,
        help="Comma-separated dynamic scenarios (default: both mixed policies).",
    )
    parser.add_argument(
        "--admission-horizon-s",
        type=float,
        default=FIG9_ADMISSION_HORIZON_S,
        help="Fig9 admission cohort horizon in seconds (default: 1.6).",
    )
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--admission-iterations", type=int, default=1)
    parser.add_argument("--admission-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-native-setup-logs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.logical_nodes = args.logical_nodes or (
        FIG9_LOGICAL_NODES if args.preset == FIG9_PRESET_NAME else MIXED_LOGICAL_NODES
    )
    args.scenarios = args.scenarios or MIXED_SCENARIOS
    if not (Path(args.model) / "config.json").is_file():
        parser.error(f"model config not found below --model {args.model!r}")
    if args.warmup_iterations < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")
    if args.admission_iterations <= 0 or args.admission_batch_size <= 0:
        parser.error("admission iterations and batch size must be positive")
    try:
        if args.preset == FIG9_PRESET_NAME:
            cases = build_fig9_case_matrix(
                args.logical_nodes,
                args.scenarios,
                admission_horizon_s=args.admission_horizon_s,
                seed=args.seed,
            )
        else:
            if args.admission_horizon_s != FIG9_ADMISSION_HORIZON_S:
                parser.error("--admission-horizon-s is only meaningful for --preset fig9_dpsk")
            cases = build_case_matrix(
                args.logical_nodes,
                args.scenarios,
                batch_size_per_gpu=BS_PER_GPU,
            )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output_dir = args.output_dir or (
        REPO_ROOT
        / "bench_logs"
        / "scheduler_overhead"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_mixed")
    )
    records = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] preset={case.preset_point_name or case.preset_name} "
            f"scenario={case.scenario_name} "
            f"nodes={case.logical_nodes} gpus={case.logical_gpus} "
            f"decode={case.decode_requests} admission={case.admission_requests} "
            f"admission_per_gpu={case.admission_requests_per_gpu or 0:.3f} "
            f"ratio={case.admission_to_decode_ratio:.5f}",
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
        print(
            f"  admission={record['centralized']['admission_scheduler_ms']['mean']:.3f}/"
            f"{record['decentralized']['admission']['scheduler_critical_ms']['mean']:.3f} ms "
            f"decode={record['centralized']['decode_scheduler_ms']['mean']:.3f}/"
            f"{record['decentralized']['decode_scheduler_ms']['mean']:.3f} ms",
            flush=True,
        )
        gc.collect()
    metadata = _metadata(args, case_count=len(records))
    json_path, csv_path, html_path, readme_path = _write_results(
        output_dir, metadata, records
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

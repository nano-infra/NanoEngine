#!/usr/bin/env python3
"""Profile NanoDeploy's decentralized control-plane overhead.

This profiler is intentionally independent of the centralized Ray/DLSLime
transport benchmark and of the older decentralized scalability harnesses.  It
measures the two control paths introduced by the hierarchical scheduler:

* the per-quantum Gloo consensus between LocalEngine leaders; and
* persistent LocalEngine-to-Router ``FrontendEventBatch`` Ray flights,
  including production ``RequestRouter`` receipt processing.

Every Ray actor requests zero GPUs.  ModelRunner, CUDA, DLSLime/RDMA, model
kernels, and worker collectives are outside the measured scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import platform
import socket
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from ray.util.scheduling_strategies import (  # noqa: E402
    NodeAffinitySchedulingStrategy,
)

from nanodeploy.engine.hierarchical_contract import (  # noqa: E402
    AddResultEvent,
    FinishEvent,
    FirstScheduleEvent,
    FirstTokenEvent,
    FrontendEventBatch,
    LoadSnapshot,
    OwnerState,
    RankLoad,
)
from nanodeploy.router.request_router import (  # noqa: E402
    RequestOwner,
    RequestRouter,
)


PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)
COMPONENTS = ("consensus", "events")
SCALING_MODES = ("strong", "weak")
EVENT_MIXES = (
    "load",
    "add",
    "first_schedule",
    "first_token",
    "finish",
    "mixed",
)
ATTENTION_SP = 8
CONSENSUS_FIELDS = 8


@dataclass(frozen=True, slots=True)
class SampleStats:
    count: int
    mean: float
    stddev: float
    minimum: float
    p50: float
    p95: float
    p99: float
    maximum: float


@dataclass(frozen=True, slots=True)
class _EventFlight:
    ref: Any
    started_ns: int


class _NoopEngineTransport:
    """Registry-only transport used by receipt-side Router profiling."""

    def __init__(self, engine_id: int) -> None:
        self.engine_id = engine_id


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_nonnegative_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {value!r}"
        ) from exc
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "all values must be finite and non-negative"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_choices(
    value: str,
    *,
    choices: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values).difference(choices))
    if not values:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label} {unknown}; choose from {tuple(choices)}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{label} values must not repeat")
    return values


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(samples: Iterable[float]) -> SampleStats:
    values = tuple(float(sample) for sample in samples)
    if not values:
        raise ValueError("at least one sample is required")
    return SampleStats(
        count=len(values),
        mean=statistics.fmean(values),
        stddev=statistics.pstdev(values),
        minimum=min(values),
        p50=_percentile(values, 50.0),
        p95=_percentile(values, 95.0),
        p99=_percentile(values, 99.0),
        maximum=max(values),
    )


def _add_stats(
    record: dict[str, Any],
    prefix: str,
    samples: Iterable[float],
) -> None:
    stats = _stats(samples)
    for name, value in asdict(stats).items():
        record[f"{prefix}_{name}"] = value


def _clear_proxy_environment() -> dict[str, str]:
    removed = {
        name: os.environ[name]
        for name in PROXY_ENV_NAMES
        if name in os.environ
    }
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    return removed


def _accelerator_assignment() -> dict[str, list[str]]:
    return {
        key: list(values)
        for key, values in ray.get_runtime_context().get_accelerator_ids().items()
    }


def _node_sort_key(node: Mapping[str, Any]) -> tuple[bool, str, str]:
    resources = node.get("Resources", {})
    is_head = bool(node.get("IsHeadNode", False)) or (
        isinstance(resources, Mapping)
        and "node:__internal_head__" in resources
    )
    return (
        not is_head,
        str(node.get("NodeManagerAddress", "")),
        str(node.get("NodeID", "")),
    )


def _select_nodes(
    nodes: Sequence[Mapping[str, Any]],
    *,
    count: int,
    requested_ips: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    alive = [dict(node) for node in nodes if node.get("Alive", False)]
    if requested_ips:
        by_ip = {
            str(node.get("NodeManagerAddress", "")): node for node in alive
        }
        missing = [node_ip for node_ip in requested_ips if node_ip not in by_ip]
        if missing:
            raise RuntimeError(f"requested Ray nodes are not alive: {missing}")
        selected = [by_ip[node_ip] for node_ip in requested_ips]
    else:
        selected = sorted(alive, key=_node_sort_key)
    if len(selected) < count:
        raise RuntimeError(
            f"need {count} live Ray nodes, found {len(selected)}"
        )
    selected = selected[:count]
    if any(not str(node.get("NodeID", "")) for node in selected):
        raise RuntimeError("one or more selected nodes have no NodeID")
    return tuple(selected)


def _node_metadata(nodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "node_id": str(node.get("NodeID", "")),
            "hostname": str(node.get("NodeManagerHostname", "")),
            "address": str(node.get("NodeManagerAddress", "")),
            "is_head": not _node_sort_key(node)[0],
        }
        for node in nodes
    )


def _busy_wait_ms(duration_ms: float) -> None:
    if duration_ms <= 0:
        return
    deadline = time.perf_counter_ns() + int(duration_ms * 1_000_000)
    while time.perf_counter_ns() < deadline:
        pass


@ray.remote(num_cpus=1, num_gpus=0)
class _ConsensusActor:
    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()
        self.group_initialized = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "requested_gpus": 0,
            "assigned_accelerators": _accelerator_assignment(),
            "gloo_socket_ifname": os.getenv("GLOO_SOCKET_IFNAME"),
        }

    def initialize(self, init_method: str | None, timeout_s: float) -> None:
        if self.world_size == 1:
            if init_method is not None:
                raise ValueError("DP1 consensus must not initialize Gloo")
            return
        if init_method is None:
            raise ValueError("multi-engine consensus requires an init method")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(seconds=timeout_s),
        )
        self.group_initialized = True

    def profile(
        self,
        iterations: int,
        *,
        overlap_work_ms: float,
        straggler_rank: int,
        straggler_delay_ms: float,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for quantum_id in range(iterations):
            if self.world_size > 1 and self.rank == straggler_rank:
                _busy_wait_ms(straggler_delay_ms)
            arrival_perf_ns = time.perf_counter_ns()
            arrival_unix_ns = time.time_ns()
            if self.world_size == 1:
                _busy_wait_ms(overlap_work_ms)
                samples.append(
                    {
                        "rank": self.rank,
                        "quantum_id": quantum_id,
                        "arrival_unix_ns": arrival_unix_ns,
                        "submit_ms": 0.0,
                        "overlap_window_ms": 0.0,
                        "exposed_wait_ms": 0.0,
                        "rendezvous_ms": 0.0,
                    }
                )
                continue

            local = torch.tensor(
                [
                    1,
                    -1,
                    quantum_id,
                    -quantum_id,
                    1,
                    -1,
                    arrival_unix_ns,
                    -arrival_unix_ns,
                ],
                dtype=torch.int64,
                device="cpu",
            )
            reduced = local.clone()
            submit_begin_ns = time.perf_counter_ns()
            work = dist.all_reduce(
                reduced,
                op=dist.ReduceOp.MAX,
                async_op=True,
            )
            submit_end_ns = time.perf_counter_ns()
            _busy_wait_ms(overlap_work_ms)
            wait_begin_ns = time.perf_counter_ns()
            work.wait()
            finished_ns = time.perf_counter_ns()
            values = tuple(int(value) for value in reduced.tolist())
            minimum = (-values[1], -values[3], -values[5])
            maximum = (values[0], values[2], values[4])
            if minimum != maximum or maximum != (1, quantum_id, 1):
                raise RuntimeError(
                    "Gloo consensus returned inconsistent wave/quantum state: "
                    f"rank={self.rank}, min={minimum}, max={maximum}"
                )
            samples.append(
                {
                    "rank": self.rank,
                    "quantum_id": quantum_id,
                    "arrival_unix_ns": arrival_unix_ns,
                    "submit_ms": (submit_end_ns - submit_begin_ns) / 1_000_000,
                    "overlap_window_ms": (
                        wait_begin_ns - submit_begin_ns
                    )
                    / 1_000_000,
                    "exposed_wait_ms": (
                        finished_ns - wait_begin_ns
                    )
                    / 1_000_000,
                    "rendezvous_ms": (
                        finished_ns - arrival_perf_ns
                    )
                    / 1_000_000,
                }
            )
        return samples

    def close(self) -> None:
        if self.group_initialized and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            self.group_initialized = False


def _rank_loads(engine_id: int, running: int) -> tuple[RankLoad, ...]:
    per_rank = math.ceil(running / ATTENTION_SP) if running else 0
    return tuple(
        RankLoad(
            global_rank=engine_id * ATTENTION_SP + sp_idx,
            sp_idx=sp_idx,
            tp_idx=0,
            master_batch_size=per_rank,
            active_master_requests=per_rank,
            free_blocks=1_000_000 - running,
            total_blocks=1_000_000,
            master_assignments=per_rank,
            mastered_decode_tokens=per_rank * 16,
            active_receiver_requests=running,
            active_dispatched_tokens=running * 8_000,
        )
        for sp_idx in range(ATTENTION_SP)
    )


def _request_ids(
    engine_id: int, quantum_id: int, event_count: int
) -> tuple[int, ...]:
    base = (engine_id + 1) * 1_000_000_000_000 + quantum_id * 1_000_000
    return tuple(base + offset for offset in range(event_count))


def _build_event_batch(
    *,
    engine_id: int,
    quantum_id: int,
    event_count: int,
    event_mix: str,
) -> FrontendEventBatch:
    if event_mix not in EVENT_MIXES:
        raise ValueError(f"unsupported event mix {event_mix!r}")
    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    request_ids = _request_ids(engine_id, quantum_id, event_count)
    include_add = event_mix in {"add", "mixed"}
    include_first_schedule = event_mix in {"first_schedule", "mixed"}
    include_first_token = event_mix in {"first_token", "mixed"}
    include_finish = event_mix in {"finish", "mixed"}
    running = 0 if event_mix == "load" else event_count
    return FrontendEventBatch(
        engine_id=engine_id,
        load=LoadSnapshot(
            engine_id=engine_id,
            ready=True,
            waiting=0,
            running=running,
            free_blocks_min=1_000_000 - running,
            wave_id=1,
            quantum_id=quantum_id,
            admission_version=quantum_id + 1,
            ingress_version=quantum_id + 1,
            capacity_epoch=quantum_id,
            useful_real_batch_size=running,
            decode_quantum_count=quantum_id,
            rank_loads=_rank_loads(engine_id, running),
        ),
        add_results=tuple(
            AddResultEvent(
                request_id=request_id,
                engine_id=engine_id,
                accepted=True,
                admission_version=quantum_id + 1,
                local_planned_queue_ms=0.1,
                local_admission_ms=0.2,
            )
            for request_id in request_ids
        )
        if include_add
        else (),
        first_schedule_events=tuple(
            FirstScheduleEvent(
                request_id=request_id,
                engine_id=engine_id,
                local_scheduler_queue_ms=0.1,
            )
            for request_id in request_ids
        )
        if include_first_schedule
        else (),
        first_token_events=tuple(
            FirstTokenEvent(
                request_id=request_id,
                engine_id=engine_id,
                generated_count=1,
            )
            for request_id in request_ids
        )
        if include_first_token
        else (),
        finish_events=tuple(
            FinishEvent(
                request_id=request_id,
                generated_count=16,
                status="FINISHED",
                engine_id=engine_id,
                first_forward_to_terminal_ms=1.0,
                final_quantum_execute_ms=1.0,
            )
            for request_id in request_ids
        )
        if include_finish
        else (),
    )


@ray.remote(num_cpus=1, num_gpus=0)
class _FrontendEventActor:
    def __init__(self, engine_id: int) -> None:
        self.engine_id = engine_id
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()
        self.event_count = 0
        self.event_mix = "load"
        self.delay_ms = 0.0
        self.quantum_id = 0

    def descriptor(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "requested_gpus": 0,
            "assigned_accelerators": _accelerator_assignment(),
        }

    def configure(
        self,
        *,
        event_count: int,
        event_mix: str,
        delay_ms: float,
    ) -> None:
        if event_count < 0 or delay_ms < 0:
            raise ValueError("event count and delay must be non-negative")
        if event_mix not in EVENT_MIXES:
            raise ValueError(f"unsupported event mix {event_mix!r}")
        self.event_count = event_count
        self.event_mix = event_mix
        self.delay_ms = delay_ms
        self.quantum_id = 0

    def drain_frontend_events(self) -> FrontendEventBatch:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        batch = _build_event_batch(
            engine_id=self.engine_id,
            quantum_id=self.quantum_id,
            event_count=self.event_count,
            event_mix=self.event_mix,
        )
        self.quantum_id += 1
        return batch


def _validate_placements(
    descriptors: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
) -> None:
    if len(descriptors) != len(nodes):
        raise RuntimeError("actor placement count mismatch")
    for descriptor, node in zip(descriptors, nodes, strict=True):
        if descriptor["node_id"] != str(node["NodeID"]):
            raise RuntimeError(
                f"actor placement mismatch: descriptor={descriptor}"
            )
        assigned_gpus = descriptor["assigned_accelerators"].get("GPU", [])
        if descriptor["requested_gpus"] != 0 or assigned_gpus:
            raise RuntimeError(
                f"CPU-only profiler actor received GPU resources: {descriptor}"
            )


def _event_counts(
    *,
    scaling_mode: str,
    engines: int,
    max_engines: int,
    batch_size: int,
) -> tuple[int, ...]:
    if scaling_mode == "weak":
        return (batch_size,) * engines
    if scaling_mode != "strong":
        raise ValueError(f"unsupported scaling mode {scaling_mode!r}")
    total = batch_size * max_engines
    base, extra = divmod(total, engines)
    return tuple(base + (engine_id < extra) for engine_id in range(engines))


def _new_router(engines: int) -> RequestRouter:
    return RequestRouter(
        {
            engine_id: _NoopEngineTransport(engine_id)
            for engine_id in range(engines)
        },
        router_policy="round_robin",
    )


def _prepare_router_state(
    router: RequestRouter, batch: FrontendEventBatch
) -> None:
    add_ids = {event.request_id for event in batch.add_results}
    for event in batch.add_results:
        router._owners[event.request_id] = RequestOwner(
            OwnerState.PENDING_ADD, event.engine_id
        )
    for event in batch.finish_events:
        if event.request_id not in add_ids:
            router._owners[event.request_id] = RequestOwner(
                OwnerState.OWNED, event.engine_id
            )


def _process_router_batches(
    router: RequestRouter,
    batches: Sequence[FrontendEventBatch],
) -> dict[str, float | int]:
    """Apply one ready-flight group after untimed precondition setup."""
    if not batches:
        raise ValueError("at least one frontend event batch is required")
    for batch in batches:
        _prepare_router_state(router, batch)
    total_begin_ns = time.perf_counter_ns()

    begin_ns = time.perf_counter_ns()
    router.record_loads(batch.load for batch in batches)
    load_ns = time.perf_counter_ns() - begin_ns

    raw_add_results = tuple(
        event for batch in batches for event in batch.add_results
    )
    begin_ns = time.perf_counter_ns()
    add_results = router.record_add_results(raw_add_results)
    add_ns = time.perf_counter_ns() - begin_ns

    raw_first_schedule = tuple(
        event for batch in batches for event in batch.first_schedule_events
    )
    begin_ns = time.perf_counter_ns()
    first_schedule = router.record_first_schedule_events(
        raw_first_schedule
    )
    first_schedule_ns = time.perf_counter_ns() - begin_ns

    raw_first_token = tuple(
        event for batch in batches for event in batch.first_token_events
    )
    batch_engine_ids = {batch.engine_id for batch in batches}
    begin_ns = time.perf_counter_ns()
    first_token = tuple(raw_first_token)
    if any(event.engine_id not in batch_engine_ids for event in first_token):
        raise RuntimeError("first-token event owner mismatch")
    first_token_ns = time.perf_counter_ns() - begin_ns

    raw_finish = tuple(
        event for batch in batches for event in batch.finish_events
    )
    begin_ns = time.perf_counter_ns()
    finish = router.record_finish_events(raw_finish)
    finish_ns = time.perf_counter_ns() - begin_ns
    total_ns = time.perf_counter_ns() - total_begin_ns

    finish_ids = {event.request_id for event in raw_finish}
    for event in raw_add_results:
        if event.request_id not in finish_ids:
            router._owners.pop(event.request_id, None)
    for event in finish:
        router._terminal.pop(event.request_id, None)
    if router.active_count != 0:
        raise RuntimeError("receipt profiler leaked Router ownership state")

    return {
        "router_load_ms": load_ns / 1_000_000,
        "router_add_ms": add_ns / 1_000_000,
        "router_first_schedule_ms": first_schedule_ns / 1_000_000,
        "frontend_first_token_ms": first_token_ns / 1_000_000,
        "router_finish_ms": finish_ns / 1_000_000,
        "router_total_ms": total_ns / 1_000_000,
        "add_events": len(add_results),
        "first_schedule_events": len(first_schedule),
        "first_token_events": len(first_token),
        "finish_events": len(finish),
    }


def _process_router_batch(
    router: RequestRouter, batch: FrontendEventBatch
) -> dict[str, float | int]:
    """Compatibility helper for focused single-batch unit tests."""
    return _process_router_batches(router, (batch,))


def _arm_event_flight(actor: Any) -> _EventFlight:
    started_ns = time.perf_counter_ns()
    return _EventFlight(actor.drain_frontend_events.remote(), started_ns)


def _run_event_phase(
    *,
    actors: Sequence[Any],
    router: RequestRouter,
    iterations: int,
    payload_bytes_by_engine: Sequence[int],
    straggler_engine: int,
    poll_interval_ms: float,
    timeout_s: float,
) -> dict[str, Any]:
    if len(payload_bytes_by_engine) != len(actors):
        raise ValueError("payload-size count must match the event actors")
    flights = {
        engine_id: _arm_event_flight(actor)
        for engine_id, actor in enumerate(actors)
    }
    completed = [0] * len(actors)
    samples: list[dict[str, Any]] = []
    poll_samples_ms: list[float] = []
    get_samples_ms: list[float] = []
    rearm_samples_ms: list[float] = []
    router_call_samples: list[dict[str, float | int]] = []
    max_pending_per_engine = [1] * len(actors)
    fast_progress_before_first_straggler: int | None = None
    started_ns = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_s

    while any(count < iterations for count in completed):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"frontend event phase timed out: {completed}/{iterations}"
            )
        refs = [flight.ref for flight in flights.values()]
        poll_begin_ns = time.perf_counter_ns()
        ready_refs, _ = ray.wait(
            refs,
            num_returns=len(refs),
            timeout=0,
        )
        poll_samples_ms.append(
            (time.perf_counter_ns() - poll_begin_ns) / 1_000_000
        )
        if not ready_refs:
            if poll_interval_ms:
                time.sleep(poll_interval_ms / 1000.0)
            continue

        ref_to_engine = {
            flight.ref: engine_id for engine_id, flight in flights.items()
        }
        ready_engine_ids = sorted(ref_to_engine[ref] for ref in ready_refs)
        ready_flights = {
            engine_id: flights[engine_id]
            for engine_id in ready_engine_ids
        }
        get_begin_ns = time.perf_counter_ns()
        batches = ray.get(
            [ready_flights[engine_id].ref for engine_id in ready_engine_ids]
        )
        get_finished_ns = time.perf_counter_ns()
        get_elapsed_ms = (get_finished_ns - get_begin_ns) / 1_000_000
        get_samples_ms.append(get_elapsed_ms)

        rearm_begin_ns = time.perf_counter_ns()
        for engine_id in ready_engine_ids:
            completed[engine_id] += 1
            if completed[engine_id] < iterations:
                flights[engine_id] = _arm_event_flight(actors[engine_id])
            else:
                flights.pop(engine_id)
        rearm_elapsed_ms = (
            time.perf_counter_ns() - rearm_begin_ns
        ) / 1_000_000
        rearm_samples_ms.append(rearm_elapsed_ms)

        for engine_id, batch in zip(
            ready_engine_ids, batches, strict=True
        ):
            if batch.engine_id != engine_id:
                raise RuntimeError(
                    "frontend event batch returned from the wrong engine"
                )
            expected_quantum = completed[engine_id] - 1
            if batch.load.quantum_id != expected_quantum:
                raise RuntimeError(
                    "frontend event quantum sequence mismatch: "
                    f"engine={engine_id}, expected={expected_quantum}, "
                    f"got={batch.load.quantum_id}"
                )

        router_metrics = _process_router_batches(router, batches)
        ready_batch_count = len(batches)
        router_call_samples.append(
            {"ready_batches": ready_batch_count, **router_metrics}
        )
        for engine_id, batch in zip(
            ready_engine_ids, batches, strict=True
        ):
            event_items = sum(
                (
                    len(batch.add_results),
                    len(batch.first_schedule_events),
                    len(batch.first_token_events),
                    len(batch.finish_events),
                )
            )
            samples.append(
                {
                    "engine_id": engine_id,
                    "quantum_id": batch.load.quantum_id,
                    "flight_age_ms": (
                        get_finished_ns
                        - ready_flights[engine_id].started_ns
                    )
                    / 1_000_000,
                    "ray_get_ms_per_ready_batch": (
                        get_elapsed_ms / len(ready_engine_ids)
                    ),
                    "rearm_ms_per_ready_batch": (
                        rearm_elapsed_ms / len(ready_engine_ids)
                    ),
                    "payload_bytes": payload_bytes_by_engine[engine_id],
                    "event_items": event_items,
                    "router_load_ms": (
                        float(router_metrics["router_load_ms"])
                        / ready_batch_count
                    ),
                    "router_add_ms": (
                        float(router_metrics["router_add_ms"])
                        / ready_batch_count
                    ),
                    "router_first_schedule_ms": (
                        float(router_metrics["router_first_schedule_ms"])
                        / ready_batch_count
                    ),
                    "frontend_first_token_ms": (
                        float(router_metrics["frontend_first_token_ms"])
                        / ready_batch_count
                    ),
                    "router_finish_ms": (
                        float(router_metrics["router_finish_ms"])
                        / ready_batch_count
                    ),
                    "router_total_ms": (
                        float(router_metrics["router_total_ms"])
                        / ready_batch_count
                    ),
                    "add_events": len(batch.add_results),
                    "first_schedule_events": len(
                        batch.first_schedule_events
                    ),
                    "first_token_events": len(batch.first_token_events),
                    "finish_events": len(batch.finish_events),
                }
            )
            if (
                engine_id == straggler_engine
                and completed[engine_id] == 1
                and fast_progress_before_first_straggler is None
            ):
                fast_progress_before_first_straggler = sum(
                    count
                    for candidate_engine, count in enumerate(completed)
                    if candidate_engine != straggler_engine
                )

    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    return {
        "samples": samples,
        "elapsed_ms": elapsed_ms,
        "poll_samples_ms": poll_samples_ms,
        "get_samples_ms": get_samples_ms,
        "rearm_samples_ms": rearm_samples_ms,
        "router_call_samples": router_call_samples,
        "max_pending_per_engine": max_pending_per_engine,
        "fast_progress_before_first_straggler": (
            fast_progress_before_first_straggler
        ),
    }


def _control_init_method(
    nodes: Sequence[Mapping[str, Any]],
    explicit_address: str | None,
) -> str:
    if explicit_address is not None:
        raw = explicit_address
        if "://" in raw:
            _scheme, raw = raw.split("://", 1)
        try:
            host, port_text = raw.rsplit(":", 1)
            port = int(port_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid --control-address {explicit_address!r}"
            ) from exc
        if not host or not 1 <= port <= 65_535:
            raise ValueError(
                f"invalid --control-address {explicit_address!r}"
            )
        return f"tcp://{host}:{port}"

    control_host = str(nodes[0].get("NodeManagerAddress", ""))
    driver_host = ray.util.get_node_ip_address()
    if not control_host or control_host != driver_host:
        raise RuntimeError(
            "automatic Gloo control-port selection requires the driver to run "
            "on the first selected Ray node; pass --control-address explicitly"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((control_host, 0))
        port = int(listener.getsockname()[1])
    return f"tcp://{control_host}:{port}"


def _create_consensus_actors(
    *,
    nodes: Sequence[Mapping[str, Any]],
    gloo_interface: str | None,
    control_address: str | None,
    timeout_s: float,
) -> tuple[list[Any], tuple[dict[str, Any], ...], str | None]:
    actors = []
    runtime_env = (
        {"env_vars": {"GLOO_SOCKET_IFNAME": gloo_interface}}
        if gloo_interface
        else None
    )
    try:
        for rank, node in enumerate(nodes):
            options: dict[str, Any] = {
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    str(node["NodeID"]), soft=False
                )
            }
            if runtime_env is not None:
                options["runtime_env"] = runtime_env
            actors.append(
                _ConsensusActor.options(**options).remote(rank, len(nodes))
            )
        descriptors = tuple(
            ray.get(
                [actor.descriptor.remote() for actor in actors],
                timeout=timeout_s,
            )
        )
        _validate_placements(descriptors, nodes)
        init_method = (
            _control_init_method(nodes, control_address)
            if len(nodes) > 1
            else None
        )
        ray.get(
            [
                actor.initialize.remote(init_method, timeout_s)
                for actor in actors
            ],
            timeout=timeout_s,
        )
        return actors, descriptors, init_method
    except BaseException:
        _kill_actors(actors, timeout_s, graceful_method="close")
        raise


def _create_event_actors(
    *,
    nodes: Sequence[Mapping[str, Any]],
    timeout_s: float,
) -> tuple[list[Any], tuple[dict[str, Any], ...]]:
    actors = [
        _FrontendEventActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                str(node["NodeID"]), soft=False
            )
        ).remote(engine_id)
        for engine_id, node in enumerate(nodes)
    ]
    try:
        descriptors = tuple(
            ray.get(
                [actor.descriptor.remote() for actor in actors],
                timeout=timeout_s,
            )
        )
        _validate_placements(descriptors, nodes)
        return actors, descriptors
    except BaseException:
        _kill_actors(actors, timeout_s)
        raise


def _kill_actors(
    actors: Iterable[Any],
    timeout_s: float,
    *,
    graceful_method: str | None = None,
) -> None:
    actor_list = tuple(actors)
    if graceful_method is not None:
        refs = []
        for actor in actor_list:
            try:
                refs.append(getattr(actor, graceful_method).remote())
            except Exception:
                pass
        if refs:
            try:
                ray.get(refs, timeout=timeout_s)
            except Exception:
                pass
    for actor in actor_list:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass


def _profile_consensus_case(
    *,
    actors: Sequence[Any],
    descriptors: Sequence[Mapping[str, Any]],
    nodes: int,
    overlap_work_ms: float,
    straggler_rank: int,
    straggler_delay_ms: float,
    warmup_iterations: int,
    iterations: int,
    timeout_s: float,
    init_method: str | None,
) -> dict[str, Any]:
    kwargs = {
        "overlap_work_ms": overlap_work_ms,
        "straggler_rank": straggler_rank,
        "straggler_delay_ms": straggler_delay_ms,
    }
    if warmup_iterations:
        ray.get(
            [
                actor.profile.remote(warmup_iterations, **kwargs)
                for actor in actors
            ],
            timeout=timeout_s,
        )
    begin_ns = time.perf_counter_ns()
    per_rank_samples = ray.get(
        [actor.profile.remote(iterations, **kwargs) for actor in actors],
        timeout=timeout_s,
    )
    elapsed_ms = (time.perf_counter_ns() - begin_ns) / 1_000_000
    if len(per_rank_samples) != nodes or any(
        len(samples) != iterations for samples in per_rank_samples
    ):
        raise RuntimeError("consensus profiler returned the wrong sample count")

    arrival_skew_ms: list[float] = []
    submit_critical_ms: list[float] = []
    overlap_critical_ms: list[float] = []
    exposed_wait_critical_ms: list[float] = []
    rendezvous_critical_ms: list[float] = []
    late_participant_collective_ms: list[float] = []
    for quantum_id in range(iterations):
        quantum_samples = [
            samples[quantum_id] for samples in per_rank_samples
        ]
        if any(
            sample["quantum_id"] != quantum_id
            or sample["rank"] != rank
            for rank, sample in enumerate(quantum_samples)
        ):
            raise RuntimeError("consensus sample identity mismatch")
        arrivals = [int(sample["arrival_unix_ns"]) for sample in quantum_samples]
        late_rank = max(range(nodes), key=arrivals.__getitem__)
        arrival_skew_ms.append((max(arrivals) - min(arrivals)) / 1_000_000)
        submit_critical_ms.append(
            max(float(sample["submit_ms"]) for sample in quantum_samples)
        )
        overlap_critical_ms.append(
            max(
                float(sample["overlap_window_ms"])
                for sample in quantum_samples
            )
        )
        exposed_wait_critical_ms.append(
            max(
                float(sample["exposed_wait_ms"])
                for sample in quantum_samples
            )
        )
        rendezvous_critical_ms.append(
            max(
                float(sample["rendezvous_ms"])
                for sample in quantum_samples
            )
        )
        late_participant_collective_ms.append(
            float(quantum_samples[late_rank]["rendezvous_ms"])
        )

    record: dict[str, Any] = {
        "component": "dp_consensus",
        "nodes": nodes,
        "engines": nodes,
        "logical_workers": nodes * ATTENTION_SP,
        "attention_sp": ATTENTION_SP,
        "message_int64_count": CONSENSUS_FIELDS,
        "logical_payload_bytes_per_engine": CONSENSUS_FIELDS * 8,
        "logical_payload_bytes_all_engines": CONSENSUS_FIELDS * 8 * nodes,
        "overlap_work_ms": overlap_work_ms,
        "straggler_rank": straggler_rank if nodes > 1 else None,
        "straggler_delay_ms": straggler_delay_ms if nodes > 1 else 0.0,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "driver_elapsed_ms": elapsed_ms,
        "quantums_per_second": iterations * 1000.0 / elapsed_ms,
        "gloo_init_method": init_method,
        "placement": tuple(descriptors),
        "correctness": {
            "all_ranks_reported": len(per_rank_samples) == nodes,
            "all_quantums_reported": all(
                len(samples) == iterations for samples in per_rank_samples
            ),
            "zero_gpu_assignment": all(
                not descriptor["assigned_accelerators"].get("GPU", [])
                for descriptor in descriptors
            ),
        },
        "per_engine": {
            str(rank): {
                "rendezvous_ms": asdict(
                    _stats(sample["rendezvous_ms"] for sample in samples)
                ),
                "exposed_wait_ms": asdict(
                    _stats(sample["exposed_wait_ms"] for sample in samples)
                ),
            }
            for rank, samples in enumerate(per_rank_samples)
        },
    }
    _add_stats(record, "leader_arrival_skew_ms", arrival_skew_ms)
    _add_stats(record, "collective_submit_critical_ms", submit_critical_ms)
    _add_stats(record, "consensus_overlap_critical_ms", overlap_critical_ms)
    _add_stats(
        record,
        "consensus_exposed_wait_critical_ms",
        exposed_wait_critical_ms,
    )
    _add_stats(
        record,
        "leader_rendezvous_critical_ms",
        rendezvous_critical_ms,
    )
    _add_stats(
        record,
        "late_participant_collective_ms",
        late_participant_collective_ms,
    )
    record["consensus_exposed_wait_mean_ms_per_decode_step"] = (
        record["consensus_exposed_wait_critical_ms_mean"] / 16
    )
    record["consensus_exposed_wait_p99_ms_per_decode_step"] = (
        record["consensus_exposed_wait_critical_ms_p99"] / 16
    )
    return record


def _event_quantum_stats(
    samples: Sequence[Mapping[str, Any]],
    *,
    engines: int,
    iterations: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_quantum: dict[int, list[Mapping[str, Any]]] = {
        quantum_id: [] for quantum_id in range(iterations)
    }
    for sample in samples:
        by_quantum[int(sample["quantum_id"])].append(sample)
    if any(len(items) != engines for items in by_quantum.values()):
        raise RuntimeError("frontend event phase missed one or more engine batches")
    critical_flight_ms = []
    router_cpu_ms = []
    payload_bytes = []
    event_items = []
    for quantum_id in range(iterations):
        quantum_samples = by_quantum[quantum_id]
        critical_flight_ms.append(
            max(float(sample["flight_age_ms"]) for sample in quantum_samples)
        )
        router_cpu_ms.append(
            sum(float(sample["router_total_ms"]) for sample in quantum_samples)
        )
        payload_bytes.append(
            sum(float(sample["payload_bytes"]) for sample in quantum_samples)
        )
        event_items.append(
            sum(float(sample["event_items"]) for sample in quantum_samples)
        )
    return critical_flight_ms, router_cpu_ms, payload_bytes, event_items


def _profile_event_case(
    *,
    actors: Sequence[Any],
    descriptors: Sequence[Mapping[str, Any]],
    nodes: int,
    max_nodes: int,
    scaling_mode: str,
    batch_size: int,
    event_mix: str,
    straggler_engine: int,
    straggler_delay_ms: float,
    warmup_iterations: int,
    iterations: int,
    poll_interval_ms: float,
    timeout_s: float,
) -> dict[str, Any]:
    event_counts = (
        (0,) * nodes
        if event_mix == "load"
        else _event_counts(
            scaling_mode=scaling_mode,
            engines=nodes,
            max_engines=max_nodes,
            batch_size=batch_size,
        )
    )
    delays = tuple(
        straggler_delay_ms if engine_id == straggler_engine else 0.0
        for engine_id in range(nodes)
    )
    payload_bytes_by_engine = tuple(
        len(
            pickle.dumps(
                _build_event_batch(
                    engine_id=engine_id,
                    quantum_id=0,
                    event_count=event_count,
                    event_mix=event_mix,
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        for engine_id, event_count in enumerate(event_counts)
    )

    def configure() -> None:
        ray.get(
            [
                actor.configure.remote(
                    event_count=event_count,
                    event_mix=event_mix,
                    delay_ms=delay_ms,
                )
                for actor, event_count, delay_ms in zip(
                    actors, event_counts, delays, strict=True
                )
            ],
            timeout=timeout_s,
        )

    if warmup_iterations:
        configure()
        _run_event_phase(
            actors=actors,
            router=_new_router(nodes),
            iterations=warmup_iterations,
            payload_bytes_by_engine=payload_bytes_by_engine,
            straggler_engine=straggler_engine,
            poll_interval_ms=poll_interval_ms,
            timeout_s=timeout_s,
        )
    configure()
    result = _run_event_phase(
        actors=actors,
        router=_new_router(nodes),
        iterations=iterations,
        payload_bytes_by_engine=payload_bytes_by_engine,
        straggler_engine=straggler_engine,
        poll_interval_ms=poll_interval_ms,
        timeout_s=timeout_s,
    )
    samples = result["samples"]
    expected_samples = nodes * iterations
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} event samples, got {len(samples)}"
        )
    (
        critical_flight_ms,
        router_cpu_ms,
        payload_bytes,
        event_items,
    ) = _event_quantum_stats(samples, engines=nodes, iterations=iterations)
    elapsed_ms = float(result["elapsed_ms"])
    total_items = sum(event_items)
    total_payload_bytes = sum(payload_bytes)
    record: dict[str, Any] = {
        "component": "local_engine_to_router_events",
        "nodes": nodes,
        "engines": nodes,
        "logical_workers": nodes * ATTENTION_SP,
        "attention_sp": ATTENTION_SP,
        "scaling_mode": scaling_mode,
        "batch_size": batch_size,
        "event_mix": event_mix,
        "event_counts_per_engine": event_counts,
        "event_items_per_quantum": event_items[0],
        "payload_size_method": "untimed_python_pickle_quantum_0",
        "straggler_engine": straggler_engine,
        "straggler_delay_ms": straggler_delay_ms,
        "poll_interval_ms": poll_interval_ms,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "driver_elapsed_ms": elapsed_ms,
        "response_batches_per_second": (
            expected_samples * 1000.0 / elapsed_ms
        ),
        "event_items_per_second": total_items * 1000.0 / elapsed_ms,
        "payload_mib_per_second": (
            total_payload_bytes * 1000.0 / elapsed_ms / (1024 * 1024)
        ),
        "fast_progress_before_first_straggler": result[
            "fast_progress_before_first_straggler"
        ],
        "placement": tuple(descriptors),
        "correctness": {
            "all_engine_batches_received": len(samples) == expected_samples,
            "one_flight_per_engine": all(
                maximum == 1 for maximum in result["max_pending_per_engine"]
            ),
            "zero_gpu_assignment": all(
                not descriptor["assigned_accelerators"].get("GPU", [])
                for descriptor in descriptors
            ),
            "router_state_released": True,
        },
    }
    _add_stats(
        record,
        "flight_age_ms",
        (sample["flight_age_ms"] for sample in samples),
    )
    _add_stats(record, "critical_flight_age_ms", critical_flight_ms)
    _add_stats(record, "router_cpu_ms_per_global_quantum", router_cpu_ms)
    _add_stats(record, "payload_bytes_per_global_quantum", payload_bytes)
    _add_stats(record, "event_items_per_global_quantum", event_items)
    for phase in (
        "router_load_ms",
        "router_add_ms",
        "router_first_schedule_ms",
        "frontend_first_token_ms",
        "router_finish_ms",
        "router_total_ms",
        "ray_get_ms_per_ready_batch",
        "rearm_ms_per_ready_batch",
    ):
        _add_stats(record, phase, (sample[phase] for sample in samples))
    _add_stats(record, "poll_call_ms", result["poll_samples_ms"])
    _add_stats(record, "ray_get_call_ms", result["get_samples_ms"])
    _add_stats(record, "rearm_call_ms", result["rearm_samples_ms"])
    router_call_samples = result["router_call_samples"]
    _add_stats(
        record,
        "ready_batches_per_router_call",
        (sample["ready_batches"] for sample in router_call_samples),
    )
    for phase in (
        "router_load_ms",
        "router_add_ms",
        "router_first_schedule_ms",
        "frontend_first_token_ms",
        "router_finish_ms",
        "router_total_ms",
    ):
        _add_stats(
            record,
            f"{phase}_per_poll_batch",
            (sample[phase] for sample in router_call_samples),
        )
    return record


def _jsonable_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_results(
    output_dir: Path,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    if not records:
        raise ValueError("at least one profiler record is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "decentralized_control_plane.json"
    csv_path = output_dir / "decentralized_control_plane.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": dict(metadata), "records": list(records)},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: _jsonable_csv_value(record.get(key))
                    for key in fieldnames
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--node-counts",
        type=_parse_positive_ints,
        default=_parse_positive_ints("1,2,4"),
    )
    parser.add_argument(
        "--node-ip",
        action="append",
        default=[],
        help="Target Ray NodeManagerAddress; repeat in desired order.",
    )
    parser.add_argument(
        "--components",
        type=lambda value: _parse_choices(
            value, choices=COMPONENTS, label="component"
        ),
        default=COMPONENTS,
    )
    parser.add_argument(
        "--scaling-modes",
        type=lambda value: _parse_choices(
            value, choices=SCALING_MODES, label="scaling mode"
        ),
        default=SCALING_MODES,
    )
    parser.add_argument(
        "--event-mixes",
        type=lambda value: _parse_choices(
            value, choices=EVENT_MIXES, label="event mix"
        ),
        default=("load", "mixed"),
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128"),
    )
    parser.add_argument(
        "--overlap-work-ms",
        type=_parse_nonnegative_floats,
        default=_parse_nonnegative_floats("0,0.25,1"),
    )
    parser.add_argument(
        "--consensus-straggler-delays-ms",
        type=_parse_nonnegative_floats,
        default=_parse_nonnegative_floats("0"),
    )
    parser.add_argument(
        "--event-straggler-delays-ms",
        type=_parse_nonnegative_floats,
        default=_parse_nonnegative_floats("0"),
    )
    parser.add_argument("--straggler-rank", type=int, default=0)
    parser.add_argument("--straggler-engine", type=int, default=0)
    parser.add_argument("--control-address")
    parser.add_argument(
        "--gloo-interface",
        default=os.getenv("GLOO_SOCKET_IFNAME"),
    )
    parser.add_argument("--poll-interval-ms", type=float, default=0.05)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--gloo-timeout-s", type=float, default=60.0)
    parser.add_argument("--case-timeout-s", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if any(node_count not in {1, 2, 4} for node_count in args.node_counts):
        parser.error("--node-counts must contain only 1,2,4")
    if args.node_ip and len(args.node_ip) < max(args.node_counts):
        parser.error("--node-ip does not provide enough nodes")
    if args.warmup_iterations < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")
    if args.poll_interval_ms < 0:
        parser.error("--poll-interval-ms must be non-negative")
    if args.gloo_timeout_s <= 0 or args.case_timeout_s <= 0:
        parser.error("timeouts must be positive")
    if args.straggler_rank < 0 or args.straggler_engine < 0:
        parser.error("straggler indices must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    removed_proxies = _clear_proxy_environment()
    if ray.is_initialized():
        raise RuntimeError("profiler requires a fresh Ray driver")

    ray_temp_dir: Path | None = None
    ray_kwargs: dict[str, Any] = {
        "address": args.ray_address,
        "_skip_env_hook": True,
        "logging_level": "ERROR",
        "log_to_driver": False,
    }
    if args.ray_address == "local":
        ray_temp_dir = Path(
            tempfile.mkdtemp(prefix="nd-decentral-cp-", dir="/tmp")
        )
        ray_kwargs.update(
            {
                "include_dashboard": False,
                "num_cpus": max(args.node_counts) + 1,
                "_temp_dir": str(ray_temp_dir),
            }
        )
    ray_context = ray.init(**ray_kwargs)
    records: list[dict[str, Any]] = []
    selected_nodes: tuple[dict[str, Any], ...] = ()
    try:
        selected_nodes = _select_nodes(
            ray.nodes(),
            count=max(args.node_counts),
            requested_ips=args.node_ip,
        )
        case_index = 0
        for node_count in args.node_counts:
            nodes = selected_nodes[:node_count]
            if "consensus" in args.components:
                actors: list[Any] = []
                try:
                    actors, descriptors, init_method = _create_consensus_actors(
                        nodes=nodes,
                        gloo_interface=args.gloo_interface,
                        control_address=args.control_address,
                        timeout_s=args.gloo_timeout_s,
                    )
                    delays = (
                        (0.0,)
                        if node_count == 1
                        else args.consensus_straggler_delays_ms
                    )
                    if node_count > 1 and args.straggler_rank >= node_count:
                        parser.error(
                            "--straggler-rank is outside one consensus case"
                        )
                    for overlap_work_ms in args.overlap_work_ms:
                        for delay_ms in delays:
                            case_index += 1
                            print(
                                f"[{case_index}] component=consensus "
                                f"nodes={node_count} overlap={overlap_work_ms}ms "
                                f"straggler={delay_ms}ms",
                                flush=True,
                            )
                            record = _profile_consensus_case(
                                actors=actors,
                                descriptors=descriptors,
                                nodes=node_count,
                                overlap_work_ms=overlap_work_ms,
                                straggler_rank=args.straggler_rank,
                                straggler_delay_ms=delay_ms,
                                warmup_iterations=args.warmup_iterations,
                                iterations=args.iterations,
                                timeout_s=args.case_timeout_s,
                                init_method=init_method,
                            )
                            records.append(record)
                            print(
                                "  exposed mean/p99="
                                f"{record['consensus_exposed_wait_critical_ms_mean']:.3f}/"
                                f"{record['consensus_exposed_wait_critical_ms_p99']:.3f} ms",
                                flush=True,
                            )
                finally:
                    _kill_actors(
                        actors,
                        args.gloo_timeout_s,
                        graceful_method="close",
                    )

            if "events" in args.components:
                event_actors: list[Any] = []
                try:
                    event_actors, descriptors = _create_event_actors(
                        nodes=nodes,
                        timeout_s=args.case_timeout_s,
                    )
                    if args.straggler_engine >= node_count:
                        parser.error(
                            "--straggler-engine is outside one event case"
                        )
                    for event_mix in args.event_mixes:
                        if event_mix == "load":
                            event_cases = (("steady", 0),)
                        else:
                            event_cases = tuple(
                                (scaling_mode, batch_size)
                                for scaling_mode in args.scaling_modes
                                for batch_size in args.batch_sizes
                            )
                        for scaling_mode, batch_size in event_cases:
                            for delay_ms in args.event_straggler_delays_ms:
                                case_index += 1
                                print(
                                    f"[{case_index}] component=events "
                                    f"nodes={node_count} mix={event_mix} "
                                    f"scaling={scaling_mode} batch={batch_size} "
                                    f"straggler={delay_ms}ms",
                                    flush=True,
                                )
                                record = _profile_event_case(
                                    actors=event_actors,
                                    descriptors=descriptors,
                                    nodes=node_count,
                                    max_nodes=max(args.node_counts),
                                    scaling_mode=scaling_mode,
                                    batch_size=batch_size,
                                    event_mix=event_mix,
                                    straggler_engine=args.straggler_engine,
                                    straggler_delay_ms=delay_ms,
                                    warmup_iterations=args.warmup_iterations,
                                    iterations=args.iterations,
                                    poll_interval_ms=args.poll_interval_ms,
                                    timeout_s=args.case_timeout_s,
                                )
                                records.append(record)
                                print(
                                    "  flight mean/p99="
                                    f"{record['flight_age_ms_mean']:.3f}/"
                                    f"{record['flight_age_ms_p99']:.3f} ms | "
                                    "router mean="
                                    f"{record['router_total_ms_mean']:.3f} ms",
                                    flush=True,
                                )
                finally:
                    _kill_actors(event_actors, args.case_timeout_s)
    finally:
        ray.shutdown()

    metadata = {
        "benchmark": "nanodeploy-decentralized-control-plane",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "driver_hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ray_version": ray.__version__,
        "torch_version": torch.__version__,
        "ray_address": ray_context.address_info.get(
            "gcs_address", args.ray_address
        ),
        "ray_temp_dir": str(ray_temp_dir) if ray_temp_dir else None,
        "selected_nodes": _node_metadata(selected_nodes),
        "removed_proxy_names": sorted(removed_proxies),
        "timed_scope": (
            "hierarchical leader Gloo consensus and persistent "
            "LocalEngine-to-Router FrontendEventBatch Ray flights, including "
            "RequestRouter receipt processing"
        ),
        "excluded_scope": (
            "centralized scheduler benchmark, Router-to-LocalEngine request "
            "ingress, ModelRunner, CUDA, GPU kernels, DLSLime/RDMA, and worker "
            "collectives"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_results(args.output_dir, metadata, records)
    print(
        f"Wrote {args.output_dir / 'decentralized_control_plane.json'}",
        flush=True,
    )
    print(
        f"Wrote {args.output_dir / 'decentralized_control_plane.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

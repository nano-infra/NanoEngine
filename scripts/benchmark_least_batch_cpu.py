#!/usr/bin/env python3
"""CPU-only benchmark for least-batch staged admission.

This benchmark deliberately stops at the scheduler boundary.  It measures:

1. The real RequestRouter dispatch/receipt pipeline with a deterministic
   in-process transport.  A configurable receipt delay models the old
   commit-gated stop-and-wait path; it is not a measurement of old code.
2. The real localhost ZMQ codec, Sequence deserialization, and
   LocalEngineCore staged-ingress method.  The scheduler is never drained,
   so a positive result proves that receipts do not wait for commit.

No CUDA context, Ray cluster, model weights, or worker is started.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import zmq

from nanodeploy._cpp import Sequence, serialize_sequence_payload
from nanodeploy.engine.frontend_transport import (
    ZmqFrontendClient,
    ZmqFrontendServer,
)
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AdmissionReservation,
    AbortResult,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    RankLoad,
)
from nanodeploy.engine.local_engine import LocalEngineCore
from nanodeploy.router.admission_planner import AdmissionPlannerConfig
from nanodeploy.router.request_router import RequestRouter


_POLL_SLEEP_S = 0.00005
_TRIAL_TIMEOUT_S = 30.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rounded(value: float) -> float:
    return round(value, 6)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _validate_parameters(
    *,
    requests: int,
    engines: int,
    batch_size: int,
    prompt_tokens: int,
) -> None:
    values = {
        "requests": requests,
        "engines": engines,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(
            "benchmark values must be positive: " + ", ".join(invalid)
        )


@dataclass(frozen=True, slots=True)
class _DelayedAdmission:
    ready_at: float
    acks: tuple[IngressAck, ...]


class _DelayedReceiptTransport:
    """Minimal transport that preserves the real Router flight semantics."""

    def __init__(self, engine_id: int, receipt_delay_ms: float) -> None:
        self.engine_id = engine_id
        self._receipt_delay_s = receipt_delay_ms / 1000.0
        self._ingress_version = 0
        self.command_request_ids: list[int] = []
        self.batch_sizes: list[int] = []
        self.poll_calls = 0

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> _DelayedAdmission:
        if len(commands) != len(reservations):
            raise AssertionError("command/reservation count mismatch")
        acks = []
        for command, reservation in zip(commands, reservations, strict=True):
            if (
                reservation.request_id != command.request_id
                or reservation.engine_id != self.engine_id
            ):
                raise AssertionError("Router emitted an invalid reservation")
            self._ingress_version += 1
            self.command_request_ids.append(command.request_id)
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=True,
                    ingress_version=self._ingress_version,
                )
            )
        self.batch_sizes.append(len(commands))
        return _DelayedAdmission(
            ready_at=time.perf_counter() + self._receipt_delay_s,
            acks=tuple(acks),
        )

    def poll_admission_batch(
        self, handle: _DelayedAdmission
    ) -> tuple[bool, tuple[IngressAck, ...] | None]:
        self.poll_calls += 1
        if time.perf_counter() < handle.ready_at:
            return False, None
        return True, handle.acks

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


def _router_load(engine_id: int, requests: int) -> LoadSnapshot:
    free_blocks = max(1_000_000, requests * 64)
    return LoadSnapshot(
        engine_id=engine_id,
        ready=True,
        waiting=0,
        running=0,
        free_blocks_min=free_blocks,
        wave_id=0,
        quantum_id=0,
        rank_loads=(
            RankLoad(
                global_rank=engine_id,
                sp_idx=0,
                tp_idx=0,
                master_batch_size=0,
                active_master_requests=0,
                free_blocks=free_blocks,
                total_blocks=free_blocks,
                master_assignments=0,
                mastered_decode_tokens=0,
                control_dummy_blocks=1,
            ),
        ),
    )


def run_router_pipeline_trial(
    *,
    requests: int,
    engines: int,
    batch_size: int,
    prompt_tokens: int,
    receipt_delay_ms: float,
) -> dict[str, Any]:
    """Run one real RequestRouter pipeline trial on CPU."""
    _validate_parameters(
        requests=requests,
        engines=engines,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
    )
    if receipt_delay_ms < 0:
        raise ValueError("receipt_delay_ms must be non-negative")

    transports = {
        engine_id: _DelayedReceiptTransport(engine_id, receipt_delay_ms)
        for engine_id in range(engines)
    }
    planner_config = AdmissionPlannerConfig(
        attention_sp=1,
        kvcache_block_size=16,
        max_num_seqs=requests + 1,
        max_num_batched_tokens=requests * prompt_tokens + 1,
        max_num_recv_seqs=requests + 1,
        reserved_blocks_per_req=0.0,
        segment_size=64,
        queue_capacity=requests + 1,
    )
    router = RequestRouter(
        transports,
        router_policy="least_batch",
        admission_batch_size=batch_size,
        admission_planner_config=planner_config,
    )
    router.record_loads(
        tuple(_router_load(engine_id, requests) for engine_id in transports)
    )

    observed_acks: list[IngressAck] = []
    started_at = time.perf_counter()
    for request_id in range(requests):
        router.submit_async(
            request_id=request_id,
            prompt_len=prompt_tokens,
            num_tokens=prompt_tokens,
            max_tokens=16,
            temperature=0.0,
            ignore_eos=True,
            sequence_payload=b"cpu-router-payload",
        )

    deadline = started_at + _TRIAL_TIMEOUT_S
    poll_iterations = 0
    while len(observed_acks) < requests:
        poll_iterations += 1
        ready = router.poll_ingress_acks()
        observed_acks.extend(ready)
        if not ready:
            time.sleep(_POLL_SLEEP_S)
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                "CPU Router benchmark timed out: "
                f"received={len(observed_acks)}/{requests}"
            )
    elapsed_s = time.perf_counter() - started_at

    metrics = router.admission_metrics()
    per_engine = tuple(metrics["per_engine"].values())
    all_sent_ids = [
        request_id
        for transport in transports.values()
        for request_id in transport.command_request_ids
    ]
    id_counts = Counter(all_sent_ids)
    batch_messages = sum(item["batch_messages"] for item in per_engine)
    positive_receipts = sum(
        item["positive_receipts"] for item in per_engine
    )
    commits = sum(item["commits"] for item in per_engine)
    pending_add_owners = sum(
        router.owner(request_id) is not None
        and router.owner(request_id).state == OwnerState.PENDING_ADD
        for request_id in range(requests)
    )
    correctness = {
        "all_positive_receipts": (
            len(observed_acks) == requests
            and all(ack.enqueued for ack in observed_acks)
        ),
        "all_pending_add_before_commit": pending_add_owners == requests,
        "zero_scheduler_commits": commits == 0,
        "payload_handed_to_transport_once": (
            len(id_counts) == requests
            and all(count == 1 for count in id_counts.values())
        ),
        "router_has_no_pending_ingress": router.pending_ingress_count == 0,
    }
    if not all(correctness.values()):
        raise AssertionError(f"Router benchmark invariant failed: {correctness}")

    return {
        "elapsed_ms": _rounded(elapsed_s * 1000),
        "requests_per_second": _rounded(requests / elapsed_s),
        "requests": requests,
        "engines": engines,
        "batch_size_limit": batch_size,
        "receipt_delay_ms": receipt_delay_ms,
        "batch_messages": batch_messages,
        "requests_per_message": _rounded(requests / batch_messages),
        "positive_receipts": positive_receipts,
        "scheduler_commits": commits,
        "pending_add": router.pending_add_count,
        "poll_iterations": poll_iterations,
        "transport_poll_calls": sum(
            transport.poll_calls for transport in transports.values()
        ),
        "per_engine_batch_sizes": {
            str(engine_id): transport.batch_sizes
            for engine_id, transport in transports.items()
        },
        "correctness": correctness,
    }


class _CommitRecorder:
    def __init__(self) -> None:
        self.commit_calls = 0

    def commit_planned_batch(self, commands, reservations, sequences):
        del commands, reservations, sequences
        self.commit_calls += 1
        raise AssertionError("CPU receipt benchmark must not drain scheduler")


def _staging_local_engine(queue_capacity: int, engine_id: int = 0):
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    engine = object.__new__(actor_class)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        hierarchical_queue_capacity=queue_capacity,
    )
    engine.engine_id = engine_id
    engine.scheduler = _CommitRecorder()
    engine._failure = None
    engine._coordinator = None
    engine._ingress_adds = queue.Queue()
    engine._ingress_head = None
    engine._ingress_lock = threading.Lock()
    engine._reserved_request_ids = set()
    engine._ingress_pending_ids = set()
    engine._admission_pending_ids = set()
    engine._cancelled_ingress_ids = set()
    engine._reserved_slots = 0
    engine._ingress_version = 0
    engine._admission_version = 0
    engine._staged_ingress_depth_max = 0
    engine._state_cv = threading.Condition()
    engine._wave_running = False
    engine._wave_id = 0
    engine._quantum_id = 0
    return engine


def _zmq_commands(
    requests: int, prompt_tokens: int
) -> tuple[tuple[AddCommand, AdmissionReservation], ...]:
    items = []
    for request_id in range(requests):
        token_ids = [
            (request_id + offset) % 32_000
            for offset in range(prompt_tokens)
        ]
        sequence = Sequence(token_ids, 0.0, 16, True)
        sequence.seq_id = request_id
        payload = serialize_sequence_payload(sequence)
        command = AddCommand(
            request_id=request_id,
            prompt_len=prompt_tokens,
            num_tokens=prompt_tokens,
            max_tokens=16,
            temperature=0.0,
            ignore_eos=True,
            wave_id=0,
            sequence_payload=payload,
        )
        reservation = AdmissionReservation(
            request_id=request_id,
            engine_id=0,
            master_sp_idx=0,
            dispatched_tokens=(prompt_tokens,),
        )
        items.append((command, reservation))
    return tuple(items)


def run_zmq_staged_receipt_trial(
    *,
    requests: int,
    batch_size: int,
    prompt_tokens: int,
) -> dict[str, Any]:
    """Measure real localhost ZMQ decode and LocalEngine staging on CPU."""
    _validate_parameters(
        requests=requests,
        engines=1,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
    )
    engine = _staging_local_engine(requests)
    items = _zmq_commands(requests, prompt_tokens)

    def unsupported_add(command, sequence):
        del command, sequence
        raise AssertionError("synchronous add is outside this benchmark")

    server = ZmqFrontendServer(
        engine_id=0,
        advertised_host="127.0.0.1",
        queue_capacity=requests,
        add=unsupported_add,
        enqueue_batch=engine.enqueue_add_batch,
        admit_batch=engine.admit_add_batch,
    )
    context: zmq.Context | None = None
    client: ZmqFrontendClient | None = None
    server.start(2.0)
    try:
        context = zmq.Context(io_threads=1)
        client = ZmqFrontendClient(
            context=context,
            address=server.address,
            deployment_epoch=server.deployment_epoch,
            engine_id=0,
            queue_capacity=requests,
            startup_timeout_s=2.0,
            request_timeout_s=_TRIAL_TIMEOUT_S,
        )

        acks: list[IngressAck] = []
        receipt_latencies_ms: list[float] = []
        batch_deserialize_ms: list[float] = []
        poll_iterations = 0
        started_at = time.perf_counter()
        deadline = started_at + _TRIAL_TIMEOUT_S
        for start in range(0, requests, batch_size):
            batch = items[start : start + batch_size]
            commands = tuple(item[0] for item in batch)
            reservations = tuple(item[1] for item in batch)
            flight_started_at = time.perf_counter()
            flight = client.admit(commands, reservations)
            while True:
                poll_iterations += 1
                ready, batch_acks = client.poll_ingress(flight)
                if not ready:
                    time.sleep(_POLL_SLEEP_S)
                    if time.perf_counter() >= deadline:
                        raise TimeoutError(
                            "CPU ZMQ benchmark timed out: "
                            f"received={len(acks)}/{requests}"
                        )
                    continue
                expected_acks = len(commands)
                if batch_acks is None or len(batch_acks) != expected_acks:
                    raise AssertionError(
                        "ZMQ admission returned an invalid ACK count"
                    )
                observed_at = time.perf_counter()
                receipt_latencies_ms.append(
                    (observed_at - flight_started_at) * 1000
                )
                if batch_acks:
                    deserialize_ms = batch_acks[0].sequence_deserialize_ms
                    if deserialize_ms is None:
                        raise AssertionError("missing Sequence decode metric")
                    batch_deserialize_ms.append(deserialize_ms)
                acks.extend(batch_acks)
                break
        elapsed_s = time.perf_counter() - started_at
    finally:
        if client is not None:
            client.close()
        if context is not None:
            context.term()
        server.close(2.0)

    payload_bytes = sum(
        ack.sequence_payload_bytes or 0 for ack in acks
    )
    expected_payload_bytes = sum(len(item[0].sequence_payload) for item in items)
    ingress_versions = [ack.ingress_version for ack in acks]
    batch_messages = len(receipt_latencies_ms)
    correctness = {
        "all_positive_receipts": (
            len(acks) == requests and all(ack.enqueued for ack in acks)
        ),
        "all_receipts_precede_commit": (
            engine.scheduler.commit_calls == 0
            and engine._admission_version == 0
        ),
        "all_requests_staged": (
            engine._reserved_slots == requests
            and engine._ingress_adds.qsize() == requests
            and len(engine._ingress_pending_ids) == requests
        ),
        "ingress_versions_are_unique": (
            len(set(ingress_versions)) == requests
            and None not in ingress_versions
        ),
        "payload_accounted_once": payload_bytes == expected_payload_bytes,
    }
    if not all(correctness.values()):
        raise AssertionError(f"ZMQ benchmark invariant failed: {correctness}")

    return {
        "elapsed_ms": _rounded(elapsed_s * 1000),
        "requests_per_second": _rounded(requests / elapsed_s),
        "requests": requests,
        "batch_size_limit": batch_size,
        "batch_messages": batch_messages,
        "max_outstanding_flights": 1,
        "requests_per_message": _rounded(requests / batch_messages),
        "receipt_latency_ms": {
            "p50": _rounded(_percentile(receipt_latencies_ms, 0.50)),
            "p99": _rounded(_percentile(receipt_latencies_ms, 0.99)),
            "max": _rounded(max(receipt_latencies_ms)),
        },
        "sequence_deserialize_ms_per_batch": {
            "p50": _rounded(_percentile(batch_deserialize_ms, 0.50)),
            "p99": _rounded(_percentile(batch_deserialize_ms, 0.99)),
            "max": _rounded(max(batch_deserialize_ms)),
        },
        "sequence_payload_bytes": payload_bytes,
        "scheduler_commit_calls": engine.scheduler.commit_calls,
        "admission_version": engine._admission_version,
        "reserved_slots": engine._reserved_slots,
        "staged_ingress_depth_max": engine._staged_ingress_depth_max,
        "poll_iterations": poll_iterations,
        "correctness": correctness,
    }


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [trial["elapsed_ms"] for trial in trials]
    throughput = [trial["requests_per_second"] for trial in trials]
    return {
        "repeats": len(trials),
        "elapsed_ms": {
            "median": _rounded(statistics.median(elapsed)),
            "min": _rounded(min(elapsed)),
            "max": _rounded(max(elapsed)),
        },
        "requests_per_second": {
            "median": _rounded(statistics.median(throughput)),
            "min": _rounded(min(throughput)),
            "max": _rounded(max(throughput)),
        },
        "trials": trials,
    }


def run_benchmark(
    *,
    requests: int,
    engines: int,
    batch_size: int,
    prompt_tokens: int,
    modeled_commit_gate_ms: float,
    repeats: int,
) -> dict[str, Any]:
    _validate_parameters(
        requests=requests,
        engines=engines,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
    )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if modeled_commit_gate_ms <= 0:
        raise ValueError("modeled_commit_gate_ms must be positive")

    staged_trials = []
    modeled_trials = []
    zmq_trials = []
    for _ in range(repeats):
        staged_trials.append(
            run_router_pipeline_trial(
                requests=requests,
                engines=engines,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                receipt_delay_ms=0.0,
            )
        )
        modeled_trials.append(
            run_router_pipeline_trial(
                requests=requests,
                engines=engines,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                receipt_delay_ms=modeled_commit_gate_ms,
            )
        )
        zmq_trials.append(
            run_zmq_staged_receipt_trial(
                requests=requests,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
            )
        )

    staged = _aggregate_trials(staged_trials)
    modeled = _aggregate_trials(modeled_trials)
    zmq_staged = _aggregate_trials(zmq_trials)
    staged_elapsed = staged["elapsed_ms"]["median"]
    modeled_elapsed = modeled["elapsed_ms"]["median"]
    return {
        "schema_version": 1,
        "scope": "CPU control plane only",
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": _cpu_model(),
            "python": platform.python_version(),
            "pyzmq": zmq.__version__,
        },
        "parameters": {
            "requests": requests,
            "engines": engines,
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "modeled_commit_gate_ms": modeled_commit_gate_ms,
            "repeats": repeats,
        },
        "router_pipeline": {
            "staged_receipt": staged,
            "modeled_commit_gated_receipt": modeled,
            "modeled_elapsed_speedup": _rounded(
                modeled_elapsed / staged_elapsed
            ),
            "baseline_notice": (
                "The commit-gated result is a delay model exercised through "
                "the current Router, not a measurement of old production code."
            ),
        },
        "localhost_zmq_staged_receipt": zmq_staged,
        "success": {
            "router_dispatched_all_before_commit": all(
                all(trial["correctness"].values())
                for trial in staged_trials
            ),
            "localhost_zmq_staged_all_before_commit": all(
                all(trial["correctness"].values())
                for trial in zmq_trials
            ),
        },
        "limitations": [
            "Does not execute a LocalScheduler commit or GPU quantum.",
            "Does not measure Ray, cross-node TCP/RDMA, model throughput, "
            "TTFT, or TPOT.",
            "The modeled commit gate quantifies stop-and-wait sensitivity only.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--engines", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--modeled-commit-gate-ms", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_benchmark(
        requests=args.requests,
        engines=args.engines,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        modeled_commit_gate_ms=args.modeled_commit_gate_ms,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

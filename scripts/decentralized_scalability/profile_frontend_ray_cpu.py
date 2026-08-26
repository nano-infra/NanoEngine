#!/usr/bin/env python3
"""Profile both production frontend directions on a CPU-only Ray cluster.

Ingress uses the real RequestRouter, SP8 AdmissionPlanner, ZMQ multipart
Sequence transport, server-side decode, LocalEngine staged-ingress method, and
positive receipt.  Egress returns production FrontendEventBatch/LoadSnapshot
objects through Ray and consumes ready actor results without waiting for a
straggler.  No scheduler commit, worker, model, RDMA, or CUDA operation runs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import queue
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import zmq
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanodeploy._cpp import Sequence as NanoDeploySequence
from nanodeploy._cpp import serialize_sequence_payload
from nanodeploy.engine.frontend_transport import (
    FrontendFlight,
    ZmqFrontendClient,
    ZmqFrontendServer,
)
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    FrontendEventBatch,
    IngressAck,
)
from nanodeploy.router.request_router import RequestRouter
from scripts.benchmark_least_batch_cpu import _staging_local_engine
from scripts.decentralized_scalability.common import (
    base_metadata,
    clear_proxy_environment,
    fixed_sp_planner_config,
    node_metadata,
    parse_positive_ints,
    parse_scaling_modes,
    requests_for_case,
    select_cluster_nodes,
    sp_load_snapshot,
    summarize,
    write_results,
)


_POLL_SLEEP_S = 0.00005


def build_frontend_event_batch(
    *,
    engine_id: int,
    attention_sp: int,
    event_count: int,
    quantum_id: int,
) -> FrontendEventBatch:
    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    request_base = engine_id * 1_000_000_000 + quantum_id * max(1, event_count)
    return FrontendEventBatch(
        engine_id=engine_id,
        load=sp_load_snapshot(
            engine_id,
            attention_sp=attention_sp,
            capacity_requests=max(1, event_count),
            running=event_count,
            ingress_version=event_count,
            admission_version=event_count,
            quantum_id=quantum_id,
        ),
        add_results=tuple(
            AddResultEvent(
                request_id=request_base + offset,
                engine_id=engine_id,
                accepted=True,
                admission_version=offset + 1,
                local_planned_queue_ms=0.1,
                local_admission_ms=0.2,
            )
            for offset in range(event_count)
        ),
    )


def _accelerator_assignment() -> dict[str, list[str]]:
    return {
        key: list(values)
        for key, values in ray.get_runtime_context().get_accelerator_ids().items()
    }


@ray.remote(num_cpus=1, num_gpus=0)
class _CpuIngressActor:
    def __init__(
        self,
        engine_id: int,
        attention_sp: int,
        queue_capacity: int,
        startup_timeout_s: float,
    ) -> None:
        self.engine_id = engine_id
        self.attention_sp = attention_sp
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()
        self.engine = _staging_local_engine(queue_capacity, engine_id)
        self.engine.config.attention_sp = attention_sp
        self.server = ZmqFrontendServer(
            engine_id=engine_id,
            advertised_host=self.node_ip,
            queue_capacity=queue_capacity,
            add=self._unsupported_add,
            enqueue_batch=self.engine.enqueue_add_batch,
            admit_batch=self.engine.admit_add_batch,
        )
        self.server.start(startup_timeout_s)

    @staticmethod
    def _unsupported_add(command, sequence):
        del command, sequence
        raise AssertionError("synchronous ADD is outside this profiler")

    def descriptor(self) -> dict[str, Any]:
        self.server.raise_if_failed()
        return {
            "engine_id": self.engine_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "frontend_address": self.server.address,
            "frontend_epoch": self.server.deployment_epoch,
            "requested_gpus": 0,
            "assigned_accelerators": _accelerator_assignment(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }

    def stats(self) -> dict[str, Any]:
        self.server.raise_if_failed()
        with self.engine._ingress_lock:
            return {
                "engine_id": self.engine_id,
                "reserved_slots": self.engine._reserved_slots,
                "pending_ingress": len(self.engine._ingress_pending_ids),
                "queue_size": self.engine._ingress_adds.qsize(),
                "ingress_version": self.engine._ingress_version,
                "scheduler_commit_calls": self.engine.scheduler.commit_calls,
            }

    def reset_staged_state(self) -> None:
        self.server.raise_if_failed()
        with self.engine._ingress_lock:
            self.engine._ingress_adds = queue.Queue()
            self.engine._ingress_head = None
            self.engine._reserved_request_ids.clear()
            self.engine._ingress_pending_ids.clear()
            self.engine._admission_pending_ids.clear()
            self.engine._cancelled_ingress_ids.clear()
            self.engine._reserved_slots = 0
            self.engine._staged_ingress_depth_max = 0

    def close(self) -> None:
        self.server.close(2.0)


@ray.remote(num_cpus=1, num_gpus=0)
class _CpuEventActor:
    def __init__(
        self,
        engine_id: int,
        attention_sp: int,
        delay_ms: float,
    ) -> None:
        self.engine_id = engine_id
        self.attention_sp = attention_sp
        self.delay_s = delay_ms / 1000.0
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()

    def descriptor(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "requested_gpus": 0,
            "assigned_accelerators": _accelerator_assignment(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }

    def drain_frontend_events(
        self, event_count: int, quantum_id: int
    ) -> FrontendEventBatch:
        if self.delay_s:
            time.sleep(self.delay_s)
        return build_frontend_event_batch(
            engine_id=self.engine_id,
            attention_sp=self.attention_sp,
            event_count=event_count,
            quantum_id=quantum_id,
        )


@dataclass(frozen=True, slots=True)
class _TimedFlight:
    flight: FrontendFlight
    started_at: float
    request_count: int


class _ZmqAdmissionTransport:
    def __init__(self, client: ZmqFrontendClient) -> None:
        self.engine_id = client.engine_id
        self.client = client
        self.request_ids: list[int] = []
        self.batch_sizes: list[int] = []
        self.receipt_latency_ms: list[float] = []
        self.deserialize_ms: list[float] = []

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> _TimedFlight:
        started_at = time.perf_counter()
        flight = self.client.admit(commands, reservations)
        self.request_ids.extend(command.request_id for command in commands)
        self.batch_sizes.append(len(commands))
        return _TimedFlight(flight, started_at, len(commands))

    def poll_admission_batch(
        self, flight: _TimedFlight
    ) -> tuple[bool, tuple[IngressAck, ...] | None]:
        ready, acks = self.client.poll_ingress(flight.flight)
        if not ready:
            return False, None
        if acks is None or len(acks) != flight.request_count:
            raise RuntimeError("ZMQ ingress ACK count mismatch")
        self.receipt_latency_ms.append(
            (time.perf_counter() - flight.started_at) * 1000.0
        )
        if acks:
            deserialize_ms = acks[0].sequence_deserialize_ms
            if deserialize_ms is None:
                raise RuntimeError("ZMQ ingress omitted deserialize timing")
            self.deserialize_ms.append(deserialize_ms)
        return True, acks

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        del allow_future_ingress
        raise RuntimeError(f"abort is outside this profiler: {request_id}")

    def clear_ingress_abort(self, request_id: int) -> None:
        raise RuntimeError(
            f"clear_ingress_abort is outside this profiler: {request_id}"
        )


def _make_payloads(
    *, request_base: int, requests: int, prompt_tokens: int
) -> tuple[bytes, ...]:
    token_ids = [token_id % 32_000 for token_id in range(prompt_tokens)]
    payloads = []
    for offset in range(requests):
        sequence = NanoDeploySequence(token_ids, 0.1, 16, True)
        sequence.seq_id = request_base + offset
        payloads.append(serialize_sequence_payload(sequence))
    return tuple(payloads)


def _run_ingress_trial(
    *,
    clients: dict[int, ZmqFrontendClient],
    actors: Sequence[Any],
    attention_sp: int,
    requests: int,
    batch_size: int,
    prompt_tokens: int,
    request_base: int,
    request_timeout_s: float,
) -> dict[str, Any]:
    transports = {
        engine_id: _ZmqAdmissionTransport(client)
        for engine_id, client in clients.items()
    }
    router = RequestRouter(
        transports,
        router_policy="least_batch",
        admission_batch_size=batch_size,
        admission_planner_config=fixed_sp_planner_config(
            attention_sp=attention_sp,
            capacity_requests=requests,
            prompt_tokens=prompt_tokens,
        ),
    )
    router.record_loads(
        tuple(
            sp_load_snapshot(
                engine_id,
                attention_sp=attention_sp,
                capacity_requests=requests,
            )
            for engine_id in transports
        )
    )
    payloads = _make_payloads(
        request_base=request_base,
        requests=requests,
        prompt_tokens=prompt_tokens,
    )

    observed: list[IngressAck] = []
    started_at = time.perf_counter()
    for offset, payload in enumerate(payloads):
        request_id = request_base + offset
        router.submit_async(
            request_id=request_id,
            prompt_len=prompt_tokens,
            num_tokens=prompt_tokens,
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            sequence_payload=payload,
        )
    deadline = started_at + request_timeout_s
    while len(observed) < requests:
        ready = router.poll_ingress_acks()
        observed.extend(ready)
        if not ready:
            time.sleep(_POLL_SLEEP_S)
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"frontend ingress timed out: {len(observed)}/{requests}"
            )
    elapsed_s = time.perf_counter() - started_at
    server_stats = tuple(ray.get([actor.stats.remote() for actor in actors]))

    sent_ids = [
        request_id
        for transport in transports.values()
        for request_id in transport.request_ids
    ]
    id_counts = Counter(sent_ids)
    expected_bytes = sum(len(payload) for payload in payloads)
    observed_bytes = sum(ack.sequence_payload_bytes or 0 for ack in observed)
    receipt_samples = [
        sample
        for transport in transports.values()
        for sample in transport.receipt_latency_ms
    ]
    deserialize_samples = [
        sample
        for transport in transports.values()
        for sample in transport.deserialize_ms
    ]
    correctness = {
        "all_receipts_positive": (
            len(observed) == requests and all(ack.enqueued for ack in observed)
        ),
        "each_payload_transferred_once": (
            expected_bytes == observed_bytes
            and len(id_counts) == requests
            and all(count == 1 for count in id_counts.values())
        ),
        "all_requests_staged": (
            sum(item["reserved_slots"] for item in server_stats) == requests
            and sum(item["pending_ingress"] for item in server_stats) == requests
            and sum(item["queue_size"] for item in server_stats) == requests
        ),
        "scheduler_never_committed": all(
            item["scheduler_commit_calls"] == 0 for item in server_stats
        ),
        "router_has_no_ingress_flight": router.pending_ingress_count == 0,
    }
    if not all(correctness.values()):
        raise AssertionError(f"frontend ingress invariant failed: {correctness}")
    return {
        "elapsed_ms": elapsed_s * 1000.0,
        "requests_per_second": requests / elapsed_s,
        "receipt_latency_ms": summarize(receipt_samples),
        "deserialize_ms_per_message": summarize(deserialize_samples),
        "payload_bytes": observed_bytes,
        "batch_messages": len(receipt_samples),
        "correctness": correctness,
    }


def _create_ingress_actors(
    *,
    nodes: Sequence[dict[str, Any]],
    attention_sp: int,
    queue_capacity: int,
    startup_timeout_s: float,
) -> tuple[list[Any], tuple[dict[str, Any], ...]]:
    actors = []
    for engine_id, node in enumerate(nodes):
        actor = _CpuIngressActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                str(node["NodeID"]), soft=False
            )
        ).remote(engine_id, attention_sp, queue_capacity, startup_timeout_s)
        actors.append(actor)
    descriptors = tuple(
        ray.get(
            [actor.descriptor.remote() for actor in actors],
            timeout=startup_timeout_s,
        )
    )
    _validate_placements(descriptors, nodes)
    return actors, descriptors


def _validate_placements(
    descriptors: Sequence[dict[str, Any]], nodes: Sequence[dict[str, Any]]
) -> None:
    if len(descriptors) != len(nodes):
        raise RuntimeError("actor placement count mismatch")
    for descriptor, node in zip(descriptors, nodes, strict=True):
        if descriptor["node_id"] != str(node["NodeID"]):
            raise RuntimeError(
                f"engine {descriptor['engine_id']} placement mismatch"
            )
        assigned_gpus = descriptor["assigned_accelerators"].get("GPU", [])
        if descriptor["requested_gpus"] != 0 or assigned_gpus:
            raise RuntimeError(
                f"CPU-only actor received GPU resources: {descriptor}"
            )


def _close_actors(actors: Iterable[Any], timeout_s: float) -> None:
    actors = tuple(actors)
    if not actors:
        return
    close_refs = []
    for actor in actors:
        try:
            close_method = getattr(actor, "close", None)
            if close_method is not None:
                close_refs.append(close_method.remote())
        except Exception:
            pass
    if close_refs:
        try:
            ray.get(close_refs, timeout=timeout_s)
        except Exception:
            pass
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass


def _profile_ingress_case(
    *,
    nodes: Sequence[dict[str, Any]],
    scaling_mode: str,
    attention_sp: int,
    requests: int,
    batch_size: int,
    prompt_tokens: int,
    repeats: int,
    startup_timeout_s: float,
    request_timeout_s: float,
) -> dict[str, Any]:
    actors: list[Any] = []
    clients: dict[int, ZmqFrontendClient] = {}
    context: zmq.Context | None = None
    try:
        actors, descriptors = _create_ingress_actors(
            nodes=nodes,
            attention_sp=attention_sp,
            queue_capacity=requests + 1,
            startup_timeout_s=startup_timeout_s,
        )
        context = zmq.Context(io_threads=1)
        for descriptor in descriptors:
            clients[descriptor["engine_id"]] = ZmqFrontendClient(
                context=context,
                address=descriptor["frontend_address"],
                deployment_epoch=descriptor["frontend_epoch"],
                engine_id=descriptor["engine_id"],
                queue_capacity=requests + 1,
                startup_timeout_s=startup_timeout_s,
                request_timeout_s=request_timeout_s,
            )
        trials = []
        for repeat_index in range(repeats):
            trials.append(
                _run_ingress_trial(
                    clients=clients,
                    actors=actors,
                    attention_sp=attention_sp,
                    requests=requests,
                    batch_size=batch_size,
                    prompt_tokens=prompt_tokens,
                    request_base=repeat_index * requests,
                    request_timeout_s=request_timeout_s,
                )
            )
            ray.get([actor.reset_staged_state.remote() for actor in actors])
        return {
            "direction": "router_to_local_engine_zmq",
            "scaling_mode": scaling_mode,
            "nodes": len(nodes),
            "engines": len(nodes),
            "attention_sp": attention_sp,
            "logical_workers": len(nodes) * attention_sp,
            "requests": requests,
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "sample_count": repeats,
            "elapsed_ms": summarize(trial["elapsed_ms"] for trial in trials),
            "items_per_second": summarize(
                trial["requests_per_second"] for trial in trials
            ),
            "latency_ms": summarize(
                trial["receipt_latency_ms"]["p99"] for trial in trials
            ),
            "server_work_ms": summarize(
                trial["deserialize_ms_per_message"]["p50"] for trial in trials
            ),
            "payload_bytes_per_sample": summarize(
                trial["payload_bytes"] for trial in trials
            ),
            "placement": descriptors,
            "correctness": [trial["correctness"] for trial in trials],
        }
    finally:
        for client in clients.values():
            client.close()
        if context is not None:
            context.term()
        _close_actors(actors, startup_timeout_s)


def _event_counts(
    *, scaling_mode: str, nodes: int, max_nodes: int, batch_size: int
) -> tuple[int, ...]:
    if scaling_mode == "weak":
        return (batch_size,) * nodes
    total = batch_size * max_nodes
    base, extra = divmod(total, nodes)
    return tuple(base + (index < extra) for index in range(nodes))


def _event_iteration(
    actors: Sequence[Any],
    event_counts: Sequence[int],
    *,
    quantum_id: int,
) -> dict[str, Any]:
    begin = time.perf_counter()
    refs = [
        actor.drain_frontend_events.remote(event_count, quantum_id)
        for actor, event_count in zip(actors, event_counts, strict=True)
    ]
    submit_end = time.perf_counter()
    pending = list(refs)
    ref_to_engine = {ref: engine_id for engine_id, ref in enumerate(refs)}
    batches: dict[int, FrontendEventBatch] = {}
    ready_latency_ms: dict[int, float] = {}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=0)
        if not ready:
            time.sleep(_POLL_SLEEP_S)
            continue
        batch = ray.get(ready[0])
        engine_id = ref_to_engine[ready[0]]
        batches[engine_id] = batch
        ready_latency_ms[engine_id] = (time.perf_counter() - begin) * 1000.0
    end = time.perf_counter()
    ordered = tuple(batches[index] for index in range(len(actors)))
    for engine_id, (batch, expected_count) in enumerate(
        zip(ordered, event_counts, strict=True)
    ):
        if (
            batch.engine_id != engine_id
            or len(batch.load.rank_loads) != 8
            or len(batch.add_results) != expected_count
        ):
            raise RuntimeError("invalid Ray frontend event batch")
    total_events = sum(event_counts)
    return {
        "submit_ms": (submit_end - begin) * 1000.0,
        "roundtrip_ms": (end - begin) * 1000.0,
        "first_ready_ms": min(ready_latency_ms.values()),
        "last_ready_ms": max(ready_latency_ms.values()),
        "events_per_second": total_events / (end - begin),
        "payload_bytes": sum(
            len(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL))
            for batch in ordered
        ),
        "ready_latency_ms": ready_latency_ms,
    }


def _profile_egress_case(
    *,
    nodes: Sequence[dict[str, Any]],
    max_nodes: int,
    scaling_mode: str,
    attention_sp: int,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
    straggler_delay_ms: float,
    startup_timeout_s: float,
) -> dict[str, Any]:
    actors = []
    try:
        for engine_id, node in enumerate(nodes):
            delay_ms = straggler_delay_ms if engine_id == 0 else 0.0
            actor = _CpuEventActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    str(node["NodeID"]), soft=False
                )
            ).remote(engine_id, attention_sp, delay_ms)
            actors.append(actor)
        descriptors = tuple(
            ray.get(
                [actor.descriptor.remote() for actor in actors],
                timeout=startup_timeout_s,
            )
        )
        _validate_placements(descriptors, nodes)
        event_counts = _event_counts(
            scaling_mode=scaling_mode,
            nodes=len(nodes),
            max_nodes=max_nodes,
            batch_size=batch_size,
        )
        for quantum_id in range(warmup_iterations):
            _event_iteration(
                actors,
                event_counts,
                quantum_id=quantum_id,
            )
        samples = [
            _event_iteration(
                actors,
                event_counts,
                quantum_id=warmup_iterations + iteration,
            )
            for iteration in range(iterations)
        ]
        return {
            "direction": "local_engine_to_router_ray_events",
            "scaling_mode": scaling_mode,
            "nodes": len(nodes),
            "engines": len(nodes),
            "attention_sp": attention_sp,
            "logical_workers": len(nodes) * attention_sp,
            "requests": sum(event_counts),
            "batch_size": batch_size,
            "prompt_tokens": None,
            "sample_count": iterations,
            "elapsed_ms": summarize(sample["roundtrip_ms"] for sample in samples),
            "items_per_second": summarize(
                sample["events_per_second"] for sample in samples
            ),
            "latency_ms": summarize(sample["last_ready_ms"] for sample in samples),
            "server_work_ms": summarize(sample["submit_ms"] for sample in samples),
            "payload_bytes_per_sample": summarize(
                sample["payload_bytes"] for sample in samples
            ),
            "straggler_delay_ms": straggler_delay_ms,
            "placement": descriptors,
            "correctness": [
                {
                    "all_engine_batches_received": len(
                        sample["ready_latency_ms"]
                    )
                    == len(nodes),
                    "nonblocking_ready_order_recorded": (
                        sample["first_ready_ms"] <= sample["last_ready_ms"]
                    ),
                    "configured_straggler_completed_last": (
                        straggler_delay_ms == 0.0
                        or max(
                            sample["ready_latency_ms"],
                            key=sample["ready_latency_ms"].__getitem__,
                        )
                        == 0
                    ),
                }
                for sample in samples
            ],
        }
    finally:
        _close_actors(actors, startup_timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--node-counts",
        type=parse_positive_ints,
        default=parse_positive_ints("1,2,3"),
    )
    parser.add_argument(
        "--node-ip",
        action="append",
        default=[],
        help="Target Ray NodeManagerAddress; repeat in desired node order.",
    )
    parser.add_argument("--attention-sp", type=int, default=8)
    parser.add_argument(
        "--scaling-modes",
        type=parse_scaling_modes,
        default=parse_scaling_modes("strong,weak"),
    )
    parser.add_argument("--strong-total-requests", type=int, default=4096)
    parser.add_argument("--requests-per-engine", type=int, default=4096)
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=parse_positive_ints("32,64,128"),
    )
    parser.add_argument(
        "--prompt-lengths",
        type=parse_positive_ints,
        default=parse_positive_ints("32,8000"),
    )
    parser.add_argument("--ingress-repeats", type=int, default=5)
    parser.add_argument("--event-warmup-iterations", type=int, default=20)
    parser.add_argument("--event-iterations", type=int, default=500)
    parser.add_argument("--straggler-delay-ms", type=float, default=0.0)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if any(node_count not in {1, 2, 3, 4} for node_count in args.node_counts):
        parser.error("--node-counts must contain only 1,2,3,4")
    if args.attention_sp != 8:
        parser.error("this production profiler requires --attention-sp 8")
    for name in (
        "strong_total_requests",
        "requests_per_engine",
        "ingress_repeats",
        "event_iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.event_warmup_iterations < 0 or args.straggler_delay_ms < 0:
        parser.error("warmup iterations and straggler delay must be non-negative")
    if args.startup_timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("timeouts must be positive")


def _annotate_scaling(records: list[dict[str, Any]]) -> None:
    baselines = {
        (
            record["direction"],
            record["scaling_mode"],
            record["batch_size"],
            record["prompt_tokens"],
        ): record["items_per_second"]["p50"]
        for record in records
        if record["nodes"] == 1
    }
    for record in records:
        baseline = baselines.get(
            (
                record["direction"],
                record["scaling_mode"],
                record["batch_size"],
                record["prompt_tokens"],
            )
        )
        ratio = (
            record["items_per_second"]["p50"] / baseline
            if baseline is not None
            else None
        )
        record["throughput_ratio_vs_one_node"] = ratio
        record["weak_scaling_efficiency_vs_one_node"] = (
            ratio / record["nodes"]
            if ratio is not None and record["scaling_mode"] == "weak"
            else None
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if ray.is_initialized():
        raise RuntimeError("profiler requires a fresh Ray driver")
    removed_proxies = clear_proxy_environment()
    ray_context = ray.init(
        address=args.ray_address,
        ignore_reinit_error=False,
        log_to_driver=False,
    )
    ingress_records: list[dict[str, Any]] = []
    egress_records: list[dict[str, Any]] = []
    try:
        max_nodes = max(args.node_counts)
        selected_nodes = select_cluster_nodes(
            ray.nodes(),
            count=max_nodes,
            requested_node_ips=args.node_ip,
        )
        ingress_total = (
            len(args.scaling_modes)
            * len(args.node_counts)
            * len(args.batch_sizes)
            * len(args.prompt_lengths)
        )
        case_index = 0
        for scaling_mode in args.scaling_modes:
            for batch_size in args.batch_sizes:
                for prompt_tokens in args.prompt_lengths:
                    for node_count in args.node_counts:
                        case_index += 1
                        requests = requests_for_case(
                            scaling_mode=scaling_mode,
                            engines=node_count,
                            strong_total_requests=args.strong_total_requests,
                            requests_per_engine=args.requests_per_engine,
                        )
                        print(
                            f"[ingress {case_index}/{ingress_total}] "
                            f"mode={scaling_mode} nodes={node_count} "
                            f"requests={requests} batch={batch_size} "
                            f"prompt={prompt_tokens}",
                            flush=True,
                        )
                        record = _profile_ingress_case(
                            nodes=selected_nodes[:node_count],
                            scaling_mode=scaling_mode,
                            attention_sp=args.attention_sp,
                            requests=requests,
                            batch_size=batch_size,
                            prompt_tokens=prompt_tokens,
                            repeats=args.ingress_repeats,
                            startup_timeout_s=args.startup_timeout_s,
                            request_timeout_s=args.request_timeout_s,
                        )
                        ingress_records.append(record)
                        print(
                            "  qps median="
                            f"{record['items_per_second']['p50']:.1f} "
                            "receipt-p99 median="
                            f"{record['latency_ms']['p50']:.3f} ms",
                            flush=True,
                        )

        egress_total = (
            len(args.scaling_modes)
            * len(args.node_counts)
            * len(args.batch_sizes)
        )
        case_index = 0
        for scaling_mode in args.scaling_modes:
            for batch_size in args.batch_sizes:
                for node_count in args.node_counts:
                    case_index += 1
                    print(
                        f"[events {case_index}/{egress_total}] "
                        f"mode={scaling_mode} nodes={node_count} "
                        f"batch={batch_size}",
                        flush=True,
                    )
                    record = _profile_egress_case(
                        nodes=selected_nodes[:node_count],
                        max_nodes=max_nodes,
                        scaling_mode=scaling_mode,
                        attention_sp=args.attention_sp,
                        batch_size=batch_size,
                        warmup_iterations=args.event_warmup_iterations,
                        iterations=args.event_iterations,
                        straggler_delay_ms=args.straggler_delay_ms,
                        startup_timeout_s=args.startup_timeout_s,
                    )
                    egress_records.append(record)
                    print(
                        "  events/s median="
                        f"{record['items_per_second']['p50']:.1f} "
                        "roundtrip p99="
                        f"{record['elapsed_ms']['p99']:.3f} ms",
                        flush=True,
                    )

        metadata = base_metadata(
            "nanodeploy-decentralized-frontend-ray-cpu-scalability"
        )
        metadata.update(
            {
                "ray_version": ray.__version__,
                "ray_address": ray_context.address_info.get(
                    "gcs_address", args.ray_address
                ),
                "proxy_variables_removed": removed_proxies,
                "selected_nodes": node_metadata(selected_nodes),
                "timed_scope": (
                    "SP8 RequestRouter through cross-node ZMQ staged receipt; "
                    "and LocalEngine-shaped SP8 FrontendEventBatch return through Ray"
                ),
                "excluded_scope": (
                    "LocalScheduler commit/decode, ModelRunner, worker collectives, "
                    "DecodeCoordinator wakeup, RDMA, model weights, CUDA, and "
                    "GPU execution"
                ),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
        )
        _annotate_scaling(ingress_records)
        _annotate_scaling(egress_records)
        ingress_json, ingress_csv = write_results(
            args.output_dir,
            stem="frontend_ingress_cpu_scalability",
            metadata=metadata,
            records=ingress_records,
        )
        egress_json, egress_csv = write_results(
            args.output_dir,
            stem="frontend_events_cpu_scalability",
            metadata=metadata,
            records=egress_records,
        )
        for path in (ingress_json, ingress_csv, egress_json, egress_csv):
            print(f"Wrote {path}")
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

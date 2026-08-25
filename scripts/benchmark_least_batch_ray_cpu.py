#!/usr/bin/env python3
"""Run the least-batch CPU control-plane benchmark across Ray nodes.

Run this module once on a Ray head node. Ray places CPU-only staged-ingress
actors on distinct nodes; the driver then routes real Sequence payloads to
them through NanoDeploy's ZMQ frontend transport. No GPU resource is requested
and no NanoDeploy worker, model, scheduler drain, or CUDA context is started.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence as TypingSequence

import ray
import zmq
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanodeploy._cpp import Sequence, serialize_sequence_payload
from nanodeploy.engine.frontend_transport import (
    FrontendFlight,
    ZmqFrontendClient,
    ZmqFrontendServer,
)
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AdmissionReservation,
    AbortResult,
    IngressAck,
    OwnerState,
)
from nanodeploy.router.admission_planner import AdmissionPlannerConfig
from nanodeploy.router.request_router import RequestRouter
from scripts.benchmark_least_batch_cpu import (
    _cpu_model,
    _percentile,
    _rounded,
    _router_load,
    _staging_local_engine,
    _validate_parameters,
)


_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
_POLL_SLEEP_S = 0.00005


def _unsupported_add(command, sequence):
    del command, sequence
    raise AssertionError("synchronous add is outside this benchmark")


@ray.remote(num_cpus=1, num_gpus=0)
class _RayCpuIngressServer:
    """One CPU-only LocalEngine staged-ingress endpoint."""

    def __init__(
        self,
        engine_id: int,
        queue_capacity: int,
        startup_timeout_s: float,
    ) -> None:
        self.engine_id = engine_id
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()
        self.engine = _staging_local_engine(queue_capacity, engine_id)
        self.server = ZmqFrontendServer(
            engine_id=engine_id,
            advertised_host=self.node_ip,
            queue_capacity=queue_capacity,
            add=_unsupported_add,
            enqueue_batch=self.engine.enqueue_add_batch,
            admit_batch=self.engine.admit_add_batch,
        )
        self.server.start(startup_timeout_s)

    def descriptor(self) -> dict[str, Any]:
        self.server.raise_if_failed()
        return {
            "engine_id": self.engine_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "frontend_address": self.server.address,
            "frontend_epoch": self.server.deployment_epoch,
            "requested_gpus": 0,
        }

    def stats(self) -> dict[str, Any]:
        self.server.raise_if_failed()
        with self.engine._ingress_lock:
            return {
                "engine_id": self.engine_id,
                "node_id": self.node_id,
                "node_ip": self.node_ip,
                "reserved_slots": self.engine._reserved_slots,
                "pending_ingress": len(self.engine._ingress_pending_ids),
                "ingress_queue_size": self.engine._ingress_adds.qsize(),
                "ingress_version": self.engine._ingress_version,
                "admission_version": self.engine._admission_version,
                "staged_ingress_depth_max": (
                    self.engine._staged_ingress_depth_max
                ),
                "scheduler_commit_calls": (
                    self.engine.scheduler.commit_calls
                ),
            }

    def close(self) -> None:
        self.server.close(2.0)


def _clear_proxy_environment() -> tuple[str, ...]:
    removed = tuple(name for name in _PROXY_ENV_NAMES if name in os.environ)
    for name in _PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    return removed


def _select_cluster_nodes(
    nodes: Iterable[dict[str, Any]],
    *,
    num_nodes: int,
    requested_node_ips: TypingSequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    alive = [
        node
        for node in nodes
        if node.get("Alive", False)
        and float(node.get("Resources", {}).get("CPU", 0.0)) >= 1.0
    ]
    if requested_node_ips:
        requested = tuple(dict.fromkeys(requested_node_ips))
        if len(requested) != num_nodes:
            raise ValueError(
                "the number of unique --node-ip values must equal --ray-nodes"
            )
        by_ip = {node.get("NodeManagerAddress"): node for node in alive}
        missing = [node_ip for node_ip in requested if node_ip not in by_ip]
        if missing:
            raise RuntimeError(
                "requested Ray nodes are not alive or do not advertise a "
                f"CPU resource: {missing}"
            )
        return tuple(by_ip[node_ip] for node_ip in requested)
    ordered = sorted(
        alive,
        key=lambda node: (
            str(node.get("NodeManagerAddress", "")),
            str(node.get("NodeID", "")),
        ),
    )
    if len(ordered) < num_nodes:
        raise RuntimeError(
            "insufficient alive Ray nodes with CPU resources: "
            f"need={num_nodes}, found={len(ordered)}"
        )
    return tuple(ordered[:num_nodes])


@dataclass(frozen=True, slots=True)
class _TimedZmqFlight:
    flight: FrontendFlight
    started_at: float
    request_count: int


class _RayZmqTransport:
    def __init__(self, client: ZmqFrontendClient) -> None:
        self.engine_id = client.engine_id
        self.client = client
        self.command_request_ids: list[int] = []
        self.batch_sizes: list[int] = []
        self.receipt_latencies_ms: list[float] = []
        self.sequence_deserialize_ms: list[float] = []

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> _TimedZmqFlight:
        started_at = time.perf_counter()
        flight = self.client.admit(commands, reservations)
        self.command_request_ids.extend(
            command.request_id for command in commands
        )
        self.batch_sizes.append(len(commands))
        return _TimedZmqFlight(flight, started_at, len(commands))

    def poll_admission_batch(
        self, handle: _TimedZmqFlight
    ) -> tuple[bool, tuple[IngressAck, ...] | None]:
        ready, acks = self.client.poll_ingress(handle.flight)
        if not ready:
            return False, None
        if acks is None or len(acks) != handle.request_count:
            raise AssertionError("remote ZMQ ACK count mismatch")
        self.receipt_latencies_ms.append(
            (time.perf_counter() - handle.started_at) * 1000
        )
        if acks:
            deserialize_ms = acks[0].sequence_deserialize_ms
            if deserialize_ms is None:
                raise AssertionError("remote ACK omitted deserialize timing")
            self.sequence_deserialize_ms.append(deserialize_ms)
        return True, acks

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        del allow_future_ingress
        raise RuntimeError(
            f"abort is outside the CPU throughput workload: {request_id}"
        )

    def clear_ingress_abort(self, request_id: int) -> None:
        raise RuntimeError(
            f"clear_ingress_abort is outside this workload: {request_id}"
        )

    def close(self) -> None:
        self.client.close()


def _sequence_payloads(
    requests: int, prompt_tokens: int
) -> tuple[bytes, ...]:
    payloads = []
    for request_id in range(requests):
        token_ids = [
            (request_id + offset) % 32_000
            for offset in range(prompt_tokens)
        ]
        sequence = Sequence(token_ids, 0.0, 16, True)
        sequence.seq_id = request_id
        payloads.append(serialize_sequence_payload(sequence))
    return tuple(payloads)


def _planner_config(
    requests: int, prompt_tokens: int
) -> AdmissionPlannerConfig:
    return AdmissionPlannerConfig(
        attention_sp=1,
        kvcache_block_size=16,
        max_num_seqs=requests + 1,
        max_num_batched_tokens=requests * prompt_tokens + 1,
        max_num_recv_seqs=requests + 1,
        reserved_blocks_per_req=0.0,
        segment_size=64,
        queue_capacity=requests + 1,
    )


def _summarize(values: Iterable[float]) -> dict[str, float]:
    samples = tuple(values)
    return {
        "p50": _rounded(_percentile(samples, 0.50)),
        "p99": _rounded(_percentile(samples, 0.99)),
        "max": _rounded(max(samples)),
    }


def _run_distributed_trial(
    *,
    selected_nodes: tuple[dict[str, Any], ...],
    requests: int,
    engines: int,
    batch_size: int,
    prompt_tokens: int,
    startup_timeout_s: float,
    request_timeout_s: float,
) -> dict[str, Any]:
    actors = []
    transports: dict[int, _RayZmqTransport] = {}
    context: zmq.Context | None = None
    try:
        for engine_id in range(engines):
            node = selected_nodes[engine_id % len(selected_nodes)]
            strategy = NodeAffinitySchedulingStrategy(
                node_id=str(node["NodeID"]), soft=False
            )
            actor = _RayCpuIngressServer.options(
                scheduling_strategy=strategy
            ).remote(engine_id, requests, startup_timeout_s)
            actors.append(actor)
        descriptors = tuple(
            ray.get(
                [actor.descriptor.remote() for actor in actors],
                timeout=startup_timeout_s,
            )
        )

        context = zmq.Context(io_threads=1)
        for descriptor in descriptors:
            client = ZmqFrontendClient(
                context=context,
                address=descriptor["frontend_address"],
                deployment_epoch=descriptor["frontend_epoch"],
                engine_id=descriptor["engine_id"],
                queue_capacity=requests,
                startup_timeout_s=startup_timeout_s,
                request_timeout_s=request_timeout_s,
            )
            transports[descriptor["engine_id"]] = _RayZmqTransport(client)

        router = RequestRouter(
            transports,
            router_policy="least_batch",
            admission_batch_size=batch_size,
            admission_planner_config=_planner_config(
                requests, prompt_tokens
            ),
        )
        router.record_loads(
            tuple(
                _router_load(engine_id, requests)
                for engine_id in transports
            )
        )
        payloads = _sequence_payloads(requests, prompt_tokens)

        acks: list[IngressAck] = []
        started_at = time.perf_counter()
        for request_id, payload in enumerate(payloads):
            router.submit_async(
                request_id=request_id,
                prompt_len=prompt_tokens,
                num_tokens=prompt_tokens,
                max_tokens=16,
                temperature=0.0,
                ignore_eos=True,
                sequence_payload=payload,
            )
        deadline = started_at + request_timeout_s
        poll_iterations = 0
        while len(acks) < requests:
            poll_iterations += 1
            ready = router.poll_ingress_acks()
            acks.extend(ready)
            if not ready:
                time.sleep(_POLL_SLEEP_S)
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "Ray CPU benchmark timed out: "
                    f"received={len(acks)}/{requests}"
                )
        elapsed_s = time.perf_counter() - started_at

        server_stats = tuple(
            ray.get(
                [actor.stats.remote() for actor in actors],
                timeout=startup_timeout_s,
            )
        )
        metrics = router.admission_metrics()
        per_engine = tuple(metrics["per_engine"].values())
        sent_ids = [
            request_id
            for transport in transports.values()
            for request_id in transport.command_request_ids
        ]
        id_counts = Counter(sent_ids)
        receipt_latencies = [
            latency
            for transport in transports.values()
            for latency in transport.receipt_latencies_ms
        ]
        deserialize_times = [
            latency
            for transport in transports.values()
            for latency in transport.sequence_deserialize_ms
        ]
        batch_messages = sum(item["batch_messages"] for item in per_engine)
        commits = sum(item["commits"] for item in per_engine)
        payload_bytes = sum(
            ack.sequence_payload_bytes or 0 for ack in acks
        )
        expected_payload_bytes = sum(len(payload) for payload in payloads)
        placed_node_ids = {item["node_id"] for item in server_stats}
        expected_node_ids = {
            str(node["NodeID"]) for node in selected_nodes
        }
        ingress_versions = [
            (ack.engine_id, ack.ingress_version) for ack in acks
        ]
        correctness = {
            "actors_cover_selected_nodes": (
                placed_node_ids == expected_node_ids
            ),
            "actors_match_requested_node_affinity": all(
                descriptor["node_id"]
                == str(
                    selected_nodes[
                        descriptor["engine_id"] % len(selected_nodes)
                    ]["NodeID"]
                )
                for descriptor in descriptors
            ),
            "all_positive_receipts": (
                len(acks) == requests and all(ack.enqueued for ack in acks)
            ),
            "all_pending_add_before_commit": (
                router.pending_add_count == requests and commits == 0
            ),
            "all_requests_staged": (
                sum(item["reserved_slots"] for item in server_stats)
                == requests
                and sum(
                    item["ingress_queue_size"] for item in server_stats
                )
                == requests
                and sum(
                    item["pending_ingress"] for item in server_stats
                )
                == requests
            ),
            "ingress_versions_are_unique_per_engine": (
                len(set(ingress_versions)) == requests
                and all(version is not None for _, version in ingress_versions)
            ),
            "remote_schedulers_untouched": all(
                item["scheduler_commit_calls"] == 0
                and item["admission_version"] == 0
                for item in server_stats
            ),
            "payload_accounted_once": (
                payload_bytes == expected_payload_bytes
                and len(id_counts) == requests
                and all(count == 1 for count in id_counts.values())
            ),
            "router_has_no_pending_ingress": (
                router.pending_ingress_count == 0
            ),
        }
        if not all(correctness.values()):
            raise AssertionError(
                f"Ray CPU benchmark invariant failed: {correctness}"
            )

        return {
            "elapsed_ms": _rounded(elapsed_s * 1000),
            "requests_per_second": _rounded(requests / elapsed_s),
            "requests": requests,
            "engines": engines,
            "nodes": len(selected_nodes),
            "batch_messages": batch_messages,
            "requests_per_message": _rounded(requests / batch_messages),
            "positive_receipts": len(acks),
            "scheduler_commits": commits,
            "pending_add": router.pending_add_count,
            "sequence_payload_bytes": payload_bytes,
            "receipt_latency_ms": _summarize(receipt_latencies),
            "sequence_deserialize_ms_per_batch": _summarize(
                deserialize_times
            ),
            "poll_iterations": poll_iterations,
            "engine_placements": descriptors,
            "server_stats": server_stats,
            "correctness": correctness,
        }
    finally:
        for transport in transports.values():
            transport.close()
        if context is not None:
            context.term()
        if actors:
            try:
                ray.get(
                    [actor.close.remote() for actor in actors],
                    timeout=startup_timeout_s,
                )
            except Exception:
                pass
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [trial["elapsed_ms"] for trial in trials]
    throughput = [trial["requests_per_second"] for trial in trials]
    receipt_p99 = [trial["receipt_latency_ms"]["p99"] for trial in trials]
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
        "receipt_p99_ms": {
            "median": _rounded(statistics.median(receipt_p99)),
            "min": _rounded(min(receipt_p99)),
            "max": _rounded(max(receipt_p99)),
        },
        "trials": trials,
    }


def run_ray_benchmark(
    *,
    ray_address: str,
    ray_nodes: int,
    requested_node_ips: TypingSequence[str],
    requests: int,
    engines: int,
    batch_size: int,
    prompt_tokens: int,
    repeats: int,
    startup_timeout_s: float,
    request_timeout_s: float,
) -> dict[str, Any]:
    _validate_parameters(
        requests=requests,
        engines=engines,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
    )
    if ray_nodes <= 0 or repeats <= 0:
        raise ValueError("ray_nodes and repeats must be positive")
    if engines < ray_nodes:
        raise ValueError("engines must be at least ray_nodes")
    if startup_timeout_s <= 0 or request_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    if ray.is_initialized():
        raise RuntimeError("benchmark requires a fresh Ray driver")

    removed_proxy_names = _clear_proxy_environment()
    ray_context = ray.init(
        address=ray_address,
        ignore_reinit_error=False,
        log_to_driver=False,
    )
    try:
        selected_nodes = _select_cluster_nodes(
            ray.nodes(),
            num_nodes=ray_nodes,
            requested_node_ips=requested_node_ips,
        )
        trials = [
            _run_distributed_trial(
                selected_nodes=selected_nodes,
                requests=requests,
                engines=engines,
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                startup_timeout_s=startup_timeout_s,
                request_timeout_s=request_timeout_s,
            )
            for _ in range(repeats)
        ]
        selected = tuple(
            {
                "node_id": str(node["NodeID"]),
                "node_ip": node.get("NodeManagerAddress"),
                "cpu_resources": node.get("Resources", {}).get("CPU", 0),
            }
            for node in selected_nodes
        )
        return {
            "schema_version": 1,
            "scope": "Ray-launched multi-node CPU control plane",
            "environment": {
                "driver_hostname": platform.node(),
                "platform": platform.platform(),
                "processor": _cpu_model(),
                "python": platform.python_version(),
                "ray": ray.__version__,
                "pyzmq": zmq.__version__,
                "ray_gcs_address": ray_context.address_info.get(
                    "gcs_address", ray_address
                ),
                "proxy_variables_removed": removed_proxy_names,
            },
            "parameters": {
                "requests": requests,
                "engines": engines,
                "ray_nodes": ray_nodes,
                "batch_size": batch_size,
                "prompt_tokens": prompt_tokens,
                "repeats": repeats,
            },
            "selected_nodes": selected,
            "distributed_router_zmq_staged_receipt": (
                _aggregate_trials(trials)
            ),
            "success": {
                "all_trials_passed_control_plane_invariants": all(
                    all(trial["correctness"].values())
                    for trial in trials
                )
            },
            "measurement_boundary": (
                "Elapsed time includes RequestRouter submit/planning and "
                "cross-node ZMQ send/decode/receipt. Ray actor startup and "
                "post-run stats RPC are outside the timed interval."
            ),
            "limitations": [
                "Ray is used for placement and lifecycle, not request payloads.",
                "No scheduler commit, CUDA context, GPU worker, or model is run.",
                "Does not measure RDMA, GPU quantum, throughput, TTFT, or TPOT.",
            ],
        }
    finally:
        ray.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-nodes", type=int, default=2)
    parser.add_argument(
        "--node-ip",
        action="append",
        default=[],
        help=(
            "Optional target NodeManagerAddress; repeat once per Ray node. "
            "Without it, the first alive CPU nodes are selected."
        ),
    )
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--engines", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: TypingSequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_ray_benchmark(
        ray_address=args.ray_address,
        ray_nodes=args.ray_nodes,
        requested_node_ips=args.node_ip,
        requests=args.requests,
        engines=args.engines,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        repeats=args.repeats,
        startup_timeout_s=args.startup_timeout_s,
        request_timeout_s=args.request_timeout_s,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

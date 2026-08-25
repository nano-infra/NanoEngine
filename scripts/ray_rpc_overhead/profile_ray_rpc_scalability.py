#!/usr/bin/env python3
"""Profile NanoDeploy's Ray invocation and result fanout overhead.

NanoDeploy's DLSlime data path still uses Ray to invoke ``ModelRunner.run`` on
every worker and to return sampled token IDs.  This CPU-only benchmark creates
real Ray actors with the same ``run`` call signature and returns
``batch_size_per_gpu * loop_count`` token IDs per logical worker.  It can send
the empty ``dp_seqs`` control argument used when ``use_dlslime_rpc=True`` or
full NanoDeploy ``Sequence`` objects to characterize Ray's former input data
path, including serialization and deserialization.

The benchmark measures Ray actor submission and result round-trip overhead.  It
does not launch ModelRunner, execute GPU kernels, or emulate DLSlime/RDMA
traffic.  By default logical workers are colocated in one isolated local Ray
instance.  An external Ray cluster and a fixed workers-per-node placement can
also be supplied to measure the same control RPC across multiple hosts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
import platform
import socket
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# This standalone profiler is deliberately launched with python3, never
# ``uv run``. Disable Ray's import-time uv parent-process discovery because it
# is both unnecessary here and unreliable in PID-namespaced containers where
# an ancestor can disappear before psutil walks the chain.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from nanodeploy._cpp import Sequence as NanoDeploySequence
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
class RayDecodeControlWorker:
    """Minimal actor implementing the Ray-facing portion of ModelRunner.run."""

    def __init__(
        self,
        logical_rank: int,
        loop_count: int,
        ray_sequence_length: int,
    ) -> None:
        self.logical_rank = logical_rank
        self.loop_count = loop_count
        self.ray_sequence_length = ray_sequence_length
        self._token_rows: list[list[int]] = []

    def configure_batch(self, batch_size_per_gpu: int) -> int:
        self._token_rows = [
            [
                (self.logical_rank + row_index + step_index) % 32_000
                for step_index in range(self.loop_count)
            ]
            for row_index in range(batch_size_per_gpu)
        ]
        return len(self._token_rows)

    def ready(self) -> int:
        return self.logical_rank

    def placement(self) -> dict[str, object]:
        return {
            "logical_rank": self.logical_rank,
            "hostname": socket.gethostname(),
            "node_id": str(ray.get_runtime_context().get_node_id()),
        }

    def run(
        self,
        dp_seqs: list[object],
        is_prefill: bool,
        enable_rpc: bool,
        send_timestamp: float,
    ) -> tuple[list[list[int]], float]:
        if is_prefill:
            raise ValueError("this profiler measures the decode control path")
        if self.ray_sequence_length > 0:
            if enable_rpc:
                raise ValueError("Ray Sequence transport must disable endpoint RPC")
            if len(dp_seqs) != len(self._token_rows):
                raise ValueError("Ray carried the wrong number of Sequences")
            if any(
                getattr(sequence, "num_tokens", None)
                != self.ray_sequence_length
                for sequence in dp_seqs
            ):
                raise ValueError("Ray carried a Sequence with the wrong length")
        else:
            if dp_seqs:
                raise ValueError("DLSlime Ray control RPC must carry empty dp_seqs")
            if not enable_rpc:
                raise ValueError("DLSlime-backed execution must enable endpoint RPC")
        if send_timestamp <= 0.0:
            raise ValueError("send_timestamp must be populated")
        return self._token_rows, time.time()


@dataclass(frozen=True)
class SampleStats:
    mean_ms: float
    stddev_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise ValueError("cannot calculate a percentile of no samples")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(samples_ms: Sequence[float]) -> SampleStats:
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    return SampleStats(
        mean_ms=statistics.fmean(samples_ms),
        stddev_ms=statistics.pstdev(samples_ms),
        min_ms=min(samples_ms),
        p50_ms=_percentile(samples_ms, 50.0),
        p95_ms=_percentile(samples_ms, 95.0),
        p99_ms=_percentile(samples_ms, 99.0),
        max_ms=max(samples_ms),
    )


def _sample_token_rows(
    logical_rank: int,
    batch_size_per_gpu: int,
    loop_count: int,
) -> list[list[int]]:
    return [
        [
            (logical_rank + row_index + step_index) % 32_000
            for step_index in range(loop_count)
        ]
        for row_index in range(batch_size_per_gpu)
    ]


def _sample_ray_sequence_batch(
    logical_rank: int,
    batch_size_per_gpu: int,
    sequence_length: int,
) -> list[NanoDeploySequence]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    token_ids = [
        (logical_rank + token_index) % 32_000
        for token_index in range(sequence_length)
    ]
    return [
        NanoDeploySequence(token_ids, 0.0, 16, True)
        for _ in range(batch_size_per_gpu)
    ]


def _estimated_pickle_bytes(
    batch_size_per_gpu: int,
    loop_count: int,
    ray_sequence_batch: list[NanoDeploySequence] | None = None,
) -> tuple[int, int]:
    input_args = (
        ray_sequence_batch if ray_sequence_batch is not None else [],
        False,
        ray_sequence_batch is None,
        0.0,
    )
    output = (_sample_token_rows(0, batch_size_per_gpu, loop_count), 0.0)
    return (
        len(pickle.dumps(input_args, protocol=pickle.HIGHEST_PROTOCOL)),
        len(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)),
    )


def _plan_actor_node_ids(
    nodes: Sequence[dict[str, object]],
    *,
    max_workers: int,
    workers_per_node: int | None,
) -> tuple[tuple[str | None, ...], list[dict[str, object]]]:
    if workers_per_node is None:
        return (None,) * max_workers, []

    alive_nodes = [node for node in nodes if node.get("Alive", False)]

    def node_sort_key(node: dict[str, object]) -> tuple[bool, str, str]:
        resources = node.get("Resources", {})
        is_head = bool(node.get("IsHeadNode", False)) or (
            isinstance(resources, dict)
            and "node:__internal_head__" in resources
        )
        return (
            not is_head,
            str(node.get("NodeManagerAddress", "")),
            str(node.get("NodeID", "")),
        )

    alive_nodes.sort(key=node_sort_key)
    required_nodes = math.ceil(max_workers / workers_per_node)
    if len(alive_nodes) < required_nodes:
        raise RuntimeError(
            f"need {required_nodes} live Ray nodes for {max_workers} workers at "
            f"{workers_per_node} workers/node, found {len(alive_nodes)}"
        )

    selected_nodes = alive_nodes[:required_nodes]
    selected_node_ids = [str(node.get("NodeID", "")) for node in selected_nodes]
    if any(not node_id for node_id in selected_node_ids):
        raise RuntimeError("one or more selected Ray nodes have no NodeID")

    assignments = tuple(
        selected_node_ids[rank // workers_per_node]
        for rank in range(max_workers)
    )
    selected_metadata = [
        {
            "node_id": str(node["NodeID"]),
            "hostname": str(node.get("NodeManagerHostname", "")),
            "address": str(node.get("NodeManagerAddress", "")),
            "is_head": index == 0 and not node_sort_key(node)[0],
        }
        for index, node in enumerate(selected_nodes)
    ]
    return assignments, selected_metadata


def _invoke_once(
    actors: Sequence[ray.actor.ActorHandle],
    ray_sequence_batches: Sequence[list[NanoDeploySequence]] | None,
) -> tuple[float, float, float, float, list[tuple[list[list[int]], float]]]:
    send_timestamp = time.time()
    begin_ns = time.perf_counter_ns()
    if ray_sequence_batches is None:
        refs = [
            actor.run.remote([], False, True, send_timestamp)
            for actor in actors
        ]
    else:
        if len(ray_sequence_batches) != len(actors):
            raise ValueError("each actor requires one Ray Sequence batch")
        refs = [
            actor.run.remote(sequences, False, False, send_timestamp)
            for actor, sequences in zip(
                actors,
                ray_sequence_batches,
                strict=True,
            )
        ]
    submit_end_ns = time.perf_counter_ns()
    results = ray.get(refs)
    end_ns = time.perf_counter_ns()
    receive_timestamp = time.time()

    submit_ms = (submit_end_ns - begin_ns) / 1_000_000.0
    ray_get_ms = (end_ns - submit_end_ns) / 1_000_000.0
    roundtrip_ms = (end_ns - begin_ns) / 1_000_000.0
    last_worker_finish = max(finished_at for _, finished_at in results)
    finish_to_get_ms = max(0.0, (receive_timestamp - last_worker_finish) * 1000.0)
    return submit_ms, ray_get_ms, roundtrip_ms, finish_to_get_ms, results


def _validate_results(
    results: Sequence[tuple[list[list[int]], float]],
    *,
    logical_workers: int,
    batch_size_per_gpu: int,
    loop_count: int,
) -> None:
    if len(results) != logical_workers:
        raise RuntimeError(
            f"received {len(results)} worker results, expected {logical_workers}"
        )
    for token_rows, finished_at in results:
        if len(token_rows) != batch_size_per_gpu:
            raise RuntimeError("worker returned the wrong number of token rows")
        if any(len(row) != loop_count for row in token_rows):
            raise RuntimeError("worker returned a token row with the wrong loop count")
        if finished_at <= 0.0:
            raise RuntimeError("worker omitted its completion timestamp")


def _profile_case(
    actors: Sequence[ray.actor.ActorHandle],
    *,
    logical_workers: int,
    batch_size_per_gpu: int,
    loop_count: int,
    ray_sequence_length: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, object]:
    active_actors = actors[:logical_workers]
    configured = ray.get(
        [
            actor.configure_batch.remote(batch_size_per_gpu)
            for actor in active_actors
        ]
    )
    if configured != [batch_size_per_gpu] * logical_workers:
        raise RuntimeError("one or more Ray actors rejected the configured batch size")

    ray_sequence_batches = None
    if ray_sequence_length > 0:
        ray_sequence_batches = [
            _sample_ray_sequence_batch(
                logical_rank=logical_rank,
                batch_size_per_gpu=batch_size_per_gpu,
                sequence_length=ray_sequence_length,
            )
            for logical_rank in range(logical_workers)
        ]

    for _ in range(warmup_iterations):
        *_, warmup_results = _invoke_once(
            active_actors,
            ray_sequence_batches,
        )
        _validate_results(
            warmup_results,
            logical_workers=logical_workers,
            batch_size_per_gpu=batch_size_per_gpu,
            loop_count=loop_count,
        )

    submit_samples_ms: list[float] = []
    ray_get_samples_ms: list[float] = []
    roundtrip_samples_ms: list[float] = []
    finish_to_get_samples_ms: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(measured_iterations):
            (
                submit_ms,
                ray_get_ms,
                roundtrip_ms,
                finish_to_get_ms,
                results,
            ) = _invoke_once(active_actors, ray_sequence_batches)
            submit_samples_ms.append(submit_ms)
            ray_get_samples_ms.append(ray_get_ms)
            roundtrip_samples_ms.append(roundtrip_ms)
            finish_to_get_samples_ms.append(finish_to_get_ms)
            _validate_results(
                results,
                logical_workers=logical_workers,
                batch_size_per_gpu=batch_size_per_gpu,
                loop_count=loop_count,
            )
            del results
    finally:
        if gc_was_enabled:
            gc.enable()

    input_bytes_per_worker, output_bytes_per_worker = _estimated_pickle_bytes(
        batch_size_per_gpu,
        loop_count,
        (
            ray_sequence_batches[0]
            if ray_sequence_batches is not None
            else None
        ),
    )
    submit = _stats(submit_samples_ms)
    ray_get_stats = _stats(ray_get_samples_ms)
    roundtrip = _stats(roundtrip_samples_ms)
    finish_to_get = _stats(finish_to_get_samples_ms)
    record: dict[str, object] = {
        "logical_workers": logical_workers,
        "batch_size_per_gpu": batch_size_per_gpu,
        "loop_count": loop_count,
        "ray_sequence_length": ray_sequence_length,
        "input_transport": (
            "ray_sequence" if ray_sequence_length > 0 else "dlslime_control"
        ),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "logical_input_token_ids": (
            logical_workers * batch_size_per_gpu * ray_sequence_length
        ),
        "logical_output_token_ids": (
            logical_workers * batch_size_per_gpu * loop_count
        ),
        "estimated_pickle_input_bytes_per_worker": input_bytes_per_worker,
        "estimated_pickle_input_bytes_total": (
            logical_workers * input_bytes_per_worker
        ),
        "estimated_pickle_output_bytes_per_worker": output_bytes_per_worker,
        "estimated_pickle_output_bytes_total": (
            logical_workers * output_bytes_per_worker
        ),
    }
    for prefix, values in (
        ("actor_submit", submit),
        ("ray_get", ray_get_stats),
        ("roundtrip", roundtrip),
        ("worker_finish_to_get", finish_to_get),
    ):
        for key, value in values.__dict__.items():
            record[f"{prefix}_{key}"] = value
    record["roundtrip_mean_ms_per_decode_step"] = roundtrip.mean_ms / loop_count
    record["roundtrip_p99_ms_per_decode_step"] = roundtrip.p99_ms / loop_count
    return record


def _write_results(
    output_dir: Path,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ray_rpc_overhead.json"
    csv_path = output_dir / "ray_rpc_overhead.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "records": records},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logical-workers",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128,256"),
        help="Comma-separated logical Ray worker counts.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128"),
        help="Comma-separated returned sequence counts per logical worker.",
    )
    parser.add_argument("--loop-count", type=int, default=16)
    parser.add_argument(
        "--ray-sequence-length",
        type=int,
        default=0,
        help=(
            "Full Sequence length sent through Ray per request; 0 preserves "
            "the production DLSlime control-only input path."
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--ray-address",
        default="local",
        help="Ray GCS address, or 'local' for a fresh single-node instance.",
    )
    parser.add_argument(
        "--workers-per-node",
        type=int,
        help=(
            "Hard-pin consecutive logical workers to Ray nodes in blocks of "
            "this size; intended for controlled multi-node measurements."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON/CSV results and the isolated Ray runtime.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.loop_count <= 0:
        parser.error("--loop-count must be positive")
    if args.ray_sequence_length < 0:
        parser.error("--ray-sequence-length must be non-negative")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.workers_per_node is not None and args.workers_per_node <= 0:
        parser.error("--workers-per-node must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir: Path | None = None
    ray_init_kwargs: dict[str, object] = {
        "address": args.ray_address,
        "_skip_env_hook": True,
        "logging_level": "ERROR",
    }
    if args.ray_address == "local":
        # Ray places Unix-domain sockets below its runtime directory. Keeping
        # this path short avoids Linux's 107-byte AF_UNIX pathname limit.
        ray_temp_dir = Path(tempfile.mkdtemp(prefix="nd-ray-", dir="/tmp"))
        ray_init_kwargs.update(
            {
                "include_dashboard": False,
                "num_cpus": min(os.cpu_count() or 1, max(args.logical_workers)),
                "_temp_dir": str(ray_temp_dir),
            }
        )

    if ray.is_initialized():
        raise RuntimeError("profiler requires a fresh isolated Ray instance")
    ray_context = ray.init(**ray_init_kwargs)
    records: list[dict[str, object]] = []
    actors: list[ray.actor.ActorHandle] = []
    actor_placements: list[dict[str, object]] = []
    selected_nodes: list[dict[str, object]] = []
    try:
        max_workers = max(args.logical_workers)
        actor_node_ids, selected_nodes = _plan_actor_node_ids(
            ray.nodes(),
            max_workers=max_workers,
            workers_per_node=args.workers_per_node,
        )
        for rank, node_id in enumerate(actor_node_ids):
            actor = (
                RayDecodeControlWorker.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id,
                        soft=False,
                    )
                ).remote(rank, args.loop_count, args.ray_sequence_length)
                if node_id is not None
                else RayDecodeControlWorker.remote(
                    rank,
                    args.loop_count,
                    args.ray_sequence_length,
                )
            )
            actors.append(actor)
        ready_ranks = ray.get([actor.ready.remote() for actor in actors])
        if ready_ranks != list(range(max_workers)):
            raise RuntimeError("Ray actors did not preserve logical rank ordering")
        actor_placements = ray.get([actor.placement.remote() for actor in actors])
        for placement, expected_node_id in zip(actor_placements, actor_node_ids):
            if (
                expected_node_id is not None
                and placement["node_id"] != expected_node_id
            ):
                raise RuntimeError(
                    f"worker {placement['logical_rank']} ran on node "
                    f"{placement['node_id']}, expected {expected_node_id}"
                )

        total_cases = len(args.batch_sizes) * len(args.logical_workers)
        case_index = 0
        for batch_size_per_gpu in args.batch_sizes:
            for logical_workers in args.logical_workers:
                case_index += 1
                print(
                    f"[{case_index}/{total_cases}] workers={logical_workers} "
                    f"bs_per_gpu={batch_size_per_gpu} loop_count={args.loop_count}",
                    flush=True,
                )
                record = _profile_case(
                    actors,
                    logical_workers=logical_workers,
                    batch_size_per_gpu=batch_size_per_gpu,
                    loop_count=args.loop_count,
                    ray_sequence_length=args.ray_sequence_length,
                    warmup_iterations=args.warmup_iterations,
                    measured_iterations=args.iterations,
                )
                records.append(record)
                print(
                    f"  submit mean={record['actor_submit_mean_ms']:.3f} ms | "
                    f"ray.get mean={record['ray_get_mean_ms']:.3f} ms | "
                    f"roundtrip mean={record['roundtrip_mean_ms']:.3f} ms "
                    f"p99={record['roundtrip_p99_ms']:.3f} ms",
                    flush=True,
                )
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        ray.shutdown()

    metadata = {
        "benchmark": "nanodeploy-ray-rpc-scalability",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ray_version": ray.__version__,
        "ray_address": ray_context.address_info.get("gcs_address", "local"),
        "ray_temp_dir": str(ray_temp_dir) if ray_temp_dir is not None else None,
        "cpu_count": os.cpu_count(),
        "selected_nodes": selected_nodes,
        "actor_placements": actor_placements,
        "timed_scope": (
            "Ray actor run.remote submission, optional full Sequence input "
            "transfer, and ray.get of decode token results"
        ),
        "excluded_scope": (
            "scheduler, DLSlime/RDMA, ModelRunner, GPU kernels, actor creation, "
            "batch payload construction, and result validation"
        ),
        "topology_limit": (
            "logical workers are colocated in one isolated Ray instance"
            if args.ray_address == "local"
            else (
                "actors use hard node affinity, but this measures Ray control "
                "RPC traffic rather than DLSlime/RDMA data-plane bandwidth"
            )
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_results(args.output_dir, metadata, records)
    print(f"Wrote {args.output_dir / 'ray_rpc_overhead.json'}")
    print(f"Wrote {args.output_dir / 'ray_rpc_overhead.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

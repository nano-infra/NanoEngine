#!/usr/bin/env python3
"""Compare full-Sequence Ray transport with NanoDeploy's DLSLime path.

Both modes use real Ray actors and the same NanoDeploy ``Sequence`` objects.
The Ray mode passes each worker's full Sequence batch as a Ray actor argument.
The DLSLime mode submits the production-shaped empty Ray control command and
sends the same Sequence batch through ``RPCServerEndpoint.send_seqs``; actors
receive it through ``RPCClientEndpoint.recv_seqs``.  Both modes return the same
``BS/GPU * loop_count`` token IDs through Ray.

Sequence/output construction and endpoint initialization are outside the
timed interval.  ModelRunner, GPU kernels, and scheduler work are not run.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TypedDict

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from nanodeploy._cpp import Sequence as NanoDeploySequence
from nanodeploy._cpp import serialize
from nanodeploy.endpoint import rpc_endpoint as rpc_endpoint_module
from nanodeploy.endpoint.rpc_endpoint import (
    RPCClientEndpoint,
    RPCServerEndpoint,
)
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


TRANSPORTS = ("ray", "dlslime")
REQUIRED_SLIME_ENV = (
    "SLIME_VISIBLE_DEVICES",
    "SLIME_GID_INDEX",
    "SLIME_QP_NUM",
)


class DLSLimeSendProfile(TypedDict):
    total_ms: float
    serialize_ms: float
    write_with_imm_ms: float
    future_wait_ms: float
    unattributed_ms: float
    future_wait_ms_by_rank: list[float]
    total_bytes: int


def _encode_transport_imm(payload_bytes: int, transport_slot: int) -> int:
    encode_imm = getattr(rpc_endpoint_module, "_encode_imm", None)
    if encode_imm is None:
        if transport_slot != 0:
            raise RuntimeError("legacy RPC endpoint only supports transport slot 0")
        return payload_bytes
    return encode_imm(payload_bytes, transport_slot)


@ray.remote(num_cpus=0, num_gpus=1)
class RayVsDLSLimeWorker:
    def __init__(
        self,
        logical_rank: int,
        loop_count: int,
        sequence_length: int,
        dlslime_buffer_bytes: int,
    ) -> None:
        self.logical_rank = logical_rank
        self.loop_count = loop_count
        self.sequence_length = sequence_length
        self.endpoint = RPCClientEndpoint(
            dlslime_buffer_bytes,
            logical_rank,
        )
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

    def init_dlslime_endpoint(self, server_info: list[object]) -> object:
        client_info = self.endpoint.init_client_endpoint()
        self.endpoint.connect(server_info)
        return client_info

    def placement(self) -> dict[str, object]:
        return {
            "logical_rank": self.logical_rank,
            "hostname": socket.gethostname(),
            "node_id": str(ray.get_runtime_context().get_node_id()),
        }

    def _validate_sequences(self, sequences: list[object]) -> None:
        if len(sequences) != len(self._token_rows):
            raise ValueError("worker received the wrong Sequence count")
        if any(
            getattr(sequence, "num_tokens", None) != self.sequence_length
            for sequence in sequences
        ):
            raise ValueError("worker received a Sequence with the wrong length")

    def run(
        self,
        dp_seqs: list[object],
        is_prefill: bool,
        enable_rpc: bool,
        send_timestamp: float,
    ) -> tuple[list[list[int]], float]:
        if is_prefill:
            raise ValueError("comparison profiler measures the decode-shaped path")
        if send_timestamp <= 0.0:
            raise ValueError("send_timestamp must be populated")
        if enable_rpc:
            if dp_seqs:
                raise ValueError("DLSLime control command must carry empty dp_seqs")
            sequences = self.endpoint.recv_seqs()
        else:
            sequences = dp_seqs
        self._validate_sequences(sequences)
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
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_transports(value: str) -> tuple[str, ...]:
    transports = tuple(item.strip() for item in value.split(","))
    if not transports or any(item not in TRANSPORTS for item in transports):
        raise argparse.ArgumentTypeError(
            f"transports must be comma-separated values from {TRANSPORTS}"
        )
    if len(set(transports)) != len(transports):
        raise argparse.ArgumentTypeError("transports must not contain duplicates")
    return transports


def _percentile(samples: Sequence[float], percentile: float) -> float:
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


def _sample_sequence_batch(
    logical_rank: int,
    batch_size_per_gpu: int,
    sequence_length: int,
) -> list[NanoDeploySequence]:
    token_ids = [
        (logical_rank + token_index) % 32_000
        for token_index in range(sequence_length)
    ]
    return [
        NanoDeploySequence(token_ids, 0.0, 16, True)
        for _ in range(batch_size_per_gpu)
    ]


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


def _plan_actor_node_ids(
    nodes: Sequence[dict[str, object]],
    *,
    logical_workers: int,
    workers_per_node: int,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
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
    required_nodes = math.ceil(logical_workers / workers_per_node)
    if len(alive_nodes) < required_nodes:
        raise RuntimeError(
            f"need {required_nodes} live Ray nodes, found {len(alive_nodes)}"
        )
    selected_nodes = alive_nodes[:required_nodes]
    node_ids = [str(node.get("NodeID", "")) for node in selected_nodes]
    if any(not node_id for node_id in node_ids):
        raise RuntimeError("one or more selected Ray nodes have no NodeID")
    assignments = tuple(
        node_ids[rank // workers_per_node]
        for rank in range(logical_workers)
    )
    metadata = [
        {
            "node_id": str(node["NodeID"]),
            "hostname": str(node.get("NodeManagerHostname", "")),
            "address": str(node.get("NodeManagerAddress", "")),
        }
        for node in selected_nodes
    ]
    return assignments, metadata


def _profiled_send_seqs(
    endpoint: RPCServerEndpoint,
    dp_seqs: Sequence[list[NanoDeploySequence]],
    *,
    is_prefill: bool,
    transport_slot: int = 0,
) -> DLSLimeSendProfile:
    """Run the production send loop with phase-level timing enabled."""
    start_ns = time.perf_counter_ns()
    if len(dp_seqs) != endpoint.world_size:
        raise ValueError("DLSLime sequence batch count does not match world size")

    num_slots = getattr(endpoint, "num_slots", 1)
    slot_size = getattr(endpoint, "slot_size", endpoint.buffer_size)
    if not 0 <= transport_slot < num_slots:
        raise ValueError(f"invalid RPC transport slot {transport_slot}")

    futures: list[Any] = []
    total_bytes = 0
    serialize_ns = 0
    write_with_imm_ns = 0
    slot_offset = transport_slot * slot_size
    for rank in range(endpoint.world_size):
        binding = endpoint.server_bindings[rank]
        buffer = binding.buffer
        buffer_ptr = (
            buffer.data_ptr() + buffer.storage_offset() + slot_offset
        )
        if not is_prefill and endpoint.optimize_decode_block_table:
            sp_rank = (rank // endpoint.attention_tp) % endpoint.attention_sp
            sp_size = endpoint.attention_sp
        else:
            sp_rank = -1
            sp_size = -1

        phase_start_ns = time.perf_counter_ns()
        payload_bytes = serialize(
            buffer_ptr,
            slot_size,
            dp_seqs[rank],
            is_prefill,
            sp_rank,
            sp_size,
        )
        serialize_ns += time.perf_counter_ns() - phase_start_ns
        total_bytes += payload_bytes

        phase_start_ns = time.perf_counter_ns()
        future = binding.endpoint.write_with_imm(
            [
                (
                    buffer_ptr,
                    binding.remote_buffer_ptr + slot_offset,
                    0,
                    0,
                    payload_bytes,
                )
            ],
            _encode_transport_imm(payload_bytes, transport_slot),
        )
        write_with_imm_ns += time.perf_counter_ns() - phase_start_ns
        futures.append(future)

    future_wait_ns_by_rank: list[int] = []
    wait_phase_start_ns = time.perf_counter_ns()
    for future in futures:
        phase_start_ns = time.perf_counter_ns()
        future.wait()
        future_wait_ns_by_rank.append(
            time.perf_counter_ns() - phase_start_ns
        )
    future_wait_ns = time.perf_counter_ns() - wait_phase_start_ns
    total_ns = time.perf_counter_ns() - start_ns
    attributed_ns = serialize_ns + write_with_imm_ns + future_wait_ns
    return {
        "total_ms": total_ns / 1_000_000,
        "serialize_ms": serialize_ns / 1_000_000,
        "write_with_imm_ms": write_with_imm_ns / 1_000_000,
        "future_wait_ms": future_wait_ns / 1_000_000,
        "unattributed_ms": max(0, total_ns - attributed_ns) / 1_000_000,
        "future_wait_ms_by_rank": [
            duration_ns / 1_000_000
            for duration_ns in future_wait_ns_by_rank
        ],
        "total_bytes": total_bytes,
    }


def _invoke_once(
    actors: Sequence[Any],
    sequence_batches: Sequence[list[NanoDeploySequence]],
    *,
    transport: str,
    dlslime_endpoint: RPCServerEndpoint | None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    DLSLimeSendProfile,
    list[tuple[list[list[int]], float]],
]:
    send_timestamp = time.time()
    begin_ns = time.perf_counter_ns()
    if transport == "ray":
        refs = [
            actor.run.remote(sequences, False, False, send_timestamp)
            for actor, sequences in zip(actors, sequence_batches, strict=True)
        ]
    elif transport == "dlslime":
        refs = [
            actor.run.remote([], False, True, send_timestamp)
            for actor in actors
        ]
    else:
        raise ValueError(f"unsupported transport: {transport}")
    submit_end_ns = time.perf_counter_ns()

    send_seqs_ms = 0.0
    send_profile: DLSLimeSendProfile = {
        "total_ms": 0.0,
        "serialize_ms": 0.0,
        "write_with_imm_ms": 0.0,
        "future_wait_ms": 0.0,
        "unattributed_ms": 0.0,
        "future_wait_ms_by_rank": [0.0] * len(actors),
        "total_bytes": 0,
    }
    if transport == "dlslime":
        if dlslime_endpoint is None:
            raise RuntimeError("DLSLime transport requires a connected endpoint")
        send_begin_ns = time.perf_counter_ns()
        observed_profile = _profiled_send_seqs(
            dlslime_endpoint,
            list(sequence_batches),
            is_prefill=False,
        )
        send_seqs_ms = (
            time.perf_counter_ns() - send_begin_ns
        ) / 1_000_000.0
        if len(observed_profile["future_wait_ms_by_rank"]) != len(actors):
            raise RuntimeError("DLSLime endpoint returned the wrong rank count")
        send_profile = observed_profile

    ray_get_begin_ns = time.perf_counter_ns()
    results = ray.get(refs)
    end_ns = time.perf_counter_ns()
    receive_timestamp = time.time()

    submit_ms = (submit_end_ns - begin_ns) / 1_000_000.0
    ray_get_ms = (end_ns - ray_get_begin_ns) / 1_000_000.0
    roundtrip_ms = (end_ns - begin_ns) / 1_000_000.0
    last_worker_finish = max(finished_at for _, finished_at in results)
    finish_to_get_ms = max(
        0.0,
        (receive_timestamp - last_worker_finish) * 1000.0,
    )
    return (
        submit_ms,
        send_seqs_ms,
        ray_get_ms,
        roundtrip_ms,
        finish_to_get_ms,
        send_profile,
        results,
    )


def _validate_results(
    results: Sequence[tuple[list[list[int]], float]],
    *,
    logical_workers: int,
    batch_size_per_gpu: int,
    loop_count: int,
) -> None:
    if len(results) != logical_workers:
        raise RuntimeError("worker result count mismatch")
    for token_rows, finished_at in results:
        if len(token_rows) != batch_size_per_gpu:
            raise RuntimeError("worker returned the wrong token row count")
        if any(len(row) != loop_count for row in token_rows):
            raise RuntimeError("worker returned a token row with the wrong length")
        if finished_at <= 0.0:
            raise RuntimeError("worker omitted its completion timestamp")


def _native_dlslime_bytes(
    endpoint: RPCServerEndpoint,
    sequence_batches: Sequence[list[NanoDeploySequence]],
) -> int:
    total_bytes = 0
    for binding, sequences in zip(
        endpoint.server_bindings,
        sequence_batches,
        strict=True,
    ):
        buffer = binding.buffer
        buffer_ptr = buffer.data_ptr() + buffer.storage_offset()
        total_bytes += serialize(
            buffer_ptr,
            buffer.numel(),
            sequences,
            False,
            0,
            1,
        )
    return total_bytes


def _profile_transport(
    actors: Sequence[Any],
    sequence_batches: Sequence[list[NanoDeploySequence]],
    *,
    transport: str,
    dlslime_endpoint: RPCServerEndpoint | None,
    logical_workers: int,
    batch_size_per_gpu: int,
    sequence_length: int,
    loop_count: int,
    warmup_iterations: int,
    measured_iterations: int,
    ray_input_bytes_per_worker: int,
    dlslime_input_bytes_total: int,
    output_bytes_per_worker: int,
) -> dict[str, object]:
    configured = ray.get(
        [
            actor.configure_batch.remote(batch_size_per_gpu)
            for actor in actors
        ]
    )
    if configured != [batch_size_per_gpu] * logical_workers:
        raise RuntimeError("one or more actors rejected the batch size")

    for _ in range(warmup_iterations):
        *_, results = _invoke_once(
            actors,
            sequence_batches,
            transport=transport,
            dlslime_endpoint=dlslime_endpoint,
        )
        _validate_results(
            results,
            logical_workers=logical_workers,
            batch_size_per_gpu=batch_size_per_gpu,
            loop_count=loop_count,
        )

    samples: dict[str, list[float]] = {
        "actor_submit": [],
        "dlslime_send_seqs": [],
        "dlslime_endpoint_total": [],
        "dlslime_serialize": [],
        "dlslime_write_with_imm": [],
        "dlslime_future_wait": [],
        "dlslime_unattributed": [],
        "ray_get": [],
        "roundtrip": [],
        "worker_finish_to_get": [],
    }
    future_wait_samples_by_rank: list[list[float]] = [
        [] for _ in range(logical_workers)
    ]
    slowest_wait_rank_counts = [0] * logical_workers
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(measured_iterations):
            (
                submit_ms,
                send_seqs_ms,
                ray_get_ms,
                roundtrip_ms,
                finish_to_get_ms,
                send_profile,
                results,
            ) = _invoke_once(
                actors,
                sequence_batches,
                transport=transport,
                dlslime_endpoint=dlslime_endpoint,
            )
            samples["actor_submit"].append(submit_ms)
            samples["dlslime_send_seqs"].append(send_seqs_ms)
            samples["dlslime_endpoint_total"].append(
                send_profile["total_ms"]
            )
            samples["dlslime_serialize"].append(
                send_profile["serialize_ms"]
            )
            samples["dlslime_write_with_imm"].append(
                send_profile["write_with_imm_ms"]
            )
            samples["dlslime_future_wait"].append(
                send_profile["future_wait_ms"]
            )
            samples["dlslime_unattributed"].append(
                send_profile["unattributed_ms"]
            )
            wait_ms_by_rank = send_profile["future_wait_ms_by_rank"]
            for rank, wait_ms in enumerate(wait_ms_by_rank):
                future_wait_samples_by_rank[rank].append(wait_ms)
            if transport == "dlslime":
                slowest_wait_rank = max(
                    range(logical_workers),
                    key=wait_ms_by_rank.__getitem__,
                )
                slowest_wait_rank_counts[slowest_wait_rank] += 1
            samples["ray_get"].append(ray_get_ms)
            samples["roundtrip"].append(roundtrip_ms)
            samples["worker_finish_to_get"].append(finish_to_get_ms)
            _validate_results(
                results,
                logical_workers=logical_workers,
                batch_size_per_gpu=batch_size_per_gpu,
                loop_count=loop_count,
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    record: dict[str, object] = {
        "transport": transport,
        "logical_workers": logical_workers,
        "batch_size_per_gpu": batch_size_per_gpu,
        "sequence_length": sequence_length,
        "loop_count": loop_count,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "logical_input_token_ids": (
            logical_workers * batch_size_per_gpu * sequence_length
        ),
        "ray_input_pickle_bytes_per_worker": (
            ray_input_bytes_per_worker if transport == "ray" else 0
        ),
        "ray_input_pickle_bytes_total": (
            logical_workers * ray_input_bytes_per_worker
            if transport == "ray"
            else 0
        ),
        "dlslime_input_bytes_total": (
            dlslime_input_bytes_total if transport == "dlslime" else 0
        ),
        "ray_output_pickle_bytes_per_worker": output_bytes_per_worker,
        "ray_output_pickle_bytes_total": (
            logical_workers * output_bytes_per_worker
        ),
    }
    for prefix, values in samples.items():
        stats = _stats(values)
        for key, value in stats.__dict__.items():
            record[f"{prefix}_{key}"] = value
    record["dlslime_future_wait_mean_ms_by_rank"] = [
        statistics.fmean(values)
        for values in future_wait_samples_by_rank
    ]
    record["dlslime_future_wait_p99_ms_by_rank"] = [
        _percentile(values, 99.0)
        for values in future_wait_samples_by_rank
    ]
    record["dlslime_slowest_wait_rank_counts"] = slowest_wait_rank_counts
    record["roundtrip_mean_ms_per_decode_step"] = (
        record["roundtrip_mean_ms"] / loop_count
    )
    record["roundtrip_p99_ms_per_decode_step"] = (
        record["roundtrip_p99_ms"] / loop_count
    )
    return record


def _write_results(
    output_dir: Path,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ray_vs_dlslime.json"
    csv_path = output_dir / "ray_vs_dlslime.csv"
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
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--logical-workers", type=int, default=16)
    parser.add_argument("--workers-per-node", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128"),
    )
    parser.add_argument("--sequence-length", type=int, default=8_000)
    parser.add_argument("--loop-count", type=int, default=16)
    parser.add_argument(
        "--transports",
        type=_parse_transports,
        default=_parse_transports("ray,dlslime"),
    )
    parser.add_argument("--dlslime-buffer-bytes", type=int, default=32 << 20)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "logical_workers",
        "workers_per_node",
        "sequence_length",
        "loop_count",
        "dlslime_buffer_bytes",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if "dlslime" in args.transports:
        missing = [name for name in REQUIRED_SLIME_ENV if not os.getenv(name)]
        if missing:
            parser.error(
                "DLSLime mode requires environment variables: "
                + ", ".join(missing)
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if ray.is_initialized():
        raise RuntimeError("profiler requires a fresh Ray driver")

    ray_context = ray.init(
        address=args.ray_address,
        _skip_env_hook=True,
        logging_level="ERROR",
    )
    actors: list[Any] = []
    server_endpoint: RPCServerEndpoint | None = None
    records: list[dict[str, object]] = []
    placements: list[dict[str, object]] = []
    selected_nodes: list[dict[str, object]] = []
    try:
        node_ids, selected_nodes = _plan_actor_node_ids(
            ray.nodes(),
            logical_workers=args.logical_workers,
            workers_per_node=args.workers_per_node,
        )
        for rank, node_id in enumerate(node_ids):
            actor = RayVsDLSLimeWorker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id,
                    soft=False,
                )
            ).remote(
                rank,
                args.loop_count,
                args.sequence_length,
                args.dlslime_buffer_bytes,
            )
            actors.append(actor)

        placements = ray.get([actor.placement.remote() for actor in actors])
        for placement, expected_node_id in zip(placements, node_ids, strict=True):
            if placement["node_id"] != expected_node_id:
                raise RuntimeError(
                    f"worker {placement['logical_rank']} placement mismatch"
                )

        total_cases = len(args.batch_sizes) * len(args.transports)
        case_index = 0
        # Keep each transport phase contiguous. In particular, do not perturb
        # a live DLSLime connection with a large Ray Sequence-transfer case
        # between two DLSLime batch sizes.
        for transport in args.transports:
            if transport == "dlslime":
                server_endpoint = RPCServerEndpoint(
                    args.dlslime_buffer_bytes,
                    args.logical_workers,
                    attention_sp=1,
                    attention_tp=1,
                    optimize_decode_block_table=True,
                )
                server_info = server_endpoint.init_server_endpoint()
                client_info = ray.get(
                    [
                        actor.init_dlslime_endpoint.remote(server_info)
                        for actor in actors
                    ]
                )
                server_endpoint.connect(client_info)

            for batch_size_per_gpu in args.batch_sizes:
                case_index += 1
                sequence_batches = [
                    _sample_sequence_batch(
                        logical_rank=rank,
                        batch_size_per_gpu=batch_size_per_gpu,
                        sequence_length=args.sequence_length,
                    )
                    for rank in range(args.logical_workers)
                ]
                ray_input_bytes_per_worker = len(
                    pickle.dumps(
                        (sequence_batches[0], False, False, 0.0),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
                output_bytes_per_worker = len(
                    pickle.dumps(
                        (
                            _sample_token_rows(
                                0,
                                batch_size_per_gpu,
                                args.loop_count,
                            ),
                            0.0,
                        ),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
                dlslime_input_bytes_total = (
                    _native_dlslime_bytes(
                        server_endpoint,
                        sequence_batches,
                    )
                    if transport == "dlslime" and server_endpoint is not None
                    else 0
                )
                print(
                    f"[{case_index}/{total_cases}] transport={transport} "
                    f"workers={args.logical_workers} "
                    f"bs_per_gpu={batch_size_per_gpu} "
                    f"sequence_length={args.sequence_length}",
                    flush=True,
                )
                record = _profile_transport(
                    actors,
                    sequence_batches,
                    transport=transport,
                    dlslime_endpoint=server_endpoint,
                    logical_workers=args.logical_workers,
                    batch_size_per_gpu=batch_size_per_gpu,
                    sequence_length=args.sequence_length,
                    loop_count=args.loop_count,
                    warmup_iterations=args.warmup_iterations,
                    measured_iterations=args.iterations,
                    ray_input_bytes_per_worker=ray_input_bytes_per_worker,
                    dlslime_input_bytes_total=dlslime_input_bytes_total,
                    output_bytes_per_worker=output_bytes_per_worker,
                )
                records.append(record)
                print(
                    f"  submit={record['actor_submit_mean_ms']:.3f} ms | "
                    f"send_seqs={record['dlslime_send_seqs_mean_ms']:.3f} ms | "
                    f"ray.get={record['ray_get_mean_ms']:.3f} ms | "
                    f"roundtrip={record['roundtrip_mean_ms']:.3f} ms "
                    f"p99={record['roundtrip_p99_ms']:.3f} ms",
                    flush=True,
                )
                if transport == "dlslime":
                    print(
                        "  send_seqs breakdown: "
                        f"serialize={record['dlslime_serialize_mean_ms']:.3f} ms | "
                        "write_with_imm="
                        f"{record['dlslime_write_with_imm_mean_ms']:.3f} ms | "
                        f"future.wait={record['dlslime_future_wait_mean_ms']:.3f} ms | "
                        f"unattributed={record['dlslime_unattributed_mean_ms']:.3f} ms",
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
        "benchmark": "nanodeploy-ray-vs-dlslime-sequence-transport",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "driver_hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ray_version": ray.__version__,
        "ray_address": ray_context.address_info.get(
            "gcs_address",
            args.ray_address,
        ),
        "selected_nodes": selected_nodes,
        "actor_placements": placements,
        "slime_environment": {
            name: os.environ[name] for name in REQUIRED_SLIME_ENV
        },
        "timed_scope": (
            "Ray actor submission; full Sequence input via Ray or production "
            "NanoDeploy serialize + DLSLime send_seqs/recv_seqs; token output "
            "via Ray; ray.get fan-in; DLSLime serialize, write_with_imm, and "
            "future.wait breakdown"
        ),
        "excluded_scope": (
            "Sequence/output construction, endpoint initialization, "
            "scheduler, ModelRunner, and GPU kernels"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_results(args.output_dir, metadata, records)
    print(f"Wrote {args.output_dir / 'ray_vs_dlslime.json'}")
    print(f"Wrote {args.output_dir / 'ray_vs_dlslime.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

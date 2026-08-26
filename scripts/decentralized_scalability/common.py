from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from nanodeploy.engine.hierarchical_contract import LoadSnapshot, RankLoad
from nanodeploy.router.admission_planner import AdmissionPlannerConfig


PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return parsed


def parse_scaling_modes(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"strong", "weak"}
    if not parsed or any(item not in allowed for item in parsed):
        raise argparse.ArgumentTypeError(
            "scaling modes must be comma-separated values from: strong, weak"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "scaling modes must not contain duplicates"
        )
    return parsed


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("summary requires at least one value")
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "stddev": statistics.pstdev(samples),
        "min": min(samples),
        "p50": percentile(samples, 50.0),
        "p95": percentile(samples, 95.0),
        "p99": percentile(samples, 99.0),
        "max": max(samples),
    }


def sp_load_snapshot(
    engine_id: int,
    *,
    attention_sp: int,
    capacity_requests: int,
    running: int = 0,
    waiting: int = 0,
    ingress_version: int = 0,
    admission_version: int = 0,
    quantum_id: int = 0,
) -> LoadSnapshot:
    if engine_id < 0:
        raise ValueError("engine_id must be non-negative")
    if attention_sp <= 0 or capacity_requests <= 0:
        raise ValueError("attention_sp and capacity_requests must be positive")
    free_blocks = max(1_000_000, capacity_requests * 256)
    base_running, extra_running = divmod(running, attention_sp)
    rank_loads = tuple(
        RankLoad(
            global_rank=engine_id * attention_sp + sp_idx,
            sp_idx=sp_idx,
            tp_idx=0,
            master_batch_size=base_running + (sp_idx < extra_running),
            active_master_requests=base_running + (sp_idx < extra_running),
            free_blocks=free_blocks,
            total_blocks=free_blocks,
            master_assignments=base_running + (sp_idx < extra_running),
            mastered_decode_tokens=0,
            control_dummy_blocks=1,
        )
        for sp_idx in range(attention_sp)
    )
    return LoadSnapshot(
        engine_id=engine_id,
        ready=True,
        waiting=waiting,
        running=running,
        free_blocks_min=free_blocks,
        wave_id=1,
        quantum_id=quantum_id,
        admission_version=admission_version,
        ingress_version=ingress_version,
        rank_loads=rank_loads,
    )


def fixed_sp_planner_config(
    *,
    attention_sp: int,
    capacity_requests: int,
    prompt_tokens: int,
) -> AdmissionPlannerConfig:
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    return AdmissionPlannerConfig(
        attention_sp=attention_sp,
        kvcache_block_size=64,
        max_num_seqs=capacity_requests + 1,
        max_num_batched_tokens=capacity_requests * prompt_tokens + 1,
        max_num_recv_seqs=capacity_requests + 1,
        reserved_blocks_per_req=0.0,
        segment_size=64,
        queue_capacity=capacity_requests + 1,
        fixed_sp_size=attention_sp,
    )


def requests_for_case(
    *,
    scaling_mode: str,
    engines: int,
    strong_total_requests: int,
    requests_per_engine: int,
) -> int:
    if scaling_mode == "strong":
        return strong_total_requests
    if scaling_mode == "weak":
        return engines * requests_per_engine
    raise ValueError(f"unknown scaling mode {scaling_mode!r}")


def clear_proxy_environment() -> tuple[str, ...]:
    removed = tuple(name for name in PROXY_ENV_NAMES if name in os.environ)
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    return removed


def select_cluster_nodes(
    nodes: Iterable[dict[str, Any]],
    *,
    count: int,
    requested_node_ips: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    if count <= 0:
        raise ValueError("node count must be positive")
    alive = [
        node
        for node in nodes
        if node.get("Alive", False)
        and float(node.get("Resources", {}).get("CPU", 0.0)) >= 1.0
    ]
    if requested_node_ips:
        requested = tuple(dict.fromkeys(requested_node_ips))
        if len(requested) < count:
            raise ValueError(
                "not enough unique --node-ip values for the largest node count"
            )
        by_ip = {node.get("NodeManagerAddress"): node for node in alive}
        missing = [node_ip for node_ip in requested[:count] if node_ip not in by_ip]
        if missing:
            raise RuntimeError(
                "requested Ray nodes are not alive or have no CPU resource: "
                f"{missing}"
            )
        return tuple(by_ip[node_ip] for node_ip in requested[:count])
    ordered = sorted(
        alive,
        key=lambda node: (
            str(node.get("NodeManagerAddress", "")),
            str(node.get("NodeID", "")),
        ),
    )
    if len(ordered) < count:
        raise RuntimeError(
            "insufficient alive Ray CPU nodes: "
            f"need={count}, found={len(ordered)}"
        )
    return tuple(ordered[:count])


def node_metadata(nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node["NodeID"]),
            "node_ip": str(node.get("NodeManagerAddress", "")),
            "cpu_resources": float(node.get("Resources", {}).get("CPU", 0.0)),
            "gpu_resources_visible_to_ray": float(
                node.get("Resources", {}).get("GPU", 0.0)
            ),
        }
        for node in nodes
    ]


def base_metadata(benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_results(
    output_dir: Path,
    *,
    stem: str,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    if not records:
        raise ValueError("cannot write an empty benchmark result")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "records": records},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    fieldnames = list(records[0])
    if any(list(record) != fieldnames for record in records):
        raise ValueError("all CSV records must have identical ordered fields")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: _csv_value(value)
                for key, value in record.items()
            }
            for record in records
        )
    return json_path, csv_path

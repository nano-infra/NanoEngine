#!/usr/bin/env python3
"""Parse successful cases from one Figure 12 vLLM run."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from e2e_config import FIG12_DIR, WORKLOADS


RESULT_BASE = FIG12_DIR / "results" / "e2e" / "vllm"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}$")
BASELINE_ORDER = (
    "dp_least_batch",
    "dp_least_cache",
    "cp2",
    "cp4",
    "cp8",
)
BASELINE_LABELS = {
    "dp_least_batch": "DP-LB",
    "dp_least_cache": "DP-LC",
    "cp2": "CP2",
    "cp4": "CP4",
    "cp8": "CP8",
}
WORKLOAD_BY_SLUG = {workload.slug: workload for workload in WORKLOADS}

try:
    import orjson

    def parse_json_bytes(raw: bytes) -> dict[str, Any]:
        return orjson.loads(raw)

except ImportError:

    def parse_json_bytes(raw: bytes) -> dict[str, Any]:
        return json.loads(raw)


def run_id_from_case_csv(path: Path, nodes: int) -> str | None:
    prefix = f"fig12-{nodes}node-"
    suffix = "_cases.csv"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def case_csv_path(node_root: Path, nodes: int, run_id: str) -> Path:
    return node_root / "_case_csv" / f"fig12-{nodes}node-{run_id}_cases.csv"


def run_start(path: Path, run_id: str) -> dt.datetime:
    if RUN_ID_PATTERN.fullmatch(run_id):
        parsed = dt.datetime.strptime(run_id, "%Y%m%dT%H%M%S")
        return parsed.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)


def next_run_start(
    node_root: Path,
    nodes: int,
    current_csv: Path,
    current_start: dt.datetime,
) -> dt.datetime | None:
    starts: list[dt.datetime] = []
    for path in (node_root / "_case_csv").glob(f"fig12-{nodes}node-*_cases.csv"):
        if path == current_csv:
            continue
        run_id = run_id_from_case_csv(path, nodes)
        if run_id is None:
            continue
        candidate = run_start(path, run_id)
        if candidate > current_start:
            starts.append(candidate)
    return min(starts, default=None)


def workload_and_baseline(case_name: str, nodes: int) -> tuple[str, str] | None:
    for slug in WORKLOAD_BY_SLUG:
        prefix = f"fig12_{nodes}node_{slug}_"
        if not case_name.startswith(prefix):
            continue
        suffix = case_name[len(prefix) :]
        baseline, separator, _ = suffix.rpartition("_rate")
        if separator and baseline in BASELINE_LABELS:
            return slug, baseline
    return None


def load_case_rows(
    path: Path,
    nodes: int,
    requested_workloads: list[str],
) -> list[dict[str, Any]]:
    requested = set(requested_workloads)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("enabled") or "").strip() not in {"1", "true", "True"}:
                continue
            parsed = workload_and_baseline(row.get("name", ""), nodes)
            if parsed is None:
                continue
            workload, baseline = parsed
            if requested and workload not in requested:
                continue
            try:
                row["request_rate"] = float(row["request_rate"])
                row["max_requests"] = int(row["max_requests"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid case row in {path}: {row}") from error
            row["workload"] = workload
            row["baseline"] = baseline
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no selected Figure 12 cases in {path}")
    return rows


def parse_manifest_time(value: object, path: Path) -> dt.datetime:
    if isinstance(value, str) and value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    try:
        parsed = dt.datetime.strptime(path.parent.name, "%Y%m%d-%H%M%S")
    except ValueError as error:
        raise ValueError(f"manifest has no usable start time: {path}") from error
    return parsed.replace(tzinfo=dt.timezone.utc)


def completed_benchmark_outputs(case_dir: Path, expected_requests: int) -> bool:
    """Accept a fully written benchmark even if Ctrl-C interrupted cleanup."""
    benchmark_dir = case_dir / "benchmark"
    summary_path = benchmark_dir / "summary.json"
    requests_path = benchmark_dir / "requests.jsonl"
    if not summary_path.is_file() or not requests_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        successful = int(summary["successful_requests"])
        failed = int(summary["failed_requests"])
        total = int(summary["total_requests"])
        written = sum(
            1 for line in requests_path.read_bytes().splitlines() if line.strip()
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        total == expected_requests and successful + failed == total and written == total
    )


def successful_manifests(
    node_root: Path,
    case_rows: list[dict[str, Any]],
    start: dt.datetime,
    end: dt.datetime | None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    rows_by_name = {str(row["name"]): row for row in case_rows}
    expected_names = set(rows_by_name)
    candidates: dict[str, list[tuple[dt.datetime, Path, dict[str, Any]]]] = defaultdict(
        list
    )
    for path in node_root.glob("*/*/*/*/case_manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_name = str(payload.get("case_name") or "")
            started_at = parse_manifest_time(payload.get("started_at"), path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if case_name not in expected_names or started_at < start:
            continue
        if end is not None and started_at >= end:
            continue
        manifest_ok = payload.get("status") == "ok" and payload.get("exit_code") == 0
        if not manifest_ok and not completed_benchmark_outputs(
            path.parent,
            int(rows_by_name[case_name]["max_requests"]),
        ):
            continue
        candidates[case_name].append((started_at, path, payload))

    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for case_name, values in candidates.items():
        _, path, payload = min(values, key=lambda item: item[0])
        selected[case_name] = (path, payload)
    return selected


def normalized_tpot_samples(path: Path) -> list[float]:
    samples: list[float] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                request = parse_json_bytes(raw_line)
                if request.get("is_error"):
                    continue
                value = request.get("tpot_by_e2e")
                if value is None:
                    e2e_ms = float(request["e2e_ms"])
                    output_tokens = int(request["actual_output_tokens"])
                    value = e2e_ms / output_tokens
                value = float(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{path}: invalid line {line_number}: {error}"
                ) from error
            if math.isfinite(value):
                samples.append(value)
    if not samples:
        raise RuntimeError(f"no successful latency samples in {path}")
    return samples


def measure_case(
    case_row: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    slo_target_ms: float,
) -> dict[str, Any]:
    benchmark_path = Path(manifest["paths"]["benchmark_dir"])
    requests_path = benchmark_path / "requests.jsonl"
    summary_path = benchmark_path / "summary.json"
    if not requests_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"benchmark output incomplete: {benchmark_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = normalized_tpot_samples(requests_path)
    successful_requests = int(summary.get("successful_requests", len(samples)))
    failed_requests = int(summary.get("failed_requests", 0))
    total_requests = int(
        summary.get("total_requests", successful_requests + failed_requests)
    )
    expected_requests = int(case_row["max_requests"])
    if total_requests != expected_requests:
        raise RuntimeError(
            f"{summary_path}: expected {expected_requests} requests, got {total_requests}"
        )
    if len(samples) != successful_requests:
        raise RuntimeError(
            f"{requests_path}: summary reports {successful_requests} successful "
            f"requests, parsed {len(samples)}"
        )

    values = np.asarray(samples, dtype=np.float64)
    slo_successes = int(np.count_nonzero(values <= slo_target_ms))
    runtime_s = float(summary["benchmark_runtime_s"])
    return {
        **case_row,
        "manifest_path": manifest_path.resolve(),
        "benchmark_path": benchmark_path.resolve(),
        "expected_requests": expected_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_ratio": failed_requests / total_requests,
        "slo_target_ms": slo_target_ms,
        "slo_attainment_pct": 100.0 * slo_successes / len(samples),
        "mean_tpot_ms": float(np.mean(values)),
        "p50_tpot_ms": float(np.percentile(values, 50)),
        "p90_tpot_ms": float(np.percentile(values, 90)),
        "p95_tpot_ms": float(np.percentile(values, 95)),
        "p99_tpot_ms": float(np.percentile(values, 99)),
        "benchmark_runtime_s": runtime_s,
        "achieved_rate_rps": float(summary["achieved_request_throughput_rps"]),
        "goodput_rps": slo_successes / runtime_s,
    }


def collect_run(
    node_root: Path,
    nodes: int,
    run_id: str,
    workloads: list[str],
) -> tuple[Path, list[dict[str, Any]], dict[str, tuple[Path, dict[str, Any]]]]:
    csv_path = case_csv_path(node_root, nodes, run_id)
    if not csv_path.is_file():
        raise FileNotFoundError(f"case CSV not found: {csv_path}")
    case_rows = load_case_rows(csv_path, nodes, workloads)
    start = run_start(csv_path, run_id)
    end = next_run_start(node_root, nodes, csv_path, start)
    manifests = successful_manifests(node_root, case_rows, start, end)
    return csv_path, case_rows, manifests


def choose_run(
    node_root: Path,
    nodes: int,
    requested_run_id: str | None,
    workloads: list[str],
) -> tuple[str, Path, list[dict[str, Any]], dict[str, tuple[Path, dict[str, Any]]]]:
    if requested_run_id:
        csv_path, rows, manifests = collect_run(
            node_root, nodes, requested_run_id, workloads
        )
        if not manifests:
            raise RuntimeError(f"run has no successful cases: {requested_run_id}")
        return requested_run_id, csv_path, rows, manifests

    csv_dir = node_root / "_case_csv"
    candidates: list[tuple[dt.datetime, str]] = []
    for path in csv_dir.glob(f"fig12-{nodes}node-*_cases.csv"):
        run_id = run_id_from_case_csv(path, nodes)
        if run_id is not None:
            candidates.append((run_start(path, run_id), run_id))
    for _, run_id in sorted(candidates, reverse=True):
        csv_path, rows, manifests = collect_run(node_root, nodes, run_id, workloads)
        if manifests:
            return run_id, csv_path, rows, manifests
    raise RuntimeError(f"no vLLM run with successful cases found under {node_root}")

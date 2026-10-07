#!/usr/bin/env python3
"""Parse completed points from one NanoDeploy E2E run."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from e2e_config import FIG12_DIR


RESULT_BASE = FIG12_DIR / "results" / "e2e" / "nanodeploy"
TOTAL_TIME_PATTERNS = (
    re.compile(r"Total time:\s*([\d.]+)s?"),
    re.compile(r"Total benchmark duration:\s*([\d.]+)s?"),
)

try:
    import orjson

    def parse_json(raw: bytes) -> dict[str, Any]:
        return orjson.loads(raw)

except ImportError:

    def parse_json(raw: bytes) -> dict[str, Any]:
        return json.loads(raw)


def resolve_log_dir(raw_path: str, workload_dir: Path) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [workload_dir / path, path]
    candidates.append(workload_dir / path.name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0]


def completed_points(
    run_dir: Path, workload_slugs: list[str]
) -> dict[str, list[dict[str, Any]]]:
    requested = set(workload_slugs)
    result: dict[str, list[dict[str, Any]]] = {}
    for workload_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        slug = workload_dir.name
        if requested and slug not in requested:
            continue
        summary_path = workload_dir / "run_summary.tsv"
        if not summary_path.is_file():
            continue

        by_rate: dict[float, dict[str, Any]] = {}
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if (row.get("status") or "").strip().lower() != "ok":
                    continue
                try:
                    rate = float(row["rate"])
                    expected = int(row["num_requests"])
                except (KeyError, TypeError, ValueError):
                    continue
                log_dir = resolve_log_dir(row.get("log_dir", ""), workload_dir)
                json_path = log_dir / "itl_samples.jsonl"
                log_path = log_dir / "driver.log"
                if json_path.is_file() and log_path.is_file():
                    by_rate[rate] = {
                        "rate": rate,
                        "expected_requests": expected,
                        "log_dir": log_dir,
                        "json_path": json_path,
                        "log_path": log_path,
                    }
        if by_rate:
            result[slug] = [by_rate[rate] for rate in sorted(by_rate)]
    return result


def choose_run(
    nodes: int, run_id: str | None, workloads: list[str]
) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    node_root = RESULT_BASE / f"{nodes}node"
    if run_id:
        run_dir = node_root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_dir}")
        points = completed_points(run_dir, workloads)
        if not points:
            raise RuntimeError(f"no completed points found in {run_dir}")
        return run_dir, points

    if not node_root.is_dir():
        raise FileNotFoundError(f"result directory not found: {node_root}")
    candidates = sorted(
        (path for path in node_root.iterdir() if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for run_dir in candidates:
        points = completed_points(run_dir, workloads)
        if points:
            return run_dir, points
    raise RuntimeError(f"no run with completed points found under {node_root}")


def normalized_tpot_samples(json_path: Path) -> list[float]:
    samples: list[float] = []
    with json_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                request = parse_json(raw_line)
                itl_samples = request.get("itl_samples") or []
                if not itl_samples:
                    continue
                queueing_ms = float(request.get("queueing_time_ms", 0.0) or 0.0)
                value = (sum(itl_samples) + queueing_ms) / len(itl_samples)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{json_path}: invalid line {line_number}: {error}"
                ) from error
            if math.isfinite(value):
                samples.append(float(value))
    if not samples:
        raise RuntimeError(f"no valid latency samples in {json_path}")
    return samples


def total_time_seconds(log_path: Path) -> float | None:
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for pattern in TOTAL_TIME_PATTERNS:
                match = pattern.search(line)
                if match:
                    return float(match.group(1))
    return None


def measure_point(point: dict[str, Any], slo_target_ms: float) -> dict[str, Any]:
    samples = normalized_tpot_samples(point["json_path"])
    expected = int(point["expected_requests"])
    if len(samples) != expected:
        raise RuntimeError(
            f"{point['json_path']}: expected {expected} requests, parsed {len(samples)}"
        )
    values = np.asarray(samples, dtype=np.float64)
    successes = int(np.count_nonzero(values <= slo_target_ms))
    duration = total_time_seconds(point["log_path"])
    return {
        **point,
        "parsed_requests": len(samples),
        "slo_target_ms": slo_target_ms,
        "slo_attainment_pct": 100.0 * successes / len(samples),
        "mean_tpot_ms": float(np.mean(values)),
        "p50_tpot_ms": float(np.percentile(values, 50)),
        "p90_tpot_ms": float(np.percentile(values, 90)),
        "p95_tpot_ms": float(np.percentile(values, 95)),
        "p99_tpot_ms": float(np.percentile(values, 99)),
        "duration_s": duration,
        "achieved_rate_rps": len(samples) / duration if duration else None,
        "goodput_rps": successes / duration if duration else None,
    }

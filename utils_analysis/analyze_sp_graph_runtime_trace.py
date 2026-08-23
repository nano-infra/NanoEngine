#!/usr/bin/env python3
"""Aggregate CUDA Graph launch and SP padding costs from PyTorch traces."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
PADDING_KERNEL_PATTERN = "zero_padded_rows_kernel"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict) or not isinstance(
        value.get("traceEvents"), list
    ):
        raise ValueError(f"{path} is not a Chrome/PyTorch trace")
    return value


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_us": None,
            "p50_us": None,
            "p95_us": None,
            "max_us": None,
            "mean_us": None,
        }
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "min_us": float(samples.min()),
        "p50_us": float(np.percentile(samples, 50)),
        "p95_us": float(np.percentile(samples, 95)),
        "max_us": float(samples.max()),
        "mean_us": float(samples.mean()),
    }


def _correlation(event: dict[str, Any]) -> int | str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    return args.get("correlation")


def _duration(event: dict[str, Any]) -> float | None:
    duration = event.get("dur")
    if isinstance(duration, (int, float)):
        return float(duration)
    return None


def _analyze_trace(path: Path) -> dict[str, Any]:
    trace = _load_trace(path)
    events = trace["traceEvents"]
    device_by_correlation: dict[int | str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in DEVICE_CATEGORIES:
            continue
        correlation = _correlation(event)
        if correlation is None or _duration(event) is None:
            continue
        device_by_correlation.setdefault(correlation, []).append(event)

    launches = [
        event
        for event in events
        if event.get("ph") == "X"
        and "cudagraphlaunch" in str(event.get("name", "")).lower()
        and _duration(event) is not None
    ]
    replay_rows = []
    for launch in launches:
        correlation = _correlation(launch)
        device_events = (
            device_by_correlation.get(correlation, [])
            if correlation is not None
            else []
        )
        device_begin = min(
            (float(event["ts"]) for event in device_events), default=None
        )
        device_end = max(
            (
                float(event["ts"]) + float(event["dur"])
                for event in device_events
            ),
            default=None,
        )
        padding_events = [
            event
            for event in device_events
            if PADDING_KERNEL_PATTERN
            in str(event.get("name", "")).lower()
        ]
        padding_durations = [float(event["dur"]) for event in padding_events]
        replay_rows.append(
            {
                "correlation": correlation,
                "host_launch_us": float(launch["dur"]),
                "device_event_count": len(device_events),
                "device_graph_span_us": (
                    device_end - device_begin
                    if device_begin is not None and device_end is not None
                    else None
                ),
                "padding_kernel_count": len(padding_events),
                "padding_kernel_sum_us": sum(padding_durations),
                "padding_kernel_durations_us": padding_durations,
            }
        )

    host_launches = [row["host_launch_us"] for row in replay_rows]
    graph_spans = [
        row["device_graph_span_us"]
        for row in replay_rows
        if row["device_graph_span_us"] is not None
    ]
    padding_individual = [
        duration
        for row in replay_rows
        for duration in row["padding_kernel_durations_us"]
    ]
    padding_sums = [row["padding_kernel_sum_us"] for row in replay_rows]
    padding_counts = [row["padding_kernel_count"] for row in replay_rows]
    return {
        "trace": str(path.resolve()),
        "launch_count": len(replay_rows),
        "unmatched_launch_count": sum(
            row["device_event_count"] == 0 for row in replay_rows
        ),
        "host_cuda_graph_launch": _summary(host_launches),
        "device_graph_span": _summary(graph_spans),
        "padding_kernel_individual": _summary(padding_individual),
        "padding_kernel_sum_per_replay": _summary(padding_sums),
        "padding_kernel_count_per_replay": {
            "values": padding_counts,
            "min": min(padding_counts) if padding_counts else None,
            "max": max(padding_counts) if padding_counts else None,
        },
        "replays": replay_rows,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def main() -> None:
    args = _parse_args()
    result = {
        "schema_version": 1,
        "padding_kernel_pattern": PADDING_KERNEL_PATTERN,
        "traces": [_analyze_trace(path) for path in args.traces],
        "notes": [
            "cudaGraphLaunch host duration is not device Graph execution time.",
            "Padding sums are attribution values and must not be added to host timing.",
        ],
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

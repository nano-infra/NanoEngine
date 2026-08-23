"""Summarize CUDA Graph replays in a PyTorch/Chrome profiler trace.

The parser is streaming and bounded-memory so multi-GB rank traces do not need
to be materialized as one Python object. Graph replays are grouped by their
kernel count, which cleanly separates Nano's recurrent MTP and target verify
graphs without relying on fragile thread or stream identifiers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

READ_SIZE = 4 * 1024 * 1024
TRACE_KEY = '"traceEvents"'


def _open_trace(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace", buffering=READ_SIZE)


def events(path: Path):
    decoder = json.JSONDecoder(strict=False)
    with _open_trace(path) as source:
        buffer = ""
        position = 0
        while True:
            chunk = source.read(READ_SIZE)
            if not chunk:
                raise ValueError(f"{path}: missing traceEvents")
            buffer += chunk
            key_position = buffer.find(TRACE_KEY)
            if key_position < 0:
                buffer = buffer[-len(TRACE_KEY) :]
                continue
            array_position = buffer.find("[", key_position + len(TRACE_KEY))
            if array_position < 0:
                continue
            position = array_position + 1
            break

        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                buffer = buffer[position:]
                position = 0
                chunk = source.read(READ_SIZE)
                if not chunk:
                    raise ValueError(f"{path}: unterminated traceEvents")
                buffer += chunk
                continue
            position = end
            if isinstance(event, dict):
                yield event
            if position > READ_SIZE:
                buffer = buffer[position:]
                position = 0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }


def _top_kernels(graphs: list[dict], limit: int | None = 40) -> list[dict]:
    totals = defaultdict(lambda: [0, 0.0, 0.0])
    for graph in graphs:
        for name, values in graph["kernels"].items():
            target = totals[name]
            target[0] += values[0]
            target[1] += values[1]
            target[2] = max(target[2], values[2])
    rows = [
        {
            "name": name,
            "count": values[0],
            "count_per_graph": values[0] / len(graphs),
            "total_us": values[1],
            "mean_us_per_graph": values[1] / len(graphs),
            "mean_us_per_call": values[1] / values[0],
            "max_us": values[2],
        }
        for name, values in totals.items()
    ]
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    return rows if limit is None else rows[:limit]


def summarize(path: Path, kernel_limit: int | None = 40) -> dict:
    launches = {}
    for index, event in enumerate(events(path), 1):
        if index % 1_000_000 == 0:
            print(f"launch pass: {index:,}", file=sys.stderr, flush=True)
        if event.get("cat") != "cuda_runtime" or event.get("name") != "cudaGraphLaunch":
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        if correlation is None:
            continue
        launches[int(correlation)] = {
            "correlation": int(correlation),
            "launch_ts": float(event.get("ts", 0.0)),
            "launch_cpu_us": float(event.get("dur", 0.0) or 0.0),
            "first_gpu_ts": None,
            "last_gpu_ts": None,
            "kernel_count": 0,
            "gpu_event_count": 0,
            "kernels": defaultdict(lambda: [0, 0.0, 0.0]),
        }

    for index, event in enumerate(events(path), 1):
        if index % 1_000_000 == 0:
            print(f"GPU pass: {index:,}", file=sys.stderr, flush=True)
        category = event.get("cat")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        graph = launches.get(correlation)
        if graph is None:
            continue
        ts = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0) or 0.0)
        end = ts + duration
        first = graph["first_gpu_ts"]
        last = graph["last_gpu_ts"]
        graph["first_gpu_ts"] = ts if first is None else min(first, ts)
        graph["last_gpu_ts"] = end if last is None else max(last, end)
        graph["gpu_event_count"] += 1
        if category == "kernel":
            graph["kernel_count"] += 1
            values = graph["kernels"][str(event.get("name", ""))]
            values[0] += 1
            values[1] += duration
            values[2] = max(values[2], duration)

    completed = []
    for graph in launches.values():
        if graph["first_gpu_ts"] is None:
            continue
        graph["gpu_span_us"] = graph["last_gpu_ts"] - graph["first_gpu_ts"]
        completed.append(graph)

    grouped = defaultdict(list)
    for graph in completed:
        grouped[graph["kernel_count"]].append(graph)

    groups = []
    for kernel_count, graphs in grouped.items():
        groups.append(
            {
                "kernel_count": kernel_count,
                "graphs": len(graphs),
                "gpu_event_count": _distribution(
                    [float(graph["gpu_event_count"]) for graph in graphs]
                ),
                "launch_cpu_us": _distribution(
                    [graph["launch_cpu_us"] for graph in graphs]
                ),
                "gpu_span_us": _distribution(
                    [graph["gpu_span_us"] for graph in graphs]
                ),
                "top_kernels": _top_kernels(graphs, kernel_limit),
            }
        )
    groups.sort(
        key=lambda group: (group["graphs"], group["kernel_count"]), reverse=True
    )

    repeated = [group for group in groups if group["graphs"] > 1]
    repeated.sort(key=lambda group: group["kernel_count"])
    labels = {}
    if repeated:
        labels["small_graph"] = repeated[0]["kernel_count"]
    if len(repeated) > 1:
        labels["large_graph"] = repeated[-1]["kernel_count"]
    return {
        "path": str(path),
        "graph_launches": len(launches),
        "completed_graphs": len(completed),
        "labels": labels,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--kernel-limit",
        type=int,
        default=40,
        help="kernels retained per graph group; 0 retains every kernel",
    )
    args = parser.parse_args()
    report = summarize(
        args.trace, None if args.kernel_limit == 0 else args.kernel_limit
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parse NanoDeploy traces into automatic per-layer, per-rank breakdowns."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

import ijson


MODEL_LAYER_COUNT = 61
FIRST_MOE_LAYER = 3
MEASURED_LAYER_RANGE = range(FIRST_MOE_LAYER, MODEL_LAYER_COUNT - 1)
MOE_EVENTS_PER_LAYER = 2
MLA_ANCHOR = "get_mla_metadata_kernel"
ATTENTION_SPLITKV_KERNEL = "flash_fwd_splitkv_mla_kernel"
ATTENTION_COMBINE_KERNEL = "flash_fwd_mla_combine_kernel"
MOE_DISPATCH_KERNEL = "deep_ep::internode_ll::dispatch"
MOE_COMBINE_KERNEL = "deep_ep::internode_ll::combine"
CP_DLSLIME_PREFIX = "dlslime::intranode"
LEGACY_CP_ALLTOALL_KERNEL = "all_to_all_intra_ll_kernel"

INTERESTING_KERNELS = (
    MLA_ANCHOR,
    ATTENTION_SPLITKV_KERNEL,
    ATTENTION_COMBINE_KERNEL,
    MOE_DISPATCH_KERNEL,
    MOE_COMBINE_KERNEL,
    CP_DLSLIME_PREFIX,
    LEGACY_CP_ALLTOALL_KERNEL,
)
METADATA_RANK_RE = re.compile(r"_rank(\d+)$")
TRACE_RANK_RE = re.compile(r"_rank_?(\d+)(?:\.|_)")

BASE_METRICS = (
    "total_us",
    "attention_splitkv_us",
    "attention_combine_us",
    "attention_us",
    "cp_alltoall_us",
    "cp_comm_us",
    "moe_dispatch_us",
    "moe_combine_us",
    "moe_dispatch_combine_us",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=TRACE_PATH",
        help=(
            "Dataset name and a NanoDeploy trace directory, per-rank trace, "
            "or merged trace. Repeat for each dataset."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=32,
        help="Required ranks per case; use 0 to disable (default: 32)",
    )
    parser.add_argument(
        "--expected-iterations",
        type=int,
        default=0,
        help=(
            "Required complete CUDA Graph iterations per rank; 0 auto-detects "
            "and requires the same positive count on every rank (default: 0)"
        ),
    )
    args = parser.parse_args()
    if args.expected_ranks < 0:
        parser.error("--expected-ranks cannot be negative")
    if args.expected_iterations < 0:
        parser.error("--expected-iterations cannot be negative")
    return args


def parse_named_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Invalid --case {value!r}; expected NAME=TRACE_PATH")
    name, raw_value = value.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()
    if not name or not raw_value:
        raise ValueError(f"Invalid --case {value!r}; name and path are required")
    return name, raw_value


def is_merged_trace(path: Path) -> bool:
    return "merged" in path.name and path.name.endswith(".gz")


def per_rank_traces(root: Path) -> tuple[Path, ...]:
    compressed = sorted(
        path
        for path in root.rglob("*.pt.trace.json.gz")
        if not is_merged_trace(path)
    )
    if compressed:
        return tuple(compressed)
    return tuple(
        sorted(
            path
            for path in root.rglob("*.pt.trace.json")
            if not is_merged_trace(path)
        )
    )


def resolve_trace_sources(raw_path: str) -> tuple[Path, ...]:
    source = Path(raw_path).expanduser().resolve()
    if source.is_dir():
        traces = per_rank_traces(source)
        if traces:
            return traces
        merged = sorted(source.rglob("*merged*.gz"))
        if len(merged) == 1:
            return (merged[0],)
        raise ValueError(
            f"Expected per-rank traces or one merged trace in {source}; "
            f"found {len(merged)} merged candidates"
        )
    if not source.is_file():
        raise ValueError(f"NanoDeploy trace does not exist: {source}")
    if is_merged_trace(source):
        siblings = per_rank_traces(source.parent)
        if siblings:
            print(
                f"[{source.name}] using {len(siblings)} per-rank sibling traces",
                flush=True,
            )
            return siblings
        return (source,)
    if source.name.endswith((".pt.trace.json", ".pt.trace.json.gz")):
        return (source,)
    raise ValueError(f"Unsupported NanoDeploy trace input: {source}")


def build_cases(values: list[str]) -> list[tuple[str, tuple[Path, ...]]]:
    cases: list[tuple[str, tuple[Path, ...]]] = []
    seen: set[str] = set()
    for value in values:
        name, raw_path = parse_named_value(value)
        if name in seen:
            raise ValueError(f"Duplicate --case name: {name}")
        seen.add(name)
        cases.append((name, resolve_trace_sources(raw_path)))
    return cases


def open_trace(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def metadata_name(event: dict[str, Any]) -> str | None:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    value = args.get("name")
    return str(value) if value is not None else None


def contains_any(name: str, fragments: Iterable[str]) -> bool:
    return any(fragment in name for fragment in fragments)


def load_interesting_events(
    path: Path,
) -> tuple[
    dict[tuple[Any, Any], str],
    dict[Any, str],
    dict[tuple[Any, Any], list[dict[str, Any]]],
]:
    thread_names: dict[tuple[Any, Any], str] = {}
    process_names: dict[Any, str] = {}
    events_by_track: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)

    with open_trace(path) as handle:
        for event in ijson.items(handle, "traceEvents.item", use_float=True):
            if not isinstance(event, dict):
                continue
            phase = event.get("ph")
            pid = event.get("pid")
            tid = event.get("tid")
            if phase == "M":
                name = metadata_name(event)
                if name is None:
                    continue
                if event.get("name") == "thread_name":
                    thread_names[(pid, tid)] = name
                elif event.get("name") == "process_name":
                    process_names[pid] = name
                continue
            if phase != "X" or event.get("cat") != "kernel":
                continue
            name = str(event.get("name", ""))
            if contains_any(name, INTERESTING_KERNELS):
                events_by_track[(pid, tid)].append(event)
    return thread_names, process_names, events_by_track


def metadata_rank(thread_name: str | None, process_name: str | None) -> int | None:
    for value in (thread_name, process_name):
        if value is None:
            continue
        match = METADATA_RANK_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def filename_rank(path: Path) -> int | None:
    match = TRACE_RANK_RE.search(path.name)
    return int(match.group(1)) if match else None


def event_correlation(event: dict[str, Any]) -> Any | None:
    args = event.get("args")
    return args.get("correlation") if isinstance(args, dict) else None


def matched_events(
    events: list[dict[str, Any]], fragment: str
) -> list[dict[str, Any]]:
    return [event for event in events if fragment in str(event.get("name", ""))]


def is_cp_alltoall(name: str) -> bool:
    normalized = name.lower().replace("_", "")
    return (
        "dlslime::intranodealltoall" in normalized
        or LEGACY_CP_ALLTOALL_KERNEL in name
    )


def cp_alltoall_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if is_cp_alltoall(str(event.get("name", "")))
    ]


def duration_sum(events: list[dict[str, Any]]) -> float:
    return sum(float(event.get("dur", 0.0)) for event in events)


def parse_track(
    dataset: str,
    trace: Path,
    pid: Any,
    tid: Any,
    thread_name: str,
    process_name: str,
    rank: int,
    events: list[dict[str, Any]],
    expected_iterations: int,
) -> list[dict[str, Any]]:
    by_correlation: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        correlation = event_correlation(event)
        if correlation is not None:
            by_correlation[correlation].append(event)

    complete: list[tuple[float, Any, list[dict[str, Any]]]] = []
    for correlation, group in by_correlation.items():
        anchors = matched_events(group, MLA_ANCHOR)
        if len(anchors) != MODEL_LAYER_COUNT:
            continue
        start = min(float(event["ts"]) for event in anchors)
        complete.append((start, correlation, group))
    complete.sort(key=lambda item: item[0])

    if expected_iterations and len(complete) != expected_iterations:
        raise ValueError(
            f"{dataset}/rank{rank}/{trace.name}: expected "
            f"{expected_iterations} complete iterations, found {len(complete)}"
        )
    if not complete:
        raise ValueError(
            f"{dataset}/rank{rank}/{trace.name}: no correlation group contains "
            f"exactly {MODEL_LAYER_COUNT} MLA anchors"
        )

    rows: list[dict[str, Any]] = []
    for iteration, (_, correlation, group) in enumerate(complete, start=1):
        group = sorted(group, key=lambda event: float(event["ts"]))
        timestamps = [float(event["ts"]) for event in group]
        anchors = sorted(
            matched_events(group, MLA_ANCHOR),
            key=lambda event: float(event["ts"]),
        )

        for layer in MEASURED_LAYER_RANGE:
            window_start = float(anchors[layer]["ts"])
            window_end = float(anchors[layer + 1]["ts"])
            begin_index = bisect_left(timestamps, window_start)
            end_index = bisect_left(timestamps, window_end)
            window_events = group[begin_index:end_index]

            attention_splitkv = matched_events(
                window_events, ATTENTION_SPLITKV_KERNEL
            )
            attention_combine = matched_events(
                window_events, ATTENTION_COMBINE_KERNEL
            )
            dispatch = matched_events(window_events, MOE_DISPATCH_KERNEL)
            combine = matched_events(window_events, MOE_COMBINE_KERNEL)
            cp_alltoall = cp_alltoall_events(window_events)

            if len(attention_splitkv) != 1 or len(attention_combine) != 1:
                raise ValueError(
                    f"{dataset}/rank{rank}/iteration{iteration}/layer{layer}: "
                    "expected one split-KV and one MLA combine kernel, found "
                    f"{len(attention_splitkv)} and {len(attention_combine)}"
                )
            if (
                len(dispatch) != MOE_EVENTS_PER_LAYER
                or len(combine) != MOE_EVENTS_PER_LAYER
            ):
                raise ValueError(
                    f"{dataset}/rank{rank}/iteration{iteration}/layer{layer}: "
                    f"expected {MOE_EVENTS_PER_LAYER} DeepEP dispatch and "
                    "combine kernels, found "
                    f"{len(dispatch)} and {len(combine)}"
                )

            attention_splitkv_us = duration_sum(attention_splitkv)
            attention_combine_us = duration_sum(attention_combine)
            attention_us = attention_splitkv_us + attention_combine_us
            dispatch_us = duration_sum(dispatch)
            combine_us = duration_sum(combine)
            dispatch_combine_us = dispatch_us + combine_us
            cp_alltoall_us = duration_sum(cp_alltoall)
            cp_comm_us = cp_alltoall_us
            total_us = window_end - window_start
            other_us = total_us - attention_us - dispatch_combine_us - cp_comm_us
            if other_us < -1e-6:
                raise ValueError(
                    f"{dataset}/rank{rank}/iteration{iteration}/layer{layer}: "
                    f"categorized durations exceed wall time by {-other_us:.3f} us"
                )

            rows.append(
                {
                    "dataset": dataset,
                    "case": "nano_dcp",
                    "strategy": "nano_dcp",
                    "trace": trace.name,
                    "global_rank": rank,
                    "pid": pid,
                    "tid": tid,
                    "thread_name": thread_name,
                    "process_name": process_name,
                    "iteration": iteration,
                    "correlation": correlation,
                    "layer": layer,
                    "window_start_us": window_start,
                    "window_end_us": window_end,
                    "layer_samples": 1,
                    "attention_splitkv_kernel_count": len(attention_splitkv),
                    "attention_combine_kernel_count": len(attention_combine),
                    "moe_dispatch_kernel_count": len(dispatch),
                    "moe_combine_kernel_count": len(combine),
                    "cp_alltoall_kernel_count": len(cp_alltoall),
                    "total_us": total_us,
                    "attention_splitkv_us": attention_splitkv_us,
                    "attention_combine_us": attention_combine_us,
                    "attention_us": attention_us,
                    "cp_alltoall_us": cp_alltoall_us,
                    "cp_comm_us": cp_comm_us,
                    "moe_dispatch_us": dispatch_us,
                    "moe_combine_us": combine_us,
                    "moe_dispatch_combine_us": dispatch_combine_us,
                    "other_us": max(other_us, 0.0),
                }
            )
    return rows


def parse_source(
    dataset: str,
    trace: Path,
    expected_iterations: int,
) -> list[dict[str, Any]]:
    print(f"[{dataset}] streaming {trace}", flush=True)
    thread_names, process_names, events_by_track = load_interesting_events(trace)
    fallback_rank = filename_rank(trace)
    candidates = [
        (track, events)
        for track, events in events_by_track.items()
        if matched_events(events, MLA_ANCHOR)
    ]
    if fallback_rank is not None and len(candidates) != 1:
        raise ValueError(
            f"{dataset}/{trace.name}: per-rank trace must contain exactly one "
            f"MLA anchor track, found {len(candidates)}"
        )

    rows: list[dict[str, Any]] = []
    for (pid, tid), events in candidates:
        thread_name = thread_names.get((pid, tid), "")
        process_name = process_names.get(pid, "")
        named_rank = metadata_rank(thread_name, process_name)
        if named_rank is not None and fallback_rank is not None:
            if named_rank != fallback_rank:
                raise ValueError(
                    f"{trace.name}: metadata rank {named_rank} differs from "
                    f"filename rank {fallback_rank}"
                )
        rank = named_rank if named_rank is not None else fallback_rank
        if rank is None:
            raise ValueError(
                f"{dataset}/{trace.name}/{pid}:{tid}: cannot infer rank from "
                "merged metadata or per-rank filename"
            )
        rows.extend(
            parse_track(
                dataset,
                trace,
                pid,
                tid,
                thread_name,
                process_name,
                rank,
                events,
                expected_iterations,
            )
        )
    return rows


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def add_median_breakdown(
    output: dict[str, Any], samples: list[dict[str, Any]]
) -> None:
    for key in BASE_METRICS:
        output[key] = median(samples, key)
    output["other_us"] = (
        output["total_us"]
        - output["attention_us"]
        - output["cp_comm_us"]
        - output["moe_dispatch_combine_us"]
    )
    if output["other_us"] < -1e-6:
        raise ValueError("Median component durations exceed median layer wall time")


def rank_summaries(
    dataset: str,
    layer_rows: list[dict[str, Any]],
    expected_ranks: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[int(row["global_rank"])].append(row)

    ranks = sorted(grouped)
    if expected_ranks and ranks != list(range(expected_ranks)):
        raise ValueError(
            f"{dataset}: expected ranks 0--{expected_ranks - 1}, found {ranks}"
        )

    summaries: list[dict[str, Any]] = []
    expected_layers = len(MEASURED_LAYER_RANGE)
    for rank in ranks:
        samples = grouped[rank]
        per_iteration: dict[tuple[str, int], int] = defaultdict(int)
        for sample in samples:
            per_iteration[(str(sample["trace"]), int(sample["iteration"]))] += 1
        if set(per_iteration.values()) != {expected_layers}:
            raise ValueError(
                f"{dataset}/rank{rank}: expected {expected_layers} layers in "
                f"every iteration, found {dict(per_iteration)}"
            )
        row: dict[str, Any] = {
            "dataset": dataset,
            "case": "nano_dcp",
            "strategy": "nano_dcp",
            "global_rank": rank,
            "trace_count": len({str(sample["trace"]) for sample in samples}),
            "model_iterations": len(per_iteration),
            "layers_per_iteration": expected_layers,
            "layer_samples": len(samples),
        }
        add_median_breakdown(row, samples)
        summaries.append(row)

    iteration_counts = {int(row["model_iterations"]) for row in summaries}
    if len(iteration_counts) != 1:
        raise ValueError(
            f"{dataset}: ranks contain different complete iteration counts: "
            f"{sorted(iteration_counts)}"
        )
    return summaries


def case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    summary: dict[str, Any] = {
        "dataset": first["dataset"],
        "case": first["case"],
        "strategy": first["strategy"],
        "ranks": len(rows),
        "model_iterations_per_rank": min(
            int(row["model_iterations"]) for row in rows
        ),
        "layers_per_iteration": len(MEASURED_LAYER_RANGE),
        "layer_samples_per_rank": min(int(row["layer_samples"]) for row in rows),
        "min_rank_total_us": min(float(row["total_us"]) for row in rows),
        "max_rank_total_us": max(float(row["total_us"]) for row in rows),
    }
    add_median_breakdown(summary, rows)
    return summary


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.3f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    print(f"Wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        cases = build_cases(args.case)
        layers_by_case: dict[str, list[dict[str, Any]]] = {}
        ranks_by_case: dict[str, list[dict[str, Any]]] = {}
        for dataset, traces in cases:
            layer_rows: list[dict[str, Any]] = []
            for trace in traces:
                layer_rows.extend(
                    parse_source(dataset, trace, args.expected_iterations)
                )
            layer_rows.sort(
                key=lambda row: (
                    int(row["global_rank"]),
                    str(row["trace"]),
                    int(row["iteration"]),
                    int(row["layer"]),
                )
            )
            rank_rows = rank_summaries(dataset, layer_rows, args.expected_ranks)
            layers_by_case[dataset] = layer_rows
            ranks_by_case[dataset] = rank_rows
    except (OSError, ValueError, ijson.JSONError) as exc:
        raise SystemExit(str(exc)) from exc

    layer_fields = (
        "dataset",
        "case",
        "strategy",
        "trace",
        "global_rank",
        "pid",
        "tid",
        "thread_name",
        "process_name",
        "iteration",
        "correlation",
        "layer",
        "window_start_us",
        "window_end_us",
        "layer_samples",
        "attention_splitkv_kernel_count",
        "attention_combine_kernel_count",
        "moe_dispatch_kernel_count",
        "moe_combine_kernel_count",
        "cp_alltoall_kernel_count",
    ) + BASE_METRICS + ("other_us",)
    rank_fields = (
        "dataset",
        "case",
        "strategy",
        "global_rank",
        "trace_count",
        "model_iterations",
        "layers_per_iteration",
        "layer_samples",
    ) + BASE_METRICS + ("other_us",)
    summary_fields = (
        "dataset",
        "case",
        "strategy",
        "ranks",
        "model_iterations_per_rank",
        "layers_per_iteration",
        "layer_samples_per_rank",
        "min_rank_total_us",
        "max_rank_total_us",
    ) + BASE_METRICS + ("other_us",)

    output_dir = args.output_dir.expanduser().resolve()
    all_layers = [row for rows in layers_by_case.values() for row in rows]
    all_ranks = [row for rows in ranks_by_case.values() for row in rows]
    write_csv(output_dir / "layer_breakdown.csv", layer_fields, all_layers)
    write_csv(output_dir / "layer_rank_summary.csv", rank_fields, all_ranks)
    write_csv(output_dir / "plot_rank_summary.csv", rank_fields, all_ranks)

    summary_rows = []
    for dataset, _ in cases:
        layer_rows = layers_by_case[dataset]
        rank_rows = ranks_by_case[dataset]
        write_csv(
            output_dir / dataset / "layer_breakdown.csv",
            layer_fields,
            layer_rows,
        )
        write_csv(
            output_dir / dataset / "layer_rank_summary.csv",
            rank_fields,
            rank_rows,
        )
        write_csv(
            output_dir / dataset / "plot_rank_summary.csv",
            rank_fields,
            rank_rows,
        )
        summary = case_summary(rank_rows)
        summary_rows.append(summary)
        print(
            f"[{dataset}] ranks={summary['ranks']}, "
            f"iterations/rank={summary['model_iterations_per_rank']}, "
            f"layers/iteration={summary['layers_per_iteration']}, "
            f"samples/rank={summary['layer_samples_per_rank']}, "
            f"median total={summary['total_us']:.3f} us, "
            f"attention={summary['attention_us']:.3f} us, "
            f"cp={summary['cp_comm_us']:.3f} us, "
            f"dispatch+combine={summary['moe_dispatch_combine_us']:.3f} us",
            flush=True,
        )
    write_csv(output_dir / "case_summary.csv", summary_fields, summary_rows)


if __name__ == "__main__":
    main()

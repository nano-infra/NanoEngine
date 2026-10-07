#!/usr/bin/env python3
"""Parse Fig. 13 torch-profiler traces into auditable plotting CSVs."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


STRATEGY_ORDER = ("nano dcp", "dp4dcp8", "dp8dcp4", "dp16cp2", "dp32")
LONG_NODE_ORDER = (1, 3, 5, 7)
QUICK_VLLM_STRATEGIES = {
    "dp2dcp8": "dp4dcp8",
    "dp4dcp4": "dp8dcp4",
    "dp8cp2": "dp16cp2",
    "dp16": "dp32",
}
CONTEXT_PARALLEL_SIZE = {
    "nano dcp": 8,
    "dp4dcp8": 8,
    "dp8dcp4": 4,
    "dp16cp2": 2,
    "dp32": 1,
}

MODEL_LAYER_COUNT = 61
FIRST_MOE_LAYER = 3
# The last layer has no following attention anchor, so its layer period cannot
# be measured with the same start-to-start rule as the other layers.
MEASURED_LAYER_RANGE = range(FIRST_MOE_LAYER, MODEL_LAYER_COUNT - 1)

ATTENTION_KERNELS = (
    "flash_fwd_splitkv_mla_kernel",
    "flash_fwd_mla_combine_kernel",
)
NANO_ATTENTION_KERNELS = ATTENTION_KERNELS + ("get_mla_metadata_kernel",)
ATTENTION_ANCHOR = "flash_fwd_splitkv_mla_kernel"
MOE_KERNELS = (
    "deep_ep::internode_ll::dispatch",
    "deep_ep::internode_ll::combine",
)
NANO_CP_KERNELS = ("dlslime::intranode_alltoall_kernel",)
VLLM_COMMON_CP_KERNELS = (
    "ncclDevKernel_AllGather_RING_LL",
    "ncclDevKernel_SendRecv",
)
VLLM_AR_KERNELS = {
    "dp16cp2": ("two_shot_all_reduce_kernel_inplace",),
    "dp8dcp4": ("multimem_all_reduce_kernel",),
    "dp4dcp8": ("multimem_all_reduce_kernel",),
}

LONG_NODE_RE = re.compile(r"mix_([1357])x512k_pernode")
VLLM_STRATEGY_RE = re.compile(r"(?:^|[/_-])(dp4dcp8|dp8dcp4|dp16cp2|dp32)(?:[/_-]|$)")
QUICK_VLLM_STRATEGY_RE = re.compile(
    r"(?:^|[/_-])(dp2dcp8|dp4dcp4|dp8cp2|dp16)(?:[/_-]|$)"
)
NANO_RANK_RE = re.compile(r"_rank_(\d+)\.")
VLLM_RANK_RE = re.compile(
    r"benchmark_dp(?P<dp>\d+)_pp(?P<pp>\d+)_tp(?P<tp>\d+)_"
    r"dcp(?P<dcp>\d+)_ep(?P<ep>\d+)_rank(?P<rank>\d+)\."
)


@dataclass(frozen=True)
class LayerSample:
    total: float
    attn: float
    moe_a2a: float
    cp_cost: float
    attention_event_count: int
    moe_event_count: int
    cp_event_count: int


@dataclass(frozen=True)
class TraceSummary:
    path: Path
    rank: str
    iterations: int
    layer_samples: int
    total: float
    attn: float
    moe_a2a: float
    cp_cost: float
    other: float
    attention_events_per_layer: float
    moe_events_per_layer: float
    cp_events_per_layer: float


@dataclass(frozen=True)
class CaseSpec:
    long_node: int
    strategy: str
    trace_root: Path
    trace_paths: tuple[Path, ...]
    nodes: int
    gpus_per_node: int


@dataclass(frozen=True)
class CaseResult:
    plot_row: dict[str, int | str | float]
    trace_rows: tuple[dict[str, int | str | float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case = subparsers.add_parser(
        "case",
        help="Parse one explicitly named profile case (also useful for quick tests)",
    )
    case.add_argument("--trace-root", type=Path, required=True)
    case.add_argument("--strategy", choices=STRATEGY_ORDER, required=True)
    case.add_argument("--long-node", type=int, required=True)
    case.add_argument("--nodes", type=int, required=True)
    case.add_argument("--gpus-per-node", type=int, default=8)
    add_output_arguments(case)

    quick = subparsers.add_parser(
        "quick",
        help="Discover and parse the available cases from a two-node quick-test run",
    )
    quick.add_argument("--run-root", type=Path, required=True)
    quick.add_argument("--nodes", type=int, default=2)
    quick.add_argument("--gpus-per-node", type=int, default=8)
    add_output_arguments(quick)

    matrix = subparsers.add_parser(
        "matrix",
        help="Discover and parse the complete four-node Fig. 13 profile matrix",
    )
    matrix.add_argument("--nano-root", type=Path, required=True)
    matrix.add_argument("--vllm-root", type=Path, required=True)
    matrix.add_argument("--nodes", type=int, default=4)
    matrix.add_argument("--gpus-per-node", type=int, default=8)
    add_output_arguments(matrix)

    args = parser.parse_args()
    for name in ("nodes", "gpus_per_node"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "case" and args.long_node <= 0:
        parser.error("--long-node must be positive")
    return args


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Plotting CSV with long_node,strategy,attn,moe_a2a,cp_cost,total",
    )
    parser.add_argument(
        "--trace-summary",
        type=Path,
        required=True,
        help="Per-trace audit CSV, including long-rank selection",
    )


def find_traces(root: Path) -> tuple[Path, ...]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Trace root is not a directory: {root}")
    paths = list(root.rglob("*.pt.trace.json"))
    paths.extend(root.rglob("*.pt.trace.json.gz"))
    return tuple(sorted(set(paths)))


def open_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Trace root must be a JSON object: {path}")
    return document


def matches_any(name: str, substrings: Iterable[str]) -> bool:
    return any(substring in name for substring in substrings)


def cp_kernel_names(strategy: str) -> tuple[str, ...]:
    if strategy == "nano dcp":
        return NANO_CP_KERNELS
    if strategy == "dp32":
        return ()
    return VLLM_COMMON_CP_KERNELS + VLLM_AR_KERNELS[strategy]


def attention_kernel_names(strategy: str) -> tuple[str, ...]:
    if strategy == "nano dcp":
        return NANO_ATTENTION_KERNELS
    return ATTENTION_KERNELS


def trace_rank(path: Path) -> str:
    nano_match = NANO_RANK_RE.search(path.name)
    if nano_match:
        return f"rank={nano_match.group(1)}"
    vllm_match = VLLM_RANK_RE.search(path.name)
    if vllm_match:
        fields = vllm_match.groupdict()
        return "/".join(
            f"{name}={fields[name]}" for name in ("dp", "tp", "dcp", "ep", "rank")
        )
    return "unknown"


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot take the median of an empty sequence")
    return float(statistics.median(materialized))


def parse_trace(path: Path, strategy: str) -> TraceSummary:
    document = open_trace(path)
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"Missing traceEvents array: {path}")

    kernels_by_correlation: dict[Any, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        args = event.get("args")
        correlation = args.get("correlation") if isinstance(args, dict) else None
        if correlation is None:
            continue
        kernels_by_correlation.setdefault(correlation, []).append(event)

    samples: list[LayerSample] = []
    iteration_count = 0
    attention_names = attention_kernel_names(strategy)
    cp_names = cp_kernel_names(strategy)
    for kernels in kernels_by_correlation.values():
        anchors = sorted(
            (
                event
                for event in kernels
                if ATTENTION_ANCHOR in str(event.get("name", ""))
            ),
            key=lambda event: float(event["ts"]),
        )
        if len(anchors) != MODEL_LAYER_COUNT:
            continue

        iteration_count += 1
        kernels.sort(key=lambda event: float(event["ts"]))
        timestamps = [float(event["ts"]) for event in kernels]
        for layer in MEASURED_LAYER_RANGE:
            start = float(anchors[layer]["ts"])
            end = float(anchors[layer + 1]["ts"])
            begin_index = bisect_left(timestamps, start)
            end_index = bisect_left(timestamps, end)
            layer_events = kernels[begin_index:end_index]

            attention_events = [
                event
                for event in layer_events
                if matches_any(str(event.get("name", "")), attention_names)
            ]
            moe_events = [
                event
                for event in layer_events
                if matches_any(str(event.get("name", "")), MOE_KERNELS)
            ]
            cp_events = [
                event
                for event in layer_events
                if matches_any(str(event.get("name", "")), cp_names)
            ]
            if len(attention_events) != len(attention_names):
                raise ValueError(
                    f"Expected {len(attention_names)} attention kernels in layer "
                    f"{layer}, found "
                    f"{len(attention_events)}: {path}"
                )
            if not moe_events:
                raise ValueError(f"No MoE dispatch/combine kernels in layer {layer}: {path}")
            if strategy != "dp32" and not cp_events:
                raise ValueError(f"No CP communication kernels in layer {layer}: {path}")

            samples.append(
                LayerSample(
                    total=end - start,
                    attn=sum(float(event.get("dur", 0.0)) for event in attention_events),
                    moe_a2a=sum(float(event.get("dur", 0.0)) for event in moe_events),
                    cp_cost=sum(float(event.get("dur", 0.0)) for event in cp_events),
                    attention_event_count=len(attention_events),
                    moe_event_count=len(moe_events),
                    cp_event_count=len(cp_events),
                )
            )

    if not samples:
        raise ValueError(
            "No complete 61-layer CUDA Graph iterations were found in "
            f"{path}. Check that this is a decode profiler trace."
        )

    total = median(sample.total for sample in samples)
    attn = median(sample.attn for sample in samples)
    moe_a2a = median(sample.moe_a2a for sample in samples)
    cp_cost = median(sample.cp_cost for sample in samples)
    other = total - attn - moe_a2a - cp_cost
    if other < -1e-6:
        raise ValueError(
            f"Negative residual for {path}: total={total}, attn={attn}, "
            f"moe_a2a={moe_a2a}, cp_cost={cp_cost}"
        )

    return TraceSummary(
        path=path,
        rank=trace_rank(path),
        iterations=iteration_count,
        layer_samples=len(samples),
        total=total,
        attn=attn,
        moe_a2a=moe_a2a,
        cp_cost=cp_cost,
        other=max(other, 0.0),
        attention_events_per_layer=median(
            float(sample.attention_event_count) for sample in samples
        ),
        moe_events_per_layer=median(float(sample.moe_event_count) for sample in samples),
        cp_events_per_layer=median(float(sample.cp_event_count) for sample in samples),
    )


def relative_trace(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def aggregate_case(case: CaseSpec) -> CaseResult:
    expected_traces = case.nodes * case.gpus_per_node
    if len(case.trace_paths) != expected_traces:
        raise ValueError(
            f"{case.strategy} long_node={case.long_node} expected {expected_traces} "
            f"traces, found {len(case.trace_paths)} below {case.trace_root}. Use a "
            "fresh output root containing exactly one trace per GPU."
        )

    summaries: list[TraceSummary] = []
    for index, path in enumerate(case.trace_paths, start=1):
        print(
            f"[{case.strategy} long={case.long_node}] parsing "
            f"{index}/{len(case.trace_paths)}: {relative_trace(path, case.trace_root)}",
            flush=True,
        )
        summaries.append(parse_trace(path, case.strategy))

    long_trace_count = min(
        expected_traces,
        case.nodes * case.long_node * CONTEXT_PARALLEL_SIZE[case.strategy],
    )
    ranked = sorted(summaries, key=lambda summary: (-summary.attn, str(summary.path)))
    selected_paths = {summary.path for summary in ranked[:long_trace_count]}
    selected = [summary for summary in summaries if summary.path in selected_paths]

    plot_row: dict[str, int | str | float] = {
        "long_node": case.long_node,
        "strategy": case.strategy,
        "attn": median(summary.attn for summary in selected),
        "moe_a2a": median(summary.moe_a2a for summary in selected),
        "cp_cost": median(summary.cp_cost for summary in selected),
        "total": median(summary.total for summary in selected),
    }
    selection_position = {summary.path: index for index, summary in enumerate(ranked, 1)}
    trace_rows: list[dict[str, int | str | float]] = []
    for summary in sorted(summaries, key=lambda item: str(item.path)):
        trace_rows.append(
            {
                "long_node": case.long_node,
                "strategy": case.strategy,
                "selected_long_rank": int(summary.path in selected_paths),
                "attention_rank": selection_position[summary.path],
                "trace": relative_trace(summary.path, case.trace_root),
                "rank": summary.rank,
                "iterations": summary.iterations,
                "layer_samples": summary.layer_samples,
                "attn": summary.attn,
                "moe_a2a": summary.moe_a2a,
                "cp_cost": summary.cp_cost,
                "other": summary.other,
                "total": summary.total,
                "attention_events_per_layer": summary.attention_events_per_layer,
                "moe_events_per_layer": summary.moe_events_per_layer,
                "cp_events_per_layer": summary.cp_events_per_layer,
            }
        )

    print(
        f"[{case.strategy} long={case.long_node}] selected "
        f"{len(selected)}/{len(summaries)} long-request traces; "
        f"attn={plot_row['attn']:.3f} us, "
        f"moe_a2a={plot_row['moe_a2a']:.3f} us, "
        f"cp_cost={plot_row['cp_cost']:.3f} us, "
        f"total={plot_row['total']:.3f} us",
        flush=True,
    )
    return CaseResult(plot_row=plot_row, trace_rows=tuple(trace_rows))


def discover_matrix_cases(args: argparse.Namespace) -> list[CaseSpec]:
    nano_root = args.nano_root.expanduser().resolve()
    vllm_root = args.vllm_root.expanduser().resolve()
    grouped: dict[tuple[int, str], list[Path]] = {}

    for path in find_traces(nano_root):
        match = LONG_NODE_RE.search(str(path.relative_to(nano_root)))
        if not match:
            raise ValueError(f"Cannot infer long_node from NanoDeploy trace path: {path}")
        grouped.setdefault((int(match.group(1)), "nano dcp"), []).append(path)

    for path in find_traces(vllm_root):
        relative = str(path.relative_to(vllm_root))
        long_match = LONG_NODE_RE.search(relative)
        strategy_match = VLLM_STRATEGY_RE.search(relative)
        if not long_match or not strategy_match:
            raise ValueError(f"Cannot infer Fig. 13 case from vLLM trace path: {path}")
        grouped.setdefault(
            (int(long_match.group(1)), strategy_match.group(1)), []
        ).append(path)

    expected = {
        (long_node, strategy)
        for long_node in LONG_NODE_ORDER
        for strategy in STRATEGY_ORDER
    }
    actual = set(grouped)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Incomplete Fig. 13 matrix; missing={missing}, extra={extra}")

    cases = []
    for long_node in LONG_NODE_ORDER:
        for strategy in STRATEGY_ORDER:
            root = nano_root if strategy == "nano dcp" else vllm_root
            cases.append(
                CaseSpec(
                    long_node=long_node,
                    strategy=strategy,
                    trace_root=root,
                    trace_paths=tuple(sorted(grouped[(long_node, strategy)])),
                    nodes=args.nodes,
                    gpus_per_node=args.gpus_per_node,
                )
            )
    return cases


def discover_quick_cases(args: argparse.Namespace) -> list[CaseSpec]:
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"Quick-test run root is not a directory: {run_root}")

    grouped: dict[str, list[Path]] = {}
    nano_root = run_root / "nano"
    if nano_root.is_dir():
        nano_traces = list(find_traces(nano_root))
        if nano_traces:
            grouped["nano dcp"] = nano_traces

    vllm_root = run_root / "vllm"
    if vllm_root.is_dir():
        for path in find_traces(vllm_root):
            relative = str(path.relative_to(vllm_root))
            match = QUICK_VLLM_STRATEGY_RE.search(relative)
            if not match:
                raise ValueError(f"Cannot infer quick-test strategy from trace path: {path}")
            strategy = QUICK_VLLM_STRATEGIES[match.group(1)]
            grouped.setdefault(strategy, []).append(path)

    if not grouped:
        raise ValueError(f"No torch-profiler traces found below {run_root}")

    cases = []
    for strategy in STRATEGY_ORDER:
        trace_paths = grouped.get(strategy)
        if not trace_paths:
            continue
        trace_root = nano_root if strategy == "nano dcp" else vllm_root
        cases.append(
            CaseSpec(
                long_node=1,
                strategy=strategy,
                trace_root=trace_root,
                trace_paths=tuple(sorted(trace_paths)),
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
            )
        )
    return cases


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
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
    if args.command == "case":
        trace_root = args.trace_root.expanduser().resolve()
        traces = find_traces(trace_root)
        if not traces:
            raise SystemExit(f"No torch-profiler traces found below {trace_root}")
        cases = [
            CaseSpec(
                long_node=args.long_node,
                strategy=args.strategy,
                trace_root=trace_root,
                trace_paths=traces,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
            )
        ]
    elif args.command == "quick":
        try:
            cases = discover_quick_cases(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        try:
            cases = discover_matrix_cases(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    results: list[CaseResult] = []
    try:
        for case in cases:
            results.append(aggregate_case(case))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    plot_rows = [result.plot_row for result in results]
    trace_rows = [row for result in results for row in result.trace_rows]
    write_csv(
        args.output,
        ("long_node", "strategy", "attn", "moe_a2a", "cp_cost", "total"),
        plot_rows,
    )
    write_csv(
        args.trace_summary,
        (
            "long_node",
            "strategy",
            "selected_long_rank",
            "attention_rank",
            "trace",
            "rank",
            "iterations",
            "layer_samples",
            "attn",
            "moe_a2a",
            "cp_cost",
            "other",
            "total",
            "attention_events_per_layer",
            "moe_events_per_layer",
            "cp_events_per_layer",
        ),
        trace_rows,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parse Fig. 15 vLLM traces into iteration and per-layer breakdowns.

The historical ``-merged-*.tar.gz`` files are gzip-compressed merged Chrome
trace JSON documents, not tar archives.  When one of those files is supplied,
this script parses the per-rank ``*.pt.trace.json.gz`` siblings in the same
directory.  Keeping ranks separate prevents correlation-ID collisions and
avoids loading a multi-gigabyte merged JSON document into memory.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


MODEL_LAYER_COUNT = 61
FIRST_MOE_LAYER = 3
MOE_LAYER_COUNT = 58
MOE_EVENTS_PER_LAYER = 2
# Layer latency is measured from one attention anchor to the next.  The final
# model layer has no following anchor, so the paper plot uses MoE layers 3--59.
MEASURED_LAYER_RANGE = range(FIRST_MOE_LAYER, MODEL_LAYER_COUNT - 1)
ATTENTION_ANCHOR = "flash_fwd_splitkv_mla_kernel"
ATTENTION_KERNELS = (
    ATTENTION_ANCHOR,
    "flash_fwd_mla_combine_kernel",
)
MOE_DISPATCH_KERNEL = "deep_ep::internode_ll::dispatch"
MOE_COMBINE_KERNEL = "deep_ep::internode_ll::combine"

# These are payload collectives introduced by TP/CP in the Fig. 15 vLLM DCP
# traces.  The scalar nccl AllReduce_Sum_u32 control collective also appears
# in pure DP traces and is deliberately not classified as CP communication.
CP_COMM_KERNELS = (
    "ncclDevKernel_AllGather",
    "ncclDevKernel_SendRecv",
    "ncclDevKernel_ReduceScatter",
    "multimem_all_reduce_kernel",
    "two_shot_all_reduce_kernel_inplace",
)

STRATEGY_RE = re.compile(
    r"(?:^|[/_.-])(dp\d+(?:dcp\d+)?(?:_(?:agrs|naive))?)(?:[/_.-]|$)"
)
TRACE_RANK_RE = re.compile(
    r"benchmark_dp(?P<dp>\d+)_pp(?P<pp>\d+)_tp(?P<tp>\d+)_"
    r"dcp(?P<dcp>\d+)_ep(?P<ep>\d+)_rank(?P<rank>\d+)\."
)

BASE_METRIC_COLUMNS = (
    "total_us",
    "attention_us",
    "cp_comm_us",
    "moe_dispatch_us",
    "moe_combine_us",
)
METRIC_COLUMNS = BASE_METRIC_COLUMNS + (
    "moe_dispatch_combine_us",
    "other_us",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    source: Path
    strategy: str
    traces: tuple[Path, ...]


@dataclass(frozen=True)
class TraceIdentity:
    dp_rank: int
    pp_rank: int
    tp_rank: int
    dcp_rank: int
    ep_rank: int
    local_rank: int

    @property
    def global_rank(self) -> int:
        # Expert parallelism spans every GPU in each evaluated DeepSeek-V3
        # configuration, so the EP rank is also the global worker rank.
        return self.ep_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=TRACE_PATH",
        help=(
            "Case label and either a trace directory, one per-rank trace, or "
            "a historical -merged-*.tar.gz file. Repeat for multiple cases."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for iteration, layer, summary, and matched-kernel CSVs.",
    )
    parser.add_argument(
        "--expected-traces",
        type=int,
        default=32,
        help="Required per-rank traces per case; use 0 to disable (default: 32).",
    )
    parser.add_argument(
        "--expected-iterations",
        type=int,
        default=31,
        help=(
            "Required complete CUDA Graph iterations per trace; use 0 to "
            "disable (default: 31)."
        ),
    )
    args = parser.parse_args()
    if args.expected_traces < 0:
        parser.error("--expected-traces cannot be negative")
    if args.expected_iterations < 0:
        parser.error("--expected-iterations cannot be negative")
    return args


def parse_case_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --case {value!r}; expected NAME=TRACE_PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"Invalid --case {value!r}; name and path are required")
    return name, Path(raw_path).expanduser()


def infer_strategy(name: str, path: Path) -> str:
    path_matches = tuple(STRATEGY_RE.finditer(str(path)))
    match = path_matches[-1] if path_matches else STRATEGY_RE.search(name)
    if not match:
        raise ValueError(
            f"Cannot infer a DP or DCP strategy for case {name!r} from {path}"
        )
    return match.group(1)


def is_merged_trace(path: Path) -> bool:
    return path.name.startswith("-merged-") and path.name.endswith(".tar.gz")


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


def discover_traces(source: Path) -> tuple[Path, ...]:
    source = source.resolve()
    if source.is_dir():
        traces = per_rank_traces(source)
        if not traces:
            raise ValueError(f"No per-rank torch-profiler traces below {source}")
        return traces
    if not source.is_file():
        raise ValueError(f"Trace input does not exist: {source}")
    if is_merged_trace(source):
        traces = per_rank_traces(source.parent)
        if not traces:
            raise ValueError(
                f"{source} is a merged gzip JSON trace, but no per-rank sibling "
                "traces are available. Supply the original per-rank traces."
            )
        print(
            f"[{source.name}] using {len(traces)} per-rank sibling traces; "
            "the merged file is not a tar archive",
            flush=True,
        )
        return traces
    if source.name.endswith((".pt.trace.json", ".pt.trace.json.gz")):
        return (source,)
    raise ValueError(f"Unsupported trace input: {source}")


def build_cases(values: Iterable[str], expected_traces: int) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for value in values:
        name, source = parse_case_argument(value)
        if name in seen:
            raise ValueError(f"Duplicate case name: {name}")
        seen.add(name)
        strategy = infer_strategy(name, source)
        traces = discover_traces(source)
        if expected_traces and len(traces) != expected_traces:
            raise ValueError(
                f"{name}: expected {expected_traces} per-rank traces, "
                f"found {len(traces)}"
            )
        cases.append(
            CaseSpec(
                name=name,
                source=source.resolve(),
                strategy=strategy,
                traces=traces,
            )
        )
    return cases


def trace_identity(path: Path) -> TraceIdentity:
    match = TRACE_RANK_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse vLLM ranks from trace name: {path.name}")
    fields = {name: int(value) for name, value in match.groupdict().items()}
    return TraceIdentity(
        dp_rank=fields["dp"],
        pp_rank=fields["pp"],
        tp_rank=fields["tp"],
        dcp_rank=fields["dcp"],
        ep_rank=fields["ep"],
        local_rank=fields["rank"],
    )


def open_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Trace root must be a JSON object: {path}")
    return document


def is_cp_communication(name: str, strategy: str) -> bool:
    if "dcp" not in strategy:
        return False
    if any(fragment in name for fragment in CP_COMM_KERNELS):
        return True
    return "ncclDevKernel_AllReduce" in name and "_Sum_u32_" not in name


def kernel_category(name: str, strategy: str) -> str | None:
    if any(fragment in name for fragment in ATTENTION_KERNELS):
        return "attention"
    if MOE_DISPATCH_KERNEL in name:
        return "moe_dispatch"
    if MOE_COMBINE_KERNEL in name:
        return "moe_combine"
    if is_cp_communication(name, strategy):
        return "cp_comm"
    return None


def parse_trace(
    case: CaseSpec,
    path: Path,
    expected_iterations: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[int, float]],
]:
    identity = trace_identity(path)
    document = open_trace(path)
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"Missing traceEvents array: {path}")

    kernels_by_correlation: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        args = event.get("args")
        correlation = args.get("correlation") if isinstance(args, dict) else None
        if correlation is None:
            continue
        kernels_by_correlation[correlation].append(event)

    complete_groups: list[tuple[float, Any, list[dict[str, Any]]]] = []
    for correlation, kernels in kernels_by_correlation.items():
        anchor_count = sum(
            ATTENTION_ANCHOR in str(event.get("name", "")) for event in kernels
        )
        if anchor_count != MODEL_LAYER_COUNT:
            continue
        start = min(float(event["ts"]) for event in kernels)
        complete_groups.append((start, correlation, kernels))
    complete_groups.sort(key=lambda item: item[0])

    if expected_iterations and len(complete_groups) != expected_iterations:
        raise ValueError(
            f"{case.name}/{path.name}: expected {expected_iterations} complete "
            f"iterations, found {len(complete_groups)}"
        )
    if not complete_groups:
        raise ValueError(
            f"{case.name}/{path.name}: no correlation group contains exactly "
            f"{MODEL_LAYER_COUNT} attention anchors"
        )

    iteration_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    audit: dict[tuple[str, str], tuple[int, float]] = {}
    for iteration, (_, correlation, kernels) in enumerate(complete_groups, start=1):
        kernels = sorted(kernels, key=lambda event: float(event["ts"]))
        timestamps = [float(event["ts"]) for event in kernels]
        anchors = [
            event
            for event in kernels
            if ATTENTION_ANCHOR in str(event.get("name", ""))
        ]
        iteration_start = min(float(event["ts"]) for event in kernels)
        iteration_end = max(
            float(event["ts"]) + float(event.get("dur", 0.0))
            for event in kernels
        )
        duration_by_category: dict[str, float] = defaultdict(float)
        count_by_category: dict[str, int] = defaultdict(int)
        for event in kernels:
            name = str(event.get("name", ""))
            category = kernel_category(name, case.strategy)
            if category is None:
                continue
            duration = float(event.get("dur", 0.0))
            duration_by_category[category] += duration
            count_by_category[category] += 1
            old_count, old_duration = audit.get((category, name), (0, 0.0))
            audit[(category, name)] = (old_count + 1, old_duration + duration)

        expected_attention_kernels = MODEL_LAYER_COUNT * len(ATTENTION_KERNELS)
        if count_by_category["attention"] != expected_attention_kernels:
            raise ValueError(
                f"{case.name}/{path.name}/iteration {iteration}: expected "
                f"{expected_attention_kernels} attention kernels, found "
                f"{count_by_category['attention']}"
            )
        expected_moe_kernels = MOE_LAYER_COUNT * MOE_EVENTS_PER_LAYER
        if count_by_category["moe_dispatch"] != expected_moe_kernels:
            raise ValueError(
                f"{case.name}/{path.name}/iteration {iteration}: expected "
                f"{expected_moe_kernels} MoE dispatch kernels, found "
                f"{count_by_category['moe_dispatch']}"
            )
        if count_by_category["moe_combine"] != expected_moe_kernels:
            raise ValueError(
                f"{case.name}/{path.name}/iteration {iteration}: expected "
                f"{expected_moe_kernels} MoE combine kernels, found "
                f"{count_by_category['moe_combine']}"
            )
        if "dcp" in case.strategy and not count_by_category["cp_comm"]:
            raise ValueError(
                f"{case.name}/{path.name}/iteration {iteration}: "
                "no CP communication kernels"
            )

        total_us = iteration_end - iteration_start
        attention_us = duration_by_category["attention"]
        cp_comm_us = duration_by_category["cp_comm"]
        dispatch_us = duration_by_category["moe_dispatch"]
        combine_us = duration_by_category["moe_combine"]
        dispatch_combine_us = dispatch_us + combine_us
        component_sum_us = attention_us + cp_comm_us + dispatch_combine_us
        iteration_rows.append(
            {
                "case": case.name,
                "strategy": case.strategy,
                "trace": path.name,
                "global_rank": identity.global_rank,
                "dp_rank": identity.dp_rank,
                "pp_rank": identity.pp_rank,
                "tp_rank": identity.tp_rank,
                "dcp_rank": identity.dcp_rank,
                "ep_rank": identity.ep_rank,
                "local_rank": identity.local_rank,
                "iteration": iteration,
                "correlation": correlation,
                "kernel_count": len(kernels),
                "attention_kernel_count": count_by_category["attention"],
                "cp_comm_kernel_count": count_by_category["cp_comm"],
                "moe_dispatch_kernel_count": count_by_category["moe_dispatch"],
                "moe_combine_kernel_count": count_by_category["moe_combine"],
                "total_us": total_us,
                "attention_us": attention_us,
                "cp_comm_us": cp_comm_us,
                "moe_dispatch_us": dispatch_us,
                "moe_combine_us": combine_us,
                "moe_dispatch_combine_us": dispatch_combine_us,
                "other_us": total_us - component_sum_us,
            }
        )

        for layer in MEASURED_LAYER_RANGE:
            layer_start = float(anchors[layer]["ts"])
            layer_end = float(anchors[layer + 1]["ts"])
            begin_index = bisect_left(timestamps, layer_start)
            end_index = bisect_left(timestamps, layer_end)
            layer_events = kernels[begin_index:end_index]

            layer_duration_by_category: dict[str, float] = defaultdict(float)
            layer_count_by_category: dict[str, int] = defaultdict(int)
            for event in layer_events:
                category = kernel_category(str(event.get("name", "")), case.strategy)
                if category is None:
                    continue
                layer_duration_by_category[category] += float(event.get("dur", 0.0))
                layer_count_by_category[category] += 1

            if layer_count_by_category["attention"] != len(ATTENTION_KERNELS):
                raise ValueError(
                    f"{case.name}/{path.name}/iteration {iteration}/layer {layer}: "
                    f"expected {len(ATTENTION_KERNELS)} attention kernels, found "
                    f"{layer_count_by_category['attention']}"
                )
            for category in ("moe_dispatch", "moe_combine"):
                if layer_count_by_category[category] != MOE_EVENTS_PER_LAYER:
                    raise ValueError(
                        f"{case.name}/{path.name}/iteration {iteration}/layer {layer}: "
                        f"expected {MOE_EVENTS_PER_LAYER} {category} kernels, found "
                        f"{layer_count_by_category[category]}"
                    )
            if "dcp" in case.strategy and not layer_count_by_category["cp_comm"]:
                raise ValueError(
                    f"{case.name}/{path.name}/iteration {iteration}/layer {layer}: "
                    "no CP communication kernels"
                )

            layer_total_us = layer_end - layer_start
            layer_attention_us = layer_duration_by_category["attention"]
            layer_cp_comm_us = layer_duration_by_category["cp_comm"]
            layer_dispatch_us = layer_duration_by_category["moe_dispatch"]
            layer_combine_us = layer_duration_by_category["moe_combine"]
            layer_dispatch_combine_us = layer_dispatch_us + layer_combine_us
            layer_component_sum_us = (
                layer_attention_us
                + layer_cp_comm_us
                + layer_dispatch_combine_us
            )
            layer_other_us = layer_total_us - layer_component_sum_us
            if layer_other_us < -1e-6:
                raise ValueError(
                    f"{case.name}/{path.name}/iteration {iteration}/layer {layer}: "
                    f"negative residual {layer_other_us:.6f} us"
                )

            layer_rows.append(
                {
                    "case": case.name,
                    "strategy": case.strategy,
                    "trace": path.name,
                    "global_rank": identity.global_rank,
                    "dp_rank": identity.dp_rank,
                    "pp_rank": identity.pp_rank,
                    "tp_rank": identity.tp_rank,
                    "dcp_rank": identity.dcp_rank,
                    "ep_rank": identity.ep_rank,
                    "local_rank": identity.local_rank,
                    "iteration": iteration,
                    "correlation": correlation,
                    "layer": layer,
                    "kernel_count": len(layer_events),
                    "attention_kernel_count": layer_count_by_category["attention"],
                    "cp_comm_kernel_count": layer_count_by_category["cp_comm"],
                    "moe_dispatch_kernel_count": layer_count_by_category[
                        "moe_dispatch"
                    ],
                    "moe_combine_kernel_count": layer_count_by_category[
                        "moe_combine"
                    ],
                    "total_us": layer_total_us,
                    "attention_us": layer_attention_us,
                    "cp_comm_us": layer_cp_comm_us,
                    "moe_dispatch_us": layer_dispatch_us,
                    "moe_combine_us": layer_combine_us,
                    "moe_dispatch_combine_us": layer_dispatch_combine_us,
                    "other_us": max(layer_other_us, 0.0),
                }
            )
    return iteration_rows, layer_rows, audit


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot take the median of an empty sequence")
    return float(statistics.median(materialized))


def add_median_breakdown(
    output: dict[str, Any], samples: list[dict[str, Any]]
) -> None:
    for metric in BASE_METRIC_COLUMNS + ("moe_dispatch_combine_us",):
        output[metric] = median(float(sample[metric]) for sample in samples)
    output["other_us"] = (
        output["total_us"]
        - output["attention_us"]
        - output["cp_comm_us"]
        - output["moe_dispatch_combine_us"]
    )


def rank_summaries(iteration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iteration_rows:
        grouped[(str(row["case"]), str(row["trace"]))].append(row)

    rows: list[dict[str, Any]] = []
    for _, samples in grouped.items():
        first = samples[0]
        row = {
            key: first[key]
            for key in (
                "case",
                "strategy",
                "trace",
                "global_rank",
                "dp_rank",
                "pp_rank",
                "tp_rank",
                "dcp_rank",
                "ep_rank",
                "local_rank",
            )
        }
        row["iterations"] = len(samples)
        add_median_breakdown(row, samples)
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["case"]), int(row["global_rank"])))


def layer_rank_summaries(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[(str(row["case"]), str(row["trace"]))].append(row)

    rows: list[dict[str, Any]] = []
    expected_layers = len(MEASURED_LAYER_RANGE)
    for _, samples in grouped.items():
        first = samples[0]
        layer_counts: dict[int, int] = defaultdict(int)
        for sample in samples:
            layer_counts[int(sample["iteration"])] += 1
        if set(layer_counts.values()) != {expected_layers}:
            raise ValueError(
                f"{first['case']}/{first['trace']}: expected {expected_layers} "
                f"measured layers in every iteration, found {dict(layer_counts)}"
            )

        row = {
            key: first[key]
            for key in (
                "case",
                "strategy",
                "trace",
                "global_rank",
                "dp_rank",
                "pp_rank",
                "tp_rank",
                "dcp_rank",
                "ep_rank",
                "local_rank",
            )
        }
        row["model_iterations"] = len(layer_counts)
        row["layers_per_iteration"] = expected_layers
        row["layer_samples"] = len(samples)
        add_median_breakdown(row, samples)
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["case"]), int(row["global_rank"])))


def case_summaries(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        grouped[str(row["case"])].append(row)

    rows: list[dict[str, Any]] = []
    for case_name, samples in grouped.items():
        row: dict[str, Any] = {
            "case": case_name,
            "strategy": samples[0]["strategy"],
            "traces": len(samples),
            "iterations_per_trace": min(
                int(sample["iterations"]) for sample in samples
            ),
            "min_rank_total_us": min(float(sample["total_us"]) for sample in samples),
            "max_rank_total_us": max(float(sample["total_us"]) for sample in samples),
        }
        add_median_breakdown(row, samples)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["case"]))


def layer_case_summaries(
    layer_rank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rank_rows:
        grouped[str(row["case"])].append(row)

    rows: list[dict[str, Any]] = []
    for case_name, samples in grouped.items():
        row: dict[str, Any] = {
            "case": case_name,
            "strategy": samples[0]["strategy"],
            "traces": len(samples),
            "model_iterations_per_trace": min(
                int(sample["model_iterations"]) for sample in samples
            ),
            "layers_per_iteration": min(
                int(sample["layers_per_iteration"]) for sample in samples
            ),
            "layer_samples_per_trace": min(
                int(sample["layer_samples"]) for sample in samples
            ),
            "min_rank_total_us": min(
                float(sample["total_us"]) for sample in samples
            ),
            "max_rank_total_us": max(
                float(sample["total_us"]) for sample in samples
            ),
        }
        add_median_breakdown(row, samples)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["case"]))


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
        cases = build_cases(args.case, args.expected_traces)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    iteration_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    audit_totals: dict[tuple[str, str, str, str], tuple[int, float]] = {}
    try:
        for case in cases:
            for index, path in enumerate(case.traces, start=1):
                print(
                    f"[{case.name}] parsing {index}/{len(case.traces)}: {path.name}",
                    flush=True,
                )
                trace_iteration_rows, trace_layer_rows, audit = parse_trace(
                    case, path, args.expected_iterations
                )
                iteration_rows.extend(trace_iteration_rows)
                layer_rows.extend(trace_layer_rows)
                for (category, kernel), (count, duration) in audit.items():
                    key = (case.name, case.strategy, category, kernel)
                    old_count, old_duration = audit_totals.get(key, (0, 0.0))
                    audit_totals[key] = (
                        old_count + count,
                        old_duration + duration,
                    )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    iteration_rows.sort(
        key=lambda row: (
            str(row["case"]),
            int(row["global_rank"]),
            int(row["iteration"]),
        )
    )
    layer_rows.sort(
        key=lambda row: (
            str(row["case"]),
            int(row["global_rank"]),
            int(row["iteration"]),
            int(row["layer"]),
        )
    )
    rank_rows = rank_summaries(iteration_rows)
    case_rows = case_summaries(rank_rows)
    layer_rank_rows = layer_rank_summaries(layer_rows)
    layer_case_rows = layer_case_summaries(layer_rank_rows)
    audit_rows = []
    for (case, strategy, category, kernel), (count, duration) in sorted(
        audit_totals.items()
    ):
        audit_rows.append(
            {
                "case": case,
                "strategy": strategy,
                "category": category,
                "kernel": kernel,
                "event_count": count,
                "total_duration_us": duration,
                "mean_duration_us": duration / count,
            }
        )

    output_dir = args.output_dir.resolve()
    identity_columns = (
        "case",
        "strategy",
        "trace",
        "global_rank",
        "dp_rank",
        "pp_rank",
        "tp_rank",
        "dcp_rank",
        "ep_rank",
        "local_rank",
    )
    write_csv(
        output_dir / "iteration_breakdown.csv",
        identity_columns
        + (
            "iteration",
            "correlation",
            "kernel_count",
            "attention_kernel_count",
            "cp_comm_kernel_count",
            "moe_dispatch_kernel_count",
            "moe_combine_kernel_count",
        )
        + METRIC_COLUMNS,
        iteration_rows,
    )
    write_csv(
        output_dir / "rank_summary.csv",
        identity_columns + ("iterations",) + METRIC_COLUMNS,
        rank_rows,
    )
    write_csv(
        output_dir / "case_summary.csv",
        (
            "case",
            "strategy",
            "traces",
            "iterations_per_trace",
            "min_rank_total_us",
            "max_rank_total_us",
        )
        + METRIC_COLUMNS,
        case_rows,
    )
    write_csv(
        output_dir / "layer_breakdown.csv",
        identity_columns
        + (
            "iteration",
            "correlation",
            "layer",
            "kernel_count",
            "attention_kernel_count",
            "cp_comm_kernel_count",
            "moe_dispatch_kernel_count",
            "moe_combine_kernel_count",
        )
        + METRIC_COLUMNS,
        layer_rows,
    )
    write_csv(
        output_dir / "layer_rank_summary.csv",
        identity_columns
        + ("model_iterations", "layers_per_iteration", "layer_samples")
        + METRIC_COLUMNS,
        layer_rank_rows,
    )
    write_csv(
        output_dir / "layer_case_summary.csv",
        (
            "case",
            "strategy",
            "traces",
            "model_iterations_per_trace",
            "layers_per_iteration",
            "layer_samples_per_trace",
            "min_rank_total_us",
            "max_rank_total_us",
        )
        + METRIC_COLUMNS,
        layer_case_rows,
    )
    write_csv(
        output_dir / "matched_kernels.csv",
        (
            "case",
            "strategy",
            "category",
            "kernel",
            "event_count",
            "total_duration_us",
            "mean_duration_us",
        ),
        audit_rows,
    )

    for row in case_rows:
        print(
            f"[{row['case']}] traces={row['traces']}, "
            f"iterations/trace={row['iterations_per_trace']}, "
            f"median total={row['total_us']:.3f} us, "
            f"attention={row['attention_us']:.3f} us, "
            f"cp={row['cp_comm_us']:.3f} us, "
            f"dispatch+combine={row['moe_dispatch_combine_us']:.3f} us",
            flush=True,
        )
    for row in layer_case_rows:
        print(
            f"[{row['case']}] per-layer/rank: "
            f"samples/trace={row['layer_samples_per_trace']}, "
            f"median total={row['total_us']:.3f} us, "
            f"attention={row['attention_us']:.3f} us, "
            f"cp={row['cp_comm_us']:.3f} us, "
            f"dispatch+combine={row['moe_dispatch_combine_us']:.3f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()

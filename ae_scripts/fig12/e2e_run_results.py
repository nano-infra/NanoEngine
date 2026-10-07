#!/usr/bin/env python3
"""Parse and plot selected NanoDeploy and vLLM Figure 12 runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig12-matplotlib-cache")
)

import nanodeploy_run_data as nano
import vllm_run_data as vllm
from e2e_config import FIG12_DIR, WORKLOADS
from standalone_plot_style import SeriesStyle, plot_matrix


DEFAULT_SLO_TARGET_MS = 50.0
WORKLOAD_BY_SLUG = {workload.slug: workload for workload in WORKLOADS}
NANO_SERIES_ID = "nano_dp4sp8"
SERIES_ORDER = (NANO_SERIES_ID, *vllm.BASELINE_ORDER)
SERIES_STYLES = {
    NANO_SERIES_ID: SeriesStyle(
        label="Ours (DCP)",
        color="#d62728",
        marker="D",
        linestyle="-",
    ),
    "dp_least_batch": SeriesStyle(
        label="vLLM (DP-LeastBatch)",
        color="#1f77b4",
        marker="o",
        linestyle="--",
    ),
    "dp_least_cache": SeriesStyle(
        label="vLLM (DP-LeastCache)",
        color="#4c78a8",
        marker="s",
        linestyle="--",
    ),
    "cp2": SeriesStyle(
        label="vLLM (CP2)",
        color="#2ca02c",
        marker="^",
        linestyle="--",
    ),
    "cp4": SeriesStyle(
        label="vLLM (CP4)",
        color="#9467bd",
        marker="v",
        linestyle="--",
    ),
    "cp8": SeriesStyle(
        label="vLLM (CP8)",
        color="#17becf",
        marker="P",
        linestyle="--",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot completed points from a NanoDeploy run, a vLLM run, or both. "
            "When both are supplied, all series are drawn in one figure."
        )
    )
    parser.add_argument(
        "--nodes",
        type=int,
        choices=(2, 4),
        default=4,
        help="cluster size used by the selected run (default: %(default)s)",
    )
    parser.add_argument(
        "--nano-run",
        help="NanoDeploy run ID or run directory",
    )
    parser.add_argument(
        "--vllm-run",
        help="vLLM run ID or _runs/<run-name> directory",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=tuple(WORKLOAD_BY_SLUG),
        default=[],
        help="workload to plot; repeat to select several (default: all found)",
    )
    parser.add_argument(
        "--slo-target",
        type=float,
        default=DEFAULT_SLO_TARGET_MS,
        help=f"normalized TPOT SLO in milliseconds (default: {DEFAULT_SLO_TARGET_MS:g})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: fig12/results/e2e/plots/<selected-runs>)",
    )
    args = parser.parse_args()
    if not args.nano_run and not args.vllm_run:
        parser.error("pass --nano-run, --vllm-run, or both")
    return args


def resolve_nano_run(value: str, nodes: int) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(f"NanoDeploy input is not a directory: {candidate}")
    if candidate.parent != Path("."):
        raise FileNotFoundError(f"NanoDeploy run directory not found: {candidate}")
    run_dir = nano.RESULT_BASE / f"{nodes}node" / value
    if not run_dir.is_dir():
        raise FileNotFoundError(f"NanoDeploy run not found: {run_dir}")
    return run_dir.resolve()


def collect_nano(
    value: str,
    nodes: int,
    workloads: list[str],
    slo_target_ms: float,
) -> tuple[str, list[dict[str, Any]]]:
    run_dir = resolve_nano_run(value, nodes)
    grouped_points = nano.completed_points(run_dir, workloads)
    if not grouped_points:
        raise RuntimeError(f"no completed NanoDeploy points found in {run_dir}")

    measured: list[dict[str, Any]] = []
    print(f"NanoDeploy run: {run_dir}")
    for workload_slug, points in grouped_points.items():
        for point in points:
            print(f"[nano] parse {workload_slug} rate={point['rate']:g}")
            row = nano.measure_point(point, slo_target_ms)
            measured.append(
                {
                    **row,
                    "system": "NanoDeploy",
                    "workload": workload_slug,
                    "plot_series": NANO_SERIES_ID,
                    "plot_rate": float(row["rate"]),
                    "successful_requests": int(row["parsed_requests"]),
                    "failed_requests": 0,
                    "runtime_s": row["duration_s"],
                    "source_path": Path(row["log_dir"]),
                }
            )
    return run_dir.name, measured


def load_vllm_run_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"vLLM run manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid vLLM run manifest: {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"vLLM run manifest has no results list: {manifest_path}")
    return manifest_path, payload


def vllm_run_id(
    manifest: dict[str, Any],
    run_dir: Path,
    nodes: int,
) -> str:
    rate_plan = str(manifest.get("rate_plan") or "")
    prefix = f"case_csv:fig12-{nodes}node-"
    suffix = "_cases.csv"
    if rate_plan.startswith(prefix) and rate_plan.endswith(suffix):
        return rate_plan[len(prefix) : -len(suffix)]
    return run_dir.name


def resolve_vllm_run(
    value: str,
    nodes: int,
) -> tuple[str, Path, Path, dict[str, Any]]:
    candidate = Path(value)
    if candidate.is_dir():
        run_dir = candidate.resolve()
        manifest_path, manifest = load_vllm_run_manifest(run_dir)
        return vllm_run_id(manifest, run_dir, nodes), run_dir, manifest_path, manifest
    if candidate.exists():
        raise ValueError(f"vLLM input must be a run ID or run directory: {candidate}")
    if candidate.parent != Path("."):
        raise FileNotFoundError(f"vLLM run directory not found: {candidate}")

    node_root = vllm.RESULT_BASE / f"{nodes}node"
    run_root = node_root / "_runs"
    direct = run_root / value
    if direct.is_dir():
        manifest_path, manifest = load_vllm_run_manifest(direct)
        return vllm_run_id(manifest, direct, nodes), direct, manifest_path, manifest

    matches: list[tuple[str, Path, Path, dict[str, Any]]] = []
    for manifest_path in sorted(run_root.glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        _, manifest = load_vllm_run_manifest(run_dir)
        resolved_id = vllm_run_id(manifest, run_dir, nodes)
        if resolved_id == value:
            matches.append((resolved_id, run_dir, manifest_path, manifest))
    if not matches:
        raise FileNotFoundError(f"vLLM run ID {value!r} not found under {run_root}")
    if len(matches) > 1:
        paths = ", ".join(str(match[1]) for match in matches)
        raise RuntimeError(f"vLLM run ID {value!r} is ambiguous: {paths}")
    return matches[0]


def resolve_result_path(value: object, run_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"vLLM run result has no {label}")
    path = Path(value)
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def collect_vllm(
    value: str,
    nodes: int,
    workloads: list[str],
    slo_target_ms: float,
) -> tuple[str, list[dict[str, Any]]]:
    run_id, run_dir, manifest_path, run_manifest = resolve_vllm_run(value, nodes)
    requested_workloads = set(workloads)

    measured: list[dict[str, Any]] = []
    measured_cases: set[str] = set()
    print(f"vLLM run: {run_id}")
    print(f"vLLM run directory: {run_dir}")
    print(f"vLLM run manifest: {manifest_path}")
    for result in run_manifest["results"]:
        if not isinstance(result, dict):
            continue
        case_name = str(result.get("case_name") or "")
        parsed = vllm.workload_and_baseline(case_name, nodes)
        if parsed is None:
            print(f"[vllm skip] {case_name or '<unnamed>'}: not a Figure 12 case")
            continue
        workload, baseline = parsed
        if requested_workloads and workload not in requested_workloads:
            continue
        if result.get("status") != "ok" or result.get("exit_code") != 0:
            print(f"[vllm skip] {case_name}: status={result.get('status')}")
            continue

        case_dir = resolve_result_path(result.get("case_dir"), run_dir, "case_dir")
        case_manifest_path = case_dir / "case_manifest.json"
        try:
            case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"vLLM case manifest not found: {case_manifest_path}"
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid vLLM case manifest: {case_manifest_path}: {error}"
            ) from error
        raw_case = case_manifest.get("case")
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"vLLM case manifest has no case object: {case_manifest_path}"
            )
        case_row = {
            **raw_case,
            "name": case_name,
            "workload": workload,
            "baseline": baseline,
        }
        print(f"[vllm] parse {case_name}")
        row = vllm.measure_case(
            case_row,
            case_manifest_path,
            case_manifest,
            slo_target_ms,
        )
        measured.append(
            {
                **row,
                "system": "vLLM",
                "plot_series": str(row["baseline"]),
                "plot_rate": float(row["request_rate"]),
                "runtime_s": row["benchmark_runtime_s"],
                "source_path": Path(row["benchmark_path"]),
            }
        )
        measured_cases.add(case_name)

    node_root = vllm.RESULT_BASE / f"{nodes}node"
    _, case_rows, recovered_manifests = vllm.collect_run(
        node_root,
        nodes,
        run_id,
        workloads,
    )
    for case_row in case_rows:
        case_name = str(case_row["name"])
        if case_name in measured_cases or case_name not in recovered_manifests:
            continue
        case_manifest_path, case_manifest = recovered_manifests[case_name]
        print(f"[vllm] recover completed benchmark outputs for {case_name}")
        row = vllm.measure_case(
            case_row,
            case_manifest_path,
            case_manifest,
            slo_target_ms,
        )
        measured.append(
            {
                **row,
                "system": "vLLM",
                "plot_series": str(row["baseline"]),
                "plot_rate": float(row["request_rate"]),
                "runtime_s": row["benchmark_runtime_s"],
                "source_path": Path(row["benchmark_path"]),
            }
        )
        measured_cases.add(case_name)
    if not measured:
        raise RuntimeError(f"no successful vLLM points found for run {run_id}")
    return run_id, measured


def display_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(FIG12_DIR.parent.resolve()))
    except ValueError:
        return str(path.resolve())


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "system",
        "workload",
        "series",
        "request_rate",
        "expected_requests",
        "successful_requests",
        "failed_requests",
        "slo_target_ms",
        "slo_attainment_pct",
        "mean_tpot_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "runtime_s",
        "achieved_rate_rps",
        "goodput_rps",
        "source_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "system": row["system"],
                    "workload": row["workload"],
                    "series": SERIES_STYLES[str(row["plot_series"])].label,
                    "request_rate": f"{row['plot_rate']:g}",
                    "expected_requests": row["expected_requests"],
                    "successful_requests": row["successful_requests"],
                    "failed_requests": row["failed_requests"],
                    "slo_target_ms": f"{row['slo_target_ms']:.3f}",
                    "slo_attainment_pct": f"{row['slo_attainment_pct']:.6f}",
                    "mean_tpot_ms": f"{row['mean_tpot_ms']:.6f}",
                    "p50_tpot_ms": f"{row['p50_tpot_ms']:.6f}",
                    "p90_tpot_ms": f"{row['p90_tpot_ms']:.6f}",
                    "p95_tpot_ms": f"{row['p95_tpot_ms']:.6f}",
                    "p99_tpot_ms": f"{row['p99_tpot_ms']:.6f}",
                    "runtime_s": (
                        "" if row["runtime_s"] is None else f"{row['runtime_s']:.6f}"
                    ),
                    "achieved_rate_rps": (
                        ""
                        if row["achieved_rate_rps"] is None
                        else f"{row['achieved_rate_rps']:.6f}"
                    ),
                    "goodput_rps": (
                        ""
                        if row["goodput_rps"] is None
                        else f"{row['goodput_rps']:.6f}"
                    ),
                    "source_path": display_source(Path(row["source_path"])),
                }
            )


def default_output_dir(nodes: int, nano_id: str | None, vllm_id: str | None) -> Path:
    parts: list[str] = []
    if nano_id:
        parts.append(f"nano-{nano_id}")
    if vllm_id:
        parts.append(f"vllm-{vllm_id}")
    return FIG12_DIR / "results" / "e2e" / "plots" / f"{nodes}node" / "__".join(parts)


def main() -> int:
    args = parse_args()
    if args.slo_target <= 0:
        raise ValueError("--slo-target must be positive")

    rows: list[dict[str, Any]] = []
    nano_id: str | None = None
    vllm_id: str | None = None
    if args.nano_run:
        nano_id, nano_rows = collect_nano(
            args.nano_run,
            args.nodes,
            args.workload,
            args.slo_target,
        )
        rows.extend(nano_rows)
    if args.vllm_run:
        vllm_id, vllm_rows = collect_vllm(
            args.vllm_run,
            args.nodes,
            args.workload,
            args.slo_target,
        )
        rows.extend(vllm_rows)

    workload_order = {workload.slug: index for index, workload in enumerate(WORKLOADS)}
    series_order = {series_id: index for index, series_id in enumerate(SERIES_ORDER)}
    rows.sort(
        key=lambda row: (
            workload_order[str(row["workload"])],
            series_order[str(row["plot_series"])],
            float(row["plot_rate"]),
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["workload"])].append(row)

    output_dir = args.output_dir or default_output_dir(args.nodes, nano_id, vllm_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "fig12_e2e_run"
    metrics_path = output_dir / "fig12_e2e_run_metrics.tsv"
    write_metrics(metrics_path, rows)
    plot_matrix(
        grouped,
        {slug: WORKLOAD_BY_SLUG[slug].label for slug in grouped},
        SERIES_ORDER,
        SERIES_STYLES,
        output_stem,
        args.slo_target,
        rate_key="plot_rate",
        series_key="plot_series",
    )

    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

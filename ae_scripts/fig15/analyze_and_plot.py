#!/usr/bin/env python3
"""Discover Fig. 15 profiler traces, parse them, and draw the final figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASETS = ("short", "issue01", "issue05")
NANO_CASE_NAMES = {
    "short": "sharegpt4o",
    "issue01": "issue01",
    "issue05": "issue05",
}
DATASET_LABELS = {
    "short": "ShareGPT4o",
    "issue01": "Issue1%",
    "issue05": "Issue5%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nano-run",
        type=Path,
        required=True,
        help="Completed run directory produced by run_nano_profiling.py",
    )
    parser.add_argument(
        "--vllm-run",
        type=Path,
        required=True,
        help="Completed run directory produced by run_vllm_profiling.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Parsed CSV root (default: fig15/results)",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=SCRIPT_DIR / "fig15_latency_breakdown",
        help=(
            "Figure base path; PDF and PNG are written "
            "(default: fig15/fig15_latency_breakdown)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate both runs and print the automatically discovered plan",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_run_manifest(
    raw_run_dir: Path,
    expected_system: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = raw_run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    manifest = load_json(run_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(
            f"Run is not completed: {run_dir} "
            f"(status={manifest.get('status')!r})"
        )
    if manifest.get("system") != expected_system:
        raise ValueError(
            f"Expected a {expected_system} run at {run_dir}; "
            f"manifest reports {manifest.get('system')!r}"
        )
    return run_dir, manifest


def trace_files(trace_dir: Path) -> tuple[Path, ...]:
    if not trace_dir.is_dir():
        raise ValueError(f"Trace directory does not exist: {trace_dir}")
    traces = tuple(
        sorted(
            path
            for path in trace_dir.rglob("*")
            if path.is_file()
            and path.name.endswith((".pt.trace.json", ".pt.trace.json.gz"))
            and "merged" not in path.name
        )
    )
    return traces


def positive_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Expected a positive integer for {description}, got {value!r}"
        )
    return value


def nonnegative_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"Expected a nonnegative integer for {description}, got {value!r}"
        )
    return value


def discover_nano(
    run_dir: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    datasets: tuple[str, ...],
) -> tuple[list[str], dict[str, Path]]:
    topology = manifest.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("NanoDeploy manifest is missing topology metadata")
    expected_ranks = positive_int(
        topology.get("attention_dp"), "NanoDeploy attention_dp"
    ) * positive_int(topology.get("attention_sp"), "NanoDeploy attention_sp")
    recorded_counts = manifest.get("trace_counts")
    if not isinstance(recorded_counts, dict):
        raise ValueError("NanoDeploy manifest is missing trace_counts")

    command = [
        sys.executable,
        str(SCRIPT_DIR / "parse_nano_layer_breakdown.py"),
    ]
    trace_dirs: dict[str, Path] = {}
    for dataset in datasets:
        recorded = positive_int(
            recorded_counts.get(dataset),
            f"NanoDeploy trace_counts[{dataset!r}]",
        )
        if recorded != expected_ranks:
            raise ValueError(
                f"NanoDeploy {dataset} records {recorded} traces; "
                f"expected {expected_ranks}"
            )
        trace_dir = run_dir / "traces" / dataset
        actual = len(trace_files(trace_dir))
        if actual != expected_ranks:
            raise ValueError(
                f"NanoDeploy {dataset} has {actual} trace files in {trace_dir}; "
                f"expected {expected_ranks}"
            )
        trace_dirs[dataset] = trace_dir
        command.extend(
            ("--case", f"{NANO_CASE_NAMES[dataset]}={trace_dir}")
        )
    command.extend(
        (
            "--output-dir",
            str(output_dir / "nano_layer_breakdown"),
            "--expected-ranks",
            str(expected_ranks),
        )
    )
    return command, trace_dirs


def index_vllm_case_manifests(run_dir: Path) -> dict[str, Path]:
    case_manifests: dict[str, Path] = {}
    for path in sorted((run_dir / "artifacts").rglob("case_manifest.json")):
        payload = load_json(path)
        name = payload.get("case_name")
        if not isinstance(name, str) or not name:
            case = payload.get("case")
            name = case.get("name") if isinstance(case, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"Cannot identify vLLM case in {path}")
        if name in case_manifests:
            raise ValueError(
                f"Duplicate vLLM case manifest for {name}: "
                f"{case_manifests[name]} and {path}"
            )
        case_manifests[name] = path
    if not case_manifests:
        raise ValueError(f"No vLLM case manifests found below {run_dir / 'artifacts'}")
    return case_manifests


def discover_vllm(
    run_dir: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    datasets: tuple[str, ...],
    vllm_configs: tuple[tuple[str, str, str], ...],
) -> tuple[list[tuple[str, list[str]]], dict[tuple[str, str], Path]]:
    cluster = manifest.get("cluster")
    if not isinstance(cluster, dict):
        raise ValueError("vLLM manifest is missing cluster metadata")
    expected_traces = positive_int(
        cluster.get("expected_trace_count_per_case"),
        "vLLM expected_trace_count_per_case",
    )
    profile = manifest.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("vLLM manifest is missing profile metadata")
    expected_iterations = positive_int(
        profile.get("max_iterations"), "vLLM profile.max_iterations"
    )

    cases = manifest.get("cases")
    results = manifest.get("results")
    if not isinstance(cases, list) or not all(isinstance(row, dict) for row in cases):
        raise ValueError("vLLM manifest is missing case metadata")
    if not isinstance(results, list) or not all(
        isinstance(row, dict) for row in results
    ):
        raise ValueError("vLLM manifest is missing result metadata")

    cases_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case in cases:
        key = (
            str(case.get("dataset")),
            str(case.get("strategy")),
            str(case.get("dispatch_policy")),
        )
        if key in cases_by_key:
            raise ValueError(f"Duplicate vLLM case metadata for {key}")
        cases_by_key[key] = case

    results_by_name: dict[str, dict[str, Any]] = {}
    for result in results:
        name = result.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("A vLLM result is missing its case name")
        if name in results_by_name:
            raise ValueError(f"Duplicate vLLM result for {name}")
        results_by_name[name] = result

    local_manifests = index_vllm_case_manifests(run_dir)
    trace_dirs: dict[tuple[str, str], Path] = {}
    commands: list[tuple[str, list[str]]] = []
    for dataset in datasets:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "parse_vllm_iteration_breakdown.py"),
        ]
        for strategy, policy, case_label in vllm_configs:
            key = (dataset, strategy, policy)
            case = cases_by_key.get(key)
            if case is None:
                raise ValueError(
                    "vLLM run is missing case "
                    f"dataset={dataset}, strategy={strategy}, policy={policy}"
                )
            name = case.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"vLLM case {key} is missing its name")
            result = results_by_name.get(name)
            if result is None:
                raise ValueError(f"vLLM run is missing the result for {name}")
            if result.get("status") != "ok":
                raise ValueError(
                    f"vLLM case {name} did not finish successfully: "
                    f"status={result.get('status')!r}"
                )
            recorded = positive_int(
                result.get("trace_count"), f"vLLM result trace_count for {name}"
            )
            if recorded != expected_traces:
                raise ValueError(
                    f"vLLM case {name} records {recorded} traces; "
                    f"expected {expected_traces}"
                )
            expected_requests = positive_int(
                case.get("expected_requests"),
                f"vLLM expected_requests for {name}",
            )
            successful_requests = positive_int(
                result.get("successful_requests"),
                f"vLLM successful_requests for {name}",
            )
            failed_requests = nonnegative_int(
                result.get("failed_requests"),
                f"vLLM failed_requests for {name}",
            )
            if successful_requests != expected_requests or failed_requests != 0:
                raise ValueError(
                    f"vLLM case {name} request validation failed: "
                    f"successful={successful_requests}, failed={failed_requests}, "
                    f"expected={expected_requests}"
                )
            case_manifest_path = local_manifests.get(name)
            if case_manifest_path is None:
                raise ValueError(f"Cannot find the local case manifest for {name}")
            trace_dir = case_manifest_path.parent / "benchmark" / "torch_profiler"
            actual = len(trace_files(trace_dir))
            if actual != expected_traces:
                raise ValueError(
                    f"vLLM case {name} has {actual} trace files in {trace_dir}; "
                    f"expected {expected_traces}"
                )
            trace_dirs[(dataset, case_label)] = trace_dir
            command.extend(("--case", f"{case_label}={trace_dir}"))
        command.extend(
            (
                "--output-dir",
                str(output_dir / f"{dataset}_iteration_breakdown"),
                "--expected-traces",
                str(expected_traces),
                "--expected-iterations",
                str(expected_iterations),
            )
        )
        commands.append((f"vLLM/{dataset}", command))
    return commands, trace_dirs


def plot_command(
    output_dir: Path,
    figure_output: Path,
    datasets: tuple[str, ...],
    expected_ranks: int,
) -> list[str]:
    nano_dir = output_dir / "nano_layer_breakdown"
    command = [sys.executable, str(SCRIPT_DIR / "plot_latency_breakdown.py")]
    for dataset in datasets:
        label = DATASET_LABELS[dataset]
        nano_csv = (
            nano_dir / NANO_CASE_NAMES[dataset] / "plot_rank_summary.csv"
        )
        vllm_csv = (
            output_dir
            / f"{dataset}_iteration_breakdown"
            / "layer_rank_summary.csv"
        )
        command.extend(
            (
                "--input",
                f"{label}={nano_csv}",
                "--input",
                f"{label}={vllm_csv}",
            )
        )
    command.extend(
        (
            "--output",
            str(figure_output),
            "--expected-ranks",
            str(expected_ranks),
        )
    )
    return command


def figure_paths(figure_output: Path) -> tuple[Path, Path]:
    base = (
        figure_output.with_suffix("")
        if figure_output.suffix.lower() in {".pdf", ".png"}
        else figure_output
    )
    return Path(f"{base}.pdf"), Path(f"{base}.png")


def print_discovery(
    nano_trace_dirs: dict[str, Path],
    vllm_trace_dirs: dict[tuple[str, str], Path],
    datasets: tuple[str, ...],
    vllm_configs: tuple[tuple[str, str, str], ...],
) -> None:
    print("Validated profiler inputs:", flush=True)
    for dataset in datasets:
        print(f"  NanoDeploy/{dataset}: {nano_trace_dirs[dataset]}", flush=True)
        for _, _, case_label in vllm_configs:
            print(
                f"  vLLM/{dataset}/{case_label}: "
                f"{vllm_trace_dirs[(dataset, case_label)]}",
                flush=True,
            )


def execute(label: str, command: list[str], dry_run: bool) -> None:
    print(f"\n===== {label} =====", flush=True)
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def selected_datasets(
    nano_manifest: dict[str, Any],
    vllm_manifest: dict[str, Any],
) -> tuple[str, ...]:
    nano_raw = nano_manifest.get("datasets")
    if not isinstance(nano_raw, list) or not all(
        isinstance(dataset, str) for dataset in nano_raw
    ):
        raise ValueError("NanoDeploy manifest is missing the datasets list")
    nano_selected = set(nano_raw)

    cases = vllm_manifest.get("cases")
    if not isinstance(cases, list) or not all(
        isinstance(case, dict) for case in cases
    ):
        raise ValueError("vLLM manifest is missing case metadata")
    vllm_selected = {str(case.get("dataset")) for case in cases}
    if nano_selected != vllm_selected:
        raise ValueError(
            "NanoDeploy and vLLM runs contain different datasets: "
            f"nano={sorted(nano_selected)}, vllm={sorted(vllm_selected)}"
        )
    unsupported = nano_selected - set(DATASETS)
    if unsupported:
        raise ValueError(f"Unsupported datasets: {sorted(unsupported)}")
    datasets = tuple(dataset for dataset in DATASETS if dataset in nano_selected)
    if not datasets:
        raise ValueError("The selected runs do not contain a Fig. 15 dataset")
    return datasets


def selected_vllm_configs(
    manifest: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not all(
        isinstance(case, dict) for case in cases
    ):
        raise ValueError("vLLM manifest is missing case metadata")
    selected: list[tuple[str, str, str]] = []
    logical_configs = (
        ("dp", "least_batch", "dp32_least_batch"),
        ("dp", "least_cache", "dp32_least_cache"),
        ("dcp", "least_batch", "dp4dcp8_least_batch"),
    )
    for mode, policy, label in logical_configs:
        strategies = {
            str(case.get("strategy"))
            for case in cases
            if case.get("mode") == mode and case.get("dispatch_policy") == policy
        }
        if not strategies:
            continue
        if len(strategies) != 1:
            raise ValueError(
                f"vLLM {mode}/{policy} cases use multiple strategies: "
                f"{sorted(strategies)}"
            )
        selected.append((strategies.pop(), policy, label))
    if not selected:
        raise ValueError("The vLLM run contains no supported Fig. 15 cases")
    return tuple(selected)


def manifest_rank_counts(
    nano_manifest: dict[str, Any],
    vllm_manifest: dict[str, Any],
) -> tuple[int, int]:
    topology = nano_manifest.get("topology")
    cluster = vllm_manifest.get("cluster")
    if not isinstance(topology, dict) or not isinstance(cluster, dict):
        raise ValueError("Run manifests are missing topology metadata")
    nano_ranks = positive_int(
        topology.get("attention_dp"), "NanoDeploy attention_dp"
    ) * positive_int(topology.get("attention_sp"), "NanoDeploy attention_sp")
    vllm_ranks = positive_int(
        cluster.get("expected_trace_count_per_case"),
        "vLLM expected_trace_count_per_case",
    )
    return nano_ranks, vllm_ranks


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    figure_output = args.figure_output.expanduser().resolve()
    try:
        nano_run, nano_manifest = load_run_manifest(args.nano_run, "NanoDeploy")
        vllm_run, vllm_manifest = load_run_manifest(args.vllm_run, "vLLM")
        datasets = selected_datasets(nano_manifest, vllm_manifest)
        vllm_configs = selected_vllm_configs(vllm_manifest)
        nano_ranks, vllm_ranks = manifest_rank_counts(
            nano_manifest, vllm_manifest
        )
        if nano_ranks != vllm_ranks:
            raise ValueError(
                "NanoDeploy and vLLM runs use different rank counts: "
                f"nano={nano_ranks}, vllm={vllm_ranks}"
            )
        nano_command, nano_trace_dirs = discover_nano(
            nano_run, nano_manifest, output_dir, datasets
        )
        vllm_commands, vllm_trace_dirs = discover_vllm(
            vllm_run, vllm_manifest, output_dir, datasets, vllm_configs
        )
        drawing_command = plot_command(
            output_dir, figure_output, datasets, nano_ranks
        )
        print_discovery(
            nano_trace_dirs, vllm_trace_dirs, datasets, vllm_configs
        )
        execute("Parse NanoDeploy", nano_command, args.dry_run)
        for label, command in vllm_commands:
            execute(f"Parse {label}", command, args.dry_run)
        execute("Draw Figure 15", drawing_command, args.dry_run)
        pdf_path, png_path = figure_paths(figure_output)
        if not args.dry_run:
            missing_outputs = [
                path for path in (pdf_path, png_path) if not path.is_file()
            ]
            if missing_outputs:
                raise ValueError(
                    "Plot command did not produce: "
                    + ", ".join(str(path) for path in missing_outputs)
                )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Fig. 15 analysis failed: {exc}") from exc

    if args.dry_run:
        print("\nDry run completed; no CSV or figure was written.", flush=True)
    else:
        print(
            f"\nFigure 15 completed: {pdf_path} and {png_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare per-request TPOT metrics for vLLM non-DBO, DBO, and Nano."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from ae_utils.paths import require_path

import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover - optional accelerator
    orjson = None


DEFAULT_NON_DBO_ROOT = Path(
    os.environ.get(
        "VLLM_DBO_NON_DBO_ROOT",
        "/vllm/offline_bench/manual_multinode",
    )
)
DEFAULT_DBO_ROOT = Path(
    os.environ.get(
        "VLLM_DBO_ARTIFACT_ROOT",
        str(SCRIPT_DIR / "results" / "dbo"),
    )
)
DEFAULT_ISSUE1_DBO_ROOT = Path(
    os.environ.get(
        "VLLM_DBO_ISSUE1_ARTIFACT_ROOT",
        str(DEFAULT_DBO_ROOT),
    )
)
DEFAULT_NANO_RUN_DIR = Path(
    os.environ.get("FIG20_NANO_RUN_DIR") or require_path("AE_FIG20_NANO_RUN_DIR")
)
DEFAULT_NANO_STAGE_GLOB = "longshort_mixed60k_deepseek_v3*"
DEFAULT_ISSUE1_NANO_STAGE_GLOB = "longshort_issue001_deepseek_v3*"
DEFAULT_RATES = (40.0, 60.0, 80.0, 100.0, 120.0)
DEFAULT_ISSUE1_RATES = (10.0, 20.0, 30.0, 40.0, 50.0)
DEFAULT_NON_DBO_GLOB = "dp32*-rate*-dur600"
DEFAULT_DBO_GLOB = "dp32-dispatch_least_batch-*-rate*-dur600"
DEFAULT_SLO_TARGET_MS = 50.0
DEFAULT_MAX_FAILURE_RATIO = 0.01
DISPATCH_TAG_PATTERN = re.compile(r"(?:^|-)dispatch_(?P<policy>[^-]+)")

CONFIG_FIELDS = (
    "model",
    "dataset",
    "strategy",
    "dispatch_policy",
    "max_num_seqs",
    "gpu_memory_utilization",
    "max_requests",
    "warmup_requests",
    "max_model_len",
    "data_parallel_size",
    "data_parallel_size_local",
    "tensor_parallel_size",
    "decode_context_parallel_size",
    "enable_expert_parallel",
    "attention_backend",
    "all2all_backend",
)


@dataclass(frozen=True)
class DatasetPlotDefaults:
    label: str
    dbo_root: Path
    nano_stage_glob: str
    rates: tuple[float, ...]
    output_stem: str


DATASET_PLOT_DEFAULTS: dict[str, DatasetPlotDefaults] = {
    "short_random": DatasetPlotDefaults(
        label="ShareGPT4o (all-short)",
        dbo_root=DEFAULT_DBO_ROOT,
        nano_stage_glob=DEFAULT_NANO_STAGE_GLOB,
        rates=DEFAULT_RATES,
        output_stem="dbo_tpot_comparison",
    ),
    "issue01_random": DatasetPlotDefaults(
        label="Issue1%",
        dbo_root=DEFAULT_ISSUE1_DBO_ROOT,
        nano_stage_glob=DEFAULT_ISSUE1_NANO_STAGE_GLOB,
        rates=DEFAULT_ISSUE1_RATES,
        output_stem="dbo_tpot_comparison_issue1",
    ),
}


@dataclass(frozen=True)
class Candidate:
    mode: str
    request_rate: float
    case_dir: Path
    scenario: str
    timestamp: str
    finished_at: str
    policy_source: str
    dbo_decode_token_threshold: int | None
    config: Mapping[str, Any]


@dataclass(frozen=True)
class NanoCandidate:
    request_rate: float
    stage_dir: Path
    summary_path: Path
    json_path: Path
    log_path: Path
    timestamp: str
    status: str
    expected_total_requests: int | None
    strategy: str
    routing: str


@dataclass(frozen=True)
class ResultPoint:
    mode: str
    request_rate: float
    tpot_mean_ms: float
    tpot_p99_ms: float
    slo_attainment_percent: float
    slo_success_count: int
    successful_requests: int
    failed_requests: int
    total_requests: int
    failure_ratio: float
    scenario: str
    timestamp: str
    policy_source: str
    dbo_decode_token_threshold: int | None
    requests_path: Path
    summary_path: Path
    config: Mapping[str, Any]


def parse_request_rates(raw: str) -> tuple[float, ...]:
    rates: list[float] = []
    for part in raw.replace(",", " ").split():
        value = float(part)
        if not math.isfinite(value) or value <= 0:
            raise argparse.ArgumentTypeError(
                "request rates must be finite positive numbers"
            )
        rates.append(value)
    if not rates:
        raise argparse.ArgumentTypeError("at least one request rate is required")
    return tuple(rates)


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return value


def failure_ratio(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0 or value > 1:
        raise argparse.ArgumentTypeError("failure ratio must be between 0 and 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse vLLM requests.jsonl files and compare TPOT Mean, TPOT P99, "
            "and SLO attainment for non-DBO, DBO, and Nano."
        )
    )
    parser.add_argument(
        "--non-dbo-root",
        type=Path,
        default=DEFAULT_NON_DBO_ROOT,
        help=f"non-DBO artifact root (default: {DEFAULT_NON_DBO_ROOT})",
    )
    parser.add_argument(
        "--dbo-root",
        type=Path,
        default=None,
        help="DBO artifact root (default: selected from --dataset)",
    )
    parser.add_argument(
        "--nano-run-dir",
        type=Path,
        default=DEFAULT_NANO_RUN_DIR,
        help=f"Nano bench run directory (default: {DEFAULT_NANO_RUN_DIR})",
    )
    parser.add_argument(
        "--nano-stage-glob",
        default=None,
        help=(
            "fnmatch pattern for Nano mixed-workload stage directories "
            "(default: selected from --dataset)"
        ),
    )
    parser.add_argument("--model", default="DPSK", help="artifact model directory")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PLOT_DEFAULTS),
        default="short_random",
        help="dataset key (default: short_random)",
    )
    parser.add_argument("--strategy", default="dp32", help="strategy key")
    parser.add_argument(
        "--non-dbo-scenario-glob",
        default=DEFAULT_NON_DBO_GLOB,
        help=(
            "fnmatch pattern for non-DBO scenarios; explicit least_batch and "
            "scenarios without a dispatch tag are accepted"
        ),
    )
    parser.add_argument(
        "--dbo-scenario-glob",
        default=DEFAULT_DBO_GLOB,
        help="fnmatch pattern for DBO scenario directory names",
    )
    parser.add_argument(
        "--request-rates",
        type=parse_request_rates,
        default=None,
        help="comma- or space-separated rates (default: selected from --dataset)",
    )
    parser.add_argument(
        "--slo-target-ms",
        type=positive_float,
        default=DEFAULT_SLO_TARGET_MS,
        help=f"per-request TPOT SLO in ms (default: {DEFAULT_SLO_TARGET_MS:g})",
    )
    parser.add_argument(
        "--max-failure-ratio",
        type=failure_ratio,
        default=DEFAULT_MAX_FAILURE_RATIO,
        help=(
            "maximum failed-request fraction accepted for a point "
            f"(default: {DEFAULT_MAX_FAILURE_RATIO:g})"
        ),
    )
    parser.add_argument(
        "--selection",
        choices=("latest", "earliest"),
        default="latest",
        help="which valid rerun to select for each rate",
    )
    parser.add_argument(
        "--allow-missing-rates",
        action="store_true",
        help="plot common rates instead of requiring every requested rate",
    )
    parser.add_argument(
        "--strict-config-match",
        action="store_true",
        help="fail instead of warning when non-DBO and DBO settings differ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output base path; .pdf and .png are appended (dataset-specific default)",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=None,
        help="TSV path (default: <output>.tsv)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="write the comparison TSV without rendering diagnostic plots",
    )
    parser.add_argument("--dpi", type=int, default=250, help="PNG resolution")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def parse_json_line(raw: bytes) -> Mapping[str, Any]:
    if orjson is not None:
        payload = orjson.loads(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("request record is not a JSON object")
    return payload


def to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_dataset_dir(root: Path, model: str, dataset: str) -> Path:
    root = root.expanduser().resolve()
    candidates = (root / model / dataset, root / dataset, root)
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == dataset:
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"cannot find dataset directory {dataset!r} below {root}; tried: {tried}"
    )


def extra_argv_tokens(case: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("shared_cli_args", "frontend_extra_args", "headless_extra_args"):
        value = case.get(key)
        if isinstance(value, list):
            tokens.extend(str(token) for token in value)
    return tokens


def parse_dbo_configuration(manifest: Mapping[str, Any]) -> tuple[bool, int | None]:
    case = manifest.get("case")
    if not isinstance(case, dict):
        return False, None
    tokens = extra_argv_tokens(case)
    enabled = "--enable-dbo" in tokens and "--no-enable-dbo" not in tokens
    threshold: int | None = None
    for index, token in enumerate(tokens):
        if token.startswith("--dbo-decode-token-threshold="):
            threshold = int(token.split("=", 1)[1])
        elif token == "--dbo-decode-token-threshold" and index + 1 < len(tokens):
            threshold = int(tokens[index + 1])
    return enabled, threshold


def scenario_dispatch_policy(scenario: str) -> str | None:
    match = DISPATCH_TAG_PATTERN.search(scenario)
    return match.group("policy") if match is not None else None


def canonical_dispatch_policy(policy: Any) -> str | None:
    if not isinstance(policy, str) or not policy:
        return None
    if policy in {"least_batch", "waiting_x4_plus_running"}:
        return "least_batch"
    return policy


def accepted_policy(
    *, mode: str, scenario: str, case: Mapping[str, Any]
) -> tuple[bool, str]:
    tagged_policy = scenario_dispatch_policy(scenario)
    manifest_policy = case.get("dispatch_policy")
    if mode == "non-DBO":
        if tagged_policy is None:
            return True, "untagged/default"
        if canonical_dispatch_policy(tagged_policy) == "least_batch":
            return True, "explicit least_batch"
        return False, f"explicit {tagged_policy}"

    effective = canonical_dispatch_policy(tagged_policy or manifest_policy)
    if effective == "least_batch":
        return True, "least_batch"
    return False, str(tagged_policy or manifest_policy or "unknown")


def policy_priority(candidate: Candidate) -> int:
    if candidate.mode == "non-DBO" and candidate.policy_source == "explicit least_batch":
        return 0
    return 1


def extract_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    case = manifest.get("case")
    strategy = manifest.get("strategy")
    case = case if isinstance(case, dict) else {}
    strategy = strategy if isinstance(strategy, dict) else {}
    config = {key: case.get(key) for key in CONFIG_FIELDS}
    for key in CONFIG_FIELDS:
        if config[key] is None and key in strategy:
            config[key] = strategy.get(key)
    config["dispatch_policy"] = canonical_dispatch_policy(
        config.get("dispatch_policy")
    )
    return config


def candidate_from_case(
    mode: str,
    case_dir: Path,
    *,
    expected_dbo: bool,
    dataset: str,
    strategy: str,
) -> Candidate:
    manifest_path = case_dir / "case_manifest.json"
    manifest = load_json(manifest_path)
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"{manifest_path}: missing case object")
    if case.get("dataset") != dataset or case.get("strategy") != strategy:
        raise ValueError(f"{manifest_path}: dataset or strategy does not match")

    dbo_enabled, threshold = parse_dbo_configuration(manifest)
    if dbo_enabled != expected_dbo:
        expected = "enabled" if expected_dbo else "disabled"
        raise ValueError(f"{manifest_path}: expected DBO {expected}")

    scenario = case_dir.parent.name
    policy_ok, policy_source = accepted_policy(
        mode=mode, scenario=scenario, case=case
    )
    if not policy_ok:
        raise ValueError(
            f"{manifest_path}: rejected dispatch policy {policy_source!r}"
        )
    rate = to_float(case.get("request_rate"))
    if rate is None:
        raise ValueError(f"{manifest_path}: missing request_rate")

    return Candidate(
        mode=mode,
        request_rate=rate,
        case_dir=case_dir,
        scenario=scenario,
        timestamp=case_dir.name,
        finished_at=str(manifest.get("finished_at") or case_dir.name),
        policy_source=policy_source,
        dbo_decode_token_threshold=threshold,
        config=extract_config(manifest),
    )


def discover_candidates(
    *,
    mode: str,
    root: Path,
    model: str,
    dataset: str,
    strategy: str,
    scenario_glob: str,
    expected_dbo: bool,
) -> list[Candidate]:
    dataset_dir = resolve_dataset_dir(root, model, dataset)
    scenario_dirs = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and fnmatch.fnmatch(path.name, scenario_glob)
    )
    if not scenario_dirs:
        raise ValueError(
            f"{mode}: no scenario below {dataset_dir} matches {scenario_glob!r}"
        )

    candidates: list[Candidate] = []
    for scenario_dir in scenario_dirs:
        for case_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
            if not (case_dir / "case_manifest.json").is_file():
                continue
            if not (case_dir / "benchmark" / "requests.jsonl").is_file():
                continue
            if not (case_dir / "benchmark" / "summary.json").is_file():
                continue
            try:
                candidates.append(
                    candidate_from_case(
                        mode,
                        case_dir,
                        expected_dbo=expected_dbo,
                        dataset=dataset,
                        strategy=strategy,
                    )
                )
            except ValueError:
                continue
    if not candidates:
        raise ValueError(f"{mode}: no usable cases found below {dataset_dir}")
    return candidates


def resolve_nano_source_path(stage_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = stage_dir / path
    return path.resolve()


def nano_timestamp_from_path(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})(?:\.[a-z0-9]+)?$", str(path))
    if match is not None:
        return match.group(1)
    return f"mtime-{path.stat().st_mtime_ns:020d}"


def discover_nano_candidates(
    run_dir: Path,
    *,
    stage_glob: str,
) -> list[NanoCandidate]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Nano: run directory not found: {run_dir}")

    stage_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and fnmatch.fnmatch(path.name, stage_glob)
    )
    if not stage_dirs:
        raise ValueError(
            f"Nano: no stage below {run_dir} matches {stage_glob!r}"
        )

    candidates: list[NanoCandidate] = []
    for stage_dir in stage_dirs:
        summary_path = stage_dir / "sweep_summary.tsv"
        if not summary_path.is_file():
            continue
        try:
            handle = summary_path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(f"cannot read {summary_path}: {exc}") from exc
        with handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                strategy = (row.get("strategy") or "").strip()
                if not strategy.lower().startswith("dp4sp8"):
                    continue
                rate = to_float(row.get("rate"))
                json_file = (row.get("json_file") or "").strip()
                log_file = (row.get("log_file") or "").strip()
                if rate is None or not json_file or not log_file:
                    continue
                json_path = resolve_nano_source_path(stage_dir, json_file)
                log_path = resolve_nano_source_path(stage_dir, log_file)
                if not json_path.is_file() or not log_path.is_file():
                    continue
                candidates.append(
                    NanoCandidate(
                        request_rate=rate,
                        stage_dir=stage_dir.resolve(),
                        summary_path=summary_path.resolve(),
                        json_path=json_path,
                        log_path=log_path,
                        timestamp=nano_timestamp_from_path(json_path),
                        status=(row.get("status") or "").strip(),
                        expected_total_requests=to_int(row.get("n_reqs")),
                        strategy=strategy,
                        routing=(row.get("routing") or "").strip(),
                    )
                )
    if not candidates:
        raise ValueError(f"Nano: no usable dp4sp8 cases found below {run_dir}")
    return candidates


def parse_tpot_values(requests_path: Path) -> np.ndarray:
    values: list[float] = []
    with requests_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = parse_json_line(raw_line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{requests_path}: invalid JSON at line {line_number}: {exc}"
                ) from exc
            if row.get("is_error"):
                continue

            tpot_ms = to_float(row.get("tpot_by_e2e"))
            if tpot_ms is None:
                e2e_ms = to_float(row.get("e2e_ms"))
                output_tokens = to_float(row.get("actual_output_tokens"))
                if e2e_ms is not None and output_tokens and output_tokens > 0:
                    tpot_ms = e2e_ms / output_tokens
            if tpot_ms is None:
                raise ValueError(
                    f"{requests_path}: line {line_number} has no usable TPOT"
                )
            values.append(tpot_ms)
    if not values:
        raise ValueError(f"{requests_path}: no successful TPOT samples")
    return np.asarray(values, dtype=np.float64)


def parse_nano_tpot_values(json_path: Path) -> np.ndarray:
    """Use the exact queue-inclusive Nano normalized-latency rule from Fig. 12."""
    values: list[float] = []
    with json_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = parse_json_line(raw_line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{json_path}: invalid JSON at line {line_number}: {exc}"
                ) from exc

            itl_samples = row.get("itl_samples")
            if not isinstance(itl_samples, list) or not itl_samples:
                continue
            try:
                sample_values = [float(sample) for sample in itl_samples]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{json_path}: invalid itl_samples at line {line_number}"
                ) from exc
            queueing_time_ms = to_float(row.get("queueing_time_ms")) or 0.0
            tpot_ms = (sum(sample_values) + queueing_time_ms) / len(sample_values)
            if not math.isfinite(tpot_ms):
                raise ValueError(
                    f"{json_path}: non-finite Nano TPOT at line {line_number}"
                )
            values.append(tpot_ms)
    if not values:
        raise ValueError(f"{json_path}: no usable Nano TPOT samples")
    return np.asarray(values, dtype=np.float64)


def parse_candidate_metrics(
    candidate: Candidate,
    *,
    slo_target_ms: float,
    max_failure_ratio: float,
) -> ResultPoint:
    benchmark_dir = candidate.case_dir / "benchmark"
    requests_path = benchmark_dir / "requests.jsonl"
    summary_path = benchmark_dir / "summary.json"
    summary = load_json(summary_path)
    values = parse_tpot_values(requests_path)

    successful_requests = int(values.size)
    explicit_failed = to_int(summary.get("failed_requests")) or 0
    summary_total = to_int(summary.get("total_requests")) or 0
    expected_total = to_int(candidate.config.get("max_requests")) or 0
    total_requests = max(
        successful_requests + explicit_failed,
        summary_total,
        expected_total,
    )
    if total_requests <= 0:
        raise ValueError(f"{summary_path}: no completed requests")
    failed_requests = max(total_requests - successful_requests, explicit_failed)
    point_failure_ratio = failed_requests / total_requests
    if point_failure_ratio > max_failure_ratio:
        raise ValueError(
            f"{summary_path}: failure ratio {point_failure_ratio:.2%} exceeds "
            f"{max_failure_ratio:.2%}"
        )

    slo_success_count = int(np.count_nonzero(values <= slo_target_ms))
    return ResultPoint(
        mode=candidate.mode,
        request_rate=candidate.request_rate,
        tpot_mean_ms=float(np.mean(values)),
        tpot_p99_ms=float(np.percentile(values, 99)),
        slo_attainment_percent=100.0 * slo_success_count / total_requests,
        slo_success_count=slo_success_count,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_requests=total_requests,
        failure_ratio=point_failure_ratio,
        scenario=candidate.scenario,
        timestamp=candidate.timestamp,
        policy_source=candidate.policy_source,
        dbo_decode_token_threshold=candidate.dbo_decode_token_threshold,
        requests_path=requests_path.resolve(),
        summary_path=summary_path.resolve(),
        config=candidate.config,
    )


def parse_nano_candidate_metrics(
    candidate: NanoCandidate,
    *,
    slo_target_ms: float,
    max_failure_ratio: float,
) -> ResultPoint:
    values = parse_nano_tpot_values(candidate.json_path)
    successful_requests = int(values.size)
    total_requests = max(
        successful_requests,
        candidate.expected_total_requests or 0,
    )
    if total_requests <= 0:
        raise ValueError(f"{candidate.json_path}: no completed requests")
    if (
        candidate.expected_total_requests is not None
        and successful_requests > candidate.expected_total_requests + 1
    ):
        raise ValueError(
            f"{candidate.json_path}: successful request count "
            f"{successful_requests} exceeds expected "
            f"{candidate.expected_total_requests}"
        )
    failed_requests = total_requests - successful_requests
    point_failure_ratio = failed_requests / total_requests
    if point_failure_ratio > max_failure_ratio:
        raise ValueError(
            f"{candidate.json_path}: failure ratio {point_failure_ratio:.2%} "
            f"exceeds {max_failure_ratio:.2%}"
        )

    slo_success_count = int(np.count_nonzero(values <= slo_target_ms))
    return ResultPoint(
        mode="Nano",
        request_rate=candidate.request_rate,
        tpot_mean_ms=float(np.mean(values)),
        tpot_p99_ms=float(np.percentile(values, 99)),
        slo_attainment_percent=100.0 * slo_success_count / total_requests,
        slo_success_count=slo_success_count,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_requests=total_requests,
        failure_ratio=point_failure_ratio,
        scenario=candidate.stage_dir.name,
        timestamp=candidate.timestamp,
        policy_source=(
            f"{candidate.strategy}/{candidate.routing or 'unknown-routing'}"
        ),
        dbo_decode_token_threshold=None,
        requests_path=candidate.json_path,
        summary_path=candidate.summary_path,
        config={
            "strategy": candidate.strategy,
            "dispatch_policy": candidate.routing,
            "status": candidate.status,
            "log_path": str(candidate.log_path),
        },
    )


def rate_key(value: float) -> float:
    return round(value, 9)


def order_candidates(
    candidates: Iterable[Candidate], selection: str
) -> list[Candidate]:
    by_priority: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_priority.setdefault(policy_priority(candidate), []).append(candidate)
    ordered: list[Candidate] = []
    for priority in sorted(by_priority):
        ordered.extend(
            sorted(
                by_priority[priority],
                key=lambda item: (item.finished_at, item.timestamp),
                reverse=selection == "latest",
            )
        )
    return ordered


def select_points(
    candidates: Iterable[Candidate],
    *,
    selection: str,
    slo_target_ms: float,
    max_failure_ratio: float,
) -> dict[float, ResultPoint]:
    grouped: dict[float, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(rate_key(candidate.request_rate), []).append(candidate)

    selected: dict[float, ResultPoint] = {}
    for rate, rate_candidates in grouped.items():
        errors: list[str] = []
        for candidate in order_candidates(rate_candidates, selection):
            try:
                selected[rate] = parse_candidate_metrics(
                    candidate,
                    slo_target_ms=slo_target_ms,
                    max_failure_ratio=max_failure_ratio,
                )
                break
            except ValueError as exc:
                errors.append(str(exc))
        if rate not in selected and errors:
            print(
                f"[{rate:g} req/s] no valid candidate: {errors[0]}",
                file=sys.stderr,
            )
    return selected


def select_nano_points(
    candidates: Iterable[NanoCandidate],
    *,
    selection: str,
    slo_target_ms: float,
    max_failure_ratio: float,
) -> dict[float, ResultPoint]:
    grouped: dict[float, list[NanoCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(rate_key(candidate.request_rate), []).append(candidate)

    selected: dict[float, ResultPoint] = {}
    for rate, rate_candidates in grouped.items():
        errors: list[str] = []
        ordered = sorted(
            rate_candidates,
            key=lambda item: item.timestamp,
            reverse=selection == "latest",
        )
        for candidate in ordered:
            try:
                selected[rate] = parse_nano_candidate_metrics(
                    candidate,
                    slo_target_ms=slo_target_ms,
                    max_failure_ratio=max_failure_ratio,
                )
                break
            except ValueError as exc:
                errors.append(str(exc))
        if rate not in selected and errors:
            print(
                f"[Nano {rate:g} req/s] no valid candidate: {errors[0]}",
                file=sys.stderr,
            )
    return selected


def find_config_mismatches(
    non_dbo: Mapping[float, ResultPoint],
    dbo: Mapping[float, ResultPoint],
    rates: Iterable[float],
) -> list[str]:
    mismatches: set[str] = set()
    for rate in rates:
        baseline = non_dbo[rate]
        enabled = dbo[rate]
        for field in CONFIG_FIELDS:
            left = baseline.config.get(field)
            right = enabled.config.get(field)
            if left != right:
                mismatches.add(
                    f"{field}: non-DBO={left!r}, DBO={right!r}"
                )
    return sorted(mismatches)


def output_base(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() in {".pdf", ".png", ".tsv"}:
        return path.with_suffix("")
    return path


def write_tsv(
    path: Path,
    rates: Iterable[float],
    non_dbo: Mapping[float, ResultPoint],
    dbo: Mapping[float, ResultPoint],
    nano: Mapping[float, ResultPoint],
    *,
    slo_target_ms: float,
) -> None:
    fields = (
        "request_rate_rps",
        "slo_target_ms",
        "non_dbo_tpot_mean_ms",
        "dbo_tpot_mean_ms",
        "nano_tpot_mean_ms",
        "non_dbo_tpot_p99_ms",
        "dbo_tpot_p99_ms",
        "nano_tpot_p99_ms",
        "non_dbo_slo_attainment_percent",
        "dbo_slo_attainment_percent",
        "nano_slo_attainment_percent",
        "non_dbo_slo_success_count",
        "dbo_slo_success_count",
        "nano_slo_success_count",
        "non_dbo_total_requests",
        "dbo_total_requests",
        "nano_total_requests",
        "non_dbo_failed_requests",
        "dbo_failed_requests",
        "nano_failed_requests",
        "non_dbo_policy_source",
        "dbo_policy_source",
        "nano_policy_source",
        "non_dbo_scenario",
        "dbo_scenario",
        "nano_scenario",
        "dbo_decode_token_threshold",
        "non_dbo_requests_jsonl",
        "dbo_requests_jsonl",
        "nano_request_json",
        "nano_log_file",
        "nano_sweep_summary",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rate in rates:
            baseline = non_dbo[rate]
            enabled = dbo[rate]
            nano_point = nano[rate]
            writer.writerow(
                {
                    "request_rate_rps": f"{rate:g}",
                    "slo_target_ms": f"{slo_target_ms:g}",
                    "non_dbo_tpot_mean_ms": f"{baseline.tpot_mean_ms:.9g}",
                    "dbo_tpot_mean_ms": f"{enabled.tpot_mean_ms:.9g}",
                    "nano_tpot_mean_ms": f"{nano_point.tpot_mean_ms:.9g}",
                    "non_dbo_tpot_p99_ms": f"{baseline.tpot_p99_ms:.9g}",
                    "dbo_tpot_p99_ms": f"{enabled.tpot_p99_ms:.9g}",
                    "nano_tpot_p99_ms": f"{nano_point.tpot_p99_ms:.9g}",
                    "non_dbo_slo_attainment_percent": (
                        f"{baseline.slo_attainment_percent:.9g}"
                    ),
                    "dbo_slo_attainment_percent": (
                        f"{enabled.slo_attainment_percent:.9g}"
                    ),
                    "nano_slo_attainment_percent": (
                        f"{nano_point.slo_attainment_percent:.9g}"
                    ),
                    "non_dbo_slo_success_count": baseline.slo_success_count,
                    "dbo_slo_success_count": enabled.slo_success_count,
                    "nano_slo_success_count": nano_point.slo_success_count,
                    "non_dbo_total_requests": baseline.total_requests,
                    "dbo_total_requests": enabled.total_requests,
                    "nano_total_requests": nano_point.total_requests,
                    "non_dbo_failed_requests": baseline.failed_requests,
                    "dbo_failed_requests": enabled.failed_requests,
                    "nano_failed_requests": nano_point.failed_requests,
                    "non_dbo_policy_source": baseline.policy_source,
                    "dbo_policy_source": enabled.policy_source,
                    "nano_policy_source": nano_point.policy_source,
                    "non_dbo_scenario": baseline.scenario,
                    "dbo_scenario": enabled.scenario,
                    "nano_scenario": nano_point.scenario,
                    "dbo_decode_token_threshold": (
                        enabled.dbo_decode_token_threshold
                    ),
                    "non_dbo_requests_jsonl": baseline.requests_path,
                    "dbo_requests_jsonl": enabled.requests_path,
                    "nano_request_json": nano_point.requests_path,
                    "nano_log_file": nano_point.config.get("log_path"),
                    "nano_sweep_summary": nano_point.summary_path,
                }
            )


def plot_comparison(
    base: Path,
    rates: list[float],
    non_dbo: Mapping[float, ResultPoint],
    dbo: Mapping[float, ResultPoint],
    nano: Mapping[float, ResultPoint],
    *,
    dataset: str,
    slo_target_ms: float,
    dpi: int,
) -> tuple[Path, Path]:
    mpl_cache = Path(tempfile.gettempdir()) / f"matplotlib-asplos-ae-{os.getuid()}"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")

    from ae_utils.plotting import get_plot_font_family

    import matplotlib.pyplot as plt

    base.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams["font.family"] = get_plot_font_family()
    baseline_color = "#555555"
    dbo_color = "#2F6DB0"
    nano_color = "#D62728"
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=True)

    metrics = (
        (
            "slo_attainment_percent",
            f"SLO attainment (%)\n(TPOT ≤ {slo_target_ms:g} ms)",
            "(a) SLO attainment",
        ),
        ("tpot_mean_ms", "Mean TPOT (ms)", "(b) Mean TPOT"),
        ("tpot_p99_ms", "P99 TPOT (ms)", "(c) P99 TPOT"),
    )
    for axis, (field, ylabel, title) in zip(axes, metrics):
        axis.plot(
            rates,
            [float(getattr(non_dbo[rate], field)) for rate in rates],
            color=baseline_color,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label="non-DBO",
        )
        axis.plot(
            rates,
            [float(getattr(dbo[rate], field)) for rate in rates],
            color=dbo_color,
            marker="s",
            linewidth=1.8,
            markersize=5,
            label="DBO",
        )
        axis.plot(
            rates,
            [float(getattr(nano[rate], field)) for rate in rates],
            color=nano_color,
            marker="D",
            linewidth=1.8,
            markersize=5,
            label="Nano (DCP)",
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Request rate (req/s)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(rates)
        axis.grid(True, linestyle="--", alpha=0.65)
        axis.set_axisbelow(True)
        if field == "slo_attainment_percent":
            axis.set_ylim(0, 105)
        else:
            axis.axhline(
                slo_target_ms,
                color="#888888",
                linestyle=":",
                linewidth=1.0,
            )
    axes[0].legend(frameon=False)

    thresholds = sorted(
        {
            dbo[rate].dbo_decode_token_threshold
            for rate in rates
            if dbo[rate].dbo_decode_token_threshold is not None
        }
    )
    threshold_suffix = (
        f", DBO decode threshold={thresholds[0]}" if len(thresholds) == 1 else ""
    )
    fig.suptitle(
        f"vLLM non-DBO vs. DBO vs. Nano: {dataset}{threshold_suffix}"
    )

    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def print_table(
    rates: Iterable[float],
    non_dbo: Mapping[float, ResultPoint],
    dbo: Mapping[float, ResultPoint],
    nano: Mapping[float, ResultPoint],
) -> None:
    print(
        f"{'Rate':>6}  {'non Mean':>9}  {'DBO Mean':>9}  {'Nano Mean':>9}  "
        f"{'non P99':>9}  {'DBO P99':>9}  {'Nano P99':>9}  "
        f"{'non SLO':>9}  {'DBO SLO':>9}  {'Nano SLO':>9}"
    )
    for rate in rates:
        baseline = non_dbo[rate]
        enabled = dbo[rate]
        nano_point = nano[rate]
        print(
            f"{rate:6g}  {baseline.tpot_mean_ms:9.2f}  "
            f"{enabled.tpot_mean_ms:9.2f}  {nano_point.tpot_mean_ms:9.2f}  "
            f"{baseline.tpot_p99_ms:9.2f}  {enabled.tpot_p99_ms:9.2f}  "
            f"{nano_point.tpot_p99_ms:9.2f}  "
            f"{baseline.slo_attainment_percent:8.2f}%  "
            f"{enabled.slo_attainment_percent:8.2f}%  "
            f"{nano_point.slo_attainment_percent:8.2f}%"
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    dataset_defaults = DATASET_PLOT_DEFAULTS[args.dataset]
    dbo_root = args.dbo_root or dataset_defaults.dbo_root
    nano_stage_glob = args.nano_stage_glob or dataset_defaults.nano_stage_glob
    request_rates = args.request_rates or dataset_defaults.rates
    requested_rates = [rate_key(rate) for rate in request_rates]
    requested_rate_set = set(requested_rates)

    try:
        non_dbo_candidates = discover_candidates(
            mode="non-DBO",
            root=args.non_dbo_root,
            model=args.model,
            dataset=args.dataset,
            strategy=args.strategy,
            scenario_glob=args.non_dbo_scenario_glob,
            expected_dbo=False,
        )
        dbo_candidates = discover_candidates(
            mode="DBO",
            root=dbo_root,
            model=args.model,
            dataset=args.dataset,
            strategy=args.strategy,
            scenario_glob=args.dbo_scenario_glob,
            expected_dbo=True,
        )
        nano_candidates = discover_nano_candidates(
            args.nano_run_dir,
            stage_glob=nano_stage_glob,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    non_dbo = select_points(
        (
            candidate
            for candidate in non_dbo_candidates
            if rate_key(candidate.request_rate) in requested_rate_set
        ),
        selection=args.selection,
        slo_target_ms=args.slo_target_ms,
        max_failure_ratio=args.max_failure_ratio,
    )
    dbo = select_points(
        (
            candidate
            for candidate in dbo_candidates
            if rate_key(candidate.request_rate) in requested_rate_set
        ),
        selection=args.selection,
        slo_target_ms=args.slo_target_ms,
        max_failure_ratio=args.max_failure_ratio,
    )
    nano = select_nano_points(
        (
            candidate
            for candidate in nano_candidates
            if rate_key(candidate.request_rate) in requested_rate_set
        ),
        selection=args.selection,
        slo_target_ms=args.slo_target_ms,
        max_failure_ratio=args.max_failure_ratio,
    )

    common_rates = [
        rate
        for rate in requested_rates
        if rate in non_dbo and rate in dbo and rate in nano
    ]
    missing_non_dbo = [rate for rate in requested_rates if rate not in non_dbo]
    missing_dbo = [rate for rate in requested_rates if rate not in dbo]
    missing_nano = [rate for rate in requested_rates if rate not in nano]
    if (
        missing_non_dbo or missing_dbo or missing_nano
    ) and not args.allow_missing_rates:
        raise SystemExit(
            "missing requested rates; "
            f"non-DBO missing={missing_non_dbo or 'none'}, "
            f"DBO missing={missing_dbo or 'none'}, "
            f"Nano missing={missing_nano or 'none'}"
        )
    if not common_rates:
        raise SystemExit(
            "non-DBO, DBO, and Nano inputs have no common requested rates"
        )

    mismatches = find_config_mismatches(non_dbo, dbo, common_rates)
    if mismatches:
        message = "configuration differences:\n  - " + "\n  - ".join(mismatches)
        if args.strict_config_match:
            raise SystemExit(message)
        print("WARNING: " + message, file=sys.stderr)

    default_output = (
        SCRIPT_DIR / dataset_defaults.output_stem
    )
    base = output_base(args.output or default_output)
    data_path = (
        args.data_output.expanduser()
        if args.data_output is not None
        else base.with_suffix(".tsv")
    )
    write_tsv(
        data_path,
        common_rates,
        non_dbo,
        dbo,
        nano,
        slo_target_ms=args.slo_target_ms,
    )
    if args.data_only:
        print_table(common_rates, non_dbo, dbo, nano)
        print(f"\nWrote {data_path}")
        return 0

    pdf_path, png_path = plot_comparison(
        base,
        common_rates,
        non_dbo,
        dbo,
        nano,
        dataset=dataset_defaults.label,
        slo_target_ms=args.slo_target_ms,
        dpi=args.dpi,
    )
    print_table(common_rates, non_dbo, dbo, nano)
    print(f"\nWrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

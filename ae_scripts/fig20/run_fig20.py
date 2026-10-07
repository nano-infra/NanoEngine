#!/usr/bin/env python3
"""Launch the Figure 20 vLLM DBO and non-DBO source experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


FIG20_DIR = Path(__file__).resolve().parent
AE_ROOT = FIG20_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

E2E_DIR = AE_ROOT / "start-e2e" / "vllm"
HARNESS = E2E_DIR / "offline_poisson_harness.py"

MODEL_NAME = "deepseek_v3_1024k"
MODEL_PATH = Path(
    os.environ.get("VLLM_DPSK_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)
VLLM_WORKDIR = Path(os.environ.get("VLLM_WORKDIR", "/vllm"))
DATASET_ROOT = Path(require_path("AE_DATASET_ROOT"))
DEFAULT_DBO_ARTIFACT_ROOT = FIG20_DIR / "results" / "dbo"
DEFAULT_NON_DBO_ARTIFACT_ROOT = FIG20_DIR / "results" / "non_dbo"
# Retain the old constant for callers that import this module.
DEFAULT_ARTIFACT_ROOT = DEFAULT_DBO_ARTIFACT_ROOT
MASTER_ADDR = os.environ.get("VLLM_4NODE_H200_MASTER_ADDR", "10.102.252.174")
REMOTE_HOSTS = tuple(
    os.environ.get(
        "VLLM_4NODE_REMOTE_HOSTS", "h200-rjob1,h200-rjob2,h200-rjob3"
    )
    .replace(",", " ")
    .split()
)

GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4
DISPATCH_POLICY = "least_batch"
GPU_MEMORY_UTILIZATION = 0.90
WARMUP_REQUESTS = 32
DATA_PARALLEL_RPC_PORT = 29550
DEFAULT_BENCH_DURATION_SEC = 600.0
DEFAULT_MAX_REQUEST_TOKENS = 1_000_000
DEFAULT_MAX_NUM_SEQS = 192
DEFAULT_DBO_DECODE_TOKEN_THRESHOLD = 2
RUN_CONFIG_SCHEMA_VERSION = 1
RUN_MODE_CHOICES = ("dbo", "non-dbo", "both")
EXECUTION_MODES = ("dbo", "non-dbo")

CSV_FIELDNAMES = (
    "enabled",
    "name",
    "cluster",
    "model",
    "dataset",
    "strategy",
    "dispatch_policy",
    "request_rate",
    "rate_phase",
    "max_num_seqs",
    "gpu_memory_utilization",
    "max_requests",
    "warmup_requests",
    "max_model_len",
    "data_parallel_rpc_port",
    "reason",
    "historical_reference",
)


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    dataset_path: Path
    request_rates: tuple[float, ...]


WORKLOADS = {
    "short_random": Workload(
        key="short_random",
        label="ShareGPT4o",
        dataset_path=Path(
            os.environ.get(
                "FIG20_SHAREGPT4O_DATASET",
                DATASET_ROOT
                / "sharegpt-4o"
                / "sharegpt4o-mixed-random-60k.csv",
            )
        ),
        request_rates=(40.0, 60.0, 80.0, 100.0, 120.0),
    ),
    "issue01_random": Workload(
        key="issue01_random",
        label="Issue1%",
        dataset_path=Path(
            os.environ.get(
                "FIG20_ISSUE1_DATASET",
                DATASET_ROOT
                / "sharegpt-4o-mixlong-0326"
                / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
            )
        ),
        request_rates=(10.0, 20.0, 30.0, 40.0, 50.0),
    ),
}


@dataclass(frozen=True)
class CaseRow:
    enabled: int
    name: str
    cluster: str
    model: str
    dataset: str
    strategy: str
    dispatch_policy: str
    request_rate: float
    rate_phase: str
    max_num_seqs: int
    gpu_memory_utilization: float
    max_requests: int
    warmup_requests: int
    max_model_len: int
    data_parallel_rpc_port: int
    reason: str = ""
    historical_reference: str = ""


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return value


def parse_rates(raw: str) -> tuple[float, ...]:
    rates = tuple(positive_float(part) for part in raw.replace(",", " ").split())
    if not rates:
        raise argparse.ArgumentTypeError("expected at least one request rate")
    return tuple(dict.fromkeys(rates))


def validate_run_name(run_name: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not run_name or any(character not in allowed for character in run_name):
        raise argparse.ArgumentTypeError(
            "run name may contain only letters, digits, dot, underscore, and hyphen"
        )
    return run_name


def require_path(path: Path, label: str, *, directory: bool) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {path}")


def filter_dataset_by_request_tokens(
    source: Path,
    destination: Path,
    max_request_tokens: int,
) -> tuple[int, int]:
    """Keep rows whose prompt and requested output fit the token limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total_rows = 0
    kept_rows = 0
    with source.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = reader.fieldnames
        if not fieldnames or not {"prompt_len", "output_len"}.issubset(fieldnames):
            raise ValueError(
                f"dataset must contain prompt_len and output_len: {source}"
            )
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    request_tokens = int(row["prompt_len"]) + int(row["output_len"])
                except (KeyError, TypeError, ValueError) as error:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(
                        f"invalid request lengths at {source}:{row_number}"
                    ) from error
                if request_tokens <= max_request_tokens:
                    writer.writerow(row)
                    kept_rows += 1
    if kept_rows == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"no requests fit max_request_tokens={max_request_tokens}: {source}"
        )
    temporary.replace(destination)
    return total_rows, kept_rows


def load_runner() -> object:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(E2E_DIR))
    import manual_multinode_poisson_runner as runner

    return runner


def selected_remote_hosts(num_nodes: int) -> tuple[str, ...]:
    remote_count = num_nodes - 1
    if len(REMOTE_HOSTS) < remote_count:
        raise ValueError(
            f"--num-nodes {num_nodes} requires {remote_count} worker SSH hosts; "
            f"only {REMOTE_HOSTS} are configured"
        )
    hosts = REMOTE_HOSTS[:remote_count]
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"worker SSH hosts must be distinct: {hosts}")
    return hosts


def topology_names(num_nodes: int) -> tuple[str, str, int]:
    dp_size = num_nodes * GPUS_PER_NODE
    return f"fig20_{num_nodes}node_h200", f"dp{dp_size}", dp_size


def configure_runner(
    runner: object,
    *,
    workload: Workload,
    dataset_path: Path,
    num_nodes: int,
    remote_hosts: tuple[str, ...],
) -> tuple[str, str, int]:
    cluster_name, strategy_name, dp_size = topology_names(num_nodes)
    runner.MODELS[MODEL_NAME] = str(MODEL_PATH.resolve())
    runner.DATASETS[workload.key] = str(dataset_path.resolve())
    runner.HARNESS_ENTRYPOINT = str(HARNESS.resolve())

    base_cluster = runner.CLUSTERS["4node_h200"]
    runner.CLUSTERS[cluster_name] = replace(
        base_cluster,
        master_addr=MASTER_ADDR,
        remote_hosts=remote_hosts,
        workdir=str(VLLM_WORKDIR.resolve()),
    )
    runner.STRATEGIES[strategy_name] = runner.StrategySpec(
        data_parallel_size=dp_size,
        data_parallel_size_local=GPUS_PER_NODE,
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        attention_backend="FLASHMLA",
        all2all_backend="deepep_low_latency",
    )
    return cluster_name, strategy_name, dp_size


def build_cases(
    workload: Workload,
    request_rates: tuple[float, ...],
    *,
    mode: str,
    cluster_name: str,
    strategy_name: str,
    bench_duration_sec: float,
    max_request_tokens: int,
    max_num_seqs: int,
) -> list[CaseRow]:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {mode}")
    mode_tag = "dbo" if mode == "dbo" else "non_dbo"
    mode_label = "DBO" if mode == "dbo" else "non-DBO"
    rate_tag = "_".join(f"{rate:g}" for rate in request_rates)
    return [
        CaseRow(
            enabled=1,
            name=(
                f"fig20_{mode_tag}_{workload.key}_{strategy_name}_"
                f"rate{rate:g}_dur{bench_duration_sec:g}"
            ),
            cluster=cluster_name,
            model=MODEL_NAME,
            dataset=workload.key,
            strategy=strategy_name,
            dispatch_policy=DISPATCH_POLICY,
            request_rate=rate,
            rate_phase=f"fig20_{mode_tag}_{workload.key}_rate{rate_tag}",
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_requests=max(1, round(rate * bench_duration_sec)),
            warmup_requests=WARMUP_REQUESTS,
            max_model_len=max_request_tokens,
            data_parallel_rpc_port=DATA_PARALLEL_RPC_PORT,
            reason=f"Figure 20 {workload.label} vLLM {mode_label} sweep",
        )
        for rate in request_rates
    ]


def write_case_csv(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    temporary.replace(path)


def load_existing_run_config(run_root: Path) -> dict[str, Any] | None:
    config_path = run_root / "run_config.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read existing run config: {config_path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"existing run config is not a JSON object: {config_path}")
        return payload
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(
            f"run directory exists without run_config.json: {run_root}; "
            "choose a different --run-name"
        )
    return None


def common_run_config(
    *,
    mode: str,
    run_name: str,
    num_nodes: int,
    remote_hosts: tuple[str, ...],
    strategy_name: str,
    dp_size: int,
    bench_duration_sec: float,
    max_request_tokens: int,
    max_num_seqs: int,
    dbo_decode_token_threshold: int,
) -> dict[str, Any]:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {mode}")
    dbo_enabled = mode == "dbo"
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "figure": 20,
        "system": "vllm_dbo" if dbo_enabled else "vllm_non_dbo",
        "run_name": run_name,
        "num_nodes": num_nodes,
        "gpus_per_node": GPUS_PER_NODE,
        "num_gpus": dp_size,
        "strategy": strategy_name,
        "data_parallel_size": dp_size,
        "expert_parallel_size": dp_size,
        "tensor_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dispatch_policy": DISPATCH_POLICY,
        "bench_duration_sec": bench_duration_sec,
        "max_request_tokens": max_request_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "dbo_decode_token_threshold": (
            dbo_decode_token_threshold if dbo_enabled else None
        ),
        "master_addr": MASTER_ADDR,
        "remote_hosts": list(remote_hosts),
        "model_path": str(MODEL_PATH.resolve()),
        "vllm_workdir": str(VLLM_WORKDIR.resolve()),
        "datasets": {},
    }


def validate_common_run_config(
    existing: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    run_root: Path,
) -> None:
    if existing is None:
        return
    mismatches = [
        key
        for key, expected_value in expected.items()
        if key != "datasets" and existing.get(key) != expected_value
    ]
    if mismatches:
        detail = ", ".join(
            f"{key}: existing={existing.get(key)!r}, requested={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(
            f"--run-name {expected['run_name']!r} has incompatible settings "
            f"under {run_root}: {detail}; choose a fresh --run-name"
        )
    if not isinstance(existing.get("datasets", {}), dict):
        raise ValueError(f"invalid datasets entry in {run_root / 'run_config.json'}")


def write_run_config(
    run_root: Path,
    common: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    workload: Workload,
    source_dataset: Path,
    prepared_dataset: Path,
    request_rates: tuple[float, ...],
    total_rows: int,
    kept_rows: int,
) -> Path:
    payload = dict(existing or common)
    datasets = dict(payload.get("datasets", {}))
    dataset_record = {
        "label": workload.label,
        "source": str(source_dataset.resolve()),
        "prepared": str(prepared_dataset.resolve()),
        "request_rates": list(request_rates),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows": total_rows - kept_rows,
    }
    previous = datasets.get(workload.key)
    if previous is not None and previous != dataset_record:
        raise ValueError(
            f"--run-name {common['run_name']!r} already records different "
            f"settings for {workload.key}; choose a fresh --run-name"
        )
    datasets[workload.key] = dataset_record
    payload["datasets"] = datasets

    config_path = run_root / "run_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config_path)
    return config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the vLLM DBO and non-DBO sweeps used by Figure 20."
    )
    parser.add_argument(
        "--run-name",
        type=validate_run_name,
        required=True,
        help="output name below the selected mode roots, for example ae_fig20_full",
    )
    parser.add_argument(
        "--mode",
        choices=RUN_MODE_CHOICES,
        default="dbo",
        help=(
            "run DBO, non-DBO, or both sequentially "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="number of 8-GPU nodes (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(WORKLOADS),
        default="short_random",
        help="paper workload to run (default: short_random)",
    )
    parser.add_argument(
        "--request-rates",
        type=parse_rates,
        default=None,
        help="comma- or space-separated rates (default: selected paper sweep)",
    )
    parser.add_argument(
        "--dbo-artifact-root",
        "--artifact-root",
        dest="dbo_artifact_root",
        type=Path,
        default=DEFAULT_DBO_ARTIFACT_ROOT,
        help="DBO result root (default: fig20/results/dbo)",
    )
    parser.add_argument(
        "--non-dbo-artifact-root",
        type=Path,
        default=DEFAULT_NON_DBO_ARTIFACT_ROOT,
        help="non-DBO result root (default: fig20/results/non_dbo)",
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=positive_float,
        default=DEFAULT_BENCH_DURATION_SEC,
        help="request-sending duration per rate (default: %(default)g)",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        default=DEFAULT_MAX_REQUEST_TOKENS,
        help=(
            "maximum prompt_len + output_len and vLLM max_model_len "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        default=DEFAULT_MAX_NUM_SEQS,
    )
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=positive_int,
        default=DEFAULT_DBO_DECODE_TOKEN_THRESHOLD,
        help="DBO decode-token threshold; ignored for non-DBO",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-historical-skips", action="store_true")
    parser.add_argument("--no-keep-going", action="store_true")
    return parser


def selected_execution_modes(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return EXECUTION_MODES
    if mode in EXECUTION_MODES:
        return (mode,)
    raise ValueError(f"unsupported execution mode: {mode}")


def run_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    runner: object,
    workload: Workload,
    request_rates: tuple[float, ...],
    remote_hosts: tuple[str, ...],
    cluster_name: str,
    strategy_name: str,
    dp_size: int,
) -> int:
    artifact_root = (
        args.dbo_artifact_root
        if mode == "dbo"
        else args.non_dbo_artifact_root
    )
    run_root = artifact_root.expanduser().resolve() / args.run_name
    existing_config = load_existing_run_config(run_root)
    requested_config = common_run_config(
        mode=mode,
        run_name=args.run_name,
        num_nodes=args.num_nodes,
        remote_hosts=remote_hosts,
        strategy_name=strategy_name,
        dp_size=dp_size,
        bench_duration_sec=args.bench_duration_sec,
        max_request_tokens=args.max_request_tokens,
        max_num_seqs=args.max_num_seqs,
        dbo_decode_token_threshold=args.dbo_decode_token_threshold,
    )
    validate_common_run_config(
        existing_config,
        requested_config,
        run_root=run_root,
    )

    filtered_dataset = (
        run_root
        / "_filtered_datasets"
        / f"{workload.key}_max{args.max_request_tokens}.csv"
    )
    total_rows, kept_rows = filter_dataset_by_request_tokens(
        workload.dataset_path,
        filtered_dataset,
        args.max_request_tokens,
    )
    config_path = write_run_config(
        run_root,
        requested_config,
        existing_config,
        workload=workload,
        source_dataset=workload.dataset_path,
        prepared_dataset=filtered_dataset,
        request_rates=request_rates,
        total_rows=total_rows,
        kept_rows=kept_rows,
    )

    rows = build_cases(
        workload,
        request_rates,
        mode=mode,
        cluster_name=cluster_name,
        strategy_name=strategy_name,
        bench_duration_sec=args.bench_duration_sec,
        max_request_tokens=args.max_request_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    case_csv = run_root / "_case_csv" / f"{workload.key}_cases.csv"
    write_case_csv(case_csv, rows)

    configure_runner(
        runner,
        workload=workload,
        dataset_path=filtered_dataset,
        num_nodes=args.num_nodes,
        remote_hosts=remote_hosts,
    )
    run_label = f"{args.run_name}-{workload.key}"
    runner_args = [
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(run_root),
        "--run-label",
        run_label,
    ]
    if mode == "dbo":
        runner_args.extend(
            [
                "--frontend-extra-arg=--enable-dbo",
                "--frontend-extra-arg=--dbo-decode-token-threshold",
                f"--frontend-extra-arg={args.dbo_decode_token_threshold}",
                "--headless-extra-arg=--enable-dbo",
                "--headless-extra-arg=--dbo-decode-token-threshold",
                f"--headless-extra-arg={args.dbo_decode_token_threshold}",
            ]
        )
    if args.dry_run:
        runner_args.append("--dry-run")
    if args.ignore_historical_skips:
        runner_args.append("--ignore-historical-skips")
    if args.no_keep_going:
        runner_args.append("--no-keep-going")

    print(f"Mode: {'DBO' if mode == 'dbo' else 'non-DBO'}", flush=True)
    print(f"Run name: {args.run_name}", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Nodes / GPUs: {args.num_nodes} / {dp_size}", flush=True)
    print(
        f"Topology: {strategy_name}+ep{dp_size}, tp1, dcp1",
        flush=True,
    )
    print(f"Workload: {workload.label}", flush=True)
    print("Rates: " + " ".join(f"{rate:g}" for rate in request_rates), flush=True)
    print(
        f"Duration per rate: {args.bench_duration_sec:g}s",
        flush=True,
    )
    print(f"Max request tokens: {args.max_request_tokens}", flush=True)
    print(
        f"Dataset rows: kept {kept_rows}/{total_rows}, "
        f"removed {total_rows - kept_rows}",
        flush=True,
    )
    print(f"Cases: {len(rows)}", flush=True)
    print(f"Run config: {config_path}", flush=True)
    print(f"Case CSV: {case_csv}", flush=True)
    try:
        runner.main(runner_args)
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(error.code, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload = WORKLOADS[args.dataset]
    request_rates = args.request_rates or workload.request_rates

    require_path(E2E_DIR, "shared vLLM E2E", directory=True)
    require_path(HARNESS, "shared vLLM harness", directory=False)
    require_path(VLLM_WORKDIR, "vLLM checkout", directory=True)
    require_path(MODEL_PATH, "DeepSeek-V3 model", directory=True)
    require_path(workload.dataset_path, workload.label + " dataset", directory=False)

    try:
        modes = selected_execution_modes(args.mode)
        remote_hosts = selected_remote_hosts(args.num_nodes)
        cluster_name, strategy_name, dp_size = topology_names(args.num_nodes)
        if len(modes) == 2:
            dbo_root = args.dbo_artifact_root.expanduser().resolve()
            non_dbo_root = args.non_dbo_artifact_root.expanduser().resolve()
            if dbo_root == non_dbo_root:
                raise ValueError(
                    "DBO and non-DBO artifact roots must differ with --mode both"
                )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    runner = load_runner()
    exit_code = 0
    for index, mode in enumerate(modes):
        if index:
            print(flush=True)
        try:
            mode_exit_code = run_mode(
                args,
                mode=mode,
                runner=runner,
                workload=workload,
                request_rates=request_rates,
                remote_hosts=remote_hosts,
                cluster_name=cluster_name,
                strategy_name=strategy_name,
                dp_size=dp_size,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if mode_exit_code:
            exit_code = mode_exit_code
            if args.no_keep_going:
                break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

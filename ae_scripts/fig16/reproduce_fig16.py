#!/usr/bin/env python3
"""Run and plot the NanoDeploy ablation used by Fig. 16."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

BENCHMARK_SCRIPT = SCRIPT_DIR / "bench_serving_overhead.py"
DEFAULT_MODEL = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_DATASET = (
    Path(require_path("AE_DATASET_MIXLONG_0326"))
    / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
)
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"
DEFAULT_RDMA_DEVICES = ",".join(f"mlx5_{index}" for index in range(8))
GPUS_PER_NODE = 8
DEFAULT_RATES = (10, 20, 30, 40)
SINGLE_RATE_TPOT_CAP_MS = 100.0
SUMMARY_FIELDNAMES = (
    "strategy",
    "routing",
    "policy",
    "rate",
    "n_reqs",
    "status",
    "itl_avg_ms",
    "itl_p99_ms",
    "queue_avg_ms",
    "queue_p99_ms",
    "decode_queue_avg_ms",
    "decode_queue_p99_ms",
    "log_file",
    "json_file",
)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    sp_backend: str
    cuda_graph_mode: str
    fixed_sp_size: int
    dynamic_sp: bool
    gpu_utilization: float


VARIANTS = (
    Variant(
        key="current_bucket_deepseek_v3",
        label="NanoDeploy",
        sp_backend="hao_basic",
        cuda_graph_mode="full",
        fixed_sp_size=0,
        dynamic_sp=True,
        gpu_utilization=0.9,
    ),
    Variant(
        key="fixed_sp8",
        label="w/ comm lib, w/ AOT graph, w/o DCP",
        sp_backend="hao_basic",
        cuda_graph_mode="full",
        fixed_sp_size=8,
        dynamic_sp=False,
        gpu_utilization=0.9,
    ),
    Variant(
        key="nccl_bucket_deepseek_v3",
        label="w/o comm lib, w/ AOT graph, w/ DCP",
        sp_backend="nccl",
        cuda_graph_mode="full",
        fixed_sp_size=0,
        dynamic_sp=True,
        gpu_utilization=0.85,
    ),
    Variant(
        key="piecewise_bucket_deepseek_v3",
        label="w/ comm lib, w/o AOT graph, w/ DCP",
        sp_backend="hao_basic",
        cuda_graph_mode="piecewise",
        fixed_sp_size=0,
        dynamic_sp=True,
        gpu_utilization=0.8,
    ),
)
POLICIES = {variant.key: variant.label for variant in VARIANTS}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Run all four variants at one or more request rates."
    )
    run.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--ray-address", default=DEFAULT_RAY_ADDRESS)
    run.add_argument("--master-address", default=DEFAULT_MASTER_ADDRESS)
    run.add_argument(
        "--num-nodes",
        type=positive_int,
        default=4,
        help="Number of 8-GPU nodes to use (default: 4).",
    )
    run.add_argument(
        "--node-ips",
        nargs="+",
        metavar="IP",
        help="Ordered Ray worker IPs to pin; the master IP must be first.",
    )
    run.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    run.add_argument(
        "--run-id",
        "--run-tag",
        dest="run_tag",
        help="Output ID; an existing compatible ID resumes automatically.",
    )
    run.add_argument(
        "--rates",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_RATES),
        metavar="RATE",
        help="Request rates to run (default: 10 20 30 40).",
    )
    run.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.key for variant in VARIANTS],
        metavar="VARIANT",
        help="Run only the listed variant keys instead of all four.",
    )
    run.add_argument(
        "--max-num-seqs",
        "--batch-size",
        dest="max_num_seqs",
        type=positive_int,
        default=256,
        help="Maximum concurrent sequences and CUDA Graph batch size (default: 256).",
    )
    run.add_argument("--duration", type=positive_int, default=600)
    run.add_argument("--max-retries", type=positive_int, default=5)
    run.add_argument("--cooldown", type=nonnegative_int, default=20)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-cluster-check", action="store_true")
    run.add_argument("--no-plot", action="store_true")

    plot = subparsers.add_parser("plot", help="Plot an existing tagged run.")
    plot.add_argument(
        "--run-id",
        "--run-tag",
        dest="run_tag",
        required=True,
        help="Run ID under the output root to plot.",
    )
    plot.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    plot.add_argument(
        "--output",
        type=Path,
        help="Optional output stem; defaults to the same location used by run.",
    )

    args = parser.parse_args()
    if args.command == "run":
        args.rates = tuple(dict.fromkeys(args.rates))
        if args.node_ips:
            if len(set(args.node_ips)) != len(args.node_ips):
                parser.error("--node-ips must not contain duplicates")
            if len(args.node_ips) != args.num_nodes:
                parser.error(
                    "--node-ips must contain exactly --num-nodes addresses"
                )
    if args.run_tag is not None:
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        if not args.run_tag or any(
            character not in allowed for character in args.run_tag
        ):
            parser.error("--run-id contains unsupported characters")
    return args


def check_path(path: Path, label: str, *, directory: bool = False) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise SystemExit(f"{label} {kind} not found: {path}")


def validate_nanodeploy_installation() -> Path:
    spec = importlib.util.find_spec("nanodeploy")
    if spec is None or spec.origin is None:
        raise SystemExit("nanodeploy is not installed in the active Python environment")
    package_dir = Path(spec.origin).resolve().parent
    config_path = package_dir / "config.py"
    check_path(config_path, "NanoDeploy config")
    config_source = config_path.read_text(encoding="utf-8")
    required_names = (
        "scheduler_arch",
        "dynamic_sp_bucket_preset",
        "dynamic_sp_bucket_policy",
        "sp_backend",
    )
    missing = [name for name in required_names if name not in config_source]
    if missing:
        raise SystemExit(
            "installed NanoDeploy does not provide the Fig. 16 API: "
            + ", ".join(missing)
        )
    return package_dir


def validate_inputs(args: argparse.Namespace) -> Path:
    check_path(BENCHMARK_SCRIPT, "Fig. 16 benchmark")
    check_path(args.model_path, "Model", directory=True)
    check_path(args.dataset_path, "Issue 1% dataset")
    with args.dataset_path.open("r", encoding="utf-8", newline="") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or ())
    missing_columns = {"prompt_len", "output_len"}.difference(fieldnames)
    if missing_columns:
        raise SystemExit(
            "Issue 1% dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    return validate_nanodeploy_installation()


def address_host(address: str) -> str | None:
    parsed = urlparse(address if "://" in address else f"//{address}")
    return parsed.hostname


def check_cluster(
    ray_address: str,
    master_address: str,
    num_nodes: int,
    node_ips: list[str] | None,
) -> None:
    try:
        import ray
    except ImportError as exc:
        raise SystemExit("Ray is required to validate and run the cluster") from exc

    ray.init(address=ray_address, ignore_reinit_error=True, logging_level="ERROR")
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive", False)]
        gpu_nodes = [node for node in nodes if node.get("Resources", {}).get("GPU", 0)]
        gpu_count = int(
            sum(node.get("Resources", {}).get("GPU", 0) for node in gpu_nodes)
        )
        expected_gpu_count = num_nodes * GPUS_PER_NODE
        details = [
            (
                node.get("NodeManagerAddress"),
                node.get("Resources", {}).get("GPU", 0),
            )
            for node in gpu_nodes
        ]
        gpu_count_by_ip = {str(node_ip): count for node_ip, count in details}
        checked_details = (
            [(node_ip, gpu_count_by_ip.get(node_ip, 0)) for node_ip in node_ips]
            if node_ips
            else details
        )
        invalid_nodes = [
            detail for detail in checked_details if int(detail[1]) != GPUS_PER_NODE
        ]
        if invalid_nodes:
            raise SystemExit(
                f"Fig. 16 requires {GPUS_PER_NODE} GPUs per Ray worker node; "
                f"found nonstandard nodes {invalid_nodes}"
            )
        if node_ips:
            master_host = address_host(master_address)
            if node_ips[0] != master_host:
                raise SystemExit(
                    f"the first --node-ips entry must match the master IP "
                    f"{master_host}; got {node_ips[0]}"
                )
            print(
                f"Ray cluster verified: pinning this run to "
                + ", ".join(node_ips)
            )
            return
        if len(gpu_nodes) < num_nodes:
            raise SystemExit(
                f"This run requires at least {num_nodes} active "
                f"{GPUS_PER_NODE}-GPU nodes and {expected_gpu_count} GPUs; "
                f"found {details} (total GPUs={gpu_count})"
            )
        print(
            f"Ray cluster verified: {len(gpu_nodes)} active "
            f"{GPUS_PER_NODE}-GPU nodes; this run will use {num_nodes}"
        )
    finally:
        ray.shutdown()


def service_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "GLOO_SOCKET_IFNAME": environment.get("GLOO_SOCKET_IFNAME", "bond0"),
            "NCCL_SOCKET_IFNAME": environment.get("NCCL_SOCKET_IFNAME", "bond0"),
            "NCCL_IB_HCA": environment.get(
                "NCCL_IB_HCA", f"={DEFAULT_RDMA_DEVICES}"
            ),
            "NCCL_IB_GID_INDEX": environment.get("NCCL_IB_GID_INDEX", "3"),
            "NCCL_IB_TC": environment.get("NCCL_IB_TC", "186"),
            "SLIME_VISIBLE_DEVICES": environment.get(
                "SLIME_VISIBLE_DEVICES", DEFAULT_RDMA_DEVICES
            ),
            "SLIME_GID_INDEX": environment.get("SLIME_GID_INDEX", "3"),
            "SLIME_QP_NUM": environment.get("SLIME_QP_NUM", "4"),
            "DEEPEP_SMS": environment.get("DEEPEP_SMS", "16"),
            "DEEPEP_MAX_TOKENS_PER_RANK": environment.get(
                "DEEPEP_MAX_TOKENS_PER_RANK", "256"
            ),
            "DEEPEP_ENABLE_MNNVL": environment.get("DEEPEP_ENABLE_MNNVL", "0"),
            "DEEPEP_MODE": environment.get("DEEPEP_MODE", "auto"),
            "NVSHMEM_QP_DEPTH": environment.get("NVSHMEM_QP_DEPTH", "1024"),
            "TORCHDYNAMO_DISABLE": environment.get("TORCHDYNAMO_DISABLE", "1"),
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "1",
            "PYTHONUNBUFFERED": "1",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    return environment


def benchmark_command(
    args: argparse.Namespace,
    variant: Variant,
    rate: int,
    itl_path: Path,
) -> list[str]:
    num_requests = int(rate * args.duration + 0.5)
    command = [
        sys.executable,
        "-u",
        str(BENCHMARK_SCRIPT),
        "--dataset",
        "csv",
        "--csv-path",
        str(args.dataset_path.resolve()),
        "--max-request-tokens",
        "0",
        "--num-requests",
        str(num_requests),
        "--request-rate",
        str(rate),
        "--sp",
        "8",
        "--dp",
        str(args.num_nodes),
        "--ep",
        str(args.num_nodes * GPUS_PER_NODE),
        "--tp",
        "1",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-limit-gb",
        "141",
        "--gpu-memory-utilization",
        str(variant.gpu_utilization),
        "--max-model-len",
        "1000000",
        "--max-input-len",
        "1000000",
        "--dummy-prefill",
        "--ray-address",
        args.ray_address,
        "--master-address",
        args.master_address,
        "--loop-count",
        "16",
        "--model-path",
        str(args.model_path.resolve()),
        "--routing-strategy",
        "LeastBatch",
        "--itl-log-path",
        str(itl_path),
        "--segment-size",
        "65536",
        "--sp-backend",
        variant.sp_backend,
        "--cuda-graph-mode",
        variant.cuda_graph_mode,
        "--scheduler-arch",
        "legacy_global",
        "--router-policy",
        "least_batch",
        "--fixed-sp-size",
        str(variant.fixed_sp_size),
        "--dynamic-sp-size-strategy",
        "bucket" if variant.dynamic_sp else "legacy",
    ]
    if variant.dynamic_sp:
        command.extend(
            [
                "--enable-dynamic-sp-size",
                "--dynamic-sp-bucket-preset",
                "deepseek_v3",
            ]
        )
    if args.node_ips:
        command.extend(["--node-ips", *args.node_ips])
    return command


def run_logged(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    append: bool,
) -> int:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=AE_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def extract_metrics(log_path: Path) -> dict[str, float]:
    section_names = {
        "--- ITL With Decode Queue (ms/token) ---": "itl",
        "--- Queueing Time (ms) ---": "queue",
        "--- Decode Queue Time (ms) ---": "decode_queue",
    }
    metrics: dict[str, dict[str, float]] = {}
    current_section: str | None = None
    metric_pattern = re.compile(r"^(Avg|P99):\s*([-+0-9.eE]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped in section_names:
            current_section = section_names[stripped]
            metrics[current_section] = {}
            continue
        if stripped.startswith("--- "):
            current_section = None
            continue
        if current_section is None:
            continue
        match = metric_pattern.match(stripped)
        if match:
            metrics[current_section][match.group(1).lower()] = float(match.group(2))

    required = (("itl", "avg"), ("itl", "p99"))
    missing = [
        f"{section}.{metric}"
        for section, metric in required
        if metric not in metrics.get(section, {})
    ]
    if missing:
        raise ValueError("missing benchmark metrics: " + ", ".join(missing))
    return {
        "itl_avg_ms": metrics["itl"]["avg"],
        "itl_p99_ms": metrics["itl"]["p99"],
        "queue_avg_ms": metrics.get("queue", {}).get("avg", float("nan")),
        "queue_p99_ms": metrics.get("queue", {}).get("p99", float("nan")),
        "decode_queue_avg_ms": metrics.get("decode_queue", {}).get(
            "avg", float("nan")
        ),
        "decode_queue_p99_ms": metrics.get("decode_queue", {}).get(
            "p99", float("nan")
        ),
    }


def write_summary_row(
    writer: csv.DictWriter,
    handle,
    *,
    variant: Variant,
    rate: int,
    num_requests: int,
    num_nodes: int,
    status: str,
    metrics: dict[str, float] | None,
    log_path: Path,
    itl_path: Path,
) -> None:
    row: dict[str, object] = {
        "strategy": f"dp{num_nodes}sp8",
        "routing": "LeastBatch",
        "policy": variant.key,
        "rate": rate,
        "n_reqs": num_requests,
        "status": status,
        "log_file": str(log_path.resolve()),
        "json_file": str(itl_path.resolve()) if itl_path.is_file() else "",
    }
    row.update(metrics or {})
    writer.writerow(row)
    handle.flush()


def build_run_config(args: argparse.Namespace, package_dir: Path) -> dict[str, object]:
    return {
        "nanodeploy_package": str(package_dir),
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "node_ips": args.node_ips,
        "rates": list(args.rates),
        "duration_seconds": args.duration,
        "max_num_seqs": args.max_num_seqs,
        "topology": {
            "nodes": args.num_nodes,
            "gpus_per_node": GPUS_PER_NODE,
            "dp": args.num_nodes,
            "sp": 8,
            "tp": 1,
            "ep": args.num_nodes * GPUS_PER_NODE,
        },
        "scheduler_arch": "legacy_global",
        "routing_strategy": "LeastBatch",
        "variants": [asdict(variant) for variant in VARIANTS],
    }


def normalize_run_config(config: dict[str, object]) -> dict[str, object]:
    normalized = dict(config)
    if "batch_size" in normalized:
        normalized.setdefault("max_num_seqs", normalized["batch_size"])
        normalized.pop("batch_size")
    variants = normalized.get("variants")
    if isinstance(variants, list):
        normalized["variants"] = [
            {name: value for name, value in variant.items() if name != "gpu_utilization"}
            if isinstance(variant, dict)
            else variant
            for variant in variants
        ]
    return normalized


def validate_resume_config(config_path: Path, expected: dict[str, object]) -> None:
    if not config_path.is_file():
        raise RuntimeError(f"cannot resume without config: {config_path}")
    existing = normalize_run_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    comparable = normalize_run_config(expected)
    differing = [
        key
        for key in sorted(set(existing) | set(comparable))
        if existing.get(key) != comparable.get(key)
    ]
    if differing:
        raise RuntimeError(
            "cannot resume with different settings ("
            + ", ".join(differing)
            + "); choose a fresh --run-id"
        )


def load_summary_rows(summary_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not summary_path.is_file():
        return {}
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(SUMMARY_FIELDNAMES):
            raise RuntimeError(f"unexpected summary schema: {summary_path}")
        for row in reader:
            try:
                key = (row["policy"], int(float(row["rate"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid summary row in {summary_path}") from exc
            rows[key] = row
    return rows


def case_is_complete(
    row: dict[str, str] | None,
    command: list[str],
    command_path: Path,
    log_path: Path,
    itl_path: Path,
) -> bool:
    if row is None or row.get("status") != "ok":
        return False
    try:
        float(row["itl_avg_ms"])
        float(row["itl_p99_ms"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_command = shlex.join(command) + "\n"
    if not command_path.is_file():
        return False
    if command_path.read_text(encoding="utf-8") != expected_command:
        return False
    for artifact in (log_path, itl_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            return False
    try:
        extract_metrics(log_path)
    except ValueError:
        return False
    return True


def initialize_summary(
    summary_path: Path,
    completed_rows: list[dict[str, str]],
) -> None:
    temporary_path = summary_path.with_name(summary_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDNAMES,
            delimiter="\t",
        )
        writer.writeheader()
        for row in completed_rows:
            writer.writerow({name: row.get(name, "") for name in SUMMARY_FIELDNAMES})
    os.replace(temporary_path, summary_path)


def run_experiments(args: argparse.Namespace) -> Path:
    package_dir = validate_inputs(args)
    tag = args.run_tag or f"fig16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root.resolve() / tag
    summary_path = output_dir / "sweep_summary.tsv"
    print(f"NanoDeploy package: {package_dir}")
    print(f"Output directory: {output_dir}")
    selected_variants = VARIANTS
    if args.variants:
        selected_variants = tuple(
            variant for variant in VARIANTS if variant.key in set(args.variants)
        )
    total_cases = len(selected_variants) * len(args.rates)
    print(
        f"Workload: {len(selected_variants)} variants x {len(args.rates)} rates "
        f"= {total_cases} cases"
    )
    print("Rates: " + ", ".join(str(rate) for rate in args.rates) + " req/s")
    print(
        f"Topology: {args.num_nodes} nodes, "
        f"DP{args.num_nodes}-SP8-EP{args.num_nodes * GPUS_PER_NODE}"
    )
    print(f"Duration: {args.duration} seconds per case")

    planned = [
        (variant, rate)
        for variant in selected_variants
        for rate in args.rates
    ]
    if args.dry_run:
        for variant, rate in planned:
            case_dir = output_dir / variant.key / f"rate_{rate}"
            command = benchmark_command(
                args, variant, rate, case_dir / "itl_samples.jsonl"
            )
            print(f"{variant.key}@{rate}: {shlex.join(command)}")
        return summary_path

    config = build_run_config(args, package_dir)
    config_path = output_dir / "config.json"
    resuming = output_dir.exists()
    if resuming:
        if args.run_tag is None:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        validate_resume_config(config_path, config)
        print(f"Resuming existing run ID: {tag}")

    existing_rows = load_summary_rows(summary_path) if resuming else {}
    completed_keys: set[tuple[str, int]] = set()
    completed_rows: list[dict[str, str]] = []
    for variant, rate in planned:
        case_dir = output_dir / variant.key / f"rate_{rate}"
        log_path = case_dir / "driver.log"
        itl_path = case_dir / "itl_samples.jsonl"
        command_path = case_dir / "command.txt"
        command = benchmark_command(args, variant, rate, itl_path)
        key = (variant.key, rate)
        row = existing_rows.get(key)
        if case_is_complete(row, command, command_path, log_path, itl_path):
            completed_keys.add(key)
            assert row is not None
            completed_rows.append(row)

    pending_cases = [
        (variant, rate)
        for variant, rate in planned
        if (variant.key, rate) not in completed_keys
    ]
    if completed_keys:
        print(f"Resume check: {len(completed_keys)} complete, skipping them")
    if pending_cases and not args.skip_cluster_check:
        check_cluster(
            args.ray_address,
            args.master_address,
            args.num_nodes,
            args.node_ips,
        )
    if not resuming:
        output_dir.mkdir(parents=True)
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    initialize_summary(summary_path, completed_rows)

    environment = service_environment()
    with summary_path.open("a", encoding="utf-8", newline="") as summary_handle:
        writer = csv.DictWriter(
            summary_handle,
            fieldnames=SUMMARY_FIELDNAMES,
            delimiter="\t",
        )
        pending_index = 0
        for case_index, (variant, rate) in enumerate(planned):
            key = (variant.key, rate)
            if key in completed_keys:
                print(
                    f"[{case_index + 1}/{total_cases}] {variant.key}@{rate} "
                    "already complete; skipping",
                    flush=True,
                )
                continue
            pending_index += 1
            case_dir = output_dir / variant.key / f"rate_{rate}"
            case_dir.mkdir(parents=True, exist_ok=True)
            log_path = case_dir / "driver.log"
            itl_path = case_dir / "itl_samples.jsonl"
            command_path = case_dir / "command.txt"
            command = benchmark_command(args, variant, rate, itl_path)
            command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
            num_requests = int(rate * args.duration + 0.5)

            metrics = None
            status = "run_failed"
            for attempt in range(1, args.max_retries + 1):
                print(
                    f"\n[{case_index + 1}/{total_cases}] {variant.key}@{rate} "
                    f"attempt {attempt}/{args.max_retries}",
                    flush=True,
                )
                append_log = log_path.is_file() and log_path.stat().st_size > 0
                if append_log:
                    marker = "RESUME" if attempt == 1 else "RETRY"
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"\n===== {marker} {attempt} =====\n")
                return_code = run_logged(
                    command,
                    environment=environment,
                    log_path=log_path,
                    append=append_log,
                )
                if return_code == 0:
                    try:
                        metrics = extract_metrics(log_path)
                    except ValueError as exc:
                        status = "parse_failed"
                        print(f"Metric parsing failed: {exc}", flush=True)
                    else:
                        status = "ok"
                        break
                else:
                    status = "run_failed"
                    print(f"Benchmark exited with status {return_code}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(10)

            write_summary_row(
                writer,
                summary_handle,
                variant=variant,
                rate=rate,
                num_requests=num_requests,
                num_nodes=args.num_nodes,
                status=status,
                metrics=metrics,
                log_path=log_path,
                itl_path=itl_path,
            )
            if status != "ok":
                raise RuntimeError(
                    f"Fig. 16 case failed after {args.max_retries} attempts: "
                    f"{variant.key}@{rate}; see {log_path}"
                )
            if pending_index < len(pending_cases) and args.cooldown:
                time.sleep(args.cooldown)

    if not args.no_plot and args.variants:
        print("Skipping plot: --variants subset selected; run `plot` after the full sweep")
    elif not args.no_plot:
        plot_output = (
            SCRIPT_DIR / "fig16"
            if args.rates == DEFAULT_RATES
            else output_dir / "fig16"
        )
        plot_summary(summary_path, plot_output)
    return summary_path


def read_summary(
    summary: Path,
) -> tuple[dict[str, dict[int, tuple[float, float]]], tuple[int, ...]]:
    check_path(summary, "Sweep summary")
    data: dict[str, dict[int, tuple[float, float]]] = {
        policy: {} for policy in POLICIES
    }
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            policy = row["policy"]
            if policy not in data or row["status"] != "ok":
                continue
            rate = int(float(row["rate"]))
            data[policy][rate] = (
                float(row["itl_avg_ms"]),
                float(row["itl_p99_ms"]),
            )

    rates = tuple(sorted({rate for values in data.values() for rate in values}))
    if not rates:
        raise SystemExit("Sweep summary contains no successful cases")
    missing = [
        f"{policy}@{rate}"
        for policy in POLICIES
        for rate in rates
        if rate not in data[policy]
    ]
    if missing:
        raise SystemExit("Incomplete sweep summary; missing: " + ", ".join(missing))
    return data, rates


def plot_summary(summary: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    if str(AE_ROOT) not in sys.path:
        sys.path.insert(0, str(AE_ROOT))
    from ae_utils.plotting import get_plot_font_family

    data, rates = read_summary(summary)
    plt.rcParams["font.family"] = get_plot_font_family()
    styles = [
        ("#2F5597", "o"),
        ("#ED7D31", "s"),
        ("#70AD47", "^"),
        ("#A64D79", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    if len(rates) == 1:
        rate = rates[0]
        positions = list(range(len(POLICIES)))
        short_labels = ("NanoDeploy", "w/o DCP", "w/o comm lib", "w/o AOT graph")
        colors = [color for color, _ in styles]
        for metric_index, (axis, title) in enumerate(
            zip(axes, ("Mean TPOT", "P99 TPOT"))
        ):
            values = [data[policy][rate][metric_index] for policy in POLICIES]
            plotted_values = [
                min(value, SINGLE_RATE_TPOT_CAP_MS) for value in values
            ]
            axis.bar(positions, plotted_values, color=colors, width=0.68)
            axis.set_ylim(0, SINGLE_RATE_TPOT_CAP_MS * 1.18)
            axis.set_yticks(range(0, int(SINGLE_RATE_TPOT_CAP_MS) + 1, 20))
            for position, value, plotted_value in zip(
                positions, values, plotted_values
            ):
                axis.text(
                    position,
                    plotted_value + 3,
                    f"{value:,.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                if value > SINGLE_RATE_TPOT_CAP_MS:
                    for x_offset in (-0.10, 0.08):
                        axis.plot(
                            [position + x_offset - 0.07, position + x_offset + 0.07],
                            [94, 101],
                            color="white",
                            linewidth=3.2,
                            clip_on=False,
                            solid_capstyle="round",
                        )
                        axis.plot(
                            [position + x_offset - 0.07, position + x_offset + 0.07],
                            [94, 101],
                            color="black",
                            linewidth=0.9,
                            clip_on=False,
                            solid_capstyle="round",
                        )
            axis.set_title(f"{title} @ {rate} req/s")
            axis.set_ylabel("TPOT (ms)")
            axis.set_xticks(positions, short_labels, rotation=18, ha="right")
            axis.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
    else:
        for (policy, label), (color, marker) in zip(POLICIES.items(), styles):
            mean = [data[policy][rate][0] for rate in rates]
            p99 = [data[policy][rate][1] for rate in rates]
            axes[0].plot(rates, mean, marker=marker, color=color, label=label)
            axes[1].plot(rates, p99, marker=marker, color=color, label=label)

        for axis, title in zip(axes, ("Mean TPOT", "P99 TPOT")):
            axis.set_title(title)
            axis.set_xlabel("Request rate (req/s)")
            axis.set_ylabel("TPOT (ms)")
            axis.set_xticks(rates)
            axis.grid(axis="y", linestyle="--", alpha=0.35)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.20, 1, 1))

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {output.with_suffix('.pdf')}")
    print(f"Created {output.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        summary = run_experiments(args)
        print(f"Summary: {summary}")
    else:
        output_dir = args.output_root.resolve() / args.run_tag
        summary = output_dir / "sweep_summary.tsv"
        if args.output is None:
            _, rates = read_summary(summary)
            output = (
                SCRIPT_DIR / "fig16"
                if rates == DEFAULT_RATES
                else output_dir / "fig16"
            )
        else:
            output = args.output
        plot_summary(summary, output)


if __name__ == "__main__":
    main()

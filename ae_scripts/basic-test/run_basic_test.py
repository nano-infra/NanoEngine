#!/usr/bin/env python3
"""Run the two-node NanoDeploy and vLLM artifact smoke tests."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


BASIC_TEST_DIR = Path(__file__).resolve().parent
AE_ROOT = BASIC_TEST_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_DRIVER = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"
VLLM_DIR = AE_ROOT / "start-e2e" / "vllm"
VLLM_RUNNER = VLLM_DIR / "manual_multinode_poisson_runner.py"
VLLM_HARNESS = VLLM_DIR / "offline_poisson_harness.py"

DEFAULT_DURATION_SEC = 60.0
DEFAULT_REQUEST_RATE = 5.0
DEFAULT_MAX_REQUEST_TOKENS = 750_000
DEFAULT_RAY_ADDRESS = os.environ.get("NANO_RAY_ADDR", "10.102.252.174:6380")
DEFAULT_NANO_MASTER_ADDRESS = os.environ.get(
    "NANO_MASTER_ADDR", "10.102.252.174:29500"
)
DEFAULT_VLLM_MASTER_ADDRESS = os.environ.get("VLLM_2NODE_H200_MASTER_ADDR")
DEFAULT_VLLM_WORKER_HOST = os.environ.get(
    "VLLM_2NODE_H200_REMOTE_HOST", "h200-rjob1"
)
DEFAULT_NANO_WORKDIR = Path(os.environ.get("NANODEPLOY_WORKDIR") or AE_ROOT.parent)
DEFAULT_VLLM_WORKDIR = Path(os.environ.get("VLLM_WORKDIR", "/vllm"))
DEFAULT_MODEL_PATH = Path(
    os.environ.get("BASIC_TEST_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)
DEFAULT_DATASET_PATH = Path(
    os.environ.get("BASIC_TEST_ISSUE1_DATASET")
    or (
        Path(require_path("AE_DATASET_MIXLONG_0326"))
        / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
    )
)
DEFAULT_RDMA_DEVICES = ",".join(f"mlx5_{index}" for index in range(8))
NANO_ENVIRONMENT_DEFAULTS = {
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_IB_HCA": f"={DEFAULT_RDMA_DEVICES}",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TC": "186",
    "SLIME_VISIBLE_DEVICES": DEFAULT_RDMA_DEVICES,
    "SLIME_GID_INDEX": "3",
    "SLIME_QP_NUM": "4",
    "DEEPEP_SMS": "16",
    "DEEPEP_MAX_TOKENS_PER_RANK": "256",
    "DEEPEP_ENABLE_MNNVL": "0",
    "DEEPEP_MODE": "auto",
    "NVSHMEM_QP_DEPTH": "1024",
}


class BasicTestError(RuntimeError):
    """Raised when a required basic-test check fails."""


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument(
        "--system",
        choices=("all", "nano", "vllm"),
        default="all",
        help="System to run (default: %(default)s).",
    )
    parser.add_argument(
        "--request-rate",
        type=positive_float,
        default=DEFAULT_REQUEST_RATE,
        help="Poisson request rate used by both systems (default: %(default)s).",
    )
    parser.add_argument(
        "--duration-sec",
        type=positive_float,
        default=DEFAULT_DURATION_SEC,
        help="Request-sending duration per system (default: %(default)s).",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        default=DEFAULT_MAX_REQUEST_TOKENS,
        help="Maximum prompt plus output tokens for the two-node test.",
    )
    parser.add_argument("--ray-address", default=DEFAULT_RAY_ADDRESS)
    parser.add_argument(
        "--nano-master-address", default=DEFAULT_NANO_MASTER_ADDRESS
    )
    parser.add_argument(
        "--vllm-master-address", default=DEFAULT_VLLM_MASTER_ADDRESS
    )
    parser.add_argument("--vllm-worker-host", default=DEFAULT_VLLM_WORKER_HOST)
    parser.add_argument("--nano-workdir", type=Path, default=DEFAULT_NANO_WORKDIR)
    parser.add_argument("--vllm-workdir", type=Path, default=DEFAULT_VLLM_WORKDIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--vllm-ssh-config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate selected launch plans without starting GPU processes.",
    )
    args = parser.parse_args()
    if args.vllm_master_address is None:
        args.vllm_master_address = args.ray_address.rsplit(":", 1)[0]
    if not args.run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    return args


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(AE_ROOT))
    except ValueError:
        return str(resolved)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def print_stage(index: int, total: int, message: str) -> None:
    border = "=" * 72
    print(f"\n{border}\nStep {index}/{total}: {message}\n{border}", flush=True)


def run_logged_command(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str],
    working_directory: Path,
) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise BasicTestError(
            f"command failed with exit code {return_code}: {shlex.join(command)}"
        )


def require_file(path: Path, label: str, *, nonempty: bool = False) -> None:
    if not path.is_file():
        raise BasicTestError(f"{label} not found: {path}")
    if nonempty and path.stat().st_size == 0:
        raise BasicTestError(f"{label} is empty: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise BasicTestError(f"{label} not found: {path}")


def validate_local_bind_address(address: str) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
    except OSError as error:
        raise BasicTestError(
            f"vLLM master address {address!r} is not assigned to the local "
            "frontend node; pass its current IP with --vllm-master-address"
        ) from error


def prepare_dataset(
    source: Path,
    destination: Path,
    max_request_tokens: int,
) -> dict[str, int]:
    require_file(source, "Issue 1% dataset", nonempty=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    total_rows = 0
    kept_rows = 0
    with source.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        required = {"prompt_len", "output_len"}
        missing = required.difference(fieldnames)
        if missing:
            raise BasicTestError(f"dataset is missing columns: {sorted(missing)}")
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                try:
                    prompt_len = int(float(row["prompt_len"]))
                    output_len = int(float(row["output_len"]))
                except (TypeError, ValueError) as error:
                    raise BasicTestError(
                        f"invalid request lengths at dataset row {total_rows + 1}"
                    ) from error
                if prompt_len + output_len <= max_request_tokens:
                    writer.writerow(row)
                    kept_rows += 1
    if kept_rows == 0:
        temporary.unlink(missing_ok=True)
        raise BasicTestError("no dataset rows remain after token filtering")
    temporary.replace(destination)
    return {"source_rows": total_rows, "kept_rows": kept_rows}


def check_ray_topology(address: str) -> dict[str, Any]:
    try:
        import ray
    except ImportError as error:
        raise BasicTestError(
            "Ray is not importable in the active environment"
        ) from error

    try:
        ray.init(
            address=address,
            ignore_reinit_error=True,
            log_to_driver=False,
            logging_level="ERROR",
        )
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        gpu_counts = [
            int(round(float(node.get("Resources", {}).get("GPU", 0))))
            for node in alive_nodes
        ]
        if len(alive_nodes) != 2 or sorted(gpu_counts) != [8, 8]:
            raise BasicTestError(
                "the NanoDeploy basic test requires exactly two alive Ray nodes "
                f"with eight GPUs each; observed nodes={len(alive_nodes)}, "
                f"gpus={gpu_counts}"
            )
        return {
            "address": address,
            "alive_nodes": len(alive_nodes),
            "gpus_per_node": gpu_counts,
            "total_gpus": sum(gpu_counts),
        }
    finally:
        if ray.is_initialized():
            ray.shutdown()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BasicTestError("cannot calculate a percentile from no values")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_nano_itl_metrics(path: Path) -> dict[str, float | int]:
    request_tpot: list[float] = []
    records = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BasicTestError(
                    f"invalid NanoDeploy ITL JSON at line {line_number}"
                ) from error
            records += 1
            value = row.get("avg_itl_with_decode_queue_ms")
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric) and numeric > 0:
                request_tpot.append(numeric)
    if records == 0 or not request_tpot:
        raise BasicTestError("NanoDeploy produced no valid ITL/TPOT samples")
    return {
        "request_records": records,
        "tpot_sample_count": len(request_tpot),
        "mean_tpot_ms": statistics.fmean(request_tpot),
        "p99_tpot_ms": percentile(request_tpot, 99.0),
    }


def run_nanodeploy(
    args: argparse.Namespace,
    run_root: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    output_dir = run_root / "nanodeploy"
    output_dir.mkdir(parents=True)
    driver_log = output_dir / "driver.log"
    itl_log = output_dir / "itl_samples.jsonl"
    command_path = output_dir / "command.txt"
    expected_requests = max(1, round(args.request_rate * args.duration_sec))
    environment = os.environ.copy()
    for proxy_name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        environment.pop(proxy_name, None)

    python_path_parts = [str(args.nano_workdir.resolve())]
    build_lib = args.nano_workdir / "build" / "lib"
    if build_lib.is_dir():
        python_path_parts.append(str(build_lib.resolve()))
    if environment.get("PYTHONPATH"):
        python_path_parts.append(environment["PYTHONPATH"])
    for name, default_value in NANO_ENVIRONMENT_DEFAULTS.items():
        if not environment.get(name):
            environment[name] = default_value
    environment.update(
        {
            "NANODEPLOY_WORKDIR": str(args.nano_workdir.resolve()),
            "PYTHONPATH": os.pathsep.join(python_path_parts),
            "TORCHDYNAMO_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "1",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    command = [
        sys.executable,
        "-u",
        str(NANO_DRIVER),
        "--dataset",
        "csv",
        "--csv-path",
        str(dataset_path.resolve()),
        "--max-request-tokens",
        str(args.max_request_tokens),
        "--max-input-len",
        str(args.max_request_tokens),
        "--num-requests",
        str(expected_requests),
        "--request-rate",
        f"{args.request_rate:g}",
        "--sp",
        "8",
        "--dp",
        "2",
        "--ep",
        "16",
        "--tp",
        "1",
        "--max-num-seqs",
        "128",
        "--gpu-memory-limit-gb",
        "141",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(args.max_request_tokens),
        "--dummy-prefill",
        "--ray-address",
        args.ray_address,
        "--master-address",
        args.nano_master_address,
        "--loop-count",
        "16",
        "--model-path",
        str(args.model_path.resolve()),
        "--routing-strategy",
        "LeastBatch",
        "--itl-log-path",
        str(itl_log.resolve()),
        "--segment-size",
        "65536",
        "--sp-backend",
        "hao_basic",
        "--cuda-graph-mode",
        "full",
        "--scheduler-arch",
        "legacy_global",
        "--fixed-sp-size",
        "0",
        "--enable-dynamic-sp-size",
        "--dynamic-sp-size-strategy",
        "bucket",
        "--dynamic-sp-bucket-preset",
        "deepseek_v3",
    ]
    recorded_environment = [
        f"{name}={environment[name]}" for name in NANO_ENVIRONMENT_DEFAULTS
    ]
    command_path.write_text(
        f"cd {shlex.quote(str(args.nano_workdir.resolve()))}\n"
        f"{shlex.join(['env', *recorded_environment, *command])}\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "pass",
        "topology": "DP2-CP8-DCP",
        "output_dir": relative_path(output_dir),
        "command": relative_path(command_path),
        "expected_requests": expected_requests,
    }
    if args.dry_run:
        print(f"DRY RUN: {shlex.join(command)}", flush=True)
        return result

    run_logged_command(
        command,
        driver_log,
        environment=environment,
        working_directory=args.nano_workdir,
    )
    require_file(driver_log, "NanoDeploy driver log", nonempty=True)
    require_file(itl_log, "NanoDeploy ITL log", nonempty=True)
    driver_text = driver_log.read_text(encoding="utf-8", errors="replace")
    sent_match = re.search(r"Requests sent:\s*(\d+)", driver_text)
    completed_match = re.search(r"Requests completed:\s*(\d+)", driver_text)
    if sent_match is None or completed_match is None:
        raise BasicTestError("NanoDeploy driver log is missing request totals")
    sent = int(sent_match.group(1))
    completed = int(completed_match.group(1))
    if sent != expected_requests or completed != expected_requests:
        raise BasicTestError(
            "NanoDeploy did not complete every request: "
            f"sent={sent}, completed={completed}, expected={expected_requests}"
        )
    result.update(
        {
            "driver_log": relative_path(driver_log),
            "itl_log": relative_path(itl_log),
            "requests_sent": sent,
            "requests_completed": completed,
            "metrics": load_nano_itl_metrics(itl_log),
        }
    )
    return result


def find_single_vllm_manifest(output_dir: Path) -> Path:
    manifests = sorted((output_dir / "_runs").glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise BasicTestError(
            f"expected one vLLM run manifest, found {len(manifests)}"
        )
    return manifests[0]


def run_vllm(
    args: argparse.Namespace,
    run_root: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    output_dir = run_root / "vllm"
    expected_requests = max(1, round(args.request_rate * args.duration_sec))
    run_label = f"basic-test-vllm-{args.run_id}"
    if str(VLLM_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_DIR))
    import manual_multinode_poisson_runner as runner

    runner.MODELS["basic_test_deepseek"] = str(args.model_path.resolve())
    runner.DATASETS["basic_test_issue01"] = str(dataset_path.resolve())
    runner.HARNESS_ENTRYPOINT = str(VLLM_HARNESS.resolve())

    base_cluster = runner.CLUSTERS["2node_h200"]
    ssh_opts = base_cluster.ssh_opts
    if args.vllm_ssh_config is not None:
        updated_ssh_opts = list(ssh_opts)
        try:
            config_path_index = updated_ssh_opts.index("-F") + 1
        except ValueError:
            updated_ssh_opts[:0] = ["-F", str(args.vllm_ssh_config.resolve())]
        else:
            updated_ssh_opts[config_path_index] = str(
                args.vllm_ssh_config.resolve()
            )
        ssh_opts = tuple(updated_ssh_opts)
    runner.CLUSTERS["basic_test_2node"] = replace(
        base_cluster,
        master_addr=args.vllm_master_address,
        remote_hosts=(args.vllm_worker_host,),
        workdir=str(args.vllm_workdir.resolve()),
        ssh_opts=ssh_opts,
    )

    case_csv = output_dir / "_case_csv" / "cases.csv"
    case_csv.parent.mkdir(parents=True)
    case_row = {
        "enabled": "1",
        "name": "basic_test_vllm_dp2tp8dcp8_issue01",
        "cluster": "basic_test_2node",
        "model": "basic_test_deepseek",
        "dataset": "basic_test_issue01",
        "strategy": "dp2dcp8",
        "dispatch_policy": "waiting_x4_plus_running",
        "request_rate": f"{args.request_rate:g}",
        "rate_phase": "basic_test",
        "max_num_seqs": "64",
        "gpu_memory_utilization": "0.85",
        "max_requests": str(expected_requests),
        "warmup_requests": "32",
        "max_model_len": str(args.max_request_tokens),
        "data_parallel_rpc_port": "29550",
        "reason": "one-minute Issue 1% basic test",
        "historical_reference": "",
    }
    with case_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=runner.CASE_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(case_row)

    runner_arguments = [
        "--case-csv",
        str(case_csv.resolve()),
        "--artifact-root",
        str(output_dir.resolve()),
        "--run-label",
        run_label,
        "--ignore-historical-skips",
        "--no-keep-going",
    ]
    if args.dry_run:
        runner_arguments.append("--dry-run")
    print(
        f"$ {sys.executable} {VLLM_RUNNER} {shlex.join(runner_arguments)}",
        flush=True,
    )
    try:
        runner.main(runner_arguments)
    except SystemExit as error:
        if error.code is None:
            exit_code = 0
        elif isinstance(error.code, int):
            exit_code = error.code
        else:
            exit_code = 1
        if exit_code != 0:
            detail = f": {error.code}" if isinstance(error.code, str) else ""
            raise BasicTestError(
                f"vLLM E2E runner failed with exit code {exit_code}{detail}"
            ) from error

    manifest_path = find_single_vllm_manifest(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = manifest.get("results", [])
    if len(results) != 1:
        raise BasicTestError(
            f"vLLM manifest must contain exactly one result; got {len(results)}"
        )
    case = results[0]
    expected_status = "dry_run" if args.dry_run else "ok"
    if case.get("status") != expected_status:
        raise BasicTestError(
            f"vLLM status is {case.get('status')!r}, expected {expected_status!r}"
        )

    result: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "pass",
        "topology": "DP2-TP8-DCP8",
        "manifest": relative_path(manifest_path),
        "case_dir": relative_path(Path(case["case_dir"])),
        "expected_requests": expected_requests,
    }
    if args.dry_run:
        return result

    summary_path = Path(case["summary_json"])
    require_file(summary_path, "vLLM benchmark summary", nonempty=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    total = int(summary.get("total_requests", 0))
    successful = int(summary.get("successful_requests", 0))
    failed = int(summary.get("failed_requests", -1))
    if total != expected_requests or successful != total or failed != 0:
        raise BasicTestError(
            "vLLM did not complete every request: "
            f"total={total}, successful={successful}, failed={failed}, "
            f"expected={expected_requests}"
        )
    tpot = summary.get("tpot_by_e2e")
    if (
        not isinstance(tpot, dict)
        or tpot.get("mean") is None
        or tpot.get("p99") is None
    ):
        raise BasicTestError("vLLM summary is missing Mean/P99 TPOT")
    result.update(
        {
            "summary": relative_path(summary_path),
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "metrics": {
                "mean_tpot_ms": float(tpot["mean"]),
                "p99_tpot_ms": float(tpot["p99"]),
            },
        }
    )
    return result


def main() -> int:
    args = parse_args()
    run_nano = args.system in {"all", "nano"}
    run_vllm_selected = args.system in {"all", "vllm"}
    total_steps = 2 + int(run_nano) + int(run_vllm_selected)
    run_root = BASIC_TEST_DIR / "results" / args.run_id
    if run_root.exists():
        print(f"ERROR: refusing to overwrite existing run: {run_root}", file=sys.stderr)
        return 1
    run_root.mkdir(parents=True)
    summary_path = run_root / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "configuration": {
            "selected_system": args.system,
            "dataset": "Issue 1%",
            "request_rate": args.request_rate,
            "duration_sec_per_system": args.duration_sec,
            "max_request_tokens": args.max_request_tokens,
            "nanodeploy_topology": "DP2-CP8-DCP",
            "vllm_topology": "DP2-TP8-DCP8",
        },
        "systems": {
            "nanodeploy": {
                "status": "not_run" if run_nano else "not_selected"
            },
            "vllm": {
                "status": "not_run" if run_vllm_selected else "not_selected"
            },
        },
    }
    write_json(summary_path, summary)

    try:
        current_step = 1
        print_stage(
            current_step,
            total_steps,
            "Validate inputs and prepare the Issue 1% subset",
        )
        if run_nano:
            require_file(NANO_DRIVER, "NanoDeploy shared E2E driver")
            require_directory(
                args.nano_workdir / "nanodeploy", "NanoDeploy checkout"
            )
        if run_vllm_selected:
            require_file(VLLM_RUNNER, "vLLM shared E2E runner")
            require_file(VLLM_HARNESS, "vLLM shared E2E harness")
            require_directory(args.vllm_workdir, "vLLM work directory")
            if not args.dry_run:
                validate_local_bind_address(args.vllm_master_address)
        require_directory(args.model_path, "DeepSeek-V3 model directory")
        filtered_dataset = run_root / "inputs" / "issue01_max_tokens.csv"
        dataset_stats = prepare_dataset(
            args.dataset_path,
            filtered_dataset,
            args.max_request_tokens,
        )
        expected_requests = max(1, round(args.request_rate * args.duration_sec))
        required_rows = expected_requests + (32 if run_vllm_selected else 0)
        if dataset_stats["kept_rows"] < required_rows:
            raise BasicTestError(
                "filtered dataset does not contain enough rows for measured "
                "and warmup requests"
            )
        summary["dataset"] = {
            **dataset_stats,
            "filtered_path": relative_path(filtered_dataset),
        }
        if run_nano and not args.dry_run:
            summary["ray_topology"] = check_ray_topology(args.ray_address)
        write_json(summary_path, summary)

        if run_nano:
            current_step += 1
            print_stage(
                current_step,
                total_steps,
                f"Run NanoDeploy DP2-CP8-DCP for {args.duration_sec:g} seconds",
            )
            summary["systems"]["nanodeploy"] = {"status": "running"}
            write_json(summary_path, summary)
            summary["systems"]["nanodeploy"] = run_nanodeploy(
                args, run_root, filtered_dataset
            )
            write_json(summary_path, summary)

        if run_vllm_selected:
            current_step += 1
            print_stage(
                current_step,
                total_steps,
                f"Run vLLM DP2-TP8-DCP8 for {args.duration_sec:g} seconds",
            )
            summary["systems"]["vllm"] = {"status": "running"}
            write_json(summary_path, summary)
            summary["systems"]["vllm"] = run_vllm(
                args, run_root, filtered_dataset
            )

        current_step += 1
        print_stage(current_step, total_steps, "Finalize the basic-test summary")
        summary["status"] = "dry_run" if args.dry_run else "pass"
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        if args.dry_run:
            if run_nano:
                print("NanoDeploy DP2-CP8-DCP: DRY RUN", flush=True)
            if run_vllm_selected:
                print("vLLM DP2-TP8-DCP8: DRY RUN", flush=True)
            print("Basic test: DRY RUN", flush=True)
        else:
            if run_nano:
                print("NanoDeploy DP2-CP8-DCP: PASS", flush=True)
            if run_vllm_selected:
                print("vLLM DP2-TP8-DCP8: PASS", flush=True)
            print("Basic test: PASS", flush=True)
        print(f"Summary: {relative_path(summary_path)}", flush=True)
        return 0
    except KeyboardInterrupt:
        summary["status"] = "failed"
        summary["finished_at"] = utc_now()
        summary["error"] = "interrupted by user"
        write_json(summary_path, summary)
        print("\nBasic test: FAIL (interrupted)", file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        summary["status"] = "failed"
        summary["finished_at"] = utc_now()
        summary["error"] = str(error)
        write_json(summary_path, summary)
        print(f"\nBasic test: FAIL: {error}", file=sys.stderr, flush=True)
        print(f"Summary: {relative_path(summary_path)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

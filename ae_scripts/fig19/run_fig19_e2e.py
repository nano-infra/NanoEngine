#!/usr/bin/env python3
"""Run the NanoDeploy service workload used by Figure 19."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_BENCHMARK = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"
DEFAULT_MODEL_PATH = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_DATASET_PATH = (
    Path(require_path("AE_DATASET_MIXLONG_0326"))
    / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
)
DEFAULT_OUTPUT_ROOT = AE_ROOT / "bench_logs" / "e2e"
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"

REQUEST_RATE = 60
GPUS_PER_NODE = 8
DEFAULT_NUM_NODES = 4
DEFAULT_SEND_DURATION_SEC = 600


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output name under bench_logs/e2e/, for example ae_fig19.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--ray-address", default=DEFAULT_RAY_ADDRESS)
    parser.add_argument("--master-address", default=DEFAULT_MASTER_ADDRESS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=DEFAULT_NUM_NODES,
        help="Number of 8-GPU nodes (default: %(default)s).",
    )
    parser.add_argument(
        "--sp",
        type=int,
        default=8,
        help="Attention SP size; the Fig. 19 collector expects 8.",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--send-duration-sec",
        type=positive_int,
        default=DEFAULT_SEND_DURATION_SEC,
        help="How long to send requests, in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the command without creating a run directory.",
    )
    args = parser.parse_args()
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_name
    ):
        parser.error("--run-name contains unsupported characters")
    return args


def num_requests(args: argparse.Namespace) -> int:
    return REQUEST_RATE * args.send_duration_sec


def attention_dp_size(args: argparse.Namespace) -> int:
    return args.num_nodes


def expert_parallel_size(args: argparse.Namespace) -> int:
    return args.num_nodes * GPUS_PER_NODE


def validate_inputs(args: argparse.Namespace) -> None:
    if not NANO_BENCHMARK.is_file():
        raise SystemExit(f"NanoDeploy E2E driver not found: {NANO_BENCHMARK}")
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")
    if not args.dataset_path.is_file():
        raise SystemExit(f"Dataset CSV not found: {args.dataset_path}")
    if args.sp != 8:
        raise SystemExit(
            f"--sp must be 8: fig19/collect_fig19_data.py only understands "
            f"cp_size=8 groups, got --sp {args.sp}"
        )

    with args.dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        fieldnames = csv.DictReader(dataset_file).fieldnames or []
    missing = {"prompt_len", "output_len"} - set(fieldnames)
    if missing:
        raise SystemExit(
            f"Dataset CSV is missing required columns {sorted(missing)}: "
            f"{args.dataset_path}"
        )


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
                "NCCL_IB_HCA",
                "=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
            ),
            "NCCL_IB_GID_INDEX": environment.get("NCCL_IB_GID_INDEX", "3"),
            "NCCL_IB_TC": environment.get("NCCL_IB_TC", "186"),
            "SLIME_VISIBLE_DEVICES": environment.get(
                "SLIME_VISIBLE_DEVICES",
                "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
            ),
            "SLIME_GID_INDEX": environment.get("SLIME_GID_INDEX", "3"),
            "SLIME_QP_NUM": environment.get("SLIME_QP_NUM", "4"),
            "DEEPEP_SMS": environment.get("DEEPEP_SMS", "16"),
            "DEEPEP_MAX_TOKENS_PER_RANK": environment.get(
                "DEEPEP_MAX_TOKENS_PER_RANK", "256"
            ),
            "DEEPEP_ENABLE_MNNVL": environment.get(
                "DEEPEP_ENABLE_MNNVL", "0"
            ),
            "DEEPEP_MODE": environment.get("DEEPEP_MODE", "auto"),
            "NVSHMEM_QP_DEPTH": environment.get("NVSHMEM_QP_DEPTH", "1024"),
            "TORCHDYNAMO_DISABLE": environment.get("TORCHDYNAMO_DISABLE", "1"),
            "PYTHONUNBUFFERED": "1",
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "1",
            "NANODEPLOY_LOG_DECODE_A2A_MASKS": "1",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    return environment


def service_command(args: argparse.Namespace, itl_log: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(NANO_BENCHMARK),
        "--dataset",
        "csv",
        "--csv-path",
        str(args.dataset_path.resolve()),
        "--max-request-tokens",
        "0",
        "--num-requests",
        str(num_requests(args)),
        "--request-rate",
        str(REQUEST_RATE),
        "--sp",
        str(args.sp),
        "--dp",
        str(attention_dp_size(args)),
        "--ep",
        str(expert_parallel_size(args)),
        "--tp",
        str(args.tp),
        "--max-num-seqs",
        "256",
        "--gpu-memory-limit-gb",
        "141",
        "--gpu-memory-utilization",
        "0.9",
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
        str(itl_log),
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


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=AE_ROOT,
            env=service_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if "'mode': 'decode_a2a_masks'" not in line:
                    print(line, end="", flush=True)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            raise
        return process.wait()


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    run_dir = args.output_root.resolve() / args.run_name
    rate_dir = run_dir / "rate_60"
    itl_log = rate_dir / "itl_samples.jsonl"
    driver_log = rate_dir / "driver.log"
    command = service_command(args, itl_log)

    config = {
        "figure": 19,
        "system": "NanoDeploy",
        "model": str(args.model_path.resolve()),
        "dataset": str(args.dataset_path.resolve()),
        "request_rate": REQUEST_RATE,
        "send_duration_sec": args.send_duration_sec,
        "num_requests": num_requests(args),
        "num_nodes": args.num_nodes,
        "num_gpus": args.num_nodes * GPUS_PER_NODE,
        "topology": (
            f"dp{attention_dp_size(args)}_sp{args.sp}_tp{args.tp}_"
            f"ep{expert_parallel_size(args)}"
        ),
        "sp_backend": "hao_basic",
        "scheduler": "legacy_global",
        "routing": "LeastBatch",
        "dynamic_sp_size_strategy": "bucket",
        "dynamic_sp_bucket_preset": "deepseek_v3",
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "command": command,
    }

    print(f"Run directory: {run_dir}")
    print(f"Command: {shlex.join(command)}")
    if args.dry_run:
        print("Dry run complete; no directory was created and no GPU work was launched.")
        return 0

    if run_dir.exists():
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different --run-name."
        )
    rate_dir.mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (rate_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    summary_path = run_dir / "run_summary.tsv"
    summary_path.write_text(
        "rate\tnum_requests\tstatus\tlog_dir\n", encoding="utf-8"
    )

    try:
        return_code = run_logged(command, driver_log)
    except KeyboardInterrupt:
        status = "interrupted"
        return_code = 130
    else:
        status = "ok" if return_code == 0 else "failed"

    with summary_path.open("a", encoding="utf-8") as summary_file:
        summary_file.write(
            f"{REQUEST_RATE}\t{num_requests(args)}\t{status}\t{rate_dir}\n"
        )

    if return_code != 0:
        print(f"Figure 19 service run {status}; see {driver_log}", file=sys.stderr)
        return return_code

    print(f"Routing log: {driver_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

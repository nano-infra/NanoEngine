#!/usr/bin/env python3
"""Run the NanoDeploy E2E workloads that provide Fig. 15 profile inputs."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_BENCHMARK = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"
DEFAULT_NANODEPLOY_WORKDIR = Path(
    os.environ.get("NANODEPLOY_WORKDIR") or AE_ROOT.parent
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("FIG15_NANO_MODEL") or require_path("AE_DPSK_MODEL")
)
DATASET_ROOT = require_path("AE_DATASET_ROOT")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "e2e_results" / "nano"
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"
COOLDOWN_SECONDS = 15
GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4


@dataclass(frozen=True)
class Workload:
    name: str
    dataset_path: Path
    request_rate: int
    num_requests: int
    max_num_seqs: int


WORKLOADS = (
    Workload(
        name="short",
        dataset_path=Path(
            os.environ.get(
                "FIG15_NANO_SHORT_DATASET",
                DATASET_ROOT / "sharegpt-4o" / "sharegpt4o-mixed-random-60k.csv",
            )
        ),
        request_rate=100,
        num_requests=60_000,
        max_num_seqs=256,
    ),
    Workload(
        name="issue01",
        dataset_path=Path(
            os.environ.get(
                "FIG15_NANO_ISSUE01_DATASET",
                DATASET_ROOT
                / "sharegpt-4o-mixlong-0326"
                / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
            )
        ),
        request_rate=60,
        num_requests=36_000,
        max_num_seqs=192,
    ),
    Workload(
        name="issue05",
        dataset_path=Path(
            os.environ.get(
                "FIG15_NANO_ISSUE05_DATASET",
                DATASET_ROOT
                / "sharegpt-4o-mixlong-0326"
                / "sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
            )
        ),
        request_rate=30,
        num_requests=18_000,
        max_num_seqs=256,
    ),
)
WORKLOAD_BY_NAME = {workload.name: workload for workload in WORKLOADS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(WORKLOAD_BY_NAME),
        default=list(WORKLOAD_BY_NAME),
        help="Workloads to run (default: short issue01 issue05).",
    )
    parser.add_argument(
        "--nanodeploy-workdir",
        type=Path,
        default=DEFAULT_NANODEPLOY_WORKDIR,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--short-dataset",
        type=Path,
        default=WORKLOAD_BY_NAME["short"].dataset_path,
    )
    parser.add_argument(
        "--issue01-dataset",
        type=Path,
        default=WORKLOAD_BY_NAME["issue01"].dataset_path,
    )
    parser.add_argument(
        "--issue05-dataset",
        type=Path,
        default=WORKLOAD_BY_NAME["issue05"].dataset_path,
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("FIG15_NANO_RAY_ADDRESS", DEFAULT_RAY_ADDRESS),
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get(
            "FIG15_NANO_MASTER_ADDRESS", DEFAULT_MASTER_ADDRESS
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="Number of 8-GPU nodes to use (default: %(default)s).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        "--run-name",
        dest="run_name",
        help="Result directory ID (default: run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.run_name and any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in args.run_name
    ):
        parser.error("--run-id contains unsupported characters")
    return args


def selected_workloads(args: argparse.Namespace) -> list[Workload]:
    dataset_paths = {
        "short": args.short_dataset,
        "issue01": args.issue01_dataset,
        "issue05": args.issue05_dataset,
    }
    node_scale = args.num_nodes / PAPER_NUM_NODES
    return [
        Workload(
            name=name,
            dataset_path=dataset_paths[name],
            request_rate=max(
                1, round(WORKLOAD_BY_NAME[name].request_rate * node_scale)
            ),
            num_requests=max(
                1, round(WORKLOAD_BY_NAME[name].num_requests * node_scale)
            ),
            max_num_seqs=max(
                1, round(WORKLOAD_BY_NAME[name].max_num_seqs * node_scale)
            ),
        )
        for name in args.datasets
    ]


def require_inputs(args: argparse.Namespace, workloads: list[Workload]) -> None:
    if not NANO_BENCHMARK.is_file():
        raise SystemExit(f"NanoDeploy E2E driver not found: {NANO_BENCHMARK}")
    if not (args.nanodeploy_workdir / "nanodeploy").is_dir():
        raise SystemExit(
            "NanoDeploy checkout not found: " f"{args.nanodeploy_workdir}"
        )
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")
    missing = [
        str(item.dataset_path)
        for item in workloads
        if not item.dataset_path.is_file()
    ]
    if missing:
        raise SystemExit("Dataset CSV not found:\n  " + "\n  ".join(missing))
    if importlib.util.find_spec("nanodeploy") is None:
        build_lib = args.nanodeploy_workdir / "build" / "lib"
        if not build_lib.is_dir():
            raise SystemExit(
                "nanodeploy is not installed and the checkout has no build/lib"
            )


def benchmark_environment(workdir: Path) -> dict[str, str]:
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

    python_paths = [str(workdir)]
    build_lib = workdir / "build" / "lib"
    if build_lib.is_dir():
        python_paths.append(str(build_lib))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])

    environment.update(
        {
            "NANODEPLOY_WORKDIR": str(workdir),
            "PYTHONPATH": os.pathsep.join(python_paths),
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
            "DEEPEP_ENABLE_MNNVL": environment.get("DEEPEP_ENABLE_MNNVL", "0"),
            "DEEPEP_MODE": environment.get("DEEPEP_MODE", "auto"),
            "NVSHMEM_QP_DEPTH": environment.get("NVSHMEM_QP_DEPTH", "1024"),
            "TORCHDYNAMO_DISABLE": environment.get("TORCHDYNAMO_DISABLE", "1"),
            "PYTHONUNBUFFERED": "1",
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "1",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    return environment


def benchmark_command(
    args: argparse.Namespace,
    workload: Workload,
    itl_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(NANO_BENCHMARK),
        "--dataset",
        "csv",
        "--csv-path",
        str(workload.dataset_path),
        "--max-request-tokens",
        "0",
        "--num-requests",
        str(workload.num_requests),
        "--request-rate",
        str(workload.request_rate),
        "--sp",
        "8",
        "--dp",
        str(args.num_nodes),
        "--ep",
        str(args.num_nodes * GPUS_PER_NODE),
        "--tp",
        "1",
        "--max-num-seqs",
        str(workload.max_num_seqs),
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
        str(args.model_path),
        "--routing-strategy",
        "LeastBatch",
        "--itl-log-path",
        str(itl_path),
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


def run_logged(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    cwd: Path,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if "sp_seq_lens" not in line:
                    print(line, end="", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def decode_record(line: str) -> dict[str, Any] | None:
    if "sp_seq_lens" not in line:
        return None
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        record = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("mode") != "decode":
        return None
    if not isinstance(record.get("sp_seq_lens"), list):
        return None
    return record


def validate_frame(sp_seq_lens: Any, num_nodes: int) -> None:
    if not isinstance(sp_seq_lens, list) or len(sp_seq_lens) != num_nodes:
        raise RuntimeError(
            f"Selected frame must contain {num_nodes} DP groups"
        )
    for dp_index, dp_group in enumerate(sp_seq_lens):
        if not isinstance(dp_group, list) or len(dp_group) != 8:
            raise RuntimeError(
                f"Selected frame DP group {dp_index} must contain eight SP ranks"
            )
        for sp_index, rank_sequences in enumerate(dp_group):
            if not isinstance(rank_sequences, list):
                raise RuntimeError(
                    f"Selected frame DP{dp_index}/SP{sp_index} must be a list"
                )
            if any(
                not isinstance(length, int)
                or isinstance(length, bool)
                or length <= 0
                for length in rank_sequences
            ):
                raise RuntimeError(
                    f"Selected frame DP{dp_index}/SP{sp_index} has an invalid "
                    "sequence length"
                )


def extract_middle_inputs(
    log_path: Path,
    nano_output_path: Path,
    vllm_output_path: Path,
    num_nodes: int,
) -> dict[str, Any]:
    total_records = 0
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if decode_record(line) is not None:
                total_records += 1
    if total_records == 0:
        raise RuntimeError(f"No decode state records found in {log_path}")

    selected_record = total_records // 2 + 1
    current_record = 0
    selected_line = 0
    selected: dict[str, Any] | None = None
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            record = decode_record(line)
            if record is None:
                continue
            current_record += 1
            if current_record == selected_record:
                selected = record
                selected_line = line_number
                break
    if selected is None:
        raise RuntimeError(f"Failed to select decode record from {log_path}")

    sp_seq_lens = selected["sp_seq_lens"]
    validate_frame(sp_seq_lens, num_nodes)
    nano_output_path.parent.mkdir(parents=True, exist_ok=False)
    vllm_output_path.parent.mkdir(parents=True, exist_ok=True)
    nano_output_path.write_text(
        json.dumps({"sp_seq_lens": sp_seq_lens}, indent=2) + "\n",
        encoding="utf-8",
    )
    vllm_output_path.write_text(
        json.dumps(sp_seq_lens, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_sizes = [
        len(rank_sequences)
        for dp_group in sp_seq_lens
        for rank_sequences in dp_group
    ]
    return {
        "decode_records": total_records,
        "selected_decode_record": selected_record,
        "selected_log_line": selected_line,
        "min_rank_batch_size": min(batch_sizes),
        "max_rank_batch_size": max(batch_sizes),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workloads = selected_workloads(args)
    require_inputs(args, workloads)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_name
    try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        (run_dir / "itl").mkdir()
    except FileExistsError as error:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different --run-id."
        ) from error

    commands = {
        workload.name: benchmark_command(
            args, workload, run_dir / "itl" / f"{workload.name}.jsonl"
        )
        for workload in workloads
    }
    manifest: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "running",
        "system": "NanoDeploy",
        "purpose": "Fig. 15 E2E source workloads for profile inputs",
        "paper_faithful": args.num_nodes == PAPER_NUM_NODES,
        "num_nodes": args.num_nodes,
        "topology": {
            "attention_dp": args.num_nodes,
            "attention_sp": 8,
            "ffn_ep": args.num_nodes * GPUS_PER_NODE,
        },
        "scheduler": {
            "architecture": "legacy_global",
            "routing": "LeastBatch",
            "dynamic_sp_size_strategy": "bucket",
            "dynamic_sp_bucket_preset": "deepseek_v3",
        },
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "model_path": str(args.model_path.resolve()),
        "workloads": {
            workload.name: {
                "dataset": str(workload.dataset_path.resolve()),
                "request_rate": workload.request_rate,
                "num_requests": workload.num_requests,
                "max_num_seqs": workload.max_num_seqs,
                "command": commands[workload.name],
            }
            for workload in workloads
        },
        "snapshots": {},
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    print(f"NanoDeploy Fig. 15 E2E output: {run_dir}", flush=True)
    if args.dry_run:
        for workload in workloads:
            print(
                f"[dry-run] {workload.name}: "
                f"{shlex.join(commands[workload.name])}",
                flush=True,
            )
        print(f"NanoDeploy Fig. 15 E2E dry run completed: {run_dir}")
        return 0

    environment = benchmark_environment(args.nanodeploy_workdir)
    try:
        for index, workload in enumerate(workloads):
            print(f"\n===== START nano-e2e/{workload.name} =====", flush=True)
            log_path = run_dir / "logs" / f"{workload.name}.log"
            return_code = run_logged(
                commands[workload.name],
                log_path,
                environment,
                args.nanodeploy_workdir,
            )
            if return_code != 0:
                raise RuntimeError(
                    f"NanoDeploy E2E failed for {workload.name} "
                    f"(exit code {return_code}); see {log_path}"
                )
            nano_input_path = (
                run_dir
                / "inputs"
                / "nano"
                / workload.name
                / "processed_input_3d.json"
            )
            vllm_input_path = (
                run_dir / "inputs" / "vllm" / f"{workload.name}.json"
            )
            snapshot = extract_middle_inputs(
                log_path,
                nano_input_path,
                vllm_input_path,
                args.num_nodes,
            )
            snapshot["nano_input"] = str(nano_input_path)
            snapshot["vllm_input"] = str(vllm_input_path)
            manifest["snapshots"][workload.name] = snapshot
            write_manifest(manifest_path, manifest)
            print(
                f"===== DONE nano-e2e/{workload.name}: selected decode "
                f"record {snapshot['selected_decode_record']}/"
                f"{snapshot['decode_records']} =====",
                flush=True,
            )
            if index + 1 < len(workloads):
                time.sleep(COOLDOWN_SECONDS)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(f"\nAll NanoDeploy Fig. 15 E2E input runs completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

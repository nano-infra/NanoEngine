#!/usr/bin/env python3
"""Run the two NanoDeploy service cases used by Figure 14."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_BENCHMARK = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"
NANO_SERVICE_DRIVER = SCRIPT_DIR / "nano_service_driver.py"
DEFAULT_MODEL_PATH = Path(
    os.environ.get("FIG14_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"
COOLDOWN_SECONDS = 15
GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4
MIN_FREE_GPU_FRACTION = 0.95


@dataclass(frozen=True)
class Case:
    name: str
    dataset_path: Path
    request_rate: int
    duration_sec: int
    max_num_seqs: int

    @property
    def num_requests(self) -> int:
        return self.request_rate * self.duration_sec


CASES = (
    Case(
        name="lb",
        dataset_path=SCRIPT_DIR / "inputs" / "issue01.csv",
        request_rate=25,
        duration_sec=360,
        max_num_seqs=192,
    ),
    Case(
        name="hol",
        dataset_path=SCRIPT_DIR / "inputs" / "issue05.csv",
        request_rate=30,
        duration_sec=600,
        max_num_seqs=256,
    ),
)
CASE_BY_NAME = {case.name: case for case in CASES}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_BY_NAME),
        default=list(CASE_BY_NAME),
        help="Cases to run (default: lb hol).",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="Number of 8-GPU nodes to use (default: %(default)s).",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        help=(
            "Maximum prompt_len + output_len for an optional reduced run; "
            "disabled by default."
        ),
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("FIG14_NANO_RAY_ADDRESS", DEFAULT_RAY_ADDRESS),
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get(
            "FIG14_NANO_MASTER_ADDRESS", DEFAULT_MASTER_ADDRESS
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Short descriptive run ID, for example ae_fig14_1.",
    )
    args = parser.parse_args()
    if args.run_id and any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    return args


def validate_inputs(args: argparse.Namespace, cases: list[Case]) -> None:
    if not NANO_BENCHMARK.is_file():
        raise SystemExit(f"NanoDeploy E2E driver not found: {NANO_BENCHMARK}")
    if not NANO_SERVICE_DRIVER.is_file():
        raise SystemExit(f"Fig. 14 NanoDeploy adapter not found: {NANO_SERVICE_DRIVER}")
    spec = importlib.util.find_spec("nanodeploy")
    if spec is None:
        raise SystemExit(
            "nanodeploy is not installed in the active Python environment"
        )
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")
    missing = [str(case.dataset_path) for case in cases if not case.dataset_path.is_file()]
    if missing:
        raise SystemExit("Dataset CSV not found:\n  " + "\n  ".join(missing))


def validate_ray_gpu_memory(
    ray_address: str,
    master_address: str,
    num_nodes: int,
) -> None:
    """Reject Ray's logical GPU view when the physical GPUs are occupied."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    def query_local_gpu_memory() -> dict[str, Any]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        node_ip = ray.util.get_node_ip_address()
        if process.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi failed on {node_ip}: {process.stdout.strip()}"
            )

        gpus = []
        for line in process.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                raise RuntimeError(
                    f"unexpected nvidia-smi output on {node_ip}: {line!r}"
                )
            index, total_mib, free_mib = (int(field) for field in fields)
            gpus.append(
                {
                    "index": index,
                    "total_mib": total_mib,
                    "free_mib": free_mib,
                }
            )
        return {"node_ip": node_ip, "gpus": gpus}

    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        gpu_nodes = [
            node
            for node in ray.nodes()
            if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) > 0
        ]
        nodes_with_active_placement_groups: set[str] = set()
        for placement_group in ray.util.placement_group_table().values():
            if placement_group.get("state") == "REMOVED":
                continue
            nodes_with_active_placement_groups.update(
                node_id
                for node_id in placement_group.get(
                    "bundles_to_node_id", {}
                ).values()
                if node_id
            )
        gpu_nodes = [
            node
            for node in gpu_nodes
            if node["NodeID"] not in nodes_with_active_placement_groups
        ]
        master_ip = master_address.rsplit(":", 1)[0]
        gpu_nodes.sort(
            key=lambda node: (
                0 if node.get("NodeManagerAddress") == master_ip else 1
            )
        )
        if not gpu_nodes or gpu_nodes[0].get("NodeManagerAddress") != master_ip:
            raise RuntimeError(
                f"NanoDeploy master node {master_ip} is unavailable in Ray"
            )
        if len(gpu_nodes) < num_nodes:
            raise RuntimeError(
                f"Fig. 14 requires {num_nodes} available Ray GPU nodes, but "
                f"only {len(gpu_nodes)} were found"
            )
        selected_nodes = gpu_nodes[:num_nodes]
        expected_gpu_count = num_nodes * GPUS_PER_NODE
        ray_gpu_count = int(
            sum(node["Resources"].get("GPU", 0) for node in selected_nodes)
        )
        if ray_gpu_count < expected_gpu_count:
            raise RuntimeError(
                f"Fig. 14 requires {expected_gpu_count} Ray GPUs on the "
                f"selected {num_nodes} nodes, but they report {ray_gpu_count}"
            )

        inspect = ray.remote(num_cpus=0)(query_local_gpu_memory)
        reports = ray.get(
            [
                inspect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"], soft=False
                    )
                ).remote()
                for node in selected_nodes
            ]
        )
    finally:
        ray.shutdown()

    physical_gpu_count = sum(len(report["gpus"]) for report in reports)
    if physical_gpu_count < expected_gpu_count:
        raise RuntimeError(
            f"Fig. 14 requires {expected_gpu_count} physical GPUs, but "
            "nvidia-smi found "
            f"{physical_gpu_count} across the active Ray nodes"
        )

    occupied = []
    for report in reports:
        for gpu in report["gpus"]:
            free_fraction = gpu["free_mib"] / gpu["total_mib"]
            if free_fraction < MIN_FREE_GPU_FRACTION:
                occupied.append(
                    f"{report['node_ip']} GPU {gpu['index']}: "
                    f"{gpu['free_mib']} / {gpu['total_mib']} MiB free"
                )
    if occupied:
        raise RuntimeError(
            "Ray reports GPU resources that are physically occupied; clear or "
            "replace these GPUs before running Fig. 14:\n  "
            + "\n  ".join(occupied)
        )

    print(
        f"Ray GPU preflight passed: {physical_gpu_count} GPUs across "
        f"{len(reports)} nodes "
        f"({', '.join(report['node_ip'] for report in reports)})",
        flush=True,
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


def service_command(
    args: argparse.Namespace,
    case: Case,
    itl_path: Path,
) -> list[str]:
    max_model_len = args.max_request_tokens or 1_000_000
    return [
        sys.executable,
        "-u",
        str(NANO_SERVICE_DRIVER),
        "--dataset",
        "csv",
        "--csv-path",
        str(case.dataset_path),
        "--max-request-tokens",
        "0" if args.max_request_tokens is None else str(args.max_request_tokens),
        "--num-requests",
        str(case.num_requests),
        "--request-rate",
        str(case.request_rate),
        "--sp",
        "8",
        "--dp",
        str(args.num_nodes),
        "--ep",
        str(args.num_nodes * GPUS_PER_NODE),
        "--tp",
        "1",
        "--max-num-seqs",
        str(case.max_num_seqs),
        "--gpu-memory-limit-gb",
        "141",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(max_model_len),
        "--max-input-len",
        str(max_model_len),
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
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
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
                log_file.write(line)
                log_file.flush()
                if "sp_seq_lens" not in line:
                    print(line, end="", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot resume from manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid manifest object: {path}")
    return payload


def case_record(case: Case, command: list[str]) -> dict[str, Any]:
    return {
        "dataset": str(case.dataset_path.resolve()),
        "request_rate": case.request_rate,
        "duration_sec": case.duration_sec,
        "num_requests": case.num_requests,
        "max_num_seqs": case.max_num_seqs,
        "command": command,
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    expected: dict[str, Any],
    cases: list[Case],
) -> None:
    differing = [
        key
        for key in (
            "system",
            "figure",
            "topology",
            "ray_address",
            "master_address",
            "model_path",
            "nanodeploy_package",
            "max_request_tokens",
        )
        if manifest.get(key) != expected.get(key)
    ]
    recorded_run_id = manifest.get("run_id")
    if recorded_run_id is not None and recorded_run_id != expected["run_id"]:
        differing.append("run_id")

    existing_cases = manifest.get("cases")
    if not isinstance(existing_cases, dict):
        differing.append("cases")
    else:
        expected_cases = expected["cases"]
        for case in cases:
            existing = existing_cases.get(case.name)
            wanted = expected_cases[case.name]
            if existing is not None and any(
                existing.get(key) != wanted.get(key)
                for key in (
                    "dataset",
                    "request_rate",
                    "duration_sec",
                    "num_requests",
                    "max_num_seqs",
                    "command",
                )
            ):
                differing.append(f"cases.{case.name}")

    if differing:
        raise RuntimeError(
            "cannot resume Fig. 14 NanoDeploy with different settings ("
            + ", ".join(sorted(set(differing)))
            + "); choose a fresh --run-id"
        )


def case_is_complete(
    case: Case,
    record: dict[str, Any] | None,
    log_path: Path,
    itl_path: Path,
) -> bool:
    if record is None or record.get("status") != "completed":
        return False
    if any(
        not path.is_file() or path.stat().st_size == 0
        for path in (log_path, itl_path)
    ):
        return False

    completed_requests: int | None = None
    with log_path.open(encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            if line.startswith("Requests completed:"):
                try:
                    completed_requests = int(line.split(":", 1)[1].strip())
                except ValueError:
                    return False
    if completed_requests != case.num_requests:
        return False

    with itl_path.open(encoding="utf-8", errors="ignore") as itl_file:
        row_count = sum(1 for line in itl_file if line.strip())
    return row_count == case.num_requests


def archive_incomplete_output(path: Path) -> None:
    if not path.exists():
        return
    attempt = 1
    while True:
        archived = path.with_name(
            f"{path.stem}.incomplete{attempt}{path.suffix}"
        )
        if not archived.exists():
            path.rename(archived)
            return
        attempt += 1


def main() -> int:
    args = parse_args()
    cases = [CASE_BY_NAME[name] for name in args.cases]
    validate_inputs(args, cases)
    run_dir = args.output_root.resolve() / args.run_id / "nano"
    commands = {
        case.name: service_command(args, case, run_dir / case.name / "itl.jsonl")
        for case in cases
    }
    package_spec = importlib.util.find_spec("nanodeploy")
    assert package_spec is not None and package_spec.origin is not None
    expected_manifest: dict[str, Any] = {
        "status": "running",
        "run_id": args.run_id,
        "started_at": utc_now(),
        "system": "NanoDeploy",
        "figure": 14,
        "topology": {
            "attention_dp": args.num_nodes,
            "attention_sp": 8,
            "ffn_ep": args.num_nodes * GPUS_PER_NODE,
        },
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "model_path": str(args.model_path.resolve()),
        "nanodeploy_package": str(package_spec.origin),
        "max_request_tokens": args.max_request_tokens,
        "cases": {
            case.name: case_record(case, commands[case.name])
            for case in cases
        },
    }
    manifest_path = run_dir / "manifest.json"
    resuming = run_dir.exists()
    if resuming:
        if not manifest_path.is_file():
            raise RuntimeError(
                f"cannot resume without NanoDeploy manifest: {manifest_path}"
            )
        manifest = load_manifest(manifest_path)
        validate_resume_manifest(manifest, expected_manifest, cases)
        manifest["run_id"] = args.run_id
        manifest["status"] = "running"
        manifest["last_resumed_at"] = utc_now()
        manifest.pop("error", None)
        manifest.pop("finished_at", None)
        existing_cases = manifest["cases"]
        for case in cases:
            if case.name not in existing_cases:
                existing_cases[case.name] = expected_manifest["cases"][case.name]
        print(f"Resuming NanoDeploy Fig. 14 run ID: {args.run_id}", flush=True)
    else:
        run_dir.mkdir(parents=True)
        manifest = expected_manifest
    write_manifest(manifest_path, manifest)

    print(f"NanoDeploy Fig. 14 output: {run_dir}", flush=True)
    completed_cases: set[str] = set()
    for case in cases:
        case_dir = run_dir / case.name
        if case_is_complete(
            case,
            manifest["cases"].get(case.name),
            case_dir / "driver.log",
            case_dir / "itl.jsonl",
        ):
            completed_cases.add(case.name)
    if completed_cases:
        print(
            f"Resume check: {len(completed_cases)} complete NanoDeploy "
            "case(s); skipping them",
            flush=True,
        )

    pending_cases = [case for case in cases if case.name not in completed_cases]
    try:
        if pending_cases:
            validate_ray_gpu_memory(
                args.ray_address,
                args.master_address,
                args.num_nodes,
            )
        environment = service_environment()
        for index, case in enumerate(pending_cases):
            case_dir = run_dir / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            log_path = case_dir / "driver.log"
            itl_path = case_dir / "itl.jsonl"
            archive_incomplete_output(log_path)
            archive_incomplete_output(itl_path)
            manifest["cases"][case.name]["status"] = "running"
            manifest["cases"][case.name]["started_at"] = utc_now()
            manifest["cases"][case.name].pop("error", None)
            write_manifest(manifest_path, manifest)
            print(f"\n===== START nano/{case.name} =====", flush=True)
            return_code = run_logged(
                commands[case.name],
                log_path,
                environment,
            )
            if return_code != 0:
                raise RuntimeError(
                    f"NanoDeploy case {case.name} failed with exit code "
                    f"{return_code}; see {log_path}"
                )
            if not case_is_complete(
                case,
                {"status": "completed"},
                log_path,
                itl_path,
            ):
                raise RuntimeError(
                    f"NanoDeploy case {case.name} exited successfully but its "
                    "log/ITL output is incomplete"
                )
            manifest["cases"][case.name]["status"] = "completed"
            manifest["cases"][case.name]["log"] = str(log_path)
            manifest["cases"][case.name]["itl"] = str(itl_path)
            manifest["cases"][case.name]["finished_at"] = utc_now()
            write_manifest(manifest_path, manifest)
            if index + 1 < len(pending_cases):
                time.sleep(COOLDOWN_SECONDS)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished_at"] = utc_now()
        if "case" in locals():
            manifest["cases"][case.name]["status"] = "failed"
            manifest["cases"][case.name]["error"] = str(error)
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now()
    write_manifest(manifest_path, manifest)
    print(f"\nAll NanoDeploy Fig. 14 cases completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

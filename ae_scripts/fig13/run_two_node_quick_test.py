#!/usr/bin/env python3
"""Run the two-node Fig. 13 profiler quick test."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

DEFAULT_VLLM_ROOT = Path(require_path("AE_VLLM_ROOT"))
DEFAULT_MODEL_PATH = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"
DEFAULT_VLLM_MASTER_ADDRESS = "10.102.252.174"
DEFAULT_VLLM_REMOTE_HOST = "h200-rjob1"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "quick_test_results"
VLLM_RUNNER = AE_ROOT / "start-e2e" / "vllm" / "manual_multinode_poisson_runner.py"
NANO_PROFILER = AE_ROOT / "start-profile" / "nano_dummy_prefill_profile.py"

NODE_COUNT = 2
GPUS_PER_NODE = 8
WORLD_SIZE = NODE_COUNT * GPUS_PER_NODE
LONG_LENGTH = 512 * 1024
SHORT_LENGTH = 2 * 1024
SHORTS_PER_GPU = 64
SHORTS_PER_NODE = GPUS_PER_NODE * SHORTS_PER_GPU
OUTPUT_LENGTH = 64
MAX_MODEL_LEN = 720_000
GPU_MEMORY_UTILIZATION = 0.9
VLLM_CONFIGS = {
    "dp2dcp8": {"dp_size": 2, "rpc_port": 29550},
    "dp4dcp4": {"dp_size": 4, "rpc_port": 29551},
    "dp8cp2": {"dp_size": 8, "rpc_port": 29552},
    "dp16": {"dp_size": 16, "rpc_port": 29553},
}
VLLM_STRATEGIES = tuple(VLLM_CONFIGS)
NETWORK_ENV_DEFAULTS = {
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_SOCKET_IFNAME": "bond0",
    "SLIME_VISIBLE_DEVICES": (
        "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
    ),
    "SLIME_GID_INDEX": "3",
    "SLIME_QP_NUM": "4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("nano", "vllm"),
        default=["nano", "vllm"],
        help="Systems to test (default: nano vllm).",
    )
    parser.add_argument("--ray-address", default=DEFAULT_RAY_ADDRESS)
    parser.add_argument("--master-address", default=DEFAULT_MASTER_ADDRESS)
    parser.add_argument(
        "--vllm-master-address", default=DEFAULT_VLLM_MASTER_ADDRESS
    )
    parser.add_argument("--vllm-remote-host", default=DEFAULT_VLLM_REMOTE_HOST)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id", default=datetime.now().strftime("quick_%Y%m%d_%H%M%S")
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--nano-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--nano-input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, '.', '_', and '-'")
    if args.nano_worker and (args.run_dir is None or args.nano_input is None):
        parser.error("internal NanoDeploy worker requires --run-dir and --nano-input")
    return args


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def quick_lengths() -> list[int]:
    lengths: list[int] = []
    for _node_index in range(NODE_COUNT):
        lengths.append(LONG_LENGTH)
        lengths.extend([SHORT_LENGTH] * SHORTS_PER_NODE)
    return lengths


def write_rank_lengths(path: Path, dp_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("prompt_len", "output_len", "data_parallel_rank"),
        )
        writer.writeheader()
        if dp_size not in (2, 4, 8, WORLD_SIZE):
            raise ValueError(f"unsupported quick-test DP size: {dp_size}")
        dp_size_local = dp_size // NODE_COUNT
        for node_index in range(NODE_COUNT):
            first_rank = node_index * dp_size_local
            writer.writerow(
                {
                    "prompt_len": LONG_LENGTH,
                    "output_len": OUTPUT_LENGTH,
                    "data_parallel_rank": first_rank,
                }
            )
            for short_index in range(SHORTS_PER_NODE):
                writer.writerow(
                    {
                        "prompt_len": SHORT_LENGTH,
                        "output_len": OUTPUT_LENGTH,
                        "data_parallel_rank": (
                            first_rank + short_index % dp_size_local
                        ),
                    }
                )


def prepare_inputs(run_dir: Path, model_path: Path) -> dict[str, Path]:
    input_root = run_dir / "inputs"
    nano_input = input_root / "nano" / "quick_2node" / "processed_input_3d.json"
    write_json(nano_input, {"sp_seq_lens": [[quick_lengths()]]})

    vllm_inputs: dict[str, Path] = {}
    for strategy, config in VLLM_CONFIGS.items():
        input_path = input_root / "vllm" / f"quick_{strategy}.lengths.csv"
        write_rank_lengths(input_path, config["dp_size"])
        vllm_inputs[strategy] = input_path

    case_csv = input_root / "vllm" / "quick_cases.csv"
    fieldnames = (
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
    with case_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        common = {
            "enabled": 1,
            "cluster": "2node_h200",
            "model": str(model_path.resolve()),
            "dispatch_policy": "waiting_x4_plus_running",
            "request_rate": 40,
            "rate_phase": "fig13_quick",
            "max_num_seqs": 64,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_requests": "csv_rows",
            "warmup_requests": 0,
            "max_model_len": MAX_MODEL_LEN,
            "reason": "Fig. 13 two-node profiler quick test",
            "historical_reference": "",
        }
        for strategy, config in VLLM_CONFIGS.items():
            writer.writerow(
                {
                    **common,
                    "name": f"fig13_quick_{strategy}",
                    "dataset": str(vllm_inputs[strategy].resolve()),
                    "strategy": strategy,
                    "data_parallel_rpc_port": config["rpc_port"],
                }
            )
    return {"nano": nano_input, "vllm_cases": case_csv}


def require_path(path: Path, label: str, *, directory: bool = False) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise SystemExit(f"{label} {kind} not found: {path}")


def validate_static_inputs(args: argparse.Namespace) -> None:
    require_path(args.model_path / "config.json", "model config")
    if "nano" in args.systems:
        require_path(NANO_PROFILER, "AE NanoDeploy profiler")
        if importlib.util.find_spec("nanodeploy") is None:
            raise SystemExit(
                "the nanodeploy package is not installed in the active Python environment"
            )
    if "vllm" in args.systems:
        require_path(args.vllm_root, "vLLM checkout", directory=True)
        require_path(VLLM_RUNNER, "AE vLLM runner")


def validate_ray_topology(address: str) -> list[dict[str, object]]:
    try:
        import ray
    except ImportError as exc:
        raise SystemExit("Ray is required for the NanoDeploy quick test") from exc

    ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
    try:
        gpu_nodes = [
            node
            for node in ray.nodes()
            if node.get("Alive") and node.get("Resources", {}).get("GPU", 0)
        ]
        details = [
            {
                "ip": node.get("NodeManagerAddress"),
                "gpus": int(node.get("Resources", {}).get("GPU", 0)),
            }
            for node in gpu_nodes
        ]
        gpu_count = sum(int(item["gpus"]) for item in details)
        if len(details) != NODE_COUNT or gpu_count != WORLD_SIZE:
            raise SystemExit(
                "Fig. 13 quick test requires exactly two Ray GPU nodes and "
                f"16 GPUs; found {details}"
            )
        if any(item["gpus"] != GPUS_PER_NODE for item in details):
            raise SystemExit(
                f"Fig. 13 quick test requires 8 GPUs per Ray node; found {details}"
            )
        return details
    finally:
        ray.shutdown()


def nano_worker_command(args: argparse.Namespace, run_dir: Path, input_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--nano-worker",
        "--run-dir",
        str(run_dir),
        "--nano-input",
        str(input_path),
        "--model-path",
        str(args.model_path),
        "--ray-address",
        args.ray_address,
        "--master-address",
        args.master_address,
    ]


def vllm_command(args: argparse.Namespace, run_dir: Path, case_csv: Path) -> list[str]:
    command = [
        sys.executable,
        str(VLLM_RUNNER),
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(run_dir / "vllm"),
        "--run-label",
        "fig13_2node_quick",
        "--ignore-historical-skips",
        "--no-keep-going",
        "--frontend-extra-arg=--routing-mode",
        "--frontend-extra-arg=explicit_rank_replay",
        "--frontend-extra-arg=--profile-after-warmup",
        "--frontend-extra-arg=--profiler-config.profiler",
        "--frontend-extra-arg=torch",
        "--frontend-extra-arg=--profiler-config.torch_profiler_dir",
        "--frontend-extra-arg={benchmark_dir}/torch_profiler",
        "--frontend-extra-arg=--profiler-config.ignore_frontend",
        "--frontend-extra-arg=true",
        "--frontend-extra-arg=--profiler-config.delay_iterations",
        "--frontend-extra-arg=2",
        "--frontend-extra-arg=--profiler-config.max_iterations",
        "--frontend-extra-arg=2",
        "--frontend-extra-arg=--profiler-config.wait_iterations",
        "--frontend-extra-arg=0",
        "--frontend-extra-arg=--profiler-config.warmup_iterations",
        "--frontend-extra-arg=0",
        "--headless-extra-arg=--profiler-config.profiler",
        "--headless-extra-arg=torch",
        "--headless-extra-arg=--profiler-config.torch_profiler_dir",
        "--headless-extra-arg={benchmark_dir}/torch_profiler",
        "--headless-extra-arg=--profiler-config.delay_iterations",
        "--headless-extra-arg=2",
        "--headless-extra-arg=--profiler-config.max_iterations",
        "--headless-extra-arg=2",
        "--headless-extra-arg=--profiler-config.wait_iterations",
        "--headless-extra-arg=0",
        "--headless-extra-arg=--profiler-config.warmup_iterations",
        "--headless-extra-arg=0",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def tee_process(
    command: list[str], log_path: Path, *, env: dict[str, str] | None = None
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def trace_ranks(paths: list[Path]) -> set[int]:
    ranks: set[int] = set()
    for path in paths:
        match = re.search(r"(?:rank_|rank)(\d+)", path.name)
        if match:
            ranks.add(int(match.group(1)))
    return ranks


def validate_nano_traces(run_dir: Path) -> int:
    traces = sorted((run_dir / "nano").rglob("*.pt.trace.json"))
    ranks = trace_ranks(traces)
    if len(traces) != WORLD_SIZE or ranks != set(range(WORLD_SIZE)):
        raise RuntimeError(
            "NanoDeploy quick test expected one trace for each of 16 ranks; "
            f"found {len(traces)} traces and ranks {sorted(ranks)}"
        )
    return len(traces)


def validate_vllm_traces(run_dir: Path) -> int:
    traces = sorted((run_dir / "vllm").rglob("*.pt.trace.json"))
    traces.extend(sorted((run_dir / "vllm").rglob("*.pt.trace.json.gz")))
    expected = WORLD_SIZE * len(VLLM_STRATEGIES)
    if len(traces) != expected:
        raise RuntimeError(
            f"vLLM quick test expected {expected} rank traces, found {len(traces)}"
        )
    for strategy in VLLM_STRATEGIES:
        strategy_traces = [
            path
            for path in traces
            if any(part.startswith(f"{strategy}-") for part in path.parts)
        ]
        if len(strategy_traces) != WORLD_SIZE:
            raise RuntimeError(
                f"vLLM {strategy} expected 16 rank traces, "
                f"found {len(strategy_traces)}"
            )
    return len(traces)


def run_nano_worker(args: argparse.Namespace) -> None:
    assert args.run_dir is not None and args.nano_input is not None
    sys.path.insert(0, str(NANO_PROFILER.parent))

    import nano_dummy_prefill_profile as profiler
    from nanodeploy.engine import ray_executor

    original_parse_args = profiler.parse_args
    original_get_config_params = profiler.get_config_params
    original_get_nodes = ray_executor.get_available_nodes_with_master_first
    original_placement_group = ray_executor.placement_group
    original_model_runner = ray_executor.ModelRunner
    target_node_ips: dict[str, str] = {}

    class ModelRunnerWithNetworkEnv:
        @staticmethod
        def options(**kwargs):
            runtime_env = dict(kwargs.get("runtime_env") or {})
            env_vars = dict(runtime_env.get("env_vars") or {})
            env_vars.update({key: os.environ[key] for key in NETWORK_ENV_DEFAULTS})
            runtime_env["env_vars"] = env_vars
            kwargs["runtime_env"] = runtime_env
            return original_model_runner.options(**kwargs)

    def get_exact_two_nodes(master_address: str):
        nodes = original_get_nodes(master_address)
        if len(nodes) != NODE_COUNT:
            details = [node.get("NodeManagerAddress") for node in nodes]
            raise RuntimeError(
                f"expected exactly two Ray nodes, found {len(nodes)}: {details}"
            )
        target_node_ips.update(
            {node["NodeID"]: node["NodeManagerAddress"] for node in nodes}
        )
        return nodes

    def hard_pinned_placement_group(
        bundles, *, strategy, name, _soft_target_node_id=None, **kwargs
    ):
        target_ip = target_node_ips.get(_soft_target_node_id)
        if target_ip is None:
            raise RuntimeError(f"unknown target Ray node: {_soft_target_node_id}")
        pinned_bundles = [dict(bundle) for bundle in bundles]
        pinned_bundles[0][f"node:{target_ip}"] = 0.001
        return original_placement_group(
            pinned_bundles, strategy=strategy, name=name, **kwargs
        )

    def parse_profiler_args():
        parsed = original_parse_args()
        if parsed.config != "dp4sp8":
            raise SystemExit("the two-node adapter expects --config dp4sp8")
        parsed.config = "dp2sp8"
        return parsed

    def get_config_params(config_name: str) -> dict[str, int]:
        if config_name != "dp2sp8":
            return original_get_config_params(config_name)
        return {
            "attention_dp": 2,
            "attention_sp": 8,
            "attention_tp": 1,
            "ffn_dp": 1,
            "ffn_ep": 16,
            "ffn_tp": 1,
        }

    profiler.parse_args = parse_profiler_args
    profiler.get_config_params = get_config_params
    ray_executor.get_available_nodes_with_master_first = get_exact_two_nodes
    ray_executor.placement_group = hard_pinned_placement_group
    ray_executor.ModelRunner = ModelRunnerWithNetworkEnv

    sys.argv = [
        str(NANO_PROFILER),
        "--config",
        "dp4sp8",
        "--sp-backend",
        "hao_basic",
        "--sp-size-policy",
        "long_short",
        "--long-request-sp-threshold",
        "100000",
        "--model-path",
        str(args.model_path),
        "--master-address",
        args.master_address,
        "--ray-address",
        args.ray_address,
        "--sp-seq-lens-file",
        str(args.nano_input),
        "--profiler-dir",
        str(args.run_dir / "nano"),
        "--profiler-start-step",
        "2",
        "--profiling-step",
        "2",
        "--max-tokens",
        str(OUTPUT_LENGTH),
        "--loop-count",
        "16",
        "--max-num-seqs",
        "64",
        "--max-num-recv-seqs",
        "32",
        "--max-num-send-seqs",
        "32",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
    ]
    profiler.main()


def run_quick_test(args: argparse.Namespace) -> None:
    args.vllm_root = args.vllm_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_static_inputs(args)
    run_dir = args.output_root / args.run_id
    try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"run directory already exists: {run_dir}\nChoose a different --run-id."
        ) from exc

    inputs = prepare_inputs(run_dir, args.model_path)
    nano_command = nano_worker_command(args, run_dir, inputs["nano"])
    vllm_launch = vllm_command(args, run_dir, inputs["vllm_cases"])
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "purpose": "Fig. 13 two-node profiler pipeline quick test",
        "paper_reproduction": False,
        "systems": args.systems,
        "topology": {"nodes": NODE_COUNT, "gpus_per_node": GPUS_PER_NODE},
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "vllm_master_address": args.vllm_master_address,
        "vllm_remote_host": args.vllm_remote_host,
        "workload": {
            "long_requests_per_node": 1,
            "long_request_length": LONG_LENGTH,
            "short_requests_per_gpu": SHORTS_PER_GPU,
            "short_requests_per_node": SHORTS_PER_NODE,
            "short_request_length": SHORT_LENGTH,
            "output_length": OUTPUT_LENGTH,
        },
        "nano_strategy": "dp2sp8",
        "vllm_strategies": list(VLLM_STRATEGIES),
        "commands": {
            "nano": nano_command,
            "vllm": vllm_launch,
        },
        "trace_counts": {},
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"Fig. 13 two-node quick-test output: {run_dir}", flush=True)
    if args.dry_run:
        if "nano" in args.systems:
            print(f"[dry-run] {shlex.join(nano_command)}", flush=True)
        if "vllm" in args.systems:
            env = os.environ.copy()
            env["VLLM_WORKDIR"] = str(args.vllm_root.resolve())
            env["VLLM_2NODE_H200_MASTER_ADDR"] = args.vllm_master_address
            env["VLLM_2NODE_H200_REMOTE_HOST"] = args.vllm_remote_host
            return_code = tee_process(
                vllm_launch, run_dir / "logs" / "vllm.log", env=env
            )
            if return_code != 0:
                manifest["status"] = "failed"
                manifest["error"] = (
                    f"vLLM dry-run failed with exit code {return_code}"
                )
                write_json(manifest_path, manifest)
                raise SystemExit(manifest["error"])
        print("Dry-run completed; no GPU processes were launched.", flush=True)
        return

    try:
        if "nano" in args.systems:
            manifest["ray_nodes"] = validate_ray_topology(args.ray_address)
            write_json(manifest_path, manifest)
            env = os.environ.copy()
            env.update(NETWORK_ENV_DEFAULTS)
            env["PYTHONUNBUFFERED"] = "1"
            print(f"[run] {shlex.join(nano_command)}", flush=True)
            return_code = tee_process(
                nano_command, run_dir / "logs" / "nano.log", env=env
            )
            if return_code != 0:
                raise RuntimeError(
                    f"NanoDeploy quick test failed with exit code {return_code}"
                )
            manifest["trace_counts"]["nano"] = validate_nano_traces(run_dir)
            write_json(manifest_path, manifest)

        if "vllm" in args.systems:
            env = os.environ.copy()
            env["VLLM_WORKDIR"] = str(args.vllm_root.resolve())
            env["VLLM_2NODE_H200_MASTER_ADDR"] = args.vllm_master_address
            env["VLLM_2NODE_H200_REMOTE_HOST"] = args.vllm_remote_host
            print(f"[run] {shlex.join(vllm_launch)}", flush=True)
            return_code = tee_process(
                vllm_launch, run_dir / "logs" / "vllm.log", env=env
            )
            if return_code != 0:
                raise RuntimeError(
                    f"vLLM quick test failed with exit code {return_code}"
                )
            manifest["trace_counts"]["vllm"] = validate_vllm_traces(run_dir)
            write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    print(
        "Fig. 13 two-node quick test completed: "
        f"trace_counts={manifest['trace_counts']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.nano_worker:
        run_nano_worker(args)
    else:
        run_quick_test(args)


if __name__ == "__main__":
    main()

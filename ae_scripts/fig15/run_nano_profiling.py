#!/usr/bin/env python3
"""Replay Fig. 15 NanoDeploy E2E snapshots and collect profiler traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_PROFILER = AE_ROOT / "start-profile" / "nano_dummy_prefill_profile.py"
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "inputs" / "nano"
DEFAULT_MODEL_PATH = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "profiling_results" / "nano"
DEFAULT_DATASETS = ("short", "issue01", "issue05")
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
DEFAULT_MASTER_ADDRESS = "10.102.252.174:29500"
GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4
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
    env_node_ips = [
        value.strip()
        for value in os.environ.get("FIG15_NANO_NODE_IPS", "").split(",")
        if value.strip()
    ]
    parser = argparse.ArgumentParser(
        description=(
            "Replay the NanoDeploy E2E snapshots for Fig. 15 and save per-rank "
            "profiler traces under profiling_results/nano."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("FIG15_NANO_RAY_ADDRESS", DEFAULT_RAY_ADDRESS),
        help=(
            f"Ray head address (default: {DEFAULT_RAY_ADDRESS}); override with "
            "FIG15_NANO_RAY_ADDRESS"
        ),
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get(
            "FIG15_NANO_MASTER_ADDRESS", DEFAULT_MASTER_ADDRESS
        ),
        help=(
            f"Distributed master host:port (default: {DEFAULT_MASTER_ADDRESS}); "
            "override with FIG15_NANO_MASTER_ADDRESS"
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="Number of 8-GPU nodes to use (default: %(default)s).",
    )
    parser.add_argument(
        "--node-ips",
        nargs="+",
        metavar="NODE_IP",
        default=env_node_ips or None,
        help=(
            "Optional ordered Ray node IPs, with the head first. The number "
            "of values must equal --num-nodes; alternatively set the "
            "comma-separated FIG15_NANO_NODE_IPS environment variable."
        ),
    )
    parser.add_argument("--max-num-seqs", type=int, default=192)
    parser.add_argument(
        "--run-id",
        "--run-name",
        dest="run_name",
        help="Result directory ID (default: run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-dataset", choices=DEFAULT_DATASETS, help=argparse.SUPPRESS
    )
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.node_ips and len(args.node_ips) != args.num_nodes:
        parser.error(
            f"--num-nodes {args.num_nodes} requires exactly "
            f"{args.num_nodes} --node-ips values"
        )
    return args


def dataset_input(input_root: Path, dataset: str) -> Path:
    return input_root / dataset / "processed_input_3d.json"


def validate_inputs(args: argparse.Namespace) -> None:
    if not NANO_PROFILER.is_file():
        raise SystemExit(f"AE NanoDeploy profiler not found: {NANO_PROFILER}")
    if importlib.util.find_spec("nanodeploy") is None:
        raise SystemExit(
            "the nanodeploy package is not installed in the active Python environment"
        )
    if not args.model_path.is_dir():
        raise SystemExit(f"Model metadata directory not found: {args.model_path}")
    missing = [
        str(dataset_input(args.input_root, dataset))
        for dataset in args.datasets
        if not dataset_input(args.input_root, dataset).is_file()
    ]
    if missing:
        raise SystemExit("Missing input JSON:\n  " + "\n  ".join(missing))


def network_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(NETWORK_ENV_DEFAULTS)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def worker_command(args: argparse.Namespace, dataset: str, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-dataset",
        dataset,
        "--run-dir",
        str(run_dir),
        "--input-root",
        str(args.input_root),
        "--model-path",
        str(args.model_path),
        "--ray-address",
        args.ray_address,
        "--master-address",
        args.master_address,
        "--num-nodes",
        str(args.num_nodes),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.node_ips:
        command.extend(("--node-ips", *args.node_ips))
    return command


def tee_process(command: list[str], log_path: Path, env: dict[str, str]) -> int:
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
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def config_name(num_nodes: int) -> str:
    return f"dp{num_nodes}sp8"


def trace_dir(run_dir: Path, dataset: str, num_nodes: int) -> Path:
    return (
        run_dir
        / "traces"
        / dataset
        / config_name(num_nodes)
        / "bucket_deepseek_v3"
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    validate_inputs(args)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_name
    try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different --run-id."
        ) from exc

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "system": "NanoDeploy",
        "purpose": "Fig. 15 replay of E2E state and trace collection",
        "paper_faithful": args.num_nodes == PAPER_NUM_NODES,
        "num_nodes": args.num_nodes,
        "topology": {
            "attention_dp": args.num_nodes,
            "attention_sp": 8,
            "ffn_ep": args.num_nodes * GPUS_PER_NODE,
        },
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "node_ips": args.node_ips or f"auto: first {args.num_nodes} Ray nodes",
        "datasets": args.datasets,
        "input_root": str(args.input_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "max_num_seqs": args.max_num_seqs,
        "sp_backend": "hao_basic",
        "sp_size_policy": "bucket",
        "dynamic_sp_bucket_preset": "deepseek_v3",
        "segment_size": 65536,
        "profiler_start_step": 3,
        "profiling_steps": 3,
        "max_tokens": 128,
        "loop_count": 16,
        "dry_run": args.dry_run,
        "commands": {
            dataset: worker_command(args, dataset, run_dir)
            for dataset in args.datasets
        },
        "trace_counts": {},
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    print(f"NanoDeploy Fig. 15 profiling output: {run_dir}", flush=True)
    if args.dry_run:
        for dataset in args.datasets:
            print(
                f"[dry-run] {dataset}: "
                f"{shlex.join(worker_command(args, dataset, run_dir))}",
                flush=True,
            )
        print(f"NanoDeploy Fig. 15 dry run completed: {run_dir}", flush=True)
        return

    env = network_environment()
    try:
        for dataset in args.datasets:
            log_path = run_dir / "logs" / f"nano_{dataset}.log"
            print(f"\n===== START nano/{dataset} =====", flush=True)
            return_code = tee_process(
                worker_command(args, dataset, run_dir), log_path, env
            )
            if return_code != 0:
                raise RuntimeError(
                    f"NanoDeploy profiling failed for {dataset} "
                    f"(exit code {return_code}); see {log_path}"
                )

            expected_traces = args.num_nodes * GPUS_PER_NODE
            traces = list(
                trace_dir(run_dir, dataset, args.num_nodes).rglob(
                    "*.pt.trace.json"
                )
            )
            if len(traces) != expected_traces:
                raise RuntimeError(
                    f"Expected {expected_traces} traces for {dataset}, found "
                    f"{len(traces)} in "
                    f"{trace_dir(run_dir, dataset, args.num_nodes)}"
                )
            manifest["trace_counts"][dataset] = len(traces)
            write_manifest(manifest_path, manifest)
            print(
                f"===== DONE nano/{dataset}: {expected_traces} traces =====",
                flush=True,
            )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(f"\nAll NanoDeploy profiling runs completed: {run_dir}", flush=True)


def run_worker(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        raise SystemExit("--run-dir is required for an internal worker run")

    sys.path.insert(0, str(NANO_PROFILER.parent))

    import nano_dummy_prefill_profile as profiler
    from nanodeploy.engine import ray_executor

    original_parse_args = profiler.parse_args
    original_get_config_params = profiler.get_config_params
    original_load_sp_seq_lens = profiler.load_sp_seq_lens
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

    def get_selected_nodes(master_address: str):
        nodes = original_get_nodes(master_address)
        if args.node_ips:
            by_ip = {node["NodeManagerAddress"]: node for node in nodes}
            missing = [
                address for address in args.node_ips if address not in by_ip
            ]
            if missing:
                raise RuntimeError(
                    f"Required Ray nodes are unavailable: {missing}"
                )
            selected = [by_ip[address] for address in args.node_ips]
        else:
            if len(nodes) < args.num_nodes:
                details = [node.get("NodeManagerAddress") for node in nodes]
                raise RuntimeError(
                    f"Fig. 15 requires {args.num_nodes} Ray GPU nodes; found "
                    f"{len(nodes)}: {details}. Pass {args.num_nodes} ordered "
                    "addresses through --node-ips to select nodes explicitly."
                )
            selected = nodes[: args.num_nodes]
        target_node_ips.update(
            {node["NodeID"]: node["NodeManagerAddress"] for node in selected}
        )
        return selected

    def hard_pinned_placement_group(
        bundles, *, strategy, name, _soft_target_node_id=None, **kwargs
    ):
        target_ip = target_node_ips.get(_soft_target_node_id)
        if target_ip is None:
            raise RuntimeError(f"Unknown target Ray node: {_soft_target_node_id}")
        pinned_bundles = [dict(bundle) for bundle in bundles]
        pinned_bundles[0][f"node:{target_ip}"] = 0.001
        return original_placement_group(
            pinned_bundles, strategy=strategy, name=name, **kwargs
        )

    def parse_profiler_args():
        parsed = original_parse_args()
        if parsed.config != "dp4sp8":
            raise SystemExit(
                "Fig. 15 NanoDeploy profiling requires --config dp4sp8"
            )
        parsed.config = config_name(args.num_nodes)
        parsed.sp_size_policy = "bucket"
        return parsed

    def get_config_params(selected_config: str) -> dict[str, int]:
        if selected_config == config_name(args.num_nodes):
            return {
                "attention_dp": args.num_nodes,
                "attention_sp": 8,
                "attention_tp": 1,
                "ffn_dp": 1,
                "ffn_ep": args.num_nodes * GPUS_PER_NODE,
                "ffn_tp": 1,
            }
        return original_get_config_params(selected_config)

    def load_sp_seq_lens(input_path: str | None):
        values = original_load_sp_seq_lens(input_path)
        if len(values) < args.num_nodes:
            raise ValueError(
                f"profile input contains {len(values)} DP groups; "
                f"--num-nodes {args.num_nodes} requires at least {args.num_nodes}"
            )
        if len(values) > args.num_nodes:
            print(
                f"Using the first {args.num_nodes} of {len(values)} input DP "
                "groups for this reduced-node run",
                flush=True,
            )
            values = values[: args.num_nodes]
        return values

    def resolve_bucket_policy(parsed, attention_sp: int):
        if attention_sp != 8:
            raise ValueError("Fig. 15 bucket replay requires attention_sp=8")
        return (
            "bucket",
            "bucket_deepseek_v3",
            {
                "segment_size": parsed.segment_size,
                "dynamic_sp_size_strategy": "bucket",
                "dynamic_sp_bucket_preset": "deepseek_v3",
                "fixed_sp_size": 0,
            },
        )

    profiler.parse_args = parse_profiler_args
    profiler.get_config_params = get_config_params
    profiler.load_sp_seq_lens = load_sp_seq_lens
    profiler.resolve_sp_policy = resolve_bucket_policy
    ray_executor.get_available_nodes_with_master_first = get_selected_nodes
    ray_executor.placement_group = hard_pinned_placement_group
    ray_executor.ModelRunner = ModelRunnerWithNetworkEnv

    input_json = dataset_input(args.input_root, args.worker_dataset)
    sys.argv = [
        str(NANO_PROFILER),
        "--config",
        "dp4sp8",
        "--sp-backend",
        "hao_basic",
        "--sp-size-policy",
        "legacy",
        "--segment-size",
        "65536",
        "--long-request-sp-threshold",
        "100000",
        "--model-path",
        str(args.model_path),
        "--master-address",
        args.master_address,
        "--ray-address",
        args.ray_address,
        "--sp-seq-lens-file",
        str(input_json),
        "--profiler-dir",
        str(args.run_dir / "traces"),
        "--profiler-start-step",
        "3",
        "--profiling-step",
        "3",
        "--max-tokens",
        "128",
        "--loop-count",
        "16",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-recv-seqs",
        "70",
        "--max-num-send-seqs",
        "70",
        "--max-model-len",
        "1000000",
    ]
    profiler.main()


def main() -> None:
    args = parse_args()
    if args.worker_dataset:
        run_worker(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()

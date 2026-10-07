#!/usr/bin/env python3
"""Profile the Fig. 15 E2E-selected states with vLLM on four nodes.

The paper configuration is four 8-GPU H200 nodes.  A two-node compatibility
mode is provided for launch-path validation on the currently available pair.
It deliberately refuses workloads that cannot fit faithfully in two-node
pure-DP mode.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

VLLM_E2E_DIR = AE_ROOT / "start-e2e" / "vllm"
VLLM_RUNNER = VLLM_E2E_DIR / "manual_multinode_poisson_runner.py"
VLLM_HARNESS = VLLM_E2E_DIR / "offline_poisson_harness.py"
DEFAULT_MODEL_PATH = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_LENS_ROOT = SCRIPT_DIR / "inputs" / "vllm"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "profiling_results" / "vllm"
DEFAULT_HEAD_HOST = "h200-rjob0"
DEFAULT_MASTER_ADDRESS = "10.102.252.174"
DEFAULT_FOUR_NODE_REMOTE_HOSTS = (
    "h200-rjob1",
    "h200-rjob2",
    "h200-rjob3",
)
DEFAULT_TWO_NODE_REMOTE_HOSTS = ("h200-rjob1",)
DATASET_FILES = {
    "short": "short.json",
    "issue01": "issue01.json",
    "issue05": "issue05.json",
}
DCP_DATASET_ORDER = ("short", "issue01", "issue05")
DP_DATASET_ORDER = ("issue01", "issue05", "short")
MODE_CHOICES = ("dcp", "dp")
POLICY_CHOICES = ("least_batch", "least_cache")
ALL2ALL_BACKEND_CHOICES = (
    "deepep_low_latency",
    "allgather_reducescatter",
    "naive",
)
CASE_CSV_FIELDS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the E2E-selected states with vLLM for Fig. 15. The "
            "default is the exact four-node paper matrix: three dp4dcp8 "
            "cases and six dp32 cases."
        )
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        help="Number of 8-GPU nodes to use (default: 4).",
    )
    parser.add_argument(
        "--topology",
        choices=("four-node", "two-node"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument("--lens-root", type=Path, default=DEFAULT_LENS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODE_CHOICES,
        help=(
            "Profile modes. Defaults to dcp+dp for four nodes and dcp only "
            "for the reduced two-node check."
        ),
    )
    parser.add_argument(
        "--dp-policies",
        nargs="+",
        choices=POLICY_CHOICES,
        default=list(POLICY_CHOICES),
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get(
            "VLLM_FIG15_MASTER_ADDR", DEFAULT_MASTER_ADDRESS
        ),
        help=(
            f"Head-node address advertised to workers (default: "
            f"{DEFAULT_MASTER_ADDRESS} on {DEFAULT_HEAD_HOST}); override with "
            "VLLM_FIG15_MASTER_ADDR"
        ),
    )
    parser.add_argument("--master-port", type=int, default=29579)
    parser.add_argument(
        "--remote-hosts",
        nargs="+",
        help=(
            "Non-head SSH hosts or aliases: exactly three for four-node and "
            "one for two-node. Defaults to h200-rjob1/2/3 or h200-rjob1; "
            "override with VLLM_FIG15_REMOTE_HOSTS."
        ),
    )
    parser.add_argument("--ssh-config", type=Path, default=Path("/root/.ssh/config"))
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-identity-file", type=Path)
    parser.add_argument(
        "--workdir",
        default=os.environ.get("VLLM_WORKDIR", "/vllm"),
        help="vLLM implementation directory shared by all nodes (default: /vllm)",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3,4,5,6,7",
        help=(
            "Comma-separated GPU indices used on every node. Four-node paper "
            "mode requires eight; two-node compatibility mode may use a subset."
        ),
    )
    parser.add_argument(
        "--all2all-backend",
        choices=ALL2ALL_BACKEND_CHOICES,
        default="deepep_low_latency",
        help=(
            "MoE communication backend. The paper uses deepep_low_latency. "
            "A reduced-GPU two-node smoke test may explicitly use "
            "allgather_reducescatter."
        ),
    )
    parser.add_argument("--request-rate", type=float, default=40.0)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--profile-delay-iterations", type=int, default=32)
    parser.add_argument("--profile-max-iterations", type=int, default=31)
    parser.add_argument(
        "--pause-before-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue with later cases after a failure",
    )
    parser.add_argument(
        "--gpu-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to launch if nvidia-smi reports an existing compute process",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-id",
        "--run-name",
        dest="run_name",
        help="Result directory ID (default: run_YYYYMMDD_HHMMSS).",
    )
    args = parser.parse_args()

    topology_nodes = {"two-node": 2, "four-node": 4}
    if args.num_nodes is None and args.topology is None:
        args.num_nodes = 4
    elif args.num_nodes is None:
        args.num_nodes = topology_nodes[args.topology]
    elif (
        args.topology is not None
        and topology_nodes[args.topology] != args.num_nodes
    ):
        parser.error("--num-nodes and --topology describe different cluster sizes")
    args.topology = "four-node" if args.num_nodes == 4 else "two-node"
    if args.modes is None:
        args.modes = list(MODE_CHOICES) if args.num_nodes == 4 else ["dcp"]

    if args.remote_hosts is None:
        environment_hosts = tuple(
            token
            for token in os.environ.get(
                "VLLM_FIG15_REMOTE_HOSTS", ""
            ).replace(",", " ").split()
            if token
        )
        args.remote_hosts = list(
            environment_hosts
            or (
                DEFAULT_FOUR_NODE_REMOTE_HOSTS
                if args.topology == "four-node"
                else DEFAULT_TWO_NODE_REMOTE_HOSTS
            )
        )

    if args.master_port <= 0:
        parser.error("--master-port must be positive")
    if args.ssh_port is not None and args.ssh_port <= 0:
        parser.error("--ssh-port must be positive")
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    for name in (
        "warmup_requests",
        "output_len",
        "profile_delay_iterations",
        "profile_max_iterations",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    try:
        device_ids = tuple(
            int(value.strip())
            for value in args.cuda_visible_devices.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error("--cuda-visible-devices must contain integer GPU indices")
    if not device_ids or len(set(device_ids)) != len(device_ids):
        parser.error("--cuda-visible-devices must contain unique GPU indices")
    if any(device < 0 for device in device_ids):
        parser.error("--cuda-visible-devices cannot contain negative indices")
    if args.topology == "four-node" and len(device_ids) != 8:
        parser.error("four-node paper mode requires exactly eight GPU indices")
    if (
        args.topology == "four-node"
        and args.all2all_backend != "deepep_low_latency"
    ):
        parser.error("four-node paper mode requires deepep_low_latency")
    args.cuda_device_ids = device_ids
    args.cuda_visible_devices = ",".join(str(device) for device in device_ids)
    if (
        args.topology == "two-node"
        and len(device_ids) != 8
        and args.all2all_backend == "deepep_low_latency"
    ):
        parser.error(
            "deepep_low_latency compatibility runs require eight GPUs per "
            "node; reduced local world sizes make DeepEP treat cross-node "
            "CUDA IPC handles as local. Use eight free GPUs per node for the "
            "real backend, or explicitly select "
            "--all2all-backend allgather_reducescatter for a non-paper smoke test."
        )
    return args


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def flatten_lengths(node: object) -> list[int]:
    values: list[int] = []

    def walk(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, int) and not isinstance(item, bool):
            if item <= 0:
                raise ValueError(f"input lengths must be positive, got {item}")
            values.append(item)
            return
        raise TypeError(
            "length JSON must contain only integers or nested lists; "
            f"got {type(item).__name__}"
        )

    walk(node)
    if not values:
        raise ValueError("length JSON contains no input lengths")
    return values


def load_lengths(path: Path) -> list[int]:
    return flatten_lengths(json.loads(path.read_text(encoding="utf-8")))


def write_lengths_csv(path: Path, lengths: list[int], output_len: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("prompt_len", "output_len"))
        writer.writeheader()
        writer.writerows(
            {"prompt_len": prompt_len, "output_len": output_len}
            for prompt_len in lengths
        )


def validate_paths(args: argparse.Namespace) -> None:
    missing = [
        path for path in (VLLM_RUNNER, VLLM_HARNESS) if not path.is_file()
    ]
    if not Path(args.workdir).is_dir():
        missing.append(Path(args.workdir))
    if not args.model_path.is_dir():
        missing.append(args.model_path)
    if args.ssh_identity_file is not None and not args.ssh_identity_file.is_file():
        missing.append(args.ssh_identity_file)
    selected = set(args.datasets)
    for dataset, filename in DATASET_FILES.items():
        if dataset in selected:
            path = args.lens_root / filename
            if not path.is_file():
                missing.append(path)
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(map(str, missing)))


def topology_values(args: argparse.Namespace) -> dict[str, Any]:
    master_address = args.master_address
    remote_hosts = tuple(args.remote_hosts)
    devices_per_node = len(args.cuda_device_ids)

    if args.topology == "four-node":
        if len(remote_hosts) != 3:
            raise SystemExit(
                "four-node topology requires exactly three --remote-hosts"
            )
        return {
            "cluster_name": "fig15_four_node",
            "master_address": master_address,
            "remote_hosts": remote_hosts,
            "dcp_strategy": "dp4dcp8",
            "dp_strategy": "dp32",
            "expected_traces": 32,
            "devices_per_node": devices_per_node,
            "local_env_script": "/root/.zshrc",
            "remote_env_script": "/root/.zshrc",
        }

    if len(remote_hosts) != 1:
        raise SystemExit(
            "two-node topology requires exactly one --remote-hosts value"
        )
    strategy_suffix = (
        ""
        if args.all2all_backend == "deepep_low_latency"
        else "_" + args.all2all_backend.replace("allgather_reducescatter", "agrs")
    )
    return {
        "cluster_name": "fig15_two_node",
        "master_address": master_address,
        "remote_hosts": remote_hosts,
        "dcp_strategy": f"dp2dcp{devices_per_node}{strategy_suffix}",
        "dp_strategy": f"dp{2 * devices_per_node}{strategy_suffix}",
        "expected_traces": 2 * devices_per_node,
        "devices_per_node": devices_per_node,
        "local_env_script": None,
        "remote_env_script": None,
    }


def validate_two_node_scope(args: argparse.Namespace) -> None:
    if args.topology != "two-node" or "dp" not in args.modes:
        return
    reasons = []
    if "issue05" in args.datasets:
        reasons.append("issue05 exceeds the safely validated two-node DP KV capacity")
    if "issue01" in args.datasets and len(args.cuda_device_ids) < 8:
        reasons.append(
            "issue01 needs eight GPUs per node in two-node pure-DP mode; with "
            "fewer GPUs, expert weights leave insufficient memory for its KV cache"
        )
    if reasons:
        raise SystemExit(
            "Two-node pure-DP selection is not safe:\n  "
            + "\n  ".join(reasons)
            + "\nUse four-node dp32 for faithful long-workload results, or select "
            "short for the current reduced-GPU compatibility check."
        )


def case_resources(
    topology: str, mode: str, dataset: str, devices_per_node: int
) -> dict[str, Any]:
    if topology == "four-node":
        if mode == "dcp":
            return {
                "max_num_seqs": 1200 if dataset == "short" else 1024,
                "gpu_memory_utilization": 0.85,
                "max_model_len": 1_000_000,
            }
        return {
            "max_num_seqs": 256,
            "gpu_memory_utilization": 0.87,
            "max_model_len": 1_000_000,
        }

    if mode == "dcp":
        return {
            "max_num_seqs": (
                1200
                if dataset == "short" and devices_per_node == 8
                else 1024
                if devices_per_node == 8
                else 256
            ),
            "gpu_memory_utilization": 0.85,
            "max_model_len": 1_000_000,
        }
    if dataset == "short":
        return {
            "max_num_seqs": 256,
            "gpu_memory_utilization": 0.87,
            "max_model_len": 20_000,
        }
    if dataset == "issue01":
        return {
            "max_num_seqs": 256,
            "gpu_memory_utilization": 0.96,
            "max_model_len": 950_000,
        }
    raise AssertionError(f"unsupported two-node DP dataset: {dataset}")


def build_inputs_and_cases(
    args: argparse.Namespace,
    run_dir: Path,
    topology: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    selected = set(args.datasets)
    stats: dict[str, dict[str, int]] = {}
    dataset_aliases: dict[str, str] = {}

    for dataset in DATASET_FILES:
        if dataset not in selected:
            continue
        source = (args.lens_root / DATASET_FILES[dataset]).resolve()
        lengths = load_lengths(source)
        csv_path = inputs_dir / f"{dataset}.lengths.csv"
        write_lengths_csv(csv_path, lengths, args.output_len)
        stats[dataset] = {
            "rows": len(lengths),
            "min_prompt_len": min(lengths),
            "max_prompt_len": max(lengths),
            "max_total_len": max(lengths) + args.output_len,
        }
        dataset_aliases[dataset] = f"fig15_{dataset}"

    cases: list[dict[str, str]] = []
    plans: list[dict[str, Any]] = []

    def add_case(mode: str, dataset: str, policy: str) -> None:
        strategy = topology[f"{mode}_strategy"]
        resources = case_resources(
            args.topology, mode, dataset, topology["devices_per_node"]
        )
        name = f"fig15_{args.topology.replace('-', '')}_{strategy}_{dataset}_{policy}"
        row = {
            "enabled": "1",
            "name": name,
            "cluster": topology["cluster_name"],
            "model": "deepseek_v3_1024k",
            "dataset": dataset_aliases[dataset],
            "strategy": strategy,
            "dispatch_policy": policy,
            "request_rate": f"{args.request_rate:g}",
            "rate_phase": "fig15",
            "max_num_seqs": str(resources["max_num_seqs"]),
            "gpu_memory_utilization": f"{resources['gpu_memory_utilization']:g}",
            "max_requests": "csv_rows",
            "warmup_requests": str(args.warmup_requests),
            "max_model_len": str(resources["max_model_len"]),
            "data_parallel_rpc_port": "29550",
            "reason": (
                "Fig. 15 replay of an E2E-selected state and trace collection"
            ),
            "historical_reference": "",
        }
        cases.append(row)
        plans.append(
            {
                "name": name,
                "mode": mode,
                "dataset": dataset,
                "dispatch_policy": policy,
                "strategy": strategy,
                "all2all_backend": args.all2all_backend,
                "expected_requests": stats[dataset]["rows"],
                "expected_traces": topology["expected_traces"],
                **resources,
            }
        )

    if "dcp" in args.modes:
        for dataset in DCP_DATASET_ORDER:
            if dataset in selected:
                add_case("dcp", dataset, "least_batch")
    if "dp" in args.modes:
        for dataset in DP_DATASET_ORDER:
            if dataset in selected:
                for policy in POLICY_CHOICES:
                    if policy in args.dp_policies:
                        add_case("dp", dataset, policy)

    if not cases:
        raise SystemExit("No cases selected")

    with (run_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(cases)
    write_json(run_dir / "workload_stats.json", stats)
    return cases, plans


def load_runner():
    path = VLLM_RUNNER
    module_name = "fig15_manual_multinode_poisson_runner"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ssh_options(args: argparse.Namespace) -> list[str]:
    options = [
        "-F",
        str(args.ssh_config),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UpdateHostKeys=no",
    ]
    if args.ssh_port is not None:
        options.extend(("-p", str(args.ssh_port)))
    if args.ssh_user:
        options.extend(("-l", args.ssh_user))
    if args.ssh_identity_file is not None:
        options.extend(("-i", str(args.ssh_identity_file.resolve())))
    return options


def configure_runner(
    runner,
    args: argparse.Namespace,
    topology: dict[str, Any],
    run_dir: Path,
) -> None:
    runner.MODELS["deepseek_v3_1024k"] = str(args.model_path.resolve())
    for dataset in args.datasets:
        runner.DATASETS[f"fig15_{dataset}"] = str(
            (run_dir / "inputs" / f"{dataset}.lengths.csv").resolve()
        )

    runner.CLUSTERS[topology["cluster_name"]] = runner.ClusterSpec(
        master_addr=topology["master_address"],
        master_port=args.master_port,
        remote_hosts=topology["remote_hosts"],
        workdir=args.workdir,
        ssh_opts=tuple(ssh_options(args)),
        local_env_script=topology["local_env_script"],
        remote_env_script=topology["remote_env_script"],
    )

    original_load_cases_from_csv = runner.load_cases_from_csv

    def load_cases_with_selected_devices(path: Path):
        return [
            runner.replace(
                case,
                cuda_visible_devices=args.cuda_visible_devices,
            )
            for case in original_load_cases_from_csv(path)
        ]

    runner.load_cases_from_csv = load_cases_with_selected_devices

    if args.topology == "two-node":
        devices_per_node = topology["devices_per_node"]
        runner.STRATEGIES[topology["dcp_strategy"]] = runner.StrategySpec(
            data_parallel_size=2,
            data_parallel_size_local=1,
            tensor_parallel_size=devices_per_node,
            decode_context_parallel_size=devices_per_node,
            data_parallel_backend="mp",
            enable_expert_parallel=True,
            attention_backend="FLASHMLA",
            all2all_backend=args.all2all_backend,
            dcp_comm_backend="a2a",
        )
        runner.STRATEGIES[topology["dp_strategy"]] = runner.StrategySpec(
            data_parallel_size=2 * devices_per_node,
            data_parallel_size_local=devices_per_node,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_backend="mp",
            enable_expert_parallel=True,
            attention_backend="FLASHMLA",
            all2all_backend=args.all2all_backend,
        )


def profiler_runner_argv(args: argparse.Namespace, run_dir: Path) -> list[str]:
    frontend_args = [
        "--no-async-scheduling",
        "--profile-after-warmup",
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        "{benchmark_dir}/torch_profiler",
        "--profiler-config.ignore_frontend",
        "true",
        "--profiler-config.delay_iterations",
        str(args.profile_delay_iterations),
        "--profiler-config.max_iterations",
        str(args.profile_max_iterations),
        "--profiler-config.wait_iterations",
        "0",
        "--profiler-config.warmup_iterations",
        "0",
    ]
    if args.pause_before_profile:
        frontend_args.append("--pause-before-profile")
    headless_args = [
        "--no-async-scheduling",
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        "{benchmark_dir}/torch_profiler",
        "--profiler-config.delay_iterations",
        str(args.profile_delay_iterations),
        "--profiler-config.max_iterations",
        str(args.profile_max_iterations),
        "--profiler-config.wait_iterations",
        "0",
        "--profiler-config.warmup_iterations",
        "0",
    ]
    argv = [
        "--artifact-root",
        str((run_dir / "artifacts").resolve()),
        "--case-csv",
        str((run_dir / "cases.csv").resolve()),
        "--run-label",
        run_dir.name,
        "--ignore-historical-skips",
        "--keep-going" if args.keep_going else "--no-keep-going",
    ]
    if args.dry_run:
        argv.append("--dry-run")
    for token in frontend_args:
        argv.append(f"--frontend-extra-arg={token}")
    for token in headless_args:
        argv.append(f"--headless-extra-arg={token}")
    return argv


def nvidia_smi_processes(
    host: str | None, args: argparse.Namespace
) -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"
    query = [
        nvidia_smi,
        f"--id={args.cuda_visible_devices}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    command = query
    label = "local node"
    process_env = os.environ.copy()
    driver_library = "/usr/local/nvidia/lib64"
    old_library_path = process_env.get("LD_LIBRARY_PATH", "")
    process_env["LD_LIBRARY_PATH"] = (
        driver_library
        if not old_library_path
        else f"{driver_library}:{old_library_path}"
    )
    if host is not None:
        label = host
        remote_query = [
            "env",
            "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
            *query,
        ]
        command = [
            "ssh",
            *ssh_options(args),
            host,
            shlex.join(remote_query),
        ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=process_env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"GPU preflight failed on {label}: {detail}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gpu_preflight(args: argparse.Namespace, topology: dict[str, Any]) -> None:
    occupied: dict[str, list[str]] = {}
    local = nvidia_smi_processes(None, args)
    if local:
        occupied["local"] = local
    for host in topology["remote_hosts"]:
        processes = nvidia_smi_processes(host, args)
        if processes:
            occupied[host] = processes
    if occupied:
        lines = ["GPU preflight found existing compute processes:"]
        for host, processes in occupied.items():
            lines.append(f"  {host}:")
            lines.extend(f"    {process}" for process in processes)
        lines.append("Wait for idle GPUs, or pass --no-gpu-preflight only if sharing is intentional.")
        raise RuntimeError("\n".join(lines))


def collect_results(
    run_dir: Path,
    plans: list[dict[str, Any]],
    *,
    dry_run: bool,
    strict: bool,
) -> list[dict[str, Any]]:
    manifests: dict[str, Path] = {}
    for path in (run_dir / "artifacts").rglob("case_manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = payload.get("case_name")
        if isinstance(name, str):
            manifests[name] = path

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for plan in plans:
        name = plan["name"]
        manifest_path = manifests.get(name)
        if manifest_path is None:
            results.append({"name": name, "status": "missing"})
            errors.append(f"missing case manifest: {name}")
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = payload.get("status")
        entry: dict[str, Any] = {
            "name": name,
            "status": status,
            "case_manifest": str(manifest_path),
        }
        expected_status = "dry_run" if dry_run else "ok"
        if status != expected_status:
            errors.append(f"{name}: status={status}, expected {expected_status}")

        paths = payload.get("paths") or {}
        benchmark_value = paths.get("benchmark_dir")
        if benchmark_value:
            benchmark_dir = Path(benchmark_value)
            entry["benchmark_dir"] = str(benchmark_dir)
            traces = list(
                (benchmark_dir / "torch_profiler").glob("*.pt.trace.json*")
            )
            entry["trace_count"] = len(traces)
            entry["expected_trace_count"] = plan["expected_traces"]
            if not dry_run and len(traces) != plan["expected_traces"]:
                errors.append(
                    f"{name}: {len(traces)} traces, expected {plan['expected_traces']}"
                )
            summary_path = benchmark_dir / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                successful = summary.get("successful_requests")
                failed = summary.get("failed_requests")
                entry["successful_requests"] = successful
                entry["failed_requests"] = failed
                entry["expected_requests"] = plan["expected_requests"]
                entry["achieved_request_throughput_rps"] = summary.get(
                    "achieved_request_throughput_rps"
                )
                if not dry_run and (
                    successful != plan["expected_requests"] or failed != 0
                ):
                    errors.append(
                        f"{name}: successful={successful}, failed={failed}, "
                        f"expected successful={plan['expected_requests']}"
                    )
            elif not dry_run:
                errors.append(f"{name}: missing summary.json")
        elif not dry_run:
            errors.append(f"{name}: manifest has no benchmark_dir")
        results.append(entry)

    if strict and errors:
        raise RuntimeError("Fig. 15 result validation failed:\n  " + "\n  ".join(errors))
    return results


def main() -> None:
    args = parse_args()
    validate_paths(args)
    validate_two_node_scope(args)
    topology = topology_values(args)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different --run-id."
        ) from exc

    _, plans = build_inputs_and_cases(args, run_dir, topology)
    free_bytes = shutil.disk_usage(run_dir).free
    manifest: dict[str, Any] = {
        "status": "preparing",
        "system": "vLLM",
        "purpose": "Fig. 15 replay of E2E-selected states and trace collection",
        "topology_mode": args.topology,
        "num_nodes": args.num_nodes,
        "paper_faithful": args.topology == "four-node",
        "cluster": {
            "master_address": topology["master_address"],
            "master_port": args.master_port,
            "remote_hosts": topology["remote_hosts"],
            "cuda_visible_devices": args.cuda_visible_devices,
            "devices_per_node": topology["devices_per_node"],
            "expected_trace_count_per_case": topology["expected_traces"],
        },
        "vllm_workdir": str(Path(args.workdir).resolve()),
        "runner": str(VLLM_RUNNER.resolve()),
        "model_path": str(args.model_path.resolve()),
        "load_format": "dummy (offline_poisson_harness default)",
        "all2all_backend": args.all2all_backend,
        "lens_root": str(args.lens_root.resolve()),
        "request_rate": args.request_rate,
        "warmup_requests": args.warmup_requests,
        "output_len": args.output_len,
        "profile": {
            "profiler": "torch",
            "delay_iterations": args.profile_delay_iterations,
            "max_iterations": args.profile_max_iterations,
            "wait_iterations": 0,
            "warmup_iterations": 0,
            "pause_before_profile": args.pause_before_profile,
            "async_scheduling": False,
        },
        "free_disk_bytes_at_start": free_bytes,
        "dry_run": args.dry_run,
        "cases": plans,
        "results": [],
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"vLLM Fig. 15 profiling output: {run_dir}", flush=True)
    print(f"Selected {len(plans)} case(s):", flush=True)
    for plan in plans:
        print(
            f"  {plan['name']} (requests={plan['expected_requests']}, "
            f"traces={plan['expected_traces']})",
            flush=True,
        )

    try:
        if not args.dry_run and args.gpu_preflight:
            gpu_preflight(args, topology)
        runner = load_runner()
        configure_runner(runner, args, topology, run_dir)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        runner.main(profiler_runner_argv(args, run_dir))
        manifest["results"] = collect_results(
            run_dir, plans, dry_run=args.dry_run, strict=True
        )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc) or type(exc).__name__
        manifest["results"] = collect_results(
            run_dir, plans, dry_run=args.dry_run, strict=False
        )
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "dry_run" if args.dry_run else "completed"
    write_json(manifest_path, manifest)
    print(
        f"All selected vLLM Fig. 15 cases validated: {run_dir}", flush=True
    )


if __name__ == "__main__":
    main()

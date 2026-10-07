#!/usr/bin/env python3
"""Launch the Figure 12 vLLM source sweeps on two or four nodes."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from e2e_config import (
    AE_ROOT,
    FIG12_DIR,
    VLLM_BASELINES,
    VLLM_MASTER_ADDR,
    VLLM_REMOTE_HOSTS,
    VLLM_WORKDIR,
    VllmBaseline,
    Workload,
    WORKLOADS,
    selected_workloads,
    filter_dataset_by_request_tokens,
)


E2E_DIR = AE_ROOT / "start-e2e" / "vllm"
HARNESS = E2E_DIR / "offline_poisson_harness.py"
DEFAULT_REMOTE_HOST = os.environ.get("FIG12_2NODE_REMOTE_HOST", "h200-rjob1")
DEFAULT_SSH_IDENTITY = Path("/root/.ssh/id_rsa_pjlab")
DEFAULT_SSH_USER = "root"
BENCH_DURATION_SEC = 600.0
DEFAULT_TIMEOUT_SEC = 3600.0
MAX_MODEL_LEN = 1_000_000
WARMUP_REQUESTS = 32
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
    reason: str
    historical_reference: str = ""


TWO_NODE_STRATEGIES = {
    "dp_least_batch": "dp16",
    "dp_least_cache": "dp16",
    "cp2": "dp8cp2",
    "cp4": "dp4dcp4",
    "cp8": "dp2dcp8",
}


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


def validate_master_addr(address: str) -> None:
    if address.startswith("127.") or address == "0.0.0.0":
        raise ValueError(f"node-0 address is not reachable by workers: {address}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
    except OSError as error:
        raise ValueError(
            f"node-0 address is not local to this container: {address}"
        ) from error


def four_node_remote_hosts() -> tuple[str, ...]:
    return tuple(VLLM_REMOTE_HOSTS.replace(",", " ").split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        type=int,
        choices=(2, 4),
        default=4,
        help="Cluster size (default: %(default)s).",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=("all", *(workload.slug for workload in WORKLOADS)),
        default=[],
        help="Workload to run; may be repeated. Defaults to all six.",
    )
    parser.add_argument(
        "--strategy",
        action="extend",
        nargs="+",
        choices=("all", *(baseline.slug for baseline in VLLM_BASELINES)),
        default=[],
        help=(
            "One or more vLLM strategies to run. Defaults to all five; "
            "provide all selected names after this option."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
        help="Output identifier shared by the selected workloads.",
    )
    parser.add_argument(
        "--rate",
        action="extend",
        nargs="+",
        type=positive_float,
        default=[],
        metavar="REQ_PER_SEC",
        help=("One or more request rates. " "Defaults to the complete scaled sweep."),
    )
    parser.add_argument(
        "--bench-duration-sec",
        type=positive_float,
        help=(
            "Request-sending duration for each rate "
            f"(default: {BENCH_DURATION_SEC:g})."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=positive_float,
        help=(
            "Wall-clock limit for each rate, including startup and benchmark "
            f"time (quick-test default: {DEFAULT_TIMEOUT_SEC:g})."
        ),
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        help=(
            "Maximum prompt_len + output_len; longer CSV rows are removed. "
            "Disabled unless passed explicitly."
        ),
    )
    parser.add_argument(
        "--master-addr",
        default=VLLM_MASTER_ADDR,
        help="Node-0 address (default: %(default)s).",
    )
    parser.add_argument(
        "--remote-host",
        default=DEFAULT_REMOTE_HOST,
        help=(
            "Two-node worker container SSH host (default: %(default)s). "
            "The SSH endpoint must enter the container directly."
        ),
    )
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-identity", type=Path, default=DEFAULT_SSH_IDENTITY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    quick_test = bool(
        args.rate
        or args.bench_duration_sec is not None
        or args.timeout_sec is not None
        or args.max_request_tokens is not None
    )
    args.rate = list(dict.fromkeys(args.rate))
    if args.nodes == 2:
        if not args.ssh_identity.is_file() and args.ssh_config is None:
            parser.error(f"SSH identity not found: {args.ssh_identity}")
    else:
        remote_hosts = four_node_remote_hosts()
        if len(remote_hosts) != 3:
            parser.error(
                "four-node execution requires exactly three worker SSH hosts; "
                f"got {remote_hosts}"
            )
    try:
        validate_master_addr(args.master_addr)
    except ValueError as error:
        parser.error(str(error))
    if (args.nodes == 2 or quick_test) and args.timeout_sec is None:
        args.timeout_sec = DEFAULT_TIMEOUT_SEC
    return args


def require_path(path: Path, label: str, directory: bool) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {path}")


def prepare_filtered_datasets(
    workloads: tuple[Workload, ...],
    output_dir: Path,
    max_request_tokens: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    for workload in workloads:
        if workload.dataset_name in prepared:
            continue
        destination = (
            output_dir / f"{workload.dataset_name}_max{max_request_tokens}.csv"
        )
        total_rows, kept_rows = filter_dataset_by_request_tokens(
            workload.dataset_path, destination, max_request_tokens
        )
        prepared[workload.dataset_name] = destination.resolve()
        print(
            f"[dataset] {workload.dataset_name}: kept {kept_rows}/{total_rows}, "
            f"removed {total_rows - kept_rows} above "
            f"{max_request_tokens} tokens",
            flush=True,
        )
    return prepared


def ssh_options(args: argparse.Namespace) -> list[str]:
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UpdateHostKeys=no",
    ]
    if args.ssh_config is not None:
        options[:0] = ["-F", str(args.ssh_config.expanduser().resolve())]
    else:
        options.extend(
            [
                "-o",
                "IdentitiesOnly=yes",
                "-l",
                args.ssh_user,
                "-i",
                str(args.ssh_identity.expanduser().resolve()),
            ]
        )
    return options


def four_node_ssh_options(args: argparse.Namespace) -> list[str]:
    config = args.ssh_config or Path("/root/.ssh/config")
    return [
        "-F",
        str(config.expanduser().resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UpdateHostKeys=no",
    ]


def preflight_two_node_worker(
    args: argparse.Namespace,
    workloads: tuple[Workload, ...],
    dataset_paths: dict[str, Path],
) -> None:
    checks: list[tuple[str, Path]] = [
        ("-d", VLLM_WORKDIR),
        ("-f", HARNESS),
    ]
    for workload in workloads:
        checks.extend(
            [
                ("-d", workload.model_path),
                ("-f", dataset_paths[workload.dataset_name]),
            ]
        )
    unique_checks = list(dict.fromkeys(checks))
    test_command = " && ".join(
        f"test {kind} {shlex.quote(str(path.resolve()))}"
        for kind, path in unique_checks
    )
    remote_command = ["zsh", "-lc", shlex.quote(test_command)]
    command = [
        "ssh",
        *ssh_options(args),
        args.remote_host,
        *remote_command,
    ]
    print(f"[preflight] checking worker container {args.remote_host}", flush=True)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"worker preflight timed out for {args.remote_host}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"worker preflight failed for {args.remote_host}: "
            f"{detail or f'exit code {completed.returncode}'}"
        )
    print("[preflight] worker SSH, container, and paths are ready", flush=True)


def preflight_four_node_workers(
    args: argparse.Namespace,
    workloads: tuple[Workload, ...],
    dataset_paths: dict[str, Path],
) -> None:
    checks: list[tuple[str, Path]] = [
        ("-d", VLLM_WORKDIR),
        ("-f", HARNESS),
    ]
    for workload in workloads:
        checks.extend(
            [
                ("-d", workload.model_path),
                (
                    "-f",
                    dataset_paths.get(
                        workload.dataset_name,
                        workload.dataset_path,
                    ),
                ),
            ]
        )
    test_command = " && ".join(
        f"test {kind} {shlex.quote(str(path.resolve()))}"
        for kind, path in dict.fromkeys(checks)
    )
    identity_marker = "FIG12_NODE_ID="
    remote_command = (
        f"{test_command} && printf {shlex.quote(identity_marker)} && "
        "cat /proc/sys/kernel/random/boot_id"
    )
    local_identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    identities = {local_identity: "node 0"}
    for host in four_node_remote_hosts():
        print(f"[preflight] checking worker container {host}", flush=True)
        command = [
            "ssh",
            *four_node_ssh_options(args),
            host,
            "zsh",
            "-lc",
            shlex.quote(remote_command),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"worker preflight timed out for {host}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"worker preflight failed for {host}: "
                f"{detail or f'exit code {completed.returncode}'}"
            )
        identity_line = next(
            (
                line
                for line in completed.stdout.splitlines()
                if line.startswith(identity_marker)
            ),
            "",
        )
        identity = identity_line.removeprefix(identity_marker).strip()
        if not identity:
            raise RuntimeError(f"worker preflight returned no node ID for {host}")
        if identity in identities:
            raise RuntimeError(
                f"worker {host} duplicates {identities[identity]}; "
                "each node must refer to a different machine"
            )
        identities[identity] = host
    print("[preflight] four distinct nodes and all paths are ready", flush=True)


def load_runner() -> object:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(E2E_DIR))
    import manual_multinode_poisson_runner as runner

    return runner


def install_completion_status_reporting(runner: object) -> None:
    """Clarify the shutdown phase after all benchmark requests finish."""
    original = runner.ManualMultinodeRunner.log_waiting

    def log_waiting(
        self: object,
        case_name: str,
        message: str,
        **kwargs: object,
    ) -> float:
        runtime = getattr(self, "_active_runtime", None)
        if (
            runtime is not None
            and message.startswith("still waiting for frontend benchmark")
            and runner.case_dir_has_complete_successful_benchmark(
                runtime.artifacts.case_dir
            )
        ):
            message = (
                "all benchmark requests completed; "
                "waiting for vLLM processes to shut down cleanly"
            )
        return original(self, case_name, message, **kwargs)

    runner.ManualMultinodeRunner.log_waiting = log_waiting


def configure_strategy(
    runner: object,
    name: str,
    *,
    dp_size: int,
    dp_local: int,
    cp_size: int,
) -> None:
    options: dict[str, object] = {
        "data_parallel_size": dp_size,
        "data_parallel_size_local": dp_local,
        "tensor_parallel_size": cp_size,
        "decode_context_parallel_size": cp_size,
        "data_parallel_backend": "mp",
        "enable_expert_parallel": True,
        "attention_backend": "FLASHMLA",
        "all2all_backend": "deepep_low_latency",
    }
    if cp_size > 1:
        options["dcp_comm_backend"] = "a2a"
    runner.STRATEGIES[name] = runner.StrategySpec(**options)


def configure_runner(
    runner: object,
    args: argparse.Namespace,
    dataset_paths: dict[str, Path],
) -> str:
    for workload in WORKLOADS:
        runner.MODELS[workload.model_name] = str(workload.model_path.resolve())
        dataset_path = dataset_paths.get(
            workload.dataset_name, workload.dataset_path.resolve()
        )
        runner.DATASETS[workload.dataset_name] = str(dataset_path)
    runner.HARNESS_ENTRYPOINT = str(HARNESS.resolve())

    if args.timeout_sec is not None:
        load_cases_from_csv = runner.load_cases_from_csv

        def load_cases_with_timeout(path: Path) -> list[object]:
            return [
                replace(case, max_bench_duration_sec=args.timeout_sec)
                for case in load_cases_from_csv(path)
            ]

        runner.load_cases_from_csv = load_cases_with_timeout

    if args.nodes == 4:
        cluster_name = "fig12_4node_h200"
        remote_hosts = four_node_remote_hosts()
        if len(remote_hosts) != 3:
            raise ValueError(
                "four-node execution requires exactly three worker SSH hosts; "
                f"got {remote_hosts}"
            )
        base_cluster = runner.CLUSTERS["4node_h200"]
        runner.CLUSTERS[cluster_name] = replace(
            base_cluster,
            master_addr=args.master_addr,
            remote_hosts=remote_hosts,
            workdir=str(VLLM_WORKDIR.resolve()),
            ssh_opts=tuple(four_node_ssh_options(args)),
        )
        return cluster_name

    cluster_name = "fig12_2node_h200"
    base_cluster = runner.CLUSTERS["2node_h200"]
    runner.CLUSTERS[cluster_name] = replace(
        base_cluster,
        master_addr=args.master_addr,
        remote_hosts=(args.remote_host,),
        workdir=str(VLLM_WORKDIR.resolve()),
        ssh_opts=tuple(ssh_options(args)),
        local_shell="zsh",
        local_shell_flags=("-lc",),
        remote_shell="zsh",
        remote_shell_flags=("-lc",),
        local_env_script=None,
        remote_env_script=None,
    )
    configure_strategy(
        runner,
        TWO_NODE_STRATEGIES["dp_least_batch"],
        dp_size=16,
        dp_local=8,
        cp_size=1,
    )
    configure_strategy(
        runner,
        TWO_NODE_STRATEGIES["cp2"],
        dp_size=8,
        dp_local=4,
        cp_size=2,
    )
    configure_strategy(
        runner,
        TWO_NODE_STRATEGIES["cp4"],
        dp_size=4,
        dp_local=2,
        cp_size=4,
    )
    configure_strategy(
        runner,
        TWO_NODE_STRATEGIES["cp8"],
        dp_size=2,
        dp_local=1,
        cp_size=8,
    )
    return cluster_name


def strategy_name(baseline: VllmBaseline, nodes: int) -> str:
    return baseline.strategy if nodes == 4 else TWO_NODE_STRATEGIES[baseline.slug]


def selected_baselines(values: list[str]) -> tuple[VllmBaseline, ...]:
    if not values or "all" in values:
        return VLLM_BASELINES
    selected = set(values)
    return tuple(baseline for baseline in VLLM_BASELINES if baseline.slug in selected)


def build_cases(
    workloads: tuple[Workload, ...],
    baselines: tuple[VllmBaseline, ...],
    cluster_name: str,
    nodes: int,
    requested_rates: tuple[float, ...],
    bench_duration_sec: float,
    max_request_tokens: int | None,
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    rate_scale = nodes / 4.0
    for workload in workloads:
        for baseline in baselines:
            gpu_memory = (
                workload.least_cache_memory
                if baseline.gpu_memory_utilization is None
                else baseline.gpu_memory_utilization
            )
            rates = requested_rates or tuple(
                paper_rate * rate_scale for paper_rate in workload.vllm_rates
            )
            for rate in rates:
                rows.append(
                    CaseRow(
                        enabled=1,
                        name=(
                            f"fig12_{nodes}node_{workload.slug}_{baseline.slug}_"
                            f"rate{rate:g}"
                        ),
                        cluster=cluster_name,
                        model=workload.model_name,
                        dataset=workload.dataset_name,
                        strategy=strategy_name(baseline, nodes),
                        dispatch_policy=baseline.dispatch_policy,
                        request_rate=rate,
                        rate_phase=(
                            f"fig12_{nodes}node_selected"
                            if requested_rates
                            else f"fig12_{nodes}node"
                        ),
                        max_num_seqs=baseline.max_num_seqs,
                        gpu_memory_utilization=gpu_memory,
                        max_requests=max(1, round(rate * bench_duration_sec)),
                        warmup_requests=WARMUP_REQUESTS,
                        max_model_len=(max_request_tokens or MAX_MODEL_LEN),
                        data_parallel_rpc_port=29550,
                        reason=(
                            f"Figure 12 {nodes}-node {workload.label} "
                            f"{baseline.slug} source sweep"
                        ),
                    )
                )
    return rows


def write_case_csv(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    workloads = selected_workloads(args.workload)
    baselines = selected_baselines(args.strategy)
    artifact_root = (
        FIG12_DIR / "results" / "e2e" / "vllm" / f"{args.nodes}node"
    ).resolve()
    run_label = f"fig12-{args.nodes}node-{args.run_id}"
    require_path(E2E_DIR, "vLLM E2E directory", directory=True)
    require_path(HARNESS, "vLLM harness", directory=False)
    require_path(VLLM_WORKDIR, "vLLM checkout", directory=True)
    for workload in workloads:
        require_path(workload.model_path, workload.label + " model", directory=True)
        require_path(
            workload.dataset_path, workload.label + " dataset", directory=False
        )
    dataset_paths: dict[str, Path] = {}
    if args.max_request_tokens is not None:
        dataset_paths = prepare_filtered_datasets(
            workloads,
            artifact_root / "_filtered_csv" / args.run_id,
            args.max_request_tokens,
        )
    if args.nodes == 2 and not args.dry_run:
        preflight_two_node_worker(args, workloads, dataset_paths)
    elif args.nodes == 4 and not args.dry_run:
        preflight_four_node_workers(args, workloads, dataset_paths)

    runner = load_runner()
    install_completion_status_reporting(runner)
    cluster_name = configure_runner(runner, args, dataset_paths)
    requested_rates = tuple(args.rate)
    bench_duration_sec = args.bench_duration_sec or BENCH_DURATION_SEC
    rows = build_cases(
        workloads,
        baselines,
        cluster_name,
        args.nodes,
        requested_rates,
        bench_duration_sec,
        args.max_request_tokens,
    )
    case_csv = artifact_root / "_case_csv" / f"{run_label}_cases.csv"
    write_case_csv(case_csv, rows)

    print(f"Nodes: {args.nodes}", flush=True)
    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Master address: {args.master_addr}", flush=True)
    if args.nodes == 2:
        print(f"Worker SSH host: {args.remote_host}", flush=True)
    else:
        print(f"Worker SSH hosts: {VLLM_REMOTE_HOSTS}", flush=True)
    if requested_rates:
        print(
            "Selected rates: " + " ".join(f"{rate:g}" for rate in requested_rates),
            flush=True,
        )
    if args.strategy:
        print(
            "Selected strategies: " + " ".join(baseline.slug for baseline in baselines),
            flush=True,
        )
    print(f"Benchmark duration per rate: {bench_duration_sec:g}s", flush=True)
    if args.max_request_tokens is not None:
        print(
            f"Max request tokens: {args.max_request_tokens} "
            "(prompt_len + output_len)",
            flush=True,
        )
    if args.timeout_sec is not None:
        print(f"Wall-clock timeout per rate: {args.timeout_sec:g}s", flush=True)
    print(f"Cases: {len(rows)}", flush=True)

    runner_args = [
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(artifact_root),
        "--run-label",
        run_label,
        "--ignore-historical-skips",
    ]
    if args.dry_run:
        runner_args.append("--dry-run")
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


if __name__ == "__main__":
    raise SystemExit(main())

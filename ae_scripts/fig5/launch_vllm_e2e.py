#!/usr/bin/env python3
"""Launch the complete multi-node E2E source experiments for Figure 5.

The shared E2E runner under ``start-e2e/vllm`` is imported read-only.  This
Figure 5-owned entry point defines either a two- or four-node cluster and the
corresponding Figure 5 cases without modifying the shared launcher or runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path


FIG5_DIR = Path(__file__).resolve().parent
AE_ROOT = FIG5_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

E2E_DIR = AE_ROOT / "start-e2e" / "vllm"
DEFAULT_HARNESS = E2E_DIR / "offline_poisson_harness.py"
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("VLLM_E2E_ARTIFACT_ROOT", FIG5_DIR / "results" / "e2e")
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("VLLM_E2E_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)
DEFAULT_DATASET_PATH = Path(
    os.environ.get("VLLM_E2E_DATASET_PATH")
    or (
        Path(require_path("AE_DATASET_MIXLONG_0326"))
        / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
    )
)
DEFAULT_VLLM_WORKDIR = Path(
    os.environ.get(
        "VLLM_WORKDIR",
        "/vllm",
    )
)
DEFAULT_ENV_SCRIPT = (
    Path(os.environ["VLLM_MULTINODE_ENV_SCRIPT"])
    if os.environ.get("VLLM_MULTINODE_ENV_SCRIPT")
    else None
)
DEFAULT_SSH_IDENTITY = Path("/root/.ssh/id_rsa_pjlab")
DEFAULT_SSH_USER = "ailab"
DEFAULT_REMOTE_CONTAINER = "ae_merged"
DEFAULT_REMOTE_HOSTS = tuple(
    os.environ.get(
        "VLLM_FIG5_REMOTE_HOSTS",
        os.environ.get(
            "VLLM_4NODE_REMOTE_HOSTS",
            "h200-rjob1,h200-rjob3,h200-rjob4",
        ),
    )
    .replace(",", " ")
    .split()
)
CLUSTER_NAME = "fig5_h200"
MODEL_NAME = "deepseek_v3_1024k"
DATASET_NAME = "issue01_random"
BENCH_DURATION_SEC = 600.0
DEFAULT_MAX_MODEL_LEN = 1_000_000
GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4
GPU_IDLE_THRESHOLD_MIB = 1024
GPU_IDLE_TIMEOUT_SEC = 180
GPU_IDLE_POLL_SEC = 5

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


@dataclass(frozen=True)
class Fig5Case:
    part: str
    dispatch_policy: str
    request_rate: float
    gpu_memory_utilization: float
    purpose: str


FIG5_CASES = (
    Fig5Case(
        part="attention",
        dispatch_policy="waiting_x4_plus_running",
        request_rate=30.0,
        gpu_memory_utilization=0.90,
        purpose="per-rank KV usage for the Attention panel",
    ),
    Fig5Case(
        part="deepep",
        dispatch_policy="least_cache",
        request_rate=30.0,
        gpu_memory_utilization=0.87,
        purpose="per-rank running requests for the DeepEP panel",
    ),
    Fig5Case(
        part="hol",
        dispatch_policy="waiting_x4_plus_running",
        request_rate=50.0,
        gpu_memory_utilization=0.90,
        purpose="queue and KV time series for the HoL panel",
    ),
)


def existing_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def resolve_ssh_hostname(remote_host: str, ssh_config: Path | None) -> str:
    command = ["ssh"]
    if ssh_config is not None:
        command.extend(["-F", str(ssh_config.expanduser())])
    command.extend(["-G", remote_host])
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return remote_host
    if result.returncode != 0:
        return remote_host
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator and key.lower() == "hostname" and value.strip():
            return value.strip()
    return remote_host


def detect_master_addr(remote_host: str, ssh_config: Path | None) -> str:
    ssh_hostname = resolve_ssh_hostname(remote_host, ssh_config)
    try:
        remote_addr = socket.gethostbyname(ssh_hostname)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((remote_addr, 22))
            master_addr = str(probe.getsockname()[0])
    except OSError as error:
        raise ValueError(
            "cannot infer the local address used to reach "
            f"{remote_host} (SSH hostname {ssh_hostname}): {error}"
        ) from error
    if master_addr.startswith("127.") or master_addr == "0.0.0.0":
        raise ValueError(
            f"inferred unusable master address {master_addr} for {remote_host}"
        )
    return master_addr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="Cluster size (default: %(default)s).",
    )
    parser.add_argument(
        "--fig5-case",
        action="append",
        choices=("all", "attention", "deepep", "hol"),
        default=[],
        help="Source case to run; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--master-addr",
        help=(
            "Node-0 container address reachable by every worker. By default "
            "it is inferred from the route to the first worker."
        ),
    )
    parser.add_argument(
        "--remote-hosts",
        nargs="+",
        help=(
            "Worker SSH aliases in node-rank order. Exactly --num-nodes minus "
            "one hosts are required."
        ),
    )
    parser.add_argument("--run-id", default="ae_run1")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-path", type=existing_path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--dataset-path", type=existing_path, default=DEFAULT_DATASET_PATH
    )
    parser.add_argument(
        "--vllm-workdir", type=existing_path, default=DEFAULT_VLLM_WORKDIR
    )
    parser.add_argument("--harness", type=existing_path, default=DEFAULT_HARNESS)
    parser.add_argument("--ssh-config", type=existing_path)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument(
        "--ssh-identity", type=existing_path, default=DEFAULT_SSH_IDENTITY
    )
    parser.add_argument("--remote-container", default=DEFAULT_REMOTE_CONTAINER)
    parser.add_argument(
        "--direct-remote",
        action="store_true",
        help=(
            "For a two-node run, execute zsh directly after SSH instead of "
            "entering --remote-container. Four-node runs always use direct "
            "worker-container SSH endpoints."
        ),
    )
    parser.add_argument("--shell", choices=("bash", "zsh"))
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--env-script", type=existing_path, default=DEFAULT_ENV_SCRIPT)
    env_group.add_argument("--no-env-script", action="store_true")
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN
    )
    parser.add_argument(
        "--bench-duration-sec", type=float, default=BENCH_DURATION_SEC
    )
    parser.add_argument("--case-csv-output", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the local and worker environments without running a case.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-historical-skips", action="store_true")
    parser.add_argument("--no-keep-going", action="store_true")
    args = parser.parse_args()

    if args.remote_hosts is None:
        args.remote_hosts = DEFAULT_REMOTE_HOSTS[: args.num_nodes - 1]
    else:
        args.remote_hosts = tuple(args.remote_hosts)
    if len(args.remote_hosts) != args.num_nodes - 1:
        parser.error(
            f"--num-nodes {args.num_nodes} requires "
            f"{args.num_nodes - 1} worker SSH host(s); got {args.remote_hosts}"
        )

    for option, path in (
        ("--model-path", args.model_path),
        ("--dataset-path", args.dataset_path),
        ("--vllm-workdir", args.vllm_workdir),
        ("--harness", args.harness),
    ):
        if not path.exists():
            parser.error(f"{option} does not exist: {path}")
    if (
        args.num_nodes == 2
        and args.ssh_config is None
        and not args.ssh_identity.exists()
    ):
        parser.error(f"--ssh-identity does not exist: {args.ssh_identity}")
    if args.master_addr is None:
        try:
            args.master_addr = detect_master_addr(
                args.remote_hosts[0], args.ssh_config
            )
        except ValueError as error:
            parser.error(f"{error}; pass --master-addr explicitly")
    if not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.bench_duration_sec <= 0:
        parser.error("--bench-duration-sec must be positive")
    if args.preflight_only and args.dry_run:
        parser.error("use either --preflight-only or --dry-run, not both")
    if not args.fig5_case:
        args.fig5_case = ["all"]
    return args


def build_cases(
    selected_parts: list[str],
    num_nodes: int,
    max_model_len: int,
    bench_duration_sec: float,
) -> list[CaseRow]:
    selected = set(selected_parts)
    if "all" in selected:
        selected = {case.part for case in FIG5_CASES}
    dp_size = num_nodes * GPUS_PER_NODE
    strategy_name = f"dp{dp_size}"
    rate_scale = num_nodes / PAPER_NUM_NODES
    rows: list[CaseRow] = []
    for case in FIG5_CASES:
        if case.part not in selected:
            continue
        request_rate = case.request_rate * rate_scale
        rows.append(
            CaseRow(
                enabled=1,
                name=(
                    f"fig5_{num_nodes}node_{case.part}_dp{dp_size}_"
                    f"rate{request_rate:g}"
                ),
                cluster=CLUSTER_NAME,
                model=MODEL_NAME,
                dataset=DATASET_NAME,
                strategy=strategy_name,
                dispatch_policy=case.dispatch_policy,
                request_rate=request_rate,
                rate_phase=case.part,
                max_num_seqs=256,
                gpu_memory_utilization=case.gpu_memory_utilization,
                max_requests=round(request_rate * bench_duration_sec),
                warmup_requests=dp_size,
                max_model_len=max_model_len,
                data_parallel_rpc_port=29550,
                reason=case.purpose,
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


def load_runner() -> object:
    if not E2E_DIR.is_dir():
        raise FileNotFoundError(f"shared E2E runner directory not found: {E2E_DIR}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(E2E_DIR))
    import manual_multinode_poisson_runner as runner

    return runner


def ssh_options(args: argparse.Namespace) -> tuple[str, ...]:
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
    elif args.num_nodes == 2:
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
    return tuple(options)


def preflight_command(args: argparse.Namespace) -> str:
    paths = (
        ("-d", args.vllm_workdir),
        ("-d", args.model_path),
        ("-f", args.dataset_path),
        ("-f", args.harness),
    )
    commands: list[str] = []
    if not args.no_env_script:
        env_script = args.env_script or Path("/root/.zshrc")
        commands.append(
            f"source {shlex.quote(str(env_script.expanduser().resolve()))}"
        )
    commands.extend(
        f"test {kind} {shlex.quote(str(path.expanduser().resolve()))}"
        for kind, path in paths
    )
    commands.extend(
        [
            f"cd {shlex.quote(str(args.vllm_workdir.expanduser().resolve()))}",
            (
                "python3 -c "
                + shlex.quote(
                    "import vllm; print('FIG5_VLLM=' + str(vllm.__file__))"
                )
            ),
            "printf FIG5_NODE_ID= && cat /proc/sys/kernel/random/boot_id",
        ]
    )
    return " && ".join(commands)


def remote_shell_command(args: argparse.Namespace, command: str) -> list[str]:
    if args.num_nodes == 4 or args.direct_remote:
        return ["zsh", "-lc", shlex.quote(command)]
    return [
        "docker",
        "exec",
        args.remote_container,
        "zsh",
        "-lc",
        shlex.quote(command),
    ]


def run_preflight(args: argparse.Namespace) -> None:
    command = preflight_command(args)
    print("[preflight] checking node 0", flush=True)
    local = subprocess.run(
        ["zsh", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if local.returncode != 0:
        detail = (local.stderr or local.stdout).strip()
        raise RuntimeError(
            f"node 0 preflight failed: {detail or f'exit code {local.returncode}'}"
        )
    identity_marker = "FIG5_NODE_ID="
    local_identity = (
        Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="utf-8")
        .strip()
    )
    identities = {local_identity: "node 0"}
    for rank, host in enumerate(args.remote_hosts, start=1):
        print(f"[preflight] checking node {rank} via {host}", flush=True)
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    *ssh_options(args),
                    host,
                    *remote_shell_command(args, command),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"worker preflight timed out: {host}") from error
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
            raise RuntimeError(f"worker preflight returned no node ID: {host}")
        if identity in identities:
            raise RuntimeError(
                f"worker {host} duplicates {identities[identity]}; "
                "each worker must refer to a different node"
            )
        identities[identity] = host
    print(
        f"[preflight] {len(identities)} distinct nodes can import vLLM",
        flush=True,
    )


def gpu_memory_command(args: argparse.Namespace) -> str:
    commands: list[str] = []
    if not args.no_env_script:
        env_script = args.env_script or Path("/root/.zshrc")
        commands.append(
            f"source {shlex.quote(str(env_script.expanduser().resolve()))}"
        )
    commands.append(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    )
    return " && ".join(commands)


def parse_gpu_memory(output: str, label: str) -> tuple[int, ...]:
    values: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if value.isdigit():
            values.append(int(value))
    if len(values) != GPUS_PER_NODE:
        raise RuntimeError(
            f"{label} returned {len(values)} GPU memory values; "
            f"expected {GPUS_PER_NODE}. Output: {output.strip()!r}"
        )
    return tuple(values)


def query_gpu_memory(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    command = gpu_memory_command(args)
    local = subprocess.run(
        ["zsh", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if local.returncode != 0:
        detail = (local.stderr or local.stdout).strip()
        raise RuntimeError(f"node 0 GPU check failed: {detail}")
    usage = {"node0": parse_gpu_memory(local.stdout, "node 0")}
    for host in args.remote_hosts:
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    *ssh_options(args),
                    host,
                    *remote_shell_command(args, command),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"GPU check timed out: {host}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"GPU check failed for {host}: {detail}")
        usage[host] = parse_gpu_memory(completed.stdout, host)
    return usage


def wait_for_idle_gpus(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + GPU_IDLE_TIMEOUT_SEC
    while True:
        usage = query_gpu_memory(args)
        busy = {
            node: values
            for node, values in usage.items()
            if max(values) > GPU_IDLE_THRESHOLD_MIB
        }
        if not busy:
            print(
                "[gpu-check] all GPUs are below "
                f"{GPU_IDLE_THRESHOLD_MIB} MiB used",
                flush=True,
            )
            return
        details = "; ".join(
            f"{node}: max {max(values)} MiB" for node, values in busy.items()
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "GPU memory did not become idle within "
                f"{GPU_IDLE_TIMEOUT_SEC}s ({details})"
            )
        print(f"[gpu-check] waiting for GPU memory to clear ({details})", flush=True)
        time.sleep(GPU_IDLE_POLL_SEC)


def install_gpu_idle_guard(runner: object, args: argparse.Namespace) -> None:
    runner_class = runner.ManualMultinodeRunner
    original_run_case = runner_class.run_case

    def run_case_with_gpu_guard(
        self: object,
        case: object,
        run_dir: Path,
    ) -> object:
        wait_for_idle_gpus(args)
        return original_run_case(self, case, run_dir)

    runner_class.run_case = run_case_with_gpu_guard


def configure_runner(runner: object, args: argparse.Namespace) -> None:
    runner.MODELS[MODEL_NAME] = str(args.model_path.expanduser().resolve())
    runner.DATASETS[DATASET_NAME] = str(args.dataset_path.expanduser().resolve())
    runner.HARNESS_ENTRYPOINT = str(args.harness.expanduser().resolve())

    base_cluster = runner.CLUSTERS[f"{args.num_nodes}node_h200"]
    local_shell = args.shell or ("zsh" if args.num_nodes == 4 else "bash")
    cluster_overrides: dict[str, object] = {
        "master_addr": args.master_addr,
        "remote_hosts": args.remote_hosts,
        "workdir": str(args.vllm_workdir.expanduser().resolve()),
        "local_shell": local_shell,
        "local_shell_flags": ("-lc",),
    }
    if args.num_nodes == 4 or args.direct_remote:
        cluster_overrides["remote_shell"] = "zsh"
        cluster_overrides["remote_shell_flags"] = ("-lc",)
    else:
        cluster_overrides["remote_shell"] = "docker"
        cluster_overrides["remote_shell_flags"] = (
            "exec",
            args.remote_container,
            "zsh",
            "-lc",
        )
    if args.no_env_script:
        cluster_overrides["local_env_script"] = None
        cluster_overrides["remote_env_script"] = None
    elif args.env_script is not None:
        env_script = str(args.env_script.expanduser().resolve())
        cluster_overrides["local_env_script"] = env_script
        cluster_overrides["remote_env_script"] = env_script
    cluster_overrides["ssh_opts"] = ssh_options(args)
    runner.CLUSTERS[CLUSTER_NAME] = replace(base_cluster, **cluster_overrides)
    dp_size = args.num_nodes * GPUS_PER_NODE
    strategy_name = f"dp{dp_size}"
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


def run_manifests(
    artifact_root: Path,
    run_label: str,
    runner: object,
) -> list[Path]:
    run_root = artifact_root / "_runs"
    if not run_root.is_dir():
        return []
    run_suffix = f"__{runner.sanitize_tag(run_label)}"
    manifests = [
        candidate / "run_manifest.json"
        for candidate in run_root.iterdir()
        if candidate.is_dir()
        and candidate.name.endswith(run_suffix)
        and (candidate / "run_manifest.json").is_file()
    ]
    manifests.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return manifests


def selected_case_failures(
    artifact_root: Path,
    run_label: str,
    expected_case_names: set[str],
    runner: object,
) -> list[str]:
    manifests = run_manifests(artifact_root, run_label, runner)
    if not manifests:
        return ["run manifest was not generated"]
    with manifests[0].open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    statuses: dict[str, str] = {}
    for result in manifest.get("results", []):
        name = str(result.get("case_name", ""))
        if name in expected_case_names:
            statuses[name] = str(result.get("status", "missing"))
    return [
        f"{name}: {statuses.get(name, 'missing')}"
        for name in sorted(expected_case_names)
        if statuses.get(name) != "ok"
    ]


def write_run_environment(
    artifact_root: Path,
    run_label: str,
    run_id: str,
    num_nodes: int,
    master_addr: str,
    runner: object,
) -> Path | None:
    manifests = run_manifests(artifact_root, run_label, runner)
    if not manifests:
        return None
    manifest_path = manifests[0]

    values = {
        "RUN_ID": run_id,
        "MASTER_ADDR": master_addr,
        "FIG5_RUN_MANIFEST": str(manifest_path.resolve()),
    }
    for candidate_manifest in manifests:
        with candidate_manifest.open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        for result in manifest.get("results", []):
            if result.get("status") != "ok":
                continue
            case_name = str(result.get("case_name", ""))
            case_dir = result.get("case_dir")
            if not case_dir:
                continue
            for part in ("attention", "deepep", "hol"):
                variable = f"{part.upper()}_LOG"
                if (
                    variable not in values
                    and f"_{num_nodes}node_{part}_" in case_name
                ):
                    log_path = (Path(case_dir) / "frontend.log").resolve()
                    if log_path.is_file():
                        values[variable] = str(log_path)
                    break

    required = {"ATTENTION_LOG", "DEEPEP_LOG", "HOL_LOG"}
    if not required.issubset(values):
        return None
    environment_path = artifact_root / f"fig5_{num_nodes}node_{run_id}_env.sh"
    lines = [
        "# Generated by fig5/launch_vllm_e2e.py.",
        *(f"export {name}={shlex.quote(value)}" for name, value in values.items()),
        "",
    ]
    temporary_path = environment_path.with_name(
        f".{environment_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(environment_path)
    return environment_path


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        try:
            run_preflight(args)
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        if args.preflight_only:
            try:
                wait_for_idle_gpus(args)
            except RuntimeError as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
            return 0
    rows = build_cases(
        args.fig5_case,
        args.num_nodes,
        args.max_model_len,
        args.bench_duration_sec,
    )
    artifact_root = args.artifact_root.expanduser().resolve()
    run_label = f"fig5-{args.num_nodes}node-{args.run_id}"
    case_csv = args.case_csv_output
    if case_csv is None:
        case_csv = artifact_root / "_case_csv" / f"{run_label}_cases.csv"
    case_csv = case_csv.expanduser().resolve()
    write_case_csv(case_csv, rows)

    runner = load_runner()
    configure_runner(runner, args)
    if not args.dry_run:
        install_gpu_idle_guard(runner, args)
    runner_arguments = [
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(artifact_root),
        "--run-label",
        run_label,
    ]
    if args.dry_run:
        runner_arguments.append("--dry-run")
    if args.ignore_historical_skips:
        runner_arguments.append("--ignore-historical-skips")
    if args.no_keep_going:
        runner_arguments.append("--no-keep-going")

    print(f"Nodes: {args.num_nodes}", flush=True)
    print(f"Master address: {args.master_addr}", flush=True)
    print(f"Worker SSH hosts: {' '.join(args.remote_hosts)}", flush=True)
    print(f"Benchmark duration: {args.bench_duration_sec:g}s", flush=True)
    print(f"Case CSV: {case_csv}", flush=True)
    for row in rows:
        print(
            f"  {row.name}: policy={row.dispatch_policy}, "
            f"rate={row.request_rate:g}, mem={row.gpu_memory_utilization:g}, "
            f"max_num_seqs={row.max_num_seqs}, "
            f"max_model_len={row.max_model_len}",
            flush=True,
        )
    exit_code = 0
    try:
        runner.main(runner_arguments)
    except SystemExit as error:
        if error.code is None:
            exit_code = 0
        elif isinstance(error.code, int):
            exit_code = error.code
        else:
            print(error.code, file=sys.stderr)
            exit_code = 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        exit_code = 1
    if exit_code != 0:
        return exit_code

    if args.dry_run:
        return 0
    failures = selected_case_failures(
        artifact_root,
        run_label,
        {row.name for row in rows},
        runner,
    )
    if failures:
        print(
            "Selected Figure 5 case(s) did not complete successfully: "
            + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    environment_path = write_run_environment(
        artifact_root,
        run_label,
        args.run_id,
        args.num_nodes,
        args.master_addr,
        runner,
    )
    if environment_path is None:
        selected_all = "all" in args.fig5_case or set(args.fig5_case) == {
            "attention",
            "deepep",
            "hol",
        }
        print(
            "The complete Figure 5 run environment was not written because "
            "successful Attention, DeepEP, and HoL logs are not all available "
            f"for RUN_ID={args.run_id}.",
            file=sys.stderr,
        )
        return 1 if selected_all else 0
    else:
        try:
            display_path = environment_path.relative_to(AE_ROOT)
        except ValueError:
            display_path = environment_path
        print(f"Run environment: {display_path}", flush=True)
        print(f"Next: source {shlex.quote(str(display_path))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

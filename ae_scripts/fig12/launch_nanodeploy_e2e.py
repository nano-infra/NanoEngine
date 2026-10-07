#!/usr/bin/env python3
"""Launch the Figure 12 NanoDeploy source workloads on two or four nodes."""

from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from e2e_config import (
    AE_ROOT,
    FIG12_DIR,
    NANODEPLOY_WORKDIR,
    NANO_MASTER_ADDR,
    NANO_RAY_ADDR,
    Workload,
    WORKLOADS,
    format_rates,
    filter_dataset_by_request_tokens,
    selected_workloads,
)


NANO_BENCHMARK = AE_ROOT / "start-e2e" / "nano" / "bench_serving_overhead.py"
SEND_DURATION_SEC = 600
DEFAULT_TIMEOUT_SEC = 3600.0
MAX_MODEL_LEN = 1_000_000
RATE_COOLDOWN_SEC = 15
DEFAULT_2NODE_REMOTE_HOST = os.environ.get(
    "FIG12_NANO_2NODE_REMOTE_HOST", "10.102.243.60"
)
SSH_USER = os.environ.get("FIG12_NANO_SSH_USER", "ailab")
SSH_KEY = Path(os.environ.get("FIG12_NANO_SSH_KEY", "/root/.ssh/id_rsa_pjlab"))
REMOTE_CONTAINER = os.environ.get("FIG12_NANO_REMOTE_CONTAINER", "ae_merged")
GPUS_PER_NODE = 8


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
        "--run-id",
        default=None,
        help="Optional output identifier for a fresh run.",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume RUN_ID, skipping rates already recorded as ok.",
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
            f"(default: {SEND_DURATION_SEC:g})."
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
        "--remote-host",
        default=DEFAULT_2NODE_REMOTE_HOST,
        help="Worker SSH host for the two-node run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume is not None and args.run_id is not None:
        parser.error("--resume and --run-id cannot be used together")
    if args.resume is not None and args.dry_run:
        parser.error("--resume and --dry-run cannot be used together")
    quick_test = bool(
        args.rate
        or args.bench_duration_sec is not None
        or args.timeout_sec is not None
        or args.max_request_tokens is not None
    )
    if quick_test and args.timeout_sec is None:
        args.timeout_sec = DEFAULT_TIMEOUT_SEC
    args.rate = list(dict.fromkeys(args.rate))
    args.run_id = (
        args.resume or args.run_id or dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    if not args.run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    return args


def require_path(path: Path, label: str, directory: bool) -> None:
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {path}")


def scaled_rates(rates: tuple[float, ...], nodes: int) -> tuple[float, ...]:
    scale = nodes / 4.0
    return tuple(rate * scale for rate in rates)


def workload_settings(
    args: argparse.Namespace,
    workload: Workload,
    output_dir: Path,
) -> tuple[tuple[float, ...], float, float | None]:
    previous_config: dict[str, object] = {}
    config_path = output_dir / "config.json"
    if args.resume is not None and config_path.is_file():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.rate:
        rates = tuple(args.rate)
    elif isinstance(previous_config.get("rates"), list):
        rates = tuple(float(rate) for rate in previous_config["rates"])
    else:
        rates = scaled_rates(workload.nano_rates, args.nodes)

    if args.bench_duration_sec is not None:
        duration_sec = args.bench_duration_sec
    elif previous_config.get("send_duration_sec") is not None:
        duration_sec = float(previous_config["send_duration_sec"])
    else:
        duration_sec = float(SEND_DURATION_SEC)

    if args.timeout_sec is not None:
        timeout_sec = args.timeout_sec
    elif previous_config.get("timeout_sec") is not None:
        timeout_sec = float(previous_config["timeout_sec"])
    elif args.nodes == 2:
        timeout_sec = DEFAULT_TIMEOUT_SEC
    else:
        timeout_sec = None
    return rates, duration_sec, timeout_sec


def rate_key(rate: float) -> str:
    return f"{rate:g}"


def completed_rates(summary_path: Path) -> set[str]:
    if not summary_path.is_file():
        return set()
    completed: set[str] = set()
    with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            try:
                completed.add(rate_key(float(row["rate"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def next_rate_dir(output_dir: Path, rate_tag: str) -> Path:
    base = output_dir / f"rate_{rate_tag}"
    if not base.exists():
        return base
    retry = 1
    while True:
        candidate = output_dir / f"rate_{rate_tag}_retry{retry}"
        if not candidate.exists():
            return candidate
        retry += 1


def split_host_port(address: str) -> tuple[str, int]:
    try:
        host, port_text = address.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid Ray address: {address!r}") from error
    if not host or not 1 <= port <= 65535:
        raise ValueError(f"invalid Ray address: {address!r}")
    return host, port


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def local_addresses() -> set[str]:
    completed = subprocess.run(
        ["hostname", "-I"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return set(completed.stdout.split())


def run_setup_command(command: list[str], label: str, timeout: int = 90) -> str:
    print(f"[ray] {label}", flush=True)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} timed out after {timeout}s") from error
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = f"\n{output}" if output else ""
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}{detail}"
        )
    return output


def ssh_container_command(host: str, command: list[str]) -> list[str]:
    inner = shlex.join(command)
    remote = (
        f"docker exec {shlex.quote(REMOTE_CONTAINER)} " f"bash -lc {shlex.quote(inner)}"
    )
    return [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{SSH_USER}@{host}",
        remote,
    ]


def ray_nodes(address: str) -> list[dict[str, object]]:
    probe = """
import json
import ray

ray.init(address=%r, logging_level="ERROR")
nodes = [
    {
        "address": node.get("NodeManagerAddress"),
        "alive": bool(node.get("Alive")),
        "gpus": float(node.get("Resources", {}).get("GPU", 0)),
    }
    for node in ray.nodes()
]
print("FIG12_RAY_NODES=" + json.dumps(nodes))
ray.shutdown()
""" % (
        address,
    )
    output = run_setup_command(
        [sys.executable, "-c", probe], "checking Ray cluster", timeout=45
    )
    marker = "FIG12_RAY_NODES="
    for line in reversed(output.splitlines()):
        if line.startswith(marker):
            value = json.loads(line[len(marker) :])
            if isinstance(value, list):
                return value
    raise RuntimeError("Ray cluster check returned no node information")


def wait_for_ray_nodes(address: str, expected_nodes: int) -> list[dict[str, object]]:
    deadline = time.monotonic() + 90
    last_nodes: list[dict[str, object]] = []
    last_error = ""
    while time.monotonic() < deadline:
        try:
            last_nodes = ray_nodes(address)
            alive = [node for node in last_nodes if node["alive"]]
            gpu_count = sum(float(node["gpus"]) for node in alive)
            if (
                len(alive) >= expected_nodes
                and gpu_count >= expected_nodes * GPUS_PER_NODE
            ):
                return alive
            last_error = f"found {len(alive)} active node(s) and {gpu_count:g} GPU(s)"
        except RuntimeError as error:
            last_error = str(error)
        time.sleep(2)
    raise RuntimeError(
        f"Ray cluster did not reach {expected_nodes} nodes with "
        f"{expected_nodes * GPUS_PER_NODE} GPUs: {last_error or last_nodes}"
    )


@dataclass
class RaySession:
    address: str
    remote_hosts: list[str]
    started_head: bool = False
    started_workers: list[str] = field(default_factory=list)

    def stop(self) -> None:
        for host in reversed(self.started_workers):
            try:
                run_setup_command(
                    ssh_container_command(host, ["ray", "stop", "--force"]),
                    f"stopping Ray worker on {host}",
                )
                self.started_workers.remove(host)
            except RuntimeError as error:
                print(f"[ray] cleanup warning: {error}", file=sys.stderr, flush=True)
        if self.started_head:
            try:
                run_setup_command(["ray", "stop", "--force"], "stopping local Ray head")
                self.started_head = False
            except RuntimeError as error:
                print(f"[ray] cleanup warning: {error}", file=sys.stderr, flush=True)


def prepare_ray_cluster(args: argparse.Namespace) -> RaySession:
    head_host, head_port = split_host_port(NANO_RAY_ADDR)
    remote_hosts = [args.remote_host] if args.nodes == 2 else []
    session = RaySession(NANO_RAY_ADDR, remote_hosts)
    try:
        if args.nodes != 2 and port_is_open(head_host, head_port):
            try:
                nodes = ray_nodes(NANO_RAY_ADDR)
                alive = [node for node in nodes if node["alive"]]
                gpu_count = sum(float(node["gpus"]) for node in alive)
                if len(alive) >= args.nodes and gpu_count >= args.nodes * GPUS_PER_NODE:
                    addresses = ", ".join(str(node["address"]) for node in alive)
                    print(
                        f"[ray] reusing {len(alive)}-node cluster ({addresses})",
                        flush=True,
                    )
                    return session
            except RuntimeError:
                if args.nodes != 2:
                    raise

        if args.nodes == 2:
            # A node can only belong to one Ray cluster. Clear stale runtimes in
            # these two dedicated containers before creating the requested pair.
            run_setup_command(
                ["ray", "stop", "--force"], "clearing stale local Ray runtime"
            )
            run_setup_command(
                ssh_container_command(args.remote_host, ["ray", "stop", "--force"]),
                f"clearing stale Ray runtime on {args.remote_host}/{REMOTE_CONTAINER}",
            )

        if not port_is_open(head_host, head_port):
            if head_host not in local_addresses():
                raise RuntimeError(
                    f"Ray head {head_host} is not this node. Set NANO_RAY_ADDR "
                    "to this node's address."
                )
            run_setup_command(
                [
                    "ray",
                    "start",
                    "--head",
                    "--node-ip-address",
                    head_host,
                    "--port",
                    str(head_port),
                    "--num-gpus",
                    str(GPUS_PER_NODE),
                    "--include-dashboard=false",
                    "--disable-usage-stats",
                ],
                f"starting local Ray head at {NANO_RAY_ADDR}",
            )
            session.started_head = True

        nodes = ray_nodes(NANO_RAY_ADDR)
        alive = [node for node in nodes if node["alive"]]
        gpu_count = sum(float(node["gpus"]) for node in alive)
        if len(alive) < args.nodes or gpu_count < args.nodes * GPUS_PER_NODE:
            if args.nodes != 2:
                raise RuntimeError(
                    f"Ray has {len(alive)} active node(s) and {gpu_count:g} GPU(s); "
                    "the four-node run requires an existing 32-GPU Ray cluster"
                )
            if not SSH_KEY.is_file():
                raise FileNotFoundError(f"SSH key not found: {SSH_KEY}")
            run_setup_command(
                ssh_container_command(
                    args.remote_host,
                    [
                        "ray",
                        "start",
                        "--address",
                        NANO_RAY_ADDR,
                        "--node-ip-address",
                        args.remote_host,
                        "--num-gpus",
                        str(GPUS_PER_NODE),
                        "--disable-usage-stats",
                    ],
                ),
                f"starting Ray worker on {args.remote_host}/{REMOTE_CONTAINER}",
            )
            session.started_workers.append(args.remote_host)

        alive = wait_for_ray_nodes(NANO_RAY_ADDR, args.nodes)
        addresses = ", ".join(str(node["address"]) for node in alive)
        print(f"[ray] ready: {len(alive)} nodes ({addresses})", flush=True)
        return session
    except BaseException:
        session.stop()
        raise


def benchmark_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "DG_PRINT_CONFIGS",
        "DG_JIT_DEBUG",
        "NANODEPLOY_MOE_GEMM_DEBUG",
        "NANODEPLOY_MOE_GEMM_DEBUG_RANKS",
        "NANODEPLOY_MOE_GEMM_DEBUG_LAYERS",
        "NANODEPLOY_MOE_GEMM_DEBUG_GEMMS",
        "NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS",
        "NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS",
    ):
        environment.pop(name, None)

    build_lib = NANODEPLOY_WORKDIR / "build" / "lib"
    python_paths = [str(NANODEPLOY_WORKDIR)]
    if build_lib.is_dir():
        python_paths.append(str(build_lib))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])

    environment.update(
        {
            "NANODEPLOY_WORKDIR": str(NANODEPLOY_WORKDIR),
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
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": environment.get(
                "NANODEPLOY_LOG_DECODE_STEP_DETAIL", "1"
            ),
            "RAY_DEDUP_LOGS": "0",
        }
    )
    return environment


def build_command(
    *,
    nodes: int,
    model_path: Path,
    dataset_path: Path,
    rate: float,
    bench_duration_sec: float,
    itl_log: Path,
    max_request_tokens: int | None,
) -> list[str]:
    dp_size = nodes
    sp_size = 8
    max_num_seqs = 256 if nodes == 4 else 128
    max_model_len = max_request_tokens or MAX_MODEL_LEN
    num_requests = max(1, round(rate * bench_duration_sec))
    return [
        sys.executable,
        "-u",
        str(NANO_BENCHMARK),
        "--dataset",
        "csv",
        "--csv-path",
        str(dataset_path),
        "--max-request-tokens",
        "0" if max_request_tokens is None else str(max_request_tokens),
        "--num-requests",
        str(num_requests),
        "--request-rate",
        f"{rate:g}",
        "--sp",
        str(sp_size),
        "--dp",
        str(dp_size),
        "--ep",
        str(dp_size * sp_size),
        "--tp",
        "1",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-limit-gb",
        "141",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(max_model_len),
        "--dummy-prefill",
        "--ray-address",
        NANO_RAY_ADDR,
        "--master-address",
        NANO_MASTER_ADDR,
        "--loop-count",
        "16",
        "--model-path",
        str(model_path),
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
        "--max-input-len",
        str(MAX_MODEL_LEN),
    ]


def run_logged(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    timeout_sec: float | None,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=NANODEPLOY_WORKDIR,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        last_report = started
        try:
            while True:
                try:
                    return process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if timeout_sec is not None and now - started >= timeout_sec:
                        print(
                            f"           timeout after {timeout_sec:g}s; "
                            "stopping benchmark...",
                            flush=True,
                        )
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=10)
                        return 124
                    if now - last_report >= 60:
                        progress = latest_benchmark_progress(log_path)
                        if progress is None:
                            progress = "initializing"
                        print(
                            f"           progress={progress}; "
                            f"elapsed={(now - started) / 60:.1f} min",
                            flush=True,
                        )
                        last_report = now
        except KeyboardInterrupt:
            print("\n           interrupt received; stopping benchmark...", flush=True)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
            raise


def latest_benchmark_progress(log_path: Path) -> str | None:
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 512 * 1024))
        text = handle.read().decode("utf-8", errors="replace")
    for line in reversed(re.split(r"[\r\n]", text)):
        if "Processing Requests:" not in line:
            continue
        match = re.search(r"(\d+)/(\d+)\s*\[([^\]]+)]", line)
        if match is None:
            continue
        completed = int(match.group(1))
        total = int(match.group(2))
        details = re.sub(r"\x1b\[[0-9;]*m", "", match.group(3))
        percent = 100.0 * completed / total
        return f"{completed}/{total} ({percent:.1f}%; {details})"
    return None


def run_workload(
    args: argparse.Namespace,
    workload: Workload,
    output_dir: Path,
    dataset_path: Path,
) -> int:
    rates, bench_duration_sec, timeout_sec = workload_settings(
        args, workload, output_dir
    )
    summary_path = output_dir / "run_summary.tsv"
    config = {
        "nodes": args.nodes,
        "topology": f"dp{args.nodes}_sp8_tp1_ep{args.nodes * 8}",
        "model": str(workload.model_path),
        "dataset": str(dataset_path),
        "rates": list(rates),
        "send_duration_sec": bench_duration_sec,
        "ray_address": NANO_RAY_ADDR,
        "master_address": NANO_MASTER_ADDR,
        "scheduler": "legacy_global/bucket",
    }
    if args.max_request_tokens is not None:
        config["source_dataset"] = str(workload.dataset_path)
        config["max_request_tokens"] = args.max_request_tokens
    if timeout_sec is not None:
        config["timeout_sec"] = timeout_sec
    config_path = output_dir / "config.json"
    if output_dir.exists():
        if args.resume is None:
            raise FileExistsError(f"result directory already exists: {output_dir}")
        if not config_path.is_file():
            raise FileNotFoundError(f"resume config not found: {config_path}")
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        checked_keys = [
            "nodes",
            "topology",
            "model",
            "dataset",
            "rates",
            "send_duration_sec",
        ]
        if timeout_sec is not None or "timeout_sec" in previous_config:
            checked_keys.append("timeout_sec")
        if (
            args.max_request_tokens is not None
            or "max_request_tokens" in previous_config
        ):
            checked_keys.append("max_request_tokens")
        for key in checked_keys:
            previous_value = previous_config.get(key)
            if key == "timeout_sec" and previous_value is None and args.nodes == 2:
                previous_value = DEFAULT_TIMEOUT_SEC
            if previous_value != config.get(key):
                raise ValueError(
                    f"resume configuration mismatch for {workload.slug}: {key}"
                )
        if not summary_path.is_file():
            raise FileNotFoundError(f"resume summary not found: {summary_path}")
    else:
        output_dir.mkdir(parents=True)
        summary_path.write_text(
            "rate\tnum_requests\tstatus\tlog_dir\n", encoding="utf-8"
        )
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    completed = completed_rates(summary_path) if args.resume is not None else set()

    environment = benchmark_environment()
    for rate_index, rate in enumerate(rates):
        key = rate_key(rate)
        if key in completed:
            print(
                f"  [skip {rate_index + 1}/{len(rates)}] rate={key} already ok",
                flush=True,
            )
            continue
        rate_tag = key.replace(".", "p")
        rate_dir = next_rate_dir(output_dir, rate_tag)
        rate_dir.mkdir()
        itl_log = rate_dir / "itl_samples.jsonl"
        driver_log = rate_dir / "driver.log"
        command = build_command(
            nodes=args.nodes,
            model_path=workload.model_path,
            dataset_path=dataset_path,
            rate=rate,
            bench_duration_sec=bench_duration_sec,
            itl_log=itl_log,
            max_request_tokens=args.max_request_tokens,
        )
        (rate_dir / "command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        num_requests = max(1, round(rate * bench_duration_sec))
        print(
            f"  [rate {rate_index + 1}/{len(rates)}] "
            f"rate={rate:g} requests={num_requests}",
            flush=True,
        )
        if args.dry_run:
            status = "planned"
        else:
            return_code = run_logged(command, driver_log, environment, timeout_sec)
            if return_code == 0:
                status = "ok"
            elif return_code == 124:
                status = "timeout"
            else:
                status = "failed"
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(f"{key}\t{num_requests}\t{status}\t{rate_dir}\n")
        if status in {"failed", "timeout"}:
            return 1
        if not args.dry_run and rate_index + 1 < len(rates):
            time.sleep(RATE_COOLDOWN_SEC)
    return 0


def main() -> int:
    args = parse_args()
    workloads = selected_workloads(args.workload)
    result_root = (
        FIG12_DIR / "results" / "e2e" / "nanodeploy" / f"{args.nodes}node" / args.run_id
    )

    require_path(NANO_BENCHMARK, "NanoDeploy benchmark", directory=False)
    require_path(NANODEPLOY_WORKDIR, "NanoDeploy checkout", directory=True)
    for workload in workloads:
        require_path(workload.model_path, workload.label + " model", directory=True)
        require_path(
            workload.dataset_path, workload.label + " dataset", directory=False
        )

    if args.resume is not None and not result_root.is_dir():
        raise FileNotFoundError(f"resume result root not found: {result_root}")

    dataset_paths: dict[str, Path] = {}
    if args.max_request_tokens is not None:
        filtered_root = result_root / "_filtered_csv"
        for workload in workloads:
            if workload.dataset_name in dataset_paths:
                continue
            destination = (
                filtered_root
                / f"{workload.dataset_name}_max{args.max_request_tokens}.csv"
            )
            total_rows, kept_rows = filter_dataset_by_request_tokens(
                workload.dataset_path,
                destination,
                args.max_request_tokens,
            )
            dataset_paths[workload.dataset_name] = destination.resolve()
            print(
                f"[dataset] {workload.dataset_name}: kept "
                f"{kept_rows}/{total_rows}, removed "
                f"{total_rows - kept_rows} above "
                f"{args.max_request_tokens} tokens",
                flush=True,
            )

    print(f"Nodes: {args.nodes}", flush=True)
    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Mode: {'resume' if args.resume is not None else 'fresh'}", flush=True)
    print(f"Result root: {result_root}", flush=True)
    print(f"Ray address: {NANO_RAY_ADDR}", flush=True)
    print(f"Gloo master address: {NANO_MASTER_ADDR}", flush=True)
    if args.max_request_tokens is not None:
        print(
            f"Max request tokens: {args.max_request_tokens} "
            "(prompt_len + output_len)",
            flush=True,
        )

    ray_session: RaySession | None = None
    if not args.dry_run:
        ray_session = prepare_ray_cluster(args)
        atexit.register(ray_session.stop)

    failures: list[str] = []
    try:
        for index, workload in enumerate(workloads, start=1):
            output_dir = result_root / workload.slug
            rates, bench_duration_sec, timeout_sec = workload_settings(
                args, workload, output_dir
            )
            timeout_text = "none" if timeout_sec is None else f"{timeout_sec:g}s"
            print(
                f"[{index}/{len(workloads)}] {workload.label}: "
                f"rates={format_rates(rates)}, "
                f"duration={bench_duration_sec:g}s, timeout={timeout_text}",
                flush=True,
            )
            dataset_path = dataset_paths.get(
                workload.dataset_name, workload.dataset_path
            )
            status = run_workload(args, workload, output_dir, dataset_path)
            if status != 0:
                failures.append(workload.slug)
    except KeyboardInterrupt:
        print("\nInterrupted; cleaning up the Ray cluster.", flush=True)
        return 130
    finally:
        if ray_session is not None:
            ray_session.stop()
            atexit.unregister(ray_session.stop)

    if failures:
        print("Failed workloads: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

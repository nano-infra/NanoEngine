#!/usr/bin/env python3
"""Launch the Figure 3 DeepEP sweep on node 0 and its SSH workers."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIG3_DIR = SCRIPT_DIR.parent
AE_ROOT = FIG3_DIR.parent
WORKER_SCRIPT = SCRIPT_DIR / "run_deepep.sh"
DEFAULT_REMOTE_HOSTS = os.environ.get(
    "FIG3_DEEPEP_REMOTE_HOSTS",
    "h200-rjob1,h200-rjob2,h200-rjob3",
)
DEFAULT_SSH_CONFIG = Path("/root/.ssh/config")
DEFAULT_ENV_SCRIPT = Path("/root/.zshrc")
FULL_BATCH_SIZES = tuple(range(1, 9)) + tuple(range(16, 257, 8))
QUICK_BATCH_SIZES = (1, 4, 8, 32, 56, 80, 104, 128, 152, 176, 200, 224, 256)
EXPECTED_DEEPEP_VERSION = "1.2.1+73b6ea4"


@dataclass
class RankProcess:
    rank: int
    host: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: object
    reader: threading.Thread


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be an integer in [1, 65535]")
    return parsed


def parse_list(raw: str) -> list[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def parse_batch_sizes(raw: str) -> tuple[int, ...]:
    values = parse_list(raw)
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    parsed: list[int] = []
    for value in values:
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"batch size is not an integer: {value}"
            ) from error
        if number <= 0:
            raise argparse.ArgumentTypeError(
                f"batch size must be positive: {value}"
            )
        if number not in parsed:
            parsed.append(number)
    return tuple(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote-hosts",
        nargs="+",
        default=parse_list(DEFAULT_REMOTE_HOSTS),
        metavar="HOST",
        help=(
            "Worker SSH aliases, in node-rank order "
            "(default: %(default)s; env: FIG3_DEEPEP_REMOTE_HOSTS)."
        ),
    )
    parser.add_argument(
        "--master-addr",
        help="Node-0 address reachable from every worker (default: bond0 IPv4).",
    )
    parser.add_argument(
        "--master-port",
        type=port,
        default=18361,
        help="Base rendezvous port (default: %(default)s).",
    )
    parser.add_argument(
        "--run-id",
        default="ae_run1",
        help="Output identifier shared by every node (default: %(default)s).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the reduced 13-point trend-check sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        help="Custom comma- or space-separated batch sizes per GPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Shared result directory (default: fig3/results/deepep/<run>_<nodes>nodes).",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python command used on every node (default: %(default)s).",
    )
    parser.add_argument(
        "--token-timeout",
        type=positive_int,
        default=300,
        metavar="SECONDS",
        help="Timeout for one token attempt (default: %(default)s).",
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=3,
        help="Attempts per token after timeout/failure (default: %(default)s).",
    )
    parser.add_argument(
        "--ssh-config",
        type=Path,
        default=DEFAULT_SSH_CONFIG,
        help="SSH configuration containing the worker aliases (default: %(default)s).",
    )
    parser.add_argument(
        "--env-script",
        type=Path,
        default=DEFAULT_ENV_SCRIPT,
        help="Per-node environment script sourced before launch (default: %(default)s).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=positive_int,
        default=10,
        metavar="SECONDS",
        help="SSH connection timeout (default: %(default)s).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check all nodes without starting the benchmark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved rank commands without SSH or benchmark execution.",
    )
    args = parser.parse_args()

    args.remote_hosts = parse_list(" ".join(args.remote_hosts))
    if not args.remote_hosts:
        parser.error("--remote-hosts must contain at least one worker")
    if len(set(args.remote_hosts)) != len(args.remote_hosts):
        parser.error("--remote-hosts contains a duplicate worker alias")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error(
            "--run-id may only contain letters, digits, dot, underscore, and hyphen"
        )
    if args.quick and args.batch_sizes is not None:
        parser.error("use either --quick or --batch-sizes, not both")
    if not args.ssh_config.is_file():
        parser.error(f"SSH configuration not found: {args.ssh_config}")
    if not args.env_script.is_file():
        parser.error(f"environment script not found: {args.env_script}")
    if not WORKER_SCRIPT.is_file():
        parser.error(f"DeepEP worker script not found: {WORKER_SCRIPT}")
    return args


def detect_bond0_ipv4() -> str:
    try:
        completed = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", "bond0"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "could not query bond0; pass --master-addr explicitly"
        ) from error
    match = re.search(r"\binet\s+(\d+(?:\.\d+){3})/", completed.stdout)
    if match is None:
        raise RuntimeError(
            "bond0 has no IPv4 address; pass --master-addr explicitly"
        )
    return match.group(1)


def validate_master_addr(address: str) -> None:
    if address == "0.0.0.0" or address.startswith("127."):
        raise RuntimeError(f"master address is not reachable by workers: {address}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
    except OSError as error:
        raise RuntimeError(
            f"master address is not assigned to node 0: {address}"
        ) from error


def ssh_options(args: argparse.Namespace) -> list[str]:
    return [
        "-F",
        str(args.ssh_config.expanduser().resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UpdateHostKeys=no",
    ]


def shell_command(command: list[str], env_script: Path) -> str:
    setup = (
        f"source {shlex.quote(str(env_script.expanduser().resolve()))}; "
        "set -euo pipefail; "
        f"cd {shlex.quote(str(AE_ROOT))}; "
    )
    return setup + "exec " + shlex.join(command)


def local_shell_argv(command: str) -> list[str]:
    return ["zsh", "-lc", command]


def remote_ssh_argv(
    args: argparse.Namespace,
    host: str,
    command: str,
) -> list[str]:
    return [
        "ssh",
        *ssh_options(args),
        host,
        shlex.join(["zsh", "-lc", command]),
    ]


def preflight_command(args: argparse.Namespace) -> str:
    python_probe = (
        "import importlib.metadata, pathlib, torch; "
        f"expected={EXPECTED_DEEPEP_VERSION!r}; "
        "actual=importlib.metadata.version('deep_ep'); "
        "assert actual == expected, f'deep_ep {actual} != {expected}'; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "assert torch.cuda.device_count() == 8, "
        "f'expected 8 GPUs, found {torch.cuda.device_count()}'; "
        "print('FIG3_PREFLIGHT=' + pathlib.Path("
        "'/proc/sys/kernel/random/boot_id').read_text().strip() + "
        "' GPUs=' + str(torch.cuda.device_count()) + ' deep_ep=' + actual)"
    )
    checks = [
        "test",
        "-f",
        str(WORKER_SCRIPT),
    ]
    command = shlex.join(checks) + " && " + shlex.join(
        [args.python, "-c", python_probe]
    )
    return shell_command(["bash", "-c", command], args.env_script)


def run_preflight(args: argparse.Namespace) -> None:
    command = preflight_command(args)
    print("[preflight] checking node 0", flush=True)
    local = subprocess.run(
        local_shell_argv(command),
        cwd=AE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if local.returncode != 0:
        detail = (local.stderr or local.stdout).strip()
        raise RuntimeError(f"node 0 preflight failed: {detail}")

    marker = "FIG3_PREFLIGHT="
    identities: dict[str, str] = {}
    for line in local.stdout.splitlines():
        if line.startswith(marker):
            identities[line.split()[0].removeprefix(marker)] = "node 0"
    if not identities:
        raise RuntimeError("node 0 preflight returned no node identity")
    print("[preflight] node 0: 8 GPUs, pinned DeepEP package", flush=True)

    for rank, host in enumerate(args.remote_hosts, start=1):
        print(f"[preflight] checking node {rank} via {host}", flush=True)
        try:
            completed = subprocess.run(
                remote_ssh_argv(args, host, command),
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"worker preflight timed out: {host}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"worker preflight failed for {host}: "
                f"{detail or f'exit code {completed.returncode}'}"
            )
        identity = ""
        for line in completed.stdout.splitlines():
            if line.startswith(marker):
                identity = line.split()[0].removeprefix(marker)
                break
        if not identity:
            raise RuntimeError(f"worker preflight returned no node identity: {host}")
        if identity in identities:
            raise RuntimeError(
                f"worker {host} duplicates {identities[identity]}; "
                "each SSH alias must refer to a different node"
            )
        identities[identity] = host
        print(
            f"[preflight] node {rank} ({host}): 8 GPUs, pinned DeepEP package",
            flush=True,
        )
    print(f"[preflight] {len(identities)} distinct nodes are ready", flush=True)


def worker_command(
    args: argparse.Namespace,
    *,
    rank: int,
    num_nodes: int,
    master_addr: str,
    batch_sizes: tuple[int, ...],
    output_dir: Path,
) -> list[str]:
    return [
        "bash",
        str(WORKER_SCRIPT.relative_to(AE_ROOT)),
        "--num-nodes",
        str(num_nodes),
        "--node-rank",
        str(rank),
        "--master-addr",
        master_addr,
        "--master-port",
        str(args.master_port),
        "--run-id",
        args.run_id,
        "--python",
        args.python,
        "--batch-sizes",
        " ".join(str(value) for value in batch_sizes),
        "--output-dir",
        str(output_dir),
        "--token-timeout",
        str(args.token_timeout),
        "--max-attempts",
        str(args.max_attempts),
    ]


def stream_output(
    stream: object,
    log_handle: object,
    *,
    show_on_console: bool,
) -> None:
    try:
        for line in stream:
            log_handle.write(line)
            log_handle.flush()
            if show_on_console:
                print(line, end="", flush=True)
    finally:
        stream.close()


def start_rank(
    args: argparse.Namespace,
    *,
    rank: int,
    host: str,
    command: list[str],
    log_path: Path,
) -> RankProcess:
    wrapped = shell_command(command, args.env_script)
    argv = (
        local_shell_argv(wrapped)
        if rank == 0
        else remote_ssh_argv(args, host, wrapped)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"\n===== launch {dt.datetime.now().astimezone().isoformat()} "
        f"rank={rank} host={host} =====\n"
    )
    log_handle.flush()
    process = subprocess.Popen(
        argv,
        cwd=AE_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    reader = threading.Thread(
        target=stream_output,
        args=(process.stdout, log_handle),
        kwargs={"show_on_console": rank == 0},
        daemon=True,
    )
    reader.start()
    return RankProcess(rank, host, process, log_path, log_handle, reader)


def stop_processes(ranks: list[RankProcess]) -> None:
    active = [item for item in ranks if item.process.poll() is None]
    for item in active:
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while active and time.monotonic() < deadline:
        active = [item for item in active if item.process.poll() is None]
        if active:
            time.sleep(0.2)
    for item in active:
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_cluster(
    args: argparse.Namespace,
    *,
    master_addr: str,
    batch_sizes: tuple[int, ...],
    output_dir: Path,
) -> None:
    num_nodes = 1 + len(args.remote_hosts)
    commands = {
        rank: worker_command(
            args,
            rank=rank,
            num_nodes=num_nodes,
            master_addr=master_addr,
            batch_sizes=batch_sizes,
            output_dir=output_dir,
        )
        for rank in range(num_nodes)
    }

    print(f"Nodes: {num_nodes}", flush=True)
    print(f"Master address: {master_addr}", flush=True)
    print(f"Worker SSH hosts: {' '.join(args.remote_hosts)}", flush=True)
    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Batch sizes ({len(batch_sizes)}): {' '.join(map(str, batch_sizes))}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    if args.dry_run:
        for rank, command in commands.items():
            host = "local" if rank == 0 else args.remote_hosts[rank - 1]
            wrapped = shell_command(command, args.env_script)
            argv = (
                local_shell_argv(wrapped)
                if rank == 0
                else remote_ssh_argv(args, host, wrapped)
            )
            print(f"\n[node {rank}: {host}]\n$ {shlex.join(argv)}", flush=True)
        return

    run_preflight(args)
    if args.preflight_only:
        return

    launcher_log_dir = output_dir / "launcher_logs"
    ranks: list[RankProcess] = []
    try:
        for rank, host in enumerate(args.remote_hosts, start=1):
            log_path = launcher_log_dir / f"node{rank}.log"
            print(f"[launch] node {rank} via {host}; log: {log_path}", flush=True)
            ranks.append(
                start_rank(
                    args,
                    rank=rank,
                    host=host,
                    command=commands[rank],
                    log_path=log_path,
                )
            )
        local_log = launcher_log_dir / "node0.log"
        print(f"[launch] node 0 locally; log: {local_log}", flush=True)
        ranks.append(
            start_rank(
                args,
                rank=0,
                host="local",
                command=commands[0],
                log_path=local_log,
            )
        )

        while True:
            failed = [
                item
                for item in ranks
                if item.process.poll() not in (None, 0)
            ]
            if failed:
                detail = "; ".join(
                    f"node {item.rank} ({item.host}) exit={item.process.returncode}"
                    for item in failed
                )
                raise RuntimeError(f"DeepEP rank failed: {detail}")
            if all(item.process.poll() == 0 for item in ranks):
                break
            time.sleep(0.5)
    except BaseException:
        stop_processes(ranks)
        raise
    finally:
        for item in ranks:
            item.reader.join(timeout=5)
            item.log_handle.close()

    missing = []
    for rank in range(num_nodes):
        summary = output_dir / f"node{rank}_summary_rank{rank * 8}.csv"
        if not summary.is_file() or summary.stat().st_size == 0:
            missing.append(summary)
    if missing:
        raise RuntimeError(
            "DeepEP ranks exited successfully but summaries are missing: "
            + ", ".join(str(path) for path in missing)
        )
    print("DeepEP cluster sweep completed successfully.", flush=True)
    for item in sorted(ranks, key=lambda rank_process: rank_process.rank):
        print(
            f"  node {item.rank} launcher log: {item.log_path}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    output_dir: Path | None = None
    try:
        master_addr = args.master_addr or detect_bond0_ipv4()
        validate_master_addr(master_addr)
        batch_sizes = (
            args.batch_sizes
            if args.batch_sizes is not None
            else QUICK_BATCH_SIZES if args.quick else FULL_BATCH_SIZES
        )
        if args.master_port + max(batch_sizes) > 65535:
            raise RuntimeError(
                "--master-port plus the largest batch size exceeds 65535"
            )
        num_nodes = 1 + len(args.remote_hosts)
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else FIG3_DIR
            / "results"
            / "deepep"
            / f"{args.run_id}_{num_nodes}nodes"
        ).expanduser().resolve()
        run_cluster(
            args,
            master_addr=master_addr,
            batch_sizes=batch_sizes,
            output_dir=output_dir,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        log_paths = (
            sorted((output_dir / "launcher_logs").glob("node*.log"))
            if output_dir is not None
            else []
        )
        for path in log_paths:
            excerpt = tail(path)
            if excerpt:
                print(f"\n--- tail: {path} ---\n{excerpt}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

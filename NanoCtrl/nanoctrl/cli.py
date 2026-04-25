from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3000


def _runtime_dir() -> Path:
    runtime = os.environ.get("NANOCTRL_RUNTIME_DIR")
    if runtime:
        return Path(runtime)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nanoctrl"
    return Path("/tmp/nanoctrl")


def _pid_file() -> Path:
    return _runtime_dir() / "nanoctrl.pid"


def _meta_file() -> Path:
    return _runtime_dir() / "nanoctrl.meta.json"


def _log_file() -> Path:
    return _runtime_dir() / "nanoctrl.log"


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid() -> int | None:
    pid_path = _pid_file()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_meta() -> dict[str, Any]:
    meta_path = _meta_file()
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_runtime(pid: int, address: str, cmd: list[str], log_path: Path) -> None:
    rt = _runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    _pid_file().write_text(str(pid), encoding="utf-8")
    _meta_file().write_text(
        json.dumps(
            {
                "pid": pid,
                "address": address,
                "cmd": cmd,
                "log": str(log_path),
                "started_at": int(time.time()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _cleanup_runtime() -> None:
    for path in (_pid_file(), _meta_file()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _normalize_address(address: str) -> str:
    if address.startswith(("http://", "https://")):
        return address.rstrip("/")
    return f"http://{address.rstrip('/')}"


def _find_manifest(config_path: Path) -> Path | None:
    candidates = [
        config_path.parent / "Cargo.toml",
        Path.cwd() / "Cargo.toml",
        Path(__file__).resolve().parents[1] / "Cargo.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _check_health(address: str, timeout: float = 1.5) -> tuple[bool, str]:
    url = _normalize_address(address)
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as c:
            r = c.get(url)
            if r.status_code == 200:
                return True, "ok"
            return False, f"http {r.status_code}"
    except httpx.HTTPError as e:
        return False, str(e)


def _resolve_command(args: argparse.Namespace) -> tuple[list[str], str, Path | None]:
    config_path = Path(args.config).resolve()
    server_address = f"{args.host}:{args.port}"

    if args.bin:
        cmd = [str(Path(args.bin).resolve()), "--config", str(config_path)]
        return cmd, server_address, None

    local_bin_candidates = [
        Path.cwd() / "target" / "release" / "nanoctrl-server",
        config_path.parent / "target" / "release" / "nanoctrl-server",
        Path(__file__).resolve().parents[1] / "target" / "release" / "nanoctrl-server",
    ]
    for local_bin in local_bin_candidates:
        if local_bin.exists() and os.access(local_bin, os.X_OK):
            cmd = [str(local_bin), "--config", str(config_path)]
            return cmd, server_address, None

    manifest_path = _find_manifest(config_path)
    if manifest_path is None:
        raise RuntimeError(
            "Cannot find NanoCtrl Cargo.toml. Use --bin to specify nanoctrl-server binary."
        )

    cmd = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(manifest_path),
        "--",
        "--config",
        str(config_path),
    ]
    return cmd, server_address, manifest_path.parent


def _start(args: argparse.Namespace) -> int:
    target_address = f"{args.host}:{args.port}"
    pid = _read_pid()
    if pid is not None and _is_pid_running(pid):
        print(f"nanoctrl is already running (pid={pid})")
        return 0
    if pid is not None:
        _cleanup_runtime()

    # If no pid file exists but health check is up, avoid launching a duplicate server.
    ok, _ = _check_health(target_address, timeout=0.8)
    if ok:
        print(
            "nanoctrl appears to be already running at "
            f"{_normalize_address(target_address)} (no pid file managed by this CLI)"
        )
        return 0

    try:
        cmd, server_address, run_cwd = _resolve_command(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    log_path = Path(args.log_file).resolve() if args.log_file else _log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=run_cwd,
        )

    _write_runtime(proc.pid, _normalize_address(server_address), cmd, log_path)

    deadline = time.time() + args.wait
    while time.time() < deadline:
        if proc.poll() is not None:
            _cleanup_runtime()
            print(
                f"nanoctrl failed to start (exit={proc.returncode}). See log: {log_path}",
                file=sys.stderr,
            )
            return 1
        ok, _ = _check_health(server_address, timeout=0.8)
        if ok:
            print(f"nanoctrl started (pid={proc.pid})")
            print(f"address: {_normalize_address(server_address)}")
            print(f"log: {log_path}")
            return 0
        time.sleep(0.2)

    print(f"nanoctrl process started (pid={proc.pid}), but health check timed out")
    print(f"address: {_normalize_address(server_address)}")
    print(f"log: {log_path}")
    return 0


def _status(args: argparse.Namespace) -> int:
    pid = _read_pid()
    meta = _read_meta()
    address = (
        args.address or meta.get("address") or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    )

    if pid is None:
        print("nanoctrl is not running (no pid file)")
        ok, detail = _check_health(address)
        print(f"health({address}): {'ok' if ok else 'down'} ({detail})")
        return 1

    running = _is_pid_running(pid)
    ok, detail = _check_health(address)
    print(f"pid: {pid}")
    print(f"process: {'running' if running else 'not running'}")
    print(f"address: {address}")
    print(f"health: {'ok' if ok else 'down'} ({detail})")
    if meta.get("log"):
        print(f"log: {meta['log']}")

    if not running:
        _cleanup_runtime()
        return 1
    return 0 if ok else 2


def _stop(args: argparse.Namespace) -> int:
    pid = _read_pid()
    if pid is None:
        print("nanoctrl is not running")
        return 0

    if not _is_pid_running(pid):
        _cleanup_runtime()
        print("nanoctrl is not running (stale pid file cleaned)")
        return 0

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not _is_pid_running(pid):
            _cleanup_runtime()
            print(f"nanoctrl stopped (pid={pid})")
            return 0
        time.sleep(0.2)

    if args.force:
        os.kill(pid, signal.SIGKILL)
        _cleanup_runtime()
        print(f"nanoctrl killed (pid={pid})")
        return 0

    print(
        f"nanoctrl did not stop within {args.timeout:.1f}s; rerun with --force",
        file=sys.stderr,
    )
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoctrl",
        description="NanoCtrl server process manager (start/status/stop)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start NanoCtrl server in background")
    p_start.add_argument(
        "-c", "--config", default="config.toml", help="Path to NanoCtrl config.toml"
    )
    p_start.add_argument("--bin", default=None, help="Path to nanoctrl-server binary")
    p_start.add_argument("--host", default=DEFAULT_HOST, help="Health-check host")
    p_start.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Health-check port"
    )
    p_start.add_argument(
        "--log-file", default=None, help="Log file path (default: runtime dir)"
    )
    p_start.add_argument(
        "--wait", type=float, default=8.0, help="Seconds to wait for health check"
    )
    p_start.set_defaults(func=_start)

    p_status = sub.add_parser("status", help="Show NanoCtrl status")
    p_status.add_argument(
        "--address",
        default=None,
        help="Server address for health check (default: from runtime metadata)",
    )
    p_status.set_defaults(func=_status)

    p_stop = sub.add_parser("stop", help="Stop NanoCtrl server")
    p_stop.add_argument(
        "--timeout", type=float, default=8.0, help="Graceful stop timeout in seconds"
    )
    p_stop.add_argument(
        "--force", action="store_true", help="Force kill if graceful stop times out"
    )
    p_stop.set_defaults(func=_stop)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

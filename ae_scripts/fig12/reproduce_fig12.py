#!/usr/bin/env python3
"""Run all fresh Figure 12 experiments and plot their results."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path


FIG12_DIR = Path(__file__).resolve().parent
AE_ROOT = FIG12_DIR.parent
NANO_LAUNCHER = FIG12_DIR / "launch_nanodeploy_e2e.py"
VLLM_LAUNCHER = FIG12_DIR / "launch_vllm_e2e.py"
PLOTTER = FIG12_DIR / "plot_fig12_end2end.py"
BENCH_DURATION_SEC = 600


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
        "--run-id",
        default=dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
        help="Run ID shared by NanoDeploy, vLLM, and plotting.",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        help=(
            "Optionally remove requests whose prompt_len + output_len exceeds "
            "this value. Disabled by default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and inspect all cases without launching experiments.",
    )
    args = parser.parse_args()
    if not args.run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    return args


def run(command: list[str]) -> None:
    print("\n$ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=AE_ROOT, check=True)


def main() -> int:
    args = parse_args()
    common = [
        "--nodes",
        str(args.nodes),
        "--run-id",
        args.run_id,
        "--bench-duration-sec",
        str(BENCH_DURATION_SEC),
    ]
    if args.max_request_tokens is not None:
        common.extend(["--max-request-tokens", str(args.max_request_tokens)])

    print(f"Nodes: {args.nodes}", flush=True)
    print(f"Run ID: {args.run_id}", flush=True)
    print(
        f"Request-sending duration per rate: {BENCH_DURATION_SEC}s",
        flush=True,
    )
    print(
        "Request-length filtering: "
        + (
            "disabled"
            if args.max_request_tokens is None
            else f"max {args.max_request_tokens} tokens"
        ),
        flush=True,
    )

    nano_command = [sys.executable, str(NANO_LAUNCHER), *common]
    vllm_command = [sys.executable, str(VLLM_LAUNCHER), *common]
    if args.dry_run:
        nano_command.append("--dry-run")
        vllm_command.append("--dry-run")

    run(nano_command)
    run(vllm_command)

    plot_command = [
        sys.executable,
        str(PLOTTER),
        "--nodes",
        str(args.nodes),
        "--nano-run",
        args.run_id,
        "--vllm-run",
        args.run_id,
    ]
    if args.dry_run:
        print(
            "\nDry run complete. After the real run, plotting will use:\n$ "
            + shlex.join(plot_command),
            flush=True,
        )
        return 0

    run(plot_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

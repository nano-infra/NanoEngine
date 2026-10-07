#!/usr/bin/env python3
"""Launch NanoDeploy torch-profiler runs for explicitly supplied inputs."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

from profile_common import parse_named_inputs, shell_command, tee_process


DEFAULT_MODEL_PATH = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
NANO_PROFILER = SCRIPT_DIR / "nano_dummy_prefill_profile.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the NanoDeploy profiler for one or more figure-owned input "
            "JSON files. The launcher does not choose a figure workload."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset name and processed_input_3d.json path; repeat as needed",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("PROFILE_NANO_RAY_ADDRESS", DEFAULT_RAY_ADDRESS),
        help=(
            f"Ray head host:port (default: {DEFAULT_RAY_ADDRESS}); override with "
            "PROFILE_NANO_RAY_ADDRESS"
        ),
    )
    parser.add_argument(
        "--master-address",
        default=os.environ.get("PROFILE_NANO_MASTER_ADDRESS"),
        help="Current distributed master host:port, or set PROFILE_NANO_MASTER_ADDRESS",
    )
    parser.add_argument("--config", default="dp4sp8")
    parser.add_argument("--sp-backend", default="hao_basic")
    parser.add_argument("--sp-size-policy", default="long_short")
    parser.add_argument("--long-request-sp-threshold", type=int, default=100_000)
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument("--max-num-seqs", type=int, default=192)
    parser.add_argument("--profiler-start-step", type=int, default=3)
    parser.add_argument("--profiling-step", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--loop-count", type=int, default=16)
    parser.add_argument("--max-num-recv-seqs", type=int, default=70)
    parser.add_argument("--max-num-send-seqs", type=int, default=70)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow adding new traces to an output tree that already has this dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        args.inputs = parse_named_inputs(args.input)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if not args.master_address:
        parser.error("--master-address is required")
    for name in (
        "long_request_sp_threshold",
        "segment_size",
        "max_num_seqs",
        "profiling_step",
        "max_tokens",
        "loop_count",
        "max_num_recv_seqs",
        "max_num_send_seqs",
        "max_model_len",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.profiler_start_step < 0:
        parser.error("--profiler-start-step must be non-negative")
    return args


def policy_tag(args: argparse.Namespace) -> str:
    if args.sp_size_policy == "long_short":
        return f"long_short_thr{args.long_request_sp_threshold}"
    if args.config == "dp4fixedsp8":
        return "fixed_sp8"
    if args.sp_size_policy == "legacy":
        return f"legacy_seg{args.segment_size}"
    return args.sp_size_policy


def main() -> None:
    args = parse_args()
    if not NANO_PROFILER.is_file():
        raise SystemExit(f"AE NanoDeploy profiler not found: {NANO_PROFILER}")
    if importlib.util.find_spec("nanodeploy") is None:
        raise SystemExit(
            "the nanodeploy package is not installed in the active Python environment"
        )

    output_root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    base_command = [
        sys.executable,
        str(NANO_PROFILER),
        "--model-path",
        str(args.model_path.expanduser().resolve()),
        "--master-address",
        args.master_address,
        "--ray-address",
        args.ray_address,
        "--config",
        args.config,
        "--sp-backend",
        args.sp_backend,
        "--sp-size-policy",
        args.sp_size_policy,
        "--long-request-sp-threshold",
        str(args.long_request_sp_threshold),
        "--segment-size",
        str(args.segment_size),
        "--profiler-start-step",
        str(args.profiler_start_step),
        "--profiling-step",
        str(args.profiling_step),
        "--max-tokens",
        str(args.max_tokens),
        "--loop-count",
        str(args.loop_count),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-recv-seqs",
        str(args.max_num_recv_seqs),
        "--max-num-send-seqs",
        str(args.max_num_send_seqs),
        "--max-model-len",
        str(args.max_model_len),
    ]

    for dataset, input_path in args.inputs:
        inferred_dataset = input_path.parent.name
        if dataset != inferred_dataset:
            raise SystemExit(
                "NanoDeploy derives the output dataset name from the input's "
                f"parent directory. Use --input {inferred_dataset}={input_path} "
                f"instead of {dataset}={input_path}."
            )
        expected_dir = output_root / dataset / args.config / policy_tag(args)
        if expected_dir.exists() and not args.allow_existing_output and not args.dry_run:
            raise SystemExit(
                f"Refusing to mix profiler traces in existing directory: {expected_dir}\n"
                "Choose a fresh --output-root or pass --allow-existing-output."
            )
        command = [
            *base_command,
            "--sp-seq-lens-file",
            str(input_path),
            "--profiler-dir",
            str(output_root),
        ]
        print(f"\n===== NanoDeploy profile: {dataset} =====", flush=True)
        print(shell_command(command), flush=True)
        if args.dry_run:
            continue

        env = os.environ.copy()
        log_path = output_root / "logs" / f"{dataset}.{policy_tag(args)}.log"
        return_code = tee_process(command, log_path, cwd=AE_ROOT, env=env)
        if return_code != 0:
            raise SystemExit(
                f"NanoDeploy profiling failed for {dataset} with exit code "
                f"{return_code}; see {log_path}"
            )


if __name__ == "__main__":
    main()

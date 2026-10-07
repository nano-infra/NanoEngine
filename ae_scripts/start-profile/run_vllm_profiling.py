#!/usr/bin/env python3
"""Launch the vLLM offline torch-profiler matrix for supplied length JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

from profile_common import parse_named_inputs, shell_command


DEFAULT_VLLM_ROOT = Path(require_path("AE_VLLM_ROOT"))
SUPPORTED_STRATEGIES = ("dp4dcp8", "dp8dcp4", "dp16cp2", "dp32")


def parse_strategy_max_num_seqs(values: list[str]) -> dict[str, int]:
    """Parse repeated STRATEGY=N overrides without changing shared defaults."""
    overrides: dict[str, int] = {}
    for item in values:
        strategy, separator, raw_value = item.partition("=")
        if not separator or not strategy or not raw_value:
            raise argparse.ArgumentTypeError(
                f"invalid --strategy-max-num-seqs value {item!r}; expected STRATEGY=N"
            )
        if strategy not in SUPPORTED_STRATEGIES:
            supported = ", ".join(SUPPORTED_STRATEGIES)
            raise argparse.ArgumentTypeError(
                f"unsupported strategy {strategy!r} in --strategy-max-num-seqs; "
                f"choose from {supported}"
            )
        if strategy in overrides:
            raise argparse.ArgumentTypeError(
                f"duplicate --strategy-max-num-seqs override for {strategy!r}"
            )
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid max-num-seqs {raw_value!r} for {strategy!r}; expected an integer"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"max-num-seqs for {strategy!r} must be positive"
            )
        overrides[strategy] = value
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run vLLM's multinode offline profiler for an explicit strategy "
            "x input matrix. Figure-specific inputs stay outside this directory."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset name and flat length-JSON path; repeat as needed",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        required=True,
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--cluster", default="4node_h200")
    parser.add_argument("--model", default="deepseek_v3_1024k")
    parser.add_argument("--dispatch-policy", default="least_batch")
    parser.add_argument("--routing-mode", default="explicit_rank_replay")
    parser.add_argument(
        "--strategy-max-num-seqs",
        action="append",
        default=[],
        metavar="STRATEGY=N",
        help=(
            "Override the vLLM wrapper's max-num-seqs for one strategy; repeat "
            "for multiple strategies. Omitted strategies retain their shared defaults."
        ),
    )
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--profile-delay-iterations", type=int, default=32)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--ignore-historical-skips", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        args.inputs = parse_named_inputs(args.input)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    try:
        args.strategy_max_num_seqs = parse_strategy_max_num_seqs(
            args.strategy_max_num_seqs
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    unused_overrides = set(args.strategy_max_num_seqs) - set(args.strategies)
    if unused_overrides:
        parser.error(
            "--strategy-max-num-seqs supplied for strategies not selected by "
            f"--strategies: {', '.join(sorted(unused_overrides))}"
        )
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.profile_delay_iterations < 0:
        parser.error("--profile-delay-iterations must be non-negative")
    if args.request_rate is not None and args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    if args.output_len is not None and args.output_len <= 0:
        parser.error("--output-len must be positive")
    return args


def case_command(
    args: argparse.Namespace,
    launcher: Path,
    strategy: str,
    dataset: str,
    input_path: Path,
) -> list[str]:
    command = [
        "zsh",
        str(launcher),
        "--artifact-root",
        str(args.artifact_root.expanduser().resolve()),
        "--cluster",
        args.cluster,
        "--strategy",
        strategy,
        "--model",
        args.model,
        "--lens-json",
        str(input_path),
        "--dispatch-policy",
        args.dispatch_policy,
        "--routing-mode",
        args.routing_mode,
        "--warmup-requests",
        str(args.warmup_requests),
        "--max-requests",
        "csv_rows",
        "--profile-delay-iterations",
        str(args.profile_delay_iterations),
        "--case-name",
        f"{dataset}_{strategy}",
    ]
    if strategy in args.strategy_max_num_seqs:
        command.extend(
            ("--max-num-seqs", str(args.strategy_max_num_seqs[strategy]))
        )
    command.append("--pause-before-profile")
    if args.request_rate is not None:
        command.extend(("--request-rate", str(args.request_rate)))
    if args.output_len is not None:
        command.extend(("--output-len", str(args.output_len)))
    if args.ignore_historical_skips:
        command.append("--ignore-historical-skips")
    return command


def main() -> None:
    args = parse_args()
    repo = args.vllm_root.expanduser().resolve()
    launcher = repo / "benchmarks" / "offline_dp_profile" / "start_multinode_offline_profile.sh"
    if not launcher.is_file():
        raise SystemExit(f"vLLM profile wrapper not found: {launcher}")
    if shutil.which("zsh") is None:
        raise SystemExit("zsh is required by the vLLM profile wrapper")

    artifact_root = args.artifact_root.expanduser().resolve()
    if not args.dry_run:
        artifact_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for strategy in args.strategies:
        for dataset, input_path in args.inputs:
            command = case_command(args, launcher, strategy, dataset, input_path)
            case = f"{dataset}/{strategy}"
            print(f"\n===== vLLM profile: {case} =====", flush=True)
            print(shell_command(command), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=repo, check=False)
            if result.returncode == 0:
                continue
            failures.append(f"{case} (exit code {result.returncode})")
            if not args.continue_on_error:
                break
        if failures and not args.continue_on_error:
            break

    if failures:
        raise SystemExit("vLLM profiling failures:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()

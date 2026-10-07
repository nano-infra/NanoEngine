#!/usr/bin/env python3
"""Measure the KIMI K2 MLA SP decode-pipeline latency grid."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

try:
    import measure_sp_pipeline as base
except ModuleNotFoundError:
    from algorithm import measure_sp_pipeline as base


ALGORITHM_DIR = Path(__file__).resolve().parent
SEARCH_SCRIPT = ALGORITHM_DIR / "search_sp_buckets.py"
BENCHMARK_SCRIPT = ALGORITHM_DIR / "test_sp_attention_cudagraph.py"
SP_SIZES = base.SP_SIZES
KIMI_NUM_HEADS = 64
KIMI_HEAD_DIM = 576
KIMI_NUM_KV_HEADS = 1
KIMI_FLASHMLA_V_HEAD_DIM = 512
KIMI_DEFAULT_MAX_SEQ_LEN = 1_000_000
DEFAULT_SAMPLES = 10
DEFAULT_WARMUP = base.DEFAULT_WARMUP
DEFAULT_ITERATIONS = base.DEFAULT_ITERATIONS
SP_BACKEND = "hao_basic"
RAW_FIELDNAMES = base.RAW_FIELDNAMES
SUMMARY_FIELDNAMES = base.SUMMARY_FIELDNAMES

if base.SP_BACKEND != SP_BACKEND:
    raise RuntimeError(
        f"KIMI K2 measurement requires SP backend {SP_BACKEND!r}; "
        f"base measurement backend is {base.SP_BACKEND!r}"
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_seq_lens(max_seq_len: int) -> tuple[int, ...]:
    if max_seq_len < 1024 or max_seq_len % 64 != 0:
        raise ValueError("--max-seq-len must be at least 1024 and divisible by 64")

    values = [1024, 2048]
    values.extend(range(4096, max_seq_len + 1, 4096))
    if values[-1] != max_seq_len:
        values.append(max_seq_len)
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output directory (default: a timestamped directory).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --output-dir and skip recorded samples.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=positive_int,
        default=KIMI_DEFAULT_MAX_SEQ_LEN,
        help=(
            "Maximum sequence length for the default grid "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seq-len",
        action="extend",
        nargs="+",
        type=positive_int,
        default=[],
        metavar="TOKENS",
        help=(
            "Sequence lengths to measure. Defaults to 1024, 2048, every "
            "4096 tokens, and --max-seq-len."
        ),
    )
    parser.add_argument(
        "--sp-size",
        action="extend",
        nargs="+",
        type=positive_int,
        choices=SP_SIZES,
        default=[],
        help="SP sizes to measure. Defaults to 1 through 8.",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=DEFAULT_SAMPLES,
        help="Samples collected at every sequence-length/SP setting.",
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=DEFAULT_WARMUP,
        help="Warmup replays per sample (default: %(default)s).",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=DEFAULT_ITERATIONS,
        help="Measured CUDA Graph replays per sample (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the planned GPU work without launching it.",
    )
    args = parser.parse_args()

    args.seq_len = tuple(dict.fromkeys(args.seq_len)) or default_seq_lens(
        args.max_seq_len
    )
    args.sp_size = tuple(dict.fromkeys(args.sp_size)) or SP_SIZES
    if any(seq_len % 64 != 0 for seq_len in args.seq_len):
        parser.error("every --seq-len must be divisible by 64")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def capture_master_latency(
    benchmark: Callable[..., object],
    *,
    seq_len: int,
    sp_size: int,
    warmup: int,
    iterations: int,
) -> float:
    latencies: list[float] = []

    def profile(frame: object, event: str, _arg: object) -> None:
        if event != "return" or getattr(frame, "f_code", None) is not benchmark.__code__:
            return
        value = frame.f_locals.get("avg_time_us")
        if isinstance(value, (int, float)):
            latencies.append(float(value))

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    previous_profile = sys.getprofile()
    try:
        sys.setprofile(profile)
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            benchmark(
                seq_len=seq_len,
                num_heads=KIMI_NUM_HEADS,
                head_dim=KIMI_HEAD_DIM,
                num_kv_heads=KIMI_NUM_KV_HEADS,
                attention_sp=sp_size,
                attention_type="MLA",
                v_head_dim=KIMI_FLASHMLA_V_HEAD_DIM,
                use_cudagraph=True,
                num_warmup=warmup,
                num_iterations=iterations,
                debug=False,
                enable_profiler=False,
            )
    except BaseException:
        sys.stdout.write(captured_stdout.getvalue())
        sys.stderr.write(captured_stderr.getvalue())
        raise
    finally:
        sys.setprofile(previous_profile)

    if len(latencies) != 1:
        raise RuntimeError(
            f"expected one latency for seq_len={seq_len}, SP={sp_size}; "
            f"captured {latencies}"
        )
    return latencies[0]


def worker_main(values: list[str]) -> int:
    args = base.parse_worker_args(values)
    seq_lens = tuple(int(value) for value in args.seq_lens.split(","))
    sample_counts = tuple(int(value) for value in args.sample_counts.split(","))
    if len(seq_lens) != len(sample_counts):
        raise ValueError("worker sequence-length and sample-count lists differ")

    rank = int(os.environ.get("RANK", "0"))
    completed = base.load_completed_samples(args.raw_output, args.sp_size)
    module = base.load_benchmark(args.benchmark_script)
    benchmark = getattr(module, "benchmark_sp_attention_with_cudagraph")

    output_handle = None
    writer = None
    if rank == 0:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.raw_output.exists()
        output_handle = args.raw_output.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(output_handle, fieldnames=RAW_FIELDNAMES)
        if new_file:
            writer.writeheader()
            output_handle.flush()

    try:
        for seq_len, sample_count in zip(seq_lens, sample_counts, strict=True):
            for sample_index in range(sample_count):
                if (seq_len, sample_index) in completed:
                    continue
                latency = capture_master_latency(
                    benchmark,
                    seq_len=seq_len,
                    sp_size=args.sp_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                if rank == 0:
                    assert writer is not None and output_handle is not None
                    writer.writerow(
                        {
                            "seq_len": seq_len,
                            "cp_size": args.sp_size,
                            "sample_index": sample_index,
                            "end_to_end_master_us": repr(latency),
                        }
                    )
                    output_handle.flush()
                    print(
                        f"SP={args.sp_size} seq_len={seq_len} "
                        f"sample={sample_index + 1}/{sample_count} "
                        f"latency={latency:.3f} us",
                        flush=True,
                    )
    finally:
        if output_handle is not None:
            output_handle.close()
        distributed = getattr(module, "dist")
        if distributed.is_initialized():
            distributed.barrier()
            distributed.destroy_process_group()
    return 0


def worker_command(
    *,
    benchmark_script: Path,
    raw_output: Path,
    sp_size: int,
    seq_lens: tuple[int, ...],
    sample_counts: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts=0",
        f"--nproc-per-node={sp_size}",
        "--local-ranks-filter=0",
        str(Path(__file__).resolve()),
        "_worker",
        "--benchmark-script",
        str(benchmark_script),
        "--raw-output",
        str(raw_output),
        "--sp-size",
        str(sp_size),
        "--seq-lens",
        ",".join(str(value) for value in seq_lens),
        "--sample-counts",
        ",".join(str(value) for value in sample_counts),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]


def write_summary(
    output_dir: Path,
    seq_lens: tuple[int, ...],
    sp_sizes: tuple[int, ...],
    sample_count: int,
) -> Path:
    expected = {
        (seq_len, sp_size): sample_count
        for seq_len in seq_lens
        for sp_size in sp_sizes
    }
    samples: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for sp_size in sp_sizes:
        raw_path = output_dir / "raw" / f"sp{sp_size}.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing raw samples: {raw_path}")
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["seq_len"]), int(row["cp_size"]))
                if key not in expected:
                    continue
                sample_index = int(row["sample_index"])
                if sample_index in samples[key]:
                    raise ValueError(f"duplicate sample index {sample_index} for {key}")
                samples[key][sample_index] = float(row["end_to_end_master_us"])

    summary_path = output_dir / "kimi_k2_h200_pipeline_p50.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for seq_len in sorted(seq_lens):
            for sp_size in sorted(sp_sizes):
                key = (seq_len, sp_size)
                indices = sorted(samples[key])
                if indices != list(range(sample_count)):
                    raise ValueError(
                        f"incomplete samples for seq_len={seq_len}, SP={sp_size}: "
                        f"expected {sample_count}, found {len(indices)}"
                    )
                values = [samples[key][index] for index in indices]
                writer.writerow(
                    {
                        "seq_len": seq_len,
                        "cp_size": sp_size,
                        "end_to_end_master_p50_us": repr(statistics.median(values)),
                        "sample_count": sample_count,
                    }
                )
    temporary.replace(summary_path)
    return summary_path


def validate_resume_backend(output_dir: Path) -> None:
    for log_path in sorted((output_dir / "logs").glob("sp*.log")):
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "backend=legacy_ll" in text:
            raise ValueError(
                f"{log_path} contains legacy_ll measurements; "
                "KIMI K2 measurement requires hao_basic. "
                "Use a fresh --output-dir instead of --resume."
            )


def main() -> int:
    args = parse_args()
    workdir = ALGORITHM_DIR.parent
    benchmark_script = BENCHMARK_SCRIPT
    if not benchmark_script.is_file():
        raise FileNotFoundError(f"Local benchmark not found: {benchmark_script}")

    seq_lens = tuple(args.seq_len)
    sp_sizes = tuple(args.sp_size)
    sample_count = args.samples
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (ALGORITHM_DIR / "results" / f"kimi_k2_sp_pipeline_{timestamp}").resolve()
    )
    total_samples = len(seq_lens) * len(sp_sizes) * sample_count

    print(f"Workspace root: {workdir}")
    print(f"Benchmark script: {benchmark_script}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence lengths: {len(seq_lens)}")
    print(f"SP sizes: {' '.join(str(value) for value in sp_sizes)}")
    print(f"Model: KIMI K2")
    print(f"Attention heads: {KIMI_NUM_HEADS}")
    print(f"MLA Q/K head dim: {KIMI_HEAD_DIM}")
    print(f"MLA FlashMLA output head dim: {KIMI_FLASHMLA_V_HEAD_DIM}")
    print(f"SP backend: {SP_BACKEND}")
    print(f"Benchmark samples: {total_samples}")
    print(f"Warmup/measured replays per sample: {args.warmup}/{args.iterations}")

    sample_counts = tuple(sample_count for _ in seq_lens)
    commands = [
        worker_command(
            benchmark_script=benchmark_script,
            raw_output=output_dir / "raw" / f"sp{sp_size}.csv",
            sp_size=sp_size,
            seq_lens=seq_lens,
            sample_counts=sample_counts,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for sp_size in sp_sizes
    ]
    if args.dry_run:
        for sp_size, command in zip(sp_sizes, commands, strict=True):
            print(f"SP={sp_size}: {' '.join(command)}")
        return 0

    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"output directory already exists; choose a fresh path: {output_dir}"
        )
    if args.resume and not output_dir.is_dir():
        raise FileNotFoundError(f"resume output directory not found: {output_dir}")
    if args.resume:
        validate_resume_backend(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    config = {
        "model": "KIMI-K2",
        "attention_type": "MLA",
        "num_heads": KIMI_NUM_HEADS,
        "head_dim": KIMI_HEAD_DIM,
        "num_kv_heads": KIMI_NUM_KV_HEADS,
        "flash_mla_v_head_dim": KIMI_FLASHMLA_V_HEAD_DIM,
        "workspace_root": str(workdir),
        "benchmark_script": str(benchmark_script),
        "seq_lens": list(seq_lens),
        "sp_sizes": list(sp_sizes),
        "sample_count": sample_count,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "sp_backend": SP_BACKEND,
        "metric": "end_to_end_master_p50_us",
    }
    base.write_or_validate_config(output_dir / "config.json", config, args.resume)

    environment = base.benchmark_environment(workdir)
    gpu_names = base.probe_gpus(workdir, environment, max(sp_sizes))
    print(f"Visible GPUs: {len(gpu_names)} ({', '.join(gpu_names)})", flush=True)
    for sp_size, command in zip(sp_sizes, commands, strict=True):
        print(f"\nRunning SP={sp_size}...", flush=True)
        base.run_logged(
            command,
            workdir=workdir,
            environment=environment,
            log_path=output_dir / "logs" / f"sp{sp_size}.log",
            append=args.resume,
        )

    summary_path = write_summary(output_dir, seq_lens, sp_sizes, sample_count)
    print(f"Fresh latency grid: {summary_path}")

    if set(sp_sizes) == set(SP_SIZES) and len(seq_lens) >= 3:
        policy_path = output_dir / "kimi_k2_policy.json"
        subprocess.run(
            [
                sys.executable,
                str(SEARCH_SCRIPT),
                "--input",
                str(summary_path),
                "--output",
                str(policy_path),
            ],
            cwd=ALGORITHM_DIR.parent,
            check=True,
        )
    else:
        print("Policy search skipped: it requires SP sizes 1 through 8 and >=3 lengths.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        raise SystemExit(worker_main(sys.argv[2:]))
    raise SystemExit(main())

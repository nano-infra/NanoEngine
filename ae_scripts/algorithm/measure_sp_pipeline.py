#!/usr/bin/env python3
"""Optionally remeasure the DeepSeek-V3 SP decode-pipeline latency grid."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import importlib.util
import inspect
import io
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable


ALGORITHM_DIR = Path(__file__).resolve().parent
SEARCH_SCRIPT = ALGORITHM_DIR / "search_sp_buckets.py"
DEFAULT_NANODEPLOY_WORKDIR = os.environ.get("NANODEPLOY_WORKDIR")
BENCHMARK_RELATIVE_PATH = Path("tests/test_sp_attention_cudagraph.py")
SP_SIZES = tuple(range(1, 9))
SEQ_LENS = (1024, 2048, *range(4096, 1_048_576 + 1, 4096))
BOUNDARY_LOW = 98_304
BOUNDARY_HIGH = 196_608
DEFAULT_BASE_SAMPLES = 10
DEFAULT_BOUNDARY_SAMPLES = 70
DEFAULT_WARMUP = 100
DEFAULT_ITERATIONS = 200
SP_BACKEND = "hao_basic"
RAW_FIELDNAMES = (
    "seq_len",
    "cp_size",
    "sample_index",
    "end_to_end_master_us",
)
SUMMARY_FIELDNAMES = (
    "seq_len",
    "cp_size",
    "end_to_end_master_p50_us",
    "sample_count",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nanodeploy-workdir",
        type=Path,
        default=(
            Path(DEFAULT_NANODEPLOY_WORKDIR)
            if DEFAULT_NANODEPLOY_WORKDIR
            else None
        ),
        help=(
            "NanoDeploy checkout containing tests/test_sp_attention_cudagraph.py. "
            "Defaults to NANODEPLOY_WORKDIR, then the installed nanodeploy "
            "package's checkout."
        ),
    )
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
        "--seq-len",
        action="extend",
        nargs="+",
        type=positive_int,
        default=[],
        metavar="TOKENS",
        help="Sequence lengths to measure. Defaults to the complete 258-point grid.",
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
        help="Use this sample count at every point instead of the paper schedule.",
    )
    parser.add_argument(
        "--base-samples",
        type=positive_int,
        default=DEFAULT_BASE_SAMPLES,
        help="Samples outside the transition region (default: %(default)s).",
    )
    parser.add_argument(
        "--boundary-samples",
        type=positive_int,
        default=DEFAULT_BOUNDARY_SAMPLES,
        help="Samples from 98,304 through 196,608 tokens (default: %(default)s).",
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

    args.seq_len = tuple(dict.fromkeys(args.seq_len)) or SEQ_LENS
    args.sp_size = tuple(dict.fromkeys(args.sp_size)) or SP_SIZES
    if any(seq_len % 64 != 0 for seq_len in args.seq_len):
        parser.error("every --seq-len must be divisible by 64")
    if args.samples is None and args.boundary_samples < args.base_samples:
        parser.error("--boundary-samples must be at least --base-samples")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def resolve_nanodeploy_workdir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    spec = importlib.util.find_spec("nanodeploy")
    if spec is not None and spec.origin is not None:
        checkout = Path(spec.origin).resolve().parent.parent
        if (checkout / BENCHMARK_RELATIVE_PATH).is_file():
            return checkout

    raise FileNotFoundError(
        "could not locate tests/test_sp_attention_cudagraph.py from the "
        "installed nanodeploy package; set NANODEPLOY_WORKDIR or pass "
        "--nanodeploy-workdir"
    )


def parse_worker_args(values: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--sp-size", type=positive_int, required=True)
    parser.add_argument("--seq-lens", required=True)
    parser.add_argument("--sample-counts", required=True)
    parser.add_argument("--warmup", type=positive_int, required=True)
    parser.add_argument("--iterations", type=positive_int, required=True)
    return parser.parse_args(values)


def sample_count_for(args: argparse.Namespace, seq_len: int) -> int:
    if args.samples is not None:
        return args.samples
    if BOUNDARY_LOW <= seq_len <= BOUNDARY_HIGH:
        return args.boundary_samples
    return args.base_samples


def benchmark_environment(workdir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    path_parts = [str(workdir)]
    if python_path:
        path_parts.append(python_path)
    environment["PYTHONPATH"] = os.pathsep.join(path_parts)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def load_benchmark(path: Path) -> object:
    checkout = path.resolve().parents[1]
    sys.path.insert(0, str(checkout))
    spec = importlib.util.spec_from_file_location("algorithm_sp_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load NanoDeploy benchmark: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Keep the communication setting identical across every measured point.
    # The benchmark helper otherwise inherits NanoDeploy's default backend.
    original_set_sp_context = getattr(module, "set_sp_context")

    if "backend" in inspect.signature(original_set_sp_context).parameters:

        def set_sp_context_with_fixed_backend(**kwargs: object) -> object:
            kwargs["backend"] = SP_BACKEND
            return original_set_sp_context(**kwargs)

        setattr(module, "set_sp_context", set_sp_context_with_fixed_backend)
    return module


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
                num_heads=128,
                head_dim=576,
                num_kv_heads=1,
                attention_sp=sp_size,
                attention_type="MLA",
                v_head_dim=512,
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


def load_completed_samples(path: Path, sp_size: int) -> set[tuple[int, int]]:
    if not path.is_file():
        return set()
    completed: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
            raise ValueError(f"unexpected raw CSV header: {path}")
        for row in reader:
            row_sp = int(row["cp_size"])
            if row_sp != sp_size:
                raise ValueError(f"{path} contains cp_size={row_sp}, expected {sp_size}")
            key = (int(row["seq_len"]), int(row["sample_index"]))
            if key in completed:
                raise ValueError(f"duplicate raw sample {key} in {path}")
            float(row["end_to_end_master_us"])
            completed.add(key)
    return completed


def worker_main(values: list[str]) -> int:
    args = parse_worker_args(values)
    seq_lens = tuple(int(value) for value in args.seq_lens.split(","))
    sample_counts = tuple(int(value) for value in args.sample_counts.split(","))
    if len(seq_lens) != len(sample_counts):
        raise ValueError("worker sequence-length and sample-count lists differ")

    rank = int(os.environ.get("RANK", "0"))
    completed = load_completed_samples(args.raw_output, args.sp_size)
    module = load_benchmark(args.benchmark_script)
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


def probe_gpus(workdir: Path, environment: dict[str, str], required: int) -> list[str]:
    probe = (
        "import json, torch; "
        "print(json.dumps([torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=workdir,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    names = json.loads(lines[-1]) if lines else []
    if len(names) < required:
        raise RuntimeError(f"requires {required} visible GPUs, found {len(names)}: {names}")
    return names


def run_logged(
    command: list[str],
    *,
    workdir: Path,
    environment: dict[str, str],
    log_path: Path,
    append: bool,
) -> None:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def write_summary(
    output_dir: Path,
    seq_lens: tuple[int, ...],
    sp_sizes: tuple[int, ...],
    sample_counts: tuple[int, ...],
) -> Path:
    expected = {
        (seq_len, sp_size): sample_count
        for seq_len, sample_count in zip(seq_lens, sample_counts, strict=True)
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

    summary_path = output_dir / "deepseek_v3_h200_pipeline_p50.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for seq_len in sorted(seq_lens):
            for sp_size in sorted(sp_sizes):
                key = (seq_len, sp_size)
                expected_count = expected[key]
                indices = sorted(samples[key])
                if indices != list(range(expected_count)):
                    raise ValueError(
                        f"incomplete samples for seq_len={seq_len}, SP={sp_size}: "
                        f"expected {expected_count}, found {len(indices)}"
                    )
                values = [samples[key][index] for index in indices]
                writer.writerow(
                    {
                        "seq_len": seq_len,
                        "cp_size": sp_size,
                        "end_to_end_master_p50_us": repr(statistics.median(values)),
                        "sample_count": expected_count,
                    }
                )
    temporary.replace(summary_path)
    return summary_path


def write_or_validate_config(
    path: Path, config: dict[str, object], resume: bool
) -> None:
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"resume configuration not found: {path}")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("resume configuration does not match config.json")
        return
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = resolve_nanodeploy_workdir(args.nanodeploy_workdir)
    benchmark_script = workdir / BENCHMARK_RELATIVE_PATH
    if not benchmark_script.is_file():
        raise FileNotFoundError(f"NanoDeploy benchmark not found: {benchmark_script}")

    seq_lens = tuple(args.seq_len)
    sp_sizes = tuple(args.sp_size)
    sample_counts = tuple(sample_count_for(args, value) for value in seq_lens)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (ALGORITHM_DIR / "results" / f"sp_pipeline_{timestamp}").resolve()
    )
    total_samples = sum(sample_counts) * len(sp_sizes)

    print(f"NanoDeploy workdir: {workdir}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence lengths: {len(seq_lens)}")
    print(f"SP sizes: {' '.join(str(value) for value in sp_sizes)}")
    print(f"SP backend: {SP_BACKEND}")
    print(f"Benchmark samples: {total_samples}")
    print(f"Warmup/measured replays per sample: {args.warmup}/{args.iterations}")

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
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    config = {
        "nanodeploy_workdir": str(workdir),
        "benchmark_script": str(benchmark_script),
        "seq_lens": list(seq_lens),
        "sp_sizes": list(sp_sizes),
        "sample_counts": list(sample_counts),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "sp_backend": SP_BACKEND,
        "metric": "end_to_end_master_p50_us",
    }
    write_or_validate_config(output_dir / "config.json", config, args.resume)

    environment = benchmark_environment(workdir)
    gpu_names = probe_gpus(workdir, environment, max(sp_sizes))
    print(f"Visible GPUs: {len(gpu_names)} ({', '.join(gpu_names)})", flush=True)
    for sp_size, command in zip(sp_sizes, commands, strict=True):
        print(f"\nRunning SP={sp_size}...", flush=True)
        run_logged(
            command,
            workdir=workdir,
            environment=environment,
            log_path=output_dir / "logs" / f"sp{sp_size}.log",
            append=args.resume,
        )

    summary_path = write_summary(output_dir, seq_lens, sp_sizes, sample_counts)
    print(f"Fresh latency grid: {summary_path}")

    if set(sp_sizes) == set(SP_SIZES) and len(seq_lens) >= 3:
        policy_path = output_dir / "reproduced_policy.json"
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

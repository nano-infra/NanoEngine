#!/usr/bin/env python3
"""Reproduce Fig. 5 with standalone FlashMLA and vLLM/DeepEP logs.

The script benchmarks the Attention inputs extracted from a new E2E run, maps
new DeepEP token logs back to the E2E ranks, parses the HoL frontend log, and
renders the three-panel figure. No vLLM Python package or C++ extension is
imported.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence


FIG5_DIR = Path(__file__).resolve().parent
MICROBENCH_DIR = FIG5_DIR.parent / "microbench"
DEFAULT_EXTERNAL_BENCHMARK = (
    MICROBENCH_DIR / "attention/benchmark_flashmla.py"
)
DEFAULT_PLOT_SCRIPT = FIG5_DIR / "plot_fig5.py"

TIME_PERCENT = 60
MAX_TOKENS_PER_GPU = 1_126_976
PAGE_SIZE = 64
NUM_HEADS = 128
HEAD_DIM = 576
V_HEAD_DIM = 512

SNAPSHOT_RE_TEMPLATE = (
    r"\[time={time_pct}% step=(\d+)/(\d+)\]\r?\n"
    r".*?rank\trunning\tgpu_kv_usage_pct\r?\n"
    r"(.*?)(?:\r?\n\r?\n|\Z)"
)
DEEPEP_FRESH_TIMING_RE = re.compile(
    r"\[rank\s+(?P<rank>\d+)(?:\s*\|\s*num_tokens\s+(?P<tokens>\d+))?\]"
    r"\s*Dispatch\s*\+\s*combine bandwidth:.*?"
    r"avg_t=(?P<avg_us>[\d.]+)\s*us"
)


def parse_args() -> argparse.Namespace:
    artifact_dir = FIG5_DIR
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Fig. 5 Attention, map DeepEP measurements to E2E "
            "ranks, parse the HoL log, and render the figure."
        )
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU ID.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python used for the benchmark and plotting subprocesses.",
    )
    parser.add_argument(
        "--external-benchmark-script",
        type=Path,
        default=DEFAULT_EXTERNAL_BENCHMARK,
        help=(
            "Attention measurement engine; defaults to the shared "
            "microbench/attention/benchmark_flashmla.py."
        ),
    )
    parser.add_argument(
        "--plot-script",
        type=Path,
        default=DEFAULT_PLOT_SCRIPT,
        help="Plot implementation; defaults to the local fig5/plot_fig5.py.",
    )
    parser.add_argument(
        "--attention-snapshot",
        type=Path,
        required=True,
        help=(
            "Either the legacy frontend_rank_quartiles_data.txt or the JSON "
            "written by fig5/extract_vllm_rank_snapshot.py."
        ),
    )
    parser.add_argument(
        "--attention-csv",
        type=Path,
        help=(
            "Skip the GPU benchmark and use an existing attention CSV. Both "
            "rank/token and external seq_len/batch_size schemas are accepted."
        ),
    )
    parser.add_argument(
        "--deepep-snapshot", type=Path, required=True
    )
    parser.add_argument(
        "--deepep-data-dir",
        type=Path,
        required=True,
        help=(
            "Fresh result directory produced on all nodes by "
            "fig5/deepep/run_deepep.sh."
        ),
    )
    parser.add_argument(
        "--deepep-local-ranks",
        type=int,
        default=8,
        help="Local ranks whose node-0 mean forms the latency lookup.",
    )
    parser.add_argument(
        "--hol-input",
        "--hol-log",
        dest="hol_log",
        type=Path,
        required=True,
        help="Raw HoL frontend.log or prepared HoL CSV.",
    )
    parser.add_argument("--hol-engines", default="8-15")
    parser.add_argument("--hol-block-size", type=float, default=64.0)
    parser.add_argument(
        "--hol-middle-window-seconds", type=float, default=800.0
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--bench-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=artifact_dir / "results" / f"run_{run_id}_pid{os.getpid()}",
        help="Fresh per-run CSV, log, and manifest directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "fig5",
        help="Figure base path or an explicit .pdf/.png path.",
    )
    args = parser.parse_args()

    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    for name in ("rep_ms", "bench_repeats", "warmup", "dpi", "year"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.hol_block_size <= 0:
        parser.error("--hol-block-size must be positive")
    if args.hol_middle_window_seconds < 0:
        parser.error("--hol-middle-window-seconds must be non-negative")
    if args.deepep_local_ranks <= 0:
        parser.error("--deepep-local-ranks must be positive")
    return args


def resolve_python(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or os.sep in value:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise FileNotFoundError(f"Python executable not found: {candidate}")
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Python executable not found on PATH: {value}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def prepare_result_root(path: Path) -> Path:
    result_root = path.expanduser().resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty result directory: {result_root}"
        )
    result_root.mkdir(parents=True, exist_ok=True)
    return result_root


def output_base(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() in {".pdf", ".png"}:
        return resolved.with_suffix("")
    return resolved


def log_message(log_path: Path, message: str = "") -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    rendered = shlex.join(str(item) for item in command)
    log_message(log_path, f"\n+ {rendered}")
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_snapshot(
    path: Path,
    time_pct: int,
) -> tuple[int, int, list[tuple[int, float, float]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        SNAPSHOT_RE_TEMPLATE.format(time_pct=time_pct), re.DOTALL
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"time={time_pct}% snapshot not found in {path}")
    rows: list[tuple[int, float, float]] = []
    for line in match.group(3).strip().splitlines():
        fields = line.strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"Malformed rank row in {path}: {line!r}")
        rows.append((int(fields[0]), float(fields[1]), float(fields[2])))
    rows.sort(key=lambda row: row[0])
    if not rows or [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError(
            f"Expected contiguous ranks starting at 0 in {path}, "
            f"found {[row[0] for row in rows]}"
        )
    return int(match.group(1)), int(match.group(2)), rows


def round_to_multiple(value: float, multiple: int) -> int:
    return int(((value + multiple / 2) // multiple) * multiple)


def derive_attention_tokens(
    snapshot: Path,
) -> tuple[list[tuple[int, int]], int, list[int], int, int, int]:
    declared_cases: list[int] | None = None
    if snapshot.suffix.lower() == ".json":
        metadata = json.loads(snapshot.read_text(encoding="utf-8"))
        time_percent = int(metadata["time_percent"])
        if time_percent != TIME_PERCENT:
            raise ValueError(
                f"Expected time={TIME_PERCENT}%, found time={time_percent}%"
            )
        tokens = [int(value) for value in metadata["attention"]["rank_tokens"]]
        rank_count = int(metadata["rank_count"])
        if rank_count <= 0 or len(tokens) != rank_count:
            raise ValueError(
                f"Expected {rank_count} Attention ranks, found {len(tokens)}"
            )
        rank_tokens = list(enumerate(tokens))
        step = int(metadata["selected_step_one_based"])
        total_steps = int(metadata["complete_snapshot_count"])
        kv_capacity_tokens = int(metadata["kv_capacity_tokens"])
        declared_cases = [
            int(value)
            for value in metadata["attention"]["microbenchmark_token_cases"]
        ]
    else:
        step, total_steps, rows = parse_snapshot(snapshot, TIME_PERCENT)
        kv_capacity_tokens = MAX_TOKENS_PER_GPU
        rank_tokens = []
        for rank, _running, usage_pct in rows:
            token = round_to_multiple(
                usage_pct / 100.0 * kv_capacity_tokens,
                PAGE_SIZE,
            )
            rank_tokens.append((rank, token))
    mean_token = round_to_multiple(
        statistics.mean(token for _rank, token in rank_tokens), PAGE_SIZE
    )
    expected_cases = sorted(
        {token for _rank, token in rank_tokens} | {mean_token}
    )
    if declared_cases is not None and declared_cases != expected_cases:
        raise ValueError("Attention microbenchmark cases do not match the snapshot")
    unique_tokens = declared_cases or expected_cases
    return (
        rank_tokens,
        mean_token,
        unique_tokens,
        step,
        total_steps,
        kv_capacity_tokens,
    )


def derive_deepep_batches(
    snapshot: Path,
) -> tuple[list[tuple[int, int]], float, list[int], int, int]:
    declared_cases: list[int] | None = None
    declared_rank_count: int | None = None
    if snapshot.suffix.lower() == ".json":
        metadata = json.loads(snapshot.read_text(encoding="utf-8"))
        time_percent = int(metadata["time_percent"])
        if time_percent != TIME_PERCENT:
            raise ValueError(
                f"Expected time={TIME_PERCENT}%, found time={time_percent}%"
            )
        batch_sizes = [
            int(value) for value in metadata["deepep"]["rank_batch_sizes"]
        ]
        declared_rank_count = int(metadata["rank_count"])
        step = int(metadata["selected_step_one_based"])
        total_steps = int(metadata["complete_snapshot_count"])
        declared_cases = [
            int(value)
            for value in metadata["deepep"]["microbenchmark_batch_cases"]
        ]
    else:
        step, total_steps, rows = parse_snapshot(snapshot, TIME_PERCENT)
        batch_sizes = [int(running) for _rank, running, _usage in rows]
    if declared_rank_count is not None and (
        declared_rank_count <= 0 or len(batch_sizes) != declared_rank_count
    ):
        raise ValueError(
            f"Expected {declared_rank_count} DeepEP ranks, "
            f"found {len(batch_sizes)}"
        )
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("DeepEP batch sizes must all be positive")
    mean_batch = statistics.mean(batch_sizes)
    expected_cases = sorted(set(batch_sizes) | {math.ceil(mean_batch)})
    if declared_cases is not None and declared_cases != expected_cases:
        raise ValueError("DeepEP microbenchmark cases do not match the snapshot")
    cases = declared_cases or expected_cases
    return list(enumerate(batch_sizes)), mean_batch, cases, step, total_steps


def parse_fresh_deepep_log(
    path: Path, token: int, local_ranks: int
) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"DeepEP token log not found: {path}")
    by_rank: dict[int, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    exit_codes = re.findall(r"^DEEPEP_SWEEP_EXIT_CODE=(\d+)\s*$", text, re.MULTILINE)
    if not exit_codes or int(exit_codes[-1]) != 0:
        raise ValueError(f"{path.name} is not a completed DeepEP token run")
    for match in DEEPEP_FRESH_TIMING_RE.finditer(text):
        rank = int(match.group("rank"))
        logged_token = match.group("tokens")
        if logged_token is not None and int(logged_token) != token:
            continue
        if 0 <= rank < local_ranks:
            if rank in by_rank:
                raise ValueError(f"{path.name} repeats rank {rank}")
            by_rank[rank] = float(match.group("avg_us"))
    expected = set(range(local_ranks))
    if set(by_rank) != expected:
        raise ValueError(
            f"{path.name} does not cover node-0 ranks 0..{local_ranks - 1}: "
            f"found {sorted(by_rank)}"
        )
    return [by_rank[rank] for rank in range(local_ranks)]


def prepare_fresh_deepep_csv(
    snapshot: Path,
    data_dir: Path,
    local_ranks: int,
    output: Path,
) -> tuple[Path, dict[str, object]]:
    rank_batches, mean_batch, cases, step, total_steps = derive_deepep_batches(
        snapshot
    )
    resolved_data_dir = data_dir.expanduser().resolve()
    logs_dir = resolved_data_dir / "logs"
    config_path = resolved_data_dir / "fig5_deepep_config.json"
    num_nodes: int | None = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured_snapshot_cases = [
            int(value) for value in config["snapshot_cases"]
        ]
        measured_cases = [int(value) for value in config["measured_cases"]]
        if configured_snapshot_cases != cases:
            raise ValueError(
                "DeepEP run configuration does not match the selected snapshot"
            )
        if int(config["local_processes"]) != local_ranks:
            raise ValueError(
                "DeepEP run local-process count does not match "
                "--deepep-local-ranks"
            )
        num_nodes = int(config["num_nodes"])
        if num_nodes * local_ranks != len(rank_batches):
            raise ValueError(
                "DeepEP run topology does not match the E2E snapshot: "
                f"{num_nodes} nodes * {local_ranks} local ranks != "
                f"{len(rank_batches)} snapshot ranks"
            )
    else:
        measured_cases = cases
    if (
        not measured_cases
        or measured_cases != sorted(measured_cases)
        or len(measured_cases) != len(set(measured_cases))
        or any(value not in cases for value in measured_cases)
    ):
        raise ValueError("DeepEP run configuration contains invalid measured cases")
    if measured_cases[0] != cases[0] or measured_cases[-1] != cases[-1]:
        raise ValueError(
            "Reduced DeepEP cases must include the smallest and largest "
            "snapshot cases"
        )
    lookup: dict[int, float] = {}
    source_logs: list[str] = []
    for token in measured_cases:
        log_path = logs_dir / f"node0_tokens_{token}.log"
        values = parse_fresh_deepep_log(log_path, token, local_ranks)
        lookup[token] = statistics.mean(values)
        source_logs.append(str(log_path))
    mean_latency = interpolate_lookup(lookup, mean_batch)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["rank", "batch_size", "time_us", "interpolated"],
        )
        writer.writeheader()
        for rank, batch_size in rank_batches:
            writer.writerow(
                {
                    "rank": rank,
                    "batch_size": batch_size,
                    "time_us": interpolate_lookup(lookup, batch_size),
                    "interpolated": batch_size not in lookup,
                }
            )
        writer.writerow(
            {
                "rank": "avg",
                "batch_size": mean_batch,
                "time_us": mean_latency,
                "interpolated": mean_batch not in lookup,
            }
        )
    interpolated_rank_count = sum(
        batch_size not in lookup for _rank, batch_size in rank_batches
    )
    validation = {
        "mode": "full" if measured_cases == cases else "interpolated",
        "time_percent": TIME_PERCENT,
        "step": step,
        "total_steps": total_steps,
        "ranks": len(rank_batches),
        "num_nodes": num_nodes,
        "local_ranks_averaged": local_ranks,
        "snapshot_cases": cases,
        "measured_cases": measured_cases,
        "interpolated_rank_count": interpolated_rank_count,
        "mean_batch_size": mean_batch,
        "mean_latency_us": mean_latency,
        "config": str(config_path) if config_path.is_file() else None,
        "source_logs": source_logs,
    }
    return output, validation


def interpolate_lookup(lookup: dict[int, float], value: float) -> float:
    if value in lookup:
        return float(lookup[int(value)])
    lower = max(key for key in lookup if key < value)
    upper = min(key for key in lookup if key > value)
    fraction = (value - lower) / (upper - lower)
    return float(lookup[lower]) + fraction * (
        float(lookup[upper]) - float(lookup[lower])
    )


def flashmla_environment(
    python: str,
    gpu: int,
) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SLIME_QP_NUM"] = "4"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig5")
    code = r"""
import importlib.metadata
import json
from pathlib import Path
import flash_mla
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
torch.cuda.set_device(0)
for name in ("get_mla_metadata", "flash_mla_with_kvcache"):
    if not callable(getattr(flash_mla, name, None)):
        raise SystemExit(f"missing flash_mla callable: {name}")
try:
    version = importlib.metadata.version("flash_mla")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"
print(json.dumps({
    "python_torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "flash_mla_version": version,
    "flash_mla_module": str(Path(flash_mla.__file__).resolve()),
}))
"""
    completed = subprocess.run(
        [python, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return env, completed.stdout.strip()


def run_external_benchmark(
    args: argparse.Namespace,
    python: str,
    benchmark_script: Path,
    unique_tokens: Sequence[int],
    result_root: Path,
    log_path: Path,
) -> tuple[Path, str]:
    env, preflight = flashmla_environment(python, args.gpu)
    log_message(log_path, "External FlashMLA preflight: " + preflight)
    output_csv = result_root / "attention_external_flashmla_unique.csv"
    command = [
        python,
        str(benchmark_script),
        "--seq_lens",
        ",".join(str(token) for token in unique_tokens),
        "--batch_sizes",
        "1",
        "--num_heads",
        str(NUM_HEADS),
        "--head_dim",
        str(HEAD_DIM),
        "--v_head_dim",
        str(V_HEAD_DIM),
        "--rep-ms",
        str(args.rep_ms),
        "--bench-repeats",
        str(args.bench_repeats),
        "--warmup",
        str(args.warmup),
        "--output",
        str(output_csv),
        "--skip-plot",
    ]
    run_logged(command, log_path, env=env)
    return output_csv, preflight


def load_external_unique_csv(
    path: Path,
    expected_tokens: Sequence[int],
) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "seq_len",
            "batch_size",
            "total_token_num",
            "time_us",
            "time_us_p10",
            "time_us_p90",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"External attention CSV missing columns: {missing}")
        rows = list(reader)
    by_token: dict[int, dict[str, float]] = {}
    for row in rows:
        token = int(row["seq_len"])
        if int(row["batch_size"]) != 1 or int(row["total_token_num"]) != token:
            raise ValueError(f"Invalid BS=1 external attention row: {row}")
        if token in by_token:
            raise ValueError(f"Duplicate external attention token: {token}")
        metrics = {
            name: float(row[name])
            for name in ("time_us", "time_us_p10", "time_us_p90")
        }
        if not (
            0 < metrics["time_us_p10"]
            <= metrics["time_us"]
            <= metrics["time_us_p90"]
        ):
            raise ValueError(f"Invalid attention quantiles for token {token}")
        by_token[token] = metrics
    if set(by_token) != set(expected_tokens):
        raise ValueError(
            "External attention CSV token set differs from the Fig. 5 snapshot: "
            f"missing={sorted(set(expected_tokens) - set(by_token))}, "
            f"extra={sorted(set(by_token) - set(expected_tokens))}"
        )
    return by_token


def write_rank_latency_csv(
    path: Path,
    rank_tokens: Sequence[tuple[int, int]],
    mean_token: int,
    by_token: dict[int, dict[str, float]],
) -> None:
    fieldnames = [
        "rank",
        "token",
        "time_us",
        "time_us_p10",
        "time_us_p90",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, token in rank_tokens:
            writer.writerow({"rank": rank, "token": token, **by_token[token]})
        writer.writerow(
            {"rank": "avg", "token": mean_token, **by_token[mean_token]}
        )


def validate_rank_latency_csv(
    path: Path,
    rank_tokens: Sequence[tuple[int, int]],
    mean_token: int,
) -> None:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "rank",
            "token",
            "time_us",
            "time_us_p10",
            "time_us_p90",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Rank attention CSV missing columns: {missing}")
        rows = list(reader)
    numeric_rows = [row for row in rows if row["rank"].strip() != "avg"]
    average_rows = [row for row in rows if row["rank"].strip() == "avg"]
    actual = sorted(
        (int(row["rank"]), int(float(row["token"]))) for row in numeric_rows
    )
    if actual != sorted(rank_tokens):
        raise ValueError("Rank attention CSV does not match the Fig. 5 snapshot")
    if len(average_rows) != 1 or int(float(average_rows[0]["token"])) != mean_token:
        raise ValueError("Rank attention CSV has an invalid avg row")
    for row in rows:
        values = [float(row[name]) for name in ("time_us_p10", "time_us", "time_us_p90")]
        if not (0 < values[0] <= values[1] <= values[2]):
            raise ValueError(f"Invalid attention timing row: {row}")


def prepare_attention_csv(
    source: Path,
    result_root: Path,
    rank_tokens: Sequence[tuple[int, int]],
    mean_token: int,
    unique_tokens: Sequence[int],
) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as input_file:
        fields = set(csv.DictReader(input_file).fieldnames or ())
    copied_source = result_root / ("attention_source" + source.suffix)
    shutil.copy2(source, copied_source)
    rank_output = result_root / "attention_rank_latency.csv"
    if {"rank", "token"}.issubset(fields):
        validate_rank_latency_csv(source, rank_tokens, mean_token)
        shutil.copy2(source, rank_output)
    elif {"seq_len", "batch_size", "total_token_num"}.issubset(fields):
        by_token = load_external_unique_csv(source, unique_tokens)
        write_rank_latency_csv(rank_output, rank_tokens, mean_token, by_token)
    else:
        raise ValueError(f"Unrecognized attention CSV schema: {sorted(fields)}")
    return copied_source, rank_output


def main() -> int:
    args = parse_args()
    result_root: Path | None = None
    try:
        python = resolve_python(args.python)
        result_root = prepare_result_root(args.result_root)
        run_log = result_root / "run.log"

        attention_snapshot = require_file(
            args.attention_snapshot, "Attention service snapshot"
        )
        deepep_snapshot = require_file(args.deepep_snapshot, "DeepEP snapshot")
        hol_log = require_file(args.hol_log, "HoL frontend.log")
        plot_script = require_file(args.plot_script, "Fig. 5 plot script")

        (
            rank_tokens,
            mean_token,
            unique_tokens,
            attention_step,
            attention_total_steps,
            attention_kv_capacity,
        ) = derive_attention_tokens(attention_snapshot)
        log_message(
            run_log,
            "Attention snapshot: "
            f"step={attention_step}/{attention_total_steps}, "
            f"{len(rank_tokens)} ranks, "
            f"{len(set(token for _rank, token in rank_tokens))} "
            f"unique rank tokens, mean token={mean_token}, "
            f"benchmark cases={len(unique_tokens)}.",
        )

        deepep_rank_csv, deepep_validation = prepare_fresh_deepep_csv(
            deepep_snapshot,
            args.deepep_data_dir,
            args.deepep_local_ranks,
            result_root / "deepep_rank_latency.csv",
        )
        if int(deepep_validation["ranks"]) != len(rank_tokens):
            raise ValueError(
                "Attention and DeepEP snapshots have different rank counts: "
                f"{len(rank_tokens)} and {deepep_validation['ranks']}"
            )
        log_message(
            run_log,
            f"DeepEP fresh results mapped to {deepep_validation['ranks']} "
            "E2E ranks: "
            f"{deepep_rank_csv}",
        )
        (result_root / "deepep_validation.json").write_text(
            json.dumps(deepep_validation, indent=2) + "\n",
            encoding="utf-8",
        )

        preflight = "plot-only: external FlashMLA was not imported"
        if args.attention_csv is None:
            benchmark_script = require_file(
                args.external_benchmark_script,
                "External FlashMLA benchmark script",
            )
            unique_csv, preflight = run_external_benchmark(
                args,
                python,
                benchmark_script,
                unique_tokens,
                result_root,
                run_log,
            )
            by_token = load_external_unique_csv(unique_csv, unique_tokens)
            rank_csv = result_root / "attention_rank_latency.csv"
            write_rank_latency_csv(rank_csv, rank_tokens, mean_token, by_token)
            attention_source = unique_csv
        else:
            attention_source, rank_csv = prepare_attention_csv(
                args.attention_csv,
                result_root,
                rank_tokens,
                mean_token,
                unique_tokens,
            )
        validate_rank_latency_csv(rank_csv, rank_tokens, mean_token)

        figure_base = output_base(args.output)
        plot_command = [
            python,
            str(plot_script),
            "--attention-input",
            str(rank_csv),
            "--hol-input",
            str(hol_log),
            "--hol-engines",
            args.hol_engines,
            "--hol-block-size",
            str(args.hol_block_size),
            "--hol-middle-window-seconds",
            str(args.hol_middle_window_seconds),
            "--year",
            str(args.year),
            "--figure-width",
            "7.0",
            "--figure-height",
            "1.05",
            "--dpi",
            str(args.dpi),
            "--output-base",
            str(figure_base),
        ]
        plot_command.extend(["--deepep-input", str(deepep_rank_csv)])
        plot_env = os.environ.copy()
        plot_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig5")
        run_logged(plot_command, run_log, env=plot_env)
        pdf_path = figure_base.with_suffix(".pdf")
        png_path = figure_base.with_suffix(".png")
        if not pdf_path.is_file() or not png_path.is_file():
            raise RuntimeError("Plot script did not create both PDF and PNG")

        environment = {
            "timestamp": dt.datetime.now().astimezone().isoformat(),
            "python": python,
            "flashmla_preflight": preflight,
            "gpu_physical_id": args.gpu,
            "rep_ms": args.rep_ms,
            "bench_repeats": args.bench_repeats,
            "warmup": args.warmup,
            "attention_case_count": len(unique_tokens),
            "attention_source": str(attention_source),
        }
        (result_root / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )

        manifest = {
            "figure": 5,
            "attention": {
                "snapshot": str(attention_snapshot),
                "time_percent": TIME_PERCENT,
                "step": attention_step,
                "total_steps": attention_total_steps,
                "kv_capacity_tokens": attention_kv_capacity,
                "rank_count": len(rank_tokens),
                "unique_rank_tokens": len(set(token for _rank, token in rank_tokens)),
                "mean_token": mean_token,
                "benchmark_cases": len(unique_tokens),
                "rank_latency_csv": str(rank_csv),
            },
            "deepep": {
                "snapshot": str(deepep_snapshot),
                "data_dir": (
                    str(args.deepep_data_dir.expanduser().resolve())
                ),
                "rank_latency_csv": str(deepep_rank_csv),
                "validation": deepep_validation,
            },
            "hol": {
                "log": str(hol_log),
                "engines": args.hol_engines,
                "block_size": args.hol_block_size,
                "middle_window_seconds": args.hol_middle_window_seconds,
            },
            "outputs": {"pdf": str(pdf_path), "png": str(png_path)},
        }
        (result_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        log_message(run_log, "\nFig. 5 reproduction completed successfully.")
        log_message(run_log, f"Result directory: {result_root}")
        log_message(run_log, f"PDF: {pdf_path}")
        log_message(run_log, f"PNG: {png_path}")
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        if result_root is not None:
            print(f"Partial results, if any: {result_root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

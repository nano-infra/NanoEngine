#!/usr/bin/env python3
"""Run the Fig. 6 microbenchmark and plot its newly profiled results.

This is the single entry point for the artifact.  It launches all 32 CUDA
Graph cases under Nsight Systems, exports a fresh node-level SQLite database
for every case, queries per-replay GPU-kernel medians, selects a representative
rank for each configuration, and finally writes the paper figure.  It invokes
``nsys`` and Python directly and never calls a shell script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import shlex
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig6")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


PAPER_SEQ_LENS = (
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
)
SCENARIO_ORDER = ("pure_dp", "tp2dcp2", "tp4dcp4", "tp8dcp8")
SCENARIO_WORLD_SIZES = {
    "pure_dp": 1,
    "tp2dcp2": 2,
    "tp4dcp4": 4,
    "tp8dcp8": 8,
}
SCENARIO_LABELS = {
    "pure_dp": "DP",
    "tp2dcp2": "CP2",
    "tp4dcp4": "CP4",
    "tp8dcp8": "CP8",
}
SCENARIO_COLORS = {
    "pure_dp": "#4C78A8",
    "tp2dcp2": "#6BAE92",
    "tp4dcp4": "#C85A54",
    "tp8dcp8": "#D2A85F",
}
COMPONENT_ORDER = ("Attention Computation", "CP Communication")
COMPONENT_HATCHES = {
    "Attention Computation": "",
    "CP Communication": "///",
}
COMPONENT_EDGECOLORS = {
    "Attention Computation": "none",
    "CP Communication": (1.0, 1.0, 1.0, 0.82),
}

CASE_RE = re.compile(
    r"^(?P<scenario>pure_dp|tp(?P<tp>\d+)dcp(?P<dcp>\d+))"
    r"_bs(?P<batch_size>\d+)_seqlen(?P<seq_len>\d+)_uniform$"
)
PROCESS_ID_MASK = (1 << 24) - 1

FIGSIZE = (6.6, 3.1)
STRATEGY_LEGEND_FONT_SIZE = 18
COMPONENT_LEGEND_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
BAR_WIDTH = 0.18

SQL_COMPONENT_MEDIANS = f"""
WITH runtime_launches AS (
    SELECT
        p.globalPid AS global_pid,
        p.pid AS pid,
        r.start AS launch_start
    FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
    JOIN StringIds AS runtime_name
      ON runtime_name.id = r.nameId
    JOIN PROCESSES AS p
      ON p.pid = (r.globalTid & {PROCESS_ID_MASK})
    WHERE lower(runtime_name.value) LIKE '%graphlaunch%'
),
launches AS (
    SELECT
        global_pid,
        pid,
        launch_start,
        LEAD(launch_start) OVER (
            PARTITION BY global_pid ORDER BY launch_start
        ) AS next_launch_start
    FROM runtime_launches
),
graph_kernels AS (
    SELECT
        k.globalPid AS global_pid,
        k.deviceId AS device_id,
        k.start AS kernel_start,
        k.end AS kernel_end,
        kernel_name.value AS kernel_name
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN StringIds AS kernel_name
      ON kernel_name.id = k.demangledName
    WHERE k.graphId IS NOT NULL
),
per_replay AS (
    SELECT
        launches.global_pid,
        launches.pid,
        graph_kernels.device_id,
        (MAX(graph_kernels.kernel_end) - MIN(graph_kernels.kernel_start))
            / 1000.0 AS graph_elapsed_us,
        SUM(CASE
            WHEN graph_kernels.kernel_name LIKE '%ncclDevKernel_Broadcast%'
            THEN graph_kernels.kernel_end - graph_kernels.kernel_start
            ELSE 0 END) / 1000.0 AS graph_sync_broadcast_us,
        SUM(CASE
            WHEN graph_kernels.kernel_name LIKE '%ncclDevKernel_AllGather%'
            THEN graph_kernels.kernel_end - graph_kernels.kernel_start
            ELSE 0 END) / 1000.0 AS query_allgather_us,
        SUM(CASE
            WHEN graph_kernels.kernel_name LIKE '%flash_fwd_splitkv_mla_kernel%'
              OR graph_kernels.kernel_name LIKE '%flash_fwd_mla_combine_kernel%'
            THEN graph_kernels.kernel_end - graph_kernels.kernel_start
            ELSE 0 END) / 1000.0 AS flashmla_us,
        SUM(CASE
            WHEN graph_kernels.kernel_name LIKE '%ncclDevKernel_SendRecv%'
            THEN graph_kernels.kernel_end - graph_kernels.kernel_start
            ELSE 0 END) / 1000.0 AS post_a2a_comm_us,
        SUM(CASE
            WHEN graph_kernels.kernel_name LIKE '%ncclDevKernel_AllReduce%'
              OR graph_kernels.kernel_name LIKE '%multimem_all_reduce_kernel%'
              OR graph_kernels.kernel_name LIKE '%allreduce_fusion_kernel_oneshot_lamport%'
            THEN graph_kernels.kernel_end - graph_kernels.kernel_start
            ELSE 0 END) / 1000.0 AS tp_allreduce_us
    FROM launches
    JOIN graph_kernels
      ON graph_kernels.global_pid = launches.global_pid
     AND graph_kernels.kernel_start >= launches.launch_start
     AND (
          launches.next_launch_start IS NULL
          OR graph_kernels.kernel_start < launches.next_launch_start
     )
    GROUP BY
        launches.global_pid,
        launches.pid,
        graph_kernels.device_id,
        launches.launch_start
)
SELECT
    pid,
    device_id,
    COUNT(*) AS replay_count,
    median(graph_elapsed_us) AS graph_elapsed_us,
    median(graph_sync_broadcast_us) AS graph_sync_broadcast_us,
    median(query_allgather_us) AS query_allgather_us,
    median(flashmla_us) AS flashmla_us,
    median(post_a2a_comm_us) AS post_a2a_comm_us,
    median(tp_allreduce_us) AS tp_allreduce_us
FROM per_replay
GROUP BY global_pid, pid, device_id
ORDER BY device_id, pid
"""


@dataclass(frozen=True)
class Case:
    path: Path
    scenario: str
    dcp_size: int
    batch_size: int
    seq_len: int


@dataclass(frozen=True)
class Point:
    case: Case
    rank: int
    replay_count: int
    graph_elapsed_us: float
    attention_us: float
    communication_us: float


class SQLiteMedian:
    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float | None) -> None:
        if value is not None:
            self.values.append(float(value))

    def finalize(self) -> float | None:
        return statistics.median(self.values) if self.values else None


def parse_args() -> argparse.Namespace:
    artifact_dir = Path(__file__).resolve().parent
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Freshly profile all 32 CUDA Graph cases for Fig. 6, parse their "
            "node-level Nsight Systems SQLite files, and render the figure."
        )
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=(
            artifact_dir
            / "benchmark"
            / "run_flashmla_dcp_cudagraph_bench.py"
        ),
        help=(
            "CUDA Graph benchmark implementation invoked for every case "
            "(default: the implementation bundled under fig6/benchmark)."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=artifact_dir / "deepseek_v3_config.json",
        help=(
            "DeepSeek-V3 config.json used to obtain model dimensions "
            "(default: fig6/deepseek_v3_config.json)."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=artifact_dir / "results" / f"run_{run_id}_pid{os.getpid()}",
        help=(
            "Directory for this fresh run. The default is a unique timestamped "
            "directory under fig6/results."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "fig6",
        help="Figure path without extension (default: fig6/fig6).",
    )
    parser.add_argument("--nsys", default="nsys", help="Nsight Systems binary.")
    parser.add_argument(
        "--python",
        default="python3",
        help="Python interpreter used for benchmark and rank processes.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Eager warmup iterations."
    )
    parser.add_argument(
        "--graph-warmup-iters",
        type=int,
        default=3,
        help="Untimed CUDA Graph replays after capture.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Profiled CUDA Graph replays per rank.",
    )
    parser.add_argument("--cp-interleave-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    args = parser.parse_args()
    if args.warmup_iters < 1:
        parser.error("--warmup-iters must be positive")
    if args.graph_warmup_iters < 0:
        parser.error("--graph-warmup-iters must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be positive")
    if args.cp_interleave_size < 1:
        parser.error("--cp-interleave-size must be positive")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def resolve_executable(
    value: str,
    label: str,
    fallback_patterns: tuple[str, ...] = (),
) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise FileNotFoundError(f"{label} executable not found: {candidate}")
    resolved = shutil.which(value)
    if resolved is not None:
        return resolved
    for pattern in fallback_patterns:
        for path_text in sorted(glob.glob(pattern), reverse=True):
            path = Path(path_text)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    raise FileNotFoundError(
        f"{label} executable not found on PATH or in standard install "
        f"locations: {value}. Pass --{label.lower()} explicitly."
    )


def validate_visible_devices() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) < 8:
        raise RuntimeError(
            "Fig. 6 requires eight visible GPUs; CUDA_VISIBLE_DEVICES currently "
            f"contains {len(devices)}: {visible!r}."
        )


def prepare_run(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, str, str]:
    benchmark_script = args.benchmark_script.expanduser().resolve()
    model_config = args.model_config.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    if not benchmark_script.is_file():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")
    validate_visible_devices()
    nsys = resolve_executable(
        args.nsys,
        "nsys",
        fallback_patterns=(
            "/usr/local/bin/nsys",
            "/usr/local/cuda/bin/nsys",
            "/usr/local/cuda/nsight-systems/bin/nsys",
            "/opt/nvidia/nsight-systems-cli/*/bin/nsys",
            "/opt/nvidia/nsight-systems-cli/*/target-linux-x64/nsys",
        ),
    )
    python = resolve_executable(args.python, "Python")

    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty result directory: {result_root}. "
            "Choose a new --result-root."
        )
    result_root.mkdir(parents=True, exist_ok=True)
    return benchmark_script, model_config, result_root, nsys, python


def tail_log(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def run_logged(command: list[str], log_path: Path, append: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_file:
        if append:
            log_file.write("\n")
        log_file.write("COMMAND: " + shlex.join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        excerpt = tail_log(log_path)
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. "
            f"See {log_path}.\nLast log lines:\n{excerpt}"
        )


INTERNAL_LAUNCH_MODE = "__launch_rank_workers__"


def choose_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def launch_rank_workers(argv: list[str]) -> int:
    """Internal nsys target: spawn one benchmark process per local rank."""
    if len(argv) < 4:
        raise ValueError(
            f"{INTERNAL_LAUNCH_MODE} expects WORLD_SIZE PYTHON BENCHMARK [ARGS...]"
        )
    world_size = int(argv[0])
    if world_size < 2:
        raise ValueError("The internal multi-rank launcher requires world_size >= 2")
    python = argv[1]
    benchmark_command = argv[2:]
    master_port = choose_local_port()
    workers: list[subprocess.Popen[bytes]] = []

    try:
        for rank in range(world_size):
            worker_env = os.environ.copy()
            worker_env.update(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": str(world_size),
                    "GROUP_RANK": "0",
                    "ROLE_RANK": str(rank),
                    "ROLE_WORLD_SIZE": str(world_size),
                    "OMP_NUM_THREADS": "1",
                }
            )
            workers.append(
                subprocess.Popen(
                    [python, *benchmark_command],
                    env=worker_env,
                )
            )

        while True:
            statuses = [worker.poll() for worker in workers]
            failure = next(
                (status for status in statuses if status not in (None, 0)),
                None,
            )
            if failure is not None:
                for worker, status in zip(workers, statuses):
                    if status is None:
                        worker.terminate()
                for worker in workers:
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
                return int(failure)
            if all(status == 0 for status in statuses):
                return 0
            time.sleep(0.1)
    except BaseException:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        raise


def profile_all_cases(
    args: argparse.Namespace,
    benchmark_script: Path,
    model_config: Path,
    result_root: Path,
    nsys: str,
    python: str,
) -> Path:
    case_dir = result_root / "nsys" / "dcpbudget1024k"
    event_dir = result_root / "online_event" / "dcpbudget1024k"
    log_dir = result_root / "logs"
    case_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fresh result root: {result_root}", flush=True)
    print("Profiling 32 CUDA Graph cases...", flush=True)
    case_index = 0
    for seq_len in PAPER_SEQ_LENS:
        batch_per_gpu = (1024 * 1024) // seq_len
        for scenario in SCENARIO_ORDER:
            case_index += 1
            world_size = SCENARIO_WORLD_SIZES[scenario]
            batch_size = batch_per_gpu * world_size
            stem = f"{scenario}_bs{batch_size}_seqlen{seq_len}_uniform"
            profile_base = case_dir / stem
            report_path = profile_base.with_suffix(".nsys-rep")
            sqlite_path = profile_base.with_suffix(".sqlite")
            log_path = log_dir / f"{stem}.log"

            print(
                f"[{case_index:02d}/32] Profile {stem} "
                f"(world_size={world_size})",
                flush=True,
            )
            bench_args = [
                str(benchmark_script),
                "--scenario",
                scenario,
                "--model-config",
                str(model_config),
                "--batch-size",
                str(batch_size),
                "--seq-len",
                str(seq_len),
                "--min-seq-len",
                "1024",
                "--seq-len-mode",
                "uniform",
                "--cp-interleave-size",
                str(args.cp_interleave_size),
                "--block-size",
                str(args.block_size),
                "--dtype",
                args.dtype,
                "--warmup-iters",
                str(args.warmup_iters),
                "--graph-warmup-iters",
                str(args.graph_warmup_iters),
                "--iters",
                str(args.iters),
                "--emit-nvtx",
                "--use-cuda-profiler-range",
                "--output-dir",
                str(event_dir),
                "--output-prefix",
                "",
            ]
            if world_size == 1:
                launcher = [python, *bench_args]
            else:
                bench_args.append("--verify-world-size")
                launcher = [
                    python,
                    str(Path(__file__).resolve()),
                    INTERNAL_LAUNCH_MODE,
                    str(world_size),
                    python,
                    *bench_args,
                ]

            profile_command = [
                nsys,
                "profile",
                "--cuda-graph-trace=node",
                "--force-overwrite=true",
                "--sample=none",
                "--trace=cuda,nvtx,osrt",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "-o",
                str(profile_base),
                *launcher,
            ]
            started = time.monotonic()
            run_logged(profile_command, log_path)
            if not report_path.is_file() or not report_path.stat().st_size:
                raise RuntimeError(
                    f"nsys did not create the expected report: {report_path}"
                )
            export_command = [
                nsys,
                "export",
                "-t",
                "sqlite",
                "-f",
                "true",
                "-o",
                str(sqlite_path),
                str(report_path),
            ]
            run_logged(export_command, log_path, append=True)
            if not sqlite_path.is_file() or not sqlite_path.stat().st_size:
                raise RuntimeError(
                    f"nsys did not create the expected SQLite file: {sqlite_path}"
                )
            elapsed = time.monotonic() - started
            print(f"           completed in {elapsed:.1f}s", flush=True)

    return case_dir


def case_from_path(path: Path) -> Case | None:
    match = CASE_RE.fullmatch(path.stem)
    if match is None:
        return None
    scenario = match.group("scenario")
    if scenario not in SCENARIO_ORDER:
        return None
    dcp_size = 1 if scenario == "pure_dp" else int(match.group("dcp"))
    return Case(
        path=path.resolve(),
        scenario=scenario,
        dcp_size=dcp_size,
        batch_size=int(match.group("batch_size")),
        seq_len=int(match.group("seq_len")),
    )


def discover_cases(input_path: Path) -> dict[tuple[int, str], Case]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    candidates = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.rglob("*.sqlite"))
    )
    selected: dict[tuple[int, str], Case] = {}
    duplicates: list[str] = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix != ".sqlite":
            continue
        case = case_from_path(candidate)
        if case is None or case.seq_len not in PAPER_SEQ_LENS:
            continue
        key = (case.seq_len, case.scenario)
        if key in selected:
            duplicates.append(f"{case.seq_len}/{case.scenario}")
        else:
            selected[key] = case

    if duplicates:
        raise ValueError(
            "Multiple SQLite files match the same paper point: "
            + ", ".join(sorted(set(duplicates)))
        )

    missing = [
        f"{format_seq_len(seq_len)} / {SCENARIO_LABELS[scenario]}"
        for seq_len in PAPER_SEQ_LENS
        for scenario in SCENARIO_ORDER
        if (seq_len, scenario) not in selected
    ]
    if missing:
        raise ValueError(
            f"Found {len(selected)} of 32 required profiles. Missing: "
            + ", ".join(missing)
        )
    return selected


def require_nsys_tables(connection: sqlite3.Connection, path: Path) -> None:
    available = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required = {
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "PROCESSES",
        "StringIds",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"{path}: missing nsys tables: {', '.join(missing)}")
    kernel_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)"
        )
    }
    if "graphId" not in kernel_columns:
        raise RuntimeError(
            f"{path}: graphId is absent; export with --cuda-graph-trace=node."
        )


def query_case(case: Case, expected_iters: int) -> Point:
    connection = sqlite3.connect(case.path)
    connection.row_factory = sqlite3.Row
    connection.create_aggregate("median", 1, SQLiteMedian)
    try:
        require_nsys_tables(connection, case.path)
        rows = [dict(row) for row in connection.execute(SQL_COMPONENT_MEDIANS)]
    finally:
        connection.close()

    if len(rows) != case.dcp_size:
        raise RuntimeError(
            f"{case.path}: expected {case.dcp_size} rank rows, found {len(rows)}."
        )

    device_ids = sorted(int(row["device_id"]) for row in rows)
    if len(set(device_ids)) != len(device_ids):
        raise RuntimeError(f"{case.path}: duplicate device IDs: {device_ids}")
    rank_by_device = {device_id: rank for rank, device_id in enumerate(device_ids)}

    if expected_iters:
        bad_counts = [
            (rank_by_device[int(row["device_id"])], int(row["replay_count"]))
            for row in rows
            if int(row["replay_count"]) != expected_iters
        ]
        if bad_counts:
            raise RuntimeError(
                f"{case.path}: expected {expected_iters} replays per rank; "
                f"observed {bad_counts}."
            )

    required_components = ["flashmla_us"]
    if case.dcp_size > 1:
        required_components.extend(
            [
                "graph_sync_broadcast_us",
                "query_allgather_us",
                "post_a2a_comm_us",
                "tp_allreduce_us",
            ]
        )
    for row in rows:
        missing_components = [
            component
            for component in required_components
            if float(row[component] or 0.0) <= 0.0
        ]
        if missing_components:
            rank = rank_by_device[int(row["device_id"])]
            raise RuntimeError(
                f"{case.path}: rank {rank} has no positive median for "
                + ", ".join(missing_components)
            )

    totals = [float(row["graph_elapsed_us"]) for row in rows]
    median_total = statistics.median(totals)
    representative = min(
        rows,
        key=lambda row: (
            abs(float(row["graph_elapsed_us"]) - median_total),
            float(row["graph_elapsed_us"]),
            rank_by_device[int(row["device_id"])],
        ),
    )
    communication_us = sum(
        float(representative[name] or 0.0)
        for name in (
            "graph_sync_broadcast_us",
            "query_allgather_us",
            "post_a2a_comm_us",
            "tp_allreduce_us",
        )
    )
    return Point(
        case=case,
        rank=rank_by_device[int(representative["device_id"])],
        replay_count=int(representative["replay_count"]),
        graph_elapsed_us=float(representative["graph_elapsed_us"]),
        attention_us=float(representative["flashmla_us"] or 0.0),
        communication_us=communication_us,
    )


def format_seq_len(seq_len: int) -> str:
    if seq_len >= 1024 * 1024 and seq_len % (1024 * 1024) == 0:
        return f"{seq_len // (1024 * 1024)}M"
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}K"
    return str(seq_len)


def batch_size_per_gpu(points: dict[tuple[int, str], Point], seq_len: int) -> int:
    values = {
        point.case.batch_size // point.case.dcp_size
        for (length, _), point in points.items()
        if length == seq_len
    }
    if len(values) != 1:
        raise RuntimeError(
            f"Inconsistent batch size per GPU for {format_seq_len(seq_len)}: "
            f"{sorted(values)}"
        )
    return values.pop()


def print_breakdown(points: dict[tuple[int, str], Point]) -> None:
    print("Fig. 6 data selected from Nsight Systems (median us per replay):")
    for seq_len in PAPER_SEQ_LENS:
        bs_per_gpu = batch_size_per_gpu(points, seq_len)
        print(f"[{format_seq_len(seq_len)} x {bs_per_gpu}]")
        for scenario in SCENARIO_ORDER:
            point = points[(seq_len, scenario)]
            total = point.attention_us + point.communication_us
            print(
                f"  {SCENARIO_LABELS[scenario]} (rank {point.rank}): "
                f"Attention Computation={point.attention_us:.3f} | "
                f"CP Communication={point.communication_us:.3f} | "
                f"PlotTotal={total:.3f} | {point.case.path}"
            )


def setup_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": get_plot_font_family(),
            "font.size": LABEL_FONT_SIZE,
            "axes.edgecolor": "#64748b",
            "axes.linewidth": 0.95,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0f172a",
            "axes.labelcolor": "#0f172a",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.0,
        }
    )


def add_legends(fig: plt.Figure, ax: plt.Axes) -> None:
    strategy_handles = [
        Patch(
            facecolor=SCENARIO_COLORS[scenario],
            edgecolor="none",
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIO_ORDER
    ]
    component_handles = [
        Patch(
            facecolor="#bfc8d3",
            edgecolor=(
                COMPONENT_EDGECOLORS[component]
                if COMPONENT_HATCHES[component]
                else "#94a3b8"
            ),
            linewidth=0.4 if COMPONENT_HATCHES[component] else 0.6,
            hatch=COMPONENT_HATCHES[component],
            label=component,
        )
        for component in COMPONENT_ORDER
    ]
    strategy_legend = fig.legend(
        handles=strategy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=STRATEGY_LEGEND_FONT_SIZE,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.legend(
        handles=component_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=COMPONENT_LEGEND_FONT_SIZE,
        handlelength=1.25,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.4,
        borderaxespad=0.15,
    )
    fig.add_artist(strategy_legend)


def render(points: dict[tuple[int, str], Point], output: Path, dpi: int) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    max_total_us = 0.0

    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        offset = (scenario_index - (len(SCENARIO_ORDER) - 1) / 2.0) * BAR_WIDTH
        for group_position, seq_len in enumerate(PAPER_SEQ_LENS):
            point = points[(seq_len, scenario)]
            values = {
                "Attention Computation": point.attention_us,
                "CP Communication": point.communication_us,
            }
            bottom = 0.0
            for component in COMPONENT_ORDER:
                value = values[component]
                hatch = COMPONENT_HATCHES[component]
                ax.bar(
                    group_position + offset,
                    value,
                    width=BAR_WIDTH * 0.9,
                    bottom=bottom,
                    color=SCENARIO_COLORS[scenario],
                    edgecolor=COMPONENT_EDGECOLORS[component],
                    linewidth=0.4 if hatch else 0.0,
                    hatch=hatch,
                    zorder=3,
                )
                bottom += value
            max_total_us = max(max_total_us, bottom)

    positions = list(range(len(PAPER_SEQ_LENS)))
    ax.set_xlim(-0.58, len(PAPER_SEQ_LENS) - 0.42)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{format_seq_len(seq_len)}×{batch_size_per_gpu(points, seq_len)}"
            for seq_len in PAPER_SEQ_LENS
        ]
    )
    ax.set_xlabel(
        r"Sequence length $\times$ batch size per GPU",
        fontsize=LABEL_FONT_SIZE,
        labelpad=2,
    )
    ax.set_ylabel("Latency (us)", fontsize=LABEL_FONT_SIZE, labelpad=8)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, pad=8, length=0)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE, length=3, width=0.9, pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.set_ylim(0, max_total_us * 1.17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.grid(axis="y", linestyle="-", linewidth=0.75, color="#e6ebf2", zorder=0)
    ax.grid(visible=False, axis="x")
    add_legends(fig, ax)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.28, top=0.92)

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output_path = output.with_suffix(f".{suffix}")
        save_kwargs: dict[str, Any] = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(
            output_path,
            format=suffix,
            bbox_inches="tight",
            pad_inches=0.02,
            **save_kwargs,
        )
        print(f"Wrote {output_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_script, model_config, result_root, nsys, python = prepare_run(args)
    sqlite_dir = profile_all_cases(
        args=args,
        benchmark_script=benchmark_script,
        model_config=model_config,
        result_root=result_root,
        nsys=nsys,
        python=python,
    )
    print("Parsing the 32 freshly exported SQLite files...", flush=True)
    cases = discover_cases(sqlite_dir)
    points = {
        key: query_case(case, args.iters)
        for key, case in sorted(cases.items())
    }
    print_breakdown(points)
    render(points, args.output, args.dpi)
    print(f"Raw profiling results: {result_root}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == INTERNAL_LAUNCH_MODE:
        raise SystemExit(launch_rank_workers(sys.argv[2:]))
    main()

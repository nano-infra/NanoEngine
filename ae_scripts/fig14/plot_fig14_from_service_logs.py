#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
NUM_KVCACHE_BLOCKS_RE = re.compile(r"num_kvcache_blocks:\s*(\d+)")
GPU_KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
VLLM_HEADER_RE = re.compile(
    r"(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine_id>\d+):"
)
RUNNING_REQS_RE = re.compile(r"Running:\s+(?P<value>\d+)\s+reqs")
GPU_KV_USAGE_RE = re.compile(r"GPU KV cache usage:\s+(?P<value>\d+(?:\.\d+)?)%")
VLLM_CAPACITY_RE = re.compile(
    r"poisson_gpu_kv_cache_capacity .*?"
    r"managed_engines=(?P<managed_engines>\d+).*?"
    r"usable_gpu_blocks=(?P<usable_gpu_blocks>[\d,]+).*?"
    r"block_size=(?P<reported_block_size>\d+)"
)
VLLM_ENGINE_METRIC_RE = re.compile(
    r"INFO\s+(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine_id>\d+):.*?"
    r"Waiting head tokens:\s+(?P<waiting_head_tokens>\d+),\s+"
    r"GPU KV cache usage:\s+(?P<gpu_kv_cache_usage_pct>\d+(?:\.\d+)?)%"
)

BASE_FONT_SIZE = 8.0
TITLE_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 8.0
TICK_LABEL_FONT_SIZE = 7.2
LEGEND_FONT_SIZE = 7.8
CAPTION_FONT_SIZE = 8.2
LEFT_SHARED_YLABEL_X = -0.19

LOAD_BALANCE_COLOR = "#1f77b4"
CV_COLOR = "purple"
FREE_LINE_COLOR = "#E67F0D"
HOL_OUTLINE_COLOR = "#2C617C"
TEXT_COLOR = "#333333"
GRID_COLOR = "#E0E0E0"

LOAD_BALANCE_ROW_LABELS = ("Batch Size", "Used Blocks")
LOAD_BALANCE_TITLES = {
    "Nano DCP": "Ours (DCP)",
    "vLLM DP": "vLLM (DP-LeastBatch)",
    "vLLM LeastCache DP": "vLLM (DP-LeastCache)",
}

FIGURE_WIDTH_INCHES = 7.2
FIGURE_HEIGHT_INCHES = 2.18


@dataclass(frozen=True)
class LoadBalanceLogSpec:
    name: str
    path: Path
    log_type: str
    center_steps: int = 0


@dataclass
class LoadBalanceParsedData:
    steps: list[int]
    used_series: list[list[float]]
    batch_series: list[list[float]]
    max_blocks_per_rank: int
    num_ranks: int
    total_steps: int
    center_start: int
    center_end: int


@dataclass(frozen=True)
class HighLoadStepRecord:
    free_blocks_per_group: list[float]
    waiting_head_blocks_per_group: list[float]
    waiting_head_blocks_per_rank: list[float]
    total_ranks: int


@dataclass(frozen=True)
class HighLoadParsedSeries:
    path: Path
    steps: list[int]
    group_free_blocks: list[list[float]]
    group_waiting_head_blocks: list[list[float]]
    group_waiting_head_blocks_by_rank: list[list[list[float]]]
    max_blocks_per_rank: float
    num_ranks: int
    num_groups: int
    group_size: int
    total_steps: int
    selected_start: int
    selected_end: int

    @property
    def group_capacity_blocks(self) -> float:
        return self.max_blocks_per_rank * self.group_size


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": get_plot_font_family(),
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_frontend_log(case_root: Path) -> Path:
    successful: list[Path] = []
    for summary_path in case_root.rglob("summary.json"):
        if summary_path.parent.name != "benchmark":
            continue
        requests_path = summary_path.with_name("requests.jsonl")
        frontend_path = summary_path.parent.parent / "frontend.log"
        if not requests_path.is_file() or not frontend_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            total = int(summary["total_requests"])
            completed = int(summary["successful_requests"])
            failed = int(summary.get("failed_requests", 0))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if total > 0 and completed == total and failed == 0:
            successful.append(frontend_path)
    if successful:
        return max(successful, key=lambda path: path.stat().st_mtime_ns)

    matches = sorted(case_root.rglob("frontend.log"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one frontend.log below {case_root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def apply_run_root_logs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        if not run_root.is_dir():
            parser.error(f"--run-root is not a directory: {run_root}")
        nano_lb = run_root / "nano" / "lb" / "driver.log"
        nano_hl = run_root / "nano" / "hol" / "driver.log"
        defaults: dict[str, Path | None] = {
            "lb_nano_log": nano_lb if nano_lb.is_file() else None,
            "hl_nano_log": nano_hl if nano_hl.is_file() else None,
        }
        for argument, case_name in (
            ("lb_vllm_log", "lb_least_batch"),
            ("lb_vllm_least_cache_log", "lb_least_cache"),
            ("hl_vllm_log", "hol_least_batch"),
        ):
            try:
                defaults[argument] = find_frontend_log(
                    run_root / "vllm" / case_name
                )
            except ValueError:
                defaults[argument] = None
        for argument, path in defaults.items():
            if getattr(args, argument) is None and path is not None:
                setattr(args, argument, path)

    missing = [
        option
        for option, argument in (
            ("--lb-nano-log", "lb_nano_log"),
            ("--lb-vllm-log", "lb_vllm_log"),
            ("--lb-vllm-least-cache-log", "lb_vllm_least_cache_log"),
            ("--hl-nano-log", "hl_nano_log"),
            ("--hl-vllm-log", "hl_vllm_log"),
        )
        if getattr(args, argument) is None
    ]
    if missing and args.run_root is None:
        parser.error(
            "provide --run-root or all five log options; missing: "
            + ", ".join(missing)
        )
    if len(missing) == 5:
        parser.error("no Fig. 14 service logs were found below --run-root")
    args.missing_log_options = tuple(missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the available Fig. 14 service logs; use all five for the "
            "complete figure."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "Fig. 14 run directory; resolves available logs and produces a "
            "partial plot until all five cases are present."
        ),
    )
    parser.add_argument("--lb-nano-log", type=Path)
    parser.add_argument("--lb-vllm-log", type=Path)
    parser.add_argument(
        "--lb-vllm-least-cache-log",
        type=Path,
    )
    parser.add_argument("--lb-nano-center-steps", type=int, default=350)
    parser.add_argument("--lb-vllm-center-steps", type=int, default=400)
    parser.add_argument("--lb-vllm-least-cache-center-steps", type=int, default=400)

    parser.add_argument("--hl-nano-log", type=Path)
    parser.add_argument("--hl-vllm-log", type=Path)
    parser.add_argument("--hl-nano-label", default="Ours (DCP)")
    parser.add_argument("--hl-vllm-label", default="vLLM (DP-LeastBatch)")
    parser.add_argument("--hl-skip-steps", type=int, default=0)
    parser.add_argument("--hl-skip-tail-steps", type=int, default=0)
    parser.add_argument("--hl-center-steps", type=int, default=0)
    parser.add_argument("--hl-group-size", type=int, default=8)
    parser.add_argument("--hl-vllm-block-size", type=float, default=64.0)
    parser.add_argument(
        "--hl-group-index",
        default="auto",
        help='8-GPU group index for panel (b), or "auto" to pick the group with the '
        "largest HoL peak.",
    )

    parser.add_argument(
        "--left-caption",
        default="(a) Load balance across all instances.",
        help="Caption shown below the load-balance panel.",
    )
    parser.add_argument(
        "--right-caption",
        default="(b) HoL demand vs free blocks.",
        help=(
            "Caption shown below the HoL panel. If it contains {gpu_range}, the "
            "selected GPU range is substituted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output base path or explicit file path. The script always writes both "
            "<base>.pdf and <base>.png."
        ),
    )
    args = parser.parse_args()
    apply_run_root_logs(parser, args)
    return args


def flatten_list(value) -> list[float]:
    flat: list[float] = []
    for item in value:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def resolve_center_window(total_steps: int, center_steps: int) -> tuple[int, int]:
    if center_steps <= 0 or center_steps >= total_steps:
        return 0, total_steps
    start = (total_steps - center_steps) // 2
    return start, start + center_steps


def center_trim(data: list[list[float]], center_steps: int) -> list[list[float]]:
    start, end = resolve_center_window(len(data), center_steps)
    return data[start:end]


def transpose_timesteps(data: list[list[float]], num_ranks: int) -> list[list[float]]:
    padded_steps: list[list[float]] = []
    for step_values in data:
        padded = list(step_values[:num_ranks])
        if len(padded) < num_ranks:
            padded.extend([0.0] * (num_ranks - len(padded)))
        padded_steps.append(padded)
    return [list(series) for series in zip(*padded_steps)]


def parse_nano_load_balance_log(path: Path, center_steps: int) -> LoadBalanceParsedData:
    if not path.is_file():
        raise FileNotFoundError(f"Nano log not found: {path}")

    max_blocks_per_rank: int | None = None
    free_steps: list[list[float]] = []
    batch_steps: list[list[float]] = []

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line)

            if max_blocks_per_rank is None:
                kv_match = NUM_KVCACHE_BLOCKS_RE.search(line)
                if kv_match is not None:
                    max_blocks_per_rank = int(kv_match.group(1))

            if "step - {" not in line:
                continue

            _, _, payload = line.partition("step - ")
            payload = payload.strip()
            if not payload.startswith("{"):
                continue

            try:
                step_data = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue

            if step_data.get("mode") != "decode":
                continue

            free_blocks = flatten_list(step_data.get("free_blocks", []))
            batch_sizes = flatten_list(step_data.get("sp_batch_sizes", []))
            if not free_blocks or not batch_sizes:
                continue

            free_steps.append(free_blocks)
            batch_steps.append(batch_sizes)

    if not free_steps or not batch_steps:
        raise ValueError(f"No decode steps parsed from Nano log: {path}")

    max_blocks = max_blocks_per_rank or 20000
    total_steps = len(free_steps)
    center_start, center_end = resolve_center_window(total_steps, center_steps)
    free_steps = center_trim(free_steps, center_steps)
    batch_steps = center_trim(batch_steps, center_steps)

    num_ranks = max(
        max(len(step) for step in free_steps),
        max(len(step) for step in batch_steps),
    )
    free_series = transpose_timesteps(free_steps, num_ranks)
    used_series = [
        [max_blocks - free for free in rank_free]
        for rank_free in free_series
    ]
    batch_series = transpose_timesteps(batch_steps, num_ranks)
    steps = list(range(1, len(free_steps) + 1))

    return LoadBalanceParsedData(
        steps=steps,
        used_series=used_series,
        batch_series=batch_series,
        max_blocks_per_rank=max_blocks,
        num_ranks=num_ranks,
        total_steps=total_steps,
        center_start=center_start,
        center_end=center_end,
    )


def parse_vllm_load_balance_log(path: Path, center_steps: int) -> LoadBalanceParsedData:
    if not path.is_file():
        raise FileNotFoundError(f"vLLM log not found: {path}")

    kv_token_sizes: list[int] = []
    timestep_map: dict[str, dict[int, tuple[int, float]]] = {}

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line)

            kv_match = GPU_KV_SIZE_RE.search(line)
            if kv_match is not None:
                kv_token_sizes.append(int(kv_match.group(1).replace(",", "")))

            if "Engine " not in line or "GPU KV cache usage:" not in line:
                continue

            header_match = VLLM_HEADER_RE.search(line)
            running_match = RUNNING_REQS_RE.search(line)
            usage_match = GPU_KV_USAGE_RE.search(line)
            if header_match is None or running_match is None or usage_match is None:
                continue

            timestamp = header_match.group("timestamp")
            engine_id = int(header_match.group("engine_id"))
            running = int(running_match.group("value"))
            usage_pct = float(usage_match.group("value"))

            engines = timestep_map.setdefault(timestamp, {})
            engines[engine_id] = (running, usage_pct)

    if not timestep_map:
        raise ValueError(f"No engine stats parsed from vLLM log: {path}")
    if not kv_token_sizes:
        raise ValueError(f"No GPU KV cache size found in vLLM log: {path}")

    timestamps = list(timestep_map.keys())
    total_steps = len(timestamps)
    center_start, center_end = resolve_center_window(total_steps, center_steps)
    timestamps = timestamps[center_start:center_end]

    max_blocks_per_rank = min(kv_token_sizes) // 64
    num_ranks = max(max(engines.keys()) for engines in timestep_map.values()) + 1

    batch_steps: list[list[float]] = []
    used_steps: list[list[float]] = []
    for timestamp in timestamps:
        engines = timestep_map[timestamp]
        batch_row: list[float] = []
        used_row: list[float] = []
        for rank in range(num_ranks):
            running, usage_pct = engines.get(rank, (0, 0.0))
            batch_row.append(float(running))
            used_row.append(max_blocks_per_rank * usage_pct / 100.0)
        batch_steps.append(batch_row)
        used_steps.append(used_row)

    return LoadBalanceParsedData(
        steps=list(range(1, len(timestamps) + 1)),
        used_series=transpose_timesteps(used_steps, num_ranks),
        batch_series=transpose_timesteps(batch_steps, num_ranks),
        max_blocks_per_rank=max_blocks_per_rank,
        num_ranks=num_ranks,
        total_steps=total_steps,
        center_start=center_start,
        center_end=center_end,
    )


def calc_stats(data_series: list[list[float]]) -> tuple[list[float], ...]:
    num_steps = len(data_series[0])
    mins: list[float] = []
    maxs: list[float] = []
    p25s: list[float] = []
    p75s: list[float] = []
    medians: list[float] = []

    for step_idx in range(num_steps):
        values = np.array([series[step_idx] for series in data_series], dtype=float)
        mins.append(float(np.min(values)))
        maxs.append(float(np.max(values)))
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        p25s.append(float(p25))
        medians.append(float(p50))
        p75s.append(float(p75))

    return mins, maxs, p25s, p75s, medians


def calc_cv(data_series: list[list[float]]) -> list[float]:
    num_steps = len(data_series[0])
    cvs: list[float] = []
    for step_idx in range(num_steps):
        values = np.array([series[step_idx] for series in data_series], dtype=float)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        cvs.append((std_value / mean_value * 100.0) if mean_value > 0 else 0.0)
    return cvs


def calc_mean_cv(data_series: list[list[float]]) -> float:
    cvs = calc_cv(data_series)
    return float(np.mean(cvs)) if cvs else 0.0


def compact_block_tick(value: float, _: object) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{int(round(value))}"


def padded_axis_upper(values: list[float], padding: float = 1.05) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return 1.0
    maximum = max(finite_values)
    return maximum * padding if maximum > 0 else 1.0


def plot_load_balance_axis(
    ax,
    steps: list[int],
    data_series: list[list[float]],
    scale: float,
    use_k_suffix: bool,
    show_x_tick_labels: bool,
    cv_y_max: float,
    show_cv_ticks: bool,
) -> tuple[object, object]:
    mins, maxs, p25s, p75s, medians = calc_stats(data_series)
    mins = [value / scale for value in mins]
    maxs = [value / scale for value in maxs]
    p25s = [value / scale for value in p25s]
    p75s = [value / scale for value in p75s]
    medians = [value / scale for value in medians]

    ax.fill_between(steps, mins, maxs, alpha=0.15, color=LOAD_BALANCE_COLOR)
    ax.fill_between(steps, p25s, p75s, alpha=0.35, color=LOAD_BALANCE_COLOR)
    ax.plot(steps, medians, color=LOAD_BALANCE_COLOR, linewidth=1.0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_ylim(0, padded_axis_upper(maxs))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.tick_params(
        axis="both",
        which="major",
        pad=0.8,
        length=2.0,
        labelsize=TICK_LABEL_FONT_SIZE,
    )
    if not show_x_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)

    if use_k_suffix:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FuncFormatter(compact_block_tick))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))

    ax2 = ax.twinx()
    cv_values = calc_cv(data_series)
    ax2.plot(
        steps,
        cv_values,
        color=CV_COLOR,
        linewidth=0.9,
        linestyle="--",
    )
    ax2.set_ylim(0, cv_y_max)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(show_cv_ticks)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}%"))
    ax2.tick_params(
        axis="y",
        which="major",
        colors=CV_COLOR,
        labelsize=TICK_LABEL_FONT_SIZE,
        pad=0.8,
        length=2.0,
        right=show_cv_ticks,
        labelright=show_cv_ticks,
    )
    return ax, ax2


def flatten_numeric_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        flat: list[float] = []
        for item in value:
            flat.extend(flatten_numeric_list(item))
        return flat
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def pad_series(values: list[float], target_len: int) -> list[float]:
    padded = list(values[:target_len])
    if len(padded) < target_len:
        padded.extend([0.0] * (target_len - len(padded)))
    return padded


def resolve_window(
    total_steps: int,
    skip_steps: int,
    skip_tail_steps: int,
    center_steps: int,
) -> tuple[int, int]:
    if total_steps <= 0:
        return 0, 0

    effective_end = total_steps - max(0, skip_tail_steps)
    if effective_end <= 0:
        return 0, 0

    effective_start = min(max(0, skip_steps), effective_end)
    available_steps = effective_end - effective_start
    if center_steps > 0 and center_steps < available_steps:
        start = effective_start + (available_steps - center_steps) // 2
        return start, start + center_steps
    return effective_start, effective_end


def chunk_sums(values: list[float], chunk_size: int) -> list[float]:
    chunk_size = max(1, chunk_size)
    return [
        float(sum(values[start : start + chunk_size]))
        for start in range(0, len(values), chunk_size)
    ]


def normalize_free_blocks(raw_value, group_size: int) -> tuple[list[float], int]:
    flat_values = flatten_numeric_list(raw_value)
    if not flat_values:
        return [], 0
    return chunk_sums(flat_values, group_size), len(flat_values)


def normalize_waiting_row(
    raw_value,
    num_groups: int,
    num_ranks: int,
    group_size: int,
) -> list[float]:
    if num_groups <= 0:
        return []

    target_len = max(num_ranks, num_groups * group_size)
    flat_values = flatten_numeric_list(raw_value)
    if not flat_values:
        return [0.0] * target_len
    if len(flat_values) == num_ranks:
        return pad_series(flat_values, target_len)
    if num_groups == 1:
        return pad_series(flat_values, target_len)
    if len(flat_values) == num_groups:
        row = [0.0] * target_len
        for group_idx, value in enumerate(pad_series(flat_values, num_groups)):
            row[group_idx * group_size] = float(value)
        return row
    if len(flat_values) == 1 and num_groups > 1:
        row = [0.0] * target_len
        for group_idx in range(num_groups):
            row[group_idx * group_size] = float(flat_values[0])
        return row
    if len(flat_values) % num_groups == 0:
        values_per_group = max(1, len(flat_values) // num_groups)
        row = [0.0] * target_len
        for group_idx in range(num_groups):
            group_values = pad_series(
                flat_values[
                    group_idx * values_per_group : (group_idx + 1) * values_per_group
                ],
                group_size,
            )
            start = group_idx * group_size
            row[start : start + group_size] = group_values[:group_size]
        return row
    return pad_series(flat_values, target_len)


def normalize_waiting_metric(
    raw_value,
    num_groups: int,
    num_ranks: int,
    group_size: int,
) -> list[float]:
    return chunk_sums(
        normalize_waiting_row(raw_value, num_groups, num_ranks, group_size),
        group_size,
    )[:num_groups]


def parse_nano_highload_log(
    path: Path,
    skip_steps: int,
    skip_tail_steps: int,
    center_steps: int,
    group_size: int,
) -> HighLoadParsedSeries:
    if not path.is_file():
        raise FileNotFoundError(f"Nano log not found: {path}")

    group_size = max(1, group_size)
    max_blocks_per_rank: int | None = None
    inferred_max_blocks = 0
    records: list[HighLoadStepRecord] = []

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line)

            if max_blocks_per_rank is None:
                kv_match = NUM_KVCACHE_BLOCKS_RE.search(line)
                if kv_match is not None:
                    max_blocks_per_rank = int(kv_match.group(1))

            if "step - {" not in line:
                continue

            _, _, payload = line.partition("step - ")
            payload = payload.strip()
            if not payload.startswith("{"):
                continue

            try:
                step_data = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue

            if step_data.get("mode") != "decode":
                continue

            free_blocks_per_group, num_ranks = normalize_free_blocks(
                step_data.get("free_blocks", []), group_size
            )
            if not free_blocks_per_group:
                continue

            flat_free_blocks = flatten_numeric_list(step_data.get("free_blocks", []))
            inferred_max_blocks = max(
                inferred_max_blocks,
                int(max(flat_free_blocks, default=0.0)),
            )
            num_groups = len(free_blocks_per_group)
            waiting_head_blocks_per_rank = normalize_waiting_row(
                step_data.get("waiting_head_blocks", 0),
                num_groups=num_groups,
                num_ranks=num_ranks,
                group_size=group_size,
            )

            records.append(
                HighLoadStepRecord(
                    free_blocks_per_group=free_blocks_per_group,
                    waiting_head_blocks_per_group=normalize_waiting_metric(
                        waiting_head_blocks_per_rank,
                        num_groups=num_groups,
                        num_ranks=num_ranks,
                        group_size=group_size,
                    ),
                    waiting_head_blocks_per_rank=waiting_head_blocks_per_rank,
                    total_ranks=num_ranks,
                )
            )

    if not records:
        raise ValueError(f"No decode steps parsed from Nano log: {path}")

    max_blocks = float(max_blocks_per_rank or inferred_max_blocks)
    total_steps = len(records)
    selected_start, selected_end = resolve_window(
        total_steps,
        skip_steps,
        skip_tail_steps,
        center_steps,
    )
    selected_records = records[selected_start:selected_end]
    if not selected_records:
        raise ValueError(f"Window selection removed every decode step from {path}")

    num_ranks = max(record.total_ranks for record in selected_records)
    num_groups = max(len(record.free_blocks_per_group) for record in selected_records)
    group_free_blocks = [[] for _ in range(num_groups)]
    group_waiting_head_blocks = [[] for _ in range(num_groups)]
    group_waiting_head_blocks_by_rank = [
        [[] for _ in range(group_size)] for _ in range(num_groups)
    ]

    for record in selected_records:
        padded_free = pad_series(record.free_blocks_per_group, num_groups)
        padded_waiting = pad_series(record.waiting_head_blocks_per_group, num_groups)
        padded_waiting_by_rank = pad_series(
            record.waiting_head_blocks_per_rank,
            num_groups * group_size,
        )
        for group_idx in range(num_groups):
            group_free_blocks[group_idx].append(padded_free[group_idx])
            group_waiting_head_blocks[group_idx].append(padded_waiting[group_idx])
            start = group_idx * group_size
            group_waiting_by_rank = padded_waiting_by_rank[start : start + group_size]
            for local_rank_idx, value in enumerate(group_waiting_by_rank):
                group_waiting_head_blocks_by_rank[group_idx][local_rank_idx].append(value)

    return HighLoadParsedSeries(
        path=path,
        steps=list(range(1, len(selected_records) + 1)),
        group_free_blocks=group_free_blocks,
        group_waiting_head_blocks=group_waiting_head_blocks,
        group_waiting_head_blocks_by_rank=group_waiting_head_blocks_by_rank,
        max_blocks_per_rank=max_blocks,
        num_ranks=num_ranks,
        num_groups=num_groups,
        group_size=group_size,
        total_steps=total_steps,
        selected_start=selected_start,
        selected_end=selected_end,
    )


def parse_vllm_highload_log(
    path: Path,
    skip_steps: int,
    skip_tail_steps: int,
    center_steps: int,
    block_size: float,
    group_size: int,
) -> HighLoadParsedSeries:
    if not path.is_file():
        raise FileNotFoundError(f"vLLM log not found: {path}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    group_size = max(1, group_size)
    kv_token_sizes: list[int] = []
    managed_engines: int | None = None
    usable_gpu_blocks_per_engine: float | None = None
    metric_batches: list[dict[int, tuple[float, float]]] = []
    current_batch: dict[int, tuple[float, float]] = {}
    previous_engine_id: int | None = None

    def flush_current_batch() -> None:
        nonlocal current_batch, previous_engine_id
        if current_batch:
            metric_batches.append(current_batch)
        current_batch = {}
        previous_engine_id = None

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line)

            kv_match = GPU_KV_SIZE_RE.search(line)
            if kv_match is not None:
                kv_token_sizes.append(int(kv_match.group(1).replace(",", "")))

            capacity_match = VLLM_CAPACITY_RE.search(line)
            if capacity_match is not None:
                managed_engines = int(capacity_match.group("managed_engines"))
                usable_gpu_blocks = float(
                    capacity_match.group("usable_gpu_blocks").replace(",", "")
                )
                usable_gpu_blocks_per_engine = usable_gpu_blocks / managed_engines

            metric_match = VLLM_ENGINE_METRIC_RE.search(line)
            if metric_match is None:
                continue

            engine_id = int(metric_match.group("engine_id"))
            waiting_head_tokens = float(metric_match.group("waiting_head_tokens"))
            gpu_kv_cache_usage_pct = float(
                metric_match.group("gpu_kv_cache_usage_pct")
            )

            if current_batch and (
                engine_id in current_batch
                or (
                    previous_engine_id is not None
                    and engine_id <= previous_engine_id
                )
            ):
                flush_current_batch()

            current_batch[engine_id] = (
                waiting_head_tokens,
                gpu_kv_cache_usage_pct,
            )
            previous_engine_id = engine_id

    flush_current_batch()

    if not metric_batches:
        raise ValueError(f"No engine stats parsed from vLLM log: {path}")
    if kv_token_sizes:
        max_blocks_per_rank = min(kv_token_sizes) / block_size
    elif usable_gpu_blocks_per_engine is not None:
        max_blocks_per_rank = usable_gpu_blocks_per_engine
    else:
        raise ValueError(f"No KV cache capacity metadata found in vLLM log: {path}")

    total_steps = len(metric_batches)
    selected_start, selected_end = resolve_window(
        total_steps,
        skip_steps,
        skip_tail_steps,
        center_steps,
    )
    selected_batches = metric_batches[selected_start:selected_end]
    if not selected_batches:
        raise ValueError(f"Window selection removed every vLLM stats step from {path}")

    inferred_num_ranks = max(
        max(engine_stats.keys(), default=-1) for engine_stats in metric_batches
    ) + 1
    num_ranks = max(inferred_num_ranks, managed_engines or 0)
    if num_ranks <= 0:
        raise ValueError(f"Could not determine vLLM engine count from {path}")

    num_groups = len(chunk_sums([0.0] * num_ranks, group_size))
    group_free_blocks = [[] for _ in range(num_groups)]
    group_waiting_head_blocks = [[] for _ in range(num_groups)]
    group_waiting_head_blocks_by_rank = [
        [[] for _ in range(group_size)] for _ in range(num_groups)
    ]

    for engine_stats in selected_batches:
        free_row = [max_blocks_per_rank] * num_ranks
        waiting_row = [0.0] * num_ranks

        for engine_id, (waiting_head_tokens, gpu_kv_cache_usage_pct) in engine_stats.items():
            if 0 <= engine_id < num_ranks:
                waiting_row[engine_id] = waiting_head_tokens / block_size
                free_row[engine_id] = max_blocks_per_rank * max(
                    0.0,
                    1.0 - gpu_kv_cache_usage_pct / 100.0,
                )

        free_per_group = chunk_sums(free_row, group_size)
        waiting_per_group = chunk_sums(waiting_row, group_size)
        for group_idx in range(num_groups):
            group_free_blocks[group_idx].append(free_per_group[group_idx])
            group_waiting_head_blocks[group_idx].append(waiting_per_group[group_idx])
            start = group_idx * group_size
            group_waiting_by_rank = pad_series(
                waiting_row[start : start + group_size],
                group_size,
            )
            for local_rank_idx, value in enumerate(group_waiting_by_rank):
                group_waiting_head_blocks_by_rank[group_idx][local_rank_idx].append(value)

    return HighLoadParsedSeries(
        path=path,
        steps=list(range(1, len(selected_batches) + 1)),
        group_free_blocks=group_free_blocks,
        group_waiting_head_blocks=group_waiting_head_blocks,
        group_waiting_head_blocks_by_rank=group_waiting_head_blocks_by_rank,
        max_blocks_per_rank=max_blocks_per_rank,
        num_ranks=num_ranks,
        num_groups=num_groups,
        group_size=group_size,
        total_steps=total_steps,
        selected_start=selected_start,
        selected_end=selected_end,
    )


def build_group_palette(group_size: int) -> list[tuple[float, float, float, float]]:
    cmap = matplotlib.colormaps["GnBu"]
    return [cmap(x) for x in np.linspace(0.40, 0.88, group_size)]


def plot_highload_axis(
    ax,
    steps: list[int],
    free_values: list[float],
    waiting_head: list[float],
    waiting_head_by_rank: list[list[float]],
    title: str,
    shared_y_max: float,
    show_x_tick_labels: bool,
) -> None:
    step_values = np.asarray(steps, dtype=float)
    free_series = np.asarray(free_values, dtype=float)
    waiting_head_series = np.asarray(waiting_head, dtype=float)
    waiting_stack = np.asarray(waiting_head_by_rank, dtype=float)
    if waiting_stack.ndim == 1:
        waiting_stack = waiting_stack[np.newaxis, :]

    if waiting_stack.size > 0 and waiting_stack.shape[-1] == len(step_values):
        ax.stackplot(
            step_values,
            waiting_stack,
            colors=build_group_palette(waiting_stack.shape[0]),
            alpha=0.92,
            edgecolor="none",
            zorder=1,
        )

    ax.plot(
        step_values,
        waiting_head_series,
        color=HOL_OUTLINE_COLOR,
        linewidth=0.8,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        step_values,
        free_series,
        color=FREE_LINE_COLOR,
        linewidth=1.2,
        solid_capstyle="round",
        zorder=3,
    )

    ax.set_title(title, fontweight="bold", pad=2.0)
    if len(step_values) > 0:
        right_limit = float(step_values[-1]) if len(step_values) > 1 else float(step_values[0]) + 1.0
        ax.set_xlim(float(step_values[0]), right_limit)
    ax.set_ylim(0.0, shared_y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_block_tick))
    ax.tick_params(
        axis="both",
        which="major",
        pad=0.8,
        length=2.0,
        labelsize=TICK_LABEL_FONT_SIZE,
    )
    if not show_x_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color=GRID_COLOR, alpha=1.0)
    ax.grid(visible=False, axis="x")


def resolve_output_base(output: Path | None) -> Path:
    if output is None:
        return Path(__file__).resolve().parent / "fig14"
    resolved = output.resolve()
    return resolved.with_suffix("") if resolved.suffix else resolved


def save_figure(fig, output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    return pdf_path, png_path


def union_bbox(axes: list[object]) -> tuple[float, float, float, float]:
    x0 = min(ax.get_position().x0 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    return x0, y0, x1, y1


def resolve_group_index(
    nano: HighLoadParsedSeries,
    vllm: HighLoadParsedSeries,
    group_index_arg: str,
) -> int:
    group_count = min(nano.num_groups, vllm.num_groups)
    if group_count <= 0:
        raise ValueError("No shared high-load groups found between Nano and vLLM.")

    if group_index_arg != "auto":
        group_index = int(group_index_arg)
        if not 0 <= group_index < group_count:
            raise ValueError(
                f"hl-group-index={group_index} is out of range [0, {group_count - 1}]"
            )
        return group_index

    scored_groups: list[tuple[float, int]] = []
    for group_index in range(group_count):
        waiting_peak = max(
            max(nano.group_waiting_head_blocks[group_index], default=0.0),
            max(vllm.group_waiting_head_blocks[group_index], default=0.0),
        )
        scored_groups.append((waiting_peak, group_index))
    return max(scored_groups)[1]


def format_right_caption(template: str, start_gpu: int, end_gpu: int) -> str:
    return template.format(gpu_range=f"{start_gpu:02d}-{end_gpu:02d}")


def resolve_partial_group_index(
    series: list[HighLoadParsedSeries],
    group_index_arg: str,
) -> int:
    group_count = min(item.num_groups for item in series)
    if group_count <= 0:
        raise ValueError("No high-load groups found in the available logs.")
    if group_index_arg != "auto":
        group_index = int(group_index_arg)
        if not 0 <= group_index < group_count:
            raise ValueError(
                f"hl-group-index={group_index} is out of range "
                f"[0, {group_count - 1}]"
            )
        return group_index

    scores: list[tuple[float, float, int]] = []
    for group_index in range(group_count):
        waiting_peak = max(
            max(item.group_waiting_head_blocks[group_index], default=0.0)
            for item in series
        )
        used_peak = max(
            item.group_capacity_blocks
            - min(item.group_free_blocks[group_index], default=0.0)
            for item in series
        )
        scores.append((waiting_peak, used_peak, -group_index))
    return -max(scores)[2]


def plot_partial_figure(
    args: argparse.Namespace,
    output_base: Path,
) -> tuple[Path, Path]:
    lb_specs = [
        LoadBalanceLogSpec(
            name="Nano DCP",
            path=args.lb_nano_log,
            log_type="nano",
            center_steps=args.lb_nano_center_steps,
        ),
        LoadBalanceLogSpec(
            name="vLLM DP",
            path=args.lb_vllm_log,
            log_type="vllm",
            center_steps=args.lb_vllm_center_steps,
        ),
        LoadBalanceLogSpec(
            name="vLLM LeastCache DP",
            path=args.lb_vllm_least_cache_log,
            log_type="vllm",
            center_steps=args.lb_vllm_least_cache_center_steps,
        ),
    ]
    lb_specs = [spec for spec in lb_specs if spec.path is not None]

    parsed_lb_specs: list[tuple[LoadBalanceLogSpec, LoadBalanceParsedData]] = []
    for spec in lb_specs:
        parsed = (
            parse_nano_load_balance_log(spec.path, spec.center_steps)
            if spec.log_type == "nano"
            else parse_vllm_load_balance_log(spec.path, spec.center_steps)
        )
        print(
            f"[load-balance] {spec.name}: center={spec.center_steps} "
            f"window=[{parsed.center_start + 1}, {parsed.center_end}] "
            f"steps={len(parsed.steps)} ranks={parsed.num_ranks} "
            f"batch_cv_mean={calc_mean_cv(parsed.batch_series):.2f}% "
            f"used_cv_mean={calc_mean_cv(parsed.used_series):.2f}%"
        )
        parsed_lb_specs.append((spec, parsed))

    parsed_hl_specs: list[tuple[str, HighLoadParsedSeries]] = []
    if args.hl_nano_log is not None:
        parsed_hl_specs.append(
            (
                args.hl_nano_label,
                parse_nano_highload_log(
                    args.hl_nano_log,
                    args.hl_skip_steps,
                    args.hl_skip_tail_steps,
                    args.hl_center_steps,
                    args.hl_group_size,
                ),
            )
        )
    if args.hl_vllm_log is not None:
        parsed_hl_specs.append(
            (
                args.hl_vllm_label,
                parse_vllm_highload_log(
                    args.hl_vllm_log,
                    args.hl_skip_steps,
                    args.hl_skip_tail_steps,
                    args.hl_center_steps,
                    args.hl_vllm_block_size,
                    args.hl_group_size,
                ),
            )
        )

    if not parsed_lb_specs or not parsed_hl_specs:
        raise ValueError(
            "Partial Fig. 14 plotting needs at least one load-balance log and "
            "one high-load log in the selected run."
        )

    highload_series = [parsed for _, parsed in parsed_hl_specs]
    group_index = resolve_partial_group_index(
        highload_series,
        str(args.hl_group_index),
    )
    max_visible_ranks = min(parsed.num_ranks for parsed in highload_series)
    start_gpu = group_index * args.hl_group_size
    end_gpu = min(max_visible_ranks, start_gpu + args.hl_group_size) - 1
    print(
        f"[high-load] selected group={group_index} "
        f"gpu={start_gpu:02d}-{end_gpu:02d} "
        + " ".join(
            f"{label}={len(parsed.steps)}steps"
            for label, parsed in parsed_hl_specs
        )
    )

    batch_cv_y_max = padded_axis_upper(
        [
            value
            for _, parsed in parsed_lb_specs
            for value in calc_cv(parsed.batch_series)
        ]
    )
    used_cv_y_max = padded_axis_upper(
        [
            value
            for _, parsed in parsed_lb_specs
            for value in calc_cv(parsed.used_series)
        ]
    )
    hl_shared_y_max = max(
        max(
            max(parsed.group_free_blocks[group_index], default=0.0),
            max(parsed.group_waiting_head_blocks[group_index], default=0.0),
            parsed.group_capacity_blocks,
        )
        for _, parsed in parsed_hl_specs
    ) * 1.03

    left_width = 1.72 * len(parsed_lb_specs)
    figure_width = max(4.55, left_width + 0.48 + 1.82)
    fig = plt.figure(figsize=(figure_width, FIGURE_HEIGHT_INCHES), dpi=300)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[left_width, 0.48, 1.82],
        left=0.075 if len(parsed_lb_specs) > 1 else 0.105,
        right=0.98,
        top=0.78,
        bottom=0.29,
        wspace=0.0,
    )
    left_grid = GridSpecFromSubplotSpec(
        2,
        len(parsed_lb_specs),
        subplot_spec=outer[0, 0],
        wspace=0.16,
        hspace=0.24,
    )
    right_grid = GridSpecFromSubplotSpec(
        len(parsed_hl_specs),
        1,
        subplot_spec=outer[0, 2],
        hspace=0.30,
    )

    left_axes: list[object] = []
    for col, (spec, parsed) in enumerate(parsed_lb_specs):
        top_ax = fig.add_subplot(left_grid[0, col])
        bottom_ax = fig.add_subplot(left_grid[1, col])
        left_axes.extend([top_ax, bottom_ax])
        top_ax.set_title(
            LOAD_BALANCE_TITLES.get(spec.name, spec.name),
            fontweight="bold",
            pad=2.0,
        )
        if col == 0:
            top_ax.set_ylabel(LOAD_BALANCE_ROW_LABELS[0], labelpad=0.0)
            bottom_ax.set_ylabel(LOAD_BALANCE_ROW_LABELS[1], labelpad=0.0)
        plot_load_balance_axis(
            top_ax,
            parsed.steps,
            parsed.batch_series,
            scale=1.0,
            use_k_suffix=False,
            show_x_tick_labels=False,
            cv_y_max=batch_cv_y_max,
            show_cv_ticks=(col == len(parsed_lb_specs) - 1),
        )
        plot_load_balance_axis(
            bottom_ax,
            parsed.steps,
            parsed.used_series,
            scale=1.0,
            use_k_suffix=True,
            show_x_tick_labels=True,
            cv_y_max=used_cv_y_max,
            show_cv_ticks=(col == len(parsed_lb_specs) - 1),
        )

    right_axes: list[object] = []
    for row, (label, parsed) in enumerate(parsed_hl_specs):
        ax = fig.add_subplot(right_grid[row, 0])
        right_axes.append(ax)
        plot_highload_axis(
            ax,
            parsed.steps,
            parsed.group_free_blocks[group_index],
            parsed.group_waiting_head_blocks[group_index],
            parsed.group_waiting_head_blocks_by_rank[group_index],
            label,
            hl_shared_y_max,
            show_x_tick_labels=(row == len(parsed_hl_specs) - 1),
        )
        ax.set_ylabel("Blocks", labelpad=2.0)

    left_x0, left_y0, left_x1, _ = union_bbox(left_axes)
    right_x0, right_y0, right_x1, _ = union_bbox(right_axes)
    left_center_x = (left_x0 + left_x1) / 2.0
    right_center_x = (right_x0 + right_x1) / 2.0
    fig.legend(
        [
            Patch(facecolor=LOAD_BALANCE_COLOR, alpha=0.15),
            Patch(facecolor=LOAD_BALANCE_COLOR, alpha=0.35),
            Line2D([0], [0], color=LOAD_BALANCE_COLOR, linewidth=1.0),
            Line2D([0], [0], color=CV_COLOR, linewidth=0.9, linestyle="--"),
        ],
        ["Min-Max", "IQR", "Median", "CV"],
        loc="upper center",
        bbox_to_anchor=(left_center_x, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=0.55,
        handlelength=1.1,
        handletextpad=0.35,
    )
    fig.legend(
        [
            Line2D([0], [0], color=FREE_LINE_COLOR, linewidth=1.2),
            Patch(
                facecolor=build_group_palette(max(args.hl_group_size, 1))[
                    max(args.hl_group_size, 1) // 2
                ],
                edgecolor="none",
                alpha=0.92,
            ),
        ],
        ["Free", "HoL Demand"],
        loc="upper center",
        bbox_to_anchor=(right_center_x, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.2,
        handletextpad=0.4,
    )
    step_y = min(left_y0, right_y0) - 0.09
    caption_y = step_y - 0.06
    fig.text(left_center_x, step_y, "Step", ha="center", va="center")
    fig.text(right_center_x, step_y, "Step", ha="center", va="center")
    fig.text(
        left_center_x,
        caption_y,
        args.left_caption,
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
    )
    fig.text(
        right_center_x,
        caption_y,
        format_right_caption(args.right_caption, start_gpu, end_gpu),
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
    )

    pdf_path, png_path = save_figure(fig, output_base)
    plt.close(fig)
    print(f"[done] partial plot saved to {pdf_path}")
    print(f"[done] partial plot saved to {png_path}")
    return pdf_path, png_path


def plot_figure(args: argparse.Namespace, output_base: Path) -> tuple[Path, Path]:
    if getattr(args, "missing_log_options", ()):
        print(
            "[partial] unavailable logs: "
            + ", ".join(args.missing_log_options)
        )
        return plot_partial_figure(args, output_base)

    lb_specs = [
        LoadBalanceLogSpec(
            name="Nano DCP",
            path=args.lb_nano_log,
            log_type="nano",
            center_steps=args.lb_nano_center_steps,
        ),
        LoadBalanceLogSpec(
            name="vLLM DP",
            path=args.lb_vllm_log,
            log_type="vllm",
            center_steps=args.lb_vllm_center_steps,
        ),
        LoadBalanceLogSpec(
            name="vLLM LeastCache DP",
            path=args.lb_vllm_least_cache_log,
            log_type="vllm",
            center_steps=args.lb_vllm_least_cache_center_steps,
        ),
    ]

    parsed_lb_specs: list[tuple[LoadBalanceLogSpec, LoadBalanceParsedData]] = []
    for spec in lb_specs:
        parsed = (
            parse_nano_load_balance_log(spec.path, spec.center_steps)
            if spec.log_type == "nano"
            else parse_vllm_load_balance_log(spec.path, spec.center_steps)
        )
        batch_cv_mean = calc_mean_cv(parsed.batch_series)
        used_cv_mean = calc_mean_cv(parsed.used_series)
        print(
            f"[load-balance] {spec.name}: center={spec.center_steps} "
            f"window=[{parsed.center_start + 1}, {parsed.center_end}] "
            f"steps={len(parsed.steps)} ranks={parsed.num_ranks} "
            f"batch_cv_mean={batch_cv_mean:.2f}% "
            f"used_cv_mean={used_cv_mean:.2f}%"
        )
        parsed_lb_specs.append((spec, parsed))

    parsed_hl_nano = parse_nano_highload_log(
        args.hl_nano_log,
        args.hl_skip_steps,
        args.hl_skip_tail_steps,
        args.hl_center_steps,
        args.hl_group_size,
    )
    parsed_hl_vllm = parse_vllm_highload_log(
        args.hl_vllm_log,
        args.hl_skip_steps,
        args.hl_skip_tail_steps,
        args.hl_center_steps,
        args.hl_vllm_block_size,
        args.hl_group_size,
    )

    group_index = resolve_group_index(
        parsed_hl_nano,
        parsed_hl_vllm,
        str(args.hl_group_index),
    )
    max_visible_ranks = min(parsed_hl_nano.num_ranks, parsed_hl_vllm.num_ranks)
    start_gpu = group_index * args.hl_group_size
    end_gpu = min(max_visible_ranks, start_gpu + args.hl_group_size) - 1
    print(
        f"[high-load] selected group={group_index} gpu={start_gpu:02d}-{end_gpu:02d} "
        f"nano_steps={len(parsed_hl_nano.steps)} vllm_steps={len(parsed_hl_vllm.steps)}"
    )

    batch_cv_y_max = padded_axis_upper(
        [
            value
            for _, parsed in parsed_lb_specs
            for value in calc_cv(parsed.batch_series)
        ]
    )
    used_cv_y_max = padded_axis_upper(
        [
            value
            for _, parsed in parsed_lb_specs
            for value in calc_cv(parsed.used_series)
        ]
    )
    hl_shared_y_max = max(
        max(parsed_hl_nano.group_free_blocks[group_index], default=0.0),
        max(parsed_hl_nano.group_waiting_head_blocks[group_index], default=0.0),
        max(parsed_hl_vllm.group_free_blocks[group_index], default=0.0),
        max(parsed_hl_vllm.group_waiting_head_blocks[group_index], default=0.0),
        parsed_hl_nano.group_capacity_blocks,
        parsed_hl_vllm.group_capacity_blocks,
        1.0,
    ) * 1.03

    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=300)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[4.95, 0.82, 1.83],
        left=0.07,
        right=0.975,
        top=0.80,
        bottom=0.28,
        wspace=0.0,
    )
    left_grid = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=outer[0, 0],
        wspace=0.16,
        hspace=0.24,
    )
    right_grid = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 2],
        hspace=0.30,
    )

    left_axes: list[object] = []
    left_top_axes: list[object] = []
    left_bottom_axes: list[object] = []
    left_handles: list[object] = []
    left_labels: list[str] = []

    for col, (spec, parsed) in enumerate(parsed_lb_specs):
        top_ax = fig.add_subplot(left_grid[0, col])
        bottom_ax = fig.add_subplot(left_grid[1, col])
        left_axes.extend([top_ax, bottom_ax])
        left_top_axes.append(top_ax)
        left_bottom_axes.append(bottom_ax)

        top_ax.set_title(
            LOAD_BALANCE_TITLES.get(spec.name, spec.name),
            fontweight="bold",
            pad=2.0,
        )
        if col == 0:
            top_ax.set_ylabel(LOAD_BALANCE_ROW_LABELS[0], labelpad=0.0)
            bottom_ax.set_ylabel(LOAD_BALANCE_ROW_LABELS[1], labelpad=0.0)
            top_ax.yaxis.set_label_coords(LEFT_SHARED_YLABEL_X, 0.5)
            bottom_ax.yaxis.set_label_coords(LEFT_SHARED_YLABEL_X, 0.5)
        top_ax_main, top_ax_cv = plot_load_balance_axis(
            top_ax,
            parsed.steps,
            parsed.batch_series,
            scale=1.0,
            use_k_suffix=False,
            show_x_tick_labels=False,
            cv_y_max=batch_cv_y_max,
            show_cv_ticks=(col == len(parsed_lb_specs) - 1),
        )
        plot_load_balance_axis(
            bottom_ax,
            parsed.steps,
            parsed.used_series,
            scale=1.0,
            use_k_suffix=True,
            show_x_tick_labels=True,
            cv_y_max=used_cv_y_max,
            show_cv_ticks=(col == len(parsed_lb_specs) - 1),
        )

        if col == 0:
            left_handles.extend(
                [
                    Patch(facecolor=LOAD_BALANCE_COLOR, alpha=0.15),
                    Patch(facecolor=LOAD_BALANCE_COLOR, alpha=0.35),
                    Line2D([0], [0], color=LOAD_BALANCE_COLOR, linewidth=1.0),
                    Line2D(
                        [0],
                        [0],
                        color=CV_COLOR,
                        linewidth=0.9,
                        linestyle="--",
                    ),
                ]
            )
            left_labels.extend(["Min-Max", "IQR", "Median", "CV"])

    right_axes: list[object] = []
    hl_nano_ax = fig.add_subplot(right_grid[0, 0])
    hl_vllm_ax = fig.add_subplot(right_grid[1, 0])
    right_axes.extend([hl_nano_ax, hl_vllm_ax])

    plot_highload_axis(
        hl_nano_ax,
        parsed_hl_nano.steps,
        parsed_hl_nano.group_free_blocks[group_index],
        parsed_hl_nano.group_waiting_head_blocks[group_index],
        parsed_hl_nano.group_waiting_head_blocks_by_rank[group_index],
        args.hl_nano_label,
        hl_shared_y_max,
        show_x_tick_labels=False,
    )
    plot_highload_axis(
        hl_vllm_ax,
        parsed_hl_vllm.steps,
        parsed_hl_vllm.group_free_blocks[group_index],
        parsed_hl_vllm.group_waiting_head_blocks[group_index],
        parsed_hl_vllm.group_waiting_head_blocks_by_rank[group_index],
        args.hl_vllm_label,
        hl_shared_y_max,
        show_x_tick_labels=True,
    )
    hl_nano_ax.set_ylabel("Blocks", labelpad=2.0)
    hl_vllm_ax.set_ylabel("Blocks", labelpad=2.0)

    right_handles = [
        Line2D([0], [0], color=FREE_LINE_COLOR, linewidth=1.2),
        Patch(
            facecolor=build_group_palette(max(args.hl_group_size, 1))[
                max(args.hl_group_size, 1) // 2
            ],
            edgecolor="none",
            alpha=0.92,
        ),
    ]
    right_labels = ["Free", "HoL Demand"]

    left_x0, left_y0, left_x1, left_y1 = union_bbox(left_axes)
    right_x0, right_y0, right_x1, right_y1 = union_bbox(right_axes)
    left_center_x = (left_x0 + left_x1) / 2.0
    right_center_x = (right_x0 + right_x1) / 2.0

    fig.legend(
        left_handles,
        left_labels,
        loc="upper center",
        bbox_to_anchor=(left_center_x, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.2,
        handletextpad=0.4,
    )
    fig.legend(
        right_handles,
        right_labels,
        loc="upper center",
        bbox_to_anchor=(right_center_x, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.2,
        handletextpad=0.4,
    )
    step_y = min(left_y0, right_y0) - 0.09
    caption_y = step_y - 0.06
    fig.text(left_center_x, step_y, "Step", ha="center", va="center")
    fig.text(right_center_x, step_y, "Step", ha="center", va="center")
    fig.text(
        left_center_x,
        caption_y,
        args.left_caption,
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
    )
    fig.text(
        right_center_x,
        caption_y,
        format_right_caption(args.right_caption, start_gpu, end_gpu),
        ha="center",
        va="center",
        fontsize=CAPTION_FONT_SIZE,
    )

    pdf_path, png_path = save_figure(fig, output_base)
    plt.close(fig)
    print(f"[done] plot saved to {pdf_path}")
    print(f"[done] plot saved to {png_path}")
    return pdf_path, png_path


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    output_base = resolve_output_base(args.output)
    plot_figure(args, output_base)


if __name__ == "__main__":
    main()

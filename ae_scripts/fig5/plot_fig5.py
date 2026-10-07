#!/usr/bin/env python3
"""Render Fig. 5 from rank latency data and HoL data.

This plotting implementation is intentionally local to ``fig5``.  It does not
load code, fonts, or configuration from the original paper plotting tree.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig5")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, MaxNLocator


DEFAULT_USABLE_GPU_BLOCKS_TOTAL = 531_360
RUN_DIR_YEAR_RE = re.compile(r"^(?P<year>\d{4})\d{4}-\d{6}$")
CAPACITY_RE = re.compile(
    r"poisson_gpu_kv_cache_capacity .*?"
    r"managed_engines=(?P<managed_engines>\d+).*?"
    r"usable_gpu_blocks=(?P<usable_gpu_blocks>[\d,]+).*?"
    r"block_size=(?P<reported_block_size>\d+)"
)
GPU_KV_CACHE_SIZE_RE = re.compile(
    r"GPU KV cache size:\s*(?P<tokens>[\d,]+)\s*tokens"
)
METRIC_RE = re.compile(
    r"INFO\s+(?P<timestamp>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine_id>\d+):.*?"
    r"Waiting head tokens:\s+(?P<waiting_head_tokens>\d+),\s+"
    r"GPU KV cache usage:\s*(?P<gpu_kv_cache_usage_pct>\d+(?:\.\d+)?)%"
)

COLORS = {
    "attention_bar": "#83A9C7",
    "deepep_bar": "#6E9BB8",
    "mean": "#D77918",
    "peak": "#A7373F",
    "gap": "#8F2D2D",
    "hol_fill": "#78AFC2",
    "hol_outline": "#2E627B",
    "free_line": "#D08B28",
    "text": "#30343A",
    "grid": "#D8DDE3",
    "spine": "#59606B",
}

@dataclass(frozen=True)
class Typography:
    tick_size: float
    label_size: float
    title_size: float
    annotation_size: float


@dataclass(frozen=True)
class CapacityInfo:
    managed_engines: int
    usable_gpu_blocks_total: int
    usable_gpu_blocks_per_engine: float
    reported_block_size: int | None
    initial_gpu_kv_tokens_per_engine: int | None


@dataclass(frozen=True)
class MetricRow:
    timestamp: datetime
    engine_id: int
    waiting_head_tokens: float
    gpu_kv_cache_usage_pct: float


@dataclass(frozen=True)
class TimeWindowInfo:
    source_start: datetime
    source_end: datetime
    selected_start: datetime
    selected_end: datetime
    requested_window_seconds: float
    applied: bool

    @property
    def selected_span_seconds(self) -> float:
        return (self.selected_end - self.selected_start).total_seconds()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the local Fig. 5 plot")
    parser.add_argument(
        "--attention-input",
        "--mla-input",
        dest="attention_input",
        type=Path,
        required=True,
        help="Attention per-rank latency CSV.",
    )
    parser.add_argument(
        "--deepep-input",
        type=Path,
        required=True,
        help="DeepEP per-rank latency CSV.",
    )
    parser.add_argument(
        "--hol-input",
        "--hol-log",
        dest="hol_input",
        type=Path,
        required=True,
        help=(
            "Raw vLLM frontend.log or a prepared CSV with time_s, "
            "total_hol_blocks, and total_free_blocks."
        ),
    )
    parser.add_argument(
        "--hol-engines",
        default="8-15",
        help='DP ids, for example "8-15" or "8,9,10,11,12,13,14,15".',
    )
    parser.add_argument("--hol-block-size", type=float, default=64.0)
    parser.add_argument("--hol-middle-window-seconds", type=float, default=800.0)
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--default-usable-gpu-blocks-total",
        type=int,
        default=DEFAULT_USABLE_GPU_BLOCKS_TOTAL,
    )
    parser.add_argument("--figure-width", type=float, default=7.0)
    parser.add_argument("--figure-height", type=float, default=1.05)
    parser.add_argument("--tick-fontsize", type=float, default=6.0)
    parser.add_argument("--label-fontsize", type=float, default=7.0)
    parser.add_argument("--title-fontsize", type=float, default=8.0)
    parser.add_argument("--annotation-fontsize", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    if args.hol_block_size <= 0:
        parser.error("--hol-block-size must be positive")
    if args.hol_middle_window_seconds < 0:
        parser.error("--hol-middle-window-seconds must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def configure_matplotlib(typography: Typography) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": get_plot_font_family(),
            "font.size": typography.label_size,
            "axes.titlesize": typography.title_size,
            "axes.labelsize": typography.label_size,
            "xtick.labelsize": typography.tick_size,
            "ytick.labelsize": typography.tick_size,
            "legend.fontsize": typography.title_size,
            "axes.edgecolor": COLORS["spine"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_engine_ids(spec: str) -> list[int]:
    normalized = spec.strip()
    if not normalized:
        raise ValueError("empty --hol-engines specification")
    if "-" in normalized and "," not in normalized:
        start_text, end_text = normalized.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"invalid engine range: {spec}")
        return list(range(start, end + 1))
    engine_ids = sorted(
        {int(token.strip()) for token in normalized.split(",") if token.strip()}
    )
    if not engine_ids:
        raise ValueError(f"invalid engine list: {spec}")
    return engine_ids


def load_rank_latency_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Rank latency CSV not found: {path}")
    rank_rows: list[tuple[int, float]] = []
    average: float | None = None
    with path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing = {"rank", "time_us"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            if row["rank"].strip() == "avg":
                average = float(row["time_us"])
            else:
                rank_rows.append((int(row["rank"]), float(row["time_us"])))
    rank_rows.sort()
    if not rank_rows or average is None:
        raise ValueError(f"{path.name} needs rank rows and one avg row")
    expected = list(range(len(rank_rows)))
    if [rank for rank, _value in rank_rows] != expected:
        raise ValueError(f"{path.name} ranks are not contiguous from zero")
    return (
        np.array([rank for rank, _value in rank_rows], dtype=np.int32),
        np.array([value for _rank, value in rank_rows], dtype=np.float64),
        average,
    )


def load_hol_timeseries_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"time_s", "total_hol_blocks", "total_free_blocks"}
    rows: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        for row in reader:
            rows.append(
                (
                    float(row["time_s"]),
                    float(row["total_hol_blocks"]),
                    float(row["total_free_blocks"]),
                )
            )
    if not rows:
        raise ValueError(f"{path.name} contains no HoL samples")
    rows.sort(key=lambda row: row[0])
    if rows[0][0] != 0.0:
        raise ValueError(f"{path.name} must start at time_s=0")
    return tuple(
        np.array([row[index] for row in rows], dtype=np.float64)
        for index in range(3)
    )


def infer_year(log_path: Path, explicit_year: int | None) -> int:
    if explicit_year is not None:
        return explicit_year
    match = RUN_DIR_YEAR_RE.match(log_path.parent.name)
    return int(match.group("year")) if match else datetime.now().year


def parse_capacity_info(
    log_path: Path,
    fallback_managed_engines: int,
    fallback_usable_gpu_blocks_total: int,
    fallback_block_size: float | None = None,
) -> CapacityInfo:
    kv_tokens: int | None = None
    parsed: tuple[int, int, int] | None = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as input_file:
        for line in input_file:
            kv_match = GPU_KV_CACHE_SIZE_RE.search(line)
            if kv_tokens is None and kv_match:
                kv_tokens = int(kv_match.group("tokens").replace(",", ""))
            capacity_match = CAPACITY_RE.search(line)
            if parsed is None and capacity_match:
                parsed = (
                    int(capacity_match.group("managed_engines")),
                    int(capacity_match.group("usable_gpu_blocks").replace(",", "")),
                    int(capacity_match.group("reported_block_size")),
                )
            if parsed is not None and kv_tokens is not None:
                break
    if parsed is None:
        managed = fallback_managed_engines
        reported_block_size = None
        if kv_tokens is not None and fallback_block_size is not None:
            usable_per_engine = kv_tokens / fallback_block_size
            usable_total = int(round(usable_per_engine * managed))
            print(
                "Warning: aggregate capacity metadata missing; derived "
                f"usable_total={usable_total} from "
                f"{kv_tokens} tokens/engine, block_size={fallback_block_size:g}, "
                f"managed_engines={managed}."
            )
        else:
            usable_total = fallback_usable_gpu_blocks_total
            print(
                "Warning: capacity metadata missing; using fallback "
                f"usable_total={usable_total}, managed_engines={managed}."
            )
    else:
        managed, usable_total, reported_block_size = parsed
    return CapacityInfo(
        managed_engines=managed,
        usable_gpu_blocks_total=usable_total,
        usable_gpu_blocks_per_engine=usable_total / managed,
        reported_block_size=reported_block_size,
        initial_gpu_kv_tokens_per_engine=kv_tokens,
    )


def parse_metric_rows(log_path: Path, year: int) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as input_file:
        for line in input_file:
            match = METRIC_RE.search(line)
            if match is None:
                continue
            rows.append(
                MetricRow(
                    timestamp=datetime.strptime(
                        f"{year}-{match.group('timestamp')}",
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    engine_id=int(match.group("engine_id")),
                    waiting_head_tokens=float(match.group("waiting_head_tokens")),
                    gpu_kv_cache_usage_pct=float(match.group("gpu_kv_cache_usage_pct")),
                )
            )
    if not rows:
        raise ValueError(f"no engine metric rows found in {log_path}")
    return rows


def select_middle_time_window(
    rows: list[MetricRow], window_seconds: float
) -> tuple[list[MetricRow], TimeWindowInfo]:
    rows = sorted(rows, key=lambda row: (row.timestamp, row.engine_id))
    source_start, source_end = rows[0].timestamp, rows[-1].timestamp
    source_span = (source_end - source_start).total_seconds()
    if window_seconds <= 0 or source_span <= window_seconds:
        return rows, TimeWindowInfo(
            source_start,
            source_end,
            source_start,
            source_end,
            window_seconds,
            False,
        )
    margin = (source_span - window_seconds) / 2.0
    target_start = source_start + timedelta(seconds=margin)
    target_end = source_end - timedelta(seconds=margin)
    timestamps = sorted(
        {row.timestamp for row in rows if target_start <= row.timestamp <= target_end}
    )
    if not timestamps:
        return rows, TimeWindowInfo(
            source_start,
            source_end,
            source_start,
            source_end,
            window_seconds,
            False,
        )
    selected_start, selected_end = timestamps[0], timestamps[-1]
    selected = [row for row in rows if selected_start <= row.timestamp <= selected_end]
    return selected, TimeWindowInfo(
        source_start,
        source_end,
        selected_start,
        selected_end,
        window_seconds,
        True,
    )


def prepare_hol_totals(
    rows: list[MetricRow],
    engine_ids: list[int],
    capacity: CapacityInfo,
    hol_block_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_ids = set(engine_ids)
    by_timestamp: dict[datetime, list[MetricRow]] = {}
    for row in rows:
        if row.engine_id in selected_ids:
            by_timestamp.setdefault(row.timestamp, []).append(row)
    if not by_timestamp:
        raise ValueError(f"no metric rows found for engines {engine_ids}")

    current_hol = {engine_id: 0.0 for engine_id in engine_ids}
    current_free = {engine_id: 0.0 for engine_id in engine_ids}
    timestamps: list[datetime] = []
    total_hol: list[float] = []
    total_free: list[float] = []
    for timestamp in sorted(by_timestamp):
        for row in by_timestamp[timestamp]:
            current_hol[row.engine_id] = row.waiting_head_tokens / hol_block_size
            current_free[row.engine_id] = max(
                0.0,
                capacity.usable_gpu_blocks_per_engine
                * (1.0 - row.gpu_kv_cache_usage_pct / 100.0),
            )
        timestamps.append(timestamp)
        total_hol.append(sum(current_hol.values()))
        total_free.append(sum(current_free.values()))

    origin = timestamps[0]
    relative_times = [(timestamp - origin).total_seconds() for timestamp in timestamps]
    return (
        np.array(relative_times, dtype=np.float64),
        np.array(total_hol, dtype=np.float64),
        np.array(total_free, dtype=np.float64),
    )


def compact_block_tick(value: float, _position: int) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        kilo = value / 1000.0
        if abs(kilo) >= 100:
            return f"{kilo:.0f}k"
        return f"{kilo:.1f}".rstrip("0").rstrip(".") + "k"
    return f"{value:.0f}"


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(COLORS["spine"])
    axis.spines["bottom"].set_color(COLORS["spine"])
    axis.grid(axis="y", linewidth=0.55, color=COLORS["grid"])
    axis.grid(visible=False, axis="x")
    axis.tick_params(axis="x", pad=1.2, length=2.2)
    axis.tick_params(axis="y", pad=1.2, length=2.2)


def add_line_label(
    axis: plt.Axes,
    y_value: float,
    text: str,
    color: str,
    fontsize: float,
    y_offset_points: float = 0.0,
) -> None:
    transform = transforms.blended_transform_factory(axis.transAxes, axis.transData)
    axis.annotate(
        text,
        xy=(0.98, y_value),
        xycoords=transform,
        xytext=(0.0, y_offset_points),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.16},
    )


def add_panel_caption(
    axis: plt.Axes, text: str, typography: Typography, y: float = -0.33
) -> None:
    axis.text(
        0.5,
        y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        # Keep captions inside their panels at the paper's compact width.
        fontsize=typography.title_size * 0.78,
        fontweight="semibold",
        color=COLORS["text"],
    )


def plot_imbalance_panel(
    axis: plt.Axes,
    title: str,
    ranks: np.ndarray,
    values: np.ndarray,
    mean_value: float,
    bar_color: str,
    typography: Typography,
    arrow_x: float,
) -> None:
    peak_value = float(values.max())
    gap_pct = (peak_value - mean_value) / peak_value * 100.0
    axis.bar(ranks, values, width=0.86, color=bar_color, edgecolor="none", zorder=2)
    axis.axhline(
        mean_value,
        color=COLORS["mean"],
        linewidth=1.15,
        linestyle=(0, (4.0, 2.2)),
        zorder=3,
    )
    axis.axhline(
        peak_value,
        color=COLORS["peak"],
        linewidth=1.05,
        linestyle=(0, (1.4, 1.4)),
        zorder=3,
    )
    axis.annotate(
        "",
        xy=(arrow_x, mean_value),
        xytext=(arrow_x, peak_value),
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["gap"],
            "lw": 0.95,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 7.0,
        },
        zorder=4,
    )
    axis.text(
        arrow_x + 1.0,
        (
            peak_value * 0.72
            if gap_pct < 10.0
            else (mean_value + peak_value) / 2.0
        ),
        f"Ideal upper bound: {gap_pct:.0f}%",
        fontsize=typography.annotation_size,
        fontweight="semibold",
        color=COLORS["gap"],
        ha="left",
        va="center",
        zorder=5,
        bbox=(
            {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2}
            if gap_pct < 10.0
            else None
        ),
    )
    line_label_offset = 4.5 if gap_pct < 10.0 else 0.0
    add_line_label(
        axis,
        peak_value,
        "Peak",
        COLORS["peak"],
        typography.title_size,
        line_label_offset,
    )
    add_line_label(
        axis,
        mean_value,
        "Mean",
        COLORS["mean"],
        typography.title_size,
        -line_label_offset,
    )
    axis.set_xlabel("Rank", labelpad=0.1)
    axis.set_ylabel("Latency (us)")
    axis.set_xlim(-0.8, float(ranks.max()) + 0.8)
    axis.set_ylim(0.0, peak_value * 1.08)
    rank_count = len(ranks)
    tick_step = max(1, int(np.ceil(rank_count / 4)))
    rank_ticks = list(range(0, rank_count, tick_step))
    if rank_ticks[-1] != rank_count - 1:
        rank_ticks.append(rank_count - 1)
    axis.set_xticks(rank_ticks)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(axis)
    add_panel_caption(axis, title, typography)


def plot_hol_panel(
    axis: plt.Axes,
    relative_times: np.ndarray,
    total_hol: np.ndarray,
    total_free: np.ndarray,
    typography: Typography,
) -> None:
    hol_fill = axis.fill_between(
        relative_times,
        total_hol,
        color=COLORS["hol_fill"],
        alpha=0.38,
        linewidth=0.0,
        label="HoL Request Demand",
        zorder=2,
    )
    axis.plot(
        relative_times,
        total_hol,
        color=COLORS["hol_outline"],
        linewidth=1.15,
        zorder=3,
    )
    (free_line,) = axis.plot(
        relative_times,
        total_free,
        color=COLORS["free_line"],
        linewidth=1.55,
        label="Aggregate Free KV Blocks",
        zorder=4,
    )
    axis.set_xlabel("Time (s)", labelpad=0.1)
    axis.set_ylabel("# KV Blocks")
    right = max(float(relative_times[-1]), 1.0)
    axis.set_xlim(0.0, right)
    axis.set_ylim(0.0, max(float(total_hol.max()), float(total_free.max()), 1.0) * 1.08)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_formatter(FuncFormatter(compact_block_tick))
    style_axis(axis)
    legend = axis.legend(
        handles=[free_line, hol_fill],
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=1,
        frameon=True,
        borderaxespad=0.25,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.22,
        borderpad=0.1,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.78)
    legend.get_frame().set_edgecolor("none")
    legend.get_texts()[0].set_color(COLORS["free_line"])
    legend.get_texts()[1].set_color(COLORS["hol_outline"])
    add_panel_caption(
        axis,
        "(c) Head of line queueing demand vs free blocks",
        typography,
    )


def save_figure(
    figure: plt.Figure, output_base: Path, dpi: int
) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    figure.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    typography = Typography(
        tick_size=args.tick_fontsize,
        label_size=args.label_fontsize,
        title_size=args.title_fontsize,
        annotation_size=args.annotation_fontsize,
    )
    configure_matplotlib(typography)

    attention_ranks, attention_values, attention_mean = load_rank_latency_csv(
        args.attention_input.expanduser().resolve()
    )
    deepep_ranks, deepep_values, deepep_mean = load_rank_latency_csv(
        args.deepep_input.expanduser().resolve()
    )

    hol_input = args.hol_input.expanduser().resolve()
    if not hol_input.is_file():
        raise FileNotFoundError(f"HoL input not found: {hol_input}")
    if hol_input.suffix.lower() == ".csv":
        relative_times, total_hol, total_free = load_hol_timeseries_csv(hol_input)
        hol_summary = (
            f"prepared_csv={hol_input.name}, "
            f"window={float(relative_times[-1]):.1f}s"
        )
    else:
        engine_ids = parse_engine_ids(args.hol_engines)
        metric_rows = parse_metric_rows(hol_input, infer_year(hol_input, args.year))
        parsed_engine_ids = sorted({row.engine_id for row in metric_rows})
        capacity = parse_capacity_info(
            hol_input,
            fallback_managed_engines=len(parsed_engine_ids),
            fallback_usable_gpu_blocks_total=args.default_usable_gpu_blocks_total,
            fallback_block_size=args.hol_block_size,
        )
        metric_rows, time_window = select_middle_time_window(
            metric_rows, args.hol_middle_window_seconds
        )
        relative_times, total_hol, total_free = prepare_hol_totals(
            metric_rows, engine_ids, capacity, args.hol_block_size
        )
        hol_summary = (
            f"engines={engine_ids[0]}-{engine_ids[-1]}, "
            f"window={time_window.selected_span_seconds:.1f}s, "
            f"reported_block_size={capacity.reported_block_size}, "
            f"hol_block_size={args.hol_block_size:g}"
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(args.figure_width, args.figure_height),
        dpi=args.dpi,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.16]},
    )
    plot_imbalance_panel(
        axes[0],
        "(a) MoE Communication Imbalance",
        deepep_ranks,
        deepep_values,
        deepep_mean,
        COLORS["deepep_bar"],
        typography,
        arrow_x=2.0,
    )
    plot_imbalance_panel(
        axes[1],
        "(b) Attention Computation Imbalance",
        attention_ranks,
        attention_values,
        attention_mean,
        COLORS["attention_bar"],
        typography,
        arrow_x=3.0,
    )
    plot_hol_panel(axes[2], relative_times, total_hol, total_free, typography)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.26, top=0.95, wspace=0.2)

    pdf_path, png_path = save_figure(
        figure, args.output_base.expanduser().resolve(), args.dpi
    )
    plt.close(figure)
    attention_gap = (
        (float(attention_values.max()) - attention_mean)
        / float(attention_values.max())
        * 100.0
    )
    deepep_gap = (
        (float(deepep_values.max()) - deepep_mean)
        / float(deepep_values.max())
        * 100.0
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print(
        f"MLA mean-to-peak gap: {attention_gap:.1f}% | "
        f"DeepEP mean-to-peak gap: {deepep_gap:.1f}%"
    )
    print(
        "HoL panel: "
        f"{hol_summary}, "
        f"free_max={float(total_free.max()):.1f}, "
        f"hol_max={float(total_hol.max()):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

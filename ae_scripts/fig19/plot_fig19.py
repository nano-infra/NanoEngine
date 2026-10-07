#!/usr/bin/env python3
"""Reproduce Fig. 19 from materialized per-iteration NanoDeploy CSVs.

The upper panel shows the cluster-wide distribution of decode context-parallel
(CP) group sizes.  The lower panel summarizes Q + residual + LSE all-to-all
latency across data-parallel (DP) ranks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


CP_COLORS = {
    1: "#5B616B",
    # CP=2..8 share a muted, perceptually ordered green-to-blue ramp.
    2: "#E4F1EC",
    3: "#D3E7E0",
    4: "#BFDCD5",
    5: "#92C8BE",
    6: "#68AAA8",
    7: "#4F88A1",
    8: "#3F6788",
}
DP_COLORS = {
    0: "#0B3C5D",
    1: "#9A031E",
    2: "#006D77",
    3: "#5F0F40",
}


@dataclass(frozen=True)
class LatencyRow:
    global_run_count: int
    dp_rank: int
    total_us: float


@dataclass(frozen=True)
class CpRow:
    global_run_count: int
    dp_rank: int
    counts: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Fig. 19 from per_iter_latencies.csv and "
            "per_iter_cp_size_hist.csv."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="directory containing per_iter_latencies.csv and per_iter_cp_size_hist.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "fig19",
        help="output basename or a .pdf/.png path (default: fig19/fig19)",
    )
    parser.add_argument(
        "--tail-start-fraction",
        type=float,
        default=0.50,
        help="start at this position in the full run timeline (default: 0.50)",
    )
    parser.add_argument(
        "--trim-tail-fraction",
        type=float,
        default=0.06,
        help="remove this fraction of the full timeline from the end (default: 0.06)",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="explicit inclusive first decode iteration; overrides the start fraction",
    )
    parser.add_argument(
        "--end-iter",
        type=int,
        default=None,
        help="explicit inclusive last decode iteration; overrides tail trimming",
    )
    parser.add_argument(
        "--latency-style",
        choices=("lines", "band"),
        default="band",
        help=(
            "lower-panel rendering: one line per DP, or a Fig. 14-style "
            "min-max/median band (default: band)"
        ),
    )
    parser.add_argument(
        "--cp-style",
        choices=("lines", "stacked-bar"),
        default="stacked-bar",
        help=(
            "upper-panel CP>1 rendering: separate step lines or stacked bars; "
            "CP=1 remains a line on the right axis (default: stacked-bar)"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution (default: 220)")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    if fieldnames is None:
        fail(f"CSV has no header: {path}")
    missing = sorted(required - set(fieldnames))
    if missing:
        fail(f"{path} is missing required columns: {', '.join(missing)}")


def parse_int(value: str, path: Path, line_number: int, column: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid integer in {column}: {value!r}"
        ) from exc


def parse_float(value: str, path: Path, line_number: int, column: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid number in {column}: {value!r}"
        ) from exc
    if not math.isfinite(result) or result <= 0:
        fail(f"{path}:{line_number}: {column} must be finite and positive, got {value!r}")
    return result


def load_latency_rows(path: Path) -> list[LatencyRow]:
    if not path.is_file():
        fail(f"latency CSV not found: {path}")

    rows: list[LatencyRow] = []
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, {"global_run_count", "dp_rank", "total_us"})
        for line_number, raw in enumerate(reader, start=2):
            run = parse_int(raw["global_run_count"], path, line_number, "global_run_count")
            dp_rank = parse_int(raw["dp_rank"], path, line_number, "dp_rank")
            key = (run, dp_rank)
            if key in seen:
                fail(f"{path}:{line_number}: duplicate (global_run_count, dp_rank)={key}")
            seen.add(key)
            rows.append(
                LatencyRow(
                    global_run_count=run,
                    dp_rank=dp_rank,
                    total_us=parse_float(raw["total_us"], path, line_number, "total_us"),
                )
            )
    if not rows:
        fail(f"latency CSV contains no data rows: {path}")
    return rows


def load_cp_rows(path: Path) -> tuple[list[CpRow], list[int]]:
    if not path.is_file():
        fail(f"CP histogram CSV not found: {path}")

    rows: list[CpRow] = []
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, {"global_run_count", "dp_rank"})
        assert reader.fieldnames is not None
        cp_columns: dict[int, str] = {}
        for column in reader.fieldnames:
            if not column.startswith("cp_"):
                continue
            suffix = column.removeprefix("cp_")
            if not suffix.isdigit() or int(suffix) <= 0:
                fail(f"{path}: malformed CP histogram column: {column!r}")
            cp_columns[int(suffix)] = column
        if not cp_columns:
            fail(f"{path} has no cp_<size> histogram columns")

        for line_number, raw in enumerate(reader, start=2):
            run = parse_int(raw["global_run_count"], path, line_number, "global_run_count")
            dp_rank = parse_int(raw["dp_rank"], path, line_number, "dp_rank")
            key = (run, dp_rank)
            if key in seen:
                fail(f"{path}:{line_number}: duplicate (global_run_count, dp_rank)={key}")
            seen.add(key)
            counts: dict[int, int] = {}
            for cp_size, column in cp_columns.items():
                count = parse_int(raw[column], path, line_number, column)
                if count < 0:
                    fail(f"{path}:{line_number}: {column} must be non-negative")
                counts[cp_size] = count
            rows.append(CpRow(global_run_count=run, dp_rank=dp_rank, counts=counts))

    if not rows:
        fail(f"CP histogram CSV contains no data rows: {path}")
    return rows, sorted(cp_columns)


def load_approximation_summary(input_dir: Path) -> dict | None:
    path = input_dir / "approximation_summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid approximation summary: {path}: {exc}")
    if summary.get("mode") != "staircase":
        fail(f"unsupported approximation mode in {path}: {summary.get('mode')!r}")
    return summary


def validate_inputs(
    latency_rows: list[LatencyRow],
    cp_rows: list[CpRow],
    cp_sizes: list[int],
) -> None:
    latency_keys = {(row.global_run_count, row.dp_rank) for row in latency_rows}
    cp_keys = {(row.global_run_count, row.dp_rank) for row in cp_rows}
    if latency_keys != cp_keys:
        latency_only = len(latency_keys - cp_keys)
        cp_only = len(cp_keys - latency_keys)
        fail(
            "the two CSVs describe different (global_run_count, dp_rank) samples: "
            f"latency-only={latency_only}, CP-only={cp_only}"
        )

    dp_ranks = {row.dp_rank for row in latency_rows}
    expected_dp_ranks = set(range(max(dp_ranks, default=-1) + 1))
    if dp_ranks != expected_dp_ranks:
        fail(
            "DP ranks must form a contiguous prefix starting at DP0, "
            f"found {sorted(dp_ranks)}"
        )
    if 1 not in cp_sizes:
        fail("Fig. 19 requires a cp_1 column for the upper panel's right axis")


def select_window(
    latency_rows: list[LatencyRow],
    cp_rows: list[CpRow],
    tail_start_fraction: float,
    trim_tail_fraction: float,
    explicit_start: int | None,
    explicit_end: int | None,
) -> tuple[list[LatencyRow], list[CpRow], int, int, int]:
    if not 0.0 <= tail_start_fraction < 1.0:
        fail("--tail-start-fraction must be in [0, 1)")
    if not 0.0 <= trim_tail_fraction < 1.0:
        fail("--trim-tail-fraction must be in [0, 1)")

    all_runs = sorted({row.global_run_count for row in latency_rows})
    run_count = len(all_runs)
    start_index = min(int(run_count * tail_start_fraction), run_count - 1)
    end_index = min(int(run_count * (1.0 - trim_tail_fraction)), run_count - 1)
    start_run = explicit_start if explicit_start is not None else all_runs[start_index]
    end_run = explicit_end if explicit_end is not None else all_runs[end_index]
    if start_run > end_run:
        fail(f"selected window is empty: start iteration {start_run} > end iteration {end_run}")

    selected_latency = [
        row for row in latency_rows if start_run <= row.global_run_count <= end_run
    ]
    selected_cp = [row for row in cp_rows if start_run <= row.global_run_count <= end_run]
    if not selected_latency or not selected_cp:
        fail(f"no samples fall in the selected iteration window [{start_run}, {end_run}]")
    return selected_latency, selected_cp, start_run, end_run, run_count


def configure_plot_style() -> str:
    plt.style.use("seaborn-v0_8-paper")
    family = get_plot_font_family()
    plt.rcParams.update(
        {
            "font.family": family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return family


def cp_color(cp_size: int) -> str:
    if cp_size not in CP_COLORS:
        fail(
            f"no paper color is defined for CP={cp_size}; "
            f"supported sizes are {sorted(CP_COLORS)}"
        )
    return CP_COLORS[cp_size]


def normalized_output_base(path: Path) -> Path:
    if path.suffix.lower() in {".pdf", ".png"}:
        path = path.with_suffix("")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_figure(
    latency_rows: list[LatencyRow],
    cp_rows: list[CpRow],
    cp_sizes: list[int],
    output_base: Path,
    dpi: int,
    latency_style: str,
    cp_style: str,
) -> tuple[Path, Path]:
    if dpi <= 0:
        fail("--dpi must be positive")

    latency_by_run: dict[int, dict[int, float]] = defaultdict(dict)
    for row in latency_rows:
        latency_by_run[row.global_run_count][row.dp_rank] = row.total_us

    cp_by_run: dict[int, dict[int, int]] = defaultdict(
        lambda: {cp_size: 0 for cp_size in cp_sizes}
    )
    for row in cp_rows:
        for cp_size in cp_sizes:
            cp_by_run[row.global_run_count][cp_size] += row.counts[cp_size]

    sorted_runs = sorted(set(latency_by_run) | set(cp_by_run))
    x_min = sorted_runs[0]
    x_max = sorted_runs[-1]
    x_pad = max(5, int((x_max - x_min) * 0.003))

    fig, (ax_cp, ax_lat) = plt.subplots(
        2,
        1,
        figsize=(16.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    # Keep the wide paper aspect ratio while making both panels about 25%
    # shorter.  The smaller hspace is sufficient once the lower legend is
    # anchored closer to its axes.
    fig.subplots_adjust(left=0.072, right=0.914, bottom=0.13, top=0.80, hspace=0.36)

    label_fs = 36
    tick_fs = 30
    legend_fs = 30
    line_w = 3.9

    non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]
    peak_non_cp1 = 0
    if cp_style == "stacked-bar":
        bottoms = np.zeros(len(sorted_runs), dtype=float)
        for cp_size in non_cp1:
            values = np.asarray(
                [cp_by_run[run][cp_size] for run in sorted_runs], dtype=float
            )
            ax_cp.bar(
                sorted_runs,
                values,
                bottom=bottoms,
                width=1.0,
                color=cp_color(cp_size),
                linewidth=0,
                label=f"CP{cp_size}",
                rasterized=True,
            )
            bottoms += values
        peak_non_cp1 = float(np.max(bottoms)) if len(bottoms) else 0
    else:
        for cp_size in non_cp1:
            values = [cp_by_run[run][cp_size] for run in sorted_runs]
            peak_non_cp1 = max(peak_non_cp1, max(values, default=0))
            ax_cp.step(
                sorted_runs,
                values,
                where="mid",
                linewidth=line_w,
                color=cp_color(cp_size),
                label=f"CP{cp_size}",
            )
    ax_cp.set_ylabel(
        "# Reqs (CP>1)",
        fontsize=label_fs,
        fontweight="bold",
        color="#111111",
    )
    ax_cp.tick_params(axis="both", labelsize=tick_fs, width=1.4, colors="#222222")
    ax_cp.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
    ax_cp.grid(False, axis="x")
    ax_cp.spines["top"].set_visible(False)
    ax_cp.spines["right"].set_visible(False)
    ax_cp.spines["left"].set_linewidth(1.4)
    ax_cp.spines["bottom"].set_linewidth(1.4)
    ax_cp.spines["left"].set_color("#333333")
    ax_cp.spines["bottom"].set_color("#333333")
    ax_cp.set_xlabel("")
    ax_cp.tick_params(axis="x", labelbottom=False)
    ax_cp.set_ylim(0, max(1, peak_non_cp1 * 1.1))
    ax_cp.set_xlim(x_min - x_pad, x_max + x_pad)
    for tick in ax_cp.get_xticklabels() + ax_cp.get_yticklabels():
        tick.set_fontweight("bold")

    cp_handles, cp_labels = ax_cp.get_legend_handles_labels()
    ax_cp_right = ax_cp.twinx()
    cp1_values = [cp_by_run[run][1] for run in sorted_runs]
    ax_cp_right.step(
        sorted_runs,
        cp1_values,
        where="mid",
        linewidth=line_w - 0.2,
        color=cp_color(1),
        linestyle="--",
        label="CP1",
    )
    ax_cp_right.set_ylim(bottom=0, top=max(1, max(cp1_values) * 1.04))
    ax_cp_right.set_ylabel(
        "# Reqs (CP=1)",
        fontsize=label_fs,
        fontweight="bold",
        color="#111111",
        rotation=270,
        labelpad=36,
    )
    ax_cp_right.tick_params(axis="y", labelsize=tick_fs, width=1.4, colors="#222222", pad=4)
    if cp_style == "stacked-bar":
        scientific_formatter = ScalarFormatter(useMathText=True)
        scientific_formatter.set_scientific(True)
        scientific_formatter.set_powerlimits((3, 3))
        ax_cp_right.yaxis.set_major_formatter(scientific_formatter)
        offset_text = ax_cp_right.yaxis.get_offset_text()
        offset_text.set_fontsize(tick_fs - 3)
        offset_text.set_fontweight("bold")
        offset_text.set_color("#222222")
    ax_cp_right.spines["top"].set_visible(False)
    ax_cp_right.spines["left"].set_visible(False)
    ax_cp_right.spines["right"].set_linewidth(1.4)
    ax_cp_right.spines["right"].set_color("#333333")
    for tick in ax_cp_right.get_yticklabels():
        tick.set_fontweight("bold")
    right_handles, right_labels = ax_cp_right.get_legend_handles_labels()
    cp_legend = ax_cp.legend(
        right_handles + cp_handles,
        right_labels + cp_labels,
        frameon=False,
        ncol=len(right_handles + cp_handles),
        fontsize=legend_fs,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    for text in cp_legend.get_texts():
        text.set_fontweight("bold")

    if latency_style == "band":
        per_run_values = [
            list(latency_by_run[run].values())
            for run in sorted_runs
        ]
        mins = [float(np.min(values)) for values in per_run_values]
        maxs = [float(np.max(values)) for values in per_run_values]
        medians = [float(np.percentile(values, 50)) for values in per_run_values]
        band_color = "#1f77b4"
        ax_lat.fill_between(
            sorted_runs, mins, maxs, color=band_color, alpha=0.15, linewidth=0
        )
        ax_lat.plot(sorted_runs, medians, color=band_color, linewidth=line_w - 0.7)
        latency_handles = [
            Patch(facecolor=band_color, alpha=0.15, label="Min-Max"),
            Line2D([0], [0], color=band_color, linewidth=line_w - 0.7, label="Median"),
        ]
    else:
        dp_ranks = sorted({row.dp_rank for row in latency_rows})
        for dp_rank in dp_ranks:
            if dp_rank not in DP_COLORS:
                fail(
                    f"no paper color is defined for DP={dp_rank}; "
                    f"supported ranks are {sorted(DP_COLORS)}"
                )
            color = DP_COLORS[dp_rank]
            x_values = [run for run in sorted_runs if dp_rank in latency_by_run[run]]
            y_values = [latency_by_run[run][dp_rank] for run in x_values]
            ax_lat.plot(
                x_values,
                y_values,
                linewidth=line_w,
                color=color,
                label=f"DP{dp_rank}",
            )
        latency_handles, _ = ax_lat.get_legend_handles_labels()
    ax_lat.set_xlabel("Decode Iter", fontsize=label_fs, fontweight="bold", color="#111111")
    ax_lat.set_ylabel("A2A Lat. (us)", fontsize=label_fs, fontweight="bold", color="#111111")
    ax_lat.tick_params(axis="both", labelsize=tick_fs, width=1.4, colors="#222222")
    ax_lat.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
    ax_lat.grid(False, axis="x")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.spines["left"].set_linewidth(1.4)
    ax_lat.spines["bottom"].set_linewidth(1.4)
    ax_lat.spines["left"].set_color("#333333")
    ax_lat.spines["bottom"].set_color("#333333")
    latency_values = [value for values in latency_by_run.values() for value in values.values()]
    latency_data_min = min(latency_values)
    latency_data_max = max(latency_values)
    latency_padding = max(1.0, (latency_data_max - latency_data_min) * 0.05)
    latency_ymin = max(
        0.0,
        float(math.floor((latency_data_min - latency_padding) / 5.0) * 5.0),
    )
    latency_ymax = float(math.ceil((latency_data_max + latency_padding) / 5.0) * 5.0)
    if latency_ymax <= latency_ymin:
        latency_ymax = latency_ymin + 5.0
    ax_lat.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_lat.set_ylim(latency_ymin, latency_ymax)
    ax_lat.set_yticks(np.arange(latency_ymin, latency_ymax + 0.1, 5))
    for tick in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
        tick.set_fontweight("bold")

    # The break marks make the intentionally truncated lower bound explicit.
    break_size = 0.016
    break_style = dict(
        transform=ax_lat.transAxes,
        color="#333333",
        clip_on=False,
        linewidth=2.0,
    )
    ax_lat.plot((-break_size, +break_size), (-break_size, +break_size), **break_style)
    ax_lat.plot(
        (-break_size, +break_size),
        (0.038 - break_size, 0.038 + break_size),
        **break_style,
    )
    dp_legend = ax_lat.legend(
        handles=latency_handles,
        frameon=False,
        ncol=len(latency_handles),
        fontsize=legend_fs,
        handlelength=3.2,
        columnspacing=1.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
    )
    for text in dp_legend.get_texts():
        text.set_fontweight("bold")

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    try:
        input_dir = args.input_dir.expanduser().resolve()
        latency_path = input_dir / "per_iter_latencies.csv"
        cp_path = input_dir / "per_iter_cp_size_hist.csv"
        approximation_summary = load_approximation_summary(input_dir)
        latency_rows = load_latency_rows(latency_path)
        cp_rows, cp_sizes = load_cp_rows(cp_path)
        validate_inputs(latency_rows, cp_rows, cp_sizes)
        selected_latency, selected_cp, start_run, end_run, full_run_count = select_window(
            latency_rows,
            cp_rows,
            args.tail_start_fraction,
            args.trim_tail_fraction,
            args.start_iter,
            args.end_iter,
        )
        family = configure_plot_style()
        output_base = normalized_output_base(args.output)
        pdf_path, png_path = plot_figure(
            selected_latency,
            selected_cp,
            cp_sizes,
            output_base,
            args.dpi,
            args.latency_style,
            args.cp_style,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selected_runs = {row.global_run_count for row in selected_latency}
    print(f"Latency CSV: {latency_path}")
    print(f"CP histogram CSV: {cp_path}")
    print(
        "Selected decode iterations: "
        f"{start_run}..{end_run} inclusive "
        f"({len(selected_runs)} of {full_run_count} unique iterations)"
    )
    print(f"CP sizes: {cp_sizes}; font: {family}")
    print(f"Latency style: {args.latency_style}")
    print(f"CP>1 style: {args.cp_style}")
    if approximation_summary is not None:
        q_summary = approximation_summary["payloads"]["Q"]
        res_summary = approximation_summary["payloads"]["Res"]
        lse_summary = approximation_summary["lse"]
        print(
            "Approximation: staircase "
            f"step={approximation_summary['bucket_step']}, "
            f"Q={q_summary['benchmark_case_count']} cases/{q_summary['bucket_count']} buckets, "
            f"Res={res_summary['benchmark_case_count']} cases/{res_summary['bucket_count']} buckets, "
            f"LSE={lse_summary['benchmark_case_count']} calibration cases"
        )
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

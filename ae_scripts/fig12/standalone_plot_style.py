#!/usr/bin/env python3
"""Shared paper-style plotting for standalone Figure 12 runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class SeriesStyle:
    label: str
    color: str
    marker: str
    linestyle: str


METRICS = (
    ("slo_attainment_pct", "SLO\nAttainment (%)"),
    ("mean_tpot_ms", "Mean\nTPOT (ms)"),
    ("p99_tpot_ms", "P99\nTPOT (ms)"),
)

FONT_FAMILY = "DejaVu Sans"
FONT_RCPARAMS: dict[str, Any] = {
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _axis_limits(rates: Sequence[float]) -> tuple[float, float]:
    low = min(rates)
    high = max(rates)
    if math.isclose(low, high):
        margin = max(1.0, abs(low) * 0.08)
        return low - margin, high + margin
    margin = 0.04 * (high - low)
    return low - margin, high + margin


def _latency_limits(
    rows: Sequence[dict[str, Any]], metric_key: str
) -> tuple[float, float]:
    values = [float(row[metric_key]) for row in rows]
    minimum = min(values)
    maximum = max(values)
    lower = min(30.0, 10.0 * math.floor((minimum - 2.0) / 10.0))
    upper = max(100.0, 10.0 * math.ceil((maximum + 2.0) / 10.0))
    if lower == upper:
        upper += 10.0
    return lower, upper


def plot_matrix(
    grouped_rows: Mapping[str, Sequence[dict[str, Any]]],
    workload_labels: Mapping[str, str],
    series_order: Sequence[str],
    series_styles: Mapping[str, SeriesStyle],
    output_stem: Path,
    slo_target_ms: float,
    *,
    rate_key: str,
    series_key: str | None = None,
) -> None:
    """Plot workloads as columns, metrics as rows, and request rate on x."""
    if not grouped_rows:
        raise RuntimeError("no rows available to plot")

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(FONT_RCPARAMS)
    workload_slugs = list(grouped_rows)
    present_series = [
        series_id
        for series_id in series_order
        if any(
            series_key is None or str(row[series_key]) == series_id
            for rows in grouped_rows.values()
            for row in rows
        )
    ]
    if not present_series:
        raise RuntimeError("no configured series available to plot")

    ncols = len(workload_slugs)
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=ncols,
        figsize=(15, 2.0 * len(METRICS) + 0.6),
        squeeze=False,
    )

    for col_index, workload_slug in enumerate(workload_slugs):
        rows = list(grouped_rows[workload_slug])
        rates = sorted({float(row[rate_key]) for row in rows})
        x_min, x_max = _axis_limits(rates)

        for row_index, (metric_key, metric_label) in enumerate(METRICS):
            axis = axes[row_index, col_index]
            if row_index == 0:
                axis.set_title(
                    workload_labels[workload_slug],
                    fontweight="bold",
                    fontsize=16,
                )
            if col_index == 0:
                axis.set_ylabel(metric_label, fontweight="bold", fontsize=14)
            if row_index == len(METRICS) - 1:
                axis.set_xlabel("Request Rate (req/s)", fontsize=14)

            for series_id in present_series:
                points = [
                    row
                    for row in rows
                    if series_key is None or str(row[series_key]) == series_id
                ]
                points.sort(key=lambda row: float(row[rate_key]))
                if not points:
                    continue
                style = series_styles[series_id]
                axis.plot(
                    [float(point[rate_key]) for point in points],
                    [float(point[metric_key]) for point in points],
                    color=style.color,
                    marker=style.marker,
                    linestyle=style.linestyle,
                    linewidth=1.8,
                    markersize=5,
                    alpha=0.95,
                )

            axis.set_xlim(x_min, x_max)
            axis.set_xticks(rates)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.tick_params(axis="both", which="major", labelsize=14)
            axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
            axis.set_axisbelow(True)

            if metric_key == "slo_attainment_pct":
                axis.set_ylim(40, 105)
                axis.set_yticks((40, 60, 80, 100))
                axis.axhline(90, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
            else:
                lower, upper = _latency_limits(rows, metric_key)
                axis.set_ylim(lower, upper)
                axis.axhline(
                    slo_target_ms,
                    color="gray",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.7,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=series_styles[series_id].color,
            marker=series_styles[series_id].marker,
            linestyle=series_styles[series_id].linestyle,
            linewidth=1.8,
            markersize=5,
        )
        for series_id in present_series
    ]
    legend_labels = [series_styles[series_id].label for series_id in present_series]

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.11, top=0.90, hspace=0.55, wspace=0.14)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(legend_labels),
        frameon=False,
        prop={"family": FONT_FAMILY, "size": 14},
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

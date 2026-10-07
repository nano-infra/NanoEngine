#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = SCRIPT_DIR / "phase_breakdown_plot_data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "fig13"

FIG_WIDTH = 7
MIN_FIG_WIDTH = 3.4
FIG_HEIGHT = 3.7
FONT_SIZE = 14
TICK_FONT_SIZE = 12
XTICK_FONT_SIZE = 13
GROUP_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 15
TITLE_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 12
PNG_DPI = 300
SUBPLOT_TOP = 0.87
SUBPLOT_BOTTOM = 0.34

LONG_NODE_ORDER = (1, 3, 5, 7)
STRATEGY_ORDER = ("nano dcp", "dp4dcp8", "dp8dcp4", "dp16cp2", "dp32")
STRATEGY_LABELS = {
    "nano dcp": "DCP",
    "dp4dcp8": "CP8",
    "dp8dcp4": "CP4",
    "dp16cp2": "CP2",
    "dp32": "DP",
}

COMPONENTS = ("attn", "moe_a2a", "cp_cost", "other")
COMP_LABELS = {
    "other": "Others",
    "cp_cost": "CP Cost",
    "moe_a2a": "Dispatch+Combine",
    "attn": "Attention",
}
COMP_HATCHES = {
    "other": "",
    "cp_cost": "///",
    "moe_a2a": "\\\\\\",
    "attn": "",
}
COMP_EDGECOLORS = {
    "other": "none",
    "cp_cost": (1, 1, 1, 0.78),
    "moe_a2a": (1, 1, 1, 0.66),
    "attn": "none",
}
COMP_COLORS = {
    "other": "#BFC8D3",
    "cp_cost": "#C85A54",
    "moe_a2a": "#6BAE92",
    "attn": "#4C78A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig. 13 phase breakdown by long requests per node."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output base path or a .pdf/.png path; both formats are written",
    )
    return parser.parse_args()


def resolve_output_base(output: Path) -> Path:
    if output.suffix.lower() in {".pdf", ".png"}:
        return output.with_suffix("")
    return output


def configure_matplotlib() -> str:
    font_name = get_plot_font_family()

    matplotlib.rcParams.update(
        {
            "font.family": font_name,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.linewidth": 0.9,
            "xtick.labelsize": XTICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "hatch.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font_name


def load_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = ("long_node", "strategy", "attn", "moe_a2a", "cp_cost", "total")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    data = data[list(required)].copy()
    if data.empty:
        raise ValueError(f"No phase-breakdown rows in {csv_path}")
    for column in ("long_node", "attn", "moe_a2a", "cp_cost", "total"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data["long_node"] % 1 != 0).any():
        raise ValueError("long_node values must be integers")
    data["long_node"] = data["long_node"].astype(int)

    duplicate = data.duplicated(subset=("long_node", "strategy"), keep=False)
    if duplicate.any():
        raise ValueError(
            "Duplicate long_node/strategy rows:\n"
            + data.loc[duplicate, ["long_node", "strategy"]].to_string(index=False)
        )

    unsupported_long_nodes = sorted(set(data["long_node"]) - set(LONG_NODE_ORDER))
    unsupported_strategies = sorted(set(data["strategy"]) - set(STRATEGY_ORDER))
    if unsupported_long_nodes:
        raise ValueError(f"Unsupported long_node values: {unsupported_long_nodes}")
    if unsupported_strategies:
        raise ValueError(f"Unsupported strategies: {unsupported_strategies}")

    if (data[["attn", "moe_a2a", "cp_cost", "total"]] < 0).any().any():
        raise ValueError("Phase latencies and totals must be non-negative")
    if (data["total"] <= 0).any():
        raise ValueError("Total latency must be positive")
    residual = data["total"] - data["attn"] - data["moe_a2a"] - data["cp_cost"]
    if (residual < -1e-6).any():
        bad = data.loc[residual < -1e-6, ["long_node", "strategy", "total"]]
        raise ValueError(f"Computed negative Others latency:\n{bad.to_string(index=False)}")
    data["other"] = residual.clip(lower=0.0)
    return data


def compute_major_ticks(y_max: float) -> np.ndarray:
    for step in (500, 250, 200, 100):
        ticks = np.arange(0, y_max + 1e-9, step)
        if 3 <= len(ticks) <= 6:
            return ticks
    return np.linspace(0, y_max, 4)


def format_ratio_label(ratio: float) -> str:
    rounded = round(ratio, 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}x"
    return f"{rounded:.1f}x"


def plot_phase_breakdown(data: pd.DataFrame, output_base: Path, font_name: str) -> None:
    available_long_nodes = set(data["long_node"])
    long_nodes = tuple(
        long_node for long_node in LONG_NODE_ORDER if long_node in available_long_nodes
    )
    figure_width = max(
        MIN_FIG_WIDTH,
        FIG_WIDTH * len(long_nodes) / len(LONG_NODE_ORDER),
    )
    fig, axes = plt.subplots(
        ncols=len(long_nodes),
        figsize=(figure_width, FIG_HEIGHT),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        wspace=0.10,
    )

    y_top = data["total"].max() * 1.14
    major_ticks = compute_major_ticks(y_top)
    annotation_offset = y_top * 0.012

    for index, (axis, long_node) in enumerate(zip(axes, long_nodes)):
        available_strategies = set(data.loc[data["long_node"] == long_node, "strategy"])
        strategies = tuple(
            strategy for strategy in STRATEGY_ORDER if strategy in available_strategies
        )
        group = (
            data[data["long_node"] == long_node]
            .set_index("strategy")
            .reindex(strategies)
            .reset_index()
        )
        x = np.arange(len(strategies)) * 1.08
        totals = group["total"].to_numpy(dtype=float)
        bottoms = np.zeros(len(strategies))

        for component in COMPONENTS:
            values = group[component].to_numpy(dtype=float)
            axis.bar(
                x,
                values,
                bottom=bottoms,
                width=0.78,
                color=COMP_COLORS[component],
                edgecolor=COMP_EDGECOLORS[component],
                linewidth=0.35 if COMP_HATCHES[component] else 0.0,
                hatch=COMP_HATCHES[component],
                label=COMP_LABELS[component] if index == 0 else None,
            )
            bottoms += values

        if "nano dcp" in strategies:
            base_index = strategies.index("nano dcp")
            base_total = totals[base_index]
            for bar_index, (bar_x, total) in enumerate(zip(x, totals)):
                if bar_index == base_index:
                    continue
                axis.text(
                    bar_x,
                    total + annotation_offset,
                    format_ratio_label(total / base_total),
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE,
                    fontfamily=font_name,
                    color="#202020",
                    clip_on=False,
                )

        axis.text(
            0.97,
            0.93,
            f"Long={long_node}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=TITLE_FONT_SIZE,
            fontweight="semibold",
            fontfamily=font_name,
            color="#202020",
            zorder=5,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [STRATEGY_LABELS[strategy] for strategy in strategies],
            rotation=90,
            ha="center",
            va="top",
        )
        axis.set_xlim(x[0] - 0.55, x[-1] + 0.55)
        axis.set_ylim(0, y_top)
        axis.set_yticks(major_ticks)
        axis.set_axisbelow(True)
        axis.grid(
            axis="y",
            which="major",
            linestyle="-",
            linewidth=0.6,
            color="#E6E9EF",
            alpha=1.0,
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if index != 0:
            axis.spines["left"].set_visible(False)
            axis.tick_params(left=False, labelleft=False)

        group_y = -0.24
        if "nano dcp" in strategies:
            axis.text(
                x[strategies.index("nano dcp")],
                group_y,
                "Ours",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=GROUP_FONT_SIZE,
                fontfamily=font_name,
                color="#202020",
                clip_on=False,
            )
        vllm_positions = [
            x[position]
            for position, strategy in enumerate(strategies)
            if strategy != "nano dcp"
        ]
        if vllm_positions:
            axis.text(
                float(np.mean(vllm_positions)),
                group_y,
                "vLLM",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=GROUP_FONT_SIZE,
                fontfamily=font_name,
                color="#202020",
                clip_on=False,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        prop={"family": font_name, "size": LEGEND_FONT_SIZE},
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    axes[0].set_ylabel(
        "Latency ($\\mu$s)",
        fontsize=LABEL_FONT_SIZE,
        fontweight="semibold",
        fontfamily=font_name,
        labelpad=10,
    )

    for axis in axes:
        axis.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        axis.tick_params(axis="x", labelsize=XTICK_FONT_SIZE, pad=4)
        for label in axis.get_yticklabels():
            label.set_fontfamily(font_name)
            label.set_fontsize(TICK_FONT_SIZE)
        for label in axis.get_xticklabels():
            label.set_fontfamily(font_name)
            label.set_fontsize(XTICK_FONT_SIZE)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    # Append the format extension instead of replacing an existing suffix in
    # the requested basename (for example, ``fig13.generated``).
    pdf_path = Path(f"{output_base}.pdf")
    png_path = Path(f"{output_base}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


def main() -> None:
    args = parse_args()
    font_name = configure_matplotlib()
    data = load_data(args.input)
    plot_phase_breakdown(data, resolve_output_base(args.output), font_name)


if __name__ == "__main__":
    main()

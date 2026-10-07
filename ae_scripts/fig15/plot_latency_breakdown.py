#!/usr/bin/env python3
"""Plot the Fig. 15 per-layer latency breakdown for every GPU rank."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "fig15_latency_breakdown"
EXPECTED_LAYERS_PER_ITERATION = 57

CASE_ORDER = (
    "nano_dcp",
    "dp4dcp8_least_batch",
    "dp32_least_batch",
    "dp32_least_cache",
)
CASE_LABELS = {
    "nano_dcp": "NanoDeploy\n(DCP)",
    "dp4dcp8_least_batch": "vLLM\n(CP8)",
    "dp32_least_batch": "vLLM\n(DP-LeastBatch)",
    "dp32_least_cache": "vLLM\n(DP-LeastCache)",
}
COMPONENTS = (
    "attention_us",
    "moe_dispatch_combine_us",
    "cp_comm_us",
    "other_us",
)
COMPONENT_LABELS = {
    "attention_us": "Attention",
    "moe_dispatch_combine_us": "Dispatch+Combine",
    "cp_comm_us": "CP Cost",
    "other_us": "Others",
}
COMPONENT_COLORS = {
    "attention_us": "#4C78A8",
    "moe_dispatch_combine_us": "#6BAE92",
    "cp_comm_us": "#C85A54",
    "other_us": "#D7DCE2",
}
COMPONENT_HATCHES = {
    "attention_us": "",
    "moe_dispatch_combine_us": "\\\\\\",
    "cp_comm_us": "///",
    "other_us": "",
}
COMPONENT_EDGECOLORS = {
    "attention_us": "none",
    "moe_dispatch_combine_us": (1, 1, 1, 0.66),
    "cp_comm_us": (1, 1, 1, 0.78),
    "other_us": "none",
}

FIG_WIDTH = 7.0
FONT_SIZE = 12
LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 15
TITLE_FONT_SIZE = 13
DATASET_FONT_SIZE = 13
PNG_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DATASET=CSV",
        help=(
            "Dataset label and rank-summary CSV. Repeat labels to combine "
            "systems in one row, or use new labels to add rows."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output base path or a .pdf/.png path; both formats are written",
    )
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=32,
        help="Required GPU ranks per case (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    return args


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --input {value!r}; expected DATASET=CSV")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Invalid --input {value!r}; label and path are required")
    return label, Path(raw_path).expanduser().resolve()


def resolve_output_base(output: Path) -> Path:
    output = output.expanduser().resolve()
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
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
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


def load_dataset(
    label: str, csv_path: Path, expected_ranks: int
) -> pd.DataFrame:
    if not csv_path.is_file():
        raise ValueError(f"Input CSV does not exist: {csv_path}")
    data = pd.read_csv(csv_path)
    required = (
        "case",
        "strategy",
        "global_rank",
        "layer_samples",
        "total_us",
        "attention_us",
        "cp_comm_us",
        "moe_dispatch_combine_us",
        "other_us",
    )
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    optional = ("model_iterations", "layers_per_iteration")
    selected_columns = list(required) + [
        column for column in optional if column in data.columns
    ]
    data = data[selected_columns].copy()
    numeric_columns = selected_columns[2:]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    integer_columns = (
        "global_rank",
        "layer_samples",
    ) + tuple(column for column in optional if column in data.columns)
    for column in integer_columns:
        if (data[column] % 1 != 0).any():
            raise ValueError(f"{column} values must be integers in {csv_path}")
        data[column] = data[column].astype(int)

    duplicate = data.duplicated(subset=("case", "global_rank"), keep=False)
    if duplicate.any():
        bad = data.loc[duplicate, ["case", "global_rank"]]
        raise ValueError(f"Duplicate case/rank rows in {csv_path}:\n{bad}")

    actual_cases = set(data["case"])
    unsupported_cases = sorted(actual_cases - set(CASE_ORDER))
    if unsupported_cases:
        raise ValueError(
            f"{csv_path}: unsupported cases {unsupported_cases}; "
            f"expected a subset of {list(CASE_ORDER)}"
        )

    for case in (item for item in CASE_ORDER if item in actual_cases):
        case_rows = data[data["case"] == case]
        ranks = tuple(sorted(case_rows["global_rank"].tolist()))
        required_ranks = tuple(range(expected_ranks))
        if ranks != required_ranks:
            raise ValueError(
                f"{csv_path}/{case}: expected ranks 0--{expected_ranks - 1}, "
                f"found {list(ranks)}"
            )
        strategies = set(case_rows["strategy"])
        if len(strategies) != 1:
            raise ValueError(
                f"{csv_path}/{case}: expected one strategy, found "
                f"{sorted(strategies)}"
            )

    missing_metadata = [column for column in optional if column not in data.columns]
    if missing_metadata:
        raise ValueError(
            f"{csv_path}: per-rank layer summaries require columns "
            f"{missing_metadata}"
        )
    if set(data["layers_per_iteration"]) != {EXPECTED_LAYERS_PER_ITERATION}:
        raise ValueError(
            f"{csv_path}: layers_per_iteration must be "
            f"{EXPECTED_LAYERS_PER_ITERATION}"
        )
    expected_samples = data["model_iterations"] * data["layers_per_iteration"]
    if not (data["layer_samples"] == expected_samples).all():
        raise ValueError(
            f"{csv_path}: layer_samples must equal "
            "model_iterations * layers_per_iteration"
        )

    latency_columns = (
        "total_us",
        "attention_us",
        "cp_comm_us",
        "moe_dispatch_combine_us",
        "other_us",
    )
    if (data[list(latency_columns)] < -1e-6).any().any():
        raise ValueError(f"Negative latency values in {csv_path}")
    component_sum = (
        data["attention_us"]
        + data["cp_comm_us"]
        + data["moe_dispatch_combine_us"]
        + data["other_us"]
    )
    if not np.allclose(data["total_us"], component_sum, atol=0.01, rtol=1e-6):
        raise ValueError(f"Stack components do not reproduce total_us in {csv_path}")

    # Remove sub-microsecond CSV rounding drift so every stack ends at total_us.
    data["other_us"] = (
        data["total_us"]
        - data["attention_us"]
        - data["cp_comm_us"]
        - data["moe_dispatch_combine_us"]
    ).clip(lower=0.0)
    data["dataset"] = label
    return data


def load_inputs(
    values: list[str], expected_ranks: int
) -> tuple[list[str], pd.DataFrame]:
    labels: list[str] = []
    frames: list[pd.DataFrame] = []
    for value in values:
        label, csv_path = parse_input(value)
        if label not in labels:
            labels.append(label)
        frames.append(load_dataset(label, csv_path, expected_ranks))
    data = pd.concat(frames, ignore_index=True)
    duplicate = data.duplicated(
        subset=("dataset", "case", "global_rank"), keep=False
    )
    if duplicate.any():
        bad = data.loc[duplicate, ["dataset", "case", "global_rank"]]
        raise ValueError(f"Duplicate dataset/case/rank rows across inputs:\n{bad}")
    case_sets = {
        label: set(data.loc[data["dataset"] == label, "case"]) for label in labels
    }
    if len({frozenset(cases) for cases in case_sets.values()}) != 1:
        raise ValueError(f"Every plot row must contain the same cases: {case_sets}")
    return labels, data


def compute_major_ticks(y_top: float) -> np.ndarray:
    for step in (2000, 1000, 500, 250, 200, 100, 50, 25):
        ticks = np.arange(0, y_top + 1e-9, step)
        if 3 <= len(ticks) <= 6:
            return ticks
    return np.linspace(0, y_top, 4)


def plot_breakdown(
    labels: list[str],
    data: pd.DataFrame,
    output_base: Path,
    font_name: str,
    expected_ranks: int,
) -> None:
    row_count = len(labels)
    available_cases = set(data["case"])
    cases = tuple(case for case in CASE_ORDER if case in available_cases)
    figure_height = 2.65 + 0.85 * row_count
    subplot_top = 0.72 if row_count == 1 else 0.81
    fig, axes = plt.subplots(
        nrows=row_count,
        ncols=len(cases),
        figsize=(FIG_WIDTH, figure_height),
        sharey="row",
        squeeze=False,
    )
    bottom_margin = 0.21 if row_count == 1 else 0.14
    fig.subplots_adjust(
        left=0.13,
        right=0.995,
        top=subplot_top,
        bottom=bottom_margin,
        wspace=0.10,
        hspace=0.26,
    )

    for row_index, dataset in enumerate(labels):
        dataset_rows = data[data["dataset"] == dataset]
        y_top = float(dataset_rows["total_us"].max()) * 1.08
        major_ticks = compute_major_ticks(y_top)

        for column_index, case in enumerate(cases):
            axis = axes[row_index, column_index]
            case_rows = (
                dataset_rows[dataset_rows["case"] == case]
                .sort_values("global_rank")
                .reset_index(drop=True)
            )
            ranks = case_rows["global_rank"].to_numpy(dtype=int)
            bottoms = np.zeros(len(ranks))

            for component in COMPONENTS:
                values = case_rows[component].to_numpy(dtype=float)
                axis.bar(
                    ranks,
                    values,
                    bottom=bottoms,
                    width=0.92,
                    color=COMPONENT_COLORS[component],
                    edgecolor=COMPONENT_EDGECOLORS[component],
                    linewidth=0.35 if COMPONENT_HATCHES[component] else 0.0,
                    hatch=COMPONENT_HATCHES[component],
                    label=(
                        COMPONENT_LABELS[component]
                        if row_index == 0 and column_index == 0
                        else None
                    ),
                )
                bottoms += values

            if row_index == 0:
                axis.set_title(
                    CASE_LABELS[case],
                    fontsize=TITLE_FONT_SIZE,
                    fontweight="semibold",
                    pad=4,
                    fontfamily=font_name,
                )
            if column_index == 0:
                axis.text(
                    0.03,
                    0.94,
                    dataset,
                    transform=axis.transAxes,
                    fontsize=DATASET_FONT_SIZE,
                    fontweight="semibold",
                    fontfamily=font_name,
                    color="#222222",
                    ha="left",
                    va="top",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.4,
                    },
                )

            axis.set_xlim(-1, expected_ranks)
            axis.set_xticks(
                tuple(
                    dict.fromkeys(
                        np.linspace(0, expected_ranks - 1, 5, dtype=int).tolist()
                    )
                )
            )
            if row_index < row_count - 1:
                axis.tick_params(axis="x", labelbottom=False)
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
            if column_index != 0:
                axis.spines["left"].set_visible(False)
                axis.tick_params(left=False, labelleft=False)

            for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
                tick_label.set_fontfamily(font_name)
                tick_label.set_fontsize(FONT_SIZE)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        prop={"family": font_name, "size": LEGEND_FONT_SIZE},
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.supxlabel(
        "GPU Rank",
        x=0.562,
        y=0.035,
        fontsize=LABEL_FONT_SIZE,
        fontweight="semibold",
        fontfamily=font_name,
    )
    fig.supylabel(
        "Per-layer latency ($\\mu$s)",
        x=0.02,
        y=(bottom_margin + subplot_top) / 2,
        fontsize=LABEL_FONT_SIZE,
        fontweight="semibold",
        fontfamily=font_name,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(f"{output_base}.pdf")
    png_path = Path(f"{output_base}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


def main() -> None:
    args = parse_args()
    try:
        labels, data = load_inputs(args.input, args.expected_ranks)
        font_name = configure_matplotlib()
        plot_breakdown(
            labels,
            data,
            resolve_output_base(args.output),
            font_name,
            args.expected_ranks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

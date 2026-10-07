#!/usr/bin/env python3
"""Plot the compact 2x2 TPOT comparison used by the paper.

Columns are datasets (ShareGPT4o and Issue1%); rows are TPOT metrics (mean
and P99).  The script consumes the TSV files produced by
``plot_dbo_tpot_comparison.py`` and writes a single-column PDF and PNG.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

DEFAULT_SHAREGPT4O_DATA = SCRIPT_DIR / "dbo_tpot_comparison.tsv"
DEFAULT_ISSUE1_DATA = SCRIPT_DIR / "dbo_tpot_comparison_issue1.tsv"
DEFAULT_OUTPUT = SCRIPT_DIR / "fig20"

FIGURE_WIDTH_INCHES = 3.35
FIGURE_HEIGHT_INCHES = 3.05
SLO_TARGET_MS = 50.0


@dataclass(frozen=True)
class SeriesStyle:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str
    linewidth: float


SERIES = (
    SeriesStyle(
        key="non_dbo",
        label="vLLM",
        color="#6B7280",
        marker="o",
        linestyle="--",
        linewidth=1.15,
    ),
    SeriesStyle(
        key="dbo",
        label="vLLM + DBO",
        color="#4C78A8",
        marker="s",
        linestyle="--",
        linewidth=1.20,
    ),
    SeriesStyle(
        key="nano",
        label="Nano (DCP)",
        color="#E76F51",
        marker="D",
        linestyle="-",
        linewidth=1.35,
    ),
)

METRICS = (
    ("tpot_mean_ms", "Mean TPOT (ms)"),
    ("tpot_p99_ms", "P99 TPOT (ms)"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a paper-ready, single-column 2x2 comparison from the "
            "ShareGPT4o and Issue1% DBO comparison TSVs."
        )
    )
    parser.add_argument(
        "--sharegpt4o-data",
        type=Path,
        default=DEFAULT_SHAREGPT4O_DATA,
        help=f"ShareGPT4o comparison TSV (default: {DEFAULT_SHAREGPT4O_DATA})",
    )
    parser.add_argument(
        "--issue1-data",
        type=Path,
        default=DEFAULT_ISSUE1_DATA,
        help=f"Issue1%% comparison TSV (default: {DEFAULT_ISSUE1_DATA})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output base path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=FIGURE_WIDTH_INCHES,
        help=f"figure width in inches (default: {FIGURE_WIDTH_INCHES:g})",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=FIGURE_HEIGHT_INCHES,
        help=f"figure height in inches (default: {FIGURE_HEIGHT_INCHES:g})",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser


def normalize_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() in {".pdf", ".png"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_float(row: dict[str, str], field: str, path: Path, line: int) -> float:
    raw = row.get(field)
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError as exc:
        raise ValueError(f"{path}:{line}: invalid {field!r}: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}:{line}: expected non-negative finite {field!r}")
    return value


def load_dataset(path: Path) -> list[dict[str, float]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input TSV not found: {path}")

    required_fields = {"request_rate_rps"}
    required_fields.update(
        f"{series.key}_{metric_key}"
        for series in SERIES
        for metric_key, _ in METRICS
    )

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required_fields - fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        for line_number, raw_row in enumerate(reader, start=2):
            parsed = {
                field: require_float(raw_row, field, path, line_number)
                for field in required_fields
            }
            rows.append(parsed)

    if not rows:
        raise ValueError(f"{path}: no data rows")
    rows.sort(key=lambda row: row["request_rate_rps"])
    rates = [row["request_rate_rps"] for row in rows]
    if len(rates) != len(set(rates)):
        raise ValueError(f"{path}: duplicate request rates")
    return rows


def configure_plot_style(font_family: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.edgecolor": "#59636E",
            "axes.linewidth": 0.65,
            "axes.labelcolor": "#263238",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
            "text.color": "#263238",
            "grid.color": "#DCE3E8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def plot_panel(axis, rows: list[dict[str, float]], metric_key: str) -> None:
    from matplotlib.ticker import MaxNLocator

    rates = [row["request_rate_rps"] for row in rows]
    all_values: list[float] = []
    for series in SERIES:
        values = [row[f"{series.key}_{metric_key}"] for row in rows]
        all_values.extend(values)
        axis.plot(
            rates,
            values,
            color=series.color,
            marker=series.marker,
            linestyle=series.linestyle,
            linewidth=series.linewidth,
            markersize=3.2,
            markerfacecolor="white" if series.key != "nano" else series.color,
            markeredgecolor=series.color,
            markeredgewidth=0.65,
            solid_capstyle="round",
            zorder=3 if series.key == "nano" else 2,
        )

    axis.axhline(
        SLO_TARGET_MS,
        color="#A6ADB4",
        linestyle=":",
        linewidth=0.75,
        alpha=0.95,
        zorder=1,
    )
    axis.set_xlim(min(rates), max(rates))
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:g}" for rate in rates])
    axis.set_ylim(0, max(all_values) * 1.10)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.grid(axis="y", linestyle="--")
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=2.3, width=0.55, pad=1.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def build_figure(
    all_short_rows: list[dict[str, float]],
    issue1_rows: list[dict[str, float]],
    output_base: Path,
    *,
    width: float,
    height: float,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(width, height), squeeze=False)
    datasets = (
        ("ShareGPT4o", all_short_rows),
        ("Issue1%", issue1_rows),
    )

    for column, (title, rows) in enumerate(datasets):
        axes[0, column].set_title(title, fontweight="bold", pad=3.0)
        for row_index, (metric_key, metric_label) in enumerate(METRICS):
            axis = axes[row_index, column]
            plot_panel(axis, rows, metric_key)
            if column == 0:
                axis.set_ylabel(metric_label, fontweight="bold", labelpad=2.5)
            if row_index == 0:
                axis.tick_params(axis="x", labelbottom=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=series.color,
            marker=series.marker,
            linestyle=series.linestyle,
            linewidth=series.linewidth,
            markersize=3.5,
            markerfacecolor="white" if series.key != "nano" else series.color,
            markeredgecolor=series.color,
            markeredgewidth=0.65,
        )
        for series in SERIES
    ]
    fig.legend(
        legend_handles,
        [series.label for series in SERIES],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.supxlabel("Request Rate (req/s)", x=0.56, y=0.025, fontsize=7.5)
    fig.subplots_adjust(
        left=0.083,
        right=0.985,
        bottom=0.120,
        top=0.875,
        wspace=0.20,
        hspace=0.16,
    )

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    # Keep the exact canvas width: bbox_inches="tight" would change the
    # nominal 3.35-inch single-column size.
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    args = build_parser().parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    mpl_cache = Path(tempfile.gettempdir()) / f"matplotlib-asplos-ae-{os.getuid()}"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")

    from ae_utils.plotting import get_plot_font_family

    try:
        all_short_rows = load_dataset(args.sharegpt4o_data)
        issue1_rows = load_dataset(args.issue1_data)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    font_family = get_plot_font_family()
    configure_plot_style(font_family)
    output_base = normalize_output(args.output)
    pdf_path, png_path = build_figure(
        all_short_rows,
        issue1_rows,
        output_base,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )

    print(f"font: {font_family}")
    print(f"figure size: {args.width:g} x {args.height:g} inches")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

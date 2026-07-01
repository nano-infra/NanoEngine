#!/usr/bin/env python3

import argparse
import ast
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HIST_RE = re.compile(r"sp_size_hist_global':\s*(\{[^}]*\})")
RATE_RE = re.compile(r"_r(\d+)_")
TOPO_RE = re.compile(r"dp(\d+)sp(\d+)(?:tp(\d+))?(?:ep(\d+))?")
MODEL_DISPLAY = {
    "deepseek-v3": "DeepSeek-V3",
}
CP_COLORS = {
    1: "#8A9099",
    2: "#0072B2",
    3: "#E69F00",
    4: "#009E73",
    5: "#D55E00",
    6: "#CC79A7",
    7: "#56B4E9",
    8: "#F0E442",
}


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B7BEC8",
            "axes.linewidth": 0.9,
            "grid.color": "#D8DDE6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-iter CP-size request histograms from NanoDeploy step logs."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A log file or a directory containing per-rate *.log files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save CSVs and plots.",
    )
    parser.add_argument(
        "--hide-header",
        action="store_true",
        help="Hide figure title and subtitle for paper-ready exports.",
    )
    return parser.parse_args()


def discover_logs(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.log"))


def parse_rate_tag(log_path: Path):
    match = RATE_RE.search(log_path.parent.name)
    if match:
        rate = int(match.group(1))
        return f"r{rate}", rate
    return log_path.parent.name, None


def infer_issue_label(log_path: Path):
    text = str(log_path).lower()
    if "issue001" in text or "r0.01" in text:
        return "Issue001"
    if "issue005" in text or "r0.05" in text:
        return "Issue005"
    return None


def infer_model_label(log_path: Path):
    model_key = log_path.parents[2].name
    return MODEL_DISPLAY.get(model_key, model_key.replace("-", " ").title())


def infer_topology_label(log_path: Path):
    match = TOPO_RE.search(log_path.parent.name)
    if not match:
        return None
    dp, sp, tp, ep = match.groups()
    parts = [f"DP{dp}", f"SP{sp}"]
    if tp:
        parts.append(f"TP{tp}")
    if ep:
        parts.append(f"EP{ep}")
    return " ".join(parts)


def infer_total_cp_size(log_path: Path):
    match = TOPO_RE.search(log_path.parent.name)
    if not match:
        return None
    _, sp, _, _ = match.groups()
    return int(sp)


def build_context(log_path: Path, rate):
    parts = [infer_model_label(log_path)]
    issue_label = infer_issue_label(log_path)
    topology = infer_topology_label(log_path)
    if issue_label:
        parts.append(issue_label)
    if topology:
        parts.append(topology)
    if rate is not None:
        parts.append(f"{rate} req/s")
    return " | ".join(parts)


def extract_rows(log_path: Path):
    rows = []
    cp_sizes = set()
    for line_no, line in enumerate(log_path.read_text(errors="ignore").splitlines(), start=1):
        match = HIST_RE.search(line)
        if not match:
            continue
        hist = ast.literal_eval(match.group(1))
        row = {"iter": len(rows), "line_no": line_no}
        for cp_size, count in hist.items():
            cp_int = int(cp_size)
            row[f"cp_{cp_int}"] = int(count)
            cp_sizes.add(cp_int)
        rows.append(row)

    cp_sizes = sorted(cp_sizes)
    for row in rows:
        for cp_size in cp_sizes:
            row.setdefault(f"cp_{cp_size}", 0)
    return rows, cp_sizes


def write_csv(rows, cp_sizes, csv_path: Path):
    fieldnames = ["iter", "line_no"] + [f"cp_{cp_size}" for cp_size in cp_sizes]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B7BEC8")
    ax.spines["bottom"].set_color("#B7BEC8")
    ax.tick_params(colors="#2E3440")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def add_figure_header(fig, context: str, variant: str):
    fig.suptitle(
        "Per-iteration request count by CP size",
        x=0.08,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(0.08, 0.948, context, ha="left", va="top", fontsize=10.5, color="#5B6573")
    fig.text(
        0.92,
        0.948,
        variant,
        ha="right",
        va="top",
        fontsize=10,
        color="#5B6573",
        fontweight="semibold",
    )


def build_legend_handles(cp_sizes, observed_cp_sizes, dual_axis: bool):
    handles = []
    labels = []
    observed = set(observed_cp_sizes)
    for cp_size in cp_sizes:
        is_observed = cp_size in observed
        alpha = 1.0 if is_observed else 0.35
        linestyle = "-" if is_observed else ":"
        if dual_axis and cp_size == 1:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--" if is_observed else ":",
                    linewidth=1.7,
                    alpha=alpha,
                )
            )
            labels.append("CP=1 (right axis)")
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cp_color(cp_size),
                    linestyle=linestyle,
                    linewidth=2.0,
                    alpha=alpha,
                )
            )
            labels.append(f"CP={cp_size}")
    return handles, labels


def reorder_legend_items(handles, labels, ncol: int):
    items = list(zip(handles, labels))
    if not items:
        return handles, labels
    nrows = (len(items) + ncol - 1) // ncol
    reordered = []
    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < len(items):
                reordered.append(items[idx])
    new_handles = [handle for handle, _ in reordered]
    new_labels = [label for _, label in reordered]
    return new_handles, new_labels


def add_legend(fig, handles, labels, hide_header: bool):
    if not handles:
        return
    ncol = min(4, len(labels))
    handles, labels = reorder_legend_items(handles, labels, ncol)
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965 if hide_header else 0.905),
        ncol=ncol,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
        title=None if hide_header else "CP size",
    )
    if legend.get_title() is not None:
        legend.get_title().set_fontweight("semibold")


def cp_color(cp_size: int):
    return CP_COLORS.get(cp_size, "#4C78A8")


def save_figure(fig, out_path: Path):
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))


def plot_line(ax, x, y, cp_size: int, label: str | None = None, alpha: float = 1.0):
    return ax.step(
        x,
        y,
        where="mid",
        linewidth=2.0,
        color=cp_color(cp_size),
        label=label or f"CP={cp_size}",
        alpha=alpha,
    )[0]


def plot_full(rows, cp_sizes, context: str, out_path: Path, hide_header: bool, legend_cp_sizes):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = [row["iter"] for row in rows]
    for cp_size in cp_sizes:
        plot_line(ax, x, [row[f"cp_{cp_size}"] for row in rows], cp_size)
    style_axes(ax)
    ax.set_xlabel("Decode Iter")
    ax.set_ylabel("Request count")
    if not hide_header:
        add_figure_header(fig, context, "All CP sizes")
    handles, labels = build_legend_handles(legend_cp_sizes, cp_sizes, dual_axis=False)
    add_legend(fig, handles, labels, hide_header)
    top = 0.86 if hide_header else 0.84
    fig.tight_layout(rect=(0.04, 0.03, 0.98, top))
    save_figure(fig, out_path)
    plt.close(fig)


def plot_dual_axis(rows, cp_sizes, context: str, out_path: Path, hide_header: bool, legend_cp_sizes):
    x = [row["iter"] for row in rows]
    fig, ax_left = plt.subplots(figsize=(9.2, 4.8))

    non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]

    if non_cp1:
        peak_non_cp1 = max(max(row[f"cp_{cp_size}"] for row in rows) for cp_size in non_cp1)
        for cp_size in non_cp1:
            plot_line(ax_left, x, [row[f"cp_{cp_size}"] for row in rows], cp_size)
        ax_left.set_ylim(0, max(1, peak_non_cp1 * 1.1))
    else:
        ax_left.set_ylim(0, 1)

    style_axes(ax_left)
    ax_left.set_xlabel("Decode Iter")
    ax_left.set_ylabel("Request count for CP>1")

    if 1 in cp_sizes:
        ax_right = ax_left.twinx()
        ax_right.plot(
            x,
            [row["cp_1"] for row in rows],
            color="gray",
            linestyle="--",
            linewidth=1.7,
            alpha=0.9,
            drawstyle="steps-mid",
        )[0]
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["right"].set_color("#B7BEC8")
        ax_right.tick_params(colors="#2E3440")
        ax_right.set_ylabel("Request count for CP=1")

    if not hide_header:
        add_figure_header(fig, context, "Dual-axis emphasis")
    handles, labels = build_legend_handles(legend_cp_sizes, cp_sizes, dual_axis=True)
    add_legend(fig, handles, labels, hide_header)
    top = 0.86 if hide_header else 0.84
    fig.tight_layout(rect=(0.04, 0.03, 0.98, top))
    save_figure(fig, out_path)
    plt.close(fig)


def plot_non_cp1(rows, cp_sizes, context: str, out_path: Path, hide_header: bool, legend_cp_sizes):
    non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]
    if not non_cp1:
        return

    x = [row["iter"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    peak_non_cp1 = max(max(row[f"cp_{cp_size}"] for row in rows) for cp_size in non_cp1)
    for cp_size in non_cp1:
        plot_line(ax, x, [row[f"cp_{cp_size}"] for row in rows], cp_size)
    style_axes(ax)
    ax.set_xlabel("Decode Iter")
    ax.set_ylabel("Request count")
    ax.set_ylim(0, max(1, peak_non_cp1 * 1.1))
    if not hide_header:
        add_figure_header(fig, context, "CP>1 focus")
    legend_non_cp1 = [cp_size for cp_size in legend_cp_sizes if cp_size != 1]
    handles, labels = build_legend_handles(legend_non_cp1, non_cp1, dual_axis=False)
    add_legend(fig, handles, labels, hide_header)
    top = 0.86 if hide_header else 0.84
    fig.tight_layout(rect=(0.04, 0.03, 0.98, top))
    save_figure(fig, out_path)
    plt.close(fig)


def main():
    args = parse_args()
    configure_plot_style()
    logs = discover_logs(args.input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for log_path in logs:
        rows, cp_sizes = extract_rows(log_path)
        if not rows:
            continue

        rate_tag, rate = parse_rate_tag(log_path)
        out_dir = args.output_dir / rate_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "iter_cp_size_hist.csv"
        plot_full_path = out_dir / "iter_cp_size_hist_all.png"
        plot_dual_path = out_dir / "iter_cp_size_hist_dual_axis.png"
        plot_non_cp1_path = out_dir / "iter_cp_size_hist_non_cp1.png"

        total_cp_size = infer_total_cp_size(log_path) or max(cp_sizes)
        legend_cp_sizes = list(range(1, total_cp_size + 1))
        for row in rows:
            for cp_size in legend_cp_sizes:
                row.setdefault(f"cp_{cp_size}", 0)

        write_csv(rows, legend_cp_sizes, csv_path)
        context = build_context(log_path, rate)
        plot_full(rows, cp_sizes, context, plot_full_path, args.hide_header, legend_cp_sizes)
        plot_dual_axis(rows, cp_sizes, context, plot_dual_path, args.hide_header, legend_cp_sizes)
        plot_non_cp1(rows, cp_sizes, context, plot_non_cp1_path, args.hide_header, legend_cp_sizes)

        peak_by_cp = {
            cp_size: max(row[f"cp_{cp_size}"] for row in rows)
            for cp_size in cp_sizes
        }
        summary_rows.append(
            {
                "rate": "" if rate is None else rate,
                "rate_tag": rate_tag,
                "iters": len(rows),
                "cp_sizes": ",".join(str(cp_size) for cp_size in cp_sizes),
                "peak_cp_1": peak_by_cp.get(1, 0),
                "peak_non_cp1": max(
                    (peak for cp_size, peak in peak_by_cp.items() if cp_size != 1),
                    default=0,
                ),
                "source_log": str(log_path),
            }
        )

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rate",
                "rate_tag",
                "iters",
                "cp_sizes",
                "peak_cp_1",
                "peak_non_cp1",
                "source_log",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"processed_logs={len(summary_rows)}")
    print(f"summary={summary_csv}")


if __name__ == "__main__":
    main()

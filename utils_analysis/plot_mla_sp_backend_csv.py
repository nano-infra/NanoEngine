from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_CHOICES = (
    "wall_mean_us",
    "all2all_mean_us",
    "comm_region_mean_us",
)


def parse_csv_choices(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = []
    for part in raw.split(","):
        value = part.strip()
        if value:
            values.append(value)
    return values or None


def convert_scalar(value: str) -> object:
    value = value.strip()
    if value == "":
        return value
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({key: convert_scalar(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def filter_rows(
    rows: list[dict[str, object]],
    *,
    cp_sizes: list[str] | None,
    patterns: list[str] | None,
    payloads: list[str] | None,
    backends: list[str] | None,
) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        if cp_sizes is not None and str(row["cp_size"]) not in cp_sizes:
            continue
        if patterns is not None and str(row["pattern"]) not in patterns:
            continue
        if payloads is not None and str(row["payload"]) not in payloads:
            continue
        if backends is not None and str(row["backend"]) not in backends:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No rows left after applying filters")
    return filtered


def unique_in_order(rows: list[dict[str, object]], key: str) -> list[str]:
    values = []
    seen = set()
    for row in rows:
        value = str(row[key])
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def series_points(
    rows: list[dict[str, object]],
    *,
    cp_size: str,
    pattern: str,
    payload: str,
    backend: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    matching = [
        row
        for row in rows
        if str(row["cp_size"]) == cp_size
        and str(row["pattern"]) == pattern
        and str(row["payload"]) == payload
        and str(row["backend"]) == backend
    ]
    matching.sort(key=lambda row: int(row["batch_size"]))
    xs = [int(row["batch_size"]) for row in matching]
    ys = [float(row[metric]) for row in matching]
    return xs, ys


def save_latency_figure(
    *,
    rows: list[dict[str, object]],
    cp_size: str,
    patterns: list[str],
    payloads: list[str],
    backends: list[str],
    metric: str,
    output_dir: Path,
    x_log2: bool,
    title_prefix: str,
) -> Path:
    fig, axes = plt.subplots(
        len(patterns),
        len(payloads),
        figsize=(5.0 * len(payloads), 3.8 * len(patterns)),
        squeeze=False,
    )

    for row_idx, pattern in enumerate(patterns):
        for col_idx, payload in enumerate(payloads):
            ax = axes[row_idx][col_idx]
            has_data = False
            for backend in backends:
                xs, ys = series_points(
                    rows,
                    cp_size=cp_size,
                    pattern=pattern,
                    payload=payload,
                    backend=backend,
                    metric=metric,
                )
                if not xs:
                    continue
                has_data = True
                ax.plot(xs, ys, marker="o", linewidth=2.0, label=backend)

            if not has_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.3)
            if x_log2:
                ax.set_xscale("log", base=2)
            if row_idx == len(patterns) - 1:
                ax.set_xlabel("Batch Size")
            if col_idx == 0:
                ax.set_ylabel(metric)
            ax.set_title(f"{pattern} / {payload}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(len(backends), 1))
    fig.suptitle(f"{title_prefix}CP={cp_size} {metric}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = output_dir / f"latency_cp{cp_size}_{metric}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_speedup_rows(
    rows: list[dict[str, object]],
    *,
    baseline_backend: str,
    candidate_backend: str,
    metric: str,
) -> list[dict[str, object]]:
    baseline_map = {}
    candidate_map = {}
    for row in rows:
        key = (
            str(row["cp_size"]),
            str(row["pattern"]),
            str(row["payload"]),
            int(row["batch_size"]),
        )
        if str(row["backend"]) == baseline_backend:
            baseline_map[key] = row
        if str(row["backend"]) == candidate_backend:
            candidate_map[key] = row

    speedup_rows = []
    for key, baseline_row in baseline_map.items():
        candidate_row = candidate_map.get(key)
        if candidate_row is None:
            continue
        baseline_value = float(baseline_row[metric])
        candidate_value = float(candidate_row[metric])
        if candidate_value <= 0:
            continue
        speedup_rows.append(
            {
                "cp_size": key[0],
                "pattern": key[1],
                "payload": key[2],
                "batch_size": key[3],
                "speedup": baseline_value / candidate_value,
            }
        )
    return speedup_rows


def save_speedup_figure(
    *,
    speedup_rows: list[dict[str, object]],
    cp_size: str,
    patterns: list[str],
    payloads: list[str],
    output_dir: Path,
    x_log2: bool,
    title_prefix: str,
    baseline_backend: str,
    candidate_backend: str,
    metric: str,
) -> Path | None:
    cp_rows = [row for row in speedup_rows if str(row["cp_size"]) == cp_size]
    if not cp_rows:
        return None

    fig, axes = plt.subplots(
        len(patterns),
        len(payloads),
        figsize=(5.0 * len(payloads), 3.8 * len(patterns)),
        squeeze=False,
    )

    for row_idx, pattern in enumerate(patterns):
        for col_idx, payload in enumerate(payloads):
            ax = axes[row_idx][col_idx]
            matching = [
                row
                for row in cp_rows
                if str(row["pattern"]) == pattern and str(row["payload"]) == payload
            ]
            matching.sort(key=lambda row: int(row["batch_size"]))
            xs = [int(row["batch_size"]) for row in matching]
            ys = [float(row["speedup"]) for row in matching]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.0, color="tab:red")
                ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
            else:
                ax.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.3)
            if x_log2:
                ax.set_xscale("log", base=2)
            if row_idx == len(patterns) - 1:
                ax.set_xlabel("Batch Size")
            if col_idx == 0:
                ax.set_ylabel("Speedup")
            ax.set_title(f"{pattern} / {payload}")

    fig.suptitle(
        f"{title_prefix}CP={cp_size} {baseline_backend} / {candidate_backend} on {metric}",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = output_dir / f"speedup_cp{cp_size}_{metric}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot latency and speedup curves from benchmark_mla_sp_backend.py CSV output."
    )
    parser.add_argument("csv_path", type=str, help="CSV produced by benchmark_mla_sp_backend.py")
    parser.add_argument(
        "--metric",
        type=str,
        default="all2all_mean_us",
        choices=METRIC_CHOICES,
    )
    parser.add_argument("--cp-sizes", type=str, default=None)
    parser.add_argument("--patterns", type=str, default=None)
    parser.add_argument("--payloads", type=str, default=None)
    parser.add_argument("--backends", type=str, default=None)
    parser.add_argument("--baseline-backend", type=str, default="hao_basic")
    parser.add_argument("--candidate-backend", type=str, default="nccl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--title-prefix", type=str, default="MLA SP Backend Curves ")
    parser.add_argument("--x-log2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    rows = read_rows(csv_path)
    rows = filter_rows(
        rows,
        cp_sizes=parse_csv_choices(args.cp_sizes),
        patterns=parse_csv_choices(args.patterns),
        payloads=parse_csv_choices(args.payloads),
        backends=parse_csv_choices(args.backends),
    )

    cp_sizes = unique_in_order(rows, "cp_size")
    patterns = parse_csv_choices(args.patterns) or unique_in_order(rows, "pattern")
    payloads = parse_csv_choices(args.payloads) or unique_in_order(rows, "payload")
    backends = parse_csv_choices(args.backends) or unique_in_order(rows, "backend")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for cp_size in cp_sizes:
        saved_paths.append(
            save_latency_figure(
                rows=rows,
                cp_size=cp_size,
                patterns=patterns,
                payloads=payloads,
                backends=backends,
                metric=args.metric,
                output_dir=output_dir,
                x_log2=args.x_log2,
                title_prefix=args.title_prefix,
            )
        )

    speedup_rows = compute_speedup_rows(
        rows,
        baseline_backend=args.baseline_backend,
        candidate_backend=args.candidate_backend,
        metric=args.metric,
    )
    for cp_size in cp_sizes:
        output_path = save_speedup_figure(
            speedup_rows=speedup_rows,
            cp_size=cp_size,
            patterns=patterns,
            payloads=payloads,
            output_dir=output_dir,
            x_log2=args.x_log2,
            title_prefix=args.title_prefix,
            baseline_backend=args.baseline_backend,
            candidate_backend=args.candidate_backend,
            metric=args.metric,
        )
        if output_path is not None:
            saved_paths.append(output_path)

    for path in saved_paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()

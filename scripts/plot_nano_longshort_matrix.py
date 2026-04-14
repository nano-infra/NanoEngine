#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

try:
    import orjson
except ImportError:
    orjson = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    stage_globs: tuple[str, ...]


DATASET_SPECS = (
    DatasetSpec(
        key="mixed60k",
        label="ShareGPT4o",
        stage_globs=("longshort_mixed60k_deepseek_v3*",),
    ),
    DatasetSpec(
        key="issue001",
        label="Issue1%",
        stage_globs=(
            "longshort_issue001_deepseek_v3*",
            "dp32_issue001_deepseek_v3*",
        ),
    ),
    DatasetSpec(
        key="issue003",
        label="Issue3%",
        stage_globs=("longshort_issue003_random_deepseek_v3*",),
    ),
    DatasetSpec(
        key="issue005",
        label="Issue5%",
        stage_globs=("longshort_issue005_random_deepseek_v3*",),
    ),
    DatasetSpec(
        key="gemini_issues",
        label="Gemini Issues",
        stage_globs=(
            "longshort_gemini_issues_deepseek_v3*",
            "dp32_gemini_issues_deepseek_v3*",
        ),
    ),
)

DATASET_ORDER = [spec.key for spec in DATASET_SPECS]
DATASET_LABEL_MAP = {spec.key: spec.label for spec in DATASET_SPECS}
SERIES_ORDER = ["dp32_baseline", "longshort_dp4sp8"]
SERIES_LABEL_MAP = {
    "dp32_baseline": "DP32 Baseline",
    "longshort_dp4sp8": "Long-Short SP8",
}
SERIES_STYLE_MAP = {
    "dp32_baseline": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "--",
    },
    "longshort_dp4sp8": {
        "color": "#f768a1",
        "marker": "D",
        "linestyle": "-",
    },
}
LATENCY_Y_MAX_MS = 100.0
LATENCY_Y_MIN_MS = 30.0
LATENCY_X_CROSS_TARGET_MS = 120.0
LATENCY_Y_TICKS = [40.0, 60.0, 80.0, 100.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse NanoDeploy long-short benchmark sweep summaries, recompute "
            "request-level SLO/NormLat metrics from JSONL outputs, and draw a "
            "dataset matrix similar to the Nano paper plots."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run-tag directory under bench_logs/, e.g. bench_logs/<RUN_TAG>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/plots_nano_longshort_matrix.",
    )
    parser.add_argument(
        "--slo-target",
        type=float,
        default=50.0,
        help="Normalized latency SLO target in ms. Default: 50.",
    )
    parser.add_argument(
        "--normlat-log-scale",
        action="store_true",
        help="Deprecated compatibility flag. Latency axes are always linear now.",
    )
    return parser.parse_args()


def parse_timestamp_from_path(path_str: str) -> datetime | None:
    match = re.search(r"(\d{8}_\d{6})(?:\.[a-z0-9]+)?$", path_str)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    float_value = to_float(value)
    if float_value is None:
        return None
    return int(float_value)


def canonical_series_key(strategy: str, stage_dir_name: str) -> str:
    strategy = (strategy or "").strip().lower()
    stage_dir_name = stage_dir_name.lower()
    if strategy.startswith("dp32") or stage_dir_name.startswith("dp32_"):
        return "dp32_baseline"
    if strategy.startswith("dp4sp8") or "longshort" in stage_dir_name:
        return "longshort_dp4sp8"
    return strategy or stage_dir_name


def parse_total_time_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    patterns = (
        r"Total time:\s*([\d.]+)s",
        r"Total time:\s*([\d.]+)",
        r"Total benchmark duration:\s*([\d.]+)s",
        r"Total benchmark duration:\s*([\d.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def parse_json_line(payload: bytes) -> dict[str, object]:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload.decode("utf-8"))


def compute_request_level_metrics(json_path: Path, slo_target_ms: float) -> dict[str, float | int | None]:
    normalized_latencies: list[float] = []
    total_requests = 0
    slo_success_count = 0

    with json_path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            request = parse_json_line(raw_line)
            itl_samples = request.get("itl_samples") or []
            if not itl_samples:
                continue

            total_requests += 1
            queueing_time_ms = float(request.get("queueing_time_ms", 0.0) or 0.0)
            normlat_ms = (sum(itl_samples) + queueing_time_ms) / len(itl_samples)
            normalized_latencies.append(normlat_ms)
            if normlat_ms <= slo_target_ms:
                slo_success_count += 1

    if not normalized_latencies:
        return {
            "total_requests": 0,
            "slo_success_count": 0,
            "normlat_avg_ms": None,
            "normlat_p50_ms": None,
            "normlat_p90_ms": None,
            "normlat_p95_ms": None,
            "normlat_p99_ms": None,
            "slo_attainment": None,
        }

    values = np.asarray(normalized_latencies, dtype=np.float64)
    slo_attainment = 100.0 * slo_success_count / total_requests if total_requests else None
    return {
        "total_requests": total_requests,
        "slo_success_count": slo_success_count,
        "normlat_avg_ms": float(np.mean(values)),
        "normlat_p50_ms": float(np.percentile(values, 50)),
        "normlat_p90_ms": float(np.percentile(values, 90)),
        "normlat_p95_ms": float(np.percentile(values, 95)),
        "normlat_p99_ms": float(np.percentile(values, 99)),
        "slo_attainment": slo_attainment,
    }


def load_stage_rows(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []

    for dataset_spec in DATASET_SPECS:
        matched_dirs: list[Path] = []
        for stage_glob in dataset_spec.stage_globs:
            matched_dirs.extend(sorted(run_dir.glob(stage_glob)))

        for stage_dir in matched_dirs:
            summary_path = stage_dir / "sweep_summary.tsv"
            if not summary_path.exists():
                continue

            with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    if (row.get("status") or "").strip().lower() != "ok":
                        continue

                    json_path_str = (row.get("json_file") or "").strip()
                    log_path_str = (row.get("log_file") or "").strip()
                    if not json_path_str or not log_path_str:
                        continue

                    json_path = Path(json_path_str)
                    log_path = Path(log_path_str)
                    if not json_path.exists() or not log_path.exists():
                        continue

                    series_key = canonical_series_key(row.get("strategy", ""), stage_dir.name)
                    timestamp = (
                        parse_timestamp_from_path(json_path_str)
                        or parse_timestamp_from_path(log_path_str)
                        or datetime.fromtimestamp(json_path.stat().st_mtime)
                    )

                    all_rows.append(
                        {
                            "dataset_key": dataset_spec.key,
                            "dataset_label": dataset_spec.label,
                            "series_key": series_key,
                            "series_label": SERIES_LABEL_MAP.get(series_key, series_key),
                            "stage_dir": stage_dir.name,
                            "summary_path": str(summary_path.resolve()),
                            "strategy": row.get("strategy"),
                            "routing": row.get("routing"),
                            "rate": to_float(row.get("rate")),
                            "n_reqs": to_int(row.get("n_reqs")),
                            "itl_avg_ms_summary": to_float(row.get("itl_avg_ms")),
                            "itl_p99_ms_summary": to_float(row.get("itl_p99_ms")),
                            "queue_avg_ms_summary": to_float(row.get("queue_avg_ms")),
                            "queue_p99_ms_summary": to_float(row.get("queue_p99_ms")),
                            "decode_queue_avg_ms_summary": to_float(row.get("decode_queue_avg_ms")),
                            "decode_queue_p99_ms_summary": to_float(row.get("decode_queue_p99_ms")),
                            "log_path": str(log_path.resolve()),
                            "json_path": str(json_path.resolve()),
                            "sample_timestamp": timestamp,
                        }
                    )

    if not all_rows:
        raise SystemExit(f"No usable sweep summary rows found under {run_dir}")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        by=["dataset_key", "series_key", "rate", "sample_timestamp", "json_path"]
    ).reset_index(drop=True)

    duplicate_mask = df.duplicated(subset=["dataset_key", "series_key", "rate"], keep="last")
    duplicates = df.loc[duplicate_mask].copy()
    deduped = df.loc[~duplicate_mask].copy()
    return deduped, duplicates


def enrich_metrics(df: pd.DataFrame, slo_target_ms: float) -> pd.DataFrame:
    enriched_rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        json_path = Path(str(row["json_path"]))
        log_path = Path(str(row["log_path"]))

        metrics = compute_request_level_metrics(json_path, slo_target_ms)
        total_time_s = parse_total_time_from_log(log_path)
        total_requests = metrics["total_requests"]
        slo_success_count = metrics["slo_success_count"]
        request_throughput = None
        goodput = None
        if total_time_s and total_time_s > 0 and total_requests:
            request_throughput = total_requests / total_time_s
            goodput = slo_success_count / total_time_s

        row.update(metrics)
        row["total_time_s"] = total_time_s
        row["request_throughput_rps"] = request_throughput
        row["goodput_rps"] = goodput
        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    dataset_rank = {key: idx for idx, key in enumerate(DATASET_ORDER)}
    series_rank = {key: idx for idx, key in enumerate(SERIES_ORDER)}
    enriched["dataset_rank"] = enriched["dataset_key"].map(dataset_rank).fillna(len(dataset_rank))
    enriched["series_rank"] = enriched["series_key"].map(series_rank).fillna(len(series_rank))
    enriched = enriched.sort_values(
        by=["dataset_rank", "series_rank", "rate", "sample_timestamp"]
    ).reset_index(drop=True)
    return enriched.drop(columns=["dataset_rank", "series_rank"])


def calculate_x_at_y(x_values: np.ndarray, y_values: np.ndarray, target_y: float) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None

    for idx in range(len(y_values) - 1):
        x1 = float(x_values[idx])
        x2 = float(x_values[idx + 1])
        y1 = float(y_values[idx])
        y2 = float(y_values[idx + 1])
        if math.isnan(y1) or math.isnan(y2):
            continue
        if y1 == y2 == target_y:
            return x1
        if (y1 >= target_y > y2) or (y1 <= target_y < y2):
            ratio = (target_y - y1) / (y2 - y1)
            return x1 + ratio * (x2 - x1)
    return None


def calculate_last_x_at_y(x_values: np.ndarray, y_values: np.ndarray, target_y: float) -> float | None:
    if len(x_values) < 1 or len(y_values) < 1:
        return None

    last_crossing = None
    for idx in range(len(y_values) - 1):
        x1 = float(x_values[idx])
        x2 = float(x_values[idx + 1])
        y1 = float(y_values[idx])
        y2 = float(y_values[idx + 1])
        if math.isnan(y1) or math.isnan(y2):
            continue

        if y1 == target_y:
            last_crossing = x1
        if y2 == target_y:
            last_crossing = x2

        if (y1 - target_y) * (y2 - target_y) < 0:
            ratio = (target_y - y1) / (y2 - y1)
            last_crossing = x1 + ratio * (x2 - x1)

    if last_crossing is None and len(y_values) == 1 and float(y_values[0]) == target_y:
        return float(x_values[0])
    return last_crossing


def compute_dataset_x_max(dataset_df: pd.DataFrame) -> float | None:
    candidate_crossings: list[float] = []
    for metric_key in ("normlat_avg_ms", "normlat_p99_ms"):
        for _, series_df in dataset_df.groupby("series_key", sort=False):
            valid_df = series_df[["rate", metric_key]].dropna().sort_values("rate")
            if valid_df.empty:
                continue
            crossing = calculate_last_x_at_y(
                valid_df["rate"].to_numpy(dtype=float),
                valid_df[metric_key].to_numpy(dtype=float),
                LATENCY_X_CROSS_TARGET_MS,
            )
            if crossing is not None:
                candidate_crossings.append(crossing)

    if candidate_crossings:
        return max(candidate_crossings)

    max_rate = dataset_df["rate"].max()
    if pd.isna(max_rate):
        return None
    return float(max_rate)


def build_crossing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_key in DATASET_ORDER:
        dataset_df = df[df["dataset_key"] == dataset_key]
        if dataset_df.empty:
            continue
        for series_key in SERIES_ORDER + sorted(
            key for key in dataset_df["series_key"].unique() if key not in SERIES_ORDER
        ):
            series_df = dataset_df[dataset_df["series_key"] == series_key].sort_values("rate")
            if series_df.empty:
                continue
            crossing_rate = calculate_x_at_y(
                series_df["rate"].to_numpy(dtype=float),
                series_df["slo_attainment"].to_numpy(dtype=float),
                90.0,
            )
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": DATASET_LABEL_MAP.get(dataset_key, dataset_key),
                    "series_key": series_key,
                    "series_label": SERIES_LABEL_MAP.get(series_key, series_key),
                    "slo90_crossing_rate": crossing_rate,
                    "min_rate": float(series_df["rate"].min()),
                    "max_rate": float(series_df["rate"].max()),
                }
            )
    return pd.DataFrame(rows)


def plot_matrix(df: pd.DataFrame, output_dir: Path, slo_target_ms: float, normlat_log_scale: bool) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        ("slo_attainment", "SLO\nAttainment (%)"),
        ("normlat_avg_ms", "Avg Norm\nLatency (ms)"),
        ("normlat_p99_ms", "P99 Norm\nLatency (ms)"),
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(DATASET_ORDER),
        figsize=(3.0 * len(DATASET_ORDER), 2.2 * len(metrics) + 0.4),
        squeeze=False,
    )

    dataset_x_max_map = {
        dataset_key: compute_dataset_x_max(df[df["dataset_key"] == dataset_key])
        for dataset_key in DATASET_ORDER
    }

    for col_idx, dataset_key in enumerate(DATASET_ORDER):
        dataset_df = df[df["dataset_key"] == dataset_key]
        for row_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for series_key in SERIES_ORDER + sorted(
                key for key in dataset_df["series_key"].unique() if key not in SERIES_ORDER
            ):
                series_df = dataset_df[dataset_df["series_key"] == series_key].sort_values("rate")
                if series_df.empty:
                    continue

                style = SERIES_STYLE_MAP.get(
                    series_key,
                    {"color": "#444444", "marker": "o", "linestyle": "-"},
                )
                label = SERIES_LABEL_MAP.get(series_key, series_key)

                ax.plot(
                    series_df["rate"],
                    series_df[metric_key],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=5,
                    label=label,
                )

                if metric_key == "slo_attainment":
                    crossing_rate = calculate_x_at_y(
                        series_df["rate"].to_numpy(dtype=float),
                        series_df["slo_attainment"].to_numpy(dtype=float),
                        90.0,
                    )
                    if crossing_rate is not None:
                        ax.axvline(
                            x=crossing_rate,
                            color=style["color"],
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.75,
                        )

            if row_idx == 0:
                ax.set_title(DATASET_LABEL_MAP.get(dataset_key, dataset_key), fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Request Rate", fontsize=11)

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

            dataset_x_max = dataset_x_max_map.get(dataset_key)
            if dataset_x_max is not None:
                ax.set_xlim(right=dataset_x_max)

            if metric_key == "slo_attainment":
                ax.set_ylim(0, 105)
                ax.axhline(90.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            else:
                ax.axhline(slo_target_ms, color="#777777", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.set_ylim(bottom=LATENCY_Y_MIN_MS, top=LATENCY_Y_MAX_MS)
                ax.set_yticks(LATENCY_Y_TICKS)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
                ax.minorticks_off()

    legend_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []
    for series_key in SERIES_ORDER:
        if series_key not in df["series_key"].unique():
            continue
        style = SERIES_STYLE_MAP[series_key]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5,
            )
        )
        legend_labels.append(SERIES_LABEL_MAP[series_key])

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=max(1, len(legend_handles)),
            frameon=False,
            fontsize=11,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, wspace=0.25, hspace=0.28)

    png_path = plot_dir / "nano_longshort_matrix.png"
    pdf_path = plot_dir / "nano_longshort_matrix.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    enriched: pd.DataFrame,
    duplicates: pd.DataFrame,
    output_dir: Path,
    run_dir: Path,
    slo_target_ms: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    enriched_to_write = enriched.copy()
    duplicates_to_write = duplicates.copy()
    for frame in (enriched_to_write, duplicates_to_write):
        if "sample_timestamp" in frame.columns:
            frame["sample_timestamp"] = frame["sample_timestamp"].map(
                lambda value: value.isoformat(sep=" ") if pd.notna(value) else ""
            )

    enriched_to_write.to_csv(data_dir / "metrics_summary.tsv", sep="\t", index=False)
    duplicates_to_write.to_csv(data_dir / "metrics_duplicates_dropped.tsv", sep="\t", index=False)

    crossing_df = build_crossing_summary(enriched)
    crossing_df.to_csv(data_dir / "slo90_crossings.tsv", sep="\t", index=False)

    metadata = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "slo_target_ms": slo_target_ms,
                "dataset_order": ",".join(DATASET_ORDER),
            }
        ]
    )
    metadata.to_csv(data_dir / "metadata.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    output_dir = (args.output_dir or (run_dir / "plots_nano_longshort_matrix")).resolve()
    deduped, duplicates = load_stage_rows(run_dir)
    enriched = enrich_metrics(deduped, slo_target_ms=args.slo_target)
    write_outputs(
        enriched,
        duplicates,
        output_dir,
        run_dir=run_dir,
        slo_target_ms=args.slo_target,
    )
    plot_matrix(
        enriched,
        output_dir=output_dir,
        slo_target_ms=args.slo_target,
        normlat_log_scale=args.normlat_log_scale,
    )

    print(f"Saved summary to: {output_dir / 'data' / 'metrics_summary.tsv'}")
    print(f"Saved dropped duplicates to: {output_dir / 'data' / 'metrics_duplicates_dropped.tsv'}")
    print(f"Saved SLO 90 crossings to: {output_dir / 'data' / 'slo90_crossings.tsv'}")
    print(f"Saved plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()

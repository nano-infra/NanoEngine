#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SECTION_AVG_TEMPLATE = r"{section}\s*\n\s*Avg:\s*([0-9]+(?:\.[0-9]+)?)"


@dataclass
class ParsedLog:
    benchmark_time: str
    model: str
    dataset: str
    strategy: str
    rate: int
    ttft_ms: float
    tpot_without_queue_ms: float
    tpot_with_queue_ms: float
    tpot_with_queue_source: str
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one run-tag's benchmark logs as one figure per model/dataset "
            "with TTFT / TPOT-with-queue / TPOT-without-queue subplots."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the run-tag directory under bench_logs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/plots_run_metrics.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def must_extract(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"missing {label}")
    return match.group(1).strip()


def extract_optional_float(section_header: str, text: str) -> float | None:
    pattern = SECTION_AVG_TEMPLATE.format(section=re.escape(section_header))
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def normalize_model_name(model_name: str, model_path: str, log_path: Path) -> str:
    model_name = model_name.strip()
    model_path = model_path.strip()
    if re.fullmatch(r"[0-9a-f]{16,}", model_name):
        repo_match = re.search(r"models--([^/]+(?:--[^/]+)*)/snapshots", model_path)
        if repo_match:
            repo_name = repo_match.group(1).replace("--", "/")
            return repo_name.split("/")[-1]
        return log_path.parts[-5]
    return model_name


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_strategy_and_rate(settings: str) -> tuple[str, int]:
    strategy_match = re.match(r"([a-zA-Z0-9]+)_", settings)
    rate_match = re.search(r"(?:^|_)r(\d+)(?:_|$)", settings)
    if not strategy_match or not rate_match:
        raise ValueError("failed to parse strategy/rate from settings")
    return strategy_match.group(1), int(rate_match.group(1))


def parse_benchmark_log(path: Path) -> ParsedLog | None:
    text = read_text(path)
    if "================= Benchmark Metadata =================" not in text:
        return None
    if "--- Benchmark Results ---" not in text:
        return None

    benchmark_time = must_extract(r"^Time \(Beijing\):\s*([^\n]+)$", text, "benchmark_time")
    model_name = must_extract(r"^Model Name:\s*([^\n]+)$", text, "model_name")
    model_path = must_extract(r"^Model Path:\s*([^\n]+)$", text, "model_path")
    dataset = must_extract(r"^Dataset:\s*([^\n]+)$", text, "dataset")
    settings = must_extract(r"^Settings:\s*([^\n]+)$", text, "settings")
    strategy, rate = parse_strategy_and_rate(settings)

    ttft_ms = float(must_extract(r"^Average TTFT:\s*([0-9]+(?:\.[0-9]+)?)\s*ms$", text, "ttft"))
    tpot_without_queue_ms = float(
        must_extract(
            SECTION_AVG_TEMPLATE.format(
                section=re.escape("--- TPOT without Queueing Time (ms/token) ---")
            ),
            text,
            "tpot_without_queue",
        )
    )

    tpot_with_queue_ms = extract_optional_float(
        "--- TPOT with Queueing Time (ms/token) ---", text
    )
    tpot_with_queue_source = "tpot_with_queueing_section"
    if tpot_with_queue_ms is None:
        tpot_with_queue_ms = extract_optional_float(
            "--- ITL With Decode Queue (ms/token) ---", text
        )
        tpot_with_queue_source = "itl_with_decode_queue_section"
    if tpot_with_queue_ms is None:
        raise ValueError("missing TPOT-with-queue metric")

    return ParsedLog(
        benchmark_time=benchmark_time,
        model=normalize_model_name(model_name, model_path, path),
        dataset=dataset,
        strategy=strategy,
        rate=rate,
        ttft_ms=ttft_ms,
        tpot_without_queue_ms=tpot_without_queue_ms,
        tpot_with_queue_ms=tpot_with_queue_ms,
        tpot_with_queue_source=tpot_with_queue_source,
        log_file=str(path.resolve()),
    )


def collect_logs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.rglob("*.log")):
        parsed = parse_benchmark_log(path)
        if parsed is None:
            continue
        rows.append(parsed.__dict__)

    if not rows:
        raise SystemExit(f"No benchmark logs found under {run_dir}")

    df = pd.DataFrame(rows)
    df["dedupe_key"] = (
        df["model"].astype(str)
        + "||"
        + df["dataset"].astype(str)
        + "||"
        + df["strategy"].astype(str)
        + "||"
        + df["rate"].astype(str)
    )
    df = df.sort_values(
        by=["model", "dataset", "strategy", "rate", "benchmark_time", "log_file"]
    ).reset_index(drop=True)

    dedupe_mask = df.duplicated(subset=["model", "dataset", "strategy", "rate"], keep="last")
    duplicates = df.loc[dedupe_mask].copy()
    deduped = df.loc[~dedupe_mask].copy()
    return deduped, duplicates


def write_outputs(df: pd.DataFrame, duplicates: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    df = df.sort_values(by=["model", "dataset", "strategy", "rate"]).reset_index(drop=True)
    df.to_csv(output_dir / "data" / "metrics_summary.tsv", sep="\t", index=False)
    duplicates.to_csv(output_dir / "data" / "metrics_duplicates_dropped.tsv", sep="\t", index=False)


def plot_combined_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    strategies = sorted(df["strategy"].unique())
    marker_map = {strategy: markers[idx % len(markers)] for idx, strategy in enumerate(strategies)}
    metric_specs = [
        ("ttft_ms", "TTFT", "Average TTFT (ms)"),
        ("tpot_with_queue_ms", "TPOT With Queue", "Average TPOT with queue (ms/token)"),
        (
            "tpot_without_queue_ms",
            "TPOT Without Queue",
            "Average TPOT without queue (ms/token)",
        ),
    ]

    grouped = df.groupby(["model", "dataset"], sort=True)
    for (model, dataset), group in grouped:
        fig, axes = plt.subplots(
            nrows=len(metric_specs),
            ncols=1,
            figsize=(10, 13),
            sharex=True,
        )
        for ax, (metric_key, title_key, y_label) in zip(axes, metric_specs):
            for strategy, strategy_df in group.groupby("strategy", sort=True):
                strategy_df = strategy_df.sort_values("rate")
                ax.plot(
                    strategy_df["rate"],
                    strategy_df[metric_key],
                    marker=marker_map[strategy],
                    linewidth=2,
                    markersize=6,
                    label=strategy,
                )

            ax.set_title(title_key)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle="--", alpha=0.35)

        axes[0].legend(title="Parallel Strategy")
        axes[-1].set_xlabel("Request Rate (req/s)")
        fig.suptitle(f"{model} | {dataset}")
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)

        file_name = f"{slugify(model)}__{slugify(dataset)}__run_metrics.png"
        fig.savefig(plot_dir / file_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "plots_run_metrics")).resolve()

    df, duplicates = collect_logs(run_dir)
    write_outputs(df, duplicates, output_dir)

    plot_combined_metrics(df, output_dir)

    print(f"Saved summary to: {output_dir / 'data' / 'metrics_summary.tsv'}")
    print(f"Saved dropped duplicates to: {output_dir / 'data' / 'metrics_duplicates_dropped.tsv'}")
    print(f"Saved plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()

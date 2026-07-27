#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
NUM_CACHE_BLOCKS_RE = re.compile(r"Set num_cache_blocks=(\d+)")
LOCAL_CACHE_BLOCKS_RE = re.compile(r"num_local_kvcache_blocks:\s*(\d+)")
LOOP_COUNT_RE = re.compile(r"(?:--loop-count|loop_count)\s*(?:=|:)?\s*(\d+)")
BLOCK_SIZE_RE = re.compile(r"(?:^|\W)kvcache_block_size(?:\W|:|=)+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize decode-step logs over time: queue state, batch sizes, "
            "SP batch sizes, and KV-cache usage."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more log files or directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. Defaults to <file>_step_plots for a file or "
            "<dir>/plots_step_log_timeseries for a directory."
        ),
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=0,
        help="Skip the first N decode steps after parsing.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Only keep the first N decode steps after skipping.",
    )
    parser.add_argument(
        "--num-kvcache-blocks",
        type=int,
        default=None,
        help="Override total KV-cache blocks per rank when the log does not contain it.",
    )
    parser.add_argument(
        "--kvcache-block-size",
        type=int,
        default=64,
        help="Default KV-cache block size when it cannot be parsed from the log.",
    )
    return parser.parse_args()


def flatten_numbers(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(flatten_numbers(item))
        return flattened
    if value is None:
        return []
    return [float(value)]


def flatten_ints(value: object) -> list[int]:
    return [int(x) for x in flatten_numbers(value)]


def collapse_scalar(value: object) -> float:
    if isinstance(value, (list, tuple)):
        flattened = flatten_numbers(value)
        if not flattened:
            return float("nan")
        if len({int(x) for x in flattened}) == 1:
            return float(flattened[0])
        return float(max(flattened))
    if value is None:
        return float("nan")
    return float(value)


def parse_ms(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text.endswith("ms"):
        text = text[:-2]
    return float(text)


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "log"


def unique_output_name(file_path: Path) -> str:
    parts = [file_path.stem]
    parent = file_path.parent.name
    if parent:
        parts.insert(0, parent)
    grandparent = file_path.parent.parent.name
    if grandparent:
        parts.insert(0, grandparent)
    return sanitize_name("__".join(parts))


def resolve_input_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            files.append(input_path.resolve())
            continue
        if input_path.is_dir():
            for child in sorted(input_path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".log", ".out", ".txt"}:
                    files.append(child.resolve())
            continue
        raise SystemExit(f"Input does not exist: {input_path}")
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for file_path in files:
        if file_path not in seen:
            unique_files.append(file_path)
            seen.add(file_path)
    return unique_files


def infer_num_kvcache_blocks(text: str, override: int | None) -> int | None:
    if override is not None:
        return override
    match = NUM_CACHE_BLOCKS_RE.search(text)
    if match:
        return int(match.group(1))
    match = LOCAL_CACHE_BLOCKS_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def infer_loop_count(text: str) -> int | None:
    match = LOOP_COUNT_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def infer_kvcache_block_size(text: str, default: int) -> int:
    match = BLOCK_SIZE_RE.search(text)
    if match:
        return int(match.group(1))
    return default


def pad_matrix(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = max(len(row) for row in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=float)
    for row_idx, row in enumerate(rows):
        if row:
            matrix[row_idx, : len(row)] = row
    return matrix


def time_edges(xs: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if xs.size == 1:
        return np.array([xs[0] - 0.5, xs[0] + 0.5], dtype=float)
    mids = (xs[:-1] + xs[1:]) / 2.0
    first = xs[0] - (mids[0] - xs[0])
    last = xs[-1] + (xs[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def safe_nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmax(arr))


def choose_time_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "decode_elapsed_s" in df.columns and df["decode_elapsed_s"].notna().all():
        return df["decode_elapsed_s"].to_numpy(dtype=float), "Estimated Decode Time (s)"
    if "wall_elapsed_s" in df.columns and df["wall_elapsed_s"].notna().any():
        return (
            df["wall_elapsed_s"].ffill().fillna(0.0).to_numpy(dtype=float),
            "Wall Clock Elapsed (s)",
        )
    return df["step_idx"].to_numpy(dtype=float), "Decode Step"


def extract_step_records(
    text: str,
    *,
    num_kvcache_blocks_override: int | None,
    default_block_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    clean_text = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")
    num_kvcache_blocks = infer_num_kvcache_blocks(clean_text, num_kvcache_blocks_override)
    loop_count = infer_loop_count(clean_text)
    kvcache_block_size = infer_kvcache_block_size(clean_text, default_block_size)

    rows: list[dict[str, object]] = []
    sp_rows: list[list[float]] = []
    free_rows: list[list[float]] = []
    used_rows: list[list[float]] = []

    wall_clock_start: datetime | None = None
    decode_elapsed_s = 0.0

    for line in clean_text.splitlines():
        if "step - {" not in line:
            continue

        timestamp = parse_timestamp(line)
        payload = line.rsplit("step - ", 1)[-1].strip()

        try:
            data = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue

        if data.get("mode") != "decode":
            continue

        sp_batch_sizes_nested = data.get("sp_batch_sizes", [])
        free_blocks_nested = data.get("free_blocks", [])

        sp_batch_sizes = flatten_ints(sp_batch_sizes_nested)
        free_blocks = flatten_ints(free_blocks_nested)
        if not sp_batch_sizes or not free_blocks:
            continue

        dp_batch_sizes = [sum(flatten_ints(group)) for group in sp_batch_sizes_nested]

        if num_kvcache_blocks is not None:
            used_blocks = [num_kvcache_blocks - value for value in free_blocks]
            kv_util_pct = [
                100.0 * (num_kvcache_blocks - value) / num_kvcache_blocks
                for value in free_blocks
            ]
        else:
            used_blocks = [float("nan")] * len(free_blocks)
            kv_util_pct = [float("nan")] * len(free_blocks)

        if wall_clock_start is None and timestamp is not None:
            wall_clock_start = timestamp
        wall_elapsed_s = (
            (timestamp - wall_clock_start).total_seconds()
            if timestamp is not None and wall_clock_start is not None
            else float("nan")
        )

        itl_ms = parse_ms(data.get("itl"))
        sch_ovhd_ms = parse_ms(data.get("sch_ovhd"))
        post_sch_ovhd_ms = parse_ms(data.get("post_sch_ovhd"))
        step_duration_ms = (
            itl_ms * loop_count if loop_count is not None and np.isfinite(itl_ms) else float("nan")
        )
        if np.isfinite(step_duration_ms):
            decode_elapsed_s += step_duration_ms / 1000.0

        waiting_head_blocks = collapse_scalar(data.get("waiting_head_blocks"))
        waiting_total_blocks = collapse_scalar(data.get("waiting_total_blocks"))

        rows.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "",
                "wall_elapsed_s": wall_elapsed_s,
                "decode_elapsed_s": decode_elapsed_s if np.isfinite(step_duration_ms) else float("nan"),
                "itl_ms": itl_ms,
                "sch_ovhd_ms": sch_ovhd_ms,
                "post_sch_ovhd_ms": post_sch_ovhd_ms,
                "step_duration_ms": step_duration_ms,
                "waiting_reqs": float(data.get("waiting_reqs", float("nan"))),
                "waiting_head_blocks": waiting_head_blocks,
                "waiting_total_blocks": waiting_total_blocks,
                "total_batch_size": float(sum(dp_batch_sizes)),
                "avg_dp_batch_size": float(np.mean(dp_batch_sizes)) if dp_batch_sizes else float("nan"),
                "max_dp_batch_size": float(np.max(dp_batch_sizes)) if dp_batch_sizes else float("nan"),
                "min_dp_batch_size": float(np.min(dp_batch_sizes)) if dp_batch_sizes else float("nan"),
                "avg_sp_batch_size": float(np.mean(sp_batch_sizes)),
                "max_sp_batch_size": float(np.max(sp_batch_sizes)),
                "min_sp_batch_size": float(np.min(sp_batch_sizes)),
                "total_free_blocks": float(np.sum(free_blocks)),
                "mean_free_blocks": float(np.mean(free_blocks)),
                "min_free_blocks": float(np.min(free_blocks)),
                "max_free_blocks": float(np.max(free_blocks)),
                "total_used_blocks": float(np.nansum(used_blocks)),
                "mean_used_blocks": safe_nanmean(used_blocks),
                "mean_kv_util_pct": safe_nanmean(kv_util_pct),
                "max_kv_util_pct": safe_nanmax(kv_util_pct),
            }
        )
        sp_rows.append([float(x) for x in sp_batch_sizes])
        free_rows.append([float(x) for x in free_blocks])
        used_rows.append([float(x) for x in used_blocks])

    if not rows:
        for line in clean_text.splitlines():
            marker = "[BENCH_DIAG] "
            if marker not in line:
                continue
            try:
                data = json.loads(line.split(marker, 1)[1])
            except (json.JSONDecodeError, ValueError):
                continue
            hierarchical = data.get("hierarchical") or {}
            per_engine = hierarchical.get("per_engine") or {}
            engine_items = sorted(
                per_engine.items(), key=lambda item: int(item[0])
            )
            rank_loads_by_engine = [
                sorted(
                    engine.get("rank_loads") or [],
                    key=lambda load: int(load["global_rank"]),
                )
                for _, engine in engine_items
            ]
            rank_loads = [
                load
                for engine_loads in rank_loads_by_engine
                for load in engine_loads
            ]
            if not rank_loads:
                continue

            sp_batch_sizes = [
                int(load["master_batch_size"]) for load in rank_loads
            ]
            free_blocks = [
                int(load["free_blocks"]) for load in rank_loads
            ]
            total_blocks = [
                int(load["total_blocks"]) for load in rank_loads
            ]
            dp_batch_sizes = [
                sum(
                    int(load["master_batch_size"])
                    for load in engine_loads
                )
                for engine_loads in rank_loads_by_engine
            ]
            used_blocks = [
                total - free
                for total, free in zip(
                    total_blocks, free_blocks, strict=True
                )
            ]
            kv_util_pct = [
                100.0 * used / total if total > 0 else float("nan")
                for used, total in zip(
                    used_blocks, total_blocks, strict=True
                )
            ]
            interval = data.get("hierarchical_interval") or {}
            itl_ms = float(
                interval.get("decode_itl_ms_mean", float("nan"))
                if interval.get("decode_itl_ms_mean") is not None
                else float("nan")
            )
            elapsed_s = float(data.get("elapsed_s", float("nan")))
            rows.append(
                {
                    "timestamp": "",
                    "wall_elapsed_s": elapsed_s,
                    "decode_elapsed_s": elapsed_s,
                    "itl_ms": itl_ms,
                    "sch_ovhd_ms": float("nan"),
                    "post_sch_ovhd_ms": float("nan"),
                    "step_duration_ms": (
                        itl_ms * loop_count
                        if loop_count is not None and np.isfinite(itl_ms)
                        else float("nan")
                    ),
                    "waiting_reqs": float(
                        hierarchical.get(
                            "waiting_requests", float("nan")
                        )
                    ),
                    "waiting_head_blocks": float("nan"),
                    "waiting_total_blocks": float("nan"),
                    "total_batch_size": float(sum(dp_batch_sizes)),
                    "avg_dp_batch_size": float(np.mean(dp_batch_sizes)),
                    "max_dp_batch_size": float(np.max(dp_batch_sizes)),
                    "min_dp_batch_size": float(np.min(dp_batch_sizes)),
                    "avg_sp_batch_size": float(np.mean(sp_batch_sizes)),
                    "max_sp_batch_size": float(np.max(sp_batch_sizes)),
                    "min_sp_batch_size": float(np.min(sp_batch_sizes)),
                    "total_free_blocks": float(np.sum(free_blocks)),
                    "mean_free_blocks": float(np.mean(free_blocks)),
                    "min_free_blocks": float(np.min(free_blocks)),
                    "max_free_blocks": float(np.max(free_blocks)),
                    "total_used_blocks": float(np.sum(used_blocks)),
                    "mean_used_blocks": float(np.mean(used_blocks)),
                    "mean_kv_util_pct": float(np.mean(kv_util_pct)),
                    "max_kv_util_pct": float(np.max(kv_util_pct)),
                }
            )
            sp_rows.append([float(x) for x in sp_batch_sizes])
            free_rows.append([float(x) for x in free_blocks])
            used_rows.append([float(x) for x in used_blocks])
            if (
                num_kvcache_blocks is None
                and len(set(total_blocks)) == 1
            ):
                num_kvcache_blocks = total_blocks[0]

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "step_idx", np.arange(1, len(df) + 1))

    metadata = {
        "num_kvcache_blocks": num_kvcache_blocks,
        "kvcache_block_size": kvcache_block_size,
        "loop_count": loop_count,
    }
    return df, pad_matrix(sp_rows), pad_matrix(free_rows), pad_matrix(used_rows), metadata


def write_matrix_tsv(
    path: Path,
    df: pd.DataFrame,
    matrix: np.ndarray,
    *,
    prefix: str,
) -> None:
    matrix_df = pd.DataFrame(matrix, columns=[f"{prefix}_{idx}" for idx in range(matrix.shape[1])])
    out_df = pd.concat(
        [
            df[["step_idx", "timestamp", "wall_elapsed_s", "decode_elapsed_s"]].reset_index(drop=True),
            matrix_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out_df.to_csv(path, sep="\t", index=False)


def plot_overview(
    df: pd.DataFrame,
    metadata: dict[str, object],
    output_path: Path,
    title: str,
) -> None:
    xs, xlabel = choose_time_axis(df)

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 24), sharex=True)

    axes[0].plot(xs, df["waiting_reqs"], color="#7c3aed", linewidth=2, label="waiting_reqs")
    axes[0].set_title("Queue: Waiting Requests")
    axes[0].set_ylabel("Requests")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        xs,
        df["waiting_head_blocks"],
        color="#f59e0b",
        linewidth=2,
        label="waiting_head_blocks",
    )
    axes[1].plot(
        xs,
        df["waiting_total_blocks"],
        color="#dc2626",
        linewidth=2,
        label="waiting_total_blocks",
    )
    axes[1].set_title("Queue: Waiting Blocks")
    axes[1].set_ylabel("Blocks")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        xs,
        df["total_batch_size"],
        color="#2563eb",
        linewidth=2,
        label="total_batch_size",
    )
    axes[2].plot(
        xs,
        df["max_dp_batch_size"],
        color="#0891b2",
        linewidth=1.8,
        label="max_dp_batch_size",
    )
    axes[2].plot(
        xs,
        df["avg_sp_batch_size"],
        color="#16a34a",
        linewidth=1.8,
        label="avg_sp_batch_size",
    )
    axes[2].plot(
        xs,
        df["max_sp_batch_size"],
        color="#65a30d",
        linewidth=1.8,
        label="max_sp_batch_size",
    )
    axes[2].set_title("Batch Size and SP Batch Size")
    axes[2].set_ylabel("Sequences")
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(loc="upper right")

    axes[3].plot(
        xs,
        df["total_free_blocks"],
        color="#0f766e",
        linewidth=2,
        label="total_free_blocks",
    )
    if metadata.get("num_kvcache_blocks") is not None:
        axes[3].plot(
            xs,
            df["total_used_blocks"],
            color="#b91c1c",
            linewidth=2,
            label="total_used_blocks",
        )
        ax3b = axes[3].twinx()
        ax3b.plot(
            xs,
            df["mean_kv_util_pct"],
            color="#7c2d12",
            linewidth=1.6,
            linestyle="--",
            label="mean_kv_util_pct",
        )
        ax3b.plot(
            xs,
            df["max_kv_util_pct"],
            color="#ea580c",
            linewidth=1.6,
            linestyle=":",
            label="max_kv_util_pct",
        )
        ax3b.set_ylabel("Utilization (%)")
        handles_a, labels_a = axes[3].get_legend_handles_labels()
        handles_b, labels_b = ax3b.get_legend_handles_labels()
        axes[3].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    else:
        axes[3].legend(loc="upper right")
    axes[3].set_title("KV Cache State")
    axes[3].set_ylabel("Blocks")
    axes[3].grid(True, linestyle="--", alpha=0.35)

    axes[4].plot(xs, df["itl_ms"], color="#1d4ed8", linewidth=2, label="itl_ms")
    axes[4].plot(
        xs,
        df["sch_ovhd_ms"],
        color="#9333ea",
        linewidth=1.8,
        label="sch_ovhd_ms",
    )
    axes[4].plot(
        xs,
        df["post_sch_ovhd_ms"],
        color="#db2777",
        linewidth=1.8,
        label="post_sch_ovhd_ms",
    )
    axes[4].set_title("Per-Step Latency and Scheduling Overhead")
    axes[4].set_ylabel("ms")
    axes[4].set_xlabel(xlabel)
    axes[4].grid(True, linestyle="--", alpha=0.35)
    axes[4].legend(loc="upper right")

    subtitle = title
    if metadata.get("num_kvcache_blocks") is not None:
        subtitle += f" | num_cache_blocks={metadata['num_kvcache_blocks']}"
    if metadata.get("loop_count") is not None:
        subtitle += f" | loop_count={metadata['loop_count']}"
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    ax: plt.Axes,
    xs: np.ndarray,
    matrix: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    if matrix.size == 0:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    x_edges = time_edges(xs)
    y_edges = np.arange(matrix.shape[1] + 1, dtype=float) - 0.5
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix.T,
        shading="auto",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_ylabel("Rank")
    plt.colorbar(mesh, ax=ax, label=colorbar_label)


def plot_heatmaps(
    df: pd.DataFrame,
    sp_matrix: np.ndarray,
    free_matrix: np.ndarray,
    used_matrix: np.ndarray,
    metadata: dict[str, object],
    output_path: Path,
    title: str,
) -> None:
    xs, xlabel = choose_time_axis(df)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18), sharex=True)

    plot_heatmap(
        axes[0],
        xs,
        sp_matrix,
        title="SP Batch Size Per Rank",
        colorbar_label="Sequences",
        cmap="viridis",
    )
    plot_heatmap(
        axes[1],
        xs,
        free_matrix,
        title="Free KV Blocks Per Rank",
        colorbar_label="Blocks",
        cmap="magma",
    )

    if metadata.get("num_kvcache_blocks") is not None and used_matrix.size > 0:
        util_matrix = 100.0 * used_matrix / float(metadata["num_kvcache_blocks"])
        plot_heatmap(
            axes[2],
            xs,
            util_matrix,
            title="KV Cache Utilization Per Rank",
            colorbar_label="Utilization (%)",
            cmap="plasma",
        )
    else:
        plot_heatmap(
            axes[2],
            xs,
            used_matrix,
            title="Used KV Blocks Per Rank",
            colorbar_label="Blocks",
            cmap="plasma",
        )

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def default_output_dir(base_output_dir: Path | None, input_path: Path) -> Path:
    if base_output_dir is not None:
        return base_output_dir
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}_step_plots"
    return input_path / "plots_step_log_timeseries"


def process_file(
    file_path: Path,
    *,
    output_base: Path,
    skip_steps: int,
    max_steps: int | None,
    num_kvcache_blocks: int | None,
    kvcache_block_size: int,
) -> Path | None:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    df, sp_matrix, free_matrix, used_matrix, metadata = extract_step_records(
        text,
        num_kvcache_blocks_override=num_kvcache_blocks,
        default_block_size=kvcache_block_size,
    )

    if df.empty:
        print(f"Skip {file_path}: no decode-step records found.")
        return None

    if skip_steps > 0:
        df = df.iloc[skip_steps:].reset_index(drop=True)
        df["step_idx"] = np.arange(1, len(df) + 1)
        sp_matrix = sp_matrix[skip_steps:, :]
        free_matrix = free_matrix[skip_steps:, :]
        used_matrix = used_matrix[skip_steps:, :]

    if max_steps is not None:
        df = df.iloc[:max_steps].reset_index(drop=True)
        df["step_idx"] = np.arange(1, len(df) + 1)
        sp_matrix = sp_matrix[:max_steps, :]
        free_matrix = free_matrix[:max_steps, :]
        used_matrix = used_matrix[:max_steps, :]

    if df.empty:
        print(f"Skip {file_path}: no decode-step records remain after filtering.")
        return None

    if output_base.name == f"{file_path.stem}_step_plots":
        output_dir = output_base
    else:
        output_dir = output_base / unique_output_name(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "timeseries_summary.tsv", sep="\t", index=False)
    write_matrix_tsv(output_dir / "sp_batch_sizes.tsv", df, sp_matrix, prefix="rank")
    write_matrix_tsv(output_dir / "free_blocks.tsv", df, free_matrix, prefix="rank")
    write_matrix_tsv(output_dir / "used_kvcache_blocks.tsv", df, used_matrix, prefix="rank")

    title = file_path.name
    plot_overview(df, metadata, output_dir / "overview.png", title)
    plot_heatmaps(df, sp_matrix, free_matrix, used_matrix, metadata, output_dir / "heatmaps.png", title)

    meta_lines = [
        f"source_log\t{file_path}",
        f"decode_steps\t{len(df)}",
        f"num_kvcache_blocks\t{metadata.get('num_kvcache_blocks', '')}",
        f"kvcache_block_size\t{metadata.get('kvcache_block_size', '')}",
        f"loop_count\t{metadata.get('loop_count', '')}",
    ]
    (output_dir / "metadata.tsv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Saved plots for {file_path} -> {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    input_files = resolve_input_files(args.inputs)
    if not input_files:
        raise SystemExit("No input files found.")

    if len(args.inputs) == 1:
        base_output = default_output_dir(args.output_dir, args.inputs[0].resolve())
    else:
        base_output = args.output_dir or Path.cwd() / "plots_step_log_timeseries"
    base_output.mkdir(parents=True, exist_ok=True)

    generated = 0
    for file_path in input_files:
        if process_file(
            file_path,
            output_base=base_output,
            skip_steps=args.skip_steps,
            max_steps=args.max_steps,
            num_kvcache_blocks=args.num_kvcache_blocks,
            kvcache_block_size=args.kvcache_block_size,
        ):
            generated += 1

    if generated == 0:
        raise SystemExit(
            "No plots were generated. These logs may not contain 'step - {...}' decode-step records."
        )


if __name__ == "__main__":
    main()

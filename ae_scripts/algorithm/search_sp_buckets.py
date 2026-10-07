#!/usr/bin/env python3
"""Derive a monotonic sequence-length-to-SP-size policy from pipeline latency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "deepseek_v3_h200_pipeline_p50.csv"
DEFAULT_OUTPUT = ROOT / "reproduced_policy.json"
METRIC = "end_to_end_master_p50_us"
SP_SIZES = tuple(range(1, 9))
MIN_POINTS_PER_BUCKET = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a monotonic sequence-length-to-SP-size bucket policy."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_grid(
    path: Path,
) -> tuple[list[int], dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    latency: dict[int, dict[int, float]] = {}
    weight: dict[int, dict[int, float]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"seq_len", "cp_size", METRIC, "sample_count"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            seq_len = int(row["seq_len"])
            sp_size = int(row["cp_size"])
            if sp_size not in SP_SIZES:
                continue
            if sp_size in latency.setdefault(seq_len, {}):
                raise ValueError(f"Duplicate setting: seq_len={seq_len}, SP={sp_size}")
            latency[seq_len][sp_size] = float(row[METRIC])
            weight.setdefault(seq_len, {})[sp_size] = float(row["sample_count"])

    seq_lens = sorted(latency)
    if not seq_lens:
        raise ValueError("Input CSV contains no measurements")
    for seq_len in seq_lens:
        missing_sp = sorted(set(SP_SIZES).difference(latency[seq_len]))
        if missing_sp:
            raise ValueError(f"seq_len={seq_len} is missing SP sizes {missing_sp}")
        if any(weight[seq_len][sp_size] <= 0 for sp_size in SP_SIZES):
            raise ValueError(f"seq_len={seq_len} has a non-positive sample count")
    return seq_lens, latency, weight


def prefix_regret(
    seq_lens: list[int],
    latency: dict[int, dict[int, float]],
    weight: dict[int, dict[int, float]],
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for sp_size in SP_SIZES:
        prefix = [0.0]
        running = 0.0
        for seq_len in seq_lens:
            oracle = min(latency[seq_len].values())
            running += (
                latency[seq_len][sp_size] - oracle
            ) * weight[seq_len][sp_size]
            prefix.append(running)
        result[sp_size] = prefix
    return result


def search_policy(
    seq_lens: list[int],
    latency: dict[int, dict[int, float]],
    weight: dict[int, dict[int, float]],
) -> list[dict[str, int]]:
    """Minimize weighted regret with non-decreasing SP sizes; SPs may be skipped."""
    point_count = len(seq_lens)
    sp_count = len(SP_SIZES)
    prefix = prefix_regret(seq_lens, latency, weight)
    infinity = float("inf")

    # cost[end][sp_idx]: best cost for points [0, end), ending with SP_SIZES[sp_idx].
    cost = [[infinity] * sp_count for _ in range(point_count + 1)]
    previous_point = [[-1] * sp_count for _ in range(point_count + 1)]
    previous_sp = [[-1] * sp_count for _ in range(point_count + 1)]

    for sp_idx, sp_size in enumerate(SP_SIZES):
        for end in range(MIN_POINTS_PER_BUCKET, point_count + 1):
            cost[end][sp_idx] = prefix[sp_size][end]
            previous_point[end][sp_idx] = 0

            for split in range(
                MIN_POINTS_PER_BUCKET,
                end - MIN_POINTS_PER_BUCKET + 1,
            ):
                prior_sp_idx = min(
                    range(sp_idx),
                    key=lambda candidate: cost[split][candidate],
                    default=-1,
                )
                if prior_sp_idx < 0:
                    continue
                candidate = cost[split][prior_sp_idx] + (
                    prefix[sp_size][end] - prefix[sp_size][split]
                )
                if candidate < cost[end][sp_idx]:
                    cost[end][sp_idx] = candidate
                    previous_point[end][sp_idx] = split
                    previous_sp[end][sp_idx] = prior_sp_idx

    sp_idx = min(range(sp_count), key=lambda idx: cost[point_count][idx])
    if cost[point_count][sp_idx] == infinity:
        raise RuntimeError("No feasible monotonic policy")

    segments: list[tuple[int, int, int]] = []
    end = point_count
    while end > 0:
        start = previous_point[end][sp_idx]
        if start < 0:
            raise RuntimeError("Invalid dynamic-programming backtrace")
        segments.append((start, end - 1, SP_SIZES[sp_idx]))
        end, sp_idx = start, previous_sp[end][sp_idx]
    segments.reverse()

    intervals: list[dict[str, int]] = []
    for index, (start, end, sp_size) in enumerate(segments):
        low = seq_lens[start] if index == 0 else intervals[-1]["seq_len_high"] + 1
        if index + 1 == len(segments):
            high = seq_lens[end]
        else:
            next_start = segments[index + 1][0]
            high = (seq_lens[end] + seq_lens[next_start]) // 2
        intervals.append(
            {"sp_size": sp_size, "seq_len_low": low, "seq_len_high": high}
        )
    return intervals


def summarize(
    seq_lens: list[int],
    latency: dict[int, dict[int, float]],
    weight: dict[int, dict[int, float]],
    intervals: list[dict[str, int]],
) -> dict[str, float | int]:
    total_weight = 0.0
    weighted_regret = 0.0
    regrets: list[float] = []
    exact_matches = 0

    for seq_len in seq_lens:
        assigned = next(
            interval["sp_size"]
            for interval in intervals
            if interval["seq_len_low"] <= seq_len <= interval["seq_len_high"]
        )
        oracle = min(
            SP_SIZES,
            key=lambda sp_size: (latency[seq_len][sp_size], sp_size),
        )
        regret = latency[seq_len][assigned] - latency[seq_len][oracle]
        sample_weight = weight[seq_len][assigned]
        regrets.append(regret)
        weighted_regret += regret * sample_weight
        total_weight += sample_weight
        exact_matches += assigned == oracle

    sorted_regrets = sorted(regrets)
    p95_index = round(0.95 * (len(sorted_regrets) - 1))
    return {
        "point_count": len(seq_lens),
        "total_sample_weight": total_weight,
        "avg_regret_us": weighted_regret / total_weight,
        "p95_regret_us": sorted_regrets[p95_index],
        "max_regret_us": max(regrets),
        "exact_best_match_points": exact_matches,
    }


def main() -> None:
    args = parse_args()
    seq_lens, latency, weight = load_grid(args.input)
    intervals = search_policy(seq_lens, latency, weight)
    summary = summarize(seq_lens, latency, weight, intervals)

    payload = {
        "input": str(args.input.resolve()),
        "metric": METRIC,
        "sp_sizes": list(SP_SIZES),
        "min_points_per_bucket": MIN_POINTS_PER_BUCKET,
        "allow_skip_sp": True,
        "intervals": intervals,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Derived monotonic SP bucket policy:")
    for interval in intervals:
        print(
            f"  SP={interval['sp_size']}: "
            f"[{interval['seq_len_low']}, {interval['seq_len_high']}]"
        )
    print(f"Saved policy to {args.output.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract the Fig. 5 rank snapshot from a raw vLLM ``frontend.log``.

The E2E launcher deliberately stops at raw logs.  This figure-owned script
aligns complete DP-rank statistic groups, selects one point in experiment
progress, and derives the Attention token count and DeepEP batch size used by
the Fig. 5 microbenchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_RANK_COUNT = 32
DEFAULT_TIME_PERCENT = 60
DEFAULT_KV_BLOCK_SIZE = 64

KV_CAPACITY_RE = re.compile(
    r"GPU KV cache size:\s*(?P<tokens>[\d,]+)\s+tokens"
)
RANK_STATS_RE = re.compile(
    r"Engine\s+(?P<rank>\d+):.*?"
    r"Running:\s*(?P<running>\d+)\s+reqs,\s*"
    r"Waiting:\s*(?P<waiting>\d+)\s+reqs,\s*"
    r"Waiting tokens:\s*(?P<waiting_tokens>\d+),\s*"
    r"Waiting head tokens:\s*(?P<waiting_head_tokens>\d+),\s*"
    r"GPU KV cache usage:\s*(?P<kv_usage>[\d.]+)%"
)


@dataclass(frozen=True)
class RawRankStat:
    rank: int
    running_requests: int
    waiting_requests: int
    waiting_tokens: int
    waiting_head_tokens: int
    gpu_kv_usage_pct: float


@dataclass(frozen=True)
class DerivedRankStat:
    rank: int
    running_requests: int
    waiting_requests: int
    waiting_tokens: int
    waiting_head_tokens: int
    gpu_kv_usage_pct: float
    kv_capacity_tokens: int
    attention_tokens: int
    deepep_batch_size: int


def percent(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("expected an integer in [0, 100]")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a complete rank snapshot and derive Fig. 5 Attention "
            "tokens and DeepEP batch sizes from a raw vLLM frontend.log."
        )
    )
    parser.add_argument("frontend_log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--time-percent", type=percent, default=DEFAULT_TIME_PERCENT
    )
    parser.add_argument(
        "--rank-count", type=positive_int, default=DEFAULT_RANK_COUNT
    )
    parser.add_argument(
        "--kv-block-size", type=positive_int, default=DEFAULT_KV_BLOCK_SIZE
    )
    parser.add_argument(
        "--kv-capacity-tokens",
        type=positive_int,
        help=(
            "Override per-rank KV capacity. By default it is read from "
            "'GPU KV cache size' in frontend.log."
        ),
    )
    return parser.parse_args()


def parse_frontend_log(
    path: Path,
    rank_count: int,
) -> tuple[list[list[RawRankStat]], set[int], int]:
    expected_ranks = set(range(rank_count))
    capacities: set[int] = set()
    snapshots: list[list[RawRankStat]] = []
    current: dict[int, RawRankStat] = {}
    last_rank: int | None = None
    discarded_partial_groups = 0

    def finish_group() -> None:
        nonlocal discarded_partial_groups
        if not current:
            return
        if set(current) == expected_ranks:
            snapshots.append([current[rank] for rank in range(rank_count)])
        else:
            discarded_partial_groups += 1

    with path.open("r", encoding="utf-8", errors="ignore") as input_file:
        for line in input_file:
            capacity_match = KV_CAPACITY_RE.search(line)
            if capacity_match:
                capacities.add(
                    int(capacity_match.group("tokens").replace(",", ""))
                )

            stats_match = RANK_STATS_RE.search(line)
            if stats_match is None:
                continue
            rank = int(stats_match.group("rank"))
            if rank not in expected_ranks:
                continue
            if last_rank is not None and (rank <= last_rank or rank in current):
                finish_group()
                current = {}
            current[rank] = RawRankStat(
                rank=rank,
                running_requests=int(stats_match.group("running")),
                waiting_requests=int(stats_match.group("waiting")),
                waiting_tokens=int(stats_match.group("waiting_tokens")),
                waiting_head_tokens=int(stats_match.group("waiting_head_tokens")),
                gpu_kv_usage_pct=float(stats_match.group("kv_usage")),
            )
            last_rank = rank

    finish_group()
    return snapshots, capacities, discarded_partial_groups


def choose_snapshot_index(snapshot_count: int, time_percent: int) -> int:
    if snapshot_count <= 0:
        raise ValueError("no complete rank snapshots were found")
    # This is intentionally identical to the historical Fig. 5 postprocessor.
    return int(round((snapshot_count - 1) * time_percent / 100.0))


def round_to_multiple(value: float, multiple: int) -> int:
    return int(math.floor(value / multiple + 0.5) * multiple)


def derive_rows(
    snapshot: list[RawRankStat],
    kv_capacity_tokens: int,
    kv_block_size: int,
) -> list[DerivedRankStat]:
    rows: list[DerivedRankStat] = []
    for raw in snapshot:
        attention_tokens = round_to_multiple(
            raw.gpu_kv_usage_pct / 100.0 * kv_capacity_tokens,
            kv_block_size,
        )
        rows.append(
            DerivedRankStat(
                **asdict(raw),
                kv_capacity_tokens=kv_capacity_tokens,
                attention_tokens=attention_tokens,
                # In the decode-dominated snapshot, each running request
                # contributes one routed token to that decode step.
                deepep_batch_size=raw.running_requests,
            )
        )
    return rows


def write_tsv(path: Path, rows: list[DerivedRankStat]) -> None:
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames, delimiter="\t"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_compatibility_snapshot(
    path: Path,
    *,
    rows: list[DerivedRankStat],
    time_percent: int,
    selected_index: int,
    snapshot_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            f"[time={time_percent}% step={selected_index + 1}/{snapshot_count}]\n"
        )
        output_file.write("rank\trunning\tgpu_kv_usage_pct\n")
        for row in rows:
            output_file.write(
                f"{row.rank}\t{row.running_requests}\t"
                f"{row.gpu_kv_usage_pct:g}\n"
            )
        output_file.write("\n")


def main() -> int:
    args = parse_args()
    frontend_log = args.frontend_log.expanduser().resolve()
    if not frontend_log.is_file():
        raise SystemExit(f"frontend.log not found: {frontend_log}")

    snapshots, capacities, discarded = parse_frontend_log(
        frontend_log, args.rank_count
    )
    selected_index = choose_snapshot_index(len(snapshots), args.time_percent)

    if args.kv_capacity_tokens is not None:
        kv_capacity_tokens = args.kv_capacity_tokens
        capacity_source = "--kv-capacity-tokens"
    elif len(capacities) == 1:
        kv_capacity_tokens = next(iter(capacities))
        capacity_source = "frontend.log"
    elif not capacities:
        raise SystemExit(
            "No 'GPU KV cache size' was found; pass --kv-capacity-tokens."
        )
    else:
        raise SystemExit(
            "Multiple KV capacities were found in frontend.log: "
            f"{sorted(capacities)}; pass --kv-capacity-tokens explicitly."
        )

    rows = derive_rows(
        snapshots[selected_index], kv_capacity_tokens, args.kv_block_size
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rank_snapshot_time{args.time_percent}"
    tsv_path = output_dir / f"{stem}.tsv"
    json_path = output_dir / f"{stem}.json"
    compatibility_path = output_dir / f"{stem}.txt"

    write_tsv(tsv_path, rows)
    write_compatibility_snapshot(
        compatibility_path,
        rows=rows,
        time_percent=args.time_percent,
        selected_index=selected_index,
        snapshot_count=len(snapshots),
    )

    attention_rank_tokens = [row.attention_tokens for row in rows]
    deepep_batch_sizes = [row.deepep_batch_size for row in rows]
    attention_mean = round_to_multiple(
        statistics.mean(attention_rank_tokens), args.kv_block_size
    )
    deepep_mean = statistics.mean(deepep_batch_sizes)
    metadata = {
        "source_frontend_log": str(frontend_log),
        "rank_count": args.rank_count,
        "time_percent": args.time_percent,
        "complete_snapshot_count": len(snapshots),
        "selected_snapshot_index_zero_based": selected_index,
        "selected_step_one_based": selected_index + 1,
        "discarded_partial_groups": discarded,
        "kv_capacity_tokens": kv_capacity_tokens,
        "kv_capacity_source": capacity_source,
        "kv_block_size": args.kv_block_size,
        "attention_token_formula": (
            "round_to_kv_block(gpu_kv_usage_pct / 100 * kv_capacity_tokens)"
        ),
        "deepep_batch_formula": "running_requests (one decode token/request)",
        "attention": {
            "rank_tokens": attention_rank_tokens,
            "mean_tokens_rounded": attention_mean,
            "microbenchmark_token_cases": sorted(
                set(attention_rank_tokens) | {attention_mean}
            ),
        },
        "deepep": {
            "rank_batch_sizes": deepep_batch_sizes,
            "mean_batch_size": deepep_mean,
            "microbenchmark_batch_cases": sorted(
                set(deepep_batch_sizes) | {math.ceil(deepep_mean)}
            ),
        },
        "outputs": {
            "tsv": str(tsv_path),
            "compatibility_snapshot": str(compatibility_path),
        },
    }
    json_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Selected time={args.time_percent}% snapshot "
        f"step={selected_index + 1}/{len(snapshots)} with {len(rows)} ranks."
    )
    print(
        f"Attention: {len(set(attention_rank_tokens))} unique rank token "
        f"counts; rounded mean={attention_mean}."
    )
    print(
        f"DeepEP: {len(set(deepep_batch_sizes))} unique rank batch sizes; "
        f"mean={deepep_mean:g}."
    )
    print(f"TSV: {tsv_path}")
    print(f"Metadata: {json_path}")
    print(f"Compatibility snapshot: {compatibility_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

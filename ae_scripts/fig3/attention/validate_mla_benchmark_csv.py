#!/usr/bin/env python3
"""Validate that a fresh MLA CSV contains the requested benchmark cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TOTAL_TOKENS = (
    "65536,131072,196608,262144,393216,524288,655360,786432,917504,1048576"
)
DEFAULT_BATCH_SIZES = "1,128,1024"
REQUIRED_COLUMNS = {
    "seq_len",
    "batch_size",
    "total_token_num",
    "time_us",
    "time_us_p10",
    "time_us_p90",
}


def validate_csv(
    csv_path: Path, total_tokens: list[int], batch_sizes: list[int]
) -> int:
    expected = {
        (total, batch, total // batch)
        for total in total_tokens
        for batch in batch_sizes
    }

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(f"missing required columns: {sorted(missing_columns)}")
        rows = list(reader)

    actual = set()
    for row in rows:
        key = (
            int(row["total_token_num"]),
            int(row["batch_size"]),
            int(row["seq_len"]),
        )
        if key in actual:
            raise ValueError(f"duplicate benchmark case: {key}")
        actual.add(key)

        p10 = float(row["time_us_p10"])
        median = float(row["time_us"])
        p90 = float(row["time_us_p90"])
        if not (0 < p10 <= median <= p90):
            raise ValueError(
                f"invalid latency statistics for {key}: "
                f"p10={p10}, median={median}, p90={p90}"
            )

    missing_cases = expected - actual
    unexpected_cases = actual - expected
    if missing_cases or unexpected_cases:
        raise ValueError(
            f"case mismatch: missing={sorted(missing_cases)}, "
            f"unexpected={sorted(unexpected_cases)}"
        )
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, found {len(rows)}")

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that an MLA CSV contains the requested cases."
    )
    parser.add_argument(
        "--total-tokens",
        default=DEFAULT_TOTAL_TOKENS,
        help="Comma-separated expected total-token counts",
    )
    parser.add_argument(
        "--batch-sizes",
        default=DEFAULT_BATCH_SIZES,
        help="Comma-separated expected batch sizes",
    )
    parser.add_argument("csv_path", type=Path, help="MLA benchmark CSV to validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_tokens = [int(value) for value in args.total_tokens.split(",") if value]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    if not total_tokens or not batch_sizes:
        raise ValueError("total-token and batch-size lists must not be empty")
    row_count = validate_csv(args.csv_path.resolve(), total_tokens, batch_sizes)
    print(
        f"Validated {row_count} requested benchmark rows in {args.csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()

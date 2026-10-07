#!/usr/bin/env python3
"""Convert per-token DeepEP logs to the CSV consumed by the figure script."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional


FIELDS = [
    "token",
    "status",
    "dispatch_combine_bw_gbps",
    "dispatch_combine_avg_us",
    "dispatch_bw_gbps",
    "dispatch_avg_us",
    "combine_bw_gbps",
    "combine_avg_us",
    "dispatch_send_us",
    "dispatch_recv_us",
    "combine_send_us",
    "combine_recv_us",
    "log_file",
]

NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def rank_prefix(rank: int) -> str:
    # Also accept locally modified tests that print "| num_tokens N".
    return rf"\[rank\s+{rank}(?:\s*\|\s*num_tokens\s+\d+)?\]"


def first_match(pattern: str, text: str) -> Optional[re.Match[str]]:
    return re.search(pattern, text, flags=re.MULTILINE)


def parse_metrics(text: str, rank: int) -> dict[str, str]:
    prefix = rank_prefix(rank)
    metrics = {
        "dispatch_combine_bw_gbps": "",
        "dispatch_combine_avg_us": "",
        "dispatch_bw_gbps": "",
        "dispatch_avg_us": "",
        "combine_bw_gbps": "",
        "combine_avg_us": "",
        "dispatch_send_us": "",
        "dispatch_recv_us": "",
        "combine_send_us": "",
        "combine_recv_us": "",
    }

    combined = first_match(
        prefix
        + rf"\s*Dispatch\s*\+\s*combine bandwidth:\s*{NUMBER}\s*GB/s,\s*"
        + rf"avg_t={NUMBER}\s*us",
        text,
    )
    if combined:
        metrics["dispatch_combine_bw_gbps"] = combined.group(1)
        metrics["dispatch_combine_avg_us"] = combined.group(2)

    separate = first_match(
        prefix
        + rf"\s*Dispatch bandwidth:\s*{NUMBER}\s*GB/s,\s*avg_t={NUMBER}\s*us\s*\|\s*"
        + rf"Combine bandwidth:\s*{NUMBER}\s*GB/s,\s*avg_t={NUMBER}\s*us",
        text,
    )
    if separate:
        metrics["dispatch_bw_gbps"] = separate.group(1)
        metrics["dispatch_avg_us"] = separate.group(2)
        metrics["combine_bw_gbps"] = separate.group(3)
        metrics["combine_avg_us"] = separate.group(4)

    # DeepEP commit 85793dd and later print:
    #   total = send + recv us | ... total = send + recv us
    detailed = first_match(
        prefix
        + rf"\s*Dispatch send/recv time:\s*{NUMBER}\s*=\s*{NUMBER}\s*\+\s*{NUMBER}\s*us\s*\|\s*"
        + rf"Combine send/recv time:\s*{NUMBER}\s*=\s*{NUMBER}\s*\+\s*{NUMBER}\s*us",
        text,
    )
    if detailed:
        metrics["dispatch_send_us"] = detailed.group(2)
        metrics["dispatch_recv_us"] = detailed.group(3)
        metrics["combine_send_us"] = detailed.group(5)
        metrics["combine_recv_us"] = detailed.group(6)
    else:
        # The pinned DeepEP revision prints only ``send + recv`` without a total.
        split_only = first_match(
            prefix
            + rf"\s*Dispatch send/recv time:\s*{NUMBER}\s*\+\s*{NUMBER}\s*us\s*\|\s*"
            + rf"Combine send/recv time:\s*{NUMBER}\s*\+\s*{NUMBER}\s*us",
            text,
        )
        if split_only:
            metrics["dispatch_send_us"] = split_only.group(1)
            metrics["dispatch_recv_us"] = split_only.group(2)
            metrics["combine_send_us"] = split_only.group(3)
            metrics["combine_recv_us"] = split_only.group(4)

    return metrics


def log_exit_code(text: str) -> Optional[int]:
    matches = re.findall(r"^DEEPEP_SWEEP_EXIT_CODE=(\d+)\s*$", text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def relative_log_path(log_path: Path, output_path: Path) -> str:
    try:
        return str(log_path.resolve().relative_to(output_path.parent.resolve()))
    except ValueError:
        return str(log_path.resolve())


def make_row(token: int, rank: int, log_path: Path, output_path: Path) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row["token"] = token
    row["log_file"] = relative_log_path(log_path, output_path)

    if not log_path.is_file():
        row["status"] = "missing_log"
        return row

    text = log_path.read_text(encoding="utf-8", errors="replace")
    row.update(parse_metrics(text, rank))
    exit_code = log_exit_code(text)

    required = (
        "dispatch_combine_avg_us",
        "dispatch_avg_us",
        "combine_avg_us",
    )
    if exit_code not in (None, 0):
        row["status"] = f"benchmark_failed_{exit_code}"
    elif not all(row[name] for name in required):
        row["status"] = "parse_error"
    else:
        row["status"] = "ok"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse DeepEP per-token logs into one node summary CSV."
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--local-processes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.node_rank < 0:
        raise ValueError("--node-rank must be non-negative")
    if args.local_processes <= 0:
        raise ValueError("--local-processes must be positive")
    if len(set(args.tokens)) != len(args.tokens) or any(t <= 0 for t in args.tokens):
        raise ValueError("--tokens must contain unique positive integers")

    first_global_rank = args.node_rank * args.local_processes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        make_row(
            token,
            first_global_rank,
            args.log_dir / f"node{args.node_rank}_tokens_{token}.log",
            args.output,
        )
        for token in args.tokens
    ]

    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(row["status"] == "ok" for row in rows)
    print(
        f"Wrote {args.output}: {ok_count}/{len(rows)} rows parsed successfully "
        f"for global rank {first_global_rank}"
    )
    if ok_count != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

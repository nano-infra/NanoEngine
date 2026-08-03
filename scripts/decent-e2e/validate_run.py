#!/usr/bin/env python3
"""Validate one decentralized E2E stage before marking it resumable."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"validation_error={message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 4:
        fail("usage: validate_run.py STAGE_DIR EXPECTED_REQUESTS EXPECTED_TRANSPORT")

    stage_dir = Path(sys.argv[1])
    try:
        expected_requests = int(sys.argv[2])
    except ValueError:
        fail(f"invalid expected request count: {sys.argv[2]!r}")
    expected_transport = sys.argv[3]

    summaries = list(stage_dir.rglob("*.summary.json"))
    if not summaries:
        fail(f"no .summary.json found below {stage_dir}")
    summary_path = max(summaries, key=lambda path: path.stat().st_mtime_ns)

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {summary_path}: {exc}")

    summary_expected = {
        "total_requests": expected_requests,
        "successful_requests": expected_requests,
        "failed_requests": 0,
        "hierarchical_worker_transport": expected_transport,
    }
    mismatches = [
        f"{key}={summary.get(key)!r} (expected {expected!r})"
        for key, expected in summary_expected.items()
        if summary.get(key) != expected
    ]
    if mismatches:
        fail(f"{summary_path}: " + "; ".join(mismatches))

    console_path = stage_dir / "console.log"
    if not console_path.is_file():
        log_candidates = list(stage_dir.rglob("*.log"))
        if not log_candidates:
            fail(f"no console.log or benchmark .log found below {stage_dir}")
        console_path = max(
            log_candidates, key=lambda path: path.stat().st_mtime_ns
        )

    marker = "[BENCH_RESULT] "
    bench_result = None
    try:
        with console_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if marker not in line:
                    continue
                try:
                    bench_result = json.loads(line.split(marker, 1)[1])
                except json.JSONDecodeError as exc:
                    fail(f"invalid BENCH_RESULT in {console_path}: {exc}")
    except OSError as exc:
        fail(f"cannot read {console_path}: {exc}")
    if bench_result is None:
        fail(f"no BENCH_RESULT found in {console_path}")

    result_expected = {
        "scheduled_requests": expected_requests,
        "dispatched_requests": expected_requests,
        "ingress_enqueued": expected_requests,
        "ingress_rejected": 0,
        "scheduler_accepted": expected_requests,
        "scheduler_rejected": 0,
        "completed_requests": expected_requests,
        "failed_requests": 0,
        "pending_ingress": 0,
        "pending_add": 0,
        "active_requests": 0,
        "hierarchical_worker_transport": expected_transport,
    }
    mismatches = [
        f"{key}={bench_result.get(key)!r} (expected {expected!r})"
        for key, expected in result_expected.items()
        if bench_result.get(key) != expected
    ]
    if mismatches:
        fail(f"{console_path}: " + "; ".join(mismatches))

    print(f"summary={summary_path}")
    print(f"bench_result_log={console_path}")
    print(f"requests={expected_requests}/{expected_requests}")
    print(f"worker_transport={expected_transport}")


if __name__ == "__main__":
    main()

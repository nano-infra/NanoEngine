#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


D_VALUES = (1, 2, 4, 8)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def number(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def integer(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
                row["_source"] = str(path)
                rows.append(row)
    return rows


def bucket_value(row: dict[str, Any], bucket_key: str, exact_key: str) -> int:
    if row.get(bucket_key) not in (None, ""):
        return integer(row[bucket_key])
    return integer(row.get(exact_key))


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("profile_kind") or "uniform",
        row.get("dataset_mode") or "",
        bucket_value(row, "B_bucket", "B"),
        bucket_value(row, "W_attn_bucket", "W_attn"),
        bucket_value(row, "L_p90_bucket", "L_p90"),
        bucket_value(row, "L_max_bucket", "L_max"),
    )


def d_value(row: dict[str, Any]) -> int:
    return integer(row.get("d_attn", row.get("dop")))


def row_metric(row: dict[str, Any], metric: str) -> float:
    if row.get(metric) in (None, ""):
        fallback = "model_p90_ms" if metric == "step_p90_ms" else "step_p90_ms"
        return number(row.get(fallback), math.inf)
    return number(row[metric], math.inf)


def stable_metric(values: list[float], *, min_repeats: int, stability_ratio: float) -> bool:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < min_repeats:
        return False
    if len(finite) <= 1:
        return True
    return max(finite) <= stability_ratio * min(finite)


def choose_entry(
    key: tuple[Any, ...],
    rows: list[dict[str, Any]],
    *,
    metric: str,
    near_optimal_ratio: float,
    abs_gain_ms: float,
    min_repeats: int,
    stability_ratio: float,
) -> dict[str, Any] | None:
    by_d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = d_value(row)
        if d in D_VALUES:
            by_d[d].append(row)

    candidates: dict[int, dict[str, Any]] = {}
    for d, d_rows in by_d.items():
        viable_rows = [
            row
            for row in d_rows
            if truthy(row.get("viable", True)) and truthy(row.get("node_local", True))
        ]
        if not viable_rows:
            continue
        values = [row_metric(row, metric) for row in viable_rows]
        candidates[d] = {
            "rows": viable_rows,
            "metric_values": values,
            "metric": statistics.median(values),
            "stable": stable_metric(
                values,
                min_repeats=min_repeats,
                stability_ratio=stability_ratio,
            ),
        }
    if not candidates:
        return None

    stable = all(info["stable"] for info in candidates.values())
    d_mem = min(candidates)
    best_d = min(candidates, key=lambda d: candidates[d]["metric"])
    best_metric = candidates[best_d]["metric"]
    near_limit = near_optimal_ratio * best_metric
    d_near = min(d for d, info in candidates.items() if info["metric"] <= near_limit)

    d_target = max(d_mem, d_near)
    performance_threshold = "none"
    threshold_reason = "data_not_stable" if not stable else "d_mem_or_near_optimal"
    if stable and d_mem in candidates and best_d > d_mem:
        current = candidates[d_mem]["metric"]
        gain = current - best_metric
        required_gain = max((near_optimal_ratio - 1.0) * current, abs_gain_ms)
        if gain >= required_gain:
            performance_threshold = f"d{d_mem}_to_d{best_d}"
            threshold_reason = "stable_p90_gain"
            d_target = max(d_mem, best_d)
        elif d_mem == 1:
            threshold_reason = "d=1_remains_near_optimal"

    profile_kind, dataset_mode, b_bucket, w_bucket, l_p90_bucket, l_max_bucket = key
    return {
        "profile_kind": profile_kind,
        "dataset_mode": dataset_mode,
        "B_bucket": b_bucket,
        "W_attn_bucket": w_bucket,
        "L_p90_bucket": l_p90_bucket,
        "L_max_bucket": l_max_bucket,
        "d_target": d_target,
        "d_mem": d_mem,
        "d_near": d_near,
        "best_d": best_d,
        "metric": metric,
        "metric_by_d": {str(d): candidates[d]["metric"] for d in sorted(candidates)},
        "repeat_count_by_d": {
            str(d): len(candidates[d]["metric_values"]) for d in sorted(candidates)
        },
        "stable_repeats": stable,
        "performance_threshold": performance_threshold,
        "threshold_reason": threshold_reason,
        "node_local": True,
    }


def build_sib(
    rows: list[dict[str, Any]],
    *,
    include_dataset_replay: bool,
    metric: str,
    near_optimal_ratio: float,
    abs_gain_ms: float,
    min_repeats: int,
    stability_ratio: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        profile_kind = row.get("profile_kind") or "uniform"
        if profile_kind == "dataset_replay" and not include_dataset_replay:
            continue
        grouped[row_key(row)].append(row)

    entries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        entry = choose_entry(
            key,
            grouped[key],
            metric=metric,
            near_optimal_ratio=near_optimal_ratio,
            abs_gain_ms=abs_gain_ms,
            min_repeats=min_repeats,
            stability_ratio=stability_ratio,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def write_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# LoongServe SP8 Decode Thresholds",
        "",
        "| profile | mode | B | W_attn | L_p90 | L_max | d_target | best_d | stable | reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| {profile_kind} | {dataset_mode} | {B_bucket} | {W_attn_bucket} | "
            "{L_p90_bucket} | {L_max_bucket} | {d_target} | {best_d} | "
            "{stable_repeats} | {threshold_reason} |".format(**entry)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a LoongServe SP<=8 decode SIB/threshold table from profile JSONL."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--include-dataset-replay", action="store_true")
    parser.add_argument("--metric", choices=("step_p90_ms", "model_p90_ms"), default="step_p90_ms")
    parser.add_argument("--near-optimal-ratio", type=float, default=1.05)
    parser.add_argument("--abs-gain-ms", type=float, default=0.1)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--stability-ratio", type=float, default=1.05)
    args = parser.parse_args()

    if args.near_optimal_ratio < 1.0:
        raise ValueError("--near-optimal-ratio must be >= 1.0")
    if args.abs_gain_ms < 0:
        raise ValueError("--abs-gain-ms must be >= 0")
    if args.min_repeats < 1:
        raise ValueError("--min-repeats must be >= 1")
    if args.stability_ratio < 1.0:
        raise ValueError("--stability-ratio must be >= 1.0")

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_json = args.out_json or Path(
        f"docs-dev/profile-results/loongserve_sp8_decode_sib_{today}.json"
    )
    out_md = args.out_md or Path(
        f"docs-dev/profile-results/loongserve_sp8_decode_thresholds_{today}.md"
    )

    rows = load_rows(args.inputs)
    entries = build_sib(
        rows,
        include_dataset_replay=args.include_dataset_replay,
        metric=args.metric,
        near_optimal_ratio=args.near_optimal_ratio,
        abs_gain_ms=args.abs_gain_ms,
        min_repeats=args.min_repeats,
        stability_ratio=args.stability_ratio,
    )

    payload = {
        "kind": "loongserve_sp8_decode_sib",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path) for path in args.inputs],
        "metric": args.metric,
        "near_optimal_ratio": args.near_optimal_ratio,
        "abs_gain_ms": args.abs_gain_ms,
        "min_repeats": args.min_repeats,
        "stability_ratio": args.stability_ratio,
        "entries": entries,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, entries)
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    print(f"entries: {len(entries)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract Fig. 19 routing cases and measure their Q/Res/LSE A2A latency."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
BENCH_SCRIPT = Path(__file__).resolve().with_name("benchmark_q_res_lse_latency.py")
PAYLOAD_NAMES = ("Q", "Res", "Lse")
PLOT_FONT_FAMILY = get_plot_font_family()
CP_COLORS = {
    1: "#5B616B",
    2: "#0B3C5D",
    3: "#8C4F00",
    4: "#006D77",
    5: "#9A031E",
    6: "#5F0F40",
    7: "#1D4E89",
    8: "#7A5C00",
}


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


configure_plot_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-iter decode A2A masks from NanoDeploy logs, benchmark unique Q/Res/Lse patterns, and materialize per-iter latency traces."
    )
    parser.add_argument("--decode-log", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--payloads", type=str, default="Q,Res,Lse")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--v-head-dim", type=int, default=512)
    parser.add_argument("--mode", type=str, default="graph", choices=["eager", "graph"])
    parser.add_argument("--preamble", type=str, default="all_reduce", choices=["none", "all_reduce", "all_gather"])
    parser.add_argument("--dlslime-impl", type=str, default="basic", choices=["basic", "tma"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--graph-inner-iters", type=int, default=20)
    parser.add_argument("--metric", type=str, default="idle_all2all_kernel_slowest_us")
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="Keep all per-rank PyTorch profiler traces (large and normally unnecessary).",
    )
    parser.add_argument(
        "--diagnostic-plots",
        action="store_true",
        help="Generate the collector's per-DP diagnostic plots in addition to the final Fig. 19 inputs.",
    )
    parser.add_argument("--max-groups", type=int, default=None, help="Only process the first N complete (iter, dp) groups")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--reuse-benchmark", action="store_true")
    parser.add_argument(
        "--reuse-extraction",
        action="store_true",
        help=(
            "Reuse unique cases, iter_case_mapping.csv, and extract_summary.json "
            "already present in --output-dir instead of parsing --decode-log again."
        ),
    )
    parser.add_argument(
        "--approximate",
        action="store_true",
        help=(
            "Use staircase approximation: bucket Q/Res by discretized "
            "(max_send_rows, max_recv_rows), benchmark representative cases, "
            "and use a calibrated constant for LSE."
        ),
    )
    parser.add_argument(
        "--bucket-step",
        type=int,
        default=2,
        help="Q/Res max-send/max-recv bucket width in rows (default: 2).",
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=2,
        help="Number of most-frequent Q/Res cases measured per occupied bucket (default: 2).",
    )
    parser.add_argument(
        "--lse-samples",
        type=int,
        default=16,
        help="Number of traffic-stratified LSE cases used for constant-latency calibration (default: 16).",
    )
    return parser.parse_args()


def parse_requested_payloads(raw: str) -> list[str]:
    values = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        if value not in PAYLOAD_NAMES:
            raise ValueError(f"Unsupported payload: {value}")
        values.append(value)
    if values != list(PAYLOAD_NAMES):
        raise ValueError("This pipeline currently expects payloads exactly equal to Q,Res,Lse")
    return values


def requested_benchmark_config(args: argparse.Namespace) -> dict:
    """Return the measurement configuration recorded in pipeline_summary.json."""
    return {
        "python_bin": args.python_bin,
        "nproc_per_node": args.nproc_per_node,
        "dtype": args.dtype,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "v_head_dim": args.v_head_dim,
        "mode": args.mode,
        "preamble": args.preamble,
        "dlslime_impl": args.dlslime_impl,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_iters": args.profile_iters,
        "graph_inner_iters": args.graph_inner_iters,
        "metric": args.metric,
        "keep_traces": args.keep_traces,
        "diagnostic_plots": args.diagnostic_plots,
    }


def requested_approximation_config(args: argparse.Namespace) -> dict:
    return {
        "enabled": bool(args.approximate),
        "bucket_features": ["max_send_rows", "max_recv_rows"],
        "bucket_step": args.bucket_step,
        "zero_is_separate_bucket": True,
        "samples_per_bucket": args.samples_per_bucket,
        "representative_selection": "highest_iteration_frequency",
        "bucket_latency_aggregation": "mean",
        "lse_samples": args.lse_samples,
        "lse_selection": "traffic_stratified",
        "lse_latency_aggregation": "median",
    }


def _infer_q_num_rows(q_mask: list[list[int]]) -> int:
    max_col = -1
    for row in q_mask:
        for col_idx, value in enumerate(row):
            if value:
                max_col = max(max_col, col_idx)
    return max_col + 1


def _parse_decode_a2a_payload(line: str) -> dict | None:
    if "'mode': 'decode_a2a_masks'" not in line:
        return None
    clean = ANSI_ESCAPE_RE.sub("", line)
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end < start:
        return None
    payload = clean[start : end + 1]
    # These decode payloads are emitted as a Python dict repr with simple scalar
    # string fields and nested integer lists. Translating to JSON first is
    # materially faster than `ast.literal_eval` on multi-GB logs.
    json_payload = payload.replace("'", '"').replace("False", "false").replace("True", "true")
    try:
        return json.loads(json_payload)
    except json.JSONDecodeError:
        return ast.literal_eval(payload)


def _decode_mask_rows(encoded: object, bits_per_row: int, field_name: str) -> list[list[int]]:
    if not isinstance(encoded, list):
        raise ValueError(f"{field_name} must be a list of rows")
    if encoded and isinstance(encoded[0], str):
        rows = []
        for entry in encoded:
            packed = np.frombuffer(base64.b64decode(entry), dtype=np.uint8)
            rows.append(np.unpackbits(packed, count=bits_per_row).astype(int).tolist())
        return rows
    return [[int(v) for v in row] for row in encoded]


def _build_case_from_group(key: tuple[int, int, int], entries: list[dict]) -> dict:
    global_run_count, loop_idx, dp_rank = key
    cp_size = int(entries[0]["cp_size"])
    max_bs = int(entries[0]["max_bs"])
    raw_per_rank = []
    for sp_rank, entry in enumerate(entries):
        if int(entry["sp_rank"]) != sp_rank:
            raise ValueError(
                f"group {key} is missing sp_rank ordering; expected {sp_rank}, got {entry['sp_rank']}"
            )
        q_mask = _decode_mask_rows(entry["q_mask"], max_bs, "q_mask")
        res_lse_mask = _decode_mask_rows(entry["res_lse_mask"], max_bs, "res_lse_mask")
        raw_per_rank.append(
            {
                "q_offsets": [int(v) for v in entry["q_offsets"]],
                "q_mask": q_mask,
                "res_lse_mask": res_lse_mask,
                "q_num_rows": int(entry["q_offsets"][sp_rank + 1]) - int(entry["q_offsets"][sp_rank]),
            }
        )
    return {
        "global_run_count": global_run_count,
        "loop_idx": loop_idx,
        "dp_rank": dp_rank,
        "cp_size": cp_size,
        "max_bs": max_bs,
        "raw_per_rank": raw_per_rank,
    }


def _build_payload_signature(case: dict, payload_name: str) -> tuple:
    if payload_name == "Q":
        return (
            int(case["cp_size"]),
            int(case["max_bs"]),
            tuple(
                (
                    tuple(int(v) for v in per_rank["q_offsets"]),
                    tuple(bytes(int(v) for v in row) for row in per_rank["q_mask"]),
                    int(per_rank["q_num_rows"]),
                )
                for per_rank in case["raw_per_rank"]
            ),
        )
    return (
        int(case["cp_size"]),
        int(case["max_bs"]),
        tuple(tuple(bytes(int(v) for v in row) for row in per_rank["res_lse_mask"]) for per_rank in case["raw_per_rank"]),
    )


def extract_unique_cases(
    log_path: Path, max_groups: int | None = None
) -> tuple[dict[str, list[dict]], list[dict], dict]:
    grouped: dict[tuple[int, int, int], dict[int, dict]] = defaultdict(dict)
    completed_keys: list[tuple[int, int, int]] = []
    completed_key_set: set[tuple[int, int, int]] = set()
    line_count = 0
    parsed_count = 0
    dummy_count = 0
    bad_count = 0

    with log_path.open(errors="ignore") as f:
        for line in f:
            line_count += 1
            if "'mode': 'decode_a2a_masks'" not in line:
                continue
            try:
                payload = _parse_decode_a2a_payload(line)
            except Exception:
                bad_count += 1
                continue
            if payload is None:
                bad_count += 1
                continue
            if payload.get("is_dummy"):
                dummy_count += 1
                continue
            parsed_count += 1
            key = (int(payload["global_run_count"]), int(payload["loop_idx"]), int(payload["dp_rank"]))
            grouped[key][int(payload["sp_rank"])] = payload
            if len(grouped[key]) == 8 and key not in completed_key_set:
                completed_keys.append(key)
                completed_key_set.add(key)
                if max_groups is not None and len(completed_keys) >= max_groups:
                    break

    mapping_rows = []
    unique_cases_by_payload = {payload_name: [] for payload_name in PAYLOAD_NAMES}
    case_id_by_signature = {payload_name: {} for payload_name in PAYLOAD_NAMES}
    complete_groups = []
    incomplete_groups = 0

    keys_to_visit = sorted(completed_keys) if max_groups is not None else sorted(grouped)
    for key in keys_to_visit:
        per_rank = grouped[key]
        if len(per_rank) != 8 or any(sp_rank not in per_rank for sp_rank in range(8)):
            incomplete_groups += 1
            continue
        complete_groups.append((key, [per_rank[sp_rank] for sp_rank in range(8)]))

    iter_ordinal_per_dp: dict[int, int] = defaultdict(int)
    for key, entries in complete_groups:
        case = _build_case_from_group(key, entries)
        dp_rank = case["dp_rank"]
        mapping_row = {
            "iter_ordinal": iter_ordinal_per_dp[dp_rank],
            "global_run_count": case["global_run_count"],
            "loop_idx": case["loop_idx"],
            "dp_rank": dp_rank,
        }
        for payload_name in PAYLOAD_NAMES:
            signature = _build_payload_signature(case, payload_name)
            cached = case_id_by_signature[payload_name].get(signature)
            if cached is None:
                digest = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()
                case_id = f"{payload_name.lower()}_pattern_{len(case_id_by_signature[payload_name]):06d}"
                case_id_by_signature[payload_name][signature] = (case_id, digest)
                unique_cases_by_payload[payload_name].append(
                    {
                        "case_id": case_id,
                        "description": f"decode iter raw {payload_name} pattern {digest[:12]}",
                        "pattern": f"raw_decode_a2a_{payload_name.lower()}",
                        "cp_size": case["cp_size"],
                        "max_bs": case["max_bs"],
                        "raw_per_rank": case["raw_per_rank"],
                    }
                )
            else:
                case_id, digest = cached
            mapping_row[f"{payload_name.lower()}_case_id"] = case_id
            mapping_row[f"{payload_name.lower()}_pattern_hash"] = digest
        mapping_rows.append(
            mapping_row
        )
        iter_ordinal_per_dp[dp_rank] += 1

    summary = {
        "log_path": str(log_path),
        "line_count": line_count,
        "parsed_decode_a2a_lines": parsed_count,
        "dummy_decode_a2a_lines": dummy_count,
        "bad_decode_a2a_lines": bad_count,
        "complete_groups": len(complete_groups),
        "incomplete_groups": incomplete_groups,
        "unique_patterns": {payload_name: len(unique_cases_by_payload[payload_name]) for payload_name in PAYLOAD_NAMES},
        "iters_per_dp": {str(dp): count for dp, count in sorted(iter_ordinal_per_dp.items())},
    }
    return unique_cases_by_payload, mapping_rows, summary


def write_unique_cases(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, separators=(",", ":")))


def load_unique_cases(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"unique cases file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload)
    if not isinstance(cases, list):
        raise ValueError(f"unique cases file must contain a list: {path}")
    return cases


def write_mapping_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter_ordinal",
                "global_run_count",
                "loop_idx",
                "dp_rank",
                "q_case_id",
                "q_pattern_hash",
                "res_case_id",
                "res_pattern_hash",
                "lse_case_id",
                "lse_pattern_hash",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_mapping_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"iteration mapping not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"iteration mapping contains no rows: {path}")
    return rows


def compute_raw_case_traffic(case: dict, payload_name: str) -> dict:
    cp_size = int(case["cp_size"])
    mask_key = "q_mask" if payload_name == "Q" else "res_lse_mask"
    send_rows_per_rank = [0] * cp_size
    recv_rows_per_rank = [0] * cp_size
    total_remote_edges = 0

    for src_rank, entry in enumerate(case["raw_per_rank"]):
        mask = entry[mask_key]
        for dst_rank in range(cp_size):
            if dst_rank == src_rank:
                continue
            edge_count = sum(int(value) for value in mask[dst_rank])
            send_rows_per_rank[src_rank] += edge_count
            recv_rows_per_rank[dst_rank] += edge_count
            total_remote_edges += edge_count

    return {
        "send_rows_per_rank": send_rows_per_rank,
        "recv_rows_per_rank": recv_rows_per_rank,
        "total_remote_edges": total_remote_edges,
        "max_send_rows": max(send_rows_per_rank, default=0),
        "max_recv_rows": max(recv_rows_per_rank, default=0),
    }


def staircase_bucket(value: int, step: int) -> int:
    if value == 0:
        return 0
    return (value + step - 1) // step


def staircase_bucket_bounds(bucket: int, step: int) -> tuple[int, int]:
    if bucket == 0:
        return 0, 0
    return (bucket - 1) * step + 1, bucket * step


def evenly_spaced_cases(cases: list[dict], sample_count: int) -> list[dict]:
    if sample_count >= len(cases):
        return list(cases)
    if sample_count == 1:
        return [cases[(len(cases) - 1) // 2]]
    indices = [index * (len(cases) - 1) // (sample_count - 1) for index in range(sample_count)]
    return [cases[index] for index in indices]


def build_approximation_plan(
    unique_cases_by_payload: dict[str, list[dict]],
    mapping_rows: list[dict],
    *,
    bucket_step: int,
    samples_per_bucket: int,
    lse_samples: int,
) -> dict:
    if bucket_step <= 0:
        raise ValueError("--bucket-step must be positive")
    if samples_per_bucket <= 0:
        raise ValueError("--samples-per-bucket must be positive")
    if lse_samples <= 0:
        raise ValueError("--lse-samples must be positive")

    frequencies = {
        payload_name: Counter(row[f"{payload_name.lower()}_case_id"] for row in mapping_rows)
        for payload_name in PAYLOAD_NAMES
    }
    traffic_by_payload: dict[str, dict[str, dict]] = {}
    benchmark_cases_by_payload: dict[str, list[dict]] = {}
    buckets_by_payload: dict[str, list[dict]] = {}

    for payload_name in ("Q", "Res"):
        traffic_by_case = {
            case["case_id"]: compute_raw_case_traffic(case, payload_name)
            for case in unique_cases_by_payload[payload_name]
        }
        traffic_by_payload[payload_name] = traffic_by_case
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for case in unique_cases_by_payload[payload_name]:
            traffic = traffic_by_case[case["case_id"]]
            key = (
                staircase_bucket(traffic["max_send_rows"], bucket_step),
                staircase_bucket(traffic["max_recv_rows"], bucket_step),
            )
            grouped[key].append(case)

        selected_cases = []
        bucket_entries = []
        for send_bucket, recv_bucket in sorted(grouped):
            members = sorted(
                grouped[(send_bucket, recv_bucket)],
                key=lambda case: (-frequencies[payload_name][case["case_id"]], case["case_id"]),
            )
            representatives = members[:samples_per_bucket]
            selected_cases.extend(representatives)
            bucket_entries.append(
                {
                    "send_bucket": send_bucket,
                    "recv_bucket": recv_bucket,
                    "send_bounds": staircase_bucket_bounds(send_bucket, bucket_step),
                    "recv_bounds": staircase_bucket_bounds(recv_bucket, bucket_step),
                    "case_ids": [case["case_id"] for case in members],
                    "representative_case_ids": [case["case_id"] for case in representatives],
                    "iteration_count": sum(frequencies[payload_name][case["case_id"]] for case in members),
                }
            )
        benchmark_cases_by_payload[payload_name] = selected_cases
        buckets_by_payload[payload_name] = bucket_entries

    lse_traffic_by_case = {
        case["case_id"]: compute_raw_case_traffic(case, "Lse")
        for case in unique_cases_by_payload["Lse"]
    }
    traffic_by_payload["Lse"] = lse_traffic_by_case
    ordered_lse_cases = sorted(
        unique_cases_by_payload["Lse"],
        key=lambda case: (
            lse_traffic_by_case[case["case_id"]]["max_send_rows"],
            lse_traffic_by_case[case["case_id"]]["max_recv_rows"],
            lse_traffic_by_case[case["case_id"]]["total_remote_edges"],
            case["case_id"],
        ),
    )
    selected_lse_cases = evenly_spaced_cases(ordered_lse_cases, min(lse_samples, len(ordered_lse_cases)))
    benchmark_cases_by_payload["Lse"] = selected_lse_cases

    return {
        "bucket_step": bucket_step,
        "samples_per_bucket": samples_per_bucket,
        "requested_lse_samples": lse_samples,
        "frequencies": frequencies,
        "traffic_by_payload": traffic_by_payload,
        "buckets_by_payload": buckets_by_payload,
        "benchmark_cases_by_payload": benchmark_cases_by_payload,
        "lse_case_ids": [case["case_id"] for case in unique_cases_by_payload["Lse"]],
        "lse_representative_case_ids": [case["case_id"] for case in selected_lse_cases],
    }


def expand_approximate_latency_maps(plan: dict, measured_latency_maps: dict[str, dict[str, float]]) -> tuple[dict, list[dict], dict]:
    expanded = {payload_name: {} for payload_name in PAYLOAD_NAMES}
    mapping_rows = []
    bucket_summaries = {"Q": [], "Res": []}

    for payload_name in ("Q", "Res"):
        for bucket in plan["buckets_by_payload"][payload_name]:
            representative_ids = bucket["representative_case_ids"]
            missing = [case_id for case_id in representative_ids if case_id not in measured_latency_maps[payload_name]]
            if missing:
                raise ValueError(
                    f"{payload_name} benchmark is missing representative cases: {', '.join(missing)}"
                )
            representative_values = [measured_latency_maps[payload_name][case_id] for case_id in representative_ids]
            assigned_latency_us = sum(representative_values) / len(representative_values)
            bucket_summaries[payload_name].append(
                {
                    "send_bucket": bucket["send_bucket"],
                    "recv_bucket": bucket["recv_bucket"],
                    "send_bounds": list(bucket["send_bounds"]),
                    "recv_bounds": list(bucket["recv_bounds"]),
                    "case_count": len(bucket["case_ids"]),
                    "iteration_count": bucket["iteration_count"],
                    "representative_case_ids": representative_ids,
                    "representative_latency_us": representative_values,
                    "assigned_latency_us": assigned_latency_us,
                }
            )
            for case_id in bucket["case_ids"]:
                expanded[payload_name][case_id] = assigned_latency_us
                traffic = plan["traffic_by_payload"][payload_name][case_id]
                mapping_rows.append(
                    {
                        "payload": payload_name,
                        "case_id": case_id,
                        "iteration_count": plan["frequencies"][payload_name][case_id],
                        "total_remote_edges": traffic["total_remote_edges"],
                        "max_send_rows": traffic["max_send_rows"],
                        "max_recv_rows": traffic["max_recv_rows"],
                        "send_bucket": bucket["send_bucket"],
                        "recv_bucket": bucket["recv_bucket"],
                        "send_range": f"{bucket['send_bounds'][0]}-{bucket['send_bounds'][1]}",
                        "recv_range": f"{bucket['recv_bounds'][0]}-{bucket['recv_bounds'][1]}",
                        "representative_case_ids": ";".join(representative_ids),
                        "assigned_latency_us": assigned_latency_us,
                    }
                )

    lse_representative_ids = plan["lse_representative_case_ids"]
    missing_lse = [case_id for case_id in lse_representative_ids if case_id not in measured_latency_maps["Lse"]]
    if missing_lse:
        raise ValueError(f"Lse benchmark is missing representative cases: {', '.join(missing_lse)}")
    lse_values = [measured_latency_maps["Lse"][case_id] for case_id in lse_representative_ids]
    lse_latency_us = float(median(lse_values))
    for case_id in plan["lse_case_ids"]:
        expanded["Lse"][case_id] = lse_latency_us
        traffic = plan["traffic_by_payload"]["Lse"][case_id]
        mapping_rows.append(
            {
                "payload": "Lse",
                "case_id": case_id,
                "iteration_count": plan["frequencies"]["Lse"][case_id],
                "total_remote_edges": traffic["total_remote_edges"],
                "max_send_rows": traffic["max_send_rows"],
                "max_recv_rows": traffic["max_recv_rows"],
                "send_bucket": "",
                "recv_bucket": "",
                "send_range": "constant",
                "recv_range": "constant",
                "representative_case_ids": ";".join(lse_representative_ids),
                "assigned_latency_us": lse_latency_us,
            }
        )

    summary = {
        "mode": "staircase",
        "bucket_features": ["max_send_rows", "max_recv_rows"],
        "bucket_step": plan["bucket_step"],
        "zero_is_separate_bucket": True,
        "samples_per_bucket": plan["samples_per_bucket"],
        "representative_selection": "highest_iteration_frequency",
        "bucket_latency_aggregation": "mean",
        "payloads": {
            payload_name: {
                "unique_case_count": len(plan["traffic_by_payload"][payload_name]),
                "bucket_count": len(plan["buckets_by_payload"][payload_name]),
                "benchmark_case_count": len(plan["benchmark_cases_by_payload"][payload_name]),
                "buckets": bucket_summaries[payload_name],
            }
            for payload_name in ("Q", "Res")
        },
        "lse": {
            "unique_case_count": len(plan["lse_case_ids"]),
            "benchmark_case_count": len(lse_representative_ids),
            "representative_case_ids": lse_representative_ids,
            "representative_latency_us": lse_values,
            "aggregation": "median",
            "assigned_latency_us": lse_latency_us,
        },
        "total_benchmark_case_count": sum(
            len(plan["benchmark_cases_by_payload"][payload_name]) for payload_name in PAYLOAD_NAMES
        ),
    }
    return expanded, mapping_rows, summary


def write_approximation_mapping_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "payload",
                "case_id",
                "iteration_count",
                "total_remote_edges",
                "max_send_rows",
                "max_recv_rows",
                "send_bucket",
                "recv_bucket",
                "send_range",
                "recv_range",
                "representative_case_ids",
                "assigned_latency_us",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_approximation_plan(plan: dict) -> dict:
    return {
        "mode": "staircase",
        "bucket_features": ["max_send_rows", "max_recv_rows"],
        "bucket_step": plan["bucket_step"],
        "zero_is_separate_bucket": True,
        "samples_per_bucket": plan["samples_per_bucket"],
        "representative_selection": "highest_iteration_frequency",
        "bucket_latency_aggregation": "mean",
        "payloads": {
            payload_name: {
                "unique_case_count": len(plan["traffic_by_payload"][payload_name]),
                "bucket_count": len(plan["buckets_by_payload"][payload_name]),
                "benchmark_case_count": len(plan["benchmark_cases_by_payload"][payload_name]),
            }
            for payload_name in ("Q", "Res")
        },
        "lse": {
            "unique_case_count": len(plan["lse_case_ids"]),
            "benchmark_case_count": len(plan["lse_representative_case_ids"]),
            "representative_case_ids": plan["lse_representative_case_ids"],
            "aggregation": "median",
        },
        "total_benchmark_case_count": sum(
            len(plan["benchmark_cases_by_payload"][payload_name]) for payload_name in PAYLOAD_NAMES
        ),
    }


def write_approximation_plan(plan: dict, path: Path) -> None:
    payload = {
        "format_version": 1,
        "bucket_step": plan["bucket_step"],
        "samples_per_bucket": plan["samples_per_bucket"],
        "requested_lse_samples": plan["requested_lse_samples"],
        "frequencies": {
            payload_name: dict(plan["frequencies"][payload_name]) for payload_name in PAYLOAD_NAMES
        },
        "traffic_by_payload": {
            payload_name: {
                case_id: {
                    "total_remote_edges": traffic["total_remote_edges"],
                    "max_send_rows": traffic["max_send_rows"],
                    "max_recv_rows": traffic["max_recv_rows"],
                }
                for case_id, traffic in plan["traffic_by_payload"][payload_name].items()
            }
            for payload_name in PAYLOAD_NAMES
        },
        "buckets_by_payload": plan["buckets_by_payload"],
        "lse_case_ids": plan["lse_case_ids"],
        "lse_representative_case_ids": plan["lse_representative_case_ids"],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_approximation_plan(path: Path, benchmark_case_paths: dict[str, Path]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"approximation plan not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported approximation plan format: {path}")
    return {
        "bucket_step": int(payload["bucket_step"]),
        "samples_per_bucket": int(payload["samples_per_bucket"]),
        "requested_lse_samples": int(payload["requested_lse_samples"]),
        "frequencies": {
            payload_name: Counter(payload["frequencies"][payload_name]) for payload_name in PAYLOAD_NAMES
        },
        "traffic_by_payload": payload["traffic_by_payload"],
        "buckets_by_payload": payload["buckets_by_payload"],
        "benchmark_cases_by_payload": {
            payload_name: load_unique_cases(benchmark_case_paths[payload_name]) for payload_name in PAYLOAD_NAMES
        },
        "lse_case_ids": payload["lse_case_ids"],
        "lse_representative_case_ids": payload["lse_representative_case_ids"],
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    payload_name: str,
    cases_path: Path,
    summary_path: Path,
    csv_path: Path,
    trace_dir: Path,
) -> None:
    cmd = [
        args.python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(BENCH_SCRIPT),
        "--cases",
        str(cases_path),
        "--payloads",
        payload_name,
        "--dtype",
        args.dtype,
        "--num-heads",
        str(args.num_heads),
        "--head-dim",
        str(args.head_dim),
        "--v-head-dim",
        str(args.v_head_dim),
        "--mode",
        args.mode,
        "--preamble",
        args.preamble,
        "--dlslime-impl",
        args.dlslime_impl,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--profile-iters",
        str(args.profile_iters),
        "--graph-inner-iters",
        str(args.graph_inner_iters),
        "--summary-path",
        str(summary_path),
        "--csv-path",
        str(csv_path),
        "--trace-dir",
        str(trace_dir),
    ]
    if args.keep_traces:
        cmd.append("--keep-traces")
    subprocess.run(cmd, check=True, cwd=str(BENCH_SCRIPT.parent))


def load_case_latencies(csv_path: Path, metric_name: str) -> dict[str, float]:
    latencies: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            latencies[row["case_id"]] = float(row[metric_name])
    return latencies


def materialize_per_iter_rows(
    mapping_rows: list[dict], *, q_latencies: dict[str, float], res_latencies: dict[str, float], lse_latencies: dict[str, float]
) -> list[dict]:
    rows = []
    for row in mapping_rows:
        q_us = float(q_latencies[row["q_case_id"]])
        res_us = float(res_latencies[row["res_case_id"]])
        lse_us = float(lse_latencies[row["lse_case_id"]])
        rows.append(
            {
                **row,
                "q_us": q_us,
                "res_us": res_us,
                "lse_us": lse_us,
                "total_us": q_us + res_us + lse_us,
            }
        )
    return rows


def derive_cp_hist_from_q_case(q_case: dict) -> dict[int, int]:
    cp_hist: dict[int, int] = defaultdict(int)
    cp_size = int(q_case["cp_size"])
    for master_sp_rank, per_rank in enumerate(q_case["raw_per_rank"]):
        num_rows = int(per_rank["q_num_rows"])
        q_mask = per_rank["q_mask"]
        if q_mask and num_rows > len(q_mask[0]):
            raise ValueError(
                f"Q case {q_case['case_id']} has q_num_rows={num_rows} exceeding q_mask width={len(q_mask[0])}"
            )
        for col_idx in range(num_rows):
            fanout = sum(int(q_mask[dst_sp_rank][col_idx]) for dst_sp_rank in range(cp_size))
            request_cp_size = 1 + fanout
            cp_hist[request_cp_size] += 1
    return dict(sorted(cp_hist.items()))


def materialize_per_iter_cp_hist_rows(mapping_rows: list[dict], q_cases: list[dict]) -> tuple[list[dict], list[int]]:
    q_hist_by_case_id = {q_case["case_id"]: derive_cp_hist_from_q_case(q_case) for q_case in q_cases}
    observed_cp_sizes = sorted(
        {
            cp_size
            for cp_hist in q_hist_by_case_id.values()
            for cp_size in cp_hist
        }
    )
    rows = []
    for mapping_row in mapping_rows:
        cp_hist = q_hist_by_case_id[mapping_row["q_case_id"]]
        row = {
            "iter_ordinal": mapping_row["iter_ordinal"],
            "global_run_count": mapping_row["global_run_count"],
            "loop_idx": mapping_row["loop_idx"],
            "dp_rank": mapping_row["dp_rank"],
        }
        for cp_size in observed_cp_sizes:
            row[f"cp_{cp_size}"] = cp_hist.get(cp_size, 0)
        rows.append(row)
    return rows, observed_cp_sizes


def write_per_iter_cp_hist_csv(rows: list[dict], cp_sizes: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter_ordinal",
                "global_run_count",
                "loop_idx",
                "dp_rank",
                *[f"cp_{cp_size}" for cp_size in cp_sizes],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_per_iter_cp_hist_csv(path: Path) -> tuple[list[dict], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"per-iteration CP histogram not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"per-iteration CP histogram has no header: {path}")
        cp_sizes = sorted(
            int(name.removeprefix("cp_"))
            for name in reader.fieldnames
            if name.startswith("cp_") and name.removeprefix("cp_").isdigit()
        )
        rows = list(reader)
    if not rows or not cp_sizes:
        raise ValueError(f"per-iteration CP histogram is empty: {path}")
    return rows, cp_sizes


def write_per_iter_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter_ordinal",
                "global_run_count",
                "loop_idx",
                "dp_rank",
                "q_case_id",
                "q_pattern_hash",
                "res_case_id",
                "res_pattern_hash",
                "lse_case_id",
                "lse_pattern_hash",
                "q_us",
                "res_us",
                "lse_us",
                "total_us",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_per_dp(rows: list[dict], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows_by_dp: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_dp[int(row["dp_rank"])].append(row)

    for dp_rank, dp_rows in rows_by_dp.items():
        dp_rows.sort(key=lambda row: int(row["iter_ordinal"]))
        x = [int(row["iter_ordinal"]) for row in dp_rows]
        q_us = [float(row["q_us"]) for row in dp_rows]
        res_us = [float(row["res_us"]) for row in dp_rows]
        lse_us = [float(row["lse_us"]) for row in dp_rows]
        total_us = [float(row["total_us"]) for row in dp_rows]

        for name, series, color in (
            ("q", q_us, "#1f77b4"),
            ("res", res_us, "#ff7f0e"),
            ("lse", lse_us, "#2ca02c"),
            ("total", total_us, "#d62728"),
        ):
            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.plot(x, series, linewidth=1.6 if name != "total" else 1.8, color=color)
            ax.set_xlabel("Decode Iter")
            ax.set_ylabel("Latency (us)")
            ax.grid(True, alpha=0.25, linewidth=0.6)
            fig.tight_layout()
            fig.savefig(plots_dir / f"dp{dp_rank}_iter_{name}_latency.png", dpi=220)
            fig.savefig(plots_dir / f"dp{dp_rank}_iter_{name}_latency.pdf")
            plt.close(fig)

    # Cross-machine total-latency views, aligned by global_run_count.
    rows_by_global_run: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        rows_by_global_run[int(row["global_run_count"])][int(row["dp_rank"])] = float(row["total_us"])

    if rows_by_global_run:
        sorted_runs = sorted(rows_by_global_run)

        fig, ax = plt.subplots(figsize=(16.5, 6.8))
        for dp_rank, color in ((0, "#0B3C5D"), (1, "#9A031E"), (2, "#006D77"), (3, "#5F0F40")):
            x = [run for run in sorted_runs if dp_rank in rows_by_global_run[run]]
            y = [rows_by_global_run[run][dp_rank] for run in x]
            if x:
                ax.plot(x, y, linewidth=4.2, color=color, label=f"DP{dp_rank}")
        ax.set_xlabel("Decode Iter", fontsize=34, fontweight="semibold", color="#111111")
        ax.set_ylabel("A2A Lat. (us)", fontsize=34, fontweight="semibold", color="#111111")
        ax.tick_params(axis="both", labelsize=28, width=1.4, colors="#222222")
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("semibold")
        ax.legend(
            frameon=False,
            ncol=4,
            fontsize=28,
            handlelength=3.4,
            columnspacing=1.8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
        )
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
        fig.savefig(plots_dir / "all_dp_iter_total_latency.png", dpi=220)
        fig.savefig(plots_dir / "all_dp_iter_total_latency.pdf")
        plt.close(fig)

        dp_coverage = max((len(per_run) for per_run in rows_by_global_run.values()), default=0)
        full_cover_runs = [
            run for run in sorted_runs if len(rows_by_global_run[run]) == dp_coverage
        ]
        if full_cover_runs:
            max_total_us = [max(rows_by_global_run[run].values()) for run in full_cover_runs]
            fig, ax = plt.subplots(figsize=(16.5, 6.2))
            ax.plot(full_cover_runs, max_total_us, linewidth=4.4, color="#111111")
            ax.set_xlabel("Decode Iter", fontsize=34, fontweight="semibold", color="#111111")
            ax.set_ylabel("Max A2A (us)", fontsize=34, fontweight="semibold", color="#111111")
            ax.tick_params(axis="both", labelsize=28, width=1.4, colors="#222222")
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
            ax.grid(False, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.4)
            ax.spines["bottom"].set_linewidth(1.4)
            ax.spines["left"].set_color("#333333")
            ax.spines["bottom"].set_color("#333333")
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("semibold")
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
            fig.savefig(plots_dir / "max_dp_iter_total_latency.png", dpi=220)
            fig.savefig(plots_dir / "max_dp_iter_total_latency.pdf")
            plt.close(fig)


def cp_color(cp_size: int) -> str:
    return CP_COLORS.get(cp_size, "#4C78A8")


def style_cp_axes(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=34, fontweight="semibold", color="#111111")
    ax.set_ylabel(ylabel, fontsize=34, fontweight="semibold", color="#111111")
    ax.tick_params(axis="both", labelsize=28, width=1.4, colors="#222222")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("semibold")


def style_cp_right_axis(ax, *, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=34, fontweight="semibold", color="#111111")
    ax.tick_params(axis="y", labelsize=28, width=1.4, colors="#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_linewidth(1.4)
    ax.spines["right"].set_color("#333333")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("semibold")


def add_cp_legend(ax, handles, labels) -> None:
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=min(4, len(labels)),
        fontsize=28,
        handlelength=3.4,
        columnspacing=1.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
    )


def plot_cp_size_per_dp(rows: list[dict], cp_sizes: list[int], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows_by_dp: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_dp[int(row["dp_rank"])].append(row)

    for dp_rank, dp_rows in rows_by_dp.items():
        dp_rows.sort(key=lambda row: int(row["iter_ordinal"]))
        x = [int(row["iter_ordinal"]) for row in dp_rows]

        non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]
        fig, ax_left = plt.subplots(figsize=(14.2, 5.6))

        peak_non_cp1 = 0
        for cp_size in non_cp1:
            y = [int(row[f"cp_{cp_size}"]) for row in dp_rows]
            if y:
                peak_non_cp1 = max(peak_non_cp1, max(y))
            ax_left.step(x, y, where="mid", linewidth=2.8, color=cp_color(cp_size), label=f"CP={cp_size}")

        style_cp_axes(ax_left, xlabel="Decode Iter", ylabel="CP>1 Cnt")
        ax_left.set_ylim(0, max(1, peak_non_cp1 * 1.1))

        if 1 in cp_sizes:
            ax_right = ax_left.twinx()
            y_cp1 = [int(row["cp_1"]) for row in dp_rows]
            ax_right.step(
                x,
                y_cp1,
                where="mid",
                linewidth=2.5,
                color=cp_color(1),
                linestyle="--",
                label="CP=1",
            )
            style_cp_right_axis(ax_right, ylabel="CP1 Cnt")
        else:
            ax_right = None

        handles, labels = ax_left.get_legend_handles_labels()
        if ax_right is not None:
            h2, l2 = ax_right.get_legend_handles_labels()
            handles = h2 + handles
            labels = l2 + labels
        add_cp_legend(ax_left, handles, labels)

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
        fig.savefig(plots_dir / f"dp{dp_rank}_iter_cp_size_dual_axis.png", dpi=220)
        fig.savefig(plots_dir / f"dp{dp_rank}_iter_cp_size_dual_axis.pdf")
        plt.close(fig)

        if non_cp1:
            fig, ax = plt.subplots(figsize=(14.2, 5.2))
            for cp_size in non_cp1:
                y = [int(row[f"cp_{cp_size}"]) for row in dp_rows]
                ax.step(x, y, where="mid", linewidth=2.8, color=cp_color(cp_size), label=f"CP={cp_size}")
            style_cp_axes(ax, xlabel="Decode Iter", ylabel="CP Cnt")
            add_cp_legend(ax, *ax.get_legend_handles_labels())
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
            fig.savefig(plots_dir / f"dp{dp_rank}_iter_cp_size_non_cp1.png", dpi=220)
            fig.savefig(plots_dir / f"dp{dp_rank}_iter_cp_size_non_cp1.pdf")
            plt.close(fig)

    rows_by_global_run: dict[int, dict[int, int]] = defaultdict(lambda: {cp_size: 0 for cp_size in cp_sizes})
    for row in rows:
        global_run = int(row["global_run_count"])
        for cp_size in cp_sizes:
            rows_by_global_run[global_run][cp_size] += int(row[f"cp_{cp_size}"])

    if rows_by_global_run:
        sorted_runs = sorted(rows_by_global_run)
        x = sorted_runs

        non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]
        fig, ax_left = plt.subplots(figsize=(14.8, 5.8))

        peak_non_cp1 = 0
        for cp_size in non_cp1:
            y = [rows_by_global_run[run][cp_size] for run in sorted_runs]
            if y:
                peak_non_cp1 = max(peak_non_cp1, max(y))
            ax_left.step(x, y, where="mid", linewidth=2.9, color=cp_color(cp_size), label=f"CP={cp_size}")

        style_cp_axes(ax_left, xlabel="Decode Iter", ylabel="CP>1 Cnt")
        ax_left.set_ylim(0, max(1, peak_non_cp1 * 1.1))

        if 1 in cp_sizes:
            ax_right = ax_left.twinx()
            y_cp1 = [rows_by_global_run[run][1] for run in sorted_runs]
            ax_right.step(
                x,
                y_cp1,
                where="mid",
                linewidth=2.6,
                color=cp_color(1),
                linestyle="--",
                label="CP=1",
            )
            style_cp_right_axis(ax_right, ylabel="CP1 Cnt")
        else:
            ax_right = None

        handles, labels = ax_left.get_legend_handles_labels()
        if ax_right is not None:
            h2, l2 = ax_right.get_legend_handles_labels()
            handles = h2 + handles
            labels = l2 + labels
        add_cp_legend(ax_left, handles, labels)

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
        fig.savefig(plots_dir / "all_dp_iter_cp_size_dual_axis.png", dpi=220)
        fig.savefig(plots_dir / "all_dp_iter_cp_size_dual_axis.pdf")
        plt.close(fig)

        if non_cp1:
            fig, ax = plt.subplots(figsize=(14.8, 5.4))
            for cp_size in non_cp1:
                y = [rows_by_global_run[run][cp_size] for run in sorted_runs]
                ax.step(x, y, where="mid", linewidth=2.9, color=cp_color(cp_size), label=f"CP={cp_size}")
            style_cp_axes(ax, xlabel="Decode Iter", ylabel="CP Cnt")
            add_cp_legend(ax, *ax.get_legend_handles_labels())
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
            fig.savefig(plots_dir / "all_dp_iter_cp_size_non_cp1.png", dpi=220)
            fig.savefig(plots_dir / "all_dp_iter_cp_size_non_cp1.pdf")
            plt.close(fig)


def plot_cluster_cp_latency_combined(latency_rows: list[dict], cp_rows: list[dict], cp_sizes: list[int], out_path: Path) -> None:
    rows_by_global_run_latency: dict[int, dict[int, float]] = defaultdict(dict)
    for row in latency_rows:
        rows_by_global_run_latency[int(row["global_run_count"])][int(row["dp_rank"])] = float(row["total_us"])

    rows_by_global_run_cp: dict[int, dict[int, int]] = defaultdict(lambda: {cp_size: 0 for cp_size in cp_sizes})
    for row in cp_rows:
        global_run = int(row["global_run_count"])
        for cp_size in cp_sizes:
            rows_by_global_run_cp[global_run][cp_size] += int(row[f"cp_{cp_size}"])

    sorted_runs = sorted(set(rows_by_global_run_latency) | set(rows_by_global_run_cp))
    if not sorted_runs:
        return
    x_min = sorted_runs[0]
    x_max = sorted_runs[-1]
    x_pad = max(5, int((x_max - x_min) * 0.003))

    fig, (ax_cp, ax_lat) = plt.subplots(
        2,
        1,
        figsize=(16.5, 11.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    fig.subplots_adjust(left=0.072, right=0.914, bottom=0.10, top=0.812, hspace=0.56)

    label_fs = 36
    tick_fs = 30
    legend_fs = 30
    line_w = 3.9

    non_cp1 = [cp_size for cp_size in cp_sizes if cp_size != 1]
    peak_non_cp1 = 0
    for cp_size in non_cp1:
        y = [rows_by_global_run_cp[run][cp_size] for run in sorted_runs]
        if y:
            peak_non_cp1 = max(peak_non_cp1, max(y))
        ax_cp.step(sorted_runs, y, where="mid", linewidth=line_w, color=cp_color(cp_size), label=f"CP={cp_size}")
    ax_cp.set_ylabel("CP>1 Cnt", fontsize=label_fs, fontweight="semibold", color="#111111")
    ax_cp.tick_params(axis="both", labelsize=tick_fs, width=1.4, colors="#222222")
    ax_cp.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
    ax_cp.grid(False, axis="x")
    ax_cp.spines["top"].set_visible(False)
    ax_cp.spines["right"].set_visible(False)
    ax_cp.spines["left"].set_linewidth(1.4)
    ax_cp.spines["bottom"].set_linewidth(1.4)
    ax_cp.spines["left"].set_color("#333333")
    ax_cp.spines["bottom"].set_color("#333333")
    ax_cp.set_xlabel("")
    ax_cp.tick_params(axis="x", labelbottom=False)
    ax_cp.set_ylim(0, max(1, peak_non_cp1 * 1.1))
    ax_cp.set_xlim(x_min - x_pad, x_max + x_pad)
    for tick in ax_cp.get_xticklabels() + ax_cp.get_yticklabels():
        tick.set_fontweight("semibold")

    cp_handles, cp_labels = ax_cp.get_legend_handles_labels()
    if 1 in cp_sizes:
        ax_cp_right = ax_cp.twinx()
        y_cp1 = [rows_by_global_run_cp[run][1] for run in sorted_runs]
        ax_cp_right.step(
            sorted_runs,
            y_cp1,
            where="mid",
            linewidth=line_w - 0.2,
            color=cp_color(1),
            linestyle="--",
            label="CP=1",
        )
        ax_cp_right.set_ylim(bottom=0, top=max(1, max(y_cp1) * 1.04))
        ax_cp_right.set_ylabel("CP=1 Cnt", fontsize=label_fs, fontweight="semibold", color="#111111", rotation=270, labelpad=30)
        ax_cp_right.tick_params(axis="y", labelsize=tick_fs, width=1.4, colors="#222222", pad=4)
        ax_cp_right.spines["top"].set_visible(False)
        ax_cp_right.spines["left"].set_visible(False)
        ax_cp_right.spines["right"].set_linewidth(1.4)
        ax_cp_right.spines["right"].set_color("#333333")
        for tick in ax_cp_right.get_yticklabels():
            tick.set_fontweight("semibold")
        h2, l2 = ax_cp_right.get_legend_handles_labels()
        cp_handles = h2 + cp_handles
        cp_labels = l2 + cp_labels
    cp_legend = ax_cp.legend(
        cp_handles,
        cp_labels,
        frameon=False,
        ncol=3,
        fontsize=legend_fs,
        handlelength=3.2,
        columnspacing=1.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    for text in cp_legend.get_texts():
        text.set_fontweight("semibold")

    for dp_rank, color in ((0, "#0B3C5D"), (1, "#9A031E"), (2, "#006D77"), (3, "#5F0F40")):
        x = [run for run in sorted_runs if dp_rank in rows_by_global_run_latency[run]]
        y = [rows_by_global_run_latency[run][dp_rank] for run in x]
        if x:
            ax_lat.plot(x, y, linewidth=line_w, color=color, label=f"DP{dp_rank}")
    lat_top = 35.0
    ax_lat.set_xlabel("Decode Iter", fontsize=label_fs, fontweight="semibold", color="#111111")
    ax_lat.set_ylabel("A2A Lat. (us)", fontsize=label_fs, fontweight="semibold", color="#111111")
    ax_lat.tick_params(axis="both", labelsize=tick_fs, width=1.4, colors="#222222")
    ax_lat.grid(True, axis="y", alpha=0.18, linewidth=0.8, color="#7A7A7A")
    ax_lat.grid(False, axis="x")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.spines["left"].set_linewidth(1.4)
    ax_lat.spines["bottom"].set_linewidth(1.4)
    ax_lat.spines["left"].set_color("#333333")
    ax_lat.spines["bottom"].set_color("#333333")
    ax_lat.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_lat.set_ylim(20, lat_top)
    ax_lat.set_yticks([20, 25, 30, 35])
    for tick in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
        tick.set_fontweight("semibold")

    # Mark the truncated latency axis since the lower bound starts at 20 us.
    d = 0.016
    kwargs = dict(transform=ax_lat.transAxes, color="#333333", clip_on=False, linewidth=2.0)
    ax_lat.plot((-d, +d), (-d, +d), **kwargs)
    ax_lat.plot((-d, +d), (0.038 - d, 0.038 + d), **kwargs)
    dp_legend = ax_lat.legend(
        frameon=False,
        ncol=4,
        fontsize=legend_fs,
        handlelength=3.2,
        columnspacing=1.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
    )
    for text in dp_legend.get_texts():
        text.set_fontweight("semibold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload_names = parse_requested_payloads(args.payloads)
    decode_log = Path(args.decode_log).resolve() if args.decode_log else None
    if args.reuse_extraction:
        if args.output_dir is None:
            raise ValueError("--reuse-extraction requires --output-dir")
    else:
        if decode_log is None:
            raise ValueError("--decode-log is required unless --reuse-extraction is used")
        if not decode_log.exists():
            raise FileNotFoundError(f"decode log not found: {decode_log}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else decode_log.parent / "iter_q_res_lse_latency"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "iter_case_mapping.csv"
    extract_summary_path = output_dir / "extract_summary.json"
    approximation_plan_path = output_dir / "approximation_plan.json"
    approximation_mapping_path = output_dir / "approximation_case_mapping.csv"
    approximation_summary_path = output_dir / "approximation_summary.json"
    trace_dir = output_dir / "traces"
    per_iter_csv_path = output_dir / "per_iter_latencies.csv"
    per_iter_cp_hist_csv_path = output_dir / "per_iter_cp_size_hist.csv"
    plots_dir = output_dir / "plots"
    cp_size_plots_dir = output_dir / "cp_size_plots"
    cases_dir = output_dir / "cases"
    case_paths = {
        payload_name: cases_dir / f"unique_{payload_name.lower()}_cases.json"
        for payload_name in payload_names
    }
    benchmark_case_paths = {
        payload_name: (
            cases_dir / f"benchmark_{payload_name.lower()}_cases.json"
            if args.approximate
            else case_paths[payload_name]
        )
        for payload_name in payload_names
    }

    compact_approximation_reuse = False
    preloaded_approximation_plan = None
    if args.reuse_extraction:
        mapping_rows = load_mapping_csv(mapping_path)
        if not extract_summary_path.is_file():
            raise FileNotFoundError(f"extraction summary not found: {extract_summary_path}")
        extract_summary = json.loads(extract_summary_path.read_text(encoding="utf-8"))
        decode_log_display = str(extract_summary.get("log_path", "reused extraction"))
        if args.approximate and approximation_plan_path.is_file():
            compact_approximation_reuse = True
            preloaded_approximation_plan = load_approximation_plan(
                approximation_plan_path,
                benchmark_case_paths,
            )
            plan_config = (
                preloaded_approximation_plan["bucket_step"],
                preloaded_approximation_plan["samples_per_bucket"],
                preloaded_approximation_plan["requested_lse_samples"],
            )
            requested_config = (args.bucket_step, args.samples_per_bucket, args.lse_samples)
            if requested_config != plan_config:
                raise ValueError(
                    "Approximation options do not match approximation_plan.json: "
                    f"requested {requested_config}, stored {plan_config}"
                )
            unique_cases_by_payload = None
            unique_case_count = {
                payload_name: len(preloaded_approximation_plan["traffic_by_payload"][payload_name])
                for payload_name in payload_names
            }
        else:
            unique_cases_by_payload = {
                payload_name: load_unique_cases(case_paths[payload_name]) for payload_name in payload_names
            }
            unique_case_count = {
                payload_name: len(unique_cases_by_payload[payload_name]) for payload_name in payload_names
            }
    else:
        unique_cases_by_payload, mapping_rows, extract_summary = extract_unique_cases(
            decode_log,
            max_groups=args.max_groups,
        )
        write_mapping_csv(mapping_rows, mapping_path)
        extract_summary["generated_at_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        extract_summary_path.write_text(json.dumps(extract_summary, indent=2))
        unique_case_count = {
            payload_name: len(unique_cases_by_payload[payload_name]) for payload_name in payload_names
        }
        if not args.approximate:
            for payload_name in payload_names:
                write_unique_cases(unique_cases_by_payload[payload_name], case_paths[payload_name])
        decode_log_display = str(decode_log)

    if not all(unique_case_count[payload_name] for payload_name in payload_names):
        raise RuntimeError("Q, Res, and Lse unique cases are all required")

    approximation_plan = None
    benchmark_cases_by_payload = unique_cases_by_payload
    if args.approximate:
        if preloaded_approximation_plan is not None:
            approximation_plan = preloaded_approximation_plan
        else:
            approximation_plan = build_approximation_plan(
                unique_cases_by_payload,
                mapping_rows,
                bucket_step=args.bucket_step,
                samples_per_bucket=args.samples_per_bucket,
                lse_samples=args.lse_samples,
            )
            write_approximation_plan(approximation_plan, approximation_plan_path)
        benchmark_cases_by_payload = approximation_plan["benchmark_cases_by_payload"]

    bench_summary_paths = {}
    bench_csv_paths = {}
    latency_maps = {}
    for payload_name in payload_names:
        payload_key = payload_name.lower()
        bench_summary_paths[payload_name] = output_dir / f"{payload_key}_latency_summary.json"
        bench_csv_paths[payload_name] = output_dir / f"{payload_key}_latency_dataset.csv"
        if args.approximate and not compact_approximation_reuse:
            write_unique_cases(benchmark_cases_by_payload[payload_name], benchmark_case_paths[payload_name])

    if compact_approximation_reuse:
        per_iter_cp_hist_rows, observed_cp_sizes = load_per_iter_cp_hist_csv(per_iter_cp_hist_csv_path)
    else:
        per_iter_cp_hist_rows, observed_cp_sizes = materialize_per_iter_cp_hist_rows(
            mapping_rows, unique_cases_by_payload["Q"]
        )
        write_per_iter_cp_hist_csv(per_iter_cp_hist_rows, observed_cp_sizes, per_iter_cp_hist_csv_path)
    if args.diagnostic_plots:
        plot_cp_size_per_dp(per_iter_cp_hist_rows, observed_cp_sizes, cp_size_plots_dir)

    if args.skip_benchmark:
        summary = {
            "decode_log": decode_log_display,
            "output_dir": str(output_dir),
            "payloads": payload_names,
            "unique_case_count": unique_case_count,
            "iter_case_count": len(mapping_rows),
            "metric": args.metric,
            "requested_benchmark_config": requested_benchmark_config(args),
            "requested_approximation_config": requested_approximation_config(args),
            "approximation_plan": summarize_approximation_plan(approximation_plan) if approximation_plan else None,
            "cases_path": {
                payload_name: str(case_paths[payload_name])
                for payload_name in payload_names
                if case_paths[payload_name].is_file()
            },
            "benchmark_cases_path": {
                payload_name: str(benchmark_case_paths[payload_name]) for payload_name in payload_names
            },
            "approximation_plan_path": str(approximation_plan_path) if approximation_plan else None,
            "mapping_path": str(mapping_path),
            "extract_summary_path": str(extract_summary_path),
            "per_iter_cp_hist_csv_path": str(per_iter_cp_hist_csv_path),
            "cp_size_plots_dir": str(cp_size_plots_dir) if args.diagnostic_plots else None,
            "benchmark_skipped": True,
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    for payload_name in payload_names:
        if not args.skip_benchmark:
            if not args.reuse_benchmark or not (
                bench_summary_paths[payload_name].exists() and bench_csv_paths[payload_name].exists()
            ):
                run_benchmark(
                    args,
                    payload_name=payload_name,
                    cases_path=benchmark_case_paths[payload_name],
                    summary_path=bench_summary_paths[payload_name],
                    csv_path=bench_csv_paths[payload_name],
                    trace_dir=trace_dir / payload_name,
                )

        if not bench_csv_paths[payload_name].exists():
            raise FileNotFoundError(f"benchmark csv not found for {payload_name}: {bench_csv_paths[payload_name]}")
        latency_maps[payload_name] = load_case_latencies(bench_csv_paths[payload_name], args.metric)

    approximation_summary = None
    if approximation_plan is not None:
        latency_maps, approximation_mapping_rows, approximation_summary = expand_approximate_latency_maps(
            approximation_plan,
            latency_maps,
        )
        write_approximation_mapping_csv(approximation_mapping_rows, approximation_mapping_path)
        approximation_summary_path.write_text(json.dumps(approximation_summary, indent=2))

    per_iter_rows = materialize_per_iter_rows(
        mapping_rows,
        q_latencies=latency_maps["Q"],
        res_latencies=latency_maps["Res"],
        lse_latencies=latency_maps["Lse"],
    )
    write_per_iter_csv(per_iter_rows, per_iter_csv_path)
    if args.diagnostic_plots:
        plot_per_dp(per_iter_rows, plots_dir)
        plot_cluster_cp_latency_combined(
            per_iter_rows,
            per_iter_cp_hist_rows,
            observed_cp_sizes,
            plots_dir / "all_dp_iter_cp_latency_combined.png",
        )

    summary = {
        "decode_log": decode_log_display,
        "output_dir": str(output_dir),
        "payloads": payload_names,
        "unique_case_count": unique_case_count,
        "iter_case_count": len(mapping_rows),
        "metric": args.metric,
        "requested_benchmark_config": requested_benchmark_config(args),
        "requested_approximation_config": requested_approximation_config(args),
        "cases_path": {
            payload_name: str(case_paths[payload_name])
            for payload_name in payload_names
            if case_paths[payload_name].is_file()
        },
        "benchmark_cases_path": {
            payload_name: str(benchmark_case_paths[payload_name]) for payload_name in payload_names
        },
        "approximation_plan_path": str(approximation_plan_path) if approximation_plan else None,
        "mapping_path": str(mapping_path),
        "benchmark_summary_path": {payload_name: str(bench_summary_paths[payload_name]) for payload_name in payload_names},
        "benchmark_csv_path": {payload_name: str(bench_csv_paths[payload_name]) for payload_name in payload_names},
        "per_iter_csv_path": str(per_iter_csv_path),
        "per_iter_cp_hist_csv_path": str(per_iter_cp_hist_csv_path),
        "plots_dir": str(plots_dir) if args.diagnostic_plots else None,
        "cp_size_plots_dir": str(cp_size_plots_dir) if args.diagnostic_plots else None,
        "approximation_summary_path": str(approximation_summary_path) if approximation_summary else None,
        "approximation_mapping_path": str(approximation_mapping_path) if approximation_summary else None,
        "approximation": approximation_summary,
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

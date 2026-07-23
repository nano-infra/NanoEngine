#!/usr/bin/env python3
"""Static 2x2 LoongServe-style Decode data-plane benchmark.

This harness deliberately bypasses ``LLMEngine.step()`` and the production
scheduler.  It installs one fixed batch and KV placement in the driver's
SPStateManagers, sends that metadata through the normal DLSlime endpoint, and
invokes the existing ModelRunner workers.  The primary measurement is the
per-inner-loop distributed CUDA-event critical path.

The four cases isolate two factors from the 2026-07-23 rate=20 run:

    T00: all requests use DoP1; master batches are balanced.
    T10: reproduce the observed DoP1/2/3 histogram; masters are balanced.
    T01: all requests use DoP1; reproduce the observed master skew.
    T11: reproduce both the observed DoP histogram and master skew.

No scheduler policy is changed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from ls_decode_issue001_profile import (
        clear_ray_proxy_env,
        formal_engine_kwargs,
        resolved_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - module-style invocation
    from scripts.ls_decode_issue001_profile import (
        clear_ray_proxy_env,
        formal_engine_kwargs,
        resolved_manifest,
    )


ATTENTION_DP = 2
ATTENTION_SP = 8
BATCH_PER_DP = 520
BLOCK_SIZE = 64
CASE_NAMES = ("T00", "T10", "T01", "T11")

# Exact values from iteration 118 of
# docs-dev/2026-07-23/ls_style_loop16_dp2sp8_r20_diag01_3.log.
OBSERVED_MASTER_COUNTS = (
    (102, 75, 0, 1, 120, 113, 58, 51),
    (16, 138, 66, 1, 36, 125, 16, 122),
)
BALANCED_MASTER_COUNTS = ((65,) * ATTENTION_SP,) * ATTENTION_DP
OBSERVED_DOP_COUNTS = (
    # DoP1, DoP2, DoP3
    (287, 174, 59),
    (287, 173, 60),
)
ALL_D1_COUNTS = ((BATCH_PER_DP, 0, 0),) * ATTENTION_DP


@dataclass(frozen=True)
class StaticSequenceLayout:
    """One request's immutable prompt placement before the pending Decode token."""

    dp_idx: int
    local_index: int
    master_sp_idx: int
    dispatched_tokens: tuple[int, ...]

    @property
    def dop(self) -> int:
        return sum(tokens > 0 for tokens in self.dispatched_tokens)

    @property
    def context_len(self) -> int:
        return sum(self.dispatched_tokens)


@dataclass(frozen=True)
class StaticCaseLayout:
    case: str
    cp_factor: str
    master_factor: str
    sequences: tuple[StaticSequenceLayout, ...]

    def sequences_for_dp(self, dp_idx: int) -> tuple[StaticSequenceLayout, ...]:
        return tuple(seq for seq in self.sequences if seq.dp_idx == dp_idx)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def _apportion(
    total: int,
    weights: Sequence[int],
    capacities: Sequence[int],
    *,
    tie_offset: int,
) -> list[int]:
    """Proportionally apportion an integer class count with exact totals."""
    if total < 0 or len(weights) != len(capacities):
        raise ValueError("invalid apportionment inputs")
    if total > sum(capacities):
        raise ValueError("apportionment exceeds capacity")
    if total == 0:
        return [0] * len(weights)

    positive_weight = sum(max(weight, 0) for weight in weights)
    if positive_weight <= 0:
        raise ValueError("positive apportionment requires a positive weight")

    ideals = [total * max(weight, 0) / positive_weight for weight in weights]
    result = [
        min(capacity, math.floor(ideal)) for ideal, capacity in zip(ideals, capacities)
    ]
    remaining = total - sum(result)
    rank_count = len(weights)
    while remaining:
        candidates = [
            rank for rank in range(rank_count) if result[rank] < capacities[rank]
        ]
        if not candidates:
            raise RuntimeError("apportionment ran out of capacity")
        rank = max(
            candidates,
            key=lambda candidate: (
                ideals[candidate] - result[candidate],
                weights[candidate],
                -((candidate - tie_offset) % rank_count),
            ),
        )
        result[rank] += 1
        remaining -= 1
    return result


def _dop_counts_by_master(
    master_counts: Sequence[int],
    dop_counts: Sequence[int],
    *,
    tie_offset: int,
) -> list[list[int]]:
    """Return [master][dop-1] counts while keeping CP rates proportional."""
    if len(dop_counts) != 3 or sum(master_counts) != sum(dop_counts):
        raise ValueError("master and DoP totals must match")

    dop3 = _apportion(
        dop_counts[2],
        master_counts,
        master_counts,
        tie_offset=tie_offset,
    )
    remaining_capacity = [
        master_counts[rank] - dop3[rank] for rank in range(len(master_counts))
    ]
    dop2 = _apportion(
        dop_counts[1],
        master_counts,
        remaining_capacity,
        tie_offset=tie_offset + 1,
    )
    return [
        [master_counts[rank] - dop2[rank] - dop3[rank], dop2[rank], dop3[rank]]
        for rank in range(len(master_counts))
    ]


def _interleave_dops(counts: Sequence[int], *, offset: int) -> list[int]:
    """Spread each master's DoP classes through request order."""
    remaining = list(counts)
    result: list[int] = []
    cursor = offset % len(remaining)
    while sum(remaining):
        candidates = [dop_idx for dop_idx, count in enumerate(remaining) if count > 0]
        dop_idx = max(
            candidates,
            key=lambda candidate: (
                remaining[candidate],
                -((candidate - cursor) % len(remaining)),
            ),
        )
        result.append(dop_idx + 1)
        remaining[dop_idx] -= 1
        cursor = (dop_idx + 1) % len(remaining)
    return result


def _round_robin_master_order(
    master_counts: Sequence[int], *, offset: int
) -> list[int]:
    remaining = list(master_counts)
    result: list[int] = []
    rank_count = len(remaining)
    while sum(remaining):
        progressed = False
        for index in range(rank_count):
            rank = (index + offset) % rank_count
            if remaining[rank] <= 0:
                continue
            result.append(rank)
            remaining[rank] -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("failed to construct master order")
    return result


def _select_owner(
    *,
    excluded: set[int],
    tokens: int,
    kv_token_load: Sequence[int],
    participant_load: Sequence[int],
    receiver_load: Sequence[int],
    max_num_recv_seqs: int,
    tie_offset: int,
    prefer_participant_balance: bool,
) -> int:
    candidates = [
        rank
        for rank in range(len(kv_token_load))
        if rank not in excluded and receiver_load[rank] < max_num_recv_seqs
    ]
    if not candidates:
        raise ValueError("no receiver rank has capacity for the requested CP layout")

    def score(rank: int) -> tuple[int, int, int, int]:
        projected_kv = kv_token_load[rank] + tokens
        projected_participants = participant_load[rank] + 1
        primary = projected_participants if prefer_participant_balance else projected_kv
        secondary = (
            projected_kv if prefer_participant_balance else projected_participants
        )
        return (
            primary,
            secondary,
            receiver_load[rank] + 1,
            (rank - tie_offset) % len(kv_token_load),
        )

    return min(candidates, key=score)


def build_case_layout(
    case: str,
    *,
    context_len: int = 800,
    cp_shard_tokens: int = 16,
    max_num_seqs: int = 256,
    max_num_recv_seqs: int = 128,
    seed: int = 0,
) -> StaticCaseLayout:
    """Build one deterministic counterfactual layout without GPU dependencies."""
    case = case.upper()
    if case not in CASE_NAMES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASE_NAMES}")
    if context_len < 3 * cp_shard_tokens:
        raise ValueError("context_len must fit the two small shards of a DoP3 request")
    if cp_shard_tokens <= 0:
        raise ValueError("cp_shard_tokens must be positive")

    cp_is_observed = case[1] == "1"
    master_is_observed = case[2] == "1"
    master_profile = (
        OBSERVED_MASTER_COUNTS if master_is_observed else BALANCED_MASTER_COUNTS
    )
    dop_profile = OBSERVED_DOP_COUNTS if cp_is_observed else ALL_D1_COUNTS

    all_layouts: list[StaticSequenceLayout] = []
    for dp_idx in range(ATTENTION_DP):
        master_counts = master_profile[dp_idx]
        if max(master_counts) > max_num_seqs:
            raise ValueError(
                f"DP{dp_idx} master batch {max(master_counts)} exceeds "
                f"max_num_seqs={max_num_seqs}"
            )
        per_master_dops = _dop_counts_by_master(
            master_counts,
            dop_profile[dp_idx],
            tie_offset=(seed + dp_idx) % ATTENTION_SP,
        )
        dop_queues = [
            _interleave_dops(
                per_master_dops[rank],
                offset=seed + dp_idx + rank,
            )
            for rank in range(ATTENTION_SP)
        ]
        master_order = _round_robin_master_order(
            master_counts,
            offset=(seed + dp_idx) % ATTENTION_SP,
        )
        request_specs: list[tuple[int, int, int]] = []
        per_master_cursor = [0] * ATTENTION_SP
        for local_index, master in enumerate(master_order):
            dop = dop_queues[master][per_master_cursor[master]]
            per_master_cursor[master] += 1
            request_specs.append((local_index, master, dop))

        # DoP1 fixes owner==master. Seed the loads with all such requests before
        # greedily placing CP fragments so both CP cases are as KV-balanced as
        # their master constraints permit.
        kv_token_load = [0] * ATTENTION_SP
        participant_load = [0] * ATTENTION_SP
        receiver_load = [0] * ATTENTION_SP
        placements: dict[int, tuple[int, ...]] = {}
        for local_index, master, dop in request_specs:
            if dop != 1:
                continue
            dispatched = [0] * ATTENTION_SP
            dispatched[master] = context_len
            placements[local_index] = tuple(dispatched)
            kv_token_load[master] += context_len
            participant_load[master] += 1

        for local_index, master, dop in request_specs:
            if dop == 1:
                continue
            dominant_tokens = context_len - cp_shard_tokens * (dop - 1)
            if dominant_tokens <= 0:
                raise ValueError("CP dominant shard must contain at least one token")

            dispatched = [0] * ATTENTION_SP
            dispatched[master] = cp_shard_tokens
            kv_token_load[master] += cp_shard_tokens
            participant_load[master] += 1

            dominant = _select_owner(
                excluded={master},
                tokens=dominant_tokens,
                kv_token_load=kv_token_load,
                participant_load=participant_load,
                receiver_load=receiver_load,
                max_num_recv_seqs=max_num_recv_seqs,
                tie_offset=seed + dp_idx + local_index,
                prefer_participant_balance=False,
            )
            dispatched[dominant] = dominant_tokens
            kv_token_load[dominant] += dominant_tokens
            participant_load[dominant] += 1
            receiver_load[dominant] += 1

            if dop == 3:
                minor = _select_owner(
                    excluded={master, dominant},
                    tokens=cp_shard_tokens,
                    kv_token_load=kv_token_load,
                    participant_load=participant_load,
                    receiver_load=receiver_load,
                    max_num_recv_seqs=max_num_recv_seqs,
                    tie_offset=seed + dp_idx + local_index + 1,
                    prefer_participant_balance=True,
                )
                dispatched[minor] = cp_shard_tokens
                kv_token_load[minor] += cp_shard_tokens
                participant_load[minor] += 1
                receiver_load[minor] += 1

            placements[local_index] = tuple(dispatched)

        for local_index, master, _ in request_specs:
            all_layouts.append(
                StaticSequenceLayout(
                    dp_idx=dp_idx,
                    local_index=local_index,
                    master_sp_idx=master,
                    dispatched_tokens=placements[local_index],
                )
            )

    layout = StaticCaseLayout(
        case=case,
        cp_factor="observed_dop_histogram" if cp_is_observed else "all_dop1",
        master_factor="observed_skew" if master_is_observed else "balanced",
        sequences=tuple(all_layouts),
    )
    validate_case_layout(
        layout,
        context_len=context_len,
        max_num_seqs=max_num_seqs,
        max_num_recv_seqs=max_num_recv_seqs,
    )
    return layout


def validate_case_layout(
    layout: StaticCaseLayout,
    *,
    context_len: int,
    max_num_seqs: int,
    max_num_recv_seqs: int,
) -> None:
    if len(layout.sequences) != ATTENTION_DP * BATCH_PER_DP:
        raise ValueError("case layout has the wrong total batch size")
    cp_is_observed = layout.case[1] == "1"
    master_is_observed = layout.case[2] == "1"
    expected_dops = OBSERVED_DOP_COUNTS if cp_is_observed else ALL_D1_COUNTS
    expected_masters = (
        OBSERVED_MASTER_COUNTS if master_is_observed else BALANCED_MASTER_COUNTS
    )

    for dp_idx in range(ATTENTION_DP):
        sequences = layout.sequences_for_dp(dp_idx)
        if len(sequences) != BATCH_PER_DP:
            raise ValueError(f"DP{dp_idx} has the wrong batch size")
        if sorted(seq.local_index for seq in sequences) != list(range(BATCH_PER_DP)):
            raise ValueError(f"DP{dp_idx} local request IDs are not unique and dense")
        if any(len(seq.dispatched_tokens) != ATTENTION_SP for seq in sequences):
            raise ValueError("placement width does not equal attention_sp")
        if any(seq.context_len != context_len for seq in sequences):
            raise ValueError("placement does not cover the configured context")

        master_counts = Counter(seq.master_sp_idx for seq in sequences)
        actual_masters = tuple(master_counts[rank] for rank in range(ATTENTION_SP))
        if actual_masters != expected_masters[dp_idx]:
            raise ValueError(
                f"DP{dp_idx} master counts {actual_masters} != "
                f"{expected_masters[dp_idx]}"
            )
        if max(actual_masters) > max_num_seqs:
            raise ValueError("master count exceeds max_num_seqs")

        dop_counts = Counter(seq.dop for seq in sequences)
        actual_dops = tuple(dop_counts[dop] for dop in (1, 2, 3))
        if actual_dops != expected_dops[dp_idx]:
            raise ValueError(
                f"DP{dp_idx} DoP counts {actual_dops} != {expected_dops[dp_idx]}"
            )

        recv_counts = [0] * ATTENTION_SP
        for seq in sequences:
            for rank, tokens in enumerate(seq.dispatched_tokens):
                if tokens > 0 and rank != seq.master_sp_idx:
                    recv_counts[rank] += 1
        if max(recv_counts) > max_num_recv_seqs:
            raise ValueError(
                f"DP{dp_idx} receiver load {max(recv_counts)} exceeds "
                f"max_num_recv_seqs={max_num_recv_seqs}"
            )


def summarize_case_layout(
    layout: StaticCaseLayout,
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    per_dp: list[dict[str, Any]] = []
    global_dop_hist: Counter[int] = Counter()
    total_attention_rank_work = 0
    for dp_idx in range(ATTENTION_DP):
        sequences = layout.sequences_for_dp(dp_idx)
        master_counts = [
            sum(seq.master_sp_idx == rank for seq in sequences)
            for rank in range(ATTENTION_SP)
        ]
        participant_counts = [
            sum(seq.dispatched_tokens[rank] > 0 for seq in sequences)
            for rank in range(ATTENTION_SP)
        ]
        receiver_counts = [
            sum(
                seq.dispatched_tokens[rank] > 0 and seq.master_sp_idx != rank
                for seq in sequences
            )
            for rank in range(ATTENTION_SP)
        ]
        kv_tokens = [
            sum(seq.dispatched_tokens[rank] for seq in sequences)
            for rank in range(ATTENTION_SP)
        ]
        kv_blocks = [
            sum(
                math.ceil(seq.dispatched_tokens[rank] / block_size)
                for seq in sequences
                if seq.dispatched_tokens[rank] > 0
            )
            for rank in range(ATTENTION_SP)
        ]
        dop_hist = Counter(seq.dop for seq in sequences)
        global_dop_hist.update(dop_hist)
        total_attention_rank_work += sum(participant_counts)
        mean_master = statistics.fmean(master_counts)
        mean_kv = statistics.fmean(kv_tokens)
        per_dp.append(
            {
                "dp_idx": dp_idx,
                "batch_size": len(sequences),
                "master_batch_by_sp": master_counts,
                "effective_model_master_batch_by_sp": [
                    max(1, count) for count in master_counts
                ],
                "participant_batch_by_sp": participant_counts,
                "receiver_batch_by_sp": receiver_counts,
                "kv_prompt_tokens_by_sp": kv_tokens,
                "kv_blocks_by_sp": kv_blocks,
                "dop_histogram": {str(dop): dop_hist[dop] for dop in sorted(dop_hist)},
                "master_max_over_mean": max(master_counts) / mean_master,
                "kv_tokens_max_over_mean": max(kv_tokens) / mean_kv,
            }
        )
    return {
        "case": layout.case,
        "cp_factor": layout.cp_factor,
        "master_factor": layout.master_factor,
        "total_batch_size": len(layout.sequences),
        "attention_sequence_rank_work": total_attention_rank_work,
        "global_dop_histogram": {
            str(dop): global_dop_hist[dop] for dop in sorted(global_dop_hist)
        },
        "average_dop": total_attention_rank_work / len(layout.sequences),
        "per_dp": per_dp,
    }


def summarize_forward_timings(
    timings_by_rank: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if not timings_by_rank or not timings_by_rank[0]:
        raise ValueError("missing model-forward timings")
    loop_count = len(timings_by_rank[0])
    if any(len(rank) != loop_count for rank in timings_by_rank):
        raise ValueError("inconsistent model-forward loop counts across ranks")

    critical_path_by_loop = [
        max(rank[loop_idx] for rank in timings_by_rank)
        for loop_idx in range(loop_count)
    ]
    rank_total_ms = [sum(rank) for rank in timings_by_rank]
    return {
        "loop_count": loop_count,
        "critical_path_ms": sum(critical_path_by_loop),
        "per_loop_ms": statistics.fmean(critical_path_by_loop),
        "critical_path_by_loop_ms": critical_path_by_loop,
        "rank_total_ms": rank_total_ms,
        "rank_per_loop_ms": [total / loop_count for total in rank_total_ms],
    }


def calculate_2x2_effects(case_values: dict[str, float]) -> dict[str, float]:
    missing = set(CASE_NAMES) - set(case_values)
    if missing:
        raise ValueError(f"missing 2x2 cases: {sorted(missing)}")
    t00, t10, t01, t11 = (case_values[name] for name in CASE_NAMES)
    return {
        "baseline_T00_ms": t00,
        "cp_effect_balanced_ms_T10_minus_T00": t10 - t00,
        "imbalance_effect_dop1_ms_T01_minus_T00": t01 - t00,
        "cp_effect_skewed_ms_T11_minus_T01": t11 - t01,
        "imbalance_effect_observed_cp_ms_T11_minus_T10": t11 - t10,
        "interaction_ms": t11 - t10 - t01 + t00,
        "current_minus_ideal_ms_T11_minus_T00": t11 - t00,
        "current_over_ideal_T11_div_T00": t11 / t00,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_output_path() -> Path:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return (
        Path("docs-dev")
        / now.strftime("%Y-%m-%d")
        / f"ls_decode_static_2x2_{stamp}.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a static T00/T10/T01/T11 Decode data-plane benchmark that "
            "isolates CP expansion from master-batch skew."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--ray-address", default="10.102.206.14:7789")
    parser.add_argument("--master-address", default="10.102.206.14:27789")
    parser.add_argument(
        "--case",
        choices=(*CASE_NAMES, "all"),
        default="all",
        help="Run one case or the complete randomized 2x2 experiment.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats-per-round", type=int, default=10)
    parser.add_argument("--loop-count", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--context-len", type=int, default=800)
    parser.add_argument("--cp-shard-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only construct and print layouts; do not import Ray or use GPUs.",
    )

    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1_024_000)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=141.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument(
        "--cuda-graph-mode",
        choices=("full", "piecewise"),
        default="full",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ls-max-num-ooe", type=int, default=8)
    parser.add_argument("--ls-batch-per-master", type=int, default=64)
    parser.add_argument("--verbose-nanodeploy-logs", action="store_true")
    args = parser.parse_args(argv)

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.warmup_repeats < 0 or args.repeats_per_round <= 0:
        parser.error("warmup repeats must be non-negative; measured repeats positive")
    if args.context_len <= 0 or args.cp_shard_tokens <= 0:
        parser.error("context and CP shard lengths must be positive")
    if args.context_len < 3 * args.cp_shard_tokens:
        parser.error("--context-len must be at least 3 * --cp-shard-tokens")
    if args.max_num_seqs <= 0 or args.max_num_recv_seqs <= 0:
        parser.error("sequence limits must be positive")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.ls_max_num_ooe < 0 or args.ls_batch_per_master <= 0:
        parser.error("LS profile values are invalid")
    return args


def _selected_cases(case: str) -> tuple[str, ...]:
    return CASE_NAMES if case == "all" else (case,)


def _build_engine(args: argparse.Namespace):
    # Imports stay here so --help and --dry-run remain CPU-only.
    from nanodeploy import LLM

    ls_kwargs = formal_engine_kwargs(
        args.ls_max_num_ooe,
        ls_decode_batch_per_master=args.ls_batch_per_master,
    )
    return LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        cuda_graph_mode=args.cuda_graph_mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        master_address=args.master_address,
        ray_address=args.ray_address,
        mode="decode",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        attention_dp=ATTENTION_DP,
        attention_sp=ATTENTION_SP,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=ATTENTION_DP * ATTENTION_SP,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=args.max_num_recv_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        loop_count=args.loop_count,
        scheduler_mode="centralized",
        segment_size=args.segment_size,
        kvcache_block_size=BLOCK_SIZE,
        fixed_sp_size=0,
        sp_backend="hao_basic",
        use_dlslime_rpc=True,
        optimize_decode_block_table=True,
        enable_non_uniform_split=False,
        enable_dynamic_sp_size=False,
        use_new_decode_dynamic_sp_scheduler=False,
        dynamic_sp_size_strategy="legacy",
        **ls_kwargs,
    )


def _allocate_case(engine: Any, layout: StaticCaseLayout) -> list[list[Any]]:
    from nanodeploy._cpp import BlockContextSlot, Sequence, SequenceStatus

    allocated_by_dp: list[list[Any]] = [[] for _ in range(ATTENTION_DP)]
    try:
        for dp_idx in range(ATTENTION_DP):
            manager = engine.scheduler.worker_state[dp_idx]
            if (
                manager.num_running_seqs != 0
                or manager.num_running_tokens != 0
                or len(manager.running) != 0
            ):
                raise RuntimeError(
                    f"DP{dp_idx} is not empty before static case allocation"
                )
            sequences: list[Any] = []
            for spec in layout.sequences_for_dp(dp_idx):
                seq = Sequence(
                    [0] * spec.context_len,
                    1.0,
                    engine.config.loop_count + 2,
                    True,
                )
                seq.assigned_dp = dp_idx
                seq.active(engine.engine_id, ATTENTION_SP, ATTENTION_DP)
                seq.status = SequenceStatus.RUNNING
                seq.num_prompt_tokens = spec.context_len
                ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
                ctx.master_sp_idx = spec.master_sp_idx
                ctx.num_dispatched_tokens = list(spec.dispatched_tokens)
                sequences.append(seq)

            manager.allocate_ls_initial_batch(sequences)
            allocated_by_dp[dp_idx] = sequences
            for seq in sequences:
                master = seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
                if not manager.may_append_on_sp(seq, master, 1):
                    raise RuntimeError(
                        f"DP{dp_idx} seq={seq.seq_id} cannot reserve pending token"
                    )
                seq.append_token(0, BlockContextSlot.ACTIVE, master)
                seq.mark_last_token_pending(BlockContextSlot.ACTIVE, master)
                manager.add_running_tokens(master, 1)
                # The production LS scheduler reserves the whole Decode chunk
                # after installing its one pending input token.  This is
                # normally invisible for the synthetic 800-token layout, but
                # it is required when this allocator is reused for an exact
                # snapshot whose master frontier is close to a block boundary.
                if not manager.may_append_on_sp(seq, master, engine.config.loop_count):
                    raise RuntimeError(
                        f"DP{dp_idx} seq={seq.seq_id} cannot reserve "
                        f"{engine.config.loop_count} Decode output tokens"
                    )
                manager.running.append(seq)
        return allocated_by_dp
    except BaseException:
        _deallocate_case(engine, allocated_by_dp)
        raise


def _deallocate_case(engine: Any, allocated_by_dp: Sequence[Sequence[Any]]) -> None:
    from nanodeploy._cpp import BlockContextSlot

    cleanup_errors: list[str] = []
    for dp_idx, sequences in enumerate(allocated_by_dp):
        manager = engine.scheduler.worker_state[dp_idx]
        for seq in reversed(sequences):
            try:
                manager.running.remove(seq)
            except (ValueError, RuntimeError):
                pass
            try:
                manager.deallocate(seq, BlockContextSlot.ACTIVE)
            except BaseException as error:  # preserve every cleanup attempt
                cleanup_errors.append(f"DP{dp_idx} seq={seq.seq_id}: {error}")
        if (
            manager.num_running_seqs != 0
            or manager.num_running_tokens != 0
            or len(manager.running) != 0
        ):
            cleanup_errors.append(
                f"DP{dp_idx} retained state: "
                f"seqs={manager.num_running_seqs}, "
                f"tokens={manager.num_running_tokens}, "
                f"running={len(manager.running)}"
            )
    if cleanup_errors:
        raise RuntimeError("static layout cleanup failed: " + "; ".join(cleanup_errors))


def _execution_sequences_by_dp(
    engine: Any,
    allocated_by_dp: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """Add the scheduler-owned dummy for every zero-master SP rank.

    The production scheduler does this before publishing ``dp_sp_seqs``.  If a
    caller invokes ModelRunner directly with no master for the local SP rank,
    ModelRunner's last-resort Python dummy has no allocated block table and
    cannot be consumed by ``prepare_decode_cpp``.
    """
    from nanodeploy._cpp import BlockContextSlot

    execution_by_dp: list[list[Any]] = []
    for dp_idx, real_sequences in enumerate(allocated_by_dp):
        sequences = list(real_sequences)
        manager = engine.scheduler.worker_state[dp_idx]
        master_counts = [
            sum(
                seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_idx
                for seq in real_sequences
            )
            for sp_idx in range(ATTENTION_SP)
        ]
        for sp_idx, master_count in enumerate(master_counts):
            if master_count:
                continue
            dummy = manager.dummy_seqs[sp_idx]
            dummy_ctx = dummy.block_ctx(BlockContextSlot.ACTIVE)
            if dummy_ctx.master_sp_idx != sp_idx or not list(
                dummy.block_table(BlockContextSlot.ACTIVE, sp_idx)
            ):
                raise RuntimeError(
                    f"DP{dp_idx}/SP{sp_idx} scheduler dummy is not allocated"
                )
            sequences.append(dummy)
        execution_by_dp.append(sequences)
    return execution_by_dp


def _run_worker_iteration(
    engine: Any,
    allocated_by_dp: Sequence[Sequence[Any]],
    *,
    loop_count: int,
    timeout_sec: float,
) -> dict[str, Any]:
    import ray

    execution_by_dp = _execution_sequences_by_dp(engine, allocated_by_dp)
    dp_sp_seqs = [
        execution_by_dp[dp_idx]
        for dp_idx in range(ATTENTION_DP)
        for _ in range(ATTENTION_SP)
    ]
    send_timestamp = time.time()
    start = time.perf_counter()
    futures = [
        worker.run.remote([], False, True, send_timestamp, loop_count)
        for worker in engine.executor.workers
    ]
    engine.executor.endpoint.send_seqs(dp_sp_seqs, False)
    results = ray.get(futures, timeout=timeout_sec)
    wall_ms = (time.perf_counter() - start) * 1000.0

    timings_by_rank: list[list[float]] = []
    output_batch_by_rank: list[int] = []
    worker_end_times: list[float] = []
    for rank, result in enumerate(results):
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "worker did not return CUDA timings; "
                "NANODEPLOY_LOG_MODEL_FORWARD_TIMING must reach every worker "
                f"(rank {rank}, result type {type(result).__name__})"
            )
        token_ids, worker_end_time, forward_gpu_ms = result
        if len(forward_gpu_ms) != loop_count:
            raise RuntimeError(
                f"rank {rank} returned {len(forward_gpu_ms)} timings, "
                f"expected {loop_count}"
            )
        output_batch_by_rank.append(len(token_ids))
        worker_end_times.append(float(worker_end_time))
        timings_by_rank.append([float(value) for value in forward_gpu_ms])

    forward = summarize_forward_timings(timings_by_rank)
    return {
        "wall_ms": wall_ms,
        "worker_finish_skew_ms": (
            (max(worker_end_times) - min(worker_end_times)) * 1000.0
        ),
        "output_batch_by_global_rank": output_batch_by_rank,
        "forward": forward,
        "timings_by_global_rank_ms": timings_by_rank,
    }


def _run_case_block(
    engine: Any,
    layout: StaticCaseLayout,
    *,
    round_idx: int,
    warmup_repeats: int,
    measured_repeats: int,
    loop_count: int,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    print(
        f"[{_utc_now()}] round={round_idx} case={layout.case} "
        f"allocate batch={len(layout.sequences)}",
        flush=True,
    )
    allocated = _allocate_case(engine, layout)
    try:
        expected_output_batch = [
            max(
                1,
                sum(
                    spec.master_sp_idx == sp_idx
                    for spec in layout.sequences_for_dp(dp_idx)
                ),
            )
            for dp_idx in range(ATTENTION_DP)
            for sp_idx in range(ATTENTION_SP)
        ]
        for warmup_idx in range(warmup_repeats):
            measurement = _run_worker_iteration(
                engine,
                allocated,
                loop_count=loop_count,
                timeout_sec=timeout_sec,
            )
            if measurement["output_batch_by_global_rank"] != expected_output_batch:
                raise RuntimeError("worker output batches do not match static masters")
            print(
                f"[{_utc_now()}] round={round_idx} case={layout.case} "
                f"warmup={warmup_idx} gpu_per_loop_ms="
                f"{measurement['forward']['per_loop_ms']:.4f}",
                flush=True,
            )

        measurements: list[dict[str, Any]] = []
        for repeat_idx in range(measured_repeats):
            measurement = _run_worker_iteration(
                engine,
                allocated,
                loop_count=loop_count,
                timeout_sec=timeout_sec,
            )
            if measurement["output_batch_by_global_rank"] != expected_output_batch:
                raise RuntimeError("worker output batches do not match static masters")
            measurement["round"] = round_idx
            measurement["repeat_in_round"] = repeat_idx
            measurements.append(measurement)
            print(
                f"[{_utc_now()}] round={round_idx} case={layout.case} "
                f"repeat={repeat_idx} gpu_per_loop_ms="
                f"{measurement['forward']['per_loop_ms']:.4f} "
                f"wall_per_loop_ms={measurement['wall_ms'] / loop_count:.4f}",
                flush=True,
            )
        return measurements
    finally:
        _deallocate_case(engine, allocated)


def _summarize_measurements(measurements: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gpu_per_loop = [
        float(measurement["forward"]["per_loop_ms"]) for measurement in measurements
    ]
    wall_per_loop = [
        float(measurement["wall_ms"]) / int(measurement["forward"]["loop_count"])
        for measurement in measurements
    ]
    worker_finish_skew = [
        float(measurement["worker_finish_skew_ms"]) for measurement in measurements
    ]
    rank_count = len(measurements[0]["forward"]["rank_per_loop_ms"])
    rank_per_loop = [
        summarize_values(
            [
                float(measurement["forward"]["rank_per_loop_ms"][rank])
                for measurement in measurements
            ]
        )
        for rank in range(rank_count)
    ]
    by_round: dict[str, Any] = {}
    for round_idx in sorted({int(item["round"]) for item in measurements}):
        values = [
            float(item["forward"]["per_loop_ms"])
            for item in measurements
            if int(item["round"]) == round_idx
        ]
        by_round[str(round_idx)] = summarize_values(values)
    return {
        "gpu_critical_path_per_loop_ms": summarize_values(gpu_per_loop),
        "wall_per_loop_ms": summarize_values(wall_per_loop),
        "worker_finish_skew_ms": summarize_values(worker_finish_skew),
        "gpu_rank_per_loop_ms": rank_per_loop,
        "gpu_critical_path_per_loop_ms_by_round": by_round,
    }


def _configuration_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attention_dp": ATTENTION_DP,
        "attention_sp": ATTENTION_SP,
        "ffn_ep": ATTENTION_DP * ATTENTION_SP,
        "batch_per_dp": BATCH_PER_DP,
        "total_batch_size": ATTENTION_DP * BATCH_PER_DP,
        "context_len": args.context_len,
        "cp_shard_tokens": args.cp_shard_tokens,
        "loop_count": args.loop_count,
        "rounds": args.rounds,
        "warmup_repeats": args.warmup_repeats,
        "repeats_per_round": args.repeats_per_round,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "max_num_recv_seqs": args.max_num_recv_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cuda_graph_mode": args.cuda_graph_mode,
        "enforce_eager": args.enforce_eager,
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "model_path": args.model_path,
        "ls_max_num_ooe": args.ls_max_num_ooe,
        "ls_batch_per_master": args.ls_batch_per_master,
        "primary_metric": "distributed CUDA-event critical path per inner loop",
        "scheduler_in_timed_path": False,
        "transport": "DLSlime sequence metadata endpoint",
    }


def _dry_run_payload(
    args: argparse.Namespace,
    layouts: dict[str, StaticCaseLayout],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ls_decode_static_layout_dry_run",
        "created_at": _utc_now(),
        "configuration": _configuration_dict(args),
        "layouts": {
            case: summarize_case_layout(layout) for case, layout in layouts.items()
        },
    }


def _round_effects(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = sorted(
        {
            int(measurement["round"])
            for case in CASE_NAMES
            for measurement in results[case]["measurements"]
        }
    )
    effects: list[dict[str, float | int]] = []
    for round_idx in rounds:
        values = {
            case: statistics.median(
                [
                    float(item["forward"]["per_loop_ms"])
                    for item in results[case]["measurements"]
                    if int(item["round"]) == round_idx
                ]
            )
            for case in CASE_NAMES
        }
        effects.append(
            {
                "round": round_idx,
                **calculate_2x2_effects(values),
            }
        )
    effect_keys = [key for key in effects[0] if key not in {"round", "baseline_T00_ms"}]
    return {
        "per_round_from_case_medians": effects,
        "summary": {
            key: summarize_values([float(effect[key]) for effect in effects])
            for key in effect_keys
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = _selected_cases(args.case)
    layouts = {
        case: build_case_layout(
            case,
            context_len=args.context_len,
            cp_shard_tokens=args.cp_shard_tokens,
            max_num_seqs=args.max_num_seqs,
            max_num_recv_seqs=args.max_num_recv_seqs,
            seed=args.seed,
        )
        for case in cases
    }

    if args.dry_run:
        payload = _dry_run_payload(args, layouts)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output_json is not None:
            _atomic_write_json(args.output_json, payload)
        return 0

    output_path = args.output_json or _default_output_path()
    cleared_proxy_env = clear_ray_proxy_env()
    os.environ["NANODEPLOY_LOG_MODEL_FORWARD_TIMING"] = "1"
    if args.verbose_nanodeploy_logs:
        os.environ["NANODEPLOY_LOG_LEVEL"] = "DEBUG"

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ls_decode_static_layout_2x2",
        "status": "initializing",
        "started_at": _utc_now(),
        "command": sys.argv,
        "cleared_proxy_env": cleared_proxy_env,
        "configuration": _configuration_dict(args),
        "source_observation": {
            "log": ("docs-dev/2026-07-23/" "ls_style_loop16_dp2sp8_r20_diag01_3.log"),
            "iteration_index": 118,
            "observed_master_counts": OBSERVED_MASTER_COUNTS,
            "observed_dop_counts": OBSERVED_DOP_COUNTS,
        },
        "layouts": {
            case: summarize_case_layout(layout) for case, layout in layouts.items()
        },
        "case_order_by_round": [],
        "results": {case: {"measurements": []} for case in cases},
    }
    _atomic_write_json(output_path, artifact)
    print(f"Writing checkpointed result to {output_path}", flush=True)

    try:
        engine = _build_engine(args)
        artifact["resolved_ls_manifest"] = resolved_manifest(
            engine.config,
            args.ls_max_num_ooe,
            expected_loop_count=args.loop_count,
        )
        artifact["status"] = "running"
        _atomic_write_json(output_path, artifact)

        rng = random.Random(args.seed)
        for round_idx in range(args.rounds):
            order = list(cases)
            rng.shuffle(order)
            artifact["case_order_by_round"].append(order)
            _atomic_write_json(output_path, artifact)
            for case in order:
                measurements = _run_case_block(
                    engine,
                    layouts[case],
                    round_idx=round_idx,
                    warmup_repeats=args.warmup_repeats,
                    measured_repeats=args.repeats_per_round,
                    loop_count=args.loop_count,
                    timeout_sec=args.timeout_sec,
                )
                artifact["results"][case]["measurements"].extend(measurements)
                artifact["results"][case]["summary"] = _summarize_measurements(
                    artifact["results"][case]["measurements"]
                )
                _atomic_write_json(output_path, artifact)

        if tuple(cases) == CASE_NAMES:
            median_values = {
                case: float(
                    artifact["results"][case]["summary"][
                        "gpu_critical_path_per_loop_ms"
                    ]["median"]
                )
                for case in CASE_NAMES
            }
            artifact["effects_from_all_measurement_medians"] = calculate_2x2_effects(
                median_values
            )
            artifact["paired_round_effects"] = _round_effects(artifact["results"])
        artifact["status"] = "success"
        artifact["finished_at"] = _utc_now()
        _atomic_write_json(output_path, artifact)
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["finished_at"] = _utc_now()
        artifact["error"] = f"{type(error).__name__}: {error}"
        _atomic_write_json(output_path, artifact)
        raise

    print(f"Static 2x2 benchmark completed: {output_path}", flush=True)
    for case in cases:
        summary = artifact["results"][case]["summary"]["gpu_critical_path_per_loop_ms"]
        print(
            f"{case}: median={summary['median']:.4f} ms "
            f"mean={summary['mean']:.4f} ms p95={summary['p95']:.4f} ms",
            flush=True,
        )
    if "effects_from_all_measurement_medians" in artifact:
        effects = artifact["effects_from_all_measurement_medians"]
        print(
            "effects: "
            f"CP@balanced={effects['cp_effect_balanced_ms_T10_minus_T00']:+.4f} ms, "
            f"skew@D1={effects['imbalance_effect_dop1_ms_T01_minus_T00']:+.4f} ms, "
            f"interaction={effects['interaction_ms']:+.4f} ms, "
            f"T11/T00={effects['current_over_ideal_T11_div_T00']:.4f}x",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract and replay one metadata-exact LoongServe-style Decode iteration.

The serving log records enough information to reconstruct every active
sequence's committed KV-token placement:

* admission records provide the initial placement ranks;
* every earlier Decode iteration provides the master rank and chunk length;
* the target iteration provides exact per-group/per-rank KV token and block
  aggregates, plus the pending output reservations.

For an initially multi-rank request, its prompt split is inferred from the
target group's residual after all single-rank prompts and historical Decode
chunks have been removed.  Extraction fails unless this split is uniquely
identifiable and every recorded token, block, DoP, master-batch, and pending
reservation aggregate matches exactly.

The replay bypasses the scheduler, regenerates physical KV block IDs, installs
the reconstructed metadata in SPStateManager, and invokes the normal DLSlime
transport and ModelRunner workers.  Thus the replay is exact for the
scheduler-visible Decode shapes, but not for token values or physical block
addresses, neither of which is present in the historical log.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    from bench_ls_decode_static_layout import (
        ATTENTION_DP,
        ATTENTION_SP,
        BLOCK_SIZE,
        StaticCaseLayout,
        StaticSequenceLayout,
        _atomic_write_json,
        _build_engine,
        _run_case_block,
        _summarize_measurements,
        _utc_now,
        clear_ray_proxy_env,
        resolved_manifest,
        summarize_case_layout,
    )
except ModuleNotFoundError:  # pragma: no cover - module-style invocation
    from scripts.bench_ls_decode_static_layout import (
        ATTENTION_DP,
        ATTENTION_SP,
        BLOCK_SIZE,
        StaticCaseLayout,
        StaticSequenceLayout,
        _atomic_write_json,
        _build_engine,
        _run_case_block,
        _summarize_measurements,
        _utc_now,
        clear_ray_proxy_env,
        resolved_manifest,
        summarize_case_layout,
    )


DEFAULT_SOURCE_LOG = Path("docs-dev/2026-07-23/ls_style_loop16_dp2sp8_r20_diag01_3.log")
DEFAULT_COMPLETION_JSONL = Path(
    "docs-dev/2026-07-23/ls_style_loop16_dp2sp8_r20_diag01_3.jsonl"
)
DEFAULT_TARGET_ITERATION = 118
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def _read_structured_log(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = ANSI_ESCAPE_RE.sub("", raw_line)
            marker = " - {'mode': '"
            if marker not in line:
                continue
            try:
                payload = ast.literal_eval(line.split(" - ", 1)[1])
            except (SyntaxError, ValueError):
                continue
            if not isinstance(payload, dict) or "mode" not in payload:
                continue
            timestamp_match = TIMESTAMP_RE.search(line)
            events.append(
                {
                    "line_number": line_number,
                    "timestamp": (
                        timestamp_match.group(1) if timestamp_match else None
                    ),
                    "payload": payload,
                }
            )
    return events


def _load_prompt_lengths(path: Path) -> dict[int, int]:
    prompt_lengths: dict[int, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            seq_id = int(row["seq_id"])
            prompt_len = int(row["prompt_len"])
            if seq_id in prompt_lengths:
                raise ValueError(
                    f"duplicate seq_id={seq_id} in {path} at line {line_number}"
                )
            if prompt_len <= 0:
                raise ValueError(f"seq_id={seq_id} has invalid prompt length")
            prompt_lengths[seq_id] = prompt_len
    return prompt_lengths


def _ceil_blocks(tokens: int, block_size: int = BLOCK_SIZE) -> int:
    return (tokens + block_size - 1) // block_size if tokens > 0 else 0


def _rank_sum(
    sequence_ids: Sequence[int],
    placements: dict[int, list[int]],
) -> list[int]:
    return [
        sum(placements[seq_id][rank] for seq_id in sequence_ids)
        for rank in range(ATTENTION_SP)
    ]


def _rank_blocks(
    sequence_ids: Sequence[int],
    placements: dict[int, list[int]],
) -> list[int]:
    return [
        sum(_ceil_blocks(placements[seq_id][rank]) for seq_id in sequence_ids)
        for rank in range(ATTENTION_SP)
    ]


def _pending_blocks(
    sequence_ids: Sequence[int],
    masters: Sequence[int],
    placements: dict[int, list[int]],
    loop_count: int,
) -> list[int]:
    pending = [0] * ATTENTION_SP
    for seq_id, master in zip(sequence_ids, masters, strict=True):
        committed = placements[seq_id][master]
        # Production first installs one pending input, then reserves the whole
        # output chunk.  Telemetry reports table blocks minus committed blocks.
        pending[master] += _ceil_blocks(committed + 1 + loop_count) - _ceil_blocks(
            committed
        )
    return pending


def _flatten_target_sequences(
    target: dict[str, Any],
) -> tuple[list[int], dict[int, tuple[int, int, int]]]:
    ordered: list[int] = []
    membership: dict[int, tuple[int, int, int]] = {}
    dp_local_index = [0] * ATTENTION_DP
    for group_index, sequence_ids in enumerate(target["iteration_sequence_ids"]):
        dp_idx = int(target["group_dp_indices"][group_index])
        for group_sequence_index, raw_seq_id in enumerate(sequence_ids):
            seq_id = int(raw_seq_id)
            if seq_id in membership:
                raise ValueError(f"target iteration contains duplicate seq_id={seq_id}")
            membership[seq_id] = (
                dp_idx,
                dp_local_index[dp_idx],
                group_sequence_index,
            )
            dp_local_index[dp_idx] += 1
            ordered.append(seq_id)
    return ordered, membership


def extract_snapshot(
    source_log: Path,
    completion_jsonl: Path,
    *,
    target_iteration_index: int,
) -> dict[str, Any]:
    """Reconstruct and validate one target iteration from historical artifacts."""
    if target_iteration_index < 0:
        raise ValueError("target_iteration_index must be non-negative")

    events = _read_structured_log(source_log)
    prompt_lengths = _load_prompt_lengths(completion_jsonl)
    iteration_index = -1
    target_event: dict[str, Any] | None = None
    prior_iterations: list[dict[str, Any]] = []
    admissions: dict[int, dict[str, Any]] = {}
    consolidations_before_target: list[dict[str, Any]] = []
    decode_summary: dict[str, Any] | None = None

    for event in events:
        payload = event["payload"]
        mode = payload.get("mode")
        if target_event is None:
            if mode == "ls_decode_admission":
                for record in payload["records"]:
                    admissions[int(record["sequence_id"])] = dict(record)
            elif mode == "ls_kv_consolidation":
                consolidations_before_target.append(event)
            elif mode == "ls_decode_iteration":
                iteration_index += 1
                if iteration_index == target_iteration_index:
                    target_event = event
                else:
                    prior_iterations.append(payload)
        elif mode == "decode":
            decode_summary = payload
            break

    if target_event is None:
        raise ValueError(
            f"iteration {target_iteration_index} was not found in {source_log}"
        )
    if decode_summary is None:
        raise ValueError("target iteration has no following decode summary")
    if consolidations_before_target:
        lines = [item["line_number"] for item in consolidations_before_target]
        raise ValueError(
            "historical KV consolidation before target is not supported; "
            f"events at lines {lines}"
        )

    target = target_event["payload"]
    loop_count = int(target["execution_loop_count"])
    ordered_ids, membership = _flatten_target_sequences(target)
    active_ids = set(ordered_ids)
    missing_prompts = sorted(active_ids - prompt_lengths.keys())
    missing_admissions = sorted(active_ids - admissions.keys())
    if missing_prompts:
        raise ValueError(f"missing prompt lengths for {missing_prompts[:8]}")
    if missing_admissions:
        raise ValueError(f"missing admission records for {missing_admissions[:8]}")

    history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for iteration in prior_iterations:
        chunk_len = int(iteration["execution_loop_count"])
        for sequence_ids, masters in zip(
            iteration["iteration_sequence_ids"],
            iteration["iteration_master_assignments"],
            strict=True,
        ):
            if len(sequence_ids) != len(masters):
                raise ValueError("historical sequence/master list length mismatch")
            for raw_seq_id, raw_master in zip(sequence_ids, masters, strict=True):
                seq_id = int(raw_seq_id)
                if seq_id in active_ids:
                    history[seq_id].append((int(raw_master), chunk_len))

    initial: dict[int, list[int]] = {}
    for seq_id in ordered_ids:
        record = admissions[seq_id]
        planned_dop = int(record["planned_kv_dop"])
        planned_ranks = [int(rank) for rank in record["planned_kv_ranks"]]
        if planned_dop != len(planned_ranks) or len(set(planned_ranks)) != planned_dop:
            raise ValueError(f"invalid admission placement for seq_id={seq_id}")
        if int(record["dp_idx"]) != membership[seq_id][0]:
            raise ValueError(f"DP mismatch for seq_id={seq_id}")
        if planned_dop == 1:
            placement = [0] * ATTENTION_SP
            placement[planned_ranks[0]] = prompt_lengths[seq_id]
            initial[seq_id] = placement

    inferred_multi_rank: list[dict[str, Any]] = []
    for group_index, raw_sequence_ids in enumerate(target["iteration_sequence_ids"]):
        sequence_ids = [int(seq_id) for seq_id in raw_sequence_ids]
        residual = [int(value) for value in target["group_used_kv_tokens"][group_index]]

        for seq_id in sequence_ids:
            for master, chunk_len in history[seq_id]:
                residual[master] -= chunk_len
            if seq_id in initial:
                for rank, tokens in enumerate(initial[seq_id]):
                    residual[rank] -= tokens

        unknown = [seq_id for seq_id in sequence_ids if seq_id not in initial]
        if not unknown:
            if any(residual):
                raise ValueError(
                    f"group {target['group_ids'][group_index]} has unexplained "
                    f"initial tokens {residual}"
                )
            continue
        if len(unknown) != 1:
            raise ValueError(
                f"group {target['group_ids'][group_index]} has {len(unknown)} "
                "initially multi-rank requests; prompt splits are not unique"
            )

        seq_id = unknown[0]
        if any(tokens < 0 for tokens in residual):
            raise ValueError(f"negative inferred prompt split for seq_id={seq_id}")
        planned_ranks = {int(rank) for rank in admissions[seq_id]["planned_kv_ranks"]}
        inferred_ranks = {rank for rank, tokens in enumerate(residual) if tokens > 0}
        if inferred_ranks != planned_ranks:
            raise ValueError(
                f"seq_id={seq_id} inferred ranks {sorted(inferred_ranks)} "
                f"!= planned ranks {sorted(planned_ranks)}"
            )
        if sum(residual) != prompt_lengths[seq_id]:
            raise ValueError(
                f"seq_id={seq_id} inferred prompt split sums to {sum(residual)}, "
                f"expected {prompt_lengths[seq_id]}"
            )
        initial[seq_id] = residual
        inferred_multi_rank.append(
            {
                "sequence_id": seq_id,
                "prompt_len": prompt_lengths[seq_id],
                "planned_ranks": sorted(planned_ranks),
                "inferred_prompt_tokens_by_sp": residual,
            }
        )

    committed: dict[int, list[int]] = {}
    for seq_id in ordered_ids:
        placement = list(initial[seq_id])
        for master, chunk_len in history[seq_id]:
            placement[master] += chunk_len
        if sum(placement) != prompt_lengths[seq_id] + sum(
            chunk_len for _, chunk_len in history[seq_id]
        ):
            raise AssertionError("internal committed-token accounting failure")
        committed[seq_id] = placement

    group_checks: list[dict[str, Any]] = []
    target_masters: dict[int, int] = {}
    all_groups_exact = True
    for group_index, raw_sequence_ids in enumerate(target["iteration_sequence_ids"]):
        sequence_ids = [int(seq_id) for seq_id in raw_sequence_ids]
        masters = [
            int(master)
            for master in target["iteration_master_assignments"][group_index]
        ]
        if len(sequence_ids) != len(masters):
            raise ValueError("target sequence/master list length mismatch")
        target_masters.update(zip(sequence_ids, masters, strict=True))

        observed_tokens = [
            int(value) for value in target["group_used_kv_tokens"][group_index]
        ]
        observed_blocks = [
            int(value) for value in target["group_used_kv_blocks"][group_index]
        ]
        observed_pending = [
            int(value)
            for value in target["pending_append_blocks_per_master"][group_index]
        ]
        reconstructed_tokens = _rank_sum(sequence_ids, committed)
        reconstructed_blocks = _rank_blocks(sequence_ids, committed)
        reconstructed_pending = _pending_blocks(
            sequence_ids, masters, committed, loop_count
        )
        reconstructed_master_batches = [
            masters.count(rank) for rank in target["master_ranks"][group_index]
        ]
        observed_master_batches = [
            int(value) for value in target["master_batch_sizes"][group_index]
        ]
        participant_ranks = {
            rank
            for seq_id in sequence_ids
            for rank, tokens in enumerate(committed[seq_id])
            if tokens > 0
        }
        exact = (
            reconstructed_tokens == observed_tokens
            and reconstructed_blocks == observed_blocks
            and reconstructed_pending == observed_pending
            and reconstructed_master_batches == observed_master_batches
            and len(participant_ranks) == int(target["kv_dops"][group_index])
        )
        all_groups_exact = all_groups_exact and exact
        group_checks.append(
            {
                "group_id": int(target["group_ids"][group_index]),
                "dp_idx": int(target["group_dp_indices"][group_index]),
                "batch_size": len(sequence_ids),
                "exact": exact,
                "observed_kv_tokens_by_sp": observed_tokens,
                "reconstructed_kv_tokens_by_sp": reconstructed_tokens,
                "observed_kv_blocks_by_sp": observed_blocks,
                "reconstructed_kv_blocks_by_sp": reconstructed_blocks,
                "observed_pending_blocks_by_sp": observed_pending,
                "reconstructed_pending_blocks_by_sp": reconstructed_pending,
                "observed_master_batch_sizes": observed_master_batches,
                "reconstructed_master_batch_sizes": reconstructed_master_batches,
            }
        )

    sequences: list[dict[str, Any]] = []
    dp_local_index = [0] * ATTENTION_DP
    for group_index, raw_sequence_ids in enumerate(target["iteration_sequence_ids"]):
        dp_idx = int(target["group_dp_indices"][group_index])
        for group_sequence_index, raw_seq_id in enumerate(raw_sequence_ids):
            seq_id = int(raw_seq_id)
            sequences.append(
                {
                    "source_sequence_id": seq_id,
                    "dp_idx": dp_idx,
                    "dp_local_index": dp_local_index[dp_idx],
                    "group_id": int(target["group_ids"][group_index]),
                    "group_sequence_index": group_sequence_index,
                    "master_sp_idx": target_masters[seq_id],
                    "prompt_len": prompt_lengths[seq_id],
                    "prior_decode_steps": len(history[seq_id]),
                    "prior_decode_tokens": sum(
                        chunk_len for _, chunk_len in history[seq_id]
                    ),
                    "admission_planned_ranks": [
                        int(rank) for rank in admissions[seq_id]["planned_kv_ranks"]
                    ],
                    "committed_tokens_by_sp": committed[seq_id],
                }
            )
            dp_local_index[dp_idx] += 1

    dop_histogram = Counter(
        sum(tokens > 0 for tokens in committed[seq_id]) for seq_id in ordered_ids
    )
    observed_dop_histogram = {
        int(dop): int(count)
        for dop, count in decode_summary["sp_size_hist_global"].items()
    }
    exact_dop = dict(sorted(dop_histogram.items())) == dict(
        sorted(observed_dop_histogram.items())
    )
    exact_batch = len(ordered_ids) == int(decode_summary["total_batch_size"])
    exact = all_groups_exact and exact_dop and exact_batch
    if not exact:
        raise ValueError(
            "snapshot reconstruction did not match every target aggregate: "
            f"groups={all_groups_exact}, dop={exact_dop}, batch={exact_batch}"
        )

    return {
        "schema_version": 1,
        "kind": "ls_decode_metadata_exact_snapshot",
        "created_at": _utc_now(),
        "source": {
            "log": str(source_log),
            "completion_jsonl": str(completion_jsonl),
            "target_iteration_index_zero_based": target_iteration_index,
            "target_log_physical_line": target_event["line_number"],
            "target_timestamp": target_event["timestamp"],
        },
        "exactness": {
            "scheduler_visible_metadata_exact": True,
            "physical_kv_block_ids_replayed": False,
            "pending_token_values_replayed": False,
            "reason": (
                "historical logs contain exact placements and shapes, but not "
                "physical allocator IDs or sampled token values"
            ),
        },
        "target": {
            "loop_count": loop_count,
            "batch_size": len(ordered_ids),
            "group_count": len(target["group_ids"]),
            "model_runner_duration_ms": float(target["model_runner_duration_ms"]),
            "model_runner_per_loop_ms": (
                float(target["model_runner_duration_ms"]) / loop_count
            ),
            "step_itl_ms": float(target["step_itl_ms"]),
            "planning_latency_ms": float(target["planning_latency_ms"]),
            "summary_scheduler_overhead_ms": decode_summary["sch_ovhd"],
            "summary_post_scheduler_overhead_ms": decode_summary["post_sch_ovhd"],
            "summary_max_kv_util_pct": decode_summary["max_kv_util_pct"],
            "summary_min_free_blocks": int(decode_summary["min_free_blocks"]),
            "dop_histogram": {
                str(dop): count for dop, count in sorted(dop_histogram.items())
            },
        },
        "inferred_initial_multi_rank_requests": inferred_multi_rank,
        "validation": {
            "exact": exact,
            "all_groups_exact": all_groups_exact,
            "dop_histogram_exact": exact_dop,
            "batch_size_exact": exact_batch,
            "group_checks": group_checks,
        },
        "sequences": sequences,
    }


def snapshot_to_layout(snapshot: dict[str, Any]) -> StaticCaseLayout:
    if not snapshot.get("validation", {}).get("exact"):
        raise ValueError("refusing to replay a snapshot without exact validation")
    sequences = tuple(
        StaticSequenceLayout(
            dp_idx=int(item["dp_idx"]),
            local_index=int(item["dp_local_index"]),
            master_sp_idx=int(item["master_sp_idx"]),
            dispatched_tokens=tuple(
                int(tokens) for tokens in item["committed_tokens_by_sp"]
            ),
        )
        for item in snapshot["sequences"]
    )
    for dp_idx in range(ATTENTION_DP):
        local_indices = sorted(
            seq.local_index for seq in sequences if seq.dp_idx == dp_idx
        )
        if local_indices != list(range(len(local_indices))):
            raise ValueError(f"snapshot DP{dp_idx} local indices are not dense")
    return StaticCaseLayout(
        case="exact",
        cp_factor="historical_exact",
        master_factor="historical_exact",
        sequences=sequences,
    )


def _default_snapshot_path() -> Path:
    return Path("docs-dev/2026-07-23/ls_decode_iteration118_snapshot.json")


def _default_output_path() -> Path:
    return Path("docs-dev/2026-07-23/ls_decode_iteration118_exact_replay.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and replay one metadata-exact LS Decode iteration."
    )
    parser.add_argument("--source-log", type=Path, default=DEFAULT_SOURCE_LOG)
    parser.add_argument(
        "--completion-jsonl", type=Path, default=DEFAULT_COMPLETION_JSONL
    )
    parser.add_argument(
        "--target-iteration", type=int, default=DEFAULT_TARGET_ITERATION
    )
    parser.add_argument("--snapshot-json", type=Path, default=_default_snapshot_path())
    parser.add_argument("--output-json", type=Path, default=_default_output_path())
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Validate and write the snapshot without connecting to Ray.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats-per-round", type=int, default=5)
    parser.add_argument("--timeout-sec", type=float, default=300.0)

    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--ray-address", default="10.102.206.14:7789")
    parser.add_argument("--master-address", default="10.102.206.14:27789")
    parser.add_argument("--loop-count", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1_024_000)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=141.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument(
        "--cuda-graph-mode", choices=("full", "piecewise"), default="full"
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ls-max-num-ooe", type=int, default=8)
    parser.add_argument("--ls-batch-per-master", type=int, default=64)
    parser.add_argument("--verbose-nanodeploy-logs", action="store_true")
    args = parser.parse_args(argv)

    if args.rounds <= 0 or args.repeats_per_round <= 0:
        parser.error("rounds and repeats must be positive")
    if args.warmup_repeats < 0:
        parser.error("warmup repeats must be non-negative")
    if args.timeout_sec <= 0:
        parser.error("timeout must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = extract_snapshot(
        args.source_log,
        args.completion_jsonl,
        target_iteration_index=args.target_iteration,
    )
    if int(snapshot["target"]["loop_count"]) != args.loop_count:
        raise ValueError(
            f"snapshot loop_count={snapshot['target']['loop_count']} "
            f"!= requested loop_count={args.loop_count}"
        )
    layout = snapshot_to_layout(snapshot)
    layout_summary = summarize_case_layout(layout)
    _atomic_write_json(args.snapshot_json, snapshot)

    print(
        f"Exact snapshot validated: iteration={args.target_iteration}, "
        f"batch={snapshot['target']['batch_size']}, "
        f"DoP={snapshot['target']['dop_histogram']}, "
        f"source model-runner/loop="
        f"{snapshot['target']['model_runner_per_loop_ms']:.4f} ms",
        flush=True,
    )
    print(f"Snapshot written to {args.snapshot_json}", flush=True)
    if args.extract_only:
        return 0

    cleared_proxy_env = clear_ray_proxy_env()
    os.environ["NANODEPLOY_LOG_MODEL_FORWARD_TIMING"] = "1"
    if args.verbose_nanodeploy_logs:
        os.environ["NANODEPLOY_LOG_LEVEL"] = "DEBUG"

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ls_decode_metadata_exact_replay",
        "status": "initializing",
        "started_at": _utc_now(),
        "command": sys.argv,
        "cleared_proxy_env": cleared_proxy_env,
        "snapshot_json": str(args.snapshot_json),
        "source_target": snapshot["target"],
        "exactness": snapshot["exactness"],
        "layout": layout_summary,
        "configuration": {
            "attention_dp": ATTENTION_DP,
            "attention_sp": ATTENTION_SP,
            "loop_count": args.loop_count,
            "rounds": args.rounds,
            "warmup_repeats": args.warmup_repeats,
            "repeats_per_round": args.repeats_per_round,
            "ray_address": args.ray_address,
            "master_address": args.master_address,
            "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cuda_graph_mode": args.cuda_graph_mode,
            "scheduler_in_timed_path": False,
            "transport": "DLSlime sequence metadata endpoint",
        },
        "measurements": [],
    }
    _atomic_write_json(args.output_json, artifact)
    print(f"Writing checkpointed replay result to {args.output_json}", flush=True)

    try:
        engine = _build_engine(args)
        artifact["resolved_ls_manifest"] = resolved_manifest(
            engine.config,
            args.ls_max_num_ooe,
            expected_loop_count=args.loop_count,
        )
        artifact["status"] = "running"
        _atomic_write_json(args.output_json, artifact)

        for round_idx in range(args.rounds):
            measurements = _run_case_block(
                engine,
                layout,
                round_idx=round_idx,
                warmup_repeats=args.warmup_repeats,
                measured_repeats=args.repeats_per_round,
                loop_count=args.loop_count,
                timeout_sec=args.timeout_sec,
            )
            artifact["measurements"].extend(measurements)
            artifact["summary"] = _summarize_measurements(artifact["measurements"])
            _atomic_write_json(args.output_json, artifact)

        wall_summary = artifact["summary"]["wall_per_loop_ms"]
        gpu_summary = artifact["summary"]["gpu_critical_path_per_loop_ms"]
        source_ms = float(snapshot["target"]["model_runner_per_loop_ms"])
        artifact["comparison_to_source"] = {
            "source_model_runner_per_loop_ms": source_ms,
            "replay_wall_per_loop_median_ms": wall_summary["median"],
            "replay_wall_minus_source_ms": wall_summary["median"] - source_ms,
            "replay_wall_over_source": wall_summary["median"] / source_ms,
            "replay_gpu_critical_path_median_ms": gpu_summary["median"],
        }
        artifact["status"] = "success"
        artifact["finished_at"] = _utc_now()
        _atomic_write_json(args.output_json, artifact)
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["finished_at"] = _utc_now()
        artifact["error"] = f"{type(error).__name__}: {error}"
        _atomic_write_json(args.output_json, artifact)
        raise

    comparison = artifact["comparison_to_source"]
    print(
        "Exact replay completed: "
        f"wall median={comparison['replay_wall_per_loop_median_ms']:.4f} ms, "
        f"source={comparison['source_model_runner_per_loop_ms']:.4f} ms, "
        f"ratio={comparison['replay_wall_over_source']:.4f}x, "
        f"GPU critical median="
        f"{comparison['replay_gpu_critical_path_median_ms']:.4f} ms",
        flush=True,
    )
    print(f"Result written to {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

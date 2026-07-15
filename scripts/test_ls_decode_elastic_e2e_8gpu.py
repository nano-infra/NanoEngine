#!/usr/bin/env python3
"""Deterministic 8-GPU LS Decode elastic scale-down/scale-up E2E.

The middle phase is arrival-idle rather than fully request-idle: one long-tail
anchor keeps Decode iterations running so the scheduler can evacuate its live
KV and release ranks.  With no live request there is no group left to scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
TELEMETRY_MODES = {
    "ls_decode_admission",
    "ls_decode_iteration",
    "ls_kv_consolidation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run burst -> arrival-idle scale-down -> burst scale-up on one "
            "DP1 x SP8 / EP8 NanoDeploy engine."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--ray-address", default="10.102.243.60:8776")
    parser.add_argument("--master-address", default="10.102.243.60:29776")
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--short-max-tokens", type=int, default=8)
    parser.add_argument("--anchor-max-tokens", type=int, default=32)
    parser.add_argument("--burst-size", type=int, default=64)
    parser.add_argument("--low-load-decode-steps", type=int, default=4)
    parser.add_argument("--max-total-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ls-batch-per-master", type=int, default=8)
    parser.add_argument("--ls-kv-consolidation-stable-steps", type=int, default=1)
    parser.add_argument("--ls-kv-consolidation-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--ls-kv-consolidation-check-interval-steps", type=int, default=1
    )
    parser.add_argument(
        "--ls-kv-consolidation-max-source-blocks-per-event", type=int, default=16
    )
    parser.add_argument(
        "--ls-kv-consolidation-migration-chunk-tokens", type=int, default=64
    )
    parser.add_argument("--kvcache-block-size", type=int, default=64)
    parser.add_argument("--segment-size", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--real-weight", dest="dummy_weight", action="store_false")
    parser.set_defaults(dummy_weight=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.prompt_len <= 0:
        parser.error("--prompt-len must be > 0")
    if args.short_max_tokens < 2:
        parser.error("--short-max-tokens must be >= 2")
    if args.anchor_max_tokens <= 2 * args.short_max_tokens + args.low_load_decode_steps:
        parser.error(
            "--anchor-max-tokens must survive both bursts and the low-load window"
        )
    if args.burst_size != 8 * args.ls_batch_per_master:
        parser.error(
            "this deterministic test requires burst-size == "
            "8 * ls-batch-per-master"
        )
    if args.low_load_decode_steps <= 0 or args.max_total_steps <= 0:
        parser.error("step limits must be > 0")
    if args.ls_kv_consolidation_max_source_blocks_per_event <= 0:
        parser.error("KV consolidation execute requires a positive block budget")
    if args.ls_kv_consolidation_migration_chunk_tokens <= 0:
        parser.error("KV consolidation execute requires positive scratch tokens")
    return args


def clear_http_proxy_env() -> list[str]:
    cleared: list[str] = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            os.environ.pop(key, None)
            cleared.append(key)
    return cleared


class TelemetryCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.msg
        if not isinstance(message, dict) or message.get("mode") not in TELEMETRY_MODES:
            return
        # Detach pybind containers and other mutable values from the log record.
        self.records.append(json.loads(json.dumps(message, default=str)))


def make_sequence(
    rng: np.random.Generator,
    prompt_len: int,
    max_tokens: int,
    temperature: float,
) -> Any:
    from nanodeploy import SamplingParams
    from nanodeploy.engine.sequence import Sequence

    prompt = rng.integers(0, 10001, size=prompt_len, dtype=np.int64).tolist()
    return Sequence(
        prompt,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("docs-dev") / f"ls_decode_elastic_e2e_8gpu_{timestamp}.json"


def group_snapshot(engine: Any) -> dict[str, list[int]]:
    return {
        str(group_id): list(engine.scheduler.get_ls_group_allocated_ranks(group_id))
        for group_id in engine.scheduler.get_ls_group_ids()
    }


def flatten_iterations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        if record.get("mode") != "ls_decode_iteration":
            continue
        group_ids = record.get("group_ids", [])
        for idx, group_id in enumerate(group_ids):
            flattened.append(
                {
                    "group_id": group_id,
                    "real_batch_size": record["real_batch_sizes"][idx],
                    "master_dop": record["master_dops"][idx],
                    "kv_dop": record["kv_dops"][idx],
                    "master_ranks": record["master_ranks"][idx],
                    "master_batch_sizes": record["master_batch_sizes"][idx],
                    "rank_allocation": record["rank_allocations"][idx],
                    "scale_reason": record["scale_reasons"][idx],
                    "new_master_ranks": record["new_master_ranks"][idx],
                    "historical_kv_migration_bytes": record[
                        "historical_kv_migration_bytes"
                    ][idx],
                }
            )
    return flattened


def find_iteration(
    records: list[dict[str, Any]],
    *,
    group_id: int,
    real_batch_size: int,
    master_dop: int,
    kv_dop: int,
    allocation_size: int,
) -> dict[str, Any] | None:
    for item in flatten_iterations(records):
        if (
            item["group_id"] == group_id
            and item["real_batch_size"] == real_batch_size
            and item["master_dop"] == master_dop
            and item["kv_dop"] == kv_dop
            and len(item["rank_allocation"]) == allocation_size
        ):
            return item
    return None


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cleared_proxy_keys = clear_http_proxy_env()
    output_path = args.output_json or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "running",
        "output_json": str(output_path),
        "cleared_proxy_env_keys": cleared_proxy_keys,
        "config": vars(args) | {"output_json": str(output_path)},
        "steps": [],
        "telemetry": [],
        "checks": {},
    }

    from nanodeploy import LLM
    from nanodeploy.logging import get_logger

    logger = get_logger()
    collector = TelemetryCollector()
    logger.addHandler(collector)
    engine: Any | None = None
    completed: dict[str, int] = {}
    step_idx = 0
    rng = np.random.default_rng(args.seed)

    def checkpoint() -> None:
        result["telemetry"] = collector.records
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def step(stage: str) -> list[dict[str, Any]]:
        nonlocal step_idx
        if engine is None:
            raise RuntimeError("engine has not been initialized")
        if step_idx >= args.max_total_steps:
            raise TimeoutError(f"exceeded {args.max_total_steps} engine steps")
        telemetry_start = len(collector.records)
        begin = time.perf_counter()
        outputs, num_tokens, batch_size, sch_ms, post_sch_ms = engine.step()
        duration_ms = (time.perf_counter() - begin) * 1000.0
        for seq_id, token_ids in outputs:
            completed[str(seq_id)] = len(token_ids)
        new_records = collector.records[telemetry_start:]
        result["steps"].append(
            {
                "step_idx": step_idx,
                "stage": stage,
                "engine_phase": (
                    "decode"
                    if num_tokens < 0
                    else "maintenance"
                    if num_tokens == 0
                    else "dummy_prefill"
                ),
                "duration_ms": duration_ms,
                "num_tokens_returned": num_tokens,
                "batch_size_returned": batch_size,
                "output_sequence_ids": [str(seq_id) for seq_id, _ in outputs],
                "scheduler_latency_ms": sch_ms,
                "post_scheduler_latency_ms": post_sch_ms,
                "group_allocations": group_snapshot(engine),
                "telemetry_modes": [record["mode"] for record in new_records],
            }
        )
        step_idx += 1
        checkpoint()
        return new_records

    def require(name: str, condition: bool, detail: Any) -> None:
        result["checks"][name] = {"passed": bool(condition), "detail": detail}
        checkpoint()
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    try:
        max_model_len = args.prompt_len + args.anchor_max_tokens + 16
        max_num_batched_tokens = args.burst_size * max_model_len + 1024
        engine = LLM(
            args.model_path,
            enforce_eager=args.enforce_eager,
            cuda_graph_mode="full",
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            master_address=args.master_address,
            ray_address=args.ray_address,
            mode="decode",
            dummy_prefill=True,
            dummy_weight=args.dummy_weight,
            perfect_eplb=True,
            attention_dp=1,
            attention_sp=8,
            attention_tp=1,
            ffn_dp=1,
            ffn_ep=8,
            ffn_tp=1,
            max_num_seqs=args.burst_size,
            max_num_recv_seqs=2 * args.burst_size,
            max_num_batched_tokens=max_num_batched_tokens,
            loop_count=1,
            routing_strategy="RoundRobin",
            scheduler_mode="centralized",
            segment_size=args.segment_size,
            kvcache_block_size=args.kvcache_block_size,
            fixed_sp_size=0,
            sp_backend="hao_basic",
            use_dlslime_rpc=True,
            optimize_decode_block_table=True,
            enable_non_uniform_split=False,
            enable_ls_decode_core_scheduler=True,
            ls_decode_initial_kv_dop=8,
            ls_decode_batch_per_master=args.ls_batch_per_master,
            ls_decode_enable_memory_scale_up=True,
            ls_kv_consolidation_mode="execute",
            ls_kv_consolidation_candidate_util=0.50,
            ls_kv_consolidation_target_high_watermark=0.80,
            ls_kv_consolidation_stable_steps=(
                args.ls_kv_consolidation_stable_steps
            ),
            ls_kv_consolidation_cooldown_steps=(
                args.ls_kv_consolidation_cooldown_steps
            ),
            ls_kv_consolidation_check_interval_steps=(
                args.ls_kv_consolidation_check_interval_steps
            ),
            ls_kv_consolidation_max_source_blocks_per_event=(
                args.ls_kv_consolidation_max_source_blocks_per_event
            ),
            ls_kv_consolidation_migration_chunk_tokens=(
                args.ls_kv_consolidation_migration_chunk_tokens
            ),
        )

        anchor = make_sequence(
            rng, args.prompt_len, args.anchor_max_tokens, args.temperature
        )
        first_short = [
            make_sequence(
                rng, args.prompt_len, args.short_max_tokens, args.temperature
            )
            for _ in range(args.burst_size - 1)
        ]
        first_short_ids = {str(seq.seq_id) for seq in first_short}
        anchor_id = str(anchor.seq_id)
        result["workload"] = {
            "anchor_sequence_id": anchor_id,
            "first_burst_short_sequence_ids": sorted(first_short_ids),
            "second_burst_short_sequence_ids": [],
            "note": (
                "The low-load window has zero new arrivals and one live anchor; "
                "a fully idle engine has no live group to consolidate."
            ),
        }
        engine.add_request([anchor, *first_short])

        first_high: dict[str, Any] | None = None
        first_group_id: int | None = None
        while not first_short_ids.issubset(completed):
            step("burst_1")
            admissions = [
                record
                for record in collector.records
                if record.get("mode") == "ls_decode_admission"
            ]
            if admissions and first_group_id is None:
                first_group_id = int(admissions[0]["group_ids"][0])
            if first_group_id is not None:
                first_high = find_iteration(
                    collector.records,
                    group_id=first_group_id,
                    real_batch_size=args.burst_size,
                    master_dop=8,
                    kv_dop=8,
                    allocation_size=8,
                )

        require("first_group_admitted", first_group_id is not None, first_group_id)
        assert first_group_id is not None
        require("first_burst_reached_8_masters", first_high is not None, first_high)
        require(
            "first_burst_no_history_migration",
            first_high is not None
            and first_high["historical_kv_migration_bytes"] == 0,
            first_high,
        )

        first_consolidation_start = len(
            [r for r in collector.records if r.get("mode") == "ls_kv_consolidation"]
        )
        low_iteration: dict[str, Any] | None = None
        while low_iteration is None:
            step("arrival_idle_scale_down")
            low_iteration = find_iteration(
                collector.records,
                group_id=first_group_id,
                real_batch_size=1,
                master_dop=1,
                kv_dop=1,
                allocation_size=1,
            )

        first_consolidations = [
            record
            for record in collector.records
            if record.get("mode") == "ls_kv_consolidation"
        ][first_consolidation_start:]
        retained_sizes = [len(record["retained_ranks"]) for record in first_consolidations]
        require(
            "automatic_consolidation_8_to_1",
            retained_sizes[:7] == [7, 6, 5, 4, 3, 2, 1],
            retained_sizes,
        )
        require("low_load_reached_1_1_1", low_iteration is not None, low_iteration)

        low_decode_seen = 0
        while low_decode_seen < args.low_load_decode_steps:
            records = step("arrival_idle_steady")
            for item in flatten_iterations(records):
                if (
                    item["group_id"] == first_group_id
                    and item["real_batch_size"] == 1
                    and item["master_dop"] == 1
                    and item["kv_dop"] == 1
                ):
                    low_decode_seen += 1
        require(
            "low_load_window_completed",
            low_decode_seen >= args.low_load_decode_steps,
            low_decode_seen,
        )

        second_short = [
            make_sequence(
                rng, args.prompt_len, args.short_max_tokens, args.temperature
            )
            for _ in range(args.burst_size - 1)
        ]
        second_short_ids = {str(seq.seq_id) for seq in second_short}
        result["workload"]["second_burst_short_sequence_ids"] = sorted(
            second_short_ids
        )
        second_telemetry_start = len(collector.records)
        engine.add_request(second_short)

        second_high: dict[str, Any] | None = None
        second_admission: dict[str, Any] | None = None
        while second_high is None:
            step("burst_2_scale_up")
            second_records = collector.records[second_telemetry_start:]
            admissions = [
                record
                for record in second_records
                if record.get("mode") == "ls_decode_admission"
            ]
            if admissions:
                second_admission = admissions[0]
            second_high = find_iteration(
                second_records,
                group_id=first_group_id,
                real_batch_size=args.burst_size,
                master_dop=8,
                kv_dop=8,
                allocation_size=8,
            )

        require(
            "second_burst_merged_same_group",
            second_admission is not None
            and second_admission["group_ids"] == [first_group_id]
            and second_admission["admission_kinds"] == ["merge"],
            second_admission,
        )
        require("second_burst_reached_8_masters", second_high is not None, second_high)
        require(
            "second_burst_no_history_migration",
            second_high is not None
            and second_high["historical_kv_migration_bytes"] == 0,
            second_high,
        )

        while not engine.is_finished():
            step("drain")

        expected_ids = first_short_ids | second_short_ids | {anchor_id}
        require(
            "all_requests_completed",
            set(completed) == expected_ids,
            {
                "expected": len(expected_ids),
                "completed": len(completed),
                "missing": sorted(expected_ids - set(completed)),
                "unexpected": sorted(set(completed) - expected_ids),
            },
        )
        require(
            "all_output_lengths_match",
            completed.get(anchor_id) == args.anchor_max_tokens
            and all(completed.get(seq_id) == args.short_max_tokens for seq_id in first_short_ids | second_short_ids),
            completed,
        )
        result["completed_output_lengths"] = completed
        result["status"] = "passed"
        result["summary"] = {
            "total_steps": step_idx,
            "total_requests": len(completed),
            "maintenance_steps": sum(
                step_record["engine_phase"] == "maintenance"
                for step_record in result["steps"]
            ),
            "first_scale_down_retained_sizes": retained_sizes[:7],
            "first_group_id": first_group_id,
            "first_high": first_high,
            "low_load": low_iteration,
            "second_high": second_high,
        }
        checkpoint()
        print("E2E_RESULT " + json.dumps(result["summary"]), flush=True)
        print(f"E2E_ARTIFACT {output_path}", flush=True)
        return result, 0
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        checkpoint()
        print("E2E_FAILURE " + json.dumps(result["error"]), flush=True)
        print(f"E2E_ARTIFACT {output_path}", flush=True)
        return result, 1
    finally:
        logger.removeHandler(collector)


def main() -> int:
    _, exit_code = run(parse_args())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

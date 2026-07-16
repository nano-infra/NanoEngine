#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, TextIO

import numpy as np


PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Long-running LS-Decode-Core serving benchmark. The default "
            "configuration runs attention DP=1, SP=8, FFN EP=8 for 10 minutes; "
            "--attention-dp 2 uses the two-node DP=2, SP=8, FFN EP=16 topology."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--ray-address", default="10.102.243.60:8776")
    parser.add_argument("--master-address", default="10.102.243.60:29690")
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--drain-timeout-sec", type=float, default=300.0)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="1.0 uses Poisson arrivals; other values use a Gamma renewal process.",
    )
    parser.add_argument(
        "--max-generated-requests",
        type=int,
        default=0,
        help="Optional accepted-request cap; 0 keeps sending until duration-sec.",
    )
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=256,
        help=(
            "Cap outstanding requests to avoid unbounded backlog. Arrivals above "
            "this cap are counted as backpressure drops. 0 disables the cap."
        ),
    )
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument(
        "--attention-dp",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Number of independent SP8 attention domains; FFN EP is 8x this value.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=None)
    parser.add_argument("--kvcache-block-size", type=int, default=64)
    parser.add_argument("--segment-size", type=int, default=65536)
    parser.add_argument("--ls-initial-kv-dop", type=int, default=0)
    parser.add_argument("--ls-batch-per-master", type=int, default=8)
    parser.add_argument(
        "--ls-kv-consolidation-mode",
        choices=["off", "shadow", "execute"],
        default="off",
    )
    parser.add_argument("--ls-kv-consolidation-candidate-util", type=float, default=0.50)
    parser.add_argument(
        "--ls-kv-consolidation-target-high-watermark", type=float, default=0.80
    )
    parser.add_argument("--ls-kv-consolidation-stable-steps", type=int, default=32)
    parser.add_argument("--ls-kv-consolidation-cooldown-steps", type=int, default=64)
    parser.add_argument(
        "--ls-kv-consolidation-check-interval-steps", type=int, default=8
    )
    parser.add_argument(
        "--ls-kv-consolidation-max-source-blocks-per-event", type=int, default=0
    )
    parser.add_argument(
        "--ls-kv-consolidation-migration-chunk-tokens", type=int, default=0
    )
    parser.add_argument(
        "--disable-ls-memory-scale-up",
        action="store_true",
        help="Disable LS decode memory scale-up.",
    )
    parser.add_argument(
        "--routing-strategy",
        choices=["RoundRobin", "LeastBatch", "LeastCache", "VLLMLoadBalance"],
        default="RoundRobin",
    )
    parser.add_argument(
        "--cuda-graph-mode",
        choices=["full", "piecewise"],
        default="full",
    )
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.set_defaults(enforce_eager=True)
    parser.add_argument("--real-weight", dest="dummy_weight", action="store_false")
    parser.set_defaults(dummy_weight=True)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--warmup-prompt-len", type=int, default=512)
    parser.add_argument("--warmup-max-tokens", type=int, default=8)
    parser.add_argument("--steady-start-sec", type=float, default=60.0)
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--completion-jsonl",
        "--itl-log-path",
        dest="completion_jsonl",
        type=Path,
        default=None,
        help="Per-completed-request JSONL path. --itl-log-path is kept as an alias.",
    )
    args = parser.parse_args()

    if args.duration_sec <= 0:
        parser.error("--duration-sec must be > 0")
    if args.drain_timeout_sec < 0:
        parser.error("--drain-timeout-sec must be >= 0")
    if args.request_rate <= 0:
        parser.error("--request-rate must be > 0")
    if args.burstiness <= 0:
        parser.error("--burstiness must be > 0")
    if args.prompt_len <= 0:
        parser.error("--prompt-len must be > 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be > 0")
    if args.max_num_recv_seqs <= 0:
        parser.error("--max-num-recv-seqs must be > 0")
    if args.ls_batch_per_master <= 0:
        parser.error("--ls-batch-per-master must be > 0")
    if (
        args.ls_kv_consolidation_mode == "execute"
        and args.ls_kv_consolidation_migration_chunk_tokens <= 0
    ):
        parser.error(
            "--ls-kv-consolidation-mode execute requires "
            "--ls-kv-consolidation-migration-chunk-tokens > 0"
        )
    if (
        args.ls_kv_consolidation_mode == "execute"
        and args.ls_kv_consolidation_max_source_blocks_per_event <= 0
    ):
        parser.error(
            "--ls-kv-consolidation-mode execute requires "
            "--ls-kv-consolidation-max-source-blocks-per-event > 0"
        )
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be >= 0")
    return args


def clear_http_proxy_env() -> list[str]:
    cleared = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            cleared.append(key)
            os.environ.pop(key, None)
    return cleared


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(percentile(values, 90)),
        "p95": float(percentile(values, 95)),
        "p99": float(percentile(values, 99)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float):
        return None
    return as_float


def finite_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = maybe_float(record.get(key))
        if value is not None:
            values.append(value)
    return values


def sample_arrival_interval(rng: np.random.Generator, rate: float, burstiness: float) -> float:
    if burstiness == 1.0:
        return float(rng.exponential(1.0 / rate))
    shape = 1.0 / (burstiness**2)
    scale = burstiness**2 / rate
    return float(rng.gamma(shape, scale))


def make_sequence(
    rng: np.random.Generator,
    prompt_len: int,
    max_tokens: int,
    temperature: float,
) -> Sequence:
    from nanodeploy import SamplingParams
    from nanodeploy.engine.sequence import Sequence

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    prompt = rng.integers(0, 10001, size=prompt_len, dtype=np.int64).tolist()
    return Sequence(prompt, sampling_params=sampling_params)


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_gpus = args.attention_dp * 8
    stem = f"ls_decode_longrun_{num_gpus}gpu_{int(args.duration_sec)}s_{timestamp}"
    return Path("docs-dev") / f"{stem}.json", Path("docs-dev") / f"{stem}.jsonl"


def build_engine(args: argparse.Namespace) -> LLM:
    from nanodeploy import LLM

    max_model_len = args.max_model_len
    if max_model_len <= 0:
        max_model_len = args.prompt_len + args.max_tokens + 16

    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_num_batched_tokens = max(
            max_model_len,
            args.max_num_seqs * (args.prompt_len + args.max_tokens) + 1024,
        )

    return LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        cuda_graph_mode=args.cuda_graph_mode,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        master_address=args.master_address,
        ray_address=args.ray_address,
        mode="decode",
        dummy_prefill=True,
        dummy_weight=args.dummy_weight,
        perfect_eplb=True,
        attention_dp=args.attention_dp,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=args.attention_dp * 8,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=args.max_num_recv_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        loop_count=1,
        routing_strategy=args.routing_strategy,
        scheduler_mode="centralized",
        segment_size=args.segment_size,
        kvcache_block_size=args.kvcache_block_size,
        fixed_sp_size=0,
        sp_backend="hao_basic",
        use_dlslime_rpc=True,
        optimize_decode_block_table=True,
        enable_non_uniform_split=False,
        enable_ls_decode_core_scheduler=True,
        ls_decode_initial_kv_dop=args.ls_initial_kv_dop,
        ls_decode_batch_per_master=args.ls_batch_per_master,
        ls_decode_enable_memory_scale_up=not args.disable_ls_memory_scale_up,
        ls_kv_consolidation_mode=args.ls_kv_consolidation_mode,
        ls_kv_consolidation_candidate_util=(
            args.ls_kv_consolidation_candidate_util
        ),
        ls_kv_consolidation_target_high_watermark=(
            args.ls_kv_consolidation_target_high_watermark
        ),
        ls_kv_consolidation_stable_steps=args.ls_kv_consolidation_stable_steps,
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


def config_dict(args: argparse.Namespace, output_json: Path, completion_jsonl: Path) -> dict[str, Any]:
    max_model_len = args.max_model_len
    if max_model_len <= 0:
        max_model_len = args.prompt_len + args.max_tokens + 16
    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_num_batched_tokens = max(
            max_model_len,
            args.max_num_seqs * (args.prompt_len + args.max_tokens) + 1024,
        )
    return {
        "model_path": args.model_path,
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "duration_sec": args.duration_sec,
        "drain_timeout_sec": args.drain_timeout_sec,
        "request_rate": args.request_rate,
        "burstiness": args.burstiness,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "max_generated_requests": args.max_generated_requests,
        "max_inflight_requests": args.max_inflight_requests,
        "attention_dp": args.attention_dp,
        "attention_sp": 8,
        "attention_tp": 1,
        "ffn_ep": args.attention_dp * 8,
        "ffn_dp": 1,
        "ffn_tp": 1,
        "fixed_sp_size": 0,
        "sp_backend": "hao_basic",
        "enable_ls_decode_core_scheduler": True,
        "ls_decode_initial_kv_dop": args.ls_initial_kv_dop,
        "ls_decode_batch_per_master": args.ls_batch_per_master,
        "ls_decode_enable_memory_scale_up": not args.disable_ls_memory_scale_up,
        "ls_kv_consolidation_mode": args.ls_kv_consolidation_mode,
        "ls_kv_consolidation_candidate_util": (
            args.ls_kv_consolidation_candidate_util
        ),
        "ls_kv_consolidation_target_high_watermark": (
            args.ls_kv_consolidation_target_high_watermark
        ),
        "ls_kv_consolidation_stable_steps": (
            args.ls_kv_consolidation_stable_steps
        ),
        "ls_kv_consolidation_cooldown_steps": (
            args.ls_kv_consolidation_cooldown_steps
        ),
        "ls_kv_consolidation_check_interval_steps": (
            args.ls_kv_consolidation_check_interval_steps
        ),
        "ls_kv_consolidation_max_source_blocks_per_event": (
            args.ls_kv_consolidation_max_source_blocks_per_event
        ),
        "ls_kv_consolidation_migration_chunk_tokens": (
            args.ls_kv_consolidation_migration_chunk_tokens
        ),
        "loop_count": 1,
        "max_num_seqs": args.max_num_seqs,
        "max_num_recv_seqs": args.max_num_recv_seqs,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
        "kvcache_block_size": args.kvcache_block_size,
        "segment_size": args.segment_size,
        "routing_strategy": args.routing_strategy,
        "scheduler_mode": "centralized",
        "dummy_prefill": True,
        "dummy_weight": args.dummy_weight,
        "enforce_eager": args.enforce_eager,
        "cuda_graph_mode": args.cuda_graph_mode,
        "warmup_requests": args.warmup_requests,
        "warmup_prompt_len": args.warmup_prompt_len,
        "warmup_max_tokens": args.warmup_max_tokens,
        "steady_start_sec": args.steady_start_sec,
        "output_json": str(output_json),
        "completion_jsonl": str(completion_jsonl),
    }


def completion_record(seq: Sequence, completed_elapsed_s: float) -> dict[str, Any]:
    metric = seq.metric
    itl_samples = [
        float(value)
        for value in list(getattr(metric, "itl_samples", []) or [])
        if maybe_float(value) is not None
    ]
    return {
        "seq_id": str(seq.seq_id),
        "completed_elapsed_s": float(completed_elapsed_s),
        "prompt_len": int(getattr(metric, "num_prompt_tokens", 0) or 0),
        "output_len": int(getattr(metric, "num_generated_tokens", 0) or 0),
        "ttft_ms": maybe_float(getattr(metric, "ttft", None)),
        "e2e_latency_ms": maybe_float(getattr(metric, "e2e_latency", None)),
        "queueing_time_ms": maybe_float(getattr(metric, "queueing_time_ms", None)),
        "decode_queue_time_ms": maybe_float(
            getattr(metric, "decode_queue_time_ms", None)
        ),
        "avg_itl_ms": maybe_float(getattr(metric, "avg_itl", None)),
        "avg_tpot_with_queueing_ms": maybe_float(
            getattr(metric, "avg_tpot_with_queueing", None)
        ),
        "avg_itl_with_decode_queue_ms": maybe_float(
            getattr(metric, "avg_itl_with_decode_queue", None)
        ),
        "itl_samples_ms": itl_samples,
    }


def run_warmup(engine: LLM, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    if args.warmup_requests == 0:
        return {"requests": 0, "steps": 0, "duration_sec": 0.0}

    warmup_sampling_start = time.perf_counter()
    warmup_seqs = [
        make_sequence(
            rng,
            args.warmup_prompt_len,
            args.warmup_max_tokens,
            args.temperature,
        )
        for _ in range(args.warmup_requests)
    ]
    engine.add_request(warmup_seqs)

    steps = 0
    completed = 0
    decode_steps = 0
    prefill_steps = 0
    maintenance_steps = 0
    start = time.perf_counter()
    print(
        "WARMUP_START "
        + json.dumps(
            {
                "requests": args.warmup_requests,
                "prompt_len": args.warmup_prompt_len,
                "max_tokens": args.warmup_max_tokens,
                "sequence_generation_sec": start - warmup_sampling_start,
            }
        ),
        flush=True,
    )
    while not engine.is_finished():
        outputs, num_tokens, _, _, _ = engine.step()
        steps += 1
        completed += len(outputs)
        if num_tokens < 0:
            decode_steps += 1
        elif num_tokens == 0:
            maintenance_steps += 1
        else:
            prefill_steps += 1
    duration = time.perf_counter() - start
    summary = {
        "requests": args.warmup_requests,
        "completed": completed,
        "steps": steps,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "maintenance_steps": maintenance_steps,
        "duration_sec": duration,
    }
    print("WARMUP_SUMMARY " + json.dumps(summary), flush=True)
    return summary


def write_completion(
    record: dict[str, Any],
    completion_fh: TextIO | None,
) -> None:
    if completion_fh is None:
        return
    completion_fh.write(json.dumps(record) + "\n")
    completion_fh.flush()


def run_longrun(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cleared_proxy_keys = clear_http_proxy_env()
    if cleared_proxy_keys:
        print(
            "CLEARED_HTTP_PROXY_ENV "
            + json.dumps({"keys": cleared_proxy_keys}),
            flush=True,
        )

    output_json, completion_jsonl = default_output_paths(args)
    if args.output_json is not None:
        output_json = args.output_json
    if args.completion_jsonl is not None:
        completion_jsonl = args.completion_jsonl
    output_json.parent.mkdir(parents=True, exist_ok=True)
    completion_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    engine = build_engine(args)
    warmup_summary = run_warmup(engine, args, rng)

    config = config_dict(args, output_json, completion_jsonl)
    print("RUN_CONFIG " + json.dumps(config, indent=2), flush=True)

    seq_map: dict[str, Sequence] = {}
    completion_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    decode_itls: list[float] = []
    steady_decode_itls: list[float] = []
    pending_maintenance_ms = 0.0

    requests_sent = 0
    requests_dropped_backpressure = 0
    max_inflight_observed = 0
    step_idx = 0
    next_arrival_s = 0.0
    drain_started_s: float | None = None
    drain_completed = False
    start = time.perf_counter()
    last_progress = start

    completion_fh: TextIO | None = completion_jsonl.open("w", encoding="utf-8")
    try:
        while True:
            now = time.perf_counter()
            elapsed_s = now - start
            can_send = elapsed_s < args.duration_sec and (
                args.max_generated_requests <= 0
                or requests_sent < args.max_generated_requests
            )

            while can_send and elapsed_s >= next_arrival_s:
                if (
                    args.max_inflight_requests > 0
                    and len(seq_map) >= args.max_inflight_requests
                ):
                    requests_dropped_backpressure += 1
                else:
                    seq = make_sequence(
                        rng,
                        args.prompt_len,
                        args.max_tokens,
                        args.temperature,
                    )
                    engine.add_request(seq)
                    seq_map[str(seq.seq_id)] = seq
                    requests_sent += 1
                    max_inflight_observed = max(max_inflight_observed, len(seq_map))

                next_arrival_s += sample_arrival_interval(
                    rng, args.request_rate, args.burstiness
                )
                can_send = elapsed_s < args.duration_sec and (
                    args.max_generated_requests <= 0
                    or requests_sent < args.max_generated_requests
                )

            if not can_send and drain_started_s is None:
                drain_started_s = elapsed_s
                print(
                    "DRAIN_START "
                    + json.dumps(
                        {
                            "elapsed_s": elapsed_s,
                            "accepted_requests": requests_sent,
                            "inflight_requests": len(seq_map),
                            "backpressure_drops": requests_dropped_backpressure,
                        }
                    ),
                    flush=True,
                )

            if not engine.is_finished():
                step_start = time.perf_counter()
                outputs, num_tokens, batch_size, sch_latency_ms, post_sch_latency_ms = (
                    engine.step()
                )
                step_duration_ms = (time.perf_counter() - step_start) * 1000.0
                elapsed_after_step_s = time.perf_counter() - start
                phase = (
                    "decode"
                    if num_tokens < 0
                    else "maintenance"
                    if num_tokens == 0
                    else "prefill"
                )
                step_record: dict[str, Any] = {
                    "step_idx": step_idx,
                    "elapsed_s": elapsed_after_step_s,
                    "phase": phase,
                    "duration_ms": step_duration_ms,
                    "num_tokens_returned": num_tokens,
                    "batch_size_returned": batch_size,
                    "outputs": len(outputs),
                    "inflight_requests": len(seq_map),
                    "scheduler_latency_ms": sch_latency_ms,
                    "post_scheduler_latency_ms": post_sch_latency_ms,
                }
                if phase == "decode":
                    itl_ms = pending_maintenance_ms + step_duration_ms
                    step_record["itl_ms"] = itl_ms
                    step_record["preceding_maintenance_ms"] = pending_maintenance_ms
                    pending_maintenance_ms = 0.0
                    decode_itls.append(itl_ms)
                    if elapsed_after_step_s >= args.steady_start_sec:
                        steady_decode_itls.append(itl_ms)
                elif phase == "maintenance":
                    pending_maintenance_ms += step_duration_ms
                step_records.append(step_record)
                step_idx += 1

                for seq_id, _ in outputs:
                    seq = seq_map.pop(str(seq_id), None)
                    if seq is None:
                        continue
                    record = completion_record(seq, elapsed_after_step_s)
                    completion_records.append(record)
                    write_completion(record, completion_fh)
            else:
                if not can_send:
                    drain_completed = True
                    break
                sleep_s = max(0.001, min(0.05, next_arrival_s - elapsed_s))
                time.sleep(sleep_s)

            if drain_started_s is not None:
                drain_elapsed_s = (time.perf_counter() - start) - drain_started_s
                if not engine.is_finished() and drain_elapsed_s > args.drain_timeout_sec:
                    print(
                        "DRAIN_TIMEOUT "
                        + json.dumps(
                            {
                                "drain_elapsed_s": drain_elapsed_s,
                                "inflight_requests": len(seq_map),
                            }
                        ),
                        flush=True,
                    )
                    break

            if now - last_progress >= args.progress_interval_sec:
                recent_decode_itls = decode_itls[-32:]
                progress = {
                    "elapsed_s": elapsed_s,
                    "phase": "send" if can_send else "drain",
                    "accepted_requests": requests_sent,
                    "completed_requests": len(completion_records),
                    "inflight_requests": len(seq_map),
                    "backpressure_drops": requests_dropped_backpressure,
                    "steps": step_idx,
                    "decode_steps": len(decode_itls),
                    "recent_decode_itl_ms": summarize(recent_decode_itls),
                }
                print("PROGRESS " + json.dumps(progress), flush=True)
                last_progress = now
    finally:
        if completion_fh is not None:
            completion_fh.close()

    total_elapsed_s = time.perf_counter() - start
    all_itl_samples = [
        sample
        for record in completion_records
        for sample in record.get("itl_samples_ms", [])
    ]
    total_input_tokens = sum(int(record.get("prompt_len") or 0) for record in completion_records)
    total_output_tokens = sum(int(record.get("output_len") or 0) for record in completion_records)
    drain_duration_s = (
        max(0.0, total_elapsed_s - drain_started_s)
        if drain_started_s is not None
        else 0.0
    )
    result = {
        "config": config,
        "warmup": warmup_summary,
        "run": {
            "total_elapsed_sec": total_elapsed_s,
            "send_duration_sec": min(args.duration_sec, drain_started_s or total_elapsed_s),
            "drain_duration_sec": drain_duration_s,
            "drain_completed": drain_completed,
            "accepted_requests": requests_sent,
            "completed_requests": len(completion_records),
            "unfinished_requests": len(seq_map),
            "backpressure_dropped_arrivals": requests_dropped_backpressure,
            "max_inflight_observed": max_inflight_observed,
            "step_count": step_idx,
            "prefill_steps": sum(1 for step in step_records if step["phase"] == "prefill"),
            "decode_steps": len(decode_itls),
            "maintenance_steps": sum(
                1 for step in step_records if step["phase"] == "maintenance"
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "request_throughput_per_sec": (
                len(completion_records) / total_elapsed_s if total_elapsed_s > 0 else 0.0
            ),
            "output_throughput_tokens_per_sec": (
                total_output_tokens / total_elapsed_s if total_elapsed_s > 0 else 0.0
            ),
        },
        "request_metrics": {
            "ttft_ms": summarize(finite_values(completion_records, "ttft_ms")),
            "e2e_latency_ms": summarize(
                finite_values(completion_records, "e2e_latency_ms")
            ),
            "queueing_time_ms": summarize(
                finite_values(completion_records, "queueing_time_ms")
            ),
            "decode_queue_time_ms": summarize(
                finite_values(completion_records, "decode_queue_time_ms")
            ),
            "avg_itl_ms": summarize(finite_values(completion_records, "avg_itl_ms")),
            "avg_tpot_with_queueing_ms": summarize(
                finite_values(completion_records, "avg_tpot_with_queueing_ms")
            ),
            "avg_itl_with_decode_queue_ms": summarize(
                finite_values(completion_records, "avg_itl_with_decode_queue_ms")
            ),
        },
        "token_itl_samples_ms": summarize(all_itl_samples),
        "step_decode_itl_ms": summarize(decode_itls),
        "steady_step_decode_itl_ms": summarize(steady_decode_itls),
        "steps": step_records,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("RESULT_SUMMARY")
    print(
        json.dumps(
            {
                "config": config,
                "run": result["run"],
                "request_metrics": result["request_metrics"],
                "token_itl_samples_ms": result["token_itl_samples_ms"],
                "step_decode_itl_ms": result["step_decode_itl_ms"],
                "steady_step_decode_itl_ms": result["steady_step_decode_itl_ms"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        "RESULT_PATHS "
        + json.dumps(
            {
                "summary_json": str(output_json),
                "completion_jsonl": str(completion_jsonl),
            }
        ),
        flush=True,
    )
    return result, 0 if drain_completed else 2


def main() -> None:
    args = parse_args()
    _, exit_code = run_longrun(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

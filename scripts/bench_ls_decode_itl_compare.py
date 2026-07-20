#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare decode ITL for an 8-GPU DP+EP baseline, a forced "
            "SP8+EP8 baseline, and the LoongServe-style LS-Decode-Core mode."
        )
    )
    parser.add_argument("--case", choices=["dp_ep", "sp8_ep", "ls"], required=True)
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--ray-address", default="10.102.243.60:8776")
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument("--ls-batch-per-master", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument(
        "--discard-decode-steps",
        type=int,
        default=1,
        help="Number of initial decode steps to exclude from steady-state stats.",
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


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
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p90_ms": percentile(values, 90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def build_engine(args: argparse.Namespace) -> LLM:
    is_ls = args.case == "ls"
    is_forced_sp8 = args.case == "sp8_ep"
    if is_ls and not 1 <= args.loop_count <= 16:
        raise ValueError("LS-Decode-Core requires --loop-count in [1, 16]")

    max_model_len = args.max_model_len
    if max_model_len <= 0:
        max_model_len = args.prompt_len + args.max_tokens + 16

    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_num_batched_tokens = max(
            max_model_len,
            args.num_requests * (args.prompt_len + args.max_tokens) + 1024,
        )

    attention_dp = 1 if (is_ls or is_forced_sp8) else 8
    attention_sp = 8 if (is_ls or is_forced_sp8) else 1
    fixed_sp_size = 8 if is_forced_sp8 else 0

    return LLM(
        args.model_path,
        enforce_eager=True,
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        loop_count=args.loop_count,
        max_num_recv_seqs=args.max_num_recv_seqs,
        kvcache_block_size=64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduler_mode="centralized",
        routing_strategy="RoundRobin",
        fixed_sp_size=fixed_sp_size,
        sp_backend="hao_basic" if (is_ls or is_forced_sp8) else "legacy_ll",
        enable_ls_decode_core_scheduler=is_ls,
        ls_decode_initial_kv_dop=0,
        ls_decode_batch_per_master=args.ls_batch_per_master,
        ls_decode_enable_memory_scale_up=True,
    )


def make_sequences(args: argparse.Namespace) -> list[Sequence]:
    rng = np.random.default_rng(args.seed)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    return [
        Sequence(
            rng.integers(0, 10001, size=args.prompt_len, dtype=np.int64).tolist(),
            sampling_params=sampling_params,
        )
        for _ in range(args.num_requests)
    ]


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    engine = build_engine(args)
    seqs = make_sequences(args)
    engine.add_request(seqs)

    steps: list[dict[str, Any]] = []
    decode_step_idx = 0
    start_time = time.perf_counter()

    for step_idx in range(args.max_steps):
        if engine.is_finished():
            break

        step_start = time.perf_counter()
        outputs, num_tokens, batch_size, sch_latency_ms, post_sch_latency_ms = engine.step()
        duration_ms = (time.perf_counter() - step_start) * 1000.0
        is_decode = num_tokens < 0

        entry: dict[str, Any] = {
            "step_idx": step_idx,
            "phase": "decode" if is_decode else "prefill",
            "duration_ms": duration_ms,
            "num_tokens_returned": num_tokens,
            "batch_size_returned": batch_size,
            "outputs": len(outputs),
            "finished_requests": sum(1 for seq in seqs if seq.is_finished),
            "scheduler_latency_ms": sch_latency_ms,
            "post_scheduler_latency_ms": post_sch_latency_ms,
        }
        if is_decode:
            execution_loop_count = engine.last_execution_loop_count
            entry["decode_step_idx"] = decode_step_idx
            entry["execution_loop_count"] = execution_loop_count
            entry["itl_ms"] = duration_ms / execution_loop_count
            decode_step_idx += 1
        steps.append(entry)

    total_duration_ms = (time.perf_counter() - start_time) * 1000.0
    if not engine.is_finished():
        raise RuntimeError(
            f"case {args.case} did not finish within {args.max_steps} steps"
        )

    decode_itls = [
        float(step["itl_ms"]) for step in steps if step["phase"] == "decode"
    ]
    steady_itls = [
        float(step["itl_ms"])
        for step in steps
        if step["phase"] == "decode"
        and int(step["decode_step_idx"]) >= args.discard_decode_steps
    ]

    config = {
        "case": args.case,
        "model_path": args.model_path,
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "total_prompt_tokens": args.num_requests * args.prompt_len,
        "attention_dp": 1 if args.case in {"ls", "sp8_ep"} else 8,
        "attention_sp": 8 if args.case in {"ls", "sp8_ep"} else 1,
        "ffn_ep": 8,
        "fixed_sp_size": 8 if args.case == "sp8_ep" else 0,
        "sp_backend": "hao_basic" if args.case in {"ls", "sp8_ep"} else "legacy_ll",
        "loop_count": args.loop_count,
        "discard_decode_steps": args.discard_decode_steps,
        "max_num_seqs": args.max_num_seqs,
        "max_num_recv_seqs": args.max_num_recv_seqs,
        "ls_batch_per_master": args.ls_batch_per_master,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    result = {
        "config": config,
        "total_duration_ms": total_duration_ms,
        "num_steps": len(steps),
        "decode_steps": len(decode_itls),
        "all_decode_itl": summarize(decode_itls),
        "steady_decode_itl": summarize(steady_itls),
        "steps": steps,
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_case(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_SUMMARY")
    print(json.dumps({k: result[k] for k in ("config", "all_decode_itl", "steady_decode_itl")}, indent=2))


if __name__ == "__main__":
    main()

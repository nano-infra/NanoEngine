#!/usr/bin/env python3
"""CSV serving benchmark for NanoDeploy's LoongServe-style Decode scheduler."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ISSUE003_DIR = SCRIPT_DIR / "issue003"
if str(ISSUE003_DIR) not in sys.path:
    sys.path.insert(0, str(ISSUE003_DIR))

import bench_serving_overhead as serving  # noqa: E402


DEFAULT_DATASET = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/"
    "sharegpt-4o-mixlong-0326/"
    "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bench-serve a CSV workload with the LoongServe-style Decode-only "
            "multi-master scheduler."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ray-address", default="10.102.243.60:8776")
    parser.add_argument("--master-address", default="10.102.243.60:29906")
    parser.add_argument("--duration-sec", type=float, default=360.0)
    parser.add_argument("--request-rate", type=float, default=20.0)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--warmup-prompt-len", type=int, default=512)
    parser.add_argument("--warmup-max-tokens", type=int, default=8)
    parser.add_argument("--verbose-nanodeploy-logs", action="store_true")

    parser.add_argument("--attention-dp", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--attention-sp", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-recv-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--max-input-len", type=int, default=1_000_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1_024_000)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=141.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--segment-size", type=int, default=65_536)
    parser.add_argument("--routing-strategy", default="LeastBatch")
    parser.add_argument("--cuda-graph-mode", choices=["full", "piecewise"], default="full")
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--ls-initial-kv-dop", type=int, default=0)
    parser.add_argument("--ls-batch-per-master", type=int, default=8)
    parser.add_argument(
        "--ls-kv-consolidation-mode",
        choices=["off", "shadow", "execute"],
        default="execute",
    )
    parser.add_argument("--ls-kv-consolidation-candidate-util", type=float, default=0.50)
    parser.add_argument(
        "--ls-kv-consolidation-target-high-watermark", type=float, default=0.80
    )
    parser.add_argument("--ls-kv-consolidation-stable-steps", type=int, default=2)
    parser.add_argument("--ls-kv-consolidation-cooldown-steps", type=int, default=2)
    parser.add_argument("--ls-kv-consolidation-check-interval-steps", type=int, default=1)
    parser.add_argument(
        "--ls-kv-consolidation-max-source-blocks-per-event", type=int, default=128
    )
    parser.add_argument(
        "--ls-kv-consolidation-migration-chunk-tokens", type=int, default=64
    )
    args = parser.parse_args()

    if args.duration_sec <= 0 or args.request_rate <= 0 or args.burstiness <= 0:
        parser.error("duration-sec, request-rate, and burstiness must be positive")
    if args.num_requests < 0:
        parser.error("num-requests must be non-negative")
    if args.warmup_requests < 0:
        parser.error("warmup-requests must be non-negative")
    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")
    if args.attention_sp != 8:
        parser.error("LoongServe-style formal benchmark requires attention-sp=8")
    if args.num_requests == 0:
        args.num_requests = int(round(args.duration_sec * args.request_rate))
    return args


def run_warmup(engine, args: argparse.Namespace) -> None:
    if args.warmup_requests == 0:
        return

    sampling_params = serving.SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=args.warmup_max_tokens,
    )
    sequences = [
        serving.Sequence(
            token_ids=np.random.randint(
                0, 10_000, size=args.warmup_prompt_len
            ).tolist(),
            sampling_params=sampling_params,
        )
        for _ in range(args.warmup_requests)
    ]
    for sequence in sequences:
        engine.add_request(sequence)

    completed = 0
    steps = 0
    started = time.perf_counter()
    while not engine.is_finished():
        outputs, _, _, _, _ = engine.step()
        completed += len(outputs)
        steps += 1
    print(
        "LS_WARMUP_SUMMARY "
        + json.dumps(
            {
                "requests": args.warmup_requests,
                "completed": completed,
                "steps": steps,
                "duration_sec": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def build_engine(args: argparse.Namespace):
    return serving.LLM(
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
        attention_dp=args.attention_dp,
        attention_sp=args.attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=args.attention_dp * args.attention_sp,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_recv_seqs=args.max_num_recv_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        loop_count=1,
        routing_strategy=args.routing_strategy,
        scheduler_mode="centralized",
        segment_size=args.segment_size,
        kvcache_block_size=64,
        fixed_sp_size=0,
        sp_backend="hao_basic",
        use_dlslime_rpc=True,
        optimize_decode_block_table=True,
        enable_non_uniform_split=False,
        enable_dynamic_sp_size=False,
        use_new_decode_dynamic_sp_scheduler=False,
        dynamic_sp_size_strategy="legacy",
        enable_ls_decode_core_scheduler=True,
        ls_decode_initial_kv_dop=args.ls_initial_kv_dop,
        ls_decode_batch_per_master=args.ls_batch_per_master,
        ls_decode_enable_memory_scale_up=True,
        ls_kv_consolidation_mode=args.ls_kv_consolidation_mode,
        ls_kv_consolidation_candidate_util=args.ls_kv_consolidation_candidate_util,
        ls_kv_consolidation_target_high_watermark=(
            args.ls_kv_consolidation_target_high_watermark
        ),
        ls_kv_consolidation_stable_steps=args.ls_kv_consolidation_stable_steps,
        ls_kv_consolidation_cooldown_steps=args.ls_kv_consolidation_cooldown_steps,
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["csv_path"] = str(args.csv_path)
    config["output_jsonl"] = str(args.output_jsonl)
    config["ffn_ep"] = args.attention_dp * args.attention_sp
    config["loop_count"] = 1
    print("LS_BENCH_CONFIG " + json.dumps(config, sort_keys=True), flush=True)

    if not args.verbose_nanodeploy_logs:
        from nanodeploy.logging import get_logger

        nanodeploy_logger = get_logger()
        nanodeploy_logger.setLevel(logging.WARNING)
        for handler in nanodeploy_logger.handlers:
            handler.setLevel(logging.WARNING)

    engine = build_engine(args)
    serving.print_model_config(engine)
    run_warmup(engine, args)

    dataset_args = argparse.Namespace(
        dataset="csv",
        csv_path=str(args.csv_path),
        num_requests=args.num_requests,
        max_input_len=args.max_input_len,
        max_model_len=args.max_model_len,
    )
    request_generator = serving.get_dataset_generator(dataset_args)
    arrival_times = serving.generate_arrival_times(
        args.num_requests, args.request_rate, args.burstiness
    )
    print(
        "LS_BENCH_ARRIVALS "
        + json.dumps(
            {
                "num_requests": args.num_requests,
                "target_duration_sec": args.duration_sec,
                "sampled_last_arrival_sec": float(arrival_times[-1]),
            }
        ),
        flush=True,
    )
    total_time, seq_map = serving.run_benchmark(
        engine, request_generator, arrival_times, args.num_requests
    )
    serving.calculate_and_print_metrics(
        total_time,
        seq_map,
        args.num_requests,
        itl_log_path=str(args.output_jsonl),
    )


if __name__ == "__main__":
    main()

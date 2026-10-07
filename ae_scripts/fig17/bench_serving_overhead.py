#!/usr/bin/env python3
"""Run the fixed 128/128-token NanoDeploy workload used by Fig. 17."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


INPUT_LENGTH = 128
OUTPUT_LENGTH = 128
WARMUP_INPUT_LENGTH = 512
WARMUP_OUTPUT_LENGTH = 256
SEED = 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requests", type=positive_int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-model-len", type=positive_int, default=1_000_000)
    parser.add_argument("--gpu-memory-utilization", type=positive_float, default=0.9)
    parser.add_argument("--gpu-memory-limit-gb", type=positive_float, default=141.0)
    parser.add_argument("--itl-log-path", type=Path, required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--tp", type=positive_int, default=1)
    parser.add_argument("--sp", type=positive_int, required=True)
    parser.add_argument("--dp", type=positive_int, required=True)
    parser.add_argument("--ep", type=positive_int, required=True)
    parser.add_argument("--max-num-seqs", type=positive_int, required=True)
    parser.add_argument("--loop-count", type=positive_int, default=16)
    parser.add_argument("--segment-size", type=positive_int, default=65_536)
    parser.add_argument(
        "--dynamic-sp-size-strategy",
        choices=("legacy", "bucket"),
        default="legacy",
        help="SP-size selection strategy.",
    )
    parser.add_argument(
        "--dynamic-sp-bucket-preset",
        choices=("none", "deepseek_v3"),
        default="none",
        help="Named sequence-length bucket policy used by bucket mode.",
    )
    parser.add_argument(
        "--routing-strategy",
        choices=("RoundRobin", "LeastBatch", "LeastCache"),
        default="LeastBatch",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        parser.error(f"model config not found: {model_path / 'config.json'}")
    if args.gpu_memory_utilization > 1:
        parser.error("--gpu-memory-utilization must not exceed 1")
    if args.dynamic_sp_size_strategy == "bucket":
        if args.dynamic_sp_bucket_preset == "none":
            parser.error(
                "strategy=bucket requires --dynamic-sp-bucket-preset"
            )
    elif args.dynamic_sp_bucket_preset != "none":
        parser.error(
            "--dynamic-sp-bucket-preset requires "
            "--dynamic-sp-size-strategy=bucket"
        )
    args.model_path = model_path
    args.itl_log_path = args.itl_log_path.expanduser().resolve()
    return args


def print_model_config(engine: object) -> None:
    print("\n" + "=" * 40)
    print("Model Configuration")
    print("=" * 40)
    config = getattr(engine, "config", None)
    if config is not None:
        try:
            values = asdict(config)
        except TypeError:
            values = vars(config) if hasattr(config, "__dict__") else {}
        for key, value in values.items():
            print(f"{key}: {value}")
    print("=" * 40 + "\n")


def make_sequence(input_length: int, output_length: int):
    prompt = np.random.randint(0, 10_000, size=input_length).tolist()
    sampling_params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=output_length,
    )
    return Sequence(token_ids=prompt, sampling_params=sampling_params)


def run_warmup(engine: object, max_num_seqs: int, world_size: int) -> None:
    num_requests = max_num_seqs * world_size
    sequences = [
        make_sequence(WARMUP_INPUT_LENGTH, WARMUP_OUTPUT_LENGTH)
        for _ in range(num_requests)
    ]
    for sequence in sequences:
        engine.add_request(sequence)

    start_time = time.perf_counter()
    completed = 0
    print(f"Running Warmup Phase: {num_requests} requests")
    with tqdm(total=num_requests, desc="Warmup Requests") as progress:
        while completed < num_requests:
            if engine.is_finished():
                time.sleep(0.001)
                continue
            outputs, _, _, _, _ = engine.step()
            completed += len(outputs)
            progress.update(len(outputs))

    elapsed = time.perf_counter() - start_time
    print(f"\nWarmup completed in {elapsed:.2f}s")


def run_benchmark(
    engine: object,
    num_requests: int,
) -> tuple[float, dict[int, object]]:
    sequences = [
        make_sequence(INPUT_LENGTH, OUTPUT_LENGTH) for _ in range(num_requests)
    ]
    sequence_by_id = {sequence.seq_id: sequence for sequence in sequences}

    print(
        f"Submitting all {num_requests} measured requests before the first "
        "engine step"
    )
    start_time = time.perf_counter()
    for sequence in sequences:
        engine.add_request(sequence)
    submission_time = time.perf_counter() - start_time
    print(f"Submitted request burst in {submission_time:.3f}s")

    with tqdm(total=num_requests, desc="Draining Request Burst") as progress:
        while not engine.is_finished():
            outputs, _, _, _, _ = engine.step()
            progress.update(len(outputs))

    return time.perf_counter() - start_time, sequence_by_id


def report_results(
    total_time: float,
    sequence_by_id: dict[int, object],
    expected_requests: int,
    output_path: Path,
) -> None:
    completed = [
        sequence
        for sequence in sequence_by_id.values()
        if sequence.metric and sequence.metric.completion_time
    ]
    all_itl_samples = [
        sample
        for sequence in completed
        for sample in (sequence.metric.itl_samples or [])
    ]
    if len(completed) != expected_requests:
        raise RuntimeError(
            "NanoDeploy did not complete every request: "
            f"completed={len(completed)}, expected={expected_requests}"
        )
    if not all_itl_samples:
        raise RuntimeError("NanoDeploy produced no ITL samples")

    total_output_tokens = sum(
        sequence.metric.num_generated_tokens for sequence in completed
    )
    print("\n" + "=" * 60)
    print("--- Benchmark Results ---")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {expected_requests}")
    print(f"Requests completed: {len(completed)}")
    print(f"Throughput: {total_output_tokens / total_time:.2f} tokens/s")
    print("\n--- TPOT without Queueing Time (ms/token) ---")
    print(f"  Avg:  {np.mean(all_itl_samples):.2f}")
    print(f"  P50:  {np.percentile(all_itl_samples, 50):.2f}")
    print(f"  P90:  {np.percentile(all_itl_samples, 90):.2f}")
    print(f"  P95:  {np.percentile(all_itl_samples, 95):.2f}")
    print(f"  P99:  {np.percentile(all_itl_samples, 99):.2f}")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with output_path.open("x", encoding="utf-8") as output_file:
        for sequence in completed:
            metric = sequence.metric
            json.dump(
                {
                    "seq_id": sequence.seq_id,
                    "itl_samples": metric.itl_samples,
                    "prompt_len": metric.num_prompt_tokens,
                    "output_len": metric.num_generated_tokens,
                    "queueing_time_ms": metric.queueing_time_ms,
                    "decode_queue_time_ms": metric.decode_queue_time_ms,
                    "avg_itl_with_decode_queue_ms": (
                        metric.avg_itl_with_decode_queue
                    ),
                },
                output_file,
            )
            output_file.write("\n")
    print(f"Saved request metrics: {output_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)

    global LLM, SamplingParams, Sequence
    from nanodeploy import LLM, SamplingParams
    from nanodeploy.engine.sequence import Sequence

    engine = LLM(
        str(args.model_path),
        cuda_graph_mode="full",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        master_address=args.master_address,
        ray_address=args.ray_address,
        mode="decode",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=False,
        moe_routing_simulation_strategy="uniform_random",
        attention_dp=args.dp,
        attention_sp=args.sp,
        attention_tp=args.tp,
        ffn_dp=1,
        ffn_ep=args.ep,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=1_024_000,
        loop_count=args.loop_count,
        routing_strategy=args.routing_strategy,
        scheduler_arch="legacy_global",
        router_policy="least_batch",
        segment_size=args.segment_size,
        kvcache_block_size=64,
        max_num_recv_seqs=16,
        enable_non_uniform_split=True,
        fixed_sp_size=0,
        sp_backend="hao_basic",
        dynamic_sp_size_strategy=args.dynamic_sp_size_strategy,
        dynamic_sp_bucket_preset=args.dynamic_sp_bucket_preset,
    )
    print_model_config(engine)
    run_warmup(engine, args.max_num_seqs, args.ep)
    total_time, sequence_by_id = run_benchmark(
        engine,
        args.num_requests,
    )
    report_results(
        total_time,
        sequence_by_id,
        args.num_requests,
        args.itl_log_path,
    )


if __name__ == "__main__":
    main()

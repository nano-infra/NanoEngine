import argparse
import os
import time
import uuid
from dataclasses import asdict
from random import randint, seed

import numpy as np
import pandas as pd
from dlengine import LLM, SamplingParams
from dlengine._rust.proto import RequestIn
from tqdm.auto import tqdm

# Constants
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
SEED = 0

# specific seed
seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Serving benchmark for DLEngine.")
    parser.add_argument(
        "--num-requests", type=int, default=256, help="Number of requests."
    )
    parser.add_argument(
        "--request-rate", type=int, default=8, help="Requests per second."
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor (1.0 = Poisson).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/models/qwen3-235B-Instruct-2507-FP8",
        help="Model path.",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Max model length."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization.",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        choices=["random", "csv"],
        help="Dataset type.",
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Path to CSV file.")

    # Distributed / Cluster arguments
    parser.add_argument(
        "--master-address", type=str, default=None, help="Ray master address."
    )
    parser.add_argument(
        "--ray-address", type=str, default=None, help="Ray cluster address."
    )

    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size.")
    parser.add_argument("--sp", type=int, default=1, help="Sequence Parallel size.")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size.")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel size.")

    # Engine args
    parser.add_argument(
        "--max-num-seqs", type=int, default=128, help="Max sequences per iteration."
    )
    parser.add_argument(
        "--dummy-prefill", action="store_true", help="Use dummy prefill."
    )
    parser.add_argument(
        "--loop-count", type=int, default=16, help="Steps per iteration."
    )

    parser.add_argument(
        "--routing-strategy",
        type=str,
        default="RoundRobin",
        choices=["RoundRobin", "LeastBatch", "LeastCache"],
        help="Routing strategy.",
    )

    args = parser.parse_args()

    if args.dataset == "csv":
        if args.csv_path is None:
            parser.error("--csv-path is required when --dataset=csv")
        if not os.path.exists(args.csv_path):
            parser.error(f"CSV file not found: {args.csv_path}")

    return args


def get_dataset(args):
    """Generates or loads the dataset of prompts and sampling params."""
    if args.dataset == "random":
        print(
            f"Generating random dataset (input: {MAX_INPUT_LEN}, output: {MAX_OUTPUT_LEN})..."
        )
        prompts = [
            [randint(0, 10000) for _ in range(MAX_INPUT_LEN)]
            for _ in range(args.num_requests)
        ]
        sampling_params_list = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_OUTPUT_LEN)
            for _ in range(args.num_requests)
        ]
        return prompts, sampling_params_list

    # CSV dataset
    print(f"Reading dataset from CSV: {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    if "prompt_len" not in df.columns or "output_len" not in df.columns:
        raise ValueError("CSV file must contain 'prompt_len' and 'output_len' columns")

    if len(df) < args.num_requests:
        print(
            f"Warning: CSV has {len(df)} rows, requested {args.num_requests}. Using {len(df)}."
        )
        args.num_requests = len(df)
    else:
        df = df.head(args.num_requests)

    prompts = []
    sampling_params_list = []
    for _, row in df.iterrows():
        prompt_len = int(row["prompt_len"])
        output_len = int(row["output_len"])
        prompts.append([randint(0, 10000) for _ in range(prompt_len)])
        sampling_params_list.append(
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        )

    print(f"Loaded {len(prompts)} requests from CSV")
    return prompts, sampling_params_list


def generate_arrival_times(num_requests, rate, burstiness):
    """Generates request arrival times based on burstiness factor."""
    if burstiness == 1.0:
        intervals = np.random.exponential(1.0 / rate, num_requests)
    else:
        shape = 1.0 / (burstiness**2)
        scale = burstiness**2 / rate
        intervals = np.random.gamma(shape, scale, num_requests)
    return np.cumsum(intervals)


def print_model_config(engine):
    """Prints the model configuration from the engine."""
    print("\n" + "=" * 40)
    print("Model Configuration")
    print("=" * 40)
    if hasattr(engine, "config"):
        # Assuming config is a dataclass or has __dict__
        conf = engine.config
        try:
            # If it's a dataclass
            conf_dict = asdict(conf)
        except TypeError:
            # Fallback if not a dataclass
            conf_dict = conf.__dict__ if hasattr(conf, "__dict__") else {}

        for k, v in conf_dict.items():
            print(f"{k}: {v}")
    else:
        print("Config not accessible directly from engine.")
    print("=" * 40 + "\n")


def run_benchmark(engine, prompts, sampling_params_list, arrival_times, num_requests):
    """Runs the main benchmark loop."""
    submit_times = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []
    completed = set()

    with tqdm(total=num_requests, desc="Processing Requests") as pbar:
        while requests_sent < num_requests or not engine.is_finished():
            current_time = time.perf_counter()
            elapsed = current_time - start_time

            # Send requests
            while (
                requests_sent < num_requests and elapsed >= arrival_times[requests_sent]
            ):

                # print(f"DEBUG: Adding request {requests_sent} at {elapsed:.4f}s")
                prompt = prompts[requests_sent]
                sp = sampling_params_list[requests_sent]

                seq_id = uuid.uuid4().int & ((1 << 63) - 1)
                engine.add_request_payload(RequestIn(seq_id, prompt, sp, 0).to_bytes())
                submit_times[seq_id] = time.perf_counter()
                requests_sent += 1

            # Engine step
            if not engine.is_finished():
                result = engine.step()

                # Update progress bar with latency info
                for event in result.outputs:
                    seq_id = int(event["seq_id"])
                    if (
                        seq_id in submit_times
                        and seq_id not in completed
                        and (event["is_finished"] or event["is_to_be_migrated"])
                    ):
                        completed.add(seq_id)
                        completed_latencies.append(
                            time.perf_counter() - submit_times[seq_id]
                        )
                        avg_lat = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_lat:.2f}s"})
                        pbar.update(1)
            else:
                time.sleep(0.001)

    total_time = time.perf_counter() - start_time
    return total_time, submit_times


def calculate_and_print_metrics(total_time, seq_map, requests_sent):
    """Calculates and prints performance metrics."""
    completed_seqs = [
        s for s in seq_map.values() if s.metric and s.metric.completion_time
    ]

    total_input = sum(s.metric.num_prompt_tokens for s in completed_seqs)
    total_output = sum(s.metric.num_generated_tokens for s in completed_seqs)

    throughput = total_output / total_time

    ttft_samples = [s.metric.ttft for s in completed_seqs if s.metric.ttft]
    avg_ttft = np.mean(ttft_samples) if ttft_samples else 0

    latency_samples = [
        s.metric.e2e_latency for s in completed_seqs if s.metric.e2e_latency
    ]
    avg_latency = np.mean(latency_samples) / 1000 if latency_samples else 0

    # TPOT stats (Inter-Token Latency)
    itls = [s.metric.avg_itl for s in completed_seqs if s.metric.avg_itl]
    tpot_stats = {}
    if itls:
        tpot_stats = {
            "avg": np.mean(itls),
            "p50": np.median(itls),
            "p90": np.percentile(itls, 90),
            "p99": np.percentile(itls, 99),
        }

    # TPOT with queueing
    tpot_wq_samples = [
        s.metric.avg_tpot_with_queueing
        for s in completed_seqs
        if s.metric.avg_tpot_with_queueing
    ]
    tpot_wq_stats = {}
    if tpot_wq_samples:
        tpot_wq_stats = {
            "avg": np.mean(tpot_wq_samples),
            "p50": np.percentile(tpot_wq_samples, 50),
            "p90": np.percentile(tpot_wq_samples, 90),
            "p95": np.percentile(tpot_wq_samples, 95),
            "p99": np.percentile(tpot_wq_samples, 99),
        }

    # Goodput
    slo_threshold = 100  # ms
    slo_success = sum(1 for s in tpot_wq_samples if s < slo_threshold)
    total_seqs = len(completed_seqs)
    goodput = (slo_success / total_seqs * 100) if total_seqs > 0 else 0

    print("\n" + "=" * 60)
    print("--- Benchmark Results ---")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {total_seqs}")
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft:.2f} ms")
    print(f"Average E2E Latency: {avg_latency:.2f} s")
    print()

    if tpot_stats:
        print("--- TPOT without Queueing Time (ms/token) ---")
        print(f"  Avg:  {tpot_stats.get('avg', 0):.2f}")
        print(f"  P50:  {tpot_stats.get('p50', 0):.2f}")
        print(f"  P90:  {tpot_stats.get('p90', 0):.2f}")
        print(f"  P99:  {tpot_stats.get('p99', 0):.2f}")
        print()

    if tpot_wq_stats:
        print("--- TPOT with Queueing Time (ms/token) ---")
        print(f"  Avg:  {tpot_wq_stats.get('avg', 0):.2f}")
        print(f"  P50:  {tpot_wq_stats.get('p50', 0):.2f}")
        print(f"  P90:  {tpot_wq_stats.get('p90', 0):.2f}")
        print(f"  P95:  {tpot_wq_stats.get('p95', 0):.2f}")
        print(f"  P99:  {tpot_wq_stats.get('p99', 0):.2f}")
        print()

    print("--- Goodput (SLO: TPOT with queueing < 100ms) ---")
    print(f"  SLO Success: {slo_success}/{total_seqs}")
    print(f"  Goodput: {goodput:.2f}%")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    print(
        f"\n--- Benchmark: {args.num_requests} reqs, {args.request_rate} req/s, burst={args.burstiness} ---"
    )

    # Initialize Engine
    from dlengine.config import Config

    # Initialize Engine
    config = Config(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        master_address=args.master_address,
        ray_address=args.ray_address,
        mode="decode",
        dummy_prefill=args.dummy_prefill,
        dummy_weight=True,
        perfect_eplb=True,
        attention_dp=args.dp,
        attention_sp=args.sp,
        attention_tp=args.tp,
        ffn_dp=1,
        ffn_ep=args.ep,
        ffn_tp=1,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=1024000,
        routing_strategy=args.routing_strategy,
    )
    engine = LLM(config)

    # Print Config
    print_model_config(engine)

    # Prepare Data
    prompts, sampling_params_list = get_dataset(args)
    arrival_times = generate_arrival_times(
        args.num_requests, args.request_rate, args.burstiness
    )

    # Run Benchmark
    total_time, seq_map = run_benchmark(
        engine, prompts, sampling_params_list, arrival_times, args.num_requests
    )

    # Report
    calculate_and_print_metrics(total_time, seq_map, args.num_requests)


if __name__ == "__main__":
    main()

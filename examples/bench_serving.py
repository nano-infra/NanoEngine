"""
Serving Benchmark for NanoDeploy

This script benchmarks the serving performance of NanoDeploy by simulating
realistic request patterns with configurable arrival time distributions.

Key Adaptations from external benchmark:
1. Uses NanoDeploy's LLM and SamplingParams instead of nanovllm
2. Uses Sequence objects to wrap requests
3. Accesses engine.scheduler.running(dp_idx) instead of engine.scheduler.running
4. Returns from step() include: outputs, num_tokens, bs, sch_latency, post_sch_latency
5. Supports multi-DP (data parallel) configuration

Metrics Collected:
- TTFT (Time to First Token): Time from request submission to first token generation
- TPOT (Time Per Output Token): Average time to generate each subsequent token
- E2E Latency: Total time from request submission to completion
- Throughput: Total output tokens per second

Burstiness Configuration:
- Burstiness factor controls the variability of request arrival times
- 1.0 (default): Poisson process with exponential inter-arrival times (CV=1)
- <1.0: More regular/uniform arrivals (e.g., 0.5 for CV=0.5)
- >1.0: More bursty arrivals (e.g., 2.0 for CV=2)
- Uses Gamma distribution with shape parameter k = 1/(burstiness^2)

Usage:
    # Standard Poisson process with random dataset
    python bench_serving.py --num-requests 256 --request-rate 8

    # More regular arrivals (less bursty)
    python bench_serving.py --num-requests 256 --request-rate 8 --burstiness 0.5

    # More bursty arrivals
    python bench_serving.py --num-requests 256 --request-rate 8 --burstiness 2.0

    # Custom model configuration
    python bench_serving.py --num-requests 100 --request-rate 4 --model-path /path/to/model

    # Use CSV dataset
    python bench_serving.py --dataset csv --csv-path dataset.csv --num-requests 1000 --request-rate 8
"""

import argparse
import os
import time
from random import randint, seed

import numpy as np
import pandas as pd
from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence
from tqdm.auto import tqdm

MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

# --- Seed for reproducibility ---
seed(0)
np.random.seed(0)


def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for NanoDeploy.")
    parser.add_argument(
        "--num-requests", type=int, default=256, help="Number of requests to process."
    )
    parser.add_argument(
        "--request-rate",
        type=int,
        default=8,
        help="Request rate (requests per second).",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request arrivals. 1.0 = Poisson process (default), "
        "<1.0 = more regular, >1.0 = more bursty.",
    )
    parser.add_argument(
        "--model-path", type=str, default="/models/qwen3-235B-Instruct-2507-FP8", help="Path to the model."
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Maximum model length."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
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
        help="Dataset type: 'random' for random generation, 'csv' to read from CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV file (required when --dataset=csv). CSV should have 'prompt_len' and 'output_len' columns.",
    )
    # Distributed / Cluster arguments
    parser.add_argument(
        "--master-address",
        type=str,
        default=None,
        help="Address of the Ray master/head node (e.g. '127.0.0.1:6379').",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the Ray cluster (e.g. 'auto' or '127.0.0.1:6379').",
    )
    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--sp", type=int, default=1, help="Sequence Parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size (for attention)")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel size")
    # Other engine args
    parser.add_argument("--max-num-seqs", type=int, default=256, help="Max number of sequences per iteration.")
    parser.add_argument("--dummy-prefill", action="store_true", help="Use dummy prefill (mock execution).")
    parser.add_argument("--loop-count", type=int, default=16, help="Number of steps per iteration (default 16).")
    args = parser.parse_args()

    NUM_REQUESTS = args.num_requests
    REQUEST_RATE = args.request_rate
    BURSTINESS = args.burstiness

    # Validate CSV path if dataset is csv
    if args.dataset == "csv":
        if args.csv_path is None:
            parser.error("--csv-path is required when --dataset=csv")
        if not os.path.exists(args.csv_path):
            parser.error(f"CSV file not found: {args.csv_path}")

    print(
        f"\n--- Running benchmark with --num-requests {NUM_REQUESTS} --request-rate {REQUEST_RATE} --burstiness {BURSTINESS} ---"
    )
    print(f"Dataset type: {args.dataset}")
    if args.dataset == "csv":
        print(f"CSV path: {args.csv_path}")

    # Initialize LLM engine
    llm = LLM(
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
        # Scaling FFN config with same params for now or customize as needed
        ffn_dp=1, # Often 1 if fully sharded or different strategy
        ffn_ep=args.ep,
        ffn_tp=1, # Assuming 1 for simplicity unless mapped to tp arg
        max_num_seqs=args.max_num_seqs,
        loop_count=args.loop_count,
        routing_strategy="LeastBatch",
    )
    engine = llm

    # --- Generate prompts ---
    if args.dataset == "random":
        # Generate prompts and sampling params with fixed lengths
        print(
            f"Generating random dataset with fixed lengths (input: {MAX_INPUT_LEN}, output: {MAX_OUTPUT_LEN})..."
        )
        prompts = [
            [randint(0, 10000) for _ in range(MAX_INPUT_LEN)]
            for _ in range(NUM_REQUESTS)
        ]
        sampling_params_list = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_OUTPUT_LEN)
            for _ in range(NUM_REQUESTS)
        ]
    else:  # csv
        # Read prompts from CSV file
        print(f"Reading dataset from CSV: {args.csv_path}...")
        df = pd.read_csv(args.csv_path)

        # Validate CSV columns
        if "prompt_len" not in df.columns or "output_len" not in df.columns:
            raise ValueError(
                "CSV file must contain 'prompt_len' and 'output_len' columns"
            )

        # Limit to NUM_REQUESTS
        if len(df) < NUM_REQUESTS:
            print(
                f"Warning: CSV has only {len(df)} rows, but {NUM_REQUESTS} requests were requested. Using {len(df)} requests."
            )
            NUM_REQUESTS = len(df)
        else:
            df = df.head(NUM_REQUESTS)

        # Generate prompts based on exact prompt_len and output_len from CSV
        prompts = []
        sampling_params_list = []
        for idx, row in df.iterrows():
            prompt_len = int(row["prompt_len"])
            output_len = int(row["output_len"])

            # Generate prompt with exact specified length
            prompt = [randint(0, 10000) for _ in range(prompt_len)]
            prompts.append(prompt)

            # Create sampling params with exact specified output length
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            sampling_params_list.append(sp)

        print(f"Loaded {len(prompts)} requests from CSV")
        if len(prompts) < 10:
             print(f"DEBUG: Prompts list: {prompts}")

    print(f"DEBUG: Total NUM_REQUESTS: {NUM_REQUESTS}")

    # --- Generate request arrival times ---
    # BURSTINESS = 1.0 for Poisson (exponential inter-arrival times)
    # BURSTINESS < 1.0 for more regular arrivals; > 1.0 for more bursty arrivals
    if BURSTINESS == 1.0:
        # Standard Poisson process (exponential inter-arrival times)
        request_intervals = np.random.exponential(1.0 / REQUEST_RATE, NUM_REQUESTS)
    else:
        # Gamma distribution to control burstiness
        # Shape parameter k controls CV: CV = 1/sqrt(k)
        # For burstiness factor b: k = 1/(b^2)
        shape = 1.0 / (BURSTINESS**2)
        scale = BURSTINESS**2 / REQUEST_RATE
        request_intervals = np.random.gamma(shape, scale, NUM_REQUESTS)

    arrival_times = np.cumsum(request_intervals)

    # --- Benchmark loop ---
    seq_map = {}  # Map seq_id to Sequence object
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=NUM_REQUESTS, desc="Processing Requests") as pbar:
        while requests_sent < NUM_REQUESTS or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            # DEBUG: Trace loop status periodically
            if requests_sent < NUM_REQUESTS and (requests_sent < 5 or requests_sent % 10 == 0):
                 pass # print(f"DEBUG: Checking arrival. Sent: {requests_sent}, Elapsed: {current_time - start_time:.4f}, Next: {arrival_times[requests_sent]:.4f}")

            while (
                requests_sent < NUM_REQUESTS
                and current_time - start_time >= arrival_times[requests_sent]
            ):
                print(f"DEBUG: Adding request {requests_sent} at {current_time - start_time:.4f}s")
                prompt = prompts[requests_sent]
                sp = sampling_params_list[requests_sent]

                # Create Sequence object for NanoDeploy
                seq = Sequence(
                    token_ids=prompt,
                    sampling_params=sp,
                )

                engine.add_request(seq)

                # Store sequence reference for later metric access
                seq_map[seq.seq_id] = seq

                requests_sent += 1

            # --- Engine step ---
            if not engine.is_finished():
                # Get outputs from step
                outputs, num_tokens, bs, sch_latency, post_sch_latency = engine.step()
                if bs > 0:
                     pass # print(f"DEBUG: Step batch size: {bs}, num_tokens: {num_tokens}")

                # Process completed sequences
                for seq_id, output_ids in outputs:
                    if seq_id in seq_map:
                        seq = seq_map[seq_id]
                        if seq.metric and seq.metric.e2e_latency is not None:
                            completed_latencies.append(
                                seq.metric.e2e_latency / 1000
                            )  # Convert ms to s
                            avg_latency = np.mean(completed_latencies)
                            pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    # Get completed sequences with metrics
    completed_seqs = [
        seq
        for seq in seq_map.values()
        if seq.metric and seq.metric.completion_time is not None
    ]

    total_input_tokens = sum(seq.metric.num_prompt_tokens for seq in completed_seqs)
    total_output_tokens = sum(seq.metric.num_generated_tokens for seq in completed_seqs)

    # TTFT and E2E latency (convert from ms to s for display)
    ttft_samples = [
        seq.metric.ttft / 1000 for seq in completed_seqs if seq.metric.ttft is not None
    ]
    avg_ttft = np.mean(ttft_samples) if ttft_samples else 0

    e2e_samples = [
        seq.metric.e2e_latency / 1000
        for seq in completed_seqs
        if seq.metric.e2e_latency is not None
    ]
    avg_latency = np.mean(e2e_samples) if e2e_samples else 0

    throughput = total_output_tokens / total_time

    # TPOT without queueing time statistics (ITL)
    # avg_itl is already in ms, convert to seconds for TPOT
    itl_samples = [
        seq.metric.avg_itl / 1000
        for seq in completed_seqs
        if seq.metric.avg_itl is not None
    ]
    if itl_samples:
        tpot_avg = np.mean(itl_samples)
        tpot_p50 = np.median(itl_samples)
        tpot_p90 = np.percentile(itl_samples, 90)
        tpot_p99 = np.percentile(itl_samples, 99)
    else:
        tpot_avg = tpot_p50 = tpot_p90 = tpot_p99 = 0

    # TPOT with queueing time statistics
    # avg_tpot_with_queueing is already in ms, convert to seconds
    tpot_with_queueing_samples = [
        seq.metric.avg_tpot_with_queueing / 1000
        for seq in completed_seqs
        if seq.metric.avg_tpot_with_queueing is not None
    ]
    if tpot_with_queueing_samples:
        tpot_wq_avg = np.mean(tpot_with_queueing_samples)
        tpot_wq_p50 = np.percentile(tpot_with_queueing_samples, 50)
        tpot_wq_p90 = np.percentile(tpot_with_queueing_samples, 90)
        tpot_wq_p95 = np.percentile(tpot_with_queueing_samples, 95)
        tpot_wq_p99 = np.percentile(tpot_with_queueing_samples, 99)
    else:
        tpot_wq_avg = tpot_wq_p50 = tpot_wq_p90 = tpot_wq_p95 = tpot_wq_p99 = 0

    # Goodput calculation (SLO: avg_tpot_with_queueing < 100ms)
    SLO_THRESHOLD_MS = 100
    slo_success_count = sum(
        1
        for seq in completed_seqs
        if seq.metric.avg_tpot_with_queueing is not None
        and seq.metric.avg_tpot_with_queueing < SLO_THRESHOLD_MS
    )
    total_sequences = len(completed_seqs)
    goodput = (slo_success_count / total_sequences * 100) if total_sequences > 0 else 0

    print("\n" + "=" * 60)
    print("--- Benchmark Results ---")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {total_sequences}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average latency: {avg_latency:.2f} s")
    print()
    print("--- TPOT without Queueing Time ---")
    print(f"  Avg:  {tpot_avg * 1000:.2f} ms/token")
    print(f"  P50:  {tpot_p50 * 1000:.2f} ms/token")
    print(f"  P90:  {tpot_p90 * 1000:.2f} ms/token")
    print(f"  P99:  {tpot_p99 * 1000:.2f} ms/token")
    print()
    print("--- TPOT with Queueing Time ---")
    print(f"  Avg:  {tpot_wq_avg * 1000:.2f} ms/token")
    print(f"  P50:  {tpot_wq_p50 * 1000:.2f} ms/token")
    print(f"  P90:  {tpot_wq_p90 * 1000:.2f} ms/token")
    print(f"  P95:  {tpot_wq_p95 * 1000:.2f} ms/token")
    print(f"  P99:  {tpot_wq_p99 * 1000:.2f} ms/token")
    print()
    print("--- Goodput (SLO: TPOT with queueing < 100ms) ---")
    print(f"  SLO Success: {slo_success_count}/{total_sequences}")
    print(f"  Goodput: {goodput:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

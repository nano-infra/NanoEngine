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
    # Standard Poisson process
    python bench_serving.py --num-requests 256 --request-rate 8
    
    # More regular arrivals (less bursty)
    python bench_serving.py --num-requests 256 --request-rate 8 --burstiness 0.5
    
    # More bursty arrivals
    python bench_serving.py --num-requests 256 --request-rate 8 --burstiness 2.0
    
    # Custom model configuration
    python bench_serving.py --num-requests 100 --request-rate 4 --model-path /path/to/model
"""

import os
import time
import numpy as np
import argparse
from random import randint, seed
from tqdm.auto import tqdm
from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence

# --- Constants ---
MODEL_PATH = os.path.expanduser("~/Data/Qwen3-0.6B/")
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

# --- Seed for reproducibility ---
seed(0)
np.random.seed(0)


class RequestMetrics:
    """Stores metrics for a single request."""
    def __init__(self, request_id, input_len, max_output_len):
        self.request_id = request_id
        self.input_len = input_len
        self.max_output_len = max_output_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_submission(self):
        self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float('nan')

    @property
    def latency(self):
        return self.completion_time - self.submission_time


def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for NanoDeploy.")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=int, default=8, help="Request rate (requests per second).")
    parser.add_argument("--burstiness", type=float, default=1.0, 
                        help="Burstiness factor for request arrivals. 1.0 = Poisson process (default), "
                             "<1.0 = more regular, >1.0 = more bursty.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the model.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode.")
    args = parser.parse_args()

    NUM_REQUESTS = args.num_requests
    REQUEST_RATE = args.request_rate
    BURSTINESS = args.burstiness

    print(f"\n--- Running benchmark with --num-requests {NUM_REQUESTS} --request-rate {REQUEST_RATE} --burstiness {BURSTINESS} ---")
    
    # Initialize LLM engine
    llm = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = llm

    # --- Generate random prompts ---
    prompts = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_REQUESTS)]
    sampling_params_list = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN)) 
        for _ in range(NUM_REQUESTS)
    ]

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
        shape = 1.0 / (BURSTINESS ** 2)
        scale = BURSTINESS ** 2 / REQUEST_RATE
        request_intervals = np.random.gamma(shape, scale, NUM_REQUESTS)
    
    arrival_times = np.cumsum(request_intervals)

    # --- Benchmark loop ---
    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=NUM_REQUESTS, desc="Processing Requests") as pbar:
        while requests_sent < NUM_REQUESTS or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            while requests_sent < NUM_REQUESTS and current_time - start_time >= arrival_times[requests_sent]:
                prompt = prompts[requests_sent]
                sp = sampling_params_list[requests_sent]
                
                # Create Sequence object for NanoDeploy
                seq = Sequence(
                    token_ids=prompt,
                    sampling_params=sp,
                )
                
                engine.add_request(seq)
                
                # Track metrics for this request
                seq_id = seq.seq_id
                req_metrics = RequestMetrics(seq_id, len(prompt), sp.max_tokens)
                req_metrics.record_submission()
                metrics[seq_id] = req_metrics
                
                requests_sent += 1

            # --- Engine step ---
            if not engine.is_finished():
                # Get outputs from step
                outputs, num_tokens, bs, sch_latency, post_sch_latency = engine.step()

                # Record first token time for all running sequences
                for dp_idx in range(engine.config.attention_dp):
                    for seq in engine.scheduler.running(dp_idx):
                        if seq.seq_id in metrics and seq.num_completed_tokens > 0:
                            metrics[seq.seq_id].record_first_token()

                # Process completed sequences
                for seq_id, output_ids in outputs:
                    if seq_id in metrics:
                        metrics[seq_id].record_first_token()  # Ensure first token time is recorded
                        metrics[seq_id].record_completion(output_ids)
                        
                        completed_latencies.append(metrics[seq_id].latency)
                        avg_latency = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)
    
    avg_ttft = np.mean([m.ttft for m in metrics.values() if m.first_token_time != -1])
    avg_tpot = np.mean([m.tpot for m in metrics.values() if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in metrics.values() if m.completion_time != -1])
    throughput = total_output_tokens / total_time

    print("\n" + "="*60)
    print("--- Benchmark Results ---")
    print("="*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {len([m for m in metrics.values() if m.completion_time != -1])}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms/token")
    print(f"Average latency: {avg_latency:.2f} s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

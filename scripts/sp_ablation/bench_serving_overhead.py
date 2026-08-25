import argparse
import json
import os
import sys
import time
from random import randint, seed
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BUILD_LIB_DIR = os.path.join(ROOT_DIR, "build", "lib")
if BUILD_LIB_DIR not in sys.path and os.path.isdir(BUILD_LIB_DIR):
    sys.path.insert(0, BUILD_LIB_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.hierarchical_contract import (
    FinishEvent,
    HIERARCHICAL_LOOP_COUNT,
)
from nanodeploy.engine.sequence import Sequence
from tqdm.auto import tqdm

# Constants
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
DEFAULT_MAX_REQUEST_TOKENS = 910_000
SEED = 0

# specific seed
seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Serving benchmark for NanoDeploy.")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests.")
    parser.add_argument("--request-rate", type=float, default=8, help="Requests per second.")
    parser.add_argument("--burstiness", type=float, default=1.0, help="Burstiness factor (1.0 = Poisson).")
    parser.add_argument("--model-path", type=str, default="/models/qwen3-235B-Instruct-2507-FP8", help="Model path.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=None, help="GPU memory limit in GB.")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode.")
    parser.add_argument(
        "--cuda-graph-mode",
        type=str,
        default="full",
        choices=["full", "piecewise"],
        help="CUDA Graph mode for decode.",
    )
    parser.add_argument("--dataset", type=str, default="random", choices=["random", "csv"], help="Dataset type.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to CSV file.")
    parser.add_argument("--max-input-len", type=int, default=None, help="Filter out CSV rows with prompt_len >= this value.")
    parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=DEFAULT_MAX_REQUEST_TOKENS,
        help=(
            "Filter out CSV rows where prompt_len + output_len exceeds this "
            f"value (default: {DEFAULT_MAX_REQUEST_TOKENS}; 0 disables)."
        ),
    )
    parser.add_argument(
        "--itl-log-path",
        type=str,
        default="itl_samples.jsonl",
        help=(
            "Compatibility path for request-level metric JSONL. Raw per-token "
            "ITL samples are not required."
        ),
    )
    parser.add_argument(
        "--request-metrics-log-path",
        type=str,
        default=None,
        help=(
            "Path to incrementally save one scalar metric record per request. "
            "Defaults to --itl-log-path for compatibility."
        ),
    )
    parser.add_argument(
        "--metrics-summary-path",
        type=str,
        default=None,
        help=(
            "Path to save aggregate metric percentiles. Defaults to "
            "<request-metrics-log-path stem>.summary.json."
        ),
    )
    
    # Distributed / Cluster arguments
    parser.add_argument("--master-address", type=str, default=None, help="Ray master address.")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address.")
    
    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size.")
    parser.add_argument("--sp", type=int, default=1, help="Sequence Parallel size.")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size.")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel size.")
    
    # Engine args
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Max sequences per iteration.")
    parser.add_argument("--dummy-prefill", action="store_true", help="Use dummy prefill.")
    parser.add_argument("--loop-count", type=int, default=16, help="Steps per iteration.")
    parser.add_argument("--segment-size", type=int, default=65536, help="Segment size for SP.")
    parser.add_argument("--sp-backend", type=str, default="hao_basic",
                        choices=["hao_basic", "nccl"],
                        help="SP all-to-all backend.")
    parser.add_argument("--disable-non-uniform-split", action="store_true",
                        help="Disable non-uniform KVCache partitioning for load balancing (enabled by default).")
    parser.add_argument("--fixed-sp-size", type=int, default=0,
                        help="Fixed number of participating SP ranks per request (0 = disabled).")
    parser.add_argument("--dynamic-sp-size-strategy", type=str, default="legacy",
                        choices=["legacy", "bucket"],
                        help="SP size selection policy for the legacy dynamic-SP path.")
    parser.add_argument(
        "--dynamic-sp-bucket-preset",
        choices=["none", "deepseek_v3", "kimi_k2"],
        default="none",
        help="Named sequence-length bucket policy.",
    )

    parser.add_argument("--routing-strategy", type=str, default="RoundRobin", 
                        choices=["RoundRobin", "LeastBatch", "LeastCache"],
                        help="Routing strategy.")
    parser.add_argument(
        "--sp-master-selector",
        type=str,
        default="LeastBatch",
        choices=["RoundRobin", "LeastBatch", "LeastCache"],
        help="SP master-rank selection policy inside each scheduler.",
    )
    parser.add_argument("--scheduler-arch", type=str, default="legacy_global",
                        choices=["legacy_global", "hierarchical"],
                        help="Scheduler architecture (default: legacy_global).")
    parser.add_argument(
        "--router-policy",
        type=str,
        default="least_batch",
        choices=[
            "round_robin",
            "least_batch",
            "least_cache",
        ],
        help="Hierarchical load-balancer policy (default: least_batch).",
    )
    
    # Profiler arguments
    parser.add_argument("--enable-profiler", action="store_true", help="Enable profiler.")
    parser.add_argument("--profiler-start-step", type=int, default=40, help="Start profiling at this step (step-based mode).")
    parser.add_argument("--profiling-step", type=int, default=16, help="Number of steps to profile (step-based mode).")
    parser.add_argument("--profiler-dir", type=str, default="./profiler_logs", help="Directory to save profiler logs.")
    parser.add_argument("--profiler-start-time", type=float, default=None, help="Start profiling after N seconds (time-based mode).")
    parser.add_argument("--profiling-duration", type=float, default=None, help="Profile for N seconds (time-based mode).")
    parser.add_argument(
        "--diagnostic-log-interval",
        type=float,
        default=0.0,
        help=(
            "Emit a structured [BENCH_DIAG] scheduler/client snapshot every N "
            "seconds (0 disables diagnostics)."
        ),
    )
    parser.add_argument(
        "--slow-add-threshold-ms",
        type=float,
        default=0.0,
        help=(
            "Emit [BENCH_SLOW_ASYNC_SUBMIT] when a due-batch submission "
            "exceeds this latency in milliseconds (0 disables it)."
        ),
    )
    parser.add_argument(
        "--hierarchical-execution-trace",
        action="store_true",
        help=(
            "Capture detailed per-rank hierarchical execution traces. This "
            "has non-trivial overhead and is intended only for diagnosis."
        ),
    )
    parser.add_argument(
        "--hierarchical-trace-log-path",
        type=str,
        default=None,
        help="JSONL output path for --hierarchical-execution-trace.",
    )
    parser.add_argument(
        "--hierarchical-quantum-log-path",
        type=str,
        default=None,
        help=(
            "Write one compact timing/load JSON record per hierarchical "
            "engine quantum. This is substantially lighter than the full "
            "execution trace."
        ),
    )
    
    args = parser.parse_args()
    
    if args.dataset == "csv":
        if args.csv_path is None:
            parser.error("--csv-path is required when --dataset=csv")
        if not os.path.exists(args.csv_path):
            parser.error(f"CSV file not found: {args.csv_path}")
    if args.diagnostic_log_interval < 0:
        parser.error("--diagnostic-log-interval must be non-negative")
    if args.slow_add_threshold_ms < 0:
        parser.error("--slow-add-threshold-ms must be non-negative")
    if args.max_request_tokens < 0:
        parser.error("--max-request-tokens must be non-negative")
    if args.hierarchical_execution_trace:
        if args.scheduler_arch != "hierarchical":
            parser.error(
                "--hierarchical-execution-trace requires "
                "--scheduler-arch hierarchical"
            )
        if args.diagnostic_log_interval <= 0:
            parser.error(
                "--hierarchical-execution-trace requires a positive "
                "--diagnostic-log-interval so traces are drained periodically"
            )
        if not args.hierarchical_trace_log_path:
            parser.error(
                "--hierarchical-trace-log-path is required with "
                "--hierarchical-execution-trace"
            )
    if (
        args.hierarchical_quantum_log_path
        and args.scheduler_arch != "hierarchical"
    ):
        parser.error(
            "--hierarchical-quantum-log-path requires "
            "--scheduler-arch hierarchical"
        )

    return args


def get_dataset_generator(args):
    """Generates the dataset of prompts and sampling params as a generator."""
    if args.dataset == "random":
        print(f"Generating random dataset generator (input: {MAX_INPUT_LEN}, output: {MAX_OUTPUT_LEN})...")
        for _ in range(args.num_requests):
            prompt = np.random.randint(0, 10000, size=MAX_INPUT_LEN).tolist()
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_OUTPUT_LEN)
            yield prompt, sp
        return

    # CSV dataset
    print(f"Reading dataset from CSV: {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    if "prompt_len" not in df.columns or "output_len" not in df.columns:
        raise ValueError("CSV file must contain 'prompt_len' and 'output_len' columns")

    if args.max_request_tokens > 0:
        orig_len = len(df)
        request_tokens = df["prompt_len"] + df["output_len"]
        df = df[request_tokens <= args.max_request_tokens].reset_index(drop=True)
        print(
            "Filtered by "
            f"max_request_tokens={args.max_request_tokens} "
            f"(prompt_len + output_len <= limit): {orig_len} -> {len(df)} rows"
        )

    if args.max_input_len is not None:
        orig_len = len(df)
        df = df[df["prompt_len"] < args.max_input_len].reset_index(drop=True)
        print(f"Filtered by max_input_len={args.max_input_len}: {orig_len} -> {len(df)} rows")

    if df.empty:
        raise ValueError("CSV dataset has no rows after applying length filters")

    if len(df) < args.num_requests:
        print(f"Warning: CSV has {len(df)} rows, requested {args.num_requests}. Cycling data to meet request count.")
        # No need to physical concat, just cycle logic in loop
    
    # Pre-calculate cycling indices to avoid mental overhead during yield
    num_rows = len(df)
    
    for i in range(args.num_requests):
        row = df.iloc[i % num_rows]
        prompt_len = int(row["prompt_len"])
        output_len = int(row["output_len"])
        
        if prompt_len > args.max_model_len or prompt_len + output_len > args.max_model_len:
            # Reserve at least 4 tokens for prompt
            if args.max_model_len - output_len < 4:
                output_len = args.max_model_len - 4
                prompt_len = 4
            else:
                prompt_len = args.max_model_len - output_len

        prompt = np.random.randint(0, 10000, size=prompt_len).tolist()
        sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        yield prompt, sp

    print(f"Generator prepared for {args.num_requests} requests from CSV")


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
    for key, value in asdict(engine.config).items():
        print(f"{key}: {value}")
    print("=" * 40 + "\n")


def run_warmup(engine, max_num_seqs, world_size):
    """Runs warmup phase before the actual benchmark."""
    warmup_input_len = 512
    warmup_output_len = 256
    num_warmup_requests = 256
    
    print(f"\n{'=' * 60}")
    print(f"Running Warmup Phase: {num_warmup_requests} requests")
    print(f"  Input tokens: {warmup_input_len}")
    print(f"  Output tokens: {warmup_output_len}")
    print("  Fixed warmup requests: 256")
    print(f"{'=' * 60}\n")
    
    # Generate warmup requests
    # Warmup is small, list is fine
    warmup_prompts = [
        np.random.randint(0, 10000, size=warmup_input_len).tolist()
        for _ in range(num_warmup_requests)
    ]
    warmup_sampling_params = SamplingParams(
        temperature=0.6, 
        ignore_eos=True, 
        max_tokens=warmup_output_len
    )
    
    warmup_seqs = []
    for prompt in warmup_prompts:
        seq = Sequence(token_ids=prompt, sampling_params=warmup_sampling_params)
        warmup_seqs.append(seq)
    if engine.config.scheduler_arch == "hierarchical":
        engine.submit_requests_async(warmup_seqs)
    else:
        engine.add_request(warmup_seqs)
    
    # Process warmup requests
    warmup_start = time.perf_counter()
    with tqdm(total=num_warmup_requests, desc="Warmup Requests") as pbar:
        completed = 0
        while completed < num_warmup_requests:
            if engine.config.scheduler_arch == "hierarchical":
                rejected_acks = [
                    ack
                    for ack in engine.poll_ingress_acks()
                    if not ack.enqueued
                ]
                rejected_adds = [
                    result
                    for result in engine.poll_add_results()
                    if not result.accepted
                ]
                if rejected_acks or rejected_adds:
                    raise RuntimeError(
                        "hierarchical warmup request rejected: "
                        f"ingress={rejected_acks}, add={rejected_adds}"
                    )
                engine.poll_first_schedule_events()
                engine.poll_first_token_events()
            if not engine.is_finished():
                outputs, _, _, _, _ = engine.step()
                for seq_id, _ in outputs:
                    completed += 1
                    pbar.update(1)
            else:
                time.sleep(0.001)
    
    warmup_time = time.perf_counter() - warmup_start
    print(f"\nWarmup completed in {warmup_time:.2f}s")
    if warmup_time > 0:
        print(
            "Warmup effective throughput: "
            f"{num_warmup_requests / warmup_time:.2f} requests/s"
        )
    print(f"{'=' * 60}\n")


def metric_percentiles(records, key):
    values = [
        float(record[key])
        for record in records
        if not record.get("is_error") and record.get(key) is not None
    ]
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(data)), 3),
        "p50": round(float(np.percentile(data, 50)), 3),
        "p90": round(float(np.percentile(data, 90)), 3),
        "p95": round(float(np.percentile(data, 95)), 3),
        "p99": round(float(np.percentile(data, 99)), 3),
        "max": round(float(np.max(data)), 3),
    }


def _weighted_metric_percentiles(samples):
    weighted = sorted(
        (
            float(sample.itl_ms),
            int(sample.token_count),
        )
        for sample in samples
        if int(sample.token_count) > 0
    )
    if not weighted:
        return None

    values = np.asarray([item[0] for item in weighted], dtype=np.float64)
    weights = np.asarray([item[1] for item in weighted], dtype=np.int64)
    cumulative = np.cumsum(weights)
    total_weight = int(cumulative[-1])

    def percentile(percent):
        position = (total_weight - 1) * percent / 100.0
        lower = int(np.floor(position))
        upper = int(np.ceil(position))

        def value_at(index):
            value_index = int(
                np.searchsorted(cumulative, index, side="right")
            )
            return float(values[value_index])

        lower_value = value_at(lower)
        upper_value = value_at(upper)
        return lower_value + (upper_value - lower_value) * (
            position - lower
        )

    return {
        "mean": round(
            float(np.average(values, weights=weights)), 3
        ),
        "p50": round(percentile(50), 3),
        "p90": round(percentile(90), 3),
        "p95": round(percentile(95), 3),
        "p99": round(percentile(99), 3),
        "max": round(float(values[-1]), 3),
        "token_intervals": total_weight,
        "quantum_samples": len(weighted),
    }


def build_hierarchical_itl_summary(samples):
    samples = tuple(samples)
    global_stats = _weighted_metric_percentiles(samples)
    if global_stats is None:
        return None
    engine_ids = sorted({int(sample.engine_id) for sample in samples})
    return {
        "definition": (
            "LocalEngine executor duration / loop_count, weighted by "
            "generated inter-token slots"
        ),
        **global_stats,
        "per_engine": {
            str(engine_id): _weighted_metric_percentiles(
                sample
                for sample in samples
                if int(sample.engine_id) == engine_id
            )
            for engine_id in engine_ids
        },
    }


def default_metrics_summary_path(request_metrics_log_path):
    if not request_metrics_log_path:
        return None
    path = Path(request_metrics_log_path)
    if path.suffix:
        return str(path.with_suffix(".summary.json"))
    return str(path.with_name(f"{path.name}.summary.json"))


def central_final_quantum_unused_decode_ms(
    metric,
    generated_count,
    raw_service_ms,
    loop_count,
):
    if generated_count <= 0:
        return None
    if loop_count <= 0:
        raise ValueError("loop_count must be positive")
    final_real_tokens = generated_count % loop_count
    final_real_tokens = final_real_tokens or loop_count
    unused_tokens = loop_count - final_real_tokens
    if unused_tokens == 0:
        return 0.0

    itl_samples = metric.itl_samples
    if itl_samples:
        final_token_slot_ms = float(itl_samples[-1])
    elif generated_count == 1:
        # record_step_tokens intentionally has no ITL sample for the first
        # token. A one-token request has exactly one quantum, so use its raw
        # service wall time as the best available full-quantum duration.
        final_token_slot_ms = float(raw_service_ms) / loop_count
    else:
        raise RuntimeError(
            "centralized partial final quantum has no ITL sample"
        )
    return max(0.0, final_token_slot_ms) * unused_tokens


def build_request_metrics_summary(records, *, slo_threshold_ms=100.0):
    successful = [
        record for record in records if not record.get("is_error")
    ]
    tpot_values = [
        float(record["tpot_with_queue_ms"])
        for record in successful
        if record.get("tpot_with_queue_ms") is not None
    ]
    slo_success = sum(value < slo_threshold_ms for value in tpot_values)
    return {
        "total_requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "e2e_ms": metric_percentiles(records, "e2e_ms"),
        "ttft_ms": metric_percentiles(records, "ttft_ms"),
        "bootstrap_ttft_ms": metric_percentiles(
            records, "bootstrap_ttft_ms"
        ),
        "model_ttft_ms": metric_percentiles(
            records, "model_ttft_ms"
        ),
        "first_schedule_latency_ms": metric_percentiles(
            records, "first_schedule_latency_ms"
        ),
        "global_capacity_queue_ms": metric_percentiles(
            records, "global_capacity_queue_ms"
        ),
        "local_scheduler_queue_ms": metric_percentiles(
            records, "local_scheduler_queue_ms"
        ),
        "first_forward_to_terminal_ms": metric_percentiles(
            records, "first_forward_to_terminal_ms"
        ),
        "first_forward_to_terminal_real_token_ms": metric_percentiles(
            records, "first_forward_to_terminal_real_token_ms"
        ),
        "final_quantum_unused_decode_ms": metric_percentiles(
            records, "final_quantum_unused_decode_ms"
        ),
        "ttft_ms_definition": (
            "dispatch to bootstrap-ready for authoritative hierarchical "
            "admission or the legacy sequence metric; model first-token "
            "latency is reported separately when observable"
        ),
        "model_ttft_ms_definition": (
            "dispatch to the first model-generated token observed by the "
            "benchmark"
        ),
        "first_schedule_latency_ms_definition": (
            "known GPU-capacity-blocked time in the global admission FIFO "
            "plus LocalScheduler enqueue to immediately before the first "
            "executor.run; excludes initial RPC/command pickup delay and "
            "forward execution"
        ),
        "first_forward_to_terminal_ms_definition": (
            "raw scheduler execution-boundary wall time through terminal "
            "completion, including any unused tail loops in the final quantum"
        ),
        "first_forward_to_terminal_real_token_ms_definition": (
            "raw execution-boundary wall time minus final-quantum decode "
            "slots that did not produce real tokens"
        ),
        "final_quantum_unused_decode_ms_definition": (
            "per-token-slot decode time multiplied by unused slots in the "
            "final fixed-size quantum"
        ),
        "tpot_with_queue_ms": metric_percentiles(
            records, "tpot_with_queue_ms"
        ),
        "tpot_with_queue_ms_definition": (
            "(real-token execution-boundary time plus GPU-capacity queue "
            "time) / generated tokens; final-quantum loops beyond the actual "
            "output length are excluded"
        ),
        "dispatch_tpot_ms": metric_percentiles(
            records, "dispatch_tpot_ms"
        ),
        "dispatch_tpot_ms_definition": (
            "(benchmark dispatch-to-observed-terminal time minus unused "
            "final-quantum decode slots) / generated tokens; includes Router, "
            "RPC, LocalEngine ingress, LocalScheduler queue, execution, and "
            "frontend observation delay"
        ),
        "dispatch_normalized_latency_ms": metric_percentiles(
            records, "dispatch_normalized_latency_ms"
        ),
        "dispatch_normalized_latency_ms_definition": (
            "legacy benchmark-observed dispatch-to-completion time / "
            "generated tokens"
        ),
        "dispatch_lag_ms": metric_percentiles(records, "dispatch_lag_ms"),
        "ingress_ack_latency_ms": metric_percentiles(
            records, "ingress_ack_latency_ms"
        ),
        "router_pending_ms": metric_percentiles(
            records, "router_pending_ms"
        ),
        "admission_rpc_ms": metric_percentiles(
            records, "admission_rpc_ms"
        ),
        "local_command_queue_ms": metric_percentiles(
            records, "local_command_queue_ms"
        ),
        "local_admission_ms": metric_percentiles(
            records, "local_admission_ms"
        ),
        "staged_queue_ms": metric_percentiles(records, "staged_queue_ms"),
        "planned_commit_ms": metric_percentiles(
            records, "planned_commit_ms"
        ),
        "sequence_deserialize_ms": metric_percentiles(
            records, "sequence_deserialize_ms"
        ),
        "sequence_payload_bytes": metric_percentiles(
            records, "sequence_payload_bytes"
        ),
        "admission_rpc_residual_ms": metric_percentiles(
            records, "admission_rpc_residual_ms"
        ),
        "frontend_ack_overhead_ms": metric_percentiles(
            records, "frontend_ack_overhead_ms"
        ),
        "dispatch_lag_ms_definition": (
            "T0->T1: scheduled arrival to actual benchmark dispatch"
        ),
        "ingress_ack_latency_ms_definition": (
            "benchmark dispatch to Router-observed bounded staged-ingress "
            "receipt"
        ),
        "router_pending_ms_definition": (
            "RequestRouter submit to admission RPC issue, accumulated across "
            "global retries; includes any separately reported GPU-capacity "
            "queue time"
        ),
        "admission_rpc_ms_definition": (
            "frontend monotonic time across LocalEngine control RPC attempts "
            "until each staged-ingress receipt is observed; scheduler commit "
            "is reported separately by AddResultEvent"
        ),
        "local_command_queue_ms_definition": (
            "compatibility queue timing for synchronous LocalEngine admission"
        ),
        "local_admission_ms_definition": (
            "compatibility timing for synchronous LocalEngine admission"
        ),
        "staged_queue_ms_definition": (
            "positive staged-ingress receipt to the final planned-commit "
            "attempt pickup inside LocalEngine"
        ),
        "planned_commit_ms_definition": (
            "final LocalEngine planned-placement validation and commit attempt"
        ),
        "sequence_deserialize_ms_definition": (
            "server-side total Sequence decode time for the enclosing ingress "
            "batch"
        ),
        "sequence_payload_bytes_definition": (
            "serialized Sequence payload bytes for this request"
        ),
        "admission_rpc_residual_ms_definition": (
            "admission RPC time minus final local command-queue and admission "
            "work; includes Ray actor mailbox/transport/polling and any prior "
            "fallback attempts"
        ),
        "frontend_ack_overhead_ms_definition": (
            "dispatch-to-receipt minus router-pending and receipt-RPC time; "
            "covers frontend submit and receipt observation overhead"
        ),
        "add_accept_latency_ms": metric_percentiles(
            records, "add_accept_latency_ms"
        ),
        "goodput": {
            "metric": "tpot_with_queue_ms",
            "threshold_ms": slo_threshold_ms,
            "successful_requests": slo_success,
            "eligible_requests": len(tpot_values),
            "attainment_percent": round(
                100.0 * slo_success / len(tpot_values), 3
            )
            if tpot_values
            else 0.0,
        },
    }


def run_benchmark(
    engine,
    request_generator,
    arrival_times,
    num_requests,
    *,
    diagnostic_log_interval=0.0,
    slow_add_threshold_ms=0.0,
    hierarchical_trace_log_path=None,
    hierarchical_quantum_log_path=None,
    request_metrics_log_path=None,
    metrics_summary_path=None,
    clock_ns=None,
    sleep_fn=None,
    show_progress=True,
):
    """Runs the main benchmark loop with rate-controlled request submission."""
    clock_ns = clock_ns or time.perf_counter_ns
    sleep_fn = sleep_fn or time.sleep
    seq_map = {}
    completed_latencies = []

    # Pre-load all request data without submitting
    print(f"Preparing {num_requests} requests for rate-controlled submission...")
    all_seqs = []
    for _ in range(num_requests):
        try:
            prompt, sp = next(request_generator)
        except StopIteration:
            break
        seq = Sequence(token_ids=prompt, sampling_params=sp)
        all_seqs.append(seq)
        seq_map[seq.seq_id] = seq

    requests_to_send = len(all_seqs)
    print(f"Prepared {requests_to_send} requests. Submitting according to arrival_times (rate-controlled).")

    if len(arrival_times) < requests_to_send:
        raise ValueError("arrival_times is shorter than the prepared requests")

    next_idx = 0
    start_ns = clock_ns()
    start_time = start_ns / 1_000_000_000
    start_wall_time = time.time()
    dispatched = 0
    ingress_enqueued = 0
    ingress_rejected = 0
    accepted = 0
    scheduler_rejected = 0
    failed = 0
    completed = 0
    dispatch_lag_ms = []
    ingress_ack_latency_ms = []
    add_accept_latency_ms = []
    observed_ttft_ms = []
    observed_e2e_ms = []
    request_times = {
        seq.seq_id: {
            "scheduled_ns": start_ns
            + int(float(arrival_times[index]) * 1_000_000_000)
        }
        for index, seq in enumerate(all_seqs)
    }
    terminal_ids = set()
    rejected_ids = set()
    last_dispatch_ns = start_ns
    slow_add_count = 0
    last_diag_time = start_time
    previous_hierarchical = None
    previous_hierarchical_time = start_time
    trace_file = None
    quantum_diagnostic_file = None
    quantum_diagnostic_count = 0
    request_metrics_file = None
    request_metric_records = []
    recorded_request_ids = set()

    if request_metrics_log_path:
        request_metrics_path = Path(request_metrics_log_path).expanduser()
        request_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        request_metrics_file = request_metrics_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        request_metrics_log_path = str(request_metrics_path)
        if metrics_summary_path is None:
            metrics_summary_path = default_metrics_summary_path(
                request_metrics_log_path
            )

    if hierarchical_quantum_log_path:
        quantum_path = Path(
            hierarchical_quantum_log_path
        ).expanduser()
        quantum_path.parent.mkdir(parents=True, exist_ok=True)
        quantum_diagnostic_file = quantum_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        hierarchical_quantum_log_path = str(quantum_path)

    def _interval_ms(timing, end_key, start_key):
        end_ns = timing.get(end_key)
        start_value_ns = timing.get(start_key)
        if end_ns is None or start_value_ns is None:
            return None
        return round((end_ns - start_value_ns) / 1_000_000, 6)

    def append_request_record(record):
        request_id = record["request_id"]
        if request_id in recorded_request_ids:
            raise RuntimeError(
                f"duplicate request metric record {request_id}"
            )
        recorded_request_ids.add(request_id)
        request_metric_records.append(record)
        if request_metrics_file is not None:
            request_metrics_file.write(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            request_metrics_file.flush()

    def build_request_record(
        *,
        request_id,
        engine_id,
        status,
        actual_output_tokens,
        observed_ns,
        finish_event=None,
        error_message=None,
    ):
        timing = request_times[request_id]
        seq = seq_map[request_id]
        timing["completion_ns"] = observed_ns
        e2e_ms = _interval_ms(timing, "completion_ns", "dispatch_ns")
        arrival_e2e_ms = _interval_ms(
            timing, "completion_ns", "scheduled_ns"
        )
        model_ttft_ms = _interval_ms(
            timing, "first_token_ns", "dispatch_ns"
        )
        bootstrap_ttft_ms = _interval_ms(
            timing, "bootstrap_ready_ns", "dispatch_ns"
        )
        ttft_source = None
        if bootstrap_ttft_ms is not None:
            ttft_ms = bootstrap_ttft_ms
            ttft_source = timing.get(
                "bootstrap_ready_source", "authoritative_admission_ack"
            )
        elif seq.metric is not None:
            metric_ttft = seq.metric.ttft
            if metric_ttft is not None:
                bootstrap_ttft_ms = round(float(metric_ttft), 6)
                ttft_ms = bootstrap_ttft_ms
                ttft_source = "sequence_metric"
            else:
                ttft_ms = model_ttft_ms
                if ttft_ms is not None:
                    ttft_source = "model_first_token_fallback"
        else:
            ttft_ms = model_ttft_ms
            if ttft_ms is not None:
                ttft_source = "model_first_token_fallback"
        first_schedule_latency_ms = timing.get(
            "first_schedule_latency_ms"
        )
        global_capacity_queue_ms = timing.get(
            "global_capacity_queue_ms"
        )
        local_scheduler_queue_ms = timing.get(
            "local_scheduler_queue_ms"
        )
        first_schedule_latency_source = None
        if first_schedule_latency_ms is not None:
            first_schedule_latency_ms = round(
                float(first_schedule_latency_ms), 6
            )
            global_capacity_queue_ms = round(
                float(global_capacity_queue_ms or 0.0), 6
            )
            local_scheduler_queue_ms = round(
                float(local_scheduler_queue_ms or 0.0), 6
            )
            first_schedule_latency_source = (
                "hierarchical_first_forward_event"
            )
        elif (
            engine.config.scheduler_arch != "hierarchical"
            and seq.metric is not None
        ):
            queueing_time_ms = seq.metric.queueing_time_ms
            if queueing_time_ms is not None:
                first_schedule_latency_ms = round(
                    float(queueing_time_ms), 6
                )
                first_schedule_latency_source = (
                    "sequence_metric_queueing_time"
                )
                global_capacity_queue_ms = first_schedule_latency_ms
                local_scheduler_queue_ms = 0.0
        dispatch_normalized_latency_ms = (
            round(e2e_ms / actual_output_tokens, 6)
            if e2e_ms is not None and actual_output_tokens > 0
            else None
        )
        ingress_ack_latency_ms = _interval_ms(
            timing, "ingress_ack_ns", "dispatch_ns"
        )
        router_pending_ms = timing.get("router_pending_ms")
        admission_rpc_ms = timing.get("admission_rpc_ms")
        local_command_queue_ms = timing.get("local_command_queue_ms")
        local_admission_ms = timing.get("local_admission_ms")
        staged_queue_ms = timing.get("staged_queue_ms")
        planned_commit_ms = timing.get("planned_commit_ms")
        sequence_deserialize_ms = timing.get("sequence_deserialize_ms")
        sequence_payload_bytes = timing.get("sequence_payload_bytes")
        admission_rpc_residual_ms = None
        if (
            admission_rpc_ms is not None
            and local_command_queue_ms is not None
            and local_admission_ms is not None
        ):
            admission_rpc_residual_ms = round(
                float(admission_rpc_ms)
                - float(local_command_queue_ms)
                - float(local_admission_ms),
                6,
            )
        frontend_ack_overhead_ms = None
        if (
            ingress_ack_latency_ms is not None
            and router_pending_ms is not None
            and admission_rpc_ms is not None
        ):
            frontend_ack_overhead_ms = round(
                float(ingress_ack_latency_ms)
                - float(router_pending_ms)
                - float(admission_rpc_ms),
                6,
            )
        first_forward_to_terminal_ms = None
        first_forward_to_terminal_real_token_ms = None
        final_quantum_real_tokens = None
        final_quantum_unused_decode_ms = None
        tpot_with_queue_ms = None
        tpot_with_queue_source = None
        dispatch_tpot_ms = None
        if finish_event is not None:
            first_forward_to_terminal_ms = (
                finish_event.first_forward_to_terminal_ms
            )
            if first_forward_to_terminal_ms is not None:
                first_forward_to_terminal_ms = round(
                    float(first_forward_to_terminal_ms), 6
                )

        if status == "FINISHED" and actual_output_tokens > 0:
            if engine.config.scheduler_arch == "hierarchical":
                if finish_event is None:
                    raise RuntimeError(
                        "hierarchical FINISHED request is missing its "
                        f"terminal event: request_id={request_id}"
                    )
                if first_forward_to_terminal_ms is None:
                    raise RuntimeError(
                        "hierarchical FINISHED request is missing "
                        "first-forward-to-terminal timing: "
                        f"request_id={request_id}"
                    )
                terminal_capacity_queue_ms = round(
                    float(finish_event.global_capacity_queue_ms), 6
                )
                if (
                    global_capacity_queue_ms is not None
                    and abs(
                        float(global_capacity_queue_ms)
                        - terminal_capacity_queue_ms
                    )
                    > 0.001
                ):
                    raise RuntimeError(
                        "first-schedule and terminal capacity queue metrics "
                        f"disagree for request {request_id}"
                    )
                global_capacity_queue_ms = terminal_capacity_queue_ms
                final_quantum_real_tokens = (
                    finish_event.final_quantum_real_tokens
                )
                final_quantum_unused_decode_ms = (
                    finish_event.final_quantum_unused_decode_ms
                )
                first_forward_to_terminal_real_token_ms = (
                    finish_event.first_forward_to_terminal_real_token_ms
                )
                if (
                    final_quantum_unused_decode_ms is None
                    or first_forward_to_terminal_real_token_ms is None
                ):
                    raise RuntimeError(
                        "hierarchical FINISHED request is missing final "
                        "quantum execution timing: "
                        f"request_id={request_id}, "
                        f"generated_count={actual_output_tokens}"
                    )
                tpot_with_queue_ms = round(
                    (
                        first_forward_to_terminal_real_token_ms
                        + global_capacity_queue_ms
                    )
                    / actual_output_tokens,
                    6,
                )
                tpot_with_queue_source = (
                    "hierarchical_real_token_execution_boundary"
                )
            else:
                if seq.metric is None:
                    raise RuntimeError(
                        "centralized FINISHED request is missing its "
                        f"sequence metric: request_id={request_id}"
                    )
                metric_tpot = seq.metric.avg_tpot_with_queueing
                metric_tpot_without_queue = (
                    seq.metric.avg_tpot_wo_queueing
                )
                if (
                    metric_tpot is None
                    or metric_tpot_without_queue is None
                ):
                    raise RuntimeError(
                        "centralized FINISHED request has incomplete "
                        f"scheduling metrics: request_id={request_id}"
                    )
                first_forward_to_terminal_ms = round(
                    float(metric_tpot_without_queue)
                    * actual_output_tokens,
                    6,
                )
                central_loop_count = int(
                    getattr(
                        engine.config,
                        "loop_count",
                        HIERARCHICAL_LOOP_COUNT,
                    )
                )
                final_quantum_real_tokens = (
                    actual_output_tokens % central_loop_count
                    or central_loop_count
                )
                final_quantum_unused_decode_ms = (
                    central_final_quantum_unused_decode_ms(
                        seq.metric,
                        actual_output_tokens,
                        first_forward_to_terminal_ms,
                        central_loop_count,
                    )
                )
                first_forward_to_terminal_real_token_ms = max(
                    0.0,
                    first_forward_to_terminal_ms
                    - final_quantum_unused_decode_ms,
                )
                raw_with_queue_ms = (
                    float(metric_tpot) * actual_output_tokens
                )
                tpot_with_queue_ms = round(
                    max(
                        0.0,
                        raw_with_queue_ms
                        - final_quantum_unused_decode_ms,
                    )
                    / actual_output_tokens,
                    6,
                )
                tpot_with_queue_source = (
                    "sequence_metric_real_token_execution_boundary"
                )
        if (
            status == "FINISHED"
            and actual_output_tokens > 0
            and e2e_ms is not None
            and final_quantum_unused_decode_ms is not None
        ):
            dispatch_tpot_ms = round(
                max(
                    0.0,
                    float(e2e_ms)
                    - float(final_quantum_unused_decode_ms),
                )
                / actual_output_tokens,
                6,
            )
        return {
            "schema_version": 3,
            "request_id": int(request_id),
            "engine_id": int(engine_id),
            "status": status,
            "is_error": status != "FINISHED",
            "error_message": error_message,
            "prompt_tokens": int(seq.num_prompt_tokens),
            "expected_output_tokens": int(seq.max_tokens),
            "actual_output_tokens": int(actual_output_tokens),
            "dispatch_offset_ms": round(
                (
                    timing.get("dispatch_ns", timing["scheduled_ns"])
                    - start_ns
                )
                / 1_000_000,
                6,
            ),
            "completion_offset_ms": round(
                (observed_ns - start_ns) / 1_000_000,
                6,
            ),
            "dispatch_lag_ms": _interval_ms(
                timing, "dispatch_ns", "scheduled_ns"
            ),
            "ingress_ack_latency_ms": ingress_ack_latency_ms,
            "router_pending_ms": router_pending_ms,
            "admission_rpc_ms": admission_rpc_ms,
            "local_command_queue_ms": local_command_queue_ms,
            "local_admission_ms": local_admission_ms,
            "staged_queue_ms": staged_queue_ms,
            "planned_commit_ms": planned_commit_ms,
            "sequence_deserialize_ms": sequence_deserialize_ms,
            "sequence_payload_bytes": sequence_payload_bytes,
            "admission_rpc_residual_ms": admission_rpc_residual_ms,
            "frontend_ack_overhead_ms": frontend_ack_overhead_ms,
            "add_accept_latency_ms": _interval_ms(
                timing, "add_result_ns", "dispatch_ns"
            ),
            "ttft_ms": ttft_ms,
            "ttft_source": ttft_source,
            "bootstrap_ttft_ms": bootstrap_ttft_ms,
            "model_ttft_ms": model_ttft_ms,
            "first_schedule_latency_ms": first_schedule_latency_ms,
            "first_schedule_latency_source": (
                first_schedule_latency_source
            ),
            "global_capacity_queue_ms": global_capacity_queue_ms,
            "local_scheduler_queue_ms": local_scheduler_queue_ms,
            "first_forward_to_terminal_ms": (
                first_forward_to_terminal_ms
            ),
            "first_forward_to_terminal_real_token_ms": (
                round(
                    float(first_forward_to_terminal_real_token_ms),
                    6,
                )
                if first_forward_to_terminal_real_token_ms is not None
                else None
            ),
            "final_quantum_real_tokens": final_quantum_real_tokens,
            "final_quantum_unused_decode_ms": (
                round(float(final_quantum_unused_decode_ms), 6)
                if final_quantum_unused_decode_ms is not None
                else None
            ),
            "e2e_ms": e2e_ms,
            "arrival_e2e_ms": arrival_e2e_ms,
            "tpot_with_queue_ms": tpot_with_queue_ms,
            "tpot_with_queue_source": tpot_with_queue_source,
            "dispatch_tpot_ms": dispatch_tpot_ms,
            "dispatch_normalized_latency_ms": (
                dispatch_normalized_latency_ms
            ),
        }

    if hierarchical_trace_log_path:
        trace_dir = os.path.dirname(
            os.path.abspath(hierarchical_trace_log_path)
        )
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = open(
            hierarchical_trace_log_path,
            "w",
            encoding="utf-8",
            buffering=1,
        )

    def drain_quantum_diagnostics():
        nonlocal quantum_diagnostic_count
        if quantum_diagnostic_file is None:
            return 0
        samples = engine.drain_hierarchical_quantum_diagnostics()
        for sample in samples:
            record = dict(sample)
            record["benchmark_elapsed_s"] = round(
                float(record["started_at_unix_s"]) - start_wall_time,
                6,
            )
            quantum_diagnostic_file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        quantum_diagnostic_file.flush()
        quantum_diagnostic_count += len(samples)
        return len(samples)

    def emit_diagnostic(*, force=False):
        nonlocal last_diag_time
        nonlocal previous_hierarchical
        nonlocal previous_hierarchical_time

        if diagnostic_log_interval <= 0:
            return
        current_time = clock_ns() / 1_000_000_000
        if (
            not force
            and current_time - last_diag_time < diagnostic_log_interval
        ):
            return

        elapsed = current_time - start_time
        payload = {
            "elapsed_s": round(elapsed, 3),
            "requests_total": requests_to_send,
            "scheduled_requests": requests_to_send,
            "dispatched_requests": dispatched,
            "ingress_enqueued": ingress_enqueued,
            "ingress_rejected": ingress_rejected,
            "scheduler_accepted": accepted,
            "scheduler_rejected": scheduler_rejected,
            "failed_requests": failed,
            "client_unsent": requests_to_send - next_idx,
            "completed": completed,
            "client_outstanding": accepted - completed,
            "pending_ingress": engine.num_pending_ingress,
            "pending_add": engine.num_pending_adds,
            "slow_add_count": slow_add_count,
        }

        if engine.config.scheduler_arch == "hierarchical":
            hierarchical = engine.hierarchical_metrics(
                refresh=False,
                include_per_engine=True,
            )
            if not hierarchical:
                payload["hierarchical"] = {}
                print(
                    "[BENCH_DIAG] "
                    + json.dumps(
                        payload, sort_keys=True, separators=(",", ":")
                    ),
                    flush=True,
                )
                last_diag_time = current_time
                return
            hierarchical_interval = current_time - previous_hierarchical_time
            if (
                previous_hierarchical is not None
                and hierarchical_interval > 0
            ):
                quantum_delta = (
                    hierarchical["decode_quantum_count"]
                    - previous_hierarchical["decode_quantum_count"]
                )
                useful_token_delta = (
                    hierarchical["useful_decode_tokens"]
                    - previous_hierarchical["useful_decode_tokens"]
                )
                execute_ms_delta = (
                    hierarchical["execute_latency_ms_total"]
                    - previous_hierarchical["execute_latency_ms_total"]
                )
                payload["hierarchical_interval"] = {
                    "seconds": round(hierarchical_interval, 3),
                    "decode_quantums": quantum_delta,
                    "useful_decode_tokens": useful_token_delta,
                    "useful_decode_tokens_per_s": round(
                        useful_token_delta / hierarchical_interval, 3
                    ),
                    "avg_execute_ms_per_engine_quantum": round(
                        execute_ms_delta / quantum_delta, 3
                    )
                    if quantum_delta
                    else None,
                    "preemptions": (
                        hierarchical["preemption_count"]
                        - previous_hierarchical["preemption_count"]
                    ),
                }
                itl_token_delta = (
                    hierarchical["decode_itl_token_count"]
                    - previous_hierarchical["decode_itl_token_count"]
                )
                itl_weighted_ms_delta = (
                    hierarchical["decode_itl_ms_weighted_total"]
                    - previous_hierarchical[
                        "decode_itl_ms_weighted_total"
                    ]
                )
                payload["hierarchical_interval"][
                    "decode_itl_ms_mean"
                ] = (
                    round(
                        itl_weighted_ms_delta / itl_token_delta,
                        3,
                    )
                    if itl_token_delta > 0
                    else None
                )
            payload["hierarchical"] = hierarchical
            previous_hierarchical = hierarchical
            previous_hierarchical_time = current_time
            if quantum_diagnostic_file is not None:
                payload["hierarchical_quantum_records_drained"] = (
                    drain_quantum_diagnostics()
                )

        if trace_file is not None:
            traces = engine.drain_execution_traces()
            for trace in traces:
                trace_file.write(
                    json.dumps(trace, separators=(",", ":")) + "\n"
                )
            trace_file.flush()
            payload["hierarchical_trace_records_drained"] = len(traces)

        print(
            "[BENCH_DIAG] "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        last_diag_time = current_time

    try:
        with tqdm(
            total=requests_to_send,
            desc="Processing Requests",
            disable=not show_progress,
        ) as pbar:
            while True:
                now_ns = clock_ns()
                elapsed_ns = now_ns - start_ns
                due_begin = next_idx
                while (
                    next_idx < requests_to_send
                    and int(
                        float(arrival_times[next_idx]) * 1_000_000_000
                    )
                    <= elapsed_ns
                ):
                    next_idx += 1

                if next_idx > due_begin:
                    due = all_seqs[due_begin:next_idx]
                    dispatch_ns = clock_ns()
                    for seq in due:
                        timing = request_times[seq.seq_id]
                        timing["dispatch_ns"] = dispatch_ns
                        dispatch_lag_ms.append(
                            (
                                dispatch_ns - timing["scheduled_ns"]
                            )
                            / 1_000_000
                        )
                    submit_begin_ns = clock_ns()
                    engine.submit_requests_async(due)
                    submit_latency_ms = (
                        clock_ns() - submit_begin_ns
                    ) / 1_000_000
                    dispatched += len(due)
                    last_dispatch_ns = dispatch_ns
                    if (
                        slow_add_threshold_ms > 0
                        and submit_latency_ms >= slow_add_threshold_ms
                    ):
                        slow_add_count += len(due)
                        print(
                            "[BENCH_SLOW_ASYNC_SUBMIT] "
                            + json.dumps(
                                {
                                    "elapsed_s": round(
                                        (dispatch_ns - start_ns)
                                        / 1_000_000_000,
                                        3,
                                    ),
                                    "request_index_begin": due_begin,
                                    "request_index_end": next_idx,
                                    "batch_size": len(due),
                                    "latency_ms": round(
                                        submit_latency_ms, 3
                                    ),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )

                for ack in engine.poll_ingress_acks():
                    observed_ns = clock_ns()
                    timing = request_times[ack.request_id]
                    timing["ingress_ack_ns"] = observed_ns
                    for metric_name in (
                        "router_pending_ms",
                        "admission_rpc_ms",
                        "local_command_queue_ms",
                        "local_admission_ms",
                        "sequence_deserialize_ms",
                        "sequence_payload_bytes",
                    ):
                        metric_value = getattr(ack, metric_name)
                        if metric_value is not None:
                            timing[metric_name] = round(
                                float(metric_value), 6
                            )
                    if (
                        ack.enqueued
                        and ack.admission_version is not None
                        and "bootstrap_ready_ns" not in timing
                    ):
                        # Centralized-compatible hierarchical admission only
                        # ACKs after the local planner commits placement and
                        # appends the bootstrap token. Use that authoritative
                        # ACK as T3 so ttft_ms matches legacy dummy-prefill
                        # semantics. The actual model token remains available
                        # separately as model_ttft_ms.
                        timing["bootstrap_ready_ns"] = observed_ns
                        timing["bootstrap_ready_source"] = (
                            "authoritative_admission_ack"
                        )
                    ingress_ack_latency_ms.append(
                        (observed_ns - timing["dispatch_ns"]) / 1_000_000
                    )
                    if ack.enqueued:
                        ingress_enqueued += 1
                        continue
                    ingress_rejected += 1
                    rejected_ids.add(ack.request_id)
                    append_request_record(
                        build_request_record(
                            request_id=ack.request_id,
                            engine_id=ack.engine_id,
                            status="INGRESS_REJECTED",
                            actual_output_tokens=0,
                            observed_ns=observed_ns,
                            error_message=ack.reason,
                        )
                    )
                    pbar.update(1)
                    print(
                        "[BENCH_INGRESS_REJECTED] "
                        + json.dumps(
                            {
                                "request_id": ack.request_id,
                                "engine_id": ack.engine_id,
                                "reason": ack.reason,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

                for result in engine.poll_add_results():
                    observed_ns = clock_ns()
                    timing = request_times[result.request_id]
                    timing["add_result_ns"] = observed_ns
                    if result.local_planned_queue_ms is not None:
                        timing["staged_queue_ms"] = round(
                            float(result.local_planned_queue_ms), 6
                        )
                    if result.local_admission_ms is not None:
                        timing["planned_commit_ms"] = round(
                            float(result.local_admission_ms), 6
                        )
                    add_accept_latency_ms.append(
                        (observed_ns - timing["dispatch_ns"]) / 1_000_000
                    )
                    if result.accepted:
                        if (
                            result.admission_version is not None
                            and "bootstrap_ready_ns" not in timing
                        ):
                            timing["bootstrap_ready_ns"] = observed_ns
                            timing["bootstrap_ready_source"] = (
                                "planned_add_result"
                            )
                        accepted += 1
                        continue
                    scheduler_rejected += 1
                    rejected_ids.add(result.request_id)
                    append_request_record(
                        build_request_record(
                            request_id=result.request_id,
                            engine_id=result.engine_id,
                            status="ADD_REJECTED",
                            actual_output_tokens=0,
                            observed_ns=observed_ns,
                            error_message=result.reason,
                        )
                    )
                    pbar.update(1)
                    print(
                        "[BENCH_ADD_REJECTED] "
                        + json.dumps(
                            {
                                "request_id": result.request_id,
                                "engine_id": result.engine_id,
                                "reason": result.reason,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

                for event in engine.poll_first_token_events():
                    observed_ns = clock_ns()
                    timing = request_times[event.request_id]
                    if "first_token_ns" not in timing:
                        timing["first_token_ns"] = observed_ns
                        timing["first_token_generated_count"] = (
                            event.generated_count
                        )
                        observed_ttft_ms.append(
                            (
                                observed_ns - timing["scheduled_ns"]
                            )
                            / 1_000_000
                        )

                for event in engine.poll_first_schedule_events():
                    timing = request_times[event.request_id]
                    if "first_schedule_latency_ms" in timing:
                        raise RuntimeError(
                            "duplicate first-schedule event for request "
                            f"{event.request_id}"
                        )
                    timing["first_schedule_latency_ms"] = (
                        event.first_schedule_latency_ms
                    )
                    timing["global_capacity_queue_ms"] = (
                        event.global_capacity_queue_ms
                    )
                    timing["local_scheduler_queue_ms"] = (
                        event.local_scheduler_queue_ms
                    )

                if engine.config.scheduler_arch == "hierarchical":
                    finish_events = engine.poll()
                else:
                    finish_events = []
                    if not engine.is_finished():
                        outputs, _, _, _, _ = engine.step()
                        finish_events = [
                            FinishEvent(
                                request_id=seq_id,
                                generated_count=len(token_ids),
                                status="FINISHED",
                                engine_id=-1,
                            )
                            for seq_id, token_ids in outputs
                        ]

                for event in finish_events:
                    request_id = event.request_id
                    if (
                        request_id in terminal_ids
                        or request_id in rejected_ids
                    ):
                        raise RuntimeError(
                            f"duplicate terminal request {request_id}"
                        )
                    terminal_ids.add(request_id)
                    observed_ns = clock_ns()
                    timing = request_times[request_id]
                    request_record = build_request_record(
                        request_id=request_id,
                        engine_id=event.engine_id,
                        status=event.status,
                        actual_output_tokens=event.generated_count,
                        observed_ns=observed_ns,
                        finish_event=event,
                        error_message=(
                            None
                            if event.status == "FINISHED"
                            else event.status
                        ),
                    )
                    append_request_record(request_record)
                    if event.status != "FINISHED":
                        failed += 1
                        pbar.update(1)
                        continue

                    completed += 1
                    observed_e2e_ms.append(
                        (observed_ns - timing["scheduled_ns"]) / 1_000_000
                    )
                    seq = seq_map[request_id]
                    if seq.metric and seq.metric.e2e_latency:
                        completed_latencies.append(
                            seq.metric.e2e_latency / 1000
                        )
                        pbar.set_postfix(
                            {
                                "Avg Latency": (
                                    f"{np.mean(completed_latencies):.2f}s"
                                )
                            }
                        )
                    pbar.update(1)

                emit_diagnostic()
                if (
                    next_idx == requests_to_send
                    and engine.num_pending_ingress == 0
                    and engine.num_pending_adds == 0
                    and engine.is_finished()
                ):
                    break

                now_ns = clock_ns()
                poll_deadline_ns = now_ns + 2_000_000
                if next_idx < requests_to_send:
                    next_arrival_ns = (
                        start_ns
                        + int(
                            float(arrival_times[next_idx])
                            * 1_000_000_000
                        )
                    )
                    poll_deadline_ns = min(
                        poll_deadline_ns, next_arrival_ns
                    )
                wait_ns = poll_deadline_ns - clock_ns()
                if wait_ns > 0:
                    sleep_fn(wait_ns / 1_000_000_000)
        emit_diagnostic(force=True)
    except KeyboardInterrupt:
        print("[BENCH_INTERRUPTED] user requested stop", flush=True)
        emit_diagnostic(force=True)
        raise
    finally:
        drain_quantum_diagnostics()
        if trace_file is not None:
            trace_file.close()
        if quantum_diagnostic_file is not None:
            quantum_diagnostic_file.close()
        if request_metrics_file is not None:
            request_metrics_file.close()

    total_time = (clock_ns() - start_ns) / 1_000_000_000
    classified = completed + len(rejected_ids) + failed
    if classified != dispatched:
        raise RuntimeError(
            "request accounting did not close: "
            f"dispatched={dispatched}, completed={completed}, "
            f"rejected={len(rejected_ids)}, failed={failed}"
        )
    if len(request_metric_records) != classified:
        raise RuntimeError(
            "request metric accounting did not close: "
            f"records={len(request_metric_records)}, "
            f"classified={classified}"
        )

    metrics_summary = build_request_metrics_summary(
        request_metric_records
    )
    hierarchical_itl_summary = None
    hierarchical_rank_loads = None
    hierarchical_worker_transport = None
    hierarchical_async_depth = None
    if engine.config.scheduler_arch == "hierarchical":
        hierarchical_worker_transport = getattr(
            engine.config, "hierarchical_worker_transport", "ray"
        )
        metrics_summary["hierarchical_worker_transport"] = (
            hierarchical_worker_transport
        )
        hierarchical_async_depth = getattr(
            engine.config, "hierarchical_async_depth", 1
        )
        metrics_summary["hierarchical_async_depth"] = (
            hierarchical_async_depth
        )
        hierarchical_itl_summary = build_hierarchical_itl_summary(
            engine.hierarchical_itl_samples()
        )
        if hierarchical_itl_summary is not None:
            metrics_summary["hierarchical_decode_itl_ms"] = (
                hierarchical_itl_summary
            )
    if engine.config.scheduler_arch == "hierarchical":
        final_hierarchical = engine.hierarchical_metrics(
            refresh=True,
            include_per_engine=True,
        )
        hierarchical_rank_loads = {
            engine_id: engine_metrics.get("rank_loads", [])
            for engine_id, engine_metrics in final_hierarchical.get(
                "per_engine", {}
            ).items()
        }
        metrics_summary["hierarchical_rank_loads"] = (
            hierarchical_rank_loads
        )
    execution_boundary_metrics = engine.execution_boundary_metrics()
    metrics_summary["execution_boundary_metrics"] = (
        execution_boundary_metrics
    )
    metrics_summary.update(
        {
            "benchmark_runtime_s": round(total_time, 6),
            "request_metrics_jsonl": request_metrics_log_path,
            "hierarchical_quantum_diagnostics_jsonl": (
                hierarchical_quantum_log_path
            ),
            "hierarchical_quantum_diagnostic_samples": (
                quantum_diagnostic_count
            ),
        }
    )
    if metrics_summary_path:
        summary_path = Path(metrics_summary_path).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_summary_path = str(summary_path)

    def latency_stats(values):
        if not values:
            return {}
        return {
            "avg": round(float(np.mean(values)), 3),
            "p50": round(float(np.percentile(values, 50)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "p99": round(float(np.percentile(values, 99)), 3),
            "max": round(float(np.max(values)), 3),
        }

    dispatch_window_s = max(
        (last_dispatch_ns - start_ns) / 1_000_000_000, 1e-9
    )
    result = {
        "scheduled_requests": requests_to_send,
        "dispatched_requests": dispatched,
        "ingress_enqueued": ingress_enqueued,
        "ingress_rejected": ingress_rejected,
        "scheduler_accepted": accepted,
        "scheduler_rejected": scheduler_rejected,
        "completed_requests": completed,
        "failed_requests": failed,
        "pending_ingress": engine.num_pending_ingress,
        "pending_add": engine.num_pending_adds,
        "active_requests": (
            engine.router.active_count
            if engine.config.scheduler_arch == "hierarchical"
            else 0
        ),
        "achieved_dispatch_rate": round(
            dispatched / dispatch_window_s, 6
        ),
        "dispatch_lag_ms": latency_stats(dispatch_lag_ms),
        "ingress_ack_latency_ms": latency_stats(
            ingress_ack_latency_ms
        ),
        "router_pending_ms": metrics_summary["router_pending_ms"],
        "admission_rpc_ms": metrics_summary["admission_rpc_ms"],
        "local_command_queue_ms": metrics_summary[
            "local_command_queue_ms"
        ],
        "local_admission_ms": metrics_summary["local_admission_ms"],
        "staged_queue_ms": metrics_summary["staged_queue_ms"],
        "planned_commit_ms": metrics_summary["planned_commit_ms"],
        "sequence_deserialize_ms": metrics_summary[
            "sequence_deserialize_ms"
        ],
        "sequence_payload_bytes": metrics_summary[
            "sequence_payload_bytes"
        ],
        "admission_rpc_residual_ms": metrics_summary[
            "admission_rpc_residual_ms"
        ],
        "frontend_ack_overhead_ms": metrics_summary[
            "frontend_ack_overhead_ms"
        ],
        "add_accept_latency_ms": latency_stats(
            add_accept_latency_ms
        ),
        "observed_ttft_ms": latency_stats(observed_ttft_ms),
        "observed_e2e_ms": latency_stats(observed_e2e_ms),
        "ttft_ms": metrics_summary["ttft_ms"],
        "bootstrap_ttft_ms": metrics_summary["bootstrap_ttft_ms"],
        "model_ttft_ms": metrics_summary["model_ttft_ms"],
        "first_schedule_latency_ms": metrics_summary[
            "first_schedule_latency_ms"
        ],
        "global_capacity_queue_ms": metrics_summary[
            "global_capacity_queue_ms"
        ],
        "local_scheduler_queue_ms": metrics_summary[
            "local_scheduler_queue_ms"
        ],
        "first_forward_to_terminal_ms": metrics_summary[
            "first_forward_to_terminal_ms"
        ],
        "e2e_ms": metrics_summary["e2e_ms"],
        "tpot_with_queue_ms": metrics_summary["tpot_with_queue_ms"],
        "dispatch_tpot_ms": metrics_summary["dispatch_tpot_ms"],
        "dispatch_normalized_latency_ms": metrics_summary[
            "dispatch_normalized_latency_ms"
        ],
        "goodput": metrics_summary["goodput"],
        "request_metrics_jsonl": request_metrics_log_path,
        "metrics_summary_json": metrics_summary_path,
        "hierarchical_quantum_diagnostics_jsonl": (
            hierarchical_quantum_log_path
        ),
        "hierarchical_quantum_diagnostic_samples": (
            quantum_diagnostic_count
        ),
        "hierarchical_worker_transport": (
            hierarchical_worker_transport
        ),
        "hierarchical_async_depth": hierarchical_async_depth,
    }
    if hierarchical_itl_summary is not None:
        result["hierarchical_decode_itl_ms"] = (
            hierarchical_itl_summary
        )
    if hierarchical_rank_loads is not None:
        result["hierarchical_rank_loads"] = hierarchical_rank_loads
    if execution_boundary_metrics is not None:
        result["execution_boundary_metrics"] = execution_boundary_metrics
    print(
        "[BENCH_RESULT] "
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return total_time, seq_map, metrics_summary


def calculate_and_print_metrics(
    total_time,
    seq_map,
    requests_sent,
    metrics_summary,
    itl_log_path=None,
):
    """Calculates and prints performance metrics."""
    completed_seqs = [s for s in seq_map.values() if s.metric and s.metric.completion_time]
    total_seqs = len(completed_seqs)
    
    total_input = sum(s.metric.num_prompt_tokens for s in completed_seqs)
    total_output = sum(s.metric.num_generated_tokens for s in completed_seqs)
    
    throughput = total_output / total_time
    
    ttft_stats = metrics_summary["ttft_ms"] or {}
    model_ttft_stats = metrics_summary["model_ttft_ms"] or {}
    e2e_stats = metrics_summary["e2e_ms"] or {}
    tpot_wq_stats = metrics_summary["tpot_with_queue_ms"] or {}
    dispatch_tpot_stats = metrics_summary["dispatch_tpot_ms"] or {}
    dispatch_normalized_stats = (
        metrics_summary["dispatch_normalized_latency_ms"] or {}
    )
    queueing_stats = metrics_summary["global_capacity_queue_ms"] or {}
    goodput_stats = metrics_summary["goodput"]

    # TPOT stats (Inter-Token Latency) - Aggregate all individual token samples
    all_itl_samples = []
    for s in completed_seqs:
        if s.metric and s.metric.itl_samples:
            all_itl_samples.extend(s.metric.itl_samples)
    
    tpot_stats = {}
    if all_itl_samples:
        tpot_stats = {
            "avg": np.mean(all_itl_samples),
            "p50": np.median(all_itl_samples),
            "p90": np.percentile(all_itl_samples, 90),
            "p95": np.percentile(all_itl_samples, 95),
            "p99": np.percentile(all_itl_samples, 99)
        }
    
    decode_queue_samples = [s.metric.decode_queue_time_ms for s in completed_seqs if s.metric.decode_queue_time_ms]
    decode_queue_stats = {}
    if decode_queue_samples:
        decode_queue_stats = {
            "avg": np.mean(decode_queue_samples),
            "p50": np.percentile(decode_queue_samples, 50),
            "p90": np.percentile(decode_queue_samples, 90),
            "p95": np.percentile(decode_queue_samples, 95),
            "p99": np.percentile(decode_queue_samples, 99)
        }

    print("\n" + "=" * 60)
    print("--- Benchmark Results ---")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Requests completed: {total_seqs}")
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(
        "Average Bootstrap TTFT: "
        f"{ttft_stats.get('mean', 0):.2f} ms"
    )
    if model_ttft_stats:
        print(
            "Average Model TTFT: "
            f"{model_ttft_stats.get('mean', 0):.2f} ms"
        )
    print(
        "Average E2E Latency: "
        f"{e2e_stats.get('mean', 0) / 1000:.2f} s"
    )
    print()
    
    if tpot_stats:
        print("--- TPOT without Queueing Time (ms/token) ---")
        print(f"  Avg:  {tpot_stats.get('avg', 0):.2f}")
        print(f"  P50:  {tpot_stats.get('p50', 0):.2f}")
        print(f"  P90:  {tpot_stats.get('p90', 0):.2f}")
        print(f"  P95:  {tpot_stats.get('p95', 0):.2f}")
        print(f"  P99:  {tpot_stats.get('p99', 0):.2f}")
        print()

    if tpot_wq_stats:
        print("--- TPOT With Queueing Time (ms/token) ---")
        print(
            "  Definition: real-token execution-boundary time plus "
            "GPU-capacity queue; unused final-quantum slots excluded"
        )
        print(f"  Avg:  {tpot_wq_stats.get('mean', 0):.2f}")
        print(f"  P50:  {tpot_wq_stats.get('p50', 0):.2f}")
        print(f"  P90:  {tpot_wq_stats.get('p90', 0):.2f}")
        print(f"  P95:  {tpot_wq_stats.get('p95', 0):.2f}")
        print(f"  P99:  {tpot_wq_stats.get('p99', 0):.2f}")
        print()

    if dispatch_tpot_stats:
        print("--- End-to-End TPOT With All Queueing (ms/token) ---")
        print(
            "  Definition: dispatch-to-terminal minus unused final-quantum "
            "slots, divided by generated tokens"
        )
        print(f"  Avg:  {dispatch_tpot_stats.get('mean', 0):.2f}")
        print(f"  P50:  {dispatch_tpot_stats.get('p50', 0):.2f}")
        print(f"  P90:  {dispatch_tpot_stats.get('p90', 0):.2f}")
        print(f"  P95:  {dispatch_tpot_stats.get('p95', 0):.2f}")
        print(f"  P99:  {dispatch_tpot_stats.get('p99', 0):.2f}")
        print()

    if dispatch_normalized_stats:
        print(
            "--- Legacy Dispatch-Normalized Latency (ms/token) ---"
        )
        print(
            f"  Avg:  {dispatch_normalized_stats.get('mean', 0):.2f}"
        )
        print(
            f"  P50:  {dispatch_normalized_stats.get('p50', 0):.2f}"
        )
        print(
            f"  P90:  {dispatch_normalized_stats.get('p90', 0):.2f}"
        )
        print(
            f"  P95:  {dispatch_normalized_stats.get('p95', 0):.2f}"
        )
        print(
            f"  P99:  {dispatch_normalized_stats.get('p99', 0):.2f}"
        )
        print()

    # ITL with decode queue
    itls_with_dq = [s.metric.avg_itl_with_decode_queue for s in completed_seqs if s.metric.avg_itl_with_decode_queue]
    if itls_with_dq:
        print("--- ITL With Decode Queue (ms/token) ---")
        print(f"  Avg:  {np.mean(itls_with_dq):.2f}")
        print(f"  P50:  {np.median(itls_with_dq):.2f}")
        print(f"  P90:  {np.percentile(itls_with_dq, 90):.2f}")
        print(f"  P95:  {np.percentile(itls_with_dq, 95):.2f}")
        print(f"  P99:  {np.percentile(itls_with_dq, 99):.2f}")
        print()

    if queueing_stats:
        print("--- Queueing Time (ms) ---")
        print("  Definition: GPU-capacity queue only")
        print(f"  Avg:  {queueing_stats.get('mean', 0):.2f}")
        print(f"  P50:  {queueing_stats.get('p50', 0):.2f}")
        print(f"  P90:  {queueing_stats.get('p90', 0):.2f}")
        print(f"  P95:  {queueing_stats.get('p95', 0):.2f}")
        print(f"  P99:  {queueing_stats.get('p99', 0):.2f}")
        print()

    if decode_queue_stats:
        print("--- Decode Queue Time (ms) ---")
        print(f"  Avg:  {decode_queue_stats.get('avg', 0):.2f}")
        print(f"  P50:  {decode_queue_stats.get('p50', 0):.2f}")
        print(f"  P90:  {decode_queue_stats.get('p90', 0):.2f}")
        print(f"  P95:  {decode_queue_stats.get('p95', 0):.2f}")
        print(f"  P99:  {decode_queue_stats.get('p99', 0):.2f}")
        print()

    print("--- Goodput (SLO: TPOT with queueing < 100ms) ---")
    print(
        "  SLO Success: "
        f"{goodput_stats['successful_requests']}/"
        f"{goodput_stats['eligible_requests']}"
    )
    print(
        f"  Goodput: {goodput_stats['attainment_percent']:.2f}%"
    )
    print("=" * 60 + "\n")
    
    if itl_log_path:
        print(
            "Per-request scalar metrics were written incrementally to "
            f"{itl_log_path}."
        )


def main():
    args = parse_args()

    print(f"\n--- Benchmark: {args.num_requests} reqs, {args.request_rate} req/s, burst={args.burstiness} ---")

    # Initialize Engine
    print(
        f"Scheduler architecture: {args.scheduler_arch}, "
        f"Routing strategy: {args.routing_strategy}, "
        f"Router policy: {args.router_policy}, "
        f"SP master selector: {args.sp_master_selector}"
    )
    engine = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        cuda_graph_mode=args.cuda_graph_mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
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
        loop_count=args.loop_count,
        routing_strategy=args.routing_strategy,
        sp_master_selector=args.sp_master_selector,
        scheduler_arch=args.scheduler_arch,
        router_policy=args.router_policy,
        segment_size=args.segment_size,
        kvcache_block_size=64,
        max_num_recv_seqs=32,
        max_num_send_seqs=16,
        enable_profiler=args.enable_profiler,
        profiler_start_step=args.profiler_start_step,
        profiling_step=args.profiling_step,
        profiler_dir=args.profiler_dir,
        profiler_start_time=args.profiler_start_time,
        profiling_duration=args.profiling_duration,
        enable_non_uniform_split=not args.disable_non_uniform_split,
        fixed_sp_size=args.fixed_sp_size,
        sp_backend=args.sp_backend,
        dynamic_sp_size_strategy=args.dynamic_sp_size_strategy,
        dynamic_sp_bucket_preset=args.dynamic_sp_bucket_preset,
        hierarchical_execution_trace=args.hierarchical_execution_trace,
        hierarchical_quantum_diagnostics=bool(
            args.hierarchical_quantum_log_path
        ),
        hierarchical_result_fastpath=(
            os.getenv("NANODEPLOY_HIER_RESULT_FASTPATH", "0") == "1"
        ),
        hierarchical_worker_transport=os.getenv(
            "NANODEPLOY_HIER_WORKER_TRANSPORT", "ray"
        ),
        hierarchical_async_depth=int(
            os.getenv("NANODEPLOY_HIER_ASYNC_DEPTH", "1")
        ),
    )
    
    # Print Config
    print_model_config(engine)

    # Run Warmup
    world_size = args.ep
    run_warmup(engine, args.max_num_seqs, world_size)
    engine.reset_execution_boundary_metrics()
    if (
        engine.config.hierarchical_quantum_diagnostics
    ):
        # Exclude warmup quantums from the benchmark diagnostic stream.
        engine.drain_hierarchical_quantum_diagnostics()

    # Prepare Data
    request_generator = get_dataset_generator(args)
    arrival_times = generate_arrival_times(args.num_requests, args.request_rate, args.burstiness)

    request_metrics_log_path = (
        args.request_metrics_log_path or args.itl_log_path
    )

    # Run Benchmark
    total_time, seq_map, metrics_summary = run_benchmark(
        engine,
        request_generator,
        arrival_times,
        args.num_requests,
        diagnostic_log_interval=args.diagnostic_log_interval,
        slow_add_threshold_ms=args.slow_add_threshold_ms,
        hierarchical_trace_log_path=args.hierarchical_trace_log_path,
        hierarchical_quantum_log_path=(
            args.hierarchical_quantum_log_path
        ),
        request_metrics_log_path=request_metrics_log_path,
        metrics_summary_path=args.metrics_summary_path,
    )

    # Report
    calculate_and_print_metrics(
        total_time,
        seq_map,
        args.num_requests,
        metrics_summary,
        itl_log_path=request_metrics_log_path,
    )


if __name__ == "__main__":
    main()

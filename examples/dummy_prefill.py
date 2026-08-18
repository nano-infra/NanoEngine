import argparse
import os

import numpy as np

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.hierarchical_contract import (
    validate_execution_trace_set,
)
from nanodeploy.engine.sequence import Sequence
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Dummy prefill decode test")
    parser.add_argument(
        "--scheduler-arch",
        type=str,
        default="legacy_global",
        choices=["legacy_global", "hierarchical"],
        help="Scheduler architecture (default: legacy_global)",
    )
    parser.add_argument(
        "--routing-strategy",
        type=str,
        default="RoundRobin",
        choices=["RoundRobin", "LeastBatch", "LeastCache"],
        help="Legacy scheduler routing strategy (default: RoundRobin)"
    )
    parser.add_argument(
        "--router-policy",
        type=str,
        default="least_batch",
        choices=[
            "round_robin",
            "least_batch",
            "least_batch_v2",
            "least_cache",
        ],
        help="Hierarchical load-balancer policy (default: least_batch)",
    )
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--moe-routing-simulation-strategy",
        type=str,
        default="model",
        choices=["model", "uniform_random", "perfect_eplb"],
        help="MoE routing behavior used by the decode model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shared worker RNG seed used by uniform_random routing.",
    )
    parser.add_argument("--long-len", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--sp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument(
        "--sp-backend",
        type=str,
        default="legacy_ll",
        choices=["legacy_ll", "hao_basic", "nccl", "nccl_compact"],
    )
    parser.add_argument(
        "--sp-size-policy",
        type=str,
        default="legacy",
        choices=["legacy", "long_short"],
        help=(
            "SP-size policy. long_short uses SP=1 at or below the threshold "
            "and the configured --sp size above it."
        ),
    )
    parser.add_argument("--segment-size", type=int, default=65536)
    parser.add_argument("--long-request-sp-threshold", type=int, default=100000)
    parser.add_argument("--enable-non-uniform-split", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument("--master-address", type=str, default="10.102.252.174:26444")
    parser.add_argument("--ray-address", type=str, default="10.102.252.174:7799")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-num-send-seqs", type=int, default=128)
    parser.add_argument("--max-num-recv-seqs", type=int, default=130)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--hierarchical-execution-trace",
        action="store_true",
        help="Capture and validate every hierarchical 16-forward quantum.",
    )
    parser.add_argument(
        "--loop-count",
        type=int,
        default=None,
        help=(
            "Steps per decode iteration. Defaults to 16 for hierarchical "
            "and 48 for legacy_global."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=200_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--enable-profiler", action="store_true")
    parser.add_argument("--profiler-start-step", type=int, default=40)
    parser.add_argument("--profiling-step", type=int, default=16)
    parser.add_argument("--profiler-dir", type=str, default="./profiler_logs")
    parser.add_argument("--profiler-start-time", type=float, default=None)
    parser.add_argument("--profiling-duration", type=float, default=None)
    args = parser.parse_args()
    if (
        args.hierarchical_execution_trace
        and args.scheduler_arch != "hierarchical"
    ):
        parser.error(
            "--hierarchical-execution-trace requires "
            "--scheduler-arch hierarchical"
        )
    path = os.path.expanduser(args.model_path)
    loop_count = args.loop_count
    if loop_count is None:
        loop_count = 16 if args.scheduler_arch == "hierarchical" else 48

    use_long_short = args.sp_size_policy == "long_short"
    if use_long_short and args.sp <= 1:
        parser.error("--sp-size-policy long_short requires --sp greater than 1")

    decode = LLM(
        path,
        enforce_eager=args.enforce_eager,
        attention_dp=args.dp,
        attention_sp=args.sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=args.ep,
        ffn_tp=1,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        moe_routing_simulation_strategy=(
            args.moe_routing_simulation_strategy
        ),
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        loop_count=loop_count,
        max_num_send_seqs=args.max_num_send_seqs,
        max_num_recv_seqs=args.max_num_recv_seqs,
        kvcache_block_size=64,
        enable_profiler=args.enable_profiler,
        profiler_start_step=args.profiler_start_step,
        profiling_step=args.profiling_step,
        profiler_dir=args.profiler_dir,
        profiler_start_time=args.profiler_start_time,
        profiling_duration=args.profiling_duration,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduler_arch=args.scheduler_arch,
        routing_strategy=args.routing_strategy,
        router_policy=args.router_policy,
        hierarchical_execution_trace=args.hierarchical_execution_trace,
        sp_backend=args.sp_backend,
        segment_size=args.segment_size,
        enable_dynamic_sp_size=use_long_short,
        dynamic_sp_size_strategy=(
            "long_short_sp8" if use_long_short else "legacy"
        ),
        dynamic_sp_long_request_threshold=args.long_request_sp_threshold,
        enable_non_uniform_split=args.enable_non_uniform_split,
    )

    print(
        f"Starting with scheduler_arch={args.scheduler_arch}, "
        f"routing_strategy={args.routing_strategy}, "
        f"router_policy={args.router_policy}, loop_count={loop_count}, "
        f"sp_backend={args.sp_backend}, "
        f"sp_size_policy={args.sp_size_policy}"
    )

    sampling_params = SamplingParams(temperature=0.1, max_tokens=args.max_tokens, ignore_eos=True)

    long_seqs = []
    if args.long_len > 0:
        long_seqs = [
            Sequence(
                np.random.randint(0, 10001, size=args.long_len).tolist(),
                sampling_params=sampling_params,
            )
        ]

    short_seqs = [
        Sequence(
            np.random.randint(0, 10001, size=args.seq_len).tolist(),
            sampling_params=sampling_params,
        )
        for _ in range(args.num_seqs)
    ]

    seqs = long_seqs + short_seqs

    decode.add_request(seqs)
    if args.num_steps > 0:
        for step_idx in range(args.num_steps):
            outputs, num_tokens, batch_size, sch_latency, post_sch_latency = decode.step()
            print(
                "step",
                step_idx,
                "num_tokens",
                num_tokens,
                "batch_size",
                batch_size,
                "outputs",
                len(outputs),
                "sch_latency_ms",
                f"{sch_latency:.4f}",
                "post_sch_latency_ms",
                f"{post_sch_latency:.4f}",
            )
    else:
        decode.generate()
    if args.hierarchical_execution_trace:
        traces = decode.drain_execution_traces()
        steps = validate_execution_trace_set(
            traces,
            range(args.dp * args.sp),
        )
        print(
            "Validated hierarchical execution trace:",
            f"ranks={args.dp * args.sp}",
            f"global_steps={len(steps)}",
            f"records={len(traces)}",
        )


if __name__ == "__main__":
    main()

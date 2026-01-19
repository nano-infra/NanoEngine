import argparse
import os

import numpy as np

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.sequence import Sequence
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Dummy prefill test with scheduler mode selection")
    parser.add_argument(
        "--scheduler-mode",
        type=str,
        default="centralized",
        choices=["centralized", "decentralized"],
        help="Scheduler mode: centralized or decentralized (default: centralized)"
    )
    parser.add_argument(
        "--routing-strategy",
        type=str,
        default="RoundRobin",
        choices=["RoundRobin", "LeastBatch", "LeastCache"],
        help="Routing strategy for decentralized scheduler (default: RoundRobin)"
    )
    args = parser.parse_args()
    path = os.path.expanduser("/models/qwen3-235B-Instruct-2507-FP8")

    decode = LLM(
        path,
        enforce_eager=False,
        attention_dp=8,
        attention_sp=1,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        master_address="10.102.97.179:26444",
        ray_address="10.102.97.179:7078",
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=128,
        max_model_len=200_000,
        max_num_batched_tokens=200_000,
        loop_count=48,
        max_num_send_seqs=128,
        max_num_recv_seqs=130,
        kvcache_block_size=256,
        enable_profiler=False,
        scheduler_mode=args.scheduler_mode,
        routing_strategy=args.routing_strategy,
    )
    
    print(f"Starting with scheduler_mode={args.scheduler_mode}, routing_strategy={args.routing_strategy}")

    sampling_params = SamplingParams(temperature=0.1, max_tokens=256, ignore_eos=True)

    long_seqs = [
        Sequence(
            np.random.randint(0, 10001, size=400000).tolist(),
            # [0] * 400000,
            sampling_params=sampling_params,
        )
    ]
    long_seqs = []

    short_seqs = [
        Sequence(
            np.random.randint(0, 10001, size=2048).tolist(),
            # [0] * 2048,
            sampling_params=sampling_params,
        )
        for _ in range(512)
    ]

    seqs = long_seqs + short_seqs

    decode.add_request(seqs)
    decode.generate()


if __name__ == "__main__":
    main()

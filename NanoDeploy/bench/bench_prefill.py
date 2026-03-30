"""Benchmark prefill performance with random prompts.

Random token IDs are generated per step to defeat prefix caching.

Usage:
    python bench_prefill.py --model /models/deepseek-v3 --prompt_len 1024 --batch_size 1 --num_steps 5 --kvcache_block_size 64
    python bench_prefill.py --config config.yaml --prompt_len 4096 --batch_size 4 --num_steps 10
"""

import random
import time

from jsonargparse import ActionConfigFile, ArgumentParser

from nanodeploy import Sequence
from nanodeploy.config import Config
from nanodeploy.llm_component import LLM
from nanodeploy.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerFast


def generate_random_token_ids(vocab_size: int, length: int) -> list[int]:
    return [random.randint(100, vocab_size - 1) for _ in range(length)]


def main():
    parser = ArgumentParser(description="Benchmark prefill performance")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)
    parser.add_argument(
        "--prompt_len", type=int, default=1024, help="Prompt length in tokens"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of sequences per batch"
    )
    parser.add_argument(
        "--num_steps", type=int, default=5, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1, help="Warmup steps excluded from stats"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1,
        help="Max decode tokens per sequence (keep small for prefill-only benchmark)",
    )
    args = parser.parse_args()

    bench_keys = (
        "config",
        "prompt_len",
        "batch_size",
        "num_steps",
        "warmup_steps",
        "max_tokens",
    )
    config_args = {k: v for k, v in vars(args).items() if k not in bench_keys}
    config = Config(**config_args)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model)
    vocab_size = tokenizer.vocab_size
    llm = LLM(config)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        ignore_eos=True,
        temperature=0.0,
    )

    total_steps = args.warmup_steps + args.num_steps
    total_tokens_per_step = args.prompt_len * args.batch_size

    print("=" * 60)
    print("Prefill Benchmark Configuration")
    print("=" * 60)
    print(f"  Model          : {config.model}")
    print(f"  Prompt length  : {args.prompt_len}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Tokens/step    : {total_tokens_per_step}")
    print(f"  Warmup steps   : {args.warmup_steps}")
    print(f"  Benchmark steps: {args.num_steps}")
    print(f"  Max decode tok : {args.max_tokens}")
    print(f"  Vocab size     : {vocab_size}")
    print("=" * 60)

    latencies: list[float] = []
    throughputs: list[float] = []

    for step in range(total_steps):
        is_warmup = step < args.warmup_steps
        tag = (
            "warmup"
            if is_warmup
            else f"step {step - args.warmup_steps + 1}/{args.num_steps}"
        )

        seqs = [
            Sequence(
                generate_random_token_ids(vocab_size, args.prompt_len),
                sampling_params=sampling_params,
            )
            for _ in range(args.batch_size)
        ]

        llm.add_request(seqs)

        t0 = time.perf_counter()
        llm.generate(use_tqdm=True)
        elapsed = time.perf_counter() - t0

        throughput = total_tokens_per_step / elapsed

        if not is_warmup:
            latencies.append(elapsed)
            throughputs.append(throughput)

        print(
            f"[{tag}] latency={elapsed * 1000:.2f}ms  "
            f"throughput={throughput:.0f} tok/s  "
            f"tokens={total_tokens_per_step}"
        )

    if throughputs:
        avg_lat = sum(latencies) / len(latencies)
        avg_tp = sum(throughputs) / len(throughputs)
        print("=" * 60)
        print("Prefill Benchmark Results")
        print("=" * 60)
        print(f"  Avg latency   : {avg_lat * 1000:.2f} ms")
        print(f"  Avg throughput : {avg_tp:.0f} tok/s")
        print(f"  Min throughput : {min(throughputs):.0f} tok/s")
        print(f"  Max throughput : {max(throughputs):.0f} tok/s")
        print("=" * 60)


if __name__ == "__main__":
    main()

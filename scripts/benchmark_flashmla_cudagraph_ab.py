#!/usr/bin/env python3
"""Compare eager and CUDA Graph latency of vLLM's FlashMLA operator."""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
import vllm

try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
        is_flashmla_dense_supported,
    )
except ImportError as exc:
    raise ImportError(
        "vLLM's FlashMLA operator is not importable. Run this script with the "
        "Python environment and source tree used by the target vLLM build."
    ) from exc


DEFAULT_CONTEXT_LENGTHS = (
    "115673,115865,121566,25711,31760,39193,43736,58672"
)


def parse_context_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths:
        raise argparse.ArgumentTypeError("at least one context length is required")
    if any(length < 0 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be non-negative")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths",
        type=parse_context_lengths,
        default=parse_context_lengths(DEFAULT_CONTEXT_LENGTHS),
        help="comma-separated valid context lengths",
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=17,
        help="physical CUDA Graph batch size; extra rows have zero context",
    )
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.graph_batch_size < len(args.context_lengths):
        raise ValueError(
            "graph batch size must be at least the number of context lengths"
        )
    if args.max_model_len < max(args.context_lengths):
        raise ValueError("max model length is smaller than an input context")
    if args.iterations < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("iterations/repeats must be positive and warmup non-negative")


def get_metadata(cache_seqlens: torch.Tensor):
    try:
        return get_mla_metadata(cache_seqlens, 128, 1)
    except TypeError:
        return get_mla_metadata(
            cache_seqlens,
            128,
            1,
            128,
            False,
            None,
        )


def time_repeated_calls(
    launch: Callable[[], object],
    iterations: int,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    supported, reason = is_flashmla_dense_supported()
    if not supported:
        raise RuntimeError(f"vLLM FlashMLA is unavailable: {reason}")

    torch.manual_seed(args.seed)
    block_size = 64
    context_lengths = args.context_lengths + [0] * (
        args.graph_batch_size - len(args.context_lengths)
    )
    page_counts = [
        (length + block_size - 1) // block_size for length in context_lengths
    ]
    total_pages = sum(page_counts)
    max_blocks_per_sequence = math.ceil(args.max_model_len / block_size)

    cache_seqlens = torch.tensor(
        context_lengths,
        dtype=torch.int32,
        device="cuda",
    )
    q = torch.randn(
        args.graph_batch_size,
        1,
        128,
        576,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.randn(
        total_pages,
        block_size,
        1,
        576,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_table = torch.full(
        (args.graph_batch_size, max_blocks_per_sequence),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    page_start = 0
    for row, page_count in enumerate(page_counts):
        if page_count == 0:
            continue
        block_table[row, :page_count] = torch.arange(
            page_start,
            page_start + page_count,
            dtype=torch.int32,
            device="cuda",
        )
        page_start += page_count

    tile_scheduler_metadata, num_splits = get_metadata(cache_seqlens)
    softmax_scale = (192**-0.5) * (0.1 * math.log(256) + 1) ** 2

    def kernel():
        return flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            512,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            causal=True,
        )

    for _ in range(args.warmup):
        eager_output, eager_lse = kernel()
    torch.cuda.synchronize()

    single_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single_graph):
        graph_output, graph_lse = kernel()
    torch.cuda.synchronize()

    unrolled_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(unrolled_graph):
        for _ in range(args.iterations):
            unrolled_output, unrolled_lse = kernel()
    torch.cuda.synchronize()

    for _ in range(args.iterations):
        single_graph.replay()
    unrolled_graph.replay()
    torch.cuda.synchronize()

    eager_samples = []
    single_graph_samples = []
    unrolled_graph_samples = []
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            eager_samples.append(time_repeated_calls(kernel, args.iterations))
            single_graph_samples.append(
                time_repeated_calls(single_graph.replay, args.iterations)
            )
        else:
            single_graph_samples.append(
                time_repeated_calls(single_graph.replay, args.iterations)
            )
            eager_samples.append(time_repeated_calls(kernel, args.iterations))
        unrolled_graph_samples.append(
            time_repeated_calls(unrolled_graph.replay, 1) / args.iterations
        )

    eager_median = statistics.median(eager_samples)
    single_graph_median = statistics.median(single_graph_samples)
    unrolled_graph_median = statistics.median(unrolled_graph_samples)

    print(f"vllm_version: {vllm.__version__}")
    print(f"vllm_path: {vllm.__file__}")
    print(f"flashmla_op_module: {flash_mla_with_kvcache.__module__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"q: {tuple(q.shape)} {q.dtype}")
    print(f"k_cache: {tuple(k_cache.shape)} {k_cache.dtype}")
    print(f"block_table: {tuple(block_table.shape)} {block_table.dtype}")
    print(f"cache_seqlens: {context_lengths}")
    print(f"active_context_tokens: {sum(context_lengths)}")
    print(f"active_pages: {total_pages}")
    print(f"output: {tuple(graph_output.shape)} {graph_output.dtype}")
    print(f"lse: {tuple(graph_lse.shape)} {graph_lse.dtype}")
    print(f"eager_samples_us: {[round(value, 3) for value in eager_samples]}")
    print(
        "single_kernel_graph_samples_us: "
        f"{[round(value, 3) for value in single_graph_samples]}"
    )
    print(
        "unrolled_graph_samples_us: "
        f"{[round(value, 3) for value in unrolled_graph_samples]}"
    )
    print(f"eager_median_us: {eager_median:.3f}")
    print(f"single_kernel_graph_median_us: {single_graph_median:.3f}")
    print(f"unrolled_graph_median_us: {unrolled_graph_median:.3f}")
    print(
        "single_graph_vs_eager_percent: "
        f"{(single_graph_median / eager_median - 1) * 100.0:+.2f}%"
    )
    print(
        "unrolled_graph_vs_eager_percent: "
        f"{(unrolled_graph_median / eager_median - 1) * 100.0:+.2f}%"
    )


if __name__ == "__main__":
    main()

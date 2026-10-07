#!/usr/bin/env python3
"""CUDA Graph benchmark for the synthetic FlashMLA + DCP decode pipeline.

Unlike ``run_flashmla_dcp_bench.py``, this benchmark captures one complete
iteration (local projections, FlashMLA, DCP collectives/combine, and the TP
tail) into a single CUDA Graph.  CUDA events surround ``graph.replay()`` and
are deliberately kept outside the captured graph.

For multi-rank scenarios, the first graph operation is a one-element NCCL
broadcast.  It absorbs residual launch skew before the measured workload.

The captured inputs and tensor shapes are static.  This is appropriate for a
microbenchmark of one fixed decode shape, but it does not model dynamic batch
or sequence-length changes between replays.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch
import torch.distributed as dist

from run_flashmla_dcp_bench import (
    SCENARIOS,
    all_gather_query_heads,
    approx_local_flashmla_flops,
    approx_local_o_proj_flops,
    approx_local_q_nope_proj_flops,
    approx_local_v_up_proj_flops,
    barrier_if_needed,
    build_flashmla_query,
    build_global_seq_lens,
    build_local_kv_cache,
    destroy_dist_if_needed,
    get_dcp_local_seq_lens,
    get_dist_env,
    init_dist_if_needed,
    load_dims,
    logical_bytes_for_a2a_lse,
    logical_bytes_for_a2a_output,
    logical_bytes_for_query_allgather,
    logical_bytes_for_tp_allreduce,
    maybe_cuda_profiler_start,
    maybe_cuda_profiler_stop,
    o_proj_local,
    tp_allreduce,
    v_up_proj_local,
)
from vllm.v1.attention.ops.dcp_alltoall import dcp_lse_combine_triton
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_dense_supported,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the complete FlashMLA + DCP decode critical path in one "
            "CUDA Graph and time graph replay."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="tp8dcp8",
        help="Predefined topology. world_size must equal tp and dcp.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "deepseek_v3_config.json",
        help="Path to DeepSeek-V3 config.json.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1024,
        help="Only used when --seq-len-mode=random.",
    )
    parser.add_argument(
        "--seq-len-mode",
        choices=["uniform", "random"],
        default="uniform",
        help="Sequence lengths are fixed for the lifetime of the captured graph.",
    )
    parser.add_argument("--cp-interleave-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Eager warmups on the capture stream before graph capture.",
    )
    parser.add_argument(
        "--graph-warmup-iters",
        type=int,
        default=int(os.environ.get("GRAPH_WARMUP_ITERS", "3")),
        help="Untimed graph replays after capture.",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--capture-error-mode",
        choices=["global", "thread_local", "relaxed"],
        default="global",
        help="cudaStreamCaptureMode passed to torch.cuda.graph.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cudagraph",
        help="Directory for rank-0 CSV outputs.",
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--use-cuda-profiler-range", action="store_true")
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="Emit an NVTX range around each graph replay.",
    )
    parser.add_argument(
        "--verify-world-size",
        action="store_true",
        help="Retained for CLI compatibility; graph mode always verifies it.",
    )
    args = parser.parse_args()
    if args.warmup_iters < 1:
        parser.error("--warmup-iters must be at least 1 for CUDA Graph capture")
    if args.graph_warmup_iters < 0:
        parser.error("--graph-warmup-iters must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    return args


def a2a_lse_reduce_graphed(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """DCP post-attention A2A without events or host synchronization."""
    batch_size, num_heads, head_dim = cp_attn_out.shape
    heads_per_rank = num_heads // world_size

    send_output = (
        cp_attn_out.contiguous()
        .view(batch_size, world_size, heads_per_rank, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    recv_output = torch.empty_like(send_output)

    send_lse = (
        cp_attn_lse.contiguous()
        .view(batch_size, world_size, heads_per_rank)
        .permute(1, 0, 2)
        .contiguous()
    )
    recv_lse = torch.empty_like(send_lse)

    output_work = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        async_op=True,
    )
    lse_work = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        async_op=True,
    )
    output_work.wait()
    lse_work.wait()
    return dcp_lse_combine_triton(recv_output, recv_lse)


def warmup_on_stream(
    body: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    iters: int,
) -> torch.Tensor:
    current_stream = torch.cuda.current_stream()
    stream.wait_stream(current_stream)
    output: torch.Tensor | None = None
    with torch.cuda.stream(stream):
        for _ in range(iters):
            output = body()
    current_stream.wait_stream(stream)
    torch.cuda.synchronize()
    assert output is not None
    return output


def replay_once_us(
    graph: torch.cuda.CUDAGraph,
    replay_stream: torch.cuda.Stream,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    emit_nvtx: bool,
) -> float:
    replay_stream.wait_stream(torch.cuda.current_stream())

    if emit_nvtx:
        torch.cuda.nvtx.range_push("cuda_graph_replay")
    try:
        with torch.cuda.stream(replay_stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
    finally:
        if emit_nvtx:
            torch.cuda.nvtx.range_pop()
    return start.elapsed_time(end) * 1000.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # Select the local device before FlashMLA capability checks or NCCL process
    # group initialization can lazily create a CUDA context.
    _, _, env_local_rank = get_dist_env()
    torch.cuda.set_device(env_local_rank)
    if not is_flashmla_dense_supported()[0]:
        raise RuntimeError(is_flashmla_dense_supported()[1])

    rank, world_size, local_rank = init_dist_if_needed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    tp_size, dcp_size, expected_world_size = SCENARIOS[args.scenario]
    if world_size != expected_world_size:
        raise ValueError(
            f"CUDA Graph scenario {args.scenario} requires "
            f"world_size={expected_world_size}, got {world_size}."
        )

    dims = load_dims(args.model_config, args.dtype)
    if dims.num_attention_heads % tp_size != 0:
        raise ValueError("num_attention_heads must be divisible by tp_size.")
    local_heads = dims.num_attention_heads // tp_size

    global_seq_lens = build_global_seq_lens(
        batch_size=args.batch_size,
        seq_len_mode=args.seq_len_mode,
        max_seq_len=args.seq_len,
        min_seq_len=args.min_seq_len,
        seed=args.seed,
        device=device,
    )
    if dcp_size > 1:
        local_seq_lens = get_dcp_local_seq_lens(
            global_seq_lens,
            dcp_size=dcp_size,
            dcp_rank=rank,
            cp_interleave_size=args.cp_interleave_size,
        )
    else:
        local_seq_lens = global_seq_lens

    block_table, kv_cache = build_local_kv_cache(
        local_seq_lens=local_seq_lens,
        block_size=args.block_size,
        head_dim=dims.kv_cache_head_dim,
        dtype=dims.dtype,
        device=device,
    )
    q_local = torch.randn(
        args.batch_size,
        local_heads,
        dims.qk_head_dim,
        device=device,
        dtype=dims.dtype,
    )
    q_nope_local, q_pe_local = q_local.split(
        [dims.qk_nope_head_dim, dims.qk_rope_head_dim], dim=-1
    )
    w_uk_t = torch.randn(
        local_heads,
        dims.qk_nope_head_dim,
        dims.kv_lora_rank,
        device=device,
        dtype=dims.dtype,
    )
    w_uv = torch.randn(
        local_heads,
        dims.kv_lora_rank,
        dims.v_head_dim,
        device=device,
        dtype=dims.dtype,
    )
    o_proj_weight_local = torch.randn(
        local_heads * dims.v_head_dim,
        dims.hidden_size,
        device=device,
        dtype=dims.dtype,
    )

    scheduler_metadata, num_splits = get_mla_metadata(
        local_seq_lens,
        dims.num_attention_heads,
        1,
    )
    softmax_scale = dims.qk_head_dim ** (-0.5)
    graph_sync_token = torch.zeros(1, dtype=torch.int32, device=device)

    def graph_body() -> torch.Tensor:
        if world_size > 1:
            # Keep this as Broadcast instead of AllReduce so nsys can separate
            # graph-head rank alignment from the real TP AllReduce below.
            dist.broadcast(graph_sync_token, src=0)

        q_flashmla_local = build_flashmla_query(q_nope_local, q_pe_local, w_uk_t)
        if dcp_size > 1:
            q_full = all_gather_query_heads(q_flashmla_local, dcp_size)
        else:
            q_full = q_flashmla_local

        flashmla_output, flashmla_lse = flash_mla_with_kvcache(
            q_full.unsqueeze(1).contiguous(),
            kv_cache,
            block_table,
            local_seq_lens,
            dims.kv_lora_rank,
            scheduler_metadata,
            num_splits,
            softmax_scale=softmax_scale,
            causal=True,
        )
        flashmla_output = flashmla_output.squeeze(1).contiguous()
        flashmla_lse = flashmla_lse.squeeze(-1).contiguous()

        if dcp_size > 1:
            attn_out_local_heads = a2a_lse_reduce_graphed(
                flashmla_output,
                flashmla_lse,
                dcp_size,
            )
        else:
            attn_out_local_heads = flashmla_output

        v_up_output = v_up_proj_local(attn_out_local_heads, w_uv)
        output_parallel = o_proj_local(v_up_output, o_proj_weight_local)
        return tp_allreduce(output_parallel, tp_size)

    # CUDA Graph capture requires all lazy kernels, libraries, allocators, and
    # NCCL communicators to have been initialized before capture begins.
    capture_stream = torch.cuda.Stream()
    barrier_if_needed(world_size)
    warmup_output = warmup_on_stream(
        graph_body,
        capture_stream,
        args.warmup_iters,
    )
    del warmup_output
    barrier_if_needed(world_size)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph,
        stream=capture_stream,
        capture_error_mode=args.capture_error_mode,
    ):
        static_output = graph_body()

    torch.cuda.synchronize()
    barrier_if_needed(world_size)

    # Initialize the reusable timing events before the first measured replay.
    # CUDA Event objects are lazy; creating them inside the timed loop can put
    # host-side event initialization between the graph launch and end record.
    replay_start = torch.cuda.Event(enable_timing=True)
    replay_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(capture_stream):
        replay_start.record()
        replay_end.record()
    replay_end.synchronize()

    for _ in range(args.graph_warmup_iters):
        _ = replay_once_us(
            graph,
            capture_stream,
            replay_start,
            replay_end,
            emit_nvtx=False,
        )

    barrier_if_needed(world_size)
    maybe_cuda_profiler_start(args.use_cuda_profiler_range)
    replay_us: list[float] = []
    for _ in range(args.iters):
        # The graph-head NCCL Broadcast is the per-replay rank rendezvous.
        # Avoid an extra out-of-graph NCCL barrier in the profiled iteration.
        replay_us.append(
            replay_once_us(
                graph,
                capture_stream,
                replay_start,
                replay_end,
                args.emit_nvtx,
            )
        )
    barrier_if_needed(world_size)
    maybe_cuda_profiler_stop(args.use_cuda_profiler_range)
    _ = static_output

    replay_avg = mean(replay_us)
    replay_min = min(replay_us)
    replay_max = max(replay_us)
    pre_q_bytes = (
        logical_bytes_for_query_allgather(
            args.batch_size,
            local_heads,
            dims.flashmla_q_head_dim,
            dims.dtype,
        )
        if dcp_size > 1
        else 0
    )
    post_output_bytes = (
        logical_bytes_for_a2a_output(
            args.batch_size,
            dims.num_attention_heads,
            dims.kv_lora_rank,
            dims.dtype,
        )
        if dcp_size > 1
        else 0
    )
    post_lse_bytes = (
        logical_bytes_for_a2a_lse(args.batch_size, dims.num_attention_heads)
        if dcp_size > 1
        else 0
    )
    tp_allreduce_bytes = (
        logical_bytes_for_tp_allreduce(
            args.batch_size,
            dims.hidden_size,
            dims.dtype,
        )
        if tp_size > 1
        else 0
    )

    per_rank_row: dict[str, Any] = {
        "execution_mode": "cuda_graph",
        "scenario": args.scenario,
        "rank": rank,
        "world_size": world_size,
        "tp_size": tp_size,
        "dcp_size": dcp_size,
        "batch_size": args.batch_size,
        "seq_len_mode": args.seq_len_mode,
        "global_max_seq_len": int(global_seq_lens.max().item()),
        "global_mean_seq_len": float(global_seq_lens.float().mean().item()),
        "local_max_seq_len": int(local_seq_lens.max().item()),
        "local_mean_seq_len": float(local_seq_lens.float().mean().item()),
        "cp_interleave_size": args.cp_interleave_size,
        "block_size": args.block_size,
        "local_heads": local_heads,
        "global_heads_after_gather": dims.num_attention_heads,
        "hidden_size": dims.hidden_size,
        "qk_head_dim": dims.qk_head_dim,
        "flashmla_q_head_dim": dims.flashmla_q_head_dim,
        "kv_cache_head_dim": dims.kv_cache_head_dim,
        "kv_lora_rank": dims.kv_lora_rank,
        "v_head_dim": dims.v_head_dim,
        "capture_error_mode": args.capture_error_mode,
        "graph_head_rank_sync": "nccl_broadcast_1el" if world_size > 1 else "none",
        "capture_warmup_iters": args.warmup_iters,
        "graph_warmup_iters": args.graph_warmup_iters,
        "timed_replay_iters": args.iters,
        "pre_query_allgather_logical_bytes": pre_q_bytes,
        "post_a2a_output_logical_bytes": post_output_bytes,
        "post_a2a_lse_logical_bytes": post_lse_bytes,
        "tp_allreduce_logical_bytes": tp_allreduce_bytes,
        "approx_local_q_nope_proj_flops": approx_local_q_nope_proj_flops(
            args.batch_size,
            local_heads,
            dims.qk_nope_head_dim,
            dims.kv_lora_rank,
        ),
        "approx_local_flashmla_flops": approx_local_flashmla_flops(
            local_seq_lens,
            dims.num_attention_heads,
            dims.flashmla_q_head_dim,
            dims.kv_lora_rank,
        ),
        "approx_local_v_up_proj_flops": approx_local_v_up_proj_flops(
            args.batch_size,
            local_heads,
            dims.kv_lora_rank,
            dims.v_head_dim,
        ),
        "approx_local_o_proj_flops": approx_local_o_proj_flops(
            args.batch_size,
            local_heads,
            dims.v_head_dim,
            dims.hidden_size,
        ),
        "graph_replay_us_avg": replay_avg,
        "graph_replay_us_min": replay_min,
        "graph_replay_us_max": replay_max,
        # Compatibility aliases for tools that only consume end-to-end time.
        "total_us_avg": replay_avg,
        "total_us_min": replay_min,
        "total_us_max": replay_max,
    }

    gathered_rows: list[dict[str, Any]] = [None] * world_size  # type: ignore[list-item]
    if world_size > 1:
        dist.all_gather_object(gathered_rows, per_rank_row)
    else:
        gathered_rows = [per_rank_row]

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = (
            f"{args.output_prefix}{args.scenario}_bs{args.batch_size}"
            f"_seqlen{args.seq_len}_{args.seq_len_mode}"
        )
        per_rank_path = output_dir / f"{base_name}_per_rank.csv"
        summary_path = output_dir / f"{base_name}_summary.csv"
        write_csv(per_rank_path, gathered_rows)

        rank_averages = [float(row["graph_replay_us_avg"]) for row in gathered_rows]
        summary_row = {
            "execution_mode": "cuda_graph",
            "scenario": args.scenario,
            "world_size": world_size,
            "tp_size": tp_size,
            "dcp_size": dcp_size,
            "batch_size": args.batch_size,
            "seq_len_mode": args.seq_len_mode,
            "graph_head_rank_sync": (
                "nccl_broadcast_1el" if world_size > 1 else "none"
            ),
            "global_max_seq_len": int(global_seq_lens.max().item()),
            "global_mean_seq_len": float(global_seq_lens.float().mean().item()),
            "capture_warmup_iters": args.warmup_iters,
            "graph_warmup_iters": args.graph_warmup_iters,
            "timed_replay_iters": args.iters,
            "graph_replay_us_avg_across_ranks": mean(rank_averages),
            "graph_replay_us_max_across_ranks": max(rank_averages),
            "total_us_avg_across_ranks": mean(rank_averages),
            "total_us_max_across_ranks": max(rank_averages),
        }
        write_csv(summary_path, [summary_row])
        print(json.dumps(summary_row, indent=2))
        print(f"Wrote {per_rank_path}")
        print(f"Wrote {summary_path}")

    # NCCL collectives captured by a CUDA Graph keep graph-specific
    # communicator resources alive.  Release the graph executable on every
    # rank before destroy_process_group(); otherwise NCCL teardown can wait on
    # resources still owned by the live CUDAGraph and never return under nsys.
    del static_output
    graph.reset()
    del graph
    torch.cuda.synchronize()

    destroy_dist_if_needed()


if __name__ == "__main__":
    main()

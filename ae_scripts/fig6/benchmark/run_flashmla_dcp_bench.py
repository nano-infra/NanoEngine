#!/usr/bin/env python3
"""
Distributed FlashMLA decode microbenchmark for DeepSeek-V3 DCP analysis.

This script benchmarks the decode attention critical path only:

1. DCP query all-gather
2. local FlashMLA decode on the local KV shard
3. DCP A2A post communication (two NCCL sendrecv kernels in nsys)
4. local Triton LSE combine
5. local v_up_proj
6. local RowParallel o_proj matmul
7. TP all-reduce on o_proj output

It is meant to reproduce the operator structure of vLLM's
`flashmla + dcp_comm_backend=a2a` path with minimal extra framework overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.distributed as dist


VLLM_REPO_ROOT = os.environ.get("VLLM_REPO_ROOT")
if VLLM_REPO_ROOT and VLLM_REPO_ROOT not in sys.path:
    sys.path.insert(0, VLLM_REPO_ROOT)

from vllm.v1.attention.ops.dcp_alltoall import dcp_lse_combine_triton
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_dense_supported,
)


SCENARIOS: dict[str, tuple[int, int, int]] = {
    "pure_dp": (1, 1, 1),
    "tp2dcp2": (2, 2, 2),
    "tp4dcp4": (4, 4, 4),
    "tp8dcp8": (8, 8, 8),
}


@dataclass
class DeepSeekV3Dims:
    hidden_size: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    dtype: torch.dtype

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def flashmla_q_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_cache_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashMLA DCP microbenchmark for DeepSeek-V3 decode."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="tp8dcp8",
        help="Predefined topology. For this script, world_size == tp == dcp.",
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
        help="Global request length generation mode.",
    )
    parser.add_argument(
        "--cp-interleave-size",
        type=int,
        default=1,
        help="Matches vLLM's cp_kv_cache_interleave_size.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="FlashMLA dense decode uses block_size=64 on Hopper.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV outputs. Rank 0 writes files here.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional basename prefix for outputs.",
    )
    parser.add_argument(
        "--use-cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around timed iterations.",
    )
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="Emit NVTX ranges around benchmark stages.",
    )
    parser.add_argument(
        "--verify-world-size",
        action="store_true",
        help="Require torch.distributed world_size to match the selected scenario.",
    )
    return parser.parse_args()


def load_dims(config_path: str, dtype_override: str) -> DeepSeekV3Dims:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return DeepSeekV3Dims(
        hidden_size=int(cfg["hidden_size"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        kv_lora_rank=int(cfg["kv_lora_rank"]),
        qk_nope_head_dim=int(cfg["qk_nope_head_dim"]),
        qk_rope_head_dim=int(cfg["qk_rope_head_dim"]),
        v_head_dim=int(cfg["v_head_dim"]),
        dtype=dtype_map[dtype_override],
    )


def get_dist_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def init_dist_if_needed() -> tuple[int, int, int]:
    rank, world_size, local_rank = get_dist_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def destroy_dist_if_needed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier_if_needed(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def maybe_cuda_profiler_start(enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.cudart().cudaProfilerStart()


def maybe_cuda_profiler_stop(enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.cudart().cudaProfilerStop()


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave_size: int,
) -> torch.Tensor:
    rank_offsets = torch.tensor([[dcp_rank]], dtype=torch.int32, device=seq_lens.device)
    seq_lens_tiled = seq_lens.to(torch.int32).unsqueeze(-1)
    base = (
        seq_lens_tiled
        // cp_interleave_size
        // dcp_size
        * cp_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_interleave_size,
        0,
        cp_interleave_size,
    )
    return (base + remainder).squeeze(1)


def build_global_seq_lens(
    batch_size: int,
    seq_len_mode: str,
    max_seq_len: int,
    min_seq_len: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if seq_len_mode == "uniform":
        return torch.full(
            (batch_size,),
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )

    rng = random.Random(seed)
    seq_lens = [
        rng.randint(min_seq_len, max_seq_len)
        for _ in range(batch_size)
    ]
    return torch.tensor(seq_lens, dtype=torch.int32, device=device)


def build_local_kv_cache(
    local_seq_lens: torch.Tensor,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(local_seq_lens.numel())
    max_local_seq_len = int(local_seq_lens.max().item())
    max_blocks = max(1, cdiv(max_local_seq_len, block_size))
    block_table = torch.arange(
        batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, max_blocks)
    kv_cache = torch.randn(
        block_table.numel(),
        block_size,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    if max_local_seq_len > 0:
        kv_cache_view = kv_cache.view(batch_size, max_blocks * block_size, 1, head_dim)
        for idx, local_len in enumerate(local_seq_lens.tolist()):
            if local_len < max_blocks * block_size:
                kv_cache_view[idx, local_len:] = float("nan")
    return block_table, kv_cache


def measure_cuda_us(fn) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0, result


def all_gather_query_heads(q_local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return q_local
    gathered = torch.empty(
        (world_size,) + tuple(q_local.shape),
        dtype=q_local.dtype,
        device=q_local.device,
    )
    dist.all_gather_into_tensor(gathered, q_local.contiguous())
    return (
        gathered.permute(1, 0, 2, 3)
        .reshape(q_local.shape[0], world_size * q_local.shape[1], q_local.shape[2])
        .contiguous()
    )


def build_flashmla_query(
    q_nope_local: torch.Tensor,
    q_pe_local: torch.Tensor,
    w_uk_t: torch.Tensor,
) -> torch.Tensor:
    # DeepSeek-V3 decode projects the NoPE query into KV latent space
    # before concatenating the RoPE component for FlashMLA.
    ql_nope = torch.bmm(q_nope_local.transpose(0, 1).contiguous(), w_uk_t)
    ql_nope = ql_nope.transpose(0, 1).contiguous()
    return torch.cat((ql_nope, q_pe_local), dim=-1).contiguous()


def a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, float]:
    if world_size == 1:
        return cp_attn_out, 0.0

    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()

    batch_size, num_heads, head_dim = local_output.shape
    heads_per_rank = num_heads // world_size

    send_output = (
        local_output.view(batch_size, world_size, heads_per_rank, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    recv_output = torch.empty_like(send_output)

    send_lse = (
        local_lse.view(batch_size, world_size, heads_per_rank)
        .permute(1, 0, 2)
        .contiguous()
    )
    recv_lse = torch.empty_like(send_lse)

    def do_comm():
        work_output = dist.all_to_all_single(
            recv_output.view(-1),
            send_output.view(-1),
            async_op=True,
        )
        work_lse = dist.all_to_all_single(
            recv_lse.view(-1),
            send_lse.view(-1),
            async_op=True,
        )
        work_output.wait()
        work_lse.wait()
        return recv_output, recv_lse

    comm_us, (recv_output, recv_lse) = measure_cuda_us(do_comm)
    reduced = dcp_lse_combine_triton(recv_output, recv_lse)
    return reduced, comm_us


def v_up_proj_local(
    attn_out_local_heads: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    x = attn_out_local_heads.transpose(0, 1).contiguous()
    out = torch.bmm(x, w_uv)
    return out.transpose(0, 1).contiguous().reshape(attn_out_local_heads.shape[0], -1)


def o_proj_local(
    v_up_output_local: torch.Tensor,
    o_proj_weight_local: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(v_up_output_local, o_proj_weight_local)


def tp_allreduce(
    output_parallel: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    if tp_size == 1:
        return output_parallel
    dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
    return output_parallel


def logical_bytes_for_query_allgather(
    batch_size: int,
    local_heads: int,
    q_head_dim: int,
    dtype: torch.dtype,
) -> int:
    return batch_size * local_heads * q_head_dim * torch.tensor([], dtype=dtype).element_size()


def logical_bytes_for_a2a_output(
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    dtype: torch.dtype,
) -> int:
    return batch_size * num_heads * kv_lora_rank * torch.tensor([], dtype=dtype).element_size()


def logical_bytes_for_a2a_lse(
    batch_size: int,
    num_heads: int,
) -> int:
    return batch_size * num_heads * torch.tensor([], dtype=torch.float32).element_size()


def logical_bytes_for_tp_allreduce(
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> int:
    return batch_size * hidden_size * torch.tensor([], dtype=dtype).element_size()


def approx_local_flashmla_flops(
    local_seq_lens: torch.Tensor,
    num_heads: int,
    flashmla_q_head_dim: int,
    kv_lora_rank: int,
) -> int:
    return int(
        2
        * int(local_seq_lens.sum().item())
        * num_heads
        * (flashmla_q_head_dim + kv_lora_rank)
    )


def approx_local_q_nope_proj_flops(
    batch_size: int,
    local_heads: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
) -> int:
    return 2 * batch_size * local_heads * qk_nope_head_dim * kv_lora_rank


def approx_local_v_up_proj_flops(
    batch_size: int,
    local_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
) -> int:
    return 2 * batch_size * local_heads * kv_lora_rank * v_head_dim


def approx_local_o_proj_flops(
    batch_size: int,
    local_heads: int,
    v_head_dim: int,
    hidden_size: int,
) -> int:
    return 2 * batch_size * (local_heads * v_head_dim) * hidden_size


def summarize(values: list[float]) -> tuple[float, float, float]:
    return mean(values), min(values), max(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not is_flashmla_dense_supported()[0]:
        raise RuntimeError(is_flashmla_dense_supported()[1])

    rank, world_size, local_rank = init_dist_if_needed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    tp_size, dcp_size, expected_world_size = SCENARIOS[args.scenario]
    if args.verify_world_size and world_size != expected_world_size:
        raise ValueError(
            f"Scenario {args.scenario} expects world_size={expected_world_size}, "
            f"got {world_size}."
        )
    if world_size != expected_world_size and world_size != 1:
        raise ValueError(
            f"Expected world_size {expected_world_size} for {args.scenario}, got {world_size}."
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
        [dims.qk_nope_head_dim, dims.qk_rope_head_dim],
        dim=-1,
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
        1 * dims.num_attention_heads,
        1,
    )
    softmax_scale = dims.qk_head_dim ** (-0.5)

    pre_q_bytes = (
        logical_bytes_for_query_allgather(
            batch_size=args.batch_size,
            local_heads=local_heads,
            q_head_dim=dims.flashmla_q_head_dim,
            dtype=dims.dtype,
        )
        if dcp_size > 1
        else 0
    )
    post_output_bytes = (
        logical_bytes_for_a2a_output(
            batch_size=args.batch_size,
            num_heads=dims.num_attention_heads,
            kv_lora_rank=dims.kv_lora_rank,
            dtype=dims.dtype,
        )
        if dcp_size > 1
        else 0
    )
    post_lse_bytes = (
        logical_bytes_for_a2a_lse(
            batch_size=args.batch_size,
            num_heads=dims.num_attention_heads,
        )
        if dcp_size > 1
        else 0
    )
    tp_allreduce_bytes = (
        logical_bytes_for_tp_allreduce(
            batch_size=args.batch_size,
            hidden_size=dims.hidden_size,
            dtype=dims.dtype,
        )
        if tp_size > 1
        else 0
    )
    local_flops = approx_local_flashmla_flops(
        local_seq_lens=local_seq_lens,
        num_heads=dims.num_attention_heads,
        flashmla_q_head_dim=dims.flashmla_q_head_dim,
        kv_lora_rank=dims.kv_lora_rank,
    )
    local_q_nope_proj_flops = approx_local_q_nope_proj_flops(
        batch_size=args.batch_size,
        local_heads=local_heads,
        qk_nope_head_dim=dims.qk_nope_head_dim,
        kv_lora_rank=dims.kv_lora_rank,
    )
    local_v_up_proj_flops = approx_local_v_up_proj_flops(
        batch_size=args.batch_size,
        local_heads=local_heads,
        kv_lora_rank=dims.kv_lora_rank,
        v_head_dim=dims.v_head_dim,
    )
    local_o_proj_flops = approx_local_o_proj_flops(
        batch_size=args.batch_size,
        local_heads=local_heads,
        v_head_dim=dims.v_head_dim,
        hidden_size=dims.hidden_size,
    )

    def run_once() -> dict[str, float]:
        with nvtx_range("q_nope_proj", args.emit_nvtx):
            q_nope_proj_us, q_flashmla_local = measure_cuda_us(
                lambda: build_flashmla_query(q_nope_local, q_pe_local, w_uk_t)
            )

        if dcp_size > 1:
            with nvtx_range("query_allgather", args.emit_nvtx):
                query_allgather_us, q_full = measure_cuda_us(
                    lambda: all_gather_query_heads(q_flashmla_local, dcp_size)
                )
        else:
            query_allgather_us = 0.0
            q_full = q_flashmla_local

        q_for_flashmla = q_full.unsqueeze(1).contiguous()

        with nvtx_range("flashmla", args.emit_nvtx):
            flashmla_us, flashmla_ret = measure_cuda_us(
                lambda: flash_mla_with_kvcache(
                    q_for_flashmla,
                    kv_cache,
                    block_table,
                    local_seq_lens,
                    dims.kv_lora_rank,
                    scheduler_metadata,
                    num_splits,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            )

        o, lse = flashmla_ret
        o = o.squeeze(1).contiguous()
        lse = lse.squeeze(-1).contiguous()

        if dcp_size > 1:
            with nvtx_range("post_a2a_comm", args.emit_nvtx):
                post_comm_us, _ = 0.0, None
                post_lse_combine_us = 0.0

                def do_a2a():
                    reduced, comm_us = a2a_lse_reduce(o, lse, dcp_size)
                    return reduced, comm_us

                a2a_total_us, (reduced_out, post_comm_us) = measure_cuda_us(do_a2a)
                post_lse_combine_us = max(a2a_total_us - post_comm_us, 0.0)
                attn_out_local_heads = reduced_out
        else:
            post_comm_us = 0.0
            post_lse_combine_us = 0.0
            attn_out_local_heads = o

        with nvtx_range("v_up_proj", args.emit_nvtx):
            v_up_proj_us, v_up_out = measure_cuda_us(
                lambda: v_up_proj_local(attn_out_local_heads, w_uv)
            )

        with nvtx_range("o_proj", args.emit_nvtx):
            o_proj_us, o_proj_out_parallel = measure_cuda_us(
                lambda: o_proj_local(v_up_out, o_proj_weight_local)
            )

        if tp_size > 1:
            with nvtx_range("tp_allreduce", args.emit_nvtx):
                tp_allreduce_us, final_hidden = measure_cuda_us(
                    lambda: tp_allreduce(o_proj_out_parallel, tp_size)
                )
                _ = final_hidden
        else:
            tp_allreduce_us = 0.0

        dcp_attention_core_total_us = (
            q_nope_proj_us
            + query_allgather_us
            + flashmla_us
            + post_comm_us
            + post_lse_combine_us
        )
        tp_attention_tail_total_us = v_up_proj_us + o_proj_us + tp_allreduce_us
        total_us = dcp_attention_core_total_us + tp_attention_tail_total_us
        return {
            "q_nope_proj_us": q_nope_proj_us,
            "query_allgather_us": query_allgather_us,
            "flashmla_us": flashmla_us,
            "post_a2a_comm_us": post_comm_us,
            "post_lse_combine_us": post_lse_combine_us,
            "dcp_attention_core_total_us": dcp_attention_core_total_us,
            "v_up_proj_us": v_up_proj_us,
            "o_proj_us": o_proj_us,
            "tp_allreduce_us": tp_allreduce_us,
            "tp_attention_tail_total_us": tp_attention_tail_total_us,
            "total_us": total_us,
        }

    for _ in range(args.warmup_iters):
        barrier_if_needed(world_size)
        _ = run_once()

    barrier_if_needed(world_size)
    maybe_cuda_profiler_start(args.use_cuda_profiler_range)

    stage_records: list[dict[str, float]] = []
    for _ in range(args.iters):
        barrier_if_needed(world_size)
        stage_records.append(run_once())

    barrier_if_needed(world_size)
    maybe_cuda_profiler_stop(args.use_cuda_profiler_range)

    per_rank_row = {
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
        "pre_query_allgather_logical_bytes": pre_q_bytes,
        "post_a2a_output_logical_bytes": post_output_bytes,
        "post_a2a_lse_logical_bytes": post_lse_bytes,
        "tp_allreduce_logical_bytes": tp_allreduce_bytes,
        "approx_local_q_nope_proj_flops": local_q_nope_proj_flops,
        "approx_local_flashmla_flops": local_flops,
        "approx_local_v_up_proj_flops": local_v_up_proj_flops,
        "approx_local_o_proj_flops": local_o_proj_flops,
    }

    for key in [
        "q_nope_proj_us",
        "query_allgather_us",
        "flashmla_us",
        "post_a2a_comm_us",
        "post_lse_combine_us",
        "dcp_attention_core_total_us",
        "v_up_proj_us",
        "o_proj_us",
        "tp_allreduce_us",
        "tp_attention_tail_total_us",
        "total_us",
    ]:
        stage_values = [rec[key] for rec in stage_records]
        avg_val, min_val, max_val = summarize(stage_values)
        per_rank_row[f"{key}_avg"] = avg_val
        per_rank_row[f"{key}_min"] = min_val
        per_rank_row[f"{key}_max"] = max_val

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

        summary_row = {
            "scenario": args.scenario,
            "world_size": world_size,
            "tp_size": tp_size,
            "dcp_size": dcp_size,
            "batch_size": args.batch_size,
            "seq_len_mode": args.seq_len_mode,
            "global_max_seq_len": int(global_seq_lens.max().item()),
            "global_mean_seq_len": float(global_seq_lens.float().mean().item()),
            "pre_query_allgather_logical_bytes": pre_q_bytes,
            "post_a2a_output_logical_bytes": post_output_bytes,
            "post_a2a_lse_logical_bytes": post_lse_bytes,
            "tp_allreduce_logical_bytes": tp_allreduce_bytes,
        }
        q_nope_proj_flop_values = [
            int(row["approx_local_q_nope_proj_flops"]) for row in gathered_rows
        ]
        summary_row["approx_local_q_nope_proj_flops_avg_across_ranks"] = int(
            mean(q_nope_proj_flop_values)
        )
        summary_row["approx_local_q_nope_proj_flops_max_across_ranks"] = max(
            q_nope_proj_flop_values
        )
        flop_values = [int(row["approx_local_flashmla_flops"]) for row in gathered_rows]
        summary_row["approx_local_flashmla_flops_avg_across_ranks"] = int(
            mean(flop_values)
        )
        summary_row["approx_local_flashmla_flops_max_across_ranks"] = max(flop_values)
        v_up_flop_values = [
            int(row["approx_local_v_up_proj_flops"]) for row in gathered_rows
        ]
        summary_row["approx_local_v_up_proj_flops_avg_across_ranks"] = int(
            mean(v_up_flop_values)
        )
        summary_row["approx_local_v_up_proj_flops_max_across_ranks"] = max(
            v_up_flop_values
        )
        o_proj_flop_values = [
            int(row["approx_local_o_proj_flops"]) for row in gathered_rows
        ]
        summary_row["approx_local_o_proj_flops_avg_across_ranks"] = int(
            mean(o_proj_flop_values)
        )
        summary_row["approx_local_o_proj_flops_max_across_ranks"] = max(
            o_proj_flop_values
        )

        metric_roots = [
            "q_nope_proj_us",
            "query_allgather_us",
            "flashmla_us",
            "post_a2a_comm_us",
            "post_lse_combine_us",
            "dcp_attention_core_total_us",
            "v_up_proj_us",
            "o_proj_us",
            "tp_allreduce_us",
            "tp_attention_tail_total_us",
            "total_us",
        ]
        for root in metric_roots:
            values = [float(row[f"{root}_avg"]) for row in gathered_rows]
            summary_row[f"{root}_avg_across_ranks"] = mean(values)
            summary_row[f"{root}_max_across_ranks"] = max(values)

        write_csv(summary_path, [summary_row])
        print(json.dumps(summary_row, indent=2))
        print(f"Wrote {per_rank_path}")
        print(f"Wrote {summary_path}")

    destroy_dist_if_needed()


if __name__ == "__main__":
    main()

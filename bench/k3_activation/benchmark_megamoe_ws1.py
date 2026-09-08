#!/usr/bin/env python3
"""Single-rank activation benchmark for the production DeepGEMM MegaMoE kernel."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist

EXPERTS = 896
HIDDEN = 3584
INTERMEDIATE = 3072
TOP_K = 16
SITU_CLAMP = 0.03125


def gib(value: int) -> float:
    return value / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[128, 512, 2048, 8192, 16384]
    )
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "activation_peaks_megamoe_ws1.csv",
    )
    args = parser.parse_args()

    import deep_gemm

    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def transform(weight: torch.Tensor, scale: torch.Tensor):
        scale_f32 = scale.view(torch.float8_e8m0fnu).to(torch.float32)
        transformed_scale = deep_gemm.transform_sf_into_required_layout(
            scale_f32,
            mn=weight.shape[1],
            k=weight.shape[2] * 2,
            recipe=(1, 32),
            num_groups=weight.shape[0],
            disable_ue8m0_cast=False,
        )
        return weight.view(torch.int8), transformed_scale

    gate_up = torch.zeros(
        EXPERTS, INTERMEDIATE * 2, HIDDEN // 2, device="cuda", dtype=torch.uint8
    )
    gate_up_scale = torch.full(
        (EXPERTS, INTERMEDIATE * 2, HIDDEN // 32), 127, device="cuda", dtype=torch.uint8
    )
    down = torch.zeros(
        EXPERTS, HIDDEN, INTERMEDIATE // 2, device="cuda", dtype=torch.uint8
    )
    down_scale = torch.full(
        (EXPERTS, HIDDEN, INTERMEDIATE // 32), 127, device="cuda", dtype=torch.uint8
    )
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        transform(gate_up, gate_up_scale), transform(down, down_scale)
    )
    del gate_up, gate_up_scale, down, down_scale
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    weight_baseline = torch.cuda.memory_allocated()
    free_before_workspace, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        EXPERTS,
        args.capacity,
        TOP_K,
        HIDDEN,
        INTERMEDIATE,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    torch.cuda.synchronize()
    workspace_live = torch.cuda.memory_allocated() - weight_baseline
    workspace_peak = torch.cuda.max_memory_allocated() - weight_baseline
    free_after_workspace, _ = torch.cuda.mem_get_info()
    workspace_device = free_before_workspace - free_after_workspace
    print(
        f"workspace device={gib(workspace_device):.3f} GiB "
        f"torch_live={gib(workspace_live):.3f} GiB "
        f"torch_peak={gib(workspace_peak):.3f} GiB",
        flush=True,
    )

    x_storage = torch.from_dlpack(buf.x).view(torch.float8_e4m3fn)

    def forward(tokens: int, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor):
        deep_gemm.mega_moe_pre_dispatch(
            x,
            ids,
            weights,
            x_storage,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=tokens,
            group_size=32,
            use_fp4_acts=False,
        )
        output = torch.empty_like(x, dtype=torch.bfloat16)
        deep_gemm.fp8_fp4_mega_moe(
            output,
            l1,
            l2,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=SITU_CLAMP,
            fast_math=True,
        )
        return output

    rows = []
    for tokens in args.tokens:
        x = torch.zeros(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.arange(tokens * TOP_K, device="cuda", dtype=torch.int32)
            .view(tokens, TOP_K)
            .remainder(EXPERTS)
        )
        weights = torch.full(
            (tokens, TOP_K), 1 / TOP_K, device="cuda", dtype=torch.float32
        )
        torch.cuda.synchronize()
        input_baseline = torch.cuda.memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        first = forward(tokens, x, ids, weights)
        torch.cuda.synchronize()
        first_peak = torch.cuda.max_memory_allocated() - input_baseline
        del first

        for _ in range(args.warmup):
            warm = forward(tokens, x, ids, weights)
            torch.cuda.synchronize()
            del warm
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = forward(tokens, x, ids, weights)
        end.record()
        torch.cuda.synchronize()
        steady_peak = torch.cuda.max_memory_allocated() - input_baseline
        latency_ms = start.elapsed_time(end)
        assert torch.isfinite(output).all()
        rows.append(
            {
                "tokens": tokens,
                "first_forward_peak_bytes": first_peak,
                "steady_forward_peak_bytes": steady_peak,
                "steady_bytes_per_token": steady_peak / tokens,
                "steady_latency_ms": latency_ms,
                "workspace_device_bytes_at_capacity": workspace_device,
                "workspace_torch_live_bytes_at_capacity": workspace_live,
                "workspace_torch_peak_bytes_at_capacity": workspace_peak,
                "capacity_tokens": args.capacity,
                "gpu": torch.cuda.get_device_name(),
            }
        )
        print(
            f"T={tokens:5d} first={gib(first_peak):7.3f} GiB steady={gib(steady_peak):7.3f} GiB latency={latency_ms:8.3f} ms",
            flush=True,
        )
        del x, ids, weights, output
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

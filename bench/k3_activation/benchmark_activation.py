#!/usr/bin/env python3
"""Measure incremental peak CUDA memory for K3-shaped Prefill operators."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


KDA_HEADS = 96
KDA_DIM = 128
MLA_HEADS = 96
MLA_QK_DIM = 192
MLA_V_DIM = 128


def shapes(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        batch, length = item.strip().lower().split("x")
        result.append((int(batch), int(length)))
    return result


def peak_call(fn, warmup: int) -> tuple[int, int]:
    for _ in range(warmup):
        output = fn()
        torch.cuda.synchronize()
        del output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return peak_allocated - baseline_allocated, peak_reserved - baseline_reserved


def kda_case(
    batch: int, length: int, warmup: int, context_length: int
) -> tuple[int, int]:
    from dlengine.runtime.kernel.triton.fla.kda import chunk_kda

    total = batch * length
    q = torch.randn(1, total, KDA_HEADS, KDA_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand(1, total, KDA_HEADS, device="cuda", dtype=torch.float32)
    state = torch.randn(
        batch, KDA_HEADS, KDA_DIM, KDA_DIM, device="cuda", dtype=torch.bfloat16
    )
    conv_state = torch.empty(
        batch, 3 * KDA_HEADS * KDA_DIM, 4, device="cuda", dtype=torch.bfloat16
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, total + 1, length, device="cuda", dtype=torch.int32)
    a_log = torch.zeros(KDA_HEADS, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(KDA_HEADS * KDA_DIM, device="cuda", dtype=torch.float32)

    def run():
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

    return peak_call(run, warmup)


def mla_case(
    batch: int, length: int, warmup: int, context_length: int
) -> tuple[int, int]:
    from flash_attn.cute import flash_attn_varlen_func

    total = batch * length
    q = torch.randn(total, MLA_HEADS, MLA_QK_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(total, MLA_HEADS, MLA_V_DIM, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, total + 1, length, device="cuda", dtype=torch.int32)
    # K3's 1M-token persistent MLA cache is present in the baseline but is not
    # read by this fresh-only Prefill attention-core measurement.
    latent_cache = torch.empty(
        context_length, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    latent_scale = torch.empty(context_length, 4, device="cuda", dtype=torch.float32)
    rope_cache = torch.empty(context_length, 64, device="cuda", dtype=torch.bfloat16)

    def run():
        result = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=length,
            max_seqlen_k=length,
            softmax_scale=MLA_QK_DIM**-0.5,
            causal=True,
        )
        return result[0] if isinstance(result, tuple) else result

    return peak_call(run, warmup)


def moe_case(
    batch: int, length: int, warmup: int, context_length: int
) -> tuple[int, int]:
    del context_length
    from dlengine.runtime.layers.backends.experts.generic import GenericExperts

    total = batch * length
    # The portable local path is BF16. It preserves K3's 896 experts, Top-16,
    # latent width 3584, and intermediate width 3072, but uses SwiGLU rather
    # than the MXFP4 MegaMoE SiTU production kernel.
    with torch.device("cuda"):
        experts = GenericExperts(
            hidden_size=3584,
            intermediate_size=3072,
            num_experts=896,
            top_k=16,
            ep_size=1,
            tp_size=1,
        )
    x = torch.randn(total, 3584, device="cuda", dtype=torch.bfloat16)
    ids = (
        torch.arange(total * 16, device="cuda", dtype=torch.int64)
        .reshape(total, 16)
        .remainder(896)
    )
    weights = torch.full((total, 16), 1 / 16, device="cuda", dtype=torch.float32)

    def run():
        return experts(x, ids, weights, is_prefill=True)

    return peak_call(run, warmup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--shapes",
        default="1x128,1x256,1x512,1x1024,1x2048,1x4096,1x8192,1x16384,4x4096,16x1024,32x512",
    )
    parser.add_argument("--targets", default="kda,mla")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=1_048_576)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.targets.split(",")]
    rows = []
    for target in requested:
        case = {"kda": kda_case, "mla": mla_case, "moe": moe_case}[target]
        for batch, length in shapes(args.shapes):
            gc.collect()
            torch.cuda.empty_cache()
            allocated, reserved = case(batch, length, args.warmup, args.context_length)
            total = batch * length
            row = {
                "target": target,
                "batch": batch,
                "length": length,
                "active_tokens": total,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "allocated_bytes_per_token": allocated / total,
            }
            rows.append(row)
            print(
                f"{target:3s} B={batch:2d} L={length:5d} T={total:5d} "
                f"peak={allocated / 2**20:9.3f} MiB "
                f"reserved_delta={reserved / 2**20:9.3f} MiB",
                flush=True,
            )

    result_stem = "_".join(requested)
    with (args.output_dir / f"activation_peaks_{result_stem}.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "dtype": "BF16",
        "resident_context_tokens": args.context_length,
        "maximum_prefill_chunk_tokens": max(
            batch * length for batch, length in shapes(args.shapes)
        ),
        "kda": {"heads": KDA_HEADS, "head_dim": KDA_DIM},
        "mla": {"heads": MLA_HEADS, "qk_head_dim": MLA_QK_DIM, "v_head_dim": MLA_V_DIM},
        "scope": "fresh-only operator core; inputs and synthetic 1M-token persistent cache/state excluded from incremental peak",
    }
    (args.output_dir / f"metadata_{result_stem}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

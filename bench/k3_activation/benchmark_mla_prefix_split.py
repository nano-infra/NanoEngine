#!/usr/bin/env python3
"""Measure K3 MLA cached-prefix chunking at 1M context and 16K fresh tokens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOTAL = 1_048_576
FRESH = 16_384
HEADS = 96
NOPE = 128
ROPE = 64
VALUE = 128
LATENT = 512
PACKED = 656


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-chunk-size", type=int, default=131_072)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.prefix_chunk_size < 0:
        raise ValueError("prefix chunk size must be non-negative")

    torch.cuda.set_device(0)
    from flash_attn.cute import flash_attn_varlen_func

    from dlengine.runtime.kernel.triton.hopper.fp8_utils import (
        restore_mla_fp8_cache_rows,
    )
    from dlengine.runtime.layers.backends.attention.mla_utils import (
        chunked_prefix_mla_attention,
    )

    cached = TOTAL - FRESH
    packed_cache = torch.zeros(cached, PACKED, device="cuda", dtype=torch.uint8)
    compressed_fresh = torch.zeros(FRESH, LATENT, device="cuda", dtype=torch.bfloat16)
    kpe_fresh = torch.zeros(FRESH, ROPE, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(FRESH, HEADS, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    kc = torch.zeros(LATENT, HEADS * NOPE, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros(LATENT, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    cached_lens = torch.tensor([cached], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, FRESH], device="cuda", dtype=torch.int32)

    k_nope_fresh = (compressed_fresh @ kc).view(FRESH, HEADS, NOPE)
    k_fresh = torch.cat(
        [k_nope_fresh, kpe_fresh[:, None, :].expand(-1, HEADS, -1)], dim=-1
    )
    v_fresh = (compressed_fresh @ vc).view(FRESH, HEADS, VALUE)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    def forward() -> torch.Tensor:
        restored = restore_mla_fp8_cache_rows(packed_cache)
        if args.prefix_chunk_size == 0:
            compressed_cached = restored[:, :LATENT]
            rope_cached = restored[:, LATENT:]
            k_nope_cached = (compressed_cached @ kc).view(cached, HEADS, NOPE)
            k_cached = torch.cat(
                [k_nope_cached, rope_cached[:, None, :].expand(-1, HEADS, -1)],
                dim=-1,
            )
            v_cached = (compressed_cached @ vc).view(cached, HEADS, VALUE)
            k = torch.cat([k_cached, k_fresh], dim=0)
            v = torch.cat([v_cached, v_fresh], dim=0)
            cu_k = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
            result = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=FRESH,
                max_seqlen_k=TOTAL,
                softmax_scale=(NOPE + ROPE) ** -0.5,
                causal=True,
            )
            return result[0] if isinstance(result, tuple) else result
        return chunked_prefix_mla_attention(
            q,
            k_fresh,
            v_fresh,
            restored,
            cached_lens,
            cu_q,
            kc,
            vc,
            chunk_size=args.prefix_chunk_size,
            softmax_scale=(NOPE + ROPE) ** -0.5,
            attention_func=flash_attn_varlen_func,
        )

    for _ in range(args.warmup):
        warmup_out = forward()
        torch.cuda.synchronize()
        del warmup_out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = forward()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated() - baseline
    live = torch.cuda.memory_allocated() - baseline
    assert out.shape == (FRESH, HEADS, VALUE)

    result = {
        "total_context": TOTAL,
        "cached_prefix": cached,
        "fresh_chunk": FRESH,
        "prefix_chunk_size": args.prefix_chunk_size,
        "incremental_peak_bytes": peak,
        "incremental_live_bytes": live,
        "steady_forward_ms": elapsed_ms,
        "warmup_iterations": args.warmup,
        "gpu": torch.cuda.get_device_name(0),
        "cache_layout": "packed mixed FP8/BF16, 656 bytes/token",
    }
    print(json.dumps(result, indent=2), flush=True)

    output = Path(__file__).parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    suffix = str(args.prefix_chunk_size) if args.prefix_chunk_size else "unsplit"
    stem = f"mla_prefix_split_{suffix}"
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()

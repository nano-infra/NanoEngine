#!/usr/bin/env python3
"""K3 MLA cached-prefix Prefill peak at 1M total context and 16K fresh tokens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOTAL = 1_048_576
FRESH = 16_384
CACHED = TOTAL - FRESH
HEADS = 96
NOPE = 128
ROPE = 64
VALUE = 128
LATENT = 512
PACKED = 656


def snapshot(stage: str, baseline: int, started: float) -> dict:
    torch.cuda.synchronize()
    row = {
        "stage": stage,
        "allocated_bytes": torch.cuda.memory_allocated() - baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"{stage:24s} live={row['allocated_bytes']/2**30:8.3f} GiB "
        f"peak={row['peak_allocated_bytes']/2**30:8.3f} GiB "
        f"elapsed={row['elapsed_seconds']:8.2f} s",
        flush=True,
    )
    return row


def main() -> None:
    global TOTAL, FRESH, CACHED
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-context", type=int, default=TOTAL)
    parser.add_argument("--fresh-chunk", type=int, default=FRESH)
    args = parser.parse_args()
    TOTAL = args.total_context
    FRESH = args.fresh_chunk
    if TOTAL < FRESH:
        raise ValueError("total context must be at least the fresh chunk")
    CACHED = TOTAL - FRESH
    torch.cuda.set_device(0)
    from dlengine.runtime.kernel.triton.hopper.fp8_utils import (
        restore_mla_fp8_cache_rows,
    )
    from flash_attn.cute import flash_attn_varlen_func

    # Persistent packed cache, fresh projections, queries, and expansion
    # weights exist before the activation baseline.
    packed_cache = torch.zeros(CACHED, PACKED, device="cuda", dtype=torch.uint8)
    compressed_fresh = torch.zeros(FRESH, LATENT, device="cuda", dtype=torch.bfloat16)
    kpe_fresh = torch.zeros(FRESH, ROPE, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(FRESH, HEADS, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    kc = torch.empty(LATENT, HEADS * NOPE, device="cuda", dtype=torch.bfloat16)
    vc = torch.empty(LATENT, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, FRESH], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = [snapshot("baseline", baseline, started)]

    restored = restore_mla_fp8_cache_rows(packed_cache)
    rows.append(snapshot("restore latent cache", baseline, started))
    comp_cached = restored[:, :LATENT]
    kpe_cached = restored[:, LATENT:]

    k_nope_cached = (comp_cached @ kc).view(CACHED, HEADS, NOPE)
    k_cached = torch.cat(
        [k_nope_cached, kpe_cached[:, None, :].expand(-1, HEADS, -1)], dim=-1
    )
    v_cached = (comp_cached @ vc).view(CACHED, HEADS, VALUE)
    rows.append(snapshot("expand cached K/V", baseline, started))

    k_nope_fresh = (compressed_fresh @ kc).view(FRESH, HEADS, NOPE)
    k_fresh = torch.cat(
        [k_nope_fresh, kpe_fresh[:, None, :].expand(-1, HEADS, -1)], dim=-1
    )
    v_fresh = (compressed_fresh @ vc).view(FRESH, HEADS, VALUE)
    k = torch.cat([k_cached, k_fresh], dim=0)
    v = torch.cat([v_cached, v_fresh], dim=0)
    rows.append(snapshot("join cached + fresh", baseline, started))

    out = flash_attn_varlen_func(
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
    if isinstance(out, tuple):
        out = out[0]
    rows.append(snapshot("FlashAttention", baseline, started))
    assert out.shape == (FRESH, HEADS, VALUE)

    output = Path(__file__).parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    result_name = f"mla_cached_total_{TOTAL}_fresh_{FRESH}"
    with (output / f"{result_name}_stages.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "total_context": TOTAL,
        "cached_prefix": CACHED,
        "fresh_chunk": FRESH,
        "cache_layout": "packed mixed FP8/BF16, 656 bytes/token",
        "expanded_k": [TOTAL, HEADS, NOPE + ROPE],
        "expanded_v": [TOTAL, HEADS, VALUE],
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / f"{result_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

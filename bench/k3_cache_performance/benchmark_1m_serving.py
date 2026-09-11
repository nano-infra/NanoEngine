#!/usr/bin/env python3
"""K3 cache reuse at 1M context with 16K serving chunks."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

TOTAL = 1_048_576
CHUNK = 16_384
HEADS = 96
KEY = 128
VALUE = 128
ROPE = 64
LATENT = 512
PACKED = 656
HIT_CHUNKS = (0, 32, 48, 56, 60, 62, 63)


def event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure_kda(warmup: int) -> list[float]:
    from dlengine.runtime.kernel.triton.fla.kda import chunk_kda

    q = torch.randn(1, CHUNK, HEADS, KEY, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, CHUNK, HEADS, VALUE, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(q)
    beta = torch.rand(1, CHUNK, HEADS, device="cuda", dtype=torch.float32)
    state = torch.randn(1, HEADS, VALUE, KEY, device="cuda", dtype=torch.bfloat16)
    state_seed = state.clone()
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, CHUNK], device="cuda", dtype=torch.int32)
    a_log = torch.zeros(HEADS, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(HEADS * KEY, device="cuda", dtype=torch.float32)

    def forward():
        return chunk_kda(
            q, k, v, g, beta,
            initial_state=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

    for _ in range(warmup):
        state.copy_(state_seed)
        forward()
    torch.cuda.synchronize()
    state.copy_(state_seed)
    torch.cuda.synchronize()
    return [event_ms(forward) for _ in range(TOTAL // CHUNK)]


def measure_mla(warmup: int) -> list[float]:
    from flash_attn.cute import flash_attn_varlen_func
    from dlengine.runtime.kernel.triton.hopper.fp8_utils import restore_mla_fp8_cache_rows
    from dlengine.runtime.layers.backends.attention.mla_utils import chunked_prefix_mla_attention

    cache = torch.zeros(TOTAL - CHUNK, PACKED, device="cuda", dtype=torch.uint8)
    compressed = torch.zeros(CHUNK, LATENT, device="cuda", dtype=torch.bfloat16)
    kpe = torch.zeros(CHUNK, ROPE, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(CHUNK, HEADS, KEY + ROPE, device="cuda", dtype=torch.bfloat16)
    kc = torch.zeros(LATENT, HEADS * KEY, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros(LATENT, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    k_fresh = torch.cat([(compressed @ kc).view(CHUNK, HEADS, KEY), kpe[:, None, :].expand(-1, HEADS, -1)], dim=-1)
    v_fresh = (compressed @ vc).view(CHUNK, HEADS, VALUE)
    cu_q = torch.tensor([0, CHUNK], device="cuda", dtype=torch.int32)

    def forward(cached: int):
        if cached == 0:
            result = flash_attn_varlen_func(
                q, k_fresh, v_fresh,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_q,
                max_seqlen_q=CHUNK,
                max_seqlen_k=CHUNK,
                softmax_scale=(KEY + ROPE) ** -0.5,
                causal=True,
            )
            return result[0] if isinstance(result, tuple) else result
        restored = restore_mla_fp8_cache_rows(cache[:cached])
        cached_lens = torch.tensor([cached], device="cuda", dtype=torch.int32)
        return chunked_prefix_mla_attention(
            q, k_fresh, v_fresh, restored, cached_lens, cu_q, kc, vc,
            chunk_size=131_072,
            softmax_scale=(KEY + ROPE) ** -0.5,
            attention_func=flash_attn_varlen_func,
        )

    # Compile both fresh-only and cached-prefix paths before measurement.
    for cached in (0, TOTAL - CHUNK):
        for _ in range(warmup):
            forward(cached)
            torch.cuda.synchronize()
    return [event_ms(lambda cached=i * CHUNK: forward(cached)) for i in range(TOTAL // CHUNK)]


def summarize(component: str, chunk_ms: list[float]) -> list[dict]:
    baseline = sum(chunk_ms)
    rows = []
    for cached_chunks in HIT_CHUNKS:
        remaining = sum(chunk_ms[cached_chunks:])
        rows.append({
            "component": component,
            "total_context": TOTAL,
            "chunk_tokens": CHUNK,
            "cached_prefix": cached_chunks * CHUNK,
            "effective_hit_rate": cached_chunks * CHUNK / TOTAL,
            "remaining_chunks": len(chunk_ms) - cached_chunks,
            "remaining_prefill_ms": remaining,
            "speedup_vs_no_hit": baseline / remaining,
            "latency_reduction": 1 - remaining / baseline,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    rows = summarize("KDA recurrence", measure_kda(args.warmup))
    rows += summarize("MLA cached attention", measure_mla(args.warmup))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cache_1m_serving.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_context": TOTAL,
        "prefill_chunk": CHUNK,
        "kda_shape": [HEADS, KEY, VALUE],
        "mla_cache": "mixed FP8/BF16, 656 bytes/token",
        "mla_prefix_chunk": 131_072,
        "measurement": "sum of 64 individually timed serving chunks; kernel/core boundaries only",
    }
    (args.output_dir / "cache_1m_serving_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

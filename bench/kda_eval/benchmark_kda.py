#!/usr/bin/env python3
"""Benchmark NanoDeploy's KDA prefill/decode kernels on CUDA.

The prefill sweep models a reusable recurrent prefix state.  For a logical
sequence length L and hit rate r, only ``L - floor(L*r)`` suffix tokens are
timed.  Prefix-state construction/restoration and input allocation are outside
the timed region.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch


def parse_csv_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def cuda_times(fn, restore, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        restore()
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        restore()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def allocation_inputs(tokens: int, batch: int, heads: int, kdim: int, vdim: int):
    # One packed token dimension and cu_seqlens are the exact layout used by
    # FlashInferKda.forward for variable-length prefill.
    total = tokens * batch
    q = torch.randn(1, total, heads, kdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, heads, vdim, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(q)
    beta = torch.rand(1, total, heads, device="cuda", dtype=torch.float32)
    cu = torch.arange(
        0, total + 1, tokens, device="cuda", dtype=torch.int32
    )
    return q, k, v, g, beta, cu


def benchmark_prefill(args) -> list[dict]:
    from dlengine.runtime.kernel.triton.fla.kda import chunk_kda

    rows = []
    heads, kdim, vdim = args.heads, args.kdim, args.vdim
    A_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(heads * kdim, device="cuda", dtype=torch.float32)
    for batch in args.batch_sizes:
        initial = torch.empty(
            batch, heads, vdim, kdim, device="cuda", dtype=torch.bfloat16
        )
        baseline = torch.randn_like(initial)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        for logical_len in args.lengths:
            seen_cached_lengths = set()
            for requested_hit_rate in args.hit_rates:
                cached = min(
                    logical_len - 1,
                    int(logical_len * requested_hit_rate / args.block_size)
                    * args.block_size,
                )
                fresh = logical_len - cached
                # Block alignment can collapse requested rates to one executable case.
                if cached in seen_cached_lengths:
                    continue
                seen_cached_lengths.add(cached)
                effective_hit_rate = cached / logical_len
                q, k, v, g, beta, cu = allocation_inputs(
                    fresh, batch, heads, kdim, vdim
                )

                def restore():
                    initial.copy_(baseline)

                def run():
                    return chunk_kda(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        initial_state=initial,
                        initial_state_indices=indices,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=cu,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        lower_bound=args.lower_bound,
                    )

                times = cuda_times(run, restore, args.warmup, args.repeats)
                p50, p90, p99 = (
                    percentile(times, 0.50),
                    percentile(times, 0.90),
                    percentile(times, 0.99),
                )
                rows.append(
                    {
                        "mode": "prefill_core",
                        "logical_length": logical_len,
                        "requested_hit_rate": requested_hit_rate,
                        "effective_hit_rate": effective_hit_rate,
                        "cached_tokens_per_seq": cached,
                        "fresh_tokens_per_seq": fresh,
                        "batch_size": batch,
                        "p50_ms": p50,
                        "p90_ms": p90,
                        "p99_ms": p99,
                        "fresh_tokens_per_s": batch * fresh / (p50 / 1000),
                        "logical_tokens_per_s": batch * logical_len / (p50 / 1000),
                    }
                )
                print(
                    f"prefill L={logical_len:6d} hit={effective_hit_rate:7.2%} "
                    f"fresh={fresh:6d} B={batch:2d} p50={p50:9.3f} ms",
                    flush=True,
                )
                del q, k, v, g, beta, cu
    return rows


def benchmark_decode(args) -> list[dict]:
    from dlengine.runtime.kernel.triton.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )

    rows = []
    H, K, V = args.heads, args.kdim, args.vdim
    A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(H * K, device="cuda", dtype=torch.float32)
    for batch in args.decode_batch_sizes:
        mixed = torch.randn(batch, 2 * H * K + H * V, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(batch, H * K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, H, device="cuda", dtype=torch.bfloat16)
        initial = torch.empty(batch, H, V, K, device="cuda", dtype=torch.bfloat16)
        baseline = torch.randn_like(initial)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        out = torch.empty(batch, 1, H, V, device="cuda", dtype=torch.bfloat16)

        def restore():
            initial.copy_(baseline)

        def run():
            fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=K**-0.5,
                initial_state=initial,
                out=out,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                lower_bound=args.lower_bound,
            )

        times = cuda_times(run, restore, args.warmup, args.repeats)
        p50, p90, p99 = [percentile(times, q) for q in (0.50, 0.90, 0.99)]
        rows.append(
            {
                "mode": "decode_core",
                "logical_length": 1,
                "requested_hit_rate": 1.0,
                "effective_hit_rate": 1.0,
                "cached_tokens_per_seq": 0,
                "fresh_tokens_per_seq": 1,
                "batch_size": batch,
                "p50_ms": p50,
                "p90_ms": p90,
                "p99_ms": p99,
                "fresh_tokens_per_s": batch / (p50 / 1000),
                "logical_tokens_per_s": batch / (p50 / 1000),
            }
        )
        print(f"decode B={batch:3d} p50={p50:9.3f} ms", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lengths", default="128,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--hit-rates", default="0,0.25,0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--decode-batch-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kdim", type=int, default=128)
    parser.add_argument("--vdim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    args.lengths = parse_csv_numbers(args.lengths, int)
    args.hit_rates = parse_csv_numbers(args.hit_rates, float)
    args.batch_sizes = parse_csv_numbers(args.batch_sizes, int)
    args.decode_batch_sizes = parse_csv_numbers(args.decode_batch_sizes, int)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "heads": args.heads,
        "kdim": args.kdim,
        "vdim": args.vdim,
        "block_size": args.block_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    try:
        import flashinfer
        metadata["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    except Exception as exc:
        metadata["flashinfer"] = f"unavailable: {exc}"

    rows = []
    if not args.skip_prefill:
        rows.extend(benchmark_prefill(args))
    if not args.skip_decode:
        rows.extend(benchmark_decode(args))
    fieldnames = list(rows[0])
    with (args.output_dir / "kda_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()

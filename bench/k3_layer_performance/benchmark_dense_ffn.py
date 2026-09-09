#!/usr/bin/env python3
"""Benchmark the single K3 dense SwiGLU FFN on one GPU."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = 7168
INTERMEDIATE = 33792


def median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    return sorted(values)[len(values) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="1024,2048,4096,8192,16384")
    parser.add_argument("--batches", default="1,8,16,32,64,128,256")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/dense_ffn.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # K3's only dense FFN uses a 33,792-wide gated intermediate.
    gate = torch.zeros(INTERMEDIATE, HIDDEN, device=device, dtype=dtype)
    up = torch.zeros_like(gate)
    down = torch.zeros(HIDDEN, INTERMEDIATE, device=device, dtype=dtype)

    def run(x):
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    rows = []
    for mode, values in (("prefill", args.chunks), ("decode", args.batches)):
        for value in (int(v) for v in values.split(",")):
            x = torch.zeros(value, HIDDEN, device=device, dtype=dtype)
            latency = median_ms(lambda: run(x), args.warmup, args.repeats)
            tokens_per_s = value * 1000.0 / latency
            # 2*(gate/up/down) matrix FLOPs plus two elementwise projections.
            flops = value * (4 * HIDDEN * INTERMEDIATE + 3 * INTERMEDIATE)
            rows.append({"mode": mode, "tokens": value, "latency_ms": latency,
                         "tokens_per_s": tokens_per_s, "gflops": flops / latency / 1e6})
            print(rows[-1], flush=True)
            del x
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()

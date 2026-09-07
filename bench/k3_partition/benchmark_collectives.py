#!/usr/bin/env python3
"""NCCL microbenchmark for K3 hidden-state boundary collectives."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist

HIDDEN = 7168
DTYPE = torch.bfloat16


def timed(op, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[16, 128, 1024, 4096, 16384]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rows = []

    for requested in args.tokens:
        tokens = ((requested + world - 1) // world) * world
        full = torch.randn(tokens, HIDDEN, device="cuda", dtype=DTYPE)
        shard = torch.randn(tokens // world, HIDDEN, device="cuda", dtype=DTYPE)
        gathered = torch.empty_like(full)
        reduced = torch.empty_like(shard)
        exchanged = torch.empty_like(full)
        operations = {
            "all_reduce": lambda: dist.all_reduce(full),
            "reduce_scatter": lambda: dist.reduce_scatter_tensor(reduced, full),
            "all_gather": lambda: dist.all_gather_into_tensor(gathered, shard),
            "all_to_all": lambda: dist.all_to_all_single(exchanged, full),
        }
        logical_bytes = tokens * HIDDEN * torch.tensor([], dtype=DTYPE).element_size()
        for name, op in operations.items():
            latency_ms = timed(op, args.warmup, args.repeats)
            if rank == 0:
                rows.append(
                    {
                        "world_size": world,
                        "requested_tokens": requested,
                        "padded_tokens": tokens,
                        "collective": name,
                        "logical_payload_bytes": logical_bytes,
                        "latency_ms": latency_ms,
                        "logical_payload_gbps": logical_bytes / latency_ms / 1e6,
                        "gpu": torch.cuda.get_device_name(),
                    }
                )
        del full, shard, gathered, reduced, exchanged

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

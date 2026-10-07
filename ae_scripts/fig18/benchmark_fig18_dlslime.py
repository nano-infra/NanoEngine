#!/usr/bin/env python3
"""Benchmark DLSlime Hao Basic against NCCL for the Fig. 18 workload."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist


PAPER_WORLD_SIZE = 8
DLSLIME_IMPLEMENTATION = "dlslime_hao_basic"
NCCL_IMPLEMENTATION = "nccl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dense Fig. 18 all-to-all workload with DLSlime "
            "AllToAllBuffer/KernelImpl.Basic and PyTorch NCCL."
        )
    )
    parser.add_argument("--batch-sizes", default="2,4,8,16,32,64,128")
    parser.add_argument("--feature-size", type=int, default=73_728)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--require-dlslime-root",
        default="",
        help=(
            "If non-empty, require the imported dlslime package to resolve "
            "under this source checkout."
        ),
    )
    args = parser.parse_args()
    for name in ("feature_size", "warmup_iters", "iters", "rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        batch_sizes = parse_int_list(args.batch_sizes)
    except ValueError as error:
        parser.error(str(error))
    if not batch_sizes:
        parser.error("--batch-sizes must contain at least one positive integer")
    if any(value <= 0 for value in batch_sizes):
        parser.error("--batch-sizes must contain only positive integers")
    return args


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.replace(" ", "").split(",") if item]
    except ValueError as error:
        raise ValueError(
            f"Expected a comma-separated integer list, got {value!r}"
        ) from error


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty latency sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "min_us": min(values),
        "p50_us": percentile(values, 50.0),
        "p90_us": percentile(values, 90.0),
        "mean_us": statistics.fmean(values),
        "max_us": max(values),
    }


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != PAPER_WORLD_SIZE:
        raise RuntimeError(
            f"Fig. 18 requires world_size={PAPER_WORLD_SIZE}; got {world_size}."
        )
    return rank, local_rank, world_size, device


def require_path_below(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Resolved dlslime to {path}, which is outside required root {root}."
        ) from error


def load_dlslime_api(required_root: str) -> tuple[Any, Any, str]:
    try:
        import dlslime
    except ImportError as error:
        raise RuntimeError(
            "DLSlime is not importable in this Python environment."
        ) from error

    buffer_cls = getattr(dlslime, "AllToAllBuffer", None)
    kernel_impl = getattr(dlslime, "KernelImpl", None)
    basic_impl = getattr(kernel_impl, "Basic", None) if kernel_impl is not None else None
    if buffer_cls is None or basic_impl is None:
        raise RuntimeError(
            "This DLSlime build does not expose AllToAllBuffer and KernelImpl.Basic; "
            "rebuild it with CUDA intra-node ops enabled."
        )

    dlslime_file = Path(dlslime.__file__).resolve()
    if required_root:
        root = Path(required_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DLSlime source root not found: {root}")
        require_path_below(dlslime_file, root)
    return buffer_cls, basic_impl, str(dlslime_file)


def connect_dlslime_buffer(
    buffer_cls: Any,
    rank: int,
    world_size: int,
    batch_size: int,
    feature_size: int,
    dtype: torch.dtype,
) -> Any:
    element_size = torch.empty((), dtype=dtype).element_size()
    buffer_size_bytes = world_size * batch_size * feature_size * element_size
    buffer = buffer_cls(rank, world_size, batch_size, buffer_size_bytes)
    local_handle = buffer.get_ipc_handle_info()
    peer_handles = [None for _ in range(world_size)]
    dist.all_gather_object(peer_handles, local_handle)
    buffer.connect_full_mesh(peer_handles)
    return buffer


def build_nccl_runner(
    input_tensor: torch.Tensor,
    world_size: int,
    batch_size: int,
    feature_size: int,
) -> Callable[[], torch.Tensor]:
    output_tensor = torch.empty_like(input_tensor)

    def run() -> torch.Tensor:
        dist.all_to_all_single(output_tensor, input_tensor)
        return output_tensor.view(world_size, batch_size, feature_size)

    return run


def check_correctness(
    collective: Callable[[], torch.Tensor],
    input_tensor: torch.Tensor,
    world_size: int,
    batch_size: int,
    feature_size: int,
    local_rank: int,
) -> None:
    actual = collective().clone()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    expected_flat = torch.empty_like(input_tensor)
    dist.all_to_all_single(expected_flat, input_tensor)
    expected = expected_flat.view(world_size, batch_size, feature_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dist.barrier(device_ids=[local_rank])


def measure_collective(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    implementation: str,
    batch_size: int,
    feature_size: int,
    dtype: torch.dtype,
    warmup_iters: int,
    measure_iters: int,
    rounds: int,
    run_check: bool,
    buffer_cls: Any,
    basic_impl: Any,
    dlslime_file: str,
) -> dict[str, object]:
    element_size = torch.empty((), dtype=dtype).element_size()
    row_bytes = feature_size * element_size
    if row_bytes % 16 != 0:
        raise ValueError(
            f"feature_size * element_size must be 16-byte aligned, got {row_bytes}"
        )

    input_tensor = torch.randn(
        (world_size * batch_size, feature_size), dtype=dtype, device=device
    )
    buffer = None
    if implementation == DLSLIME_IMPLEMENTATION:
        buffer = connect_dlslime_buffer(
            buffer_cls, rank, world_size, batch_size, feature_size, dtype
        )

        def collective() -> torch.Tensor:
            return buffer.all_to_all(
                input_tensor,
                impl=basic_impl,
                is_transpose=True,
            )

        backend = "dlslime"
        kernel_impl = "basic"
    elif implementation == NCCL_IMPLEMENTATION:
        collective = build_nccl_runner(
            input_tensor, world_size, batch_size, feature_size
        )
        backend = "pytorch_nccl"
        kernel_impl = "all_to_all_single"
    else:
        raise ValueError(f"Unsupported implementation: {implementation}")

    if run_check:
        check_correctness(
            collective,
            input_tensor,
            world_size,
            batch_size,
            feature_size,
            local_rank,
        )

    for _ in range(warmup_iters):
        collective()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    local_round_latencies: list[float] = []
    max_rank_round_latencies: list[float] = []

    for _ in range(rounds):
        dist.barrier(device_ids=[local_rank])
        start_event.record()
        for _ in range(measure_iters):
            collective()
        end_event.record()
        end_event.synchronize()
        local_latency_us = (
            start_event.elapsed_time(end_event) * 1000.0 / measure_iters
        )

        latency_tensor = torch.tensor(
            [local_latency_us], dtype=torch.float64, device=device
        )
        gathered = [torch.empty_like(latency_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, latency_tensor)
        rank_latencies = [float(value.item()) for value in gathered]
        local_round_latencies.append(local_latency_us)
        max_rank_round_latencies.append(max(rank_latencies))

    dist.barrier(device_ids=[local_rank])
    local_summary = summarize(local_round_latencies)
    end_to_end_summary = summarize(max_rank_round_latencies)
    input_bytes = input_tensor.numel() * input_tensor.element_size()
    bytes_per_rank = input_bytes * (world_size - 1) // world_size
    p50_s = end_to_end_summary["p50_us"] / 1_000_000.0

    dist.barrier(device_ids=[local_rank])
    collective = None
    if buffer is not None:
        del buffer
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_rank])

    return {
        "mode": "alltoall",
        "implementation": implementation,
        "backend": backend,
        "kernel_impl": kernel_impl,
        "dlslime_file": dlslime_file,
        "world_size": world_size,
        "batch_size": batch_size,
        "feature_size": feature_size,
        "dtype": str(dtype).replace("torch.", ""),
        "row_bytes": row_bytes,
        "bytes_per_rank": bytes_per_rank,
        "rounds": rounds,
        "iters_per_round": measure_iters,
        "warmup_iters": warmup_iters,
        "checked": run_check,
        "rank0_min_us": local_summary["min_us"],
        "rank0_p50_us": local_summary["p50_us"],
        "rank0_p90_us": local_summary["p90_us"],
        "rank0_mean_us": local_summary["mean_us"],
        "rank0_max_us": local_summary["max_us"],
        "e2e_min_us": end_to_end_summary["min_us"],
        "e2e_p50_us": end_to_end_summary["p50_us"],
        "e2e_p90_us": end_to_end_summary["p90_us"],
        "e2e_mean_us": end_to_end_summary["mean_us"],
        "e2e_max_us": end_to_end_summary["max_us"],
        "effective_gbps_per_rank_p50": (bytes_per_rank / 1e9) / p50_s,
        "raw_rank0_us": local_round_latencies,
        "raw_e2e_us": max_rank_round_latencies,
    }


def write_outputs(
    rows: list[dict[str, object]], csv_path: Path, json_path: Path
) -> None:
    csv_path = csv_path.expanduser().resolve()
    json_path = json_path.expanduser().resolve()
    if csv_path == json_path:
        raise ValueError("--csv-output and --json-output must be different paths")
    for path in (csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")

    summary_keys = [key for key in rows[0] if not key.startswith("raw_")]
    with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in summary_keys})
    with json_path.open("x", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2)

    print(f"wrote_csv={csv_path}")
    print(f"wrote_json={json_path}")


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    try:
        buffer_cls, basic_impl, dlslime_file = load_dlslime_api(
            args.require_dlslime_root
        )
        dtype = dtype_from_name(args.dtype)
        rows: list[dict[str, object]] = []
        for batch_size in parse_int_list(args.batch_sizes):
            for implementation in (DLSLIME_IMPLEMENTATION, NCCL_IMPLEMENTATION):
                row = measure_collective(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    device=device,
                    implementation=implementation,
                    batch_size=batch_size,
                    feature_size=args.feature_size,
                    dtype=dtype,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.iters,
                    rounds=args.rounds,
                    run_check=args.check,
                    buffer_cls=buffer_cls,
                    basic_impl=basic_impl,
                    dlslime_file=dlslime_file,
                )
                if rank == 0:
                    rows.append(row)
                    print(
                        f"impl={implementation} mode=alltoall ws={world_size} "
                        f"bs={batch_size} feature={args.feature_size} "
                        f"e2e_p50={row['e2e_p50_us']:.2f}us "
                        f"gbps={row['effective_gbps_per_rank_p50']:.2f}",
                        flush=True,
                    )

        if rank == 0:
            write_outputs(rows, args.csv_output, args.json_output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

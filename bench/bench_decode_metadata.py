"""Compare decode metadata transport and unpack paths on one CUDA device.

Run from the repository root:

    python3 -m bench.bench_decode_metadata --output results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dlengine.kernel.jit.nano.decode_metadata_bench import (
    launch_device,
    launch_mapped,
    register_mapped,
    unregister_mapped,
)

MAGIC = 0x444D444C
VERSION = 1
HEADER = struct.Struct("<IHH" + "I" * 14)
ALIGNMENT = 16


@dataclass(frozen=True)
class DecodePayload:
    data: bytes
    fields: dict[str, np.ndarray]
    expected: dict[str, np.ndarray]
    max_num_seqs: int
    max_num_blocks: int


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_payload(
    batch_size: int,
    context_len: int,
    block_size: int,
    max_num_seqs: int,
    seed: int,
) -> DecodePayload:
    if not 0 < batch_size <= max_num_seqs:
        raise ValueError("batch_size must be in [1, max_num_seqs]")
    if context_len <= 0 or block_size <= 0:
        raise ValueError("context_len and block_size must be positive")

    rng = np.random.default_rng(seed)
    position_jitter = min(block_size - 1, context_len - 1)
    positions = (
        context_len
        - 1
        - rng.integers(0, position_jitter + 1, batch_size, dtype=np.int64)
    )
    row_lengths = ((positions + block_size) // block_size).astype(np.uint32)
    max_num_blocks = int(row_lengths.max())
    row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    np.cumsum(row_lengths, out=row_offsets[1:])
    block_ids = np.arange(1, int(row_offsets[-1]) + 1, dtype=np.int32)
    block_ids += seed * 100_000

    fields = {
        "input_ids": rng.integers(1, 150_000, batch_size, dtype=np.int64),
        "positions": positions,
        "temperatures": rng.uniform(0.1, 1.5, batch_size).astype(np.float32),
        "state_slots": rng.integers(-2, max_num_seqs + 2, batch_size, dtype=np.int64),
        "hisparse_slots": rng.integers(
            -2, max_num_seqs + 2, batch_size, dtype=np.int64
        ),
        "block_row_offsets": row_offsets,
        "block_ids": block_ids,
    }

    offsets: dict[str, int] = {}
    cursor = HEADER.size
    for name, array in fields.items():
        cursor = _align(cursor)
        offsets[name] = cursor
        cursor += array.nbytes
    payload_bytes = _align(cursor)
    payload = bytearray(payload_bytes)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        0,
        payload_bytes,
        batch_size,
        max_num_seqs,
        max_num_blocks,
        block_size,
        offsets["input_ids"],
        offsets["positions"],
        offsets["temperatures"],
        offsets["state_slots"],
        offsets["hisparse_slots"],
        offsets["block_row_offsets"],
        offsets["block_ids"],
        0,
        0,
    )
    payload[: HEADER.size] = header
    for name, array in fields.items():
        offset = offsets[name]
        payload[offset : offset + array.nbytes] = array.tobytes()

    expected = {
        "input_ids": np.zeros(max_num_seqs, dtype=np.int64),
        "positions": np.zeros(max_num_seqs, dtype=np.int64),
        "temperatures": np.ones(max_num_seqs, dtype=np.float32),
        "state_slots": np.full(max_num_seqs, max_num_seqs, dtype=np.int64),
        "hisparse_slots": np.full(max_num_seqs, max_num_seqs, dtype=np.int64),
        "slot_mapping": np.full(max_num_seqs, -1, dtype=np.int32),
        "context_lens": np.zeros(max_num_seqs, dtype=np.int32),
        "block_tables": np.zeros((max_num_seqs, max_num_blocks), dtype=np.int32),
    }
    expected["input_ids"][:batch_size] = fields["input_ids"]
    expected["positions"][:batch_size] = positions
    expected["temperatures"][:batch_size] = fields["temperatures"]
    for name in ("state_slots", "hisparse_slots"):
        values = fields[name]
        expected[name][:batch_size] = np.where(
            (values >= 0) & (values < max_num_seqs), values, max_num_seqs
        )
    expected["context_lens"][:batch_size] = positions.astype(np.int32) + 1
    for row in range(batch_size):
        begin = int(row_offsets[row])
        end = int(row_offsets[row + 1])
        expected["block_tables"][row, : end - begin] = block_ids[begin:end]
        logical_block = int(positions[row]) // block_size
        expected["slot_mapping"][row] = (
            int(block_ids[begin + logical_block]) * block_size
            + int(positions[row]) % block_size
        )

    return DecodePayload(bytes(payload), fields, expected, max_num_seqs, max_num_blocks)


OUTPUT_NAMES = (
    "input_ids",
    "positions",
    "temperatures",
    "state_slots",
    "hisparse_slots",
    "slot_mapping",
    "context_lens",
    "block_tables",
)


def allocate_outputs(payload: DecodePayload) -> tuple[torch.Tensor, ...]:
    n = payload.max_num_seqs
    return (
        torch.empty(n, dtype=torch.int64, device="cuda"),
        torch.empty(n, dtype=torch.int64, device="cuda"),
        torch.empty(n, dtype=torch.float32, device="cuda"),
        torch.empty(n, dtype=torch.int64, device="cuda"),
        torch.empty(n, dtype=torch.int64, device="cuda"),
        torch.empty(n, dtype=torch.int32, device="cuda"),
        torch.empty(n, dtype=torch.int32, device="cuda"),
        torch.empty((n, payload.max_num_blocks), dtype=torch.int32, device="cuda"),
    )


def assert_outputs(payload: DecodePayload, outputs: tuple[torch.Tensor, ...]) -> None:
    for name, output in zip(OUTPUT_NAMES, outputs, strict=True):
        actual = output.cpu().numpy()
        np.testing.assert_array_equal(actual, payload.expected[name], err_msg=name)


def _percentiles(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50_us": float(np.percentile(values, 50)),
        "p99_us": float(np.percentile(values, 99)),
        "mean_us": float(values.mean()),
    }


def _event_time_us(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end) * 1000)


def _run_mapped(
    variants: list[DecodePayload],
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    exemplar = variants[0]
    slab = torch.empty(len(exemplar.data), dtype=torch.uint8)
    sources = [
        torch.frombuffer(bytearray(item.data), dtype=torch.uint8) for item in variants
    ]
    outputs = allocate_outputs(exemplar)
    prepare_samples: list[float] = []
    gpu_samples: list[float] = []
    register_mapped(slab)
    try:
        for step in range(warmup + iterations):
            source = sources[step % len(sources)]
            begin = time.perf_counter_ns()
            slab.copy_(source)
            prepare_us = (time.perf_counter_ns() - begin) / 1000
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch_mapped(slab, outputs)
            end.record()
            gpu_us = _event_time_us(start, end)
            if step >= warmup:
                prepare_samples.append(prepare_us)
                gpu_samples.append(gpu_us)
        assert_outputs(variants[(warmup + iterations - 1) % len(variants)], outputs)
    finally:
        unregister_mapped(slab)
    return {
        "host_prepare": _percentiles(prepare_samples),
        "gpu": _percentiles(gpu_samples),
    }


def _run_memcpy(
    variants: list[DecodePayload],
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    exemplar = variants[0]
    slab = torch.empty(len(exemplar.data), dtype=torch.uint8, pin_memory=True)
    device_payload = torch.empty_like(slab, device="cuda")
    sources = [
        torch.frombuffer(bytearray(item.data), dtype=torch.uint8) for item in variants
    ]
    outputs = allocate_outputs(exemplar)
    prepare_samples: list[float] = []
    copy_samples: list[float] = []
    kernel_samples: list[float] = []
    gpu_samples: list[float] = []
    for step in range(warmup + iterations):
        source = sources[step % len(sources)]
        begin = time.perf_counter_ns()
        slab.copy_(source)
        prepare_us = (time.perf_counter_ns() - begin) / 1000
        start = torch.cuda.Event(enable_timing=True)
        copied = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        device_payload.copy_(slab, non_blocking=True)
        copied.record()
        launch_device(device_payload, outputs)
        end.record()
        total_us = _event_time_us(start, end)
        if step >= warmup:
            prepare_samples.append(prepare_us)
            copy_samples.append(float(start.elapsed_time(copied) * 1000))
            kernel_samples.append(float(copied.elapsed_time(end) * 1000))
            gpu_samples.append(total_us)
    assert_outputs(variants[(warmup + iterations - 1) % len(variants)], outputs)
    return {
        "host_prepare": _percentiles(prepare_samples),
        "copy": _percentiles(copy_samples),
        "kernel": _percentiles(kernel_samples),
        "gpu": _percentiles(gpu_samples),
    }


def _run_per_field(
    variants: list[DecodePayload],
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    prepare_samples: list[float] = []
    gpu_samples: list[float] = []
    final: dict[str, torch.Tensor] = {}
    torch_dtypes = {
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float32): torch.float32,
    }
    for step in range(warmup + iterations):
        payload = variants[step % len(variants)]
        begin = time.perf_counter_ns()
        host_fields = {
            name: torch.tensor(
                value.tolist(),
                dtype=torch_dtypes[value.dtype],
                pin_memory=True,
            )
            for name, value in payload.expected.items()
        }
        prepare_us = (time.perf_counter_ns() - begin) / 1000
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        final = {
            name: value.cuda(non_blocking=True) for name, value in host_fields.items()
        }
        end.record()
        gpu_us = _event_time_us(start, end)
        if step >= warmup:
            prepare_samples.append(prepare_us)
            gpu_samples.append(gpu_us)
    for name, output in final.items():
        np.testing.assert_array_equal(output.cpu().numpy(), payload.expected[name])
    return {
        "host_prepare": _percentiles(prepare_samples),
        "gpu": _percentiles(gpu_samples),
    }


def run_case(
    batch_size: int,
    context_len: int,
    block_size: int,
    max_num_seqs: int,
    warmup: int,
    iterations: int,
    variants: int,
) -> dict[str, object]:
    payloads = [
        build_payload(batch_size, context_len, block_size, max_num_seqs, seed)
        for seed in range(variants)
    ]
    sizes = {len(item.data) for item in payloads}
    if len(sizes) != 1:
        raise RuntimeError("payload variants must have a stable slab size")
    return {
        "batch_size": batch_size,
        "context_len": context_len,
        "block_size": block_size,
        "max_num_seqs": max_num_seqs,
        "max_num_blocks": payloads[0].max_num_blocks,
        "payload_bytes": len(payloads[0].data),
        "paths": {
            "mapped_uva": _run_mapped(payloads, warmup, iterations),
            "aggregated_memcpy": _run_memcpy(payloads, warmup, iterations),
            "per_field_torch": _run_per_field(payloads, warmup, iterations),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--variants", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = (
        commit_result.stdout.strip() if commit_result.returncode == 0 else "unavailable"
    )
    result = {
        "schema_version": 1,
        "environment": {
            "gpu": device.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_bytes": device.total_memory,
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "git_commit": commit,
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "variants": args.variants,
            "command": [
                sys.executable,
                "-m",
                "bench.bench_decode_metadata",
                *sys.argv[1:],
            ],
        },
        "cases": [],
    }
    for context_len in args.context_lens:
        for batch_size in args.batch_sizes:
            print(f"running batch={batch_size} context={context_len}", flush=True)
            result["cases"].append(
                run_case(
                    batch_size,
                    context_len,
                    args.block_size,
                    args.max_num_seqs,
                    args.warmup,
                    args.iterations,
                    args.variants,
                )
            )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prototype one-launch Triton fusion for SP Graph routing metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from triton.runtime import driver as triton_driver


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import nanodeploy
from nanodeploy.worker.sp_graph_policy import (
    copy_graph_actual_attn_bs,
    copy_graph_q_dst_rows,
    materialize_sp_graph_padding,
)


BLOCK_SIZE = 256
REFERENCE_RANGE = "nanodeploy.prototype.routing_metadata.reference"
FUSED_RANGE = "nanodeploy.prototype.routing_metadata.fused"
DESTINATION_NAMES = (
    "q_dst_row_indices",
    "actual_attn_bs",
    "context_lens",
    "global_context_lens",
    "context_lens_for_attn",
    "block_tables",
    "res_slice_get_to_buffer_output",
    "res_slice_fill_to_buffer_output",
    "res_to_buffer_output_mask",
)


@triton.jit(
    do_not_specialize=[
        "q_work",
        "actual_attn_bs",
        "graph_attn_bs",
        "actual_master_bs",
        "graph_master_bs",
        "block_table_width",
        "sp_rank",
        "max_num_seqs",
    ]
)
def routing_metadata_fusion_kernel(
    q_dst_source_ptr,
    graph_q_dst_ptr,
    graph_actual_attn_bs_ptr,
    actual_block_tables_ptr,
    graph_context_lens_ptr,
    graph_global_context_lens_ptr,
    graph_context_lens_for_attn_ptr,
    graph_block_tables_ptr,
    graph_res_get_ptr,
    graph_res_fill_ptr,
    graph_res_mask_ptr,
    q_work,
    actual_attn_bs,
    graph_attn_bs,
    actual_master_bs,
    graph_master_bs,
    block_table_width,
    sp_rank,
    max_num_seqs,
    actual_block_stride_row,
    actual_block_stride_col,
    context_lens_stride_sp,
    context_lens_stride_row,
    global_context_lens_stride_sp,
    global_context_lens_stride_row,
    context_lens_for_attn_stride,
    graph_block_stride_row,
    graph_block_stride_col,
    res_get_stride,
    res_fill_stride,
    res_mask_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    q_mask = offsets < q_work
    q_values = tl.load(q_dst_source_ptr + offsets, mask=q_mask)
    tl.store(graph_q_dst_ptr + offsets, q_values, mask=q_mask)

    tl.store(
        graph_actual_attn_bs_ptr + offsets,
        actual_attn_bs,
        mask=offsets == 0,
    )

    attention_tail = graph_attn_bs - actual_attn_bs
    attention_row_mask = offsets < attention_tail
    tl.store(
        graph_context_lens_for_attn_ptr
        + (actual_attn_bs + offsets) * context_lens_for_attn_stride,
        1,
        mask=attention_row_mask,
    )

    attention_work = attention_tail * block_table_width
    block_mask = offsets < attention_work
    tail_rows = offsets // block_table_width
    columns = offsets - tail_rows * block_table_width
    block_values = tl.load(
        actual_block_tables_ptr
        + 0 * actual_block_stride_row
        + columns * actual_block_stride_col,
        mask=block_mask,
    )
    tl.store(
        graph_block_tables_ptr
        + (actual_attn_bs + tail_rows) * graph_block_stride_row
        + columns * graph_block_stride_col,
        block_values,
        mask=block_mask,
    )

    master_tail = graph_master_bs - actual_master_bs
    master_mask = offsets < master_tail
    master_rows = actual_master_bs + offsets
    tl.store(
        graph_context_lens_ptr
        + sp_rank * context_lens_stride_sp
        + master_rows * context_lens_stride_row,
        1,
        mask=master_mask,
    )
    tl.store(
        graph_global_context_lens_ptr
        + sp_rank * global_context_lens_stride_sp
        + master_rows * global_context_lens_stride_row,
        1,
        mask=master_mask,
    )
    dummy_attention_row = tl.where(
        actual_attn_bs < graph_attn_bs, actual_attn_bs, 0
    )
    tl.store(
        graph_res_get_ptr + master_rows * res_get_stride,
        dummy_attention_row,
        mask=master_mask,
    )
    tl.store(
        graph_res_fill_ptr + master_rows * res_fill_stride,
        sp_rank * max_num_seqs + master_rows,
        mask=master_mask,
    )
    tl.store(
        graph_res_mask_ptr + master_rows * res_mask_stride,
        1,
        mask=master_mask,
    )


@dataclass(frozen=True)
class MetadataCapacities:
    attention_sp: int
    max_num_seqs: int
    max_num_recv_seqs: int
    max_master_bs: int
    max_attn_bs: int
    max_num_blocks: int


@dataclass(frozen=True)
class MetadataCaseShape:
    actual_master_bs: int
    graph_master_bs: int
    actual_attn_bs: int
    graph_attn_bs: int
    block_table_width: int
    sp_rank: int
    max_num_seqs: int


@dataclass(frozen=True)
class MetadataSources:
    q_dst_row_indices: torch.Tensor
    block_tables: torch.Tensor


MetadataDestinations = dict[str, torch.Tensor]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-json",
        type=Path,
        required=True,
        help="ModelRunner runtime-overhead summary containing observed shapes.",
    )
    parser.add_argument(
        "--cases",
        default="no_padding,issue1_padded",
        help="Comma-separated timed cases: no_padding,issue1_padded.",
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=200)
    parser.add_argument("--iterations", type=_positive_int, default=2_000)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Optional Chrome trace for one warmed invocation of each path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest(device: torch.device) -> dict[str, Any]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "git_sha": _git_sha(),
        "nanodeploy_file": str(Path(nanodeploy.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_index": device_index,
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [properties.major, properties.minor],
        "cuda_device_total_memory_bytes": properties.total_memory,
        "pid": os.getpid(),
    }


def _most_common_shape(summary: dict[str, Any]) -> dict[str, int]:
    shapes = summary.get("shape_counts")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("shape JSON has no observed shape_counts")
    shape = shapes[0]
    required = {
        "actual_master_bs",
        "graph_master_bs",
        "actual_attn_bs",
        "graph_attn_bs",
        "block_table_width",
    }
    missing = required.difference(shape)
    if missing:
        raise ValueError(f"observed shape is missing: {sorted(missing)}")
    return {key: int(shape[key]) for key in required}


def _metadata_capacities(summary: dict[str, Any]) -> MetadataCapacities:
    required = (
        "attention_sp",
        "max_num_seqs",
        "max_num_recv_seqs",
        "max_model_len",
        "kvcache_block_size",
    )
    missing = [name for name in required if name not in summary]
    if missing:
        raise ValueError(f"shape JSON is missing capacities: {missing}")
    attention_sp = int(summary["attention_sp"])
    max_num_seqs = int(summary["max_num_seqs"])
    max_num_recv_seqs = int(summary["max_num_recv_seqs"])
    max_model_len = int(summary["max_model_len"])
    block_size = int(summary["kvcache_block_size"])
    if min(
        attention_sp,
        max_num_seqs,
        max_model_len,
        block_size,
    ) <= 0 or max_num_recv_seqs < 0:
        raise ValueError("shape JSON contains invalid non-positive capacities")
    max_master_bs = min(max_num_seqs, 512)
    return MetadataCapacities(
        attention_sp=attention_sp,
        max_num_seqs=max_num_seqs,
        max_num_recv_seqs=max_num_recv_seqs,
        max_master_bs=max_master_bs,
        max_attn_bs=max_master_bs + max_num_recv_seqs,
        max_num_blocks=(max_model_len + block_size - 1) // block_size,
    )


def _timed_case_shape(
    case_name: str,
    observed_shape: dict[str, int],
    *,
    sp_rank: int,
    max_num_seqs: int,
) -> MetadataCaseShape:
    if case_name == "no_padding":
        actual_master_bs = observed_shape["graph_master_bs"]
        actual_attn_bs = observed_shape["graph_attn_bs"]
    elif case_name == "issue1_padded":
        actual_master_bs = observed_shape["actual_master_bs"]
        actual_attn_bs = observed_shape["actual_attn_bs"]
        if (
            actual_master_bs == observed_shape["graph_master_bs"]
            and actual_attn_bs == observed_shape["graph_attn_bs"]
        ):
            raise ValueError(
                "issue1_padded requires an observed shape with a padded tail"
            )
    else:
        raise ValueError(f"Unsupported timed case: {case_name}")
    return MetadataCaseShape(
        actual_master_bs=actual_master_bs,
        graph_master_bs=observed_shape["graph_master_bs"],
        actual_attn_bs=actual_attn_bs,
        graph_attn_bs=observed_shape["graph_attn_bs"],
        block_table_width=observed_shape["block_table_width"],
        sp_rank=sp_rank,
        max_num_seqs=max_num_seqs,
    )


def _validate_case_shape(
    shape: MetadataCaseShape, capacities: MetadataCapacities
) -> None:
    if not 0 < shape.actual_attn_bs <= shape.graph_attn_bs:
        raise ValueError(f"invalid attention batch sizes: {asdict(shape)}")
    if not 0 <= shape.actual_master_bs <= shape.graph_master_bs:
        raise ValueError(f"invalid master batch sizes: {asdict(shape)}")
    if shape.graph_master_bs > capacities.max_master_bs:
        raise ValueError("Graph master batch exceeds configured capacity")
    if shape.graph_attn_bs > capacities.max_attn_bs:
        raise ValueError("Graph attention batch exceeds configured capacity")
    if not 0 < shape.block_table_width <= capacities.max_num_blocks:
        raise ValueError("block-table width exceeds configured capacity")
    if not 0 <= shape.sp_rank < capacities.attention_sp:
        raise ValueError("SP rank exceeds configured attention SP size")
    if shape.max_num_seqs != capacities.max_num_seqs:
        raise ValueError("case max_num_seqs differs from configured capacity")


def _make_case_data(
    capacities: MetadataCapacities,
    shape: MetadataCaseShape,
    *,
    device: torch.device,
    seed: int,
) -> tuple[MetadataDestinations, MetadataSources]:
    _validate_case_shape(shape, capacities)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def random_tensor(*sizes: int) -> torch.Tensor:
        return torch.randint(
            -1_000_000,
            1_000_000,
            sizes,
            dtype=torch.int32,
            device=device,
            generator=generator,
        )

    q_dst_row_indices = random_tensor(
        capacities.attention_sp, capacities.max_num_seqs
    )
    q_dst_row_indices[:, -max(1, capacities.max_num_seqs // 4) :] = -1
    if q_dst_row_indices.numel() > 2:
        q_dst_row_indices.reshape(-1)[1] = 17

    sources = MetadataSources(
        q_dst_row_indices=q_dst_row_indices,
        block_tables=random_tensor(
            shape.actual_attn_bs, shape.block_table_width
        ),
    )
    destinations = {
        "q_dst_row_indices": random_tensor(
            capacities.attention_sp, capacities.max_num_seqs
        ),
        "actual_attn_bs": random_tensor(),
        "context_lens": random_tensor(
            capacities.attention_sp, capacities.max_master_bs
        ),
        "global_context_lens": random_tensor(
            capacities.attention_sp, capacities.max_master_bs
        ),
        "context_lens_for_attn": random_tensor(capacities.max_attn_bs),
        "block_tables": random_tensor(
            capacities.max_attn_bs, capacities.max_num_blocks
        ),
        "res_slice_get_to_buffer_output": random_tensor(
            capacities.max_master_bs
        ),
        "res_slice_fill_to_buffer_output": random_tensor(
            capacities.max_master_bs
        ),
        "res_to_buffer_output_mask": random_tensor(
            capacities.max_master_bs
        ),
    }
    return destinations, sources


def _clone_destinations(
    destinations: MetadataDestinations,
) -> MetadataDestinations:
    return {name: tensor.clone() for name, tensor in destinations.items()}


def _reference_routing_metadata_once(
    destinations: MetadataDestinations,
    sources: MetadataSources,
    shape: MetadataCaseShape,
) -> None:
    copy_graph_q_dst_rows(
        destinations["q_dst_row_indices"], sources.q_dst_row_indices
    )
    copy_graph_actual_attn_bs(
        destinations["actual_attn_bs"], shape.actual_attn_bs
    )
    materialize_sp_graph_padding(
        destinations,
        actual_block_tables=sources.block_tables,
        actual_attn_bs=shape.actual_attn_bs,
        graph_attn_bs=shape.graph_attn_bs,
        actual_master_bs=shape.actual_master_bs,
        graph_master_bs=shape.graph_master_bs,
        local_result_rows=shape.actual_master_bs,
        sp_rank=shape.sp_rank,
        max_num_seqs=shape.max_num_seqs,
    )


def _validate_fused_inputs(
    destinations: MetadataDestinations,
    sources: MetadataSources,
    shape: MetadataCaseShape,
) -> torch.device:
    missing = [name for name in DESTINATION_NAMES if name not in destinations]
    if missing:
        raise ValueError(f"missing fused destinations: {missing}")
    tensors = [
        sources.q_dst_row_indices,
        sources.block_tables,
        *(destinations[name] for name in DESTINATION_NAMES),
    ]
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all fused metadata tensors must be CUDA tensors")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise ValueError("all fused metadata tensors must use torch.int32")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("all fused metadata tensors must be on one device")
    device = devices.pop()

    q_source = sources.q_dst_row_indices
    q_destination = destinations["q_dst_row_indices"]
    if q_source.ndim != 2 or q_destination.shape != q_source.shape:
        raise ValueError("q-destination source and destination shapes must match")
    if q_source.size(1) != shape.max_num_seqs:
        raise ValueError("q-destination source is not at full sequence capacity")
    if not q_source.is_contiguous() or not q_destination.is_contiguous():
        raise ValueError("q-destination tensors must be contiguous")
    if not bool(torch.any(q_source == -1).item()):
        raise ValueError("q-destination source must contain inactive -1 sentinels")

    actual_block_tables = sources.block_tables
    if (
        actual_block_tables.ndim != 2
        or actual_block_tables.size(0) < shape.actual_attn_bs
        or actual_block_tables.size(1) != shape.block_table_width
        or actual_block_tables.numel() == 0
    ):
        raise ValueError("actual block tables must cover all actual attention rows")
    if actual_block_tables.stride(1) <= 0:
        raise ValueError("actual block-table column stride must be positive")

    graph_actual_attn_bs = destinations["actual_attn_bs"]
    if graph_actual_attn_bs.numel() != 1 or not graph_actual_attn_bs.is_contiguous():
        raise ValueError("Graph actual attention batch must be one device scalar")

    context_lens = destinations["context_lens"]
    global_context_lens = destinations["global_context_lens"]
    for name, tensor in (
        ("context_lens", context_lens),
        ("global_context_lens", global_context_lens),
    ):
        if (
            tensor.ndim != 2
            or tensor.size(0) <= shape.sp_rank
            or tensor.size(1) < shape.graph_master_bs
            or any(stride <= 0 for stride in tensor.stride())
        ):
            raise ValueError(f"{name} does not cover the selected Graph rows")

    context_lens_for_attn = destinations["context_lens_for_attn"]
    if (
        context_lens_for_attn.ndim != 1
        or context_lens_for_attn.numel() < shape.graph_attn_bs
        or context_lens_for_attn.stride(0) <= 0
    ):
        raise ValueError("attention context lengths do not cover the Graph batch")

    graph_block_tables = destinations["block_tables"]
    if (
        graph_block_tables.ndim != 2
        or graph_block_tables.size(0) < shape.graph_attn_bs
        or graph_block_tables.size(1) < shape.block_table_width
        or any(stride <= 0 for stride in graph_block_tables.stride())
    ):
        raise ValueError("Graph block tables do not cover the selected Graph rows")

    for name in (
        "res_slice_get_to_buffer_output",
        "res_slice_fill_to_buffer_output",
        "res_to_buffer_output_mask",
    ):
        tensor = destinations[name]
        if (
            tensor.ndim != 1
            or tensor.numel() < shape.graph_master_bs
            or tensor.stride(0) <= 0
        ):
            raise ValueError(f"{name} does not cover the Graph master batch")

    return device


def _prepare_fused_operation(
    destinations: MetadataDestinations,
    sources: MetadataSources,
    shape: MetadataCaseShape,
) -> tuple[Callable[[], None], torch.device]:
    device = _validate_fused_inputs(destinations, sources, shape)
    q_work = sources.q_dst_row_indices.numel()
    attention_work = (
        shape.graph_attn_bs - shape.actual_attn_bs
    ) * shape.block_table_width
    master_work = shape.graph_master_bs - shape.actual_master_bs
    total_work = max(q_work, attention_work, master_work, 1)
    grid = (triton.cdiv(total_work, BLOCK_SIZE),)
    kernel_arguments = (
        sources.q_dst_row_indices,
        destinations["q_dst_row_indices"],
        destinations["actual_attn_bs"],
        sources.block_tables,
        destinations["context_lens"],
        destinations["global_context_lens"],
        destinations["context_lens_for_attn"],
        destinations["block_tables"],
        destinations["res_slice_get_to_buffer_output"],
        destinations["res_slice_fill_to_buffer_output"],
        destinations["res_to_buffer_output_mask"],
        q_work,
        shape.actual_attn_bs,
        shape.graph_attn_bs,
        shape.actual_master_bs,
        shape.graph_master_bs,
        shape.block_table_width,
        shape.sp_rank,
        shape.max_num_seqs,
        sources.block_tables.stride(0),
        sources.block_tables.stride(1),
        destinations["context_lens"].stride(0),
        destinations["context_lens"].stride(1),
        destinations["global_context_lens"].stride(0),
        destinations["global_context_lens"].stride(1),
        destinations["context_lens_for_attn"].stride(0),
        destinations["block_tables"].stride(0),
        destinations["block_tables"].stride(1),
        destinations["res_slice_get_to_buffer_output"].stride(0),
        destinations["res_slice_fill_to_buffer_output"].stride(0),
        destinations["res_to_buffer_output_mask"].stride(0),
    )
    compiled_kernel = routing_metadata_fusion_kernel.warmup(
        *kernel_arguments,
        grid=grid,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    if compiled_kernel is None:
        raise RuntimeError("Triton warmup did not return a compiled kernel")
    launch_grid = (grid[0], 1, 1)
    compiled_runner = compiled_kernel[launch_grid]
    launch_arguments = (*kernel_arguments, BLOCK_SIZE)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = triton_driver.active.get_current_stream(device_index)

    def operation() -> None:
        compiled_runner(*launch_arguments, stream=stream)

    return operation, device


def _compare_destinations(
    reference: MetadataDestinations,
    fused: MetadataDestinations,
) -> tuple[bool, dict[str, dict[str, int | bool]]]:
    tensor_results: dict[str, dict[str, int | bool]] = {}
    passed = True
    for name in DESTINATION_NAMES:
        equal = torch.equal(reference[name], fused[name])
        mismatch_count = 0
        if not equal:
            mismatch_count = int(
                torch.count_nonzero(reference[name] != fused[name]).item()
            )
            passed = False
        tensor_results[name] = {
            "passed": equal,
            "mismatch_count": mismatch_count,
        }
    return passed, tensor_results


def _correctness_shapes(
    observed_shape: dict[str, int], capacities: MetadataCapacities
) -> list[tuple[str, MetadataCaseShape]]:
    graph_master_bs = observed_shape["graph_master_bs"]
    graph_attn_bs = observed_shape["graph_attn_bs"]
    shape_variants = (
        ("no_padding", graph_master_bs, graph_attn_bs),
        (
            "issue1_padded",
            observed_shape["actual_master_bs"],
            observed_shape["actual_attn_bs"],
        ),
        ("attention_only_tail", graph_master_bs, graph_attn_bs - 4),
        ("master_only_tail", graph_master_bs - 10, graph_attn_bs),
        ("minimal_tails", graph_master_bs - 1, graph_attn_bs - 1),
        ("larger_valid_tails", graph_master_bs - 16, graph_attn_bs - 16),
    )
    sp_ranks = (0, 1, 7)
    widths = (1, 17, 1384)
    if capacities.attention_sp <= max(sp_ranks):
        raise ValueError("required correctness ranks need attention_sp >= 8")
    if capacities.max_num_blocks < max(widths):
        raise ValueError("required correctness widths exceed block-table capacity")

    cases = []
    for variant_name, actual_master_bs, actual_attn_bs in shape_variants:
        for sp_rank in sp_ranks:
            for width in widths:
                shape = MetadataCaseShape(
                    actual_master_bs=actual_master_bs,
                    graph_master_bs=graph_master_bs,
                    actual_attn_bs=actual_attn_bs,
                    graph_attn_bs=graph_attn_bs,
                    block_table_width=width,
                    sp_rank=sp_rank,
                    max_num_seqs=capacities.max_num_seqs,
                )
                cases.append(
                    (f"{variant_name}.rank{sp_rank}.width{width}", shape)
                )
    return cases


def _run_correctness_matrix(
    observed_shape: dict[str, int],
    capacities: MetadataCapacities,
    *,
    device: torch.device,
) -> dict[str, Any]:
    results = []
    all_passed = True
    for index, (case_name, shape) in enumerate(
        _correctness_shapes(observed_shape, capacities)
    ):
        base, sources = _make_case_data(
            capacities,
            shape,
            device=device,
            seed=12_340 + index,
        )
        reference = _clone_destinations(base)
        fused = _clone_destinations(base)
        _reference_routing_metadata_once(reference, sources, shape)
        fused_operation, _ = _prepare_fused_operation(fused, sources, shape)
        fused_operation()
        torch.cuda.synchronize(device)
        passed, tensor_results = _compare_destinations(reference, fused)
        all_passed = all_passed and passed
        results.append(
            {
                "name": case_name,
                "shape": asdict(shape),
                "passed": passed,
                "tensors": tensor_results,
            }
        )
        del base, sources, reference, fused, fused_operation

    correctness = {
        "passed": all_passed,
        "case_count": len(results),
        "cases": results,
    }
    if not all_passed:
        failed = [result["name"] for result in results if not result["passed"]]
        raise AssertionError(f"fused correctness failed for cases: {failed}")
    return correctness


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "min_us": float(samples.min()),
        "p50_us": float(np.percentile(samples, 50)),
        "p95_us": float(np.percentile(samples, 95)),
        "max_us": float(samples.max()),
        "mean_us": float(samples.mean()),
    }


def _measure_host_repeat(
    operation: Callable[[], None],
    *,
    iterations: int,
    device: torch.device,
) -> float:
    elapsed_ns = 0
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        begin_ns = time.perf_counter_ns()
        operation()
        elapsed_ns += time.perf_counter_ns() - begin_ns
    torch.cuda.synchronize(device)
    return elapsed_ns / iterations / 1_000


def _measure_device_repeat(
    operation: Callable[[], None],
    *,
    iterations: int,
) -> float:
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    begin_event.record()
    for _ in range(iterations):
        operation()
    end_event.record()
    end_event.synchronize()
    return begin_event.elapsed_time(end_event) * 1_000 / iterations


def _benchmark_operation_pair(
    reference: Callable[[], None],
    fused: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    operations = {"reference": reference, "fused": fused}
    for _ in range(warmup):
        reference()
        fused()
    torch.cuda.synchronize(device)

    host_samples = {"reference": [], "fused": []}
    for repeat in range(repeats):
        order = ("reference", "fused")
        if repeat % 2:
            order = tuple(reversed(order))
        for name in order:
            host_samples[name].append(
                _measure_host_repeat(
                    operations[name], iterations=iterations, device=device
                )
            )

    device_samples = {"reference": [], "fused": []}
    for repeat in range(repeats):
        order = ("reference", "fused")
        if repeat % 2:
            order = tuple(reversed(order))
        for name in order:
            device_samples[name].append(
                _measure_device_repeat(
                    operations[name], iterations=iterations
                )
            )

    implementations = {}
    for name in operations:
        implementations[name] = {
            "host_us_per_call": host_samples[name],
            "host_summary": _summarize(host_samples[name]),
            "device_span_us_per_call": device_samples[name],
            "device_span_summary": _summarize(device_samples[name]),
        }

    reference_host = implementations["reference"]["host_summary"]["p50_us"]
    fused_host = implementations["fused"]["host_summary"]["p50_us"]
    reference_device = implementations["reference"]["device_span_summary"][
        "p50_us"
    ]
    fused_device = implementations["fused"]["device_span_summary"]["p50_us"]
    comparisons = {
        "host_p50_speedup": reference_host / fused_host,
        "host_p50_delta_us": reference_host - fused_host,
        "device_span_p50_speedup": reference_device / fused_device,
        "device_span_p50_delta_us": reference_device - fused_device,
    }
    return {"implementations": implementations, "comparisons": comparisons}


def _trace_cuda_operations(
    trace_events: list[dict[str, Any]], range_name: str
) -> list[dict[str, Any]]:
    gpu_ranges = [
        event
        for event in trace_events
        if event.get("name") == range_name
        and event.get("cat") == "gpu_user_annotation"
    ]
    if len(gpu_ranges) != 1:
        raise RuntimeError(
            f"expected one GPU profiler range {range_name!r}, got "
            f"{len(gpu_ranges)}"
        )
    gpu_range = gpu_ranges[0]
    range_begin = float(gpu_range["ts"])
    range_end = range_begin + float(gpu_range["dur"])
    device_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    operations = []
    for event in trace_events:
        if event.get("cat") not in device_categories:
            continue
        begin = float(event.get("ts", float("-inf")))
        end = begin + float(event.get("dur", 0.0))
        if begin >= range_begin and end <= range_end + 0.01:
            operations.append(
                {
                    "category": event["cat"],
                    "name": event.get("name", "unknown"),
                    "duration_us": float(event.get("dur", 0.0)),
                }
            )
    return operations


def _profile_operations(
    reference: Callable[[], None],
    fused: Callable[[], None],
    *,
    trace_output: Path,
    device: torch.device,
) -> dict[str, Any]:
    reference()
    fused()
    torch.cuda.synchronize(device)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        acc_events=True,
    ) as profiler:
        with torch.profiler.record_function(REFERENCE_RANGE):
            reference()
        with torch.profiler.record_function(FUSED_RANGE):
            fused()
        torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(trace_output))

    with trace_output.open(encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    trace_events = trace.get("traceEvents")
    if not isinstance(trace_events, list):
        raise RuntimeError("Chrome trace has no traceEvents list")

    cuda_operations = {}
    for range_name in (REFERENCE_RANGE, FUSED_RANGE):
        operations = _trace_cuda_operations(trace_events, range_name)
        cuda_operations[range_name.rsplit(".", 1)[-1]] = {
            "count": len(operations),
            "operations": operations,
        }

    reference_count = cuda_operations["reference"]["count"]
    fused_count = cuda_operations["fused"]["count"]
    verification = {
        "reference_has_approximately_ten_operations": 8
        <= reference_count
        <= 12,
        "fused_has_one_kernel_launch": fused_count == 1,
    }
    if not all(verification.values()):
        raise RuntimeError(
            "unexpected profiler operation counts: "
            f"reference={reference_count}, fused={fused_count}"
        )
    return {
        "trace_path": str(trace_output.resolve()),
        "ranges": {"reference": REFERENCE_RANGE, "fused": FUSED_RANGE},
        "cuda_operations": cuda_operations,
        "verification": verification,
        "note": "Profiler attribution is not used as a performance result.",
    }


def _parse_case_names(value: str) -> list[str]:
    case_names = [name.strip() for name in value.split(",") if name.strip()]
    if not case_names:
        raise ValueError("--cases must not be empty")
    supported = {"no_padding", "issue1_padded"}
    unsupported = set(case_names).difference(supported)
    if unsupported:
        raise ValueError(f"unsupported timed cases: {sorted(unsupported)}")
    if len(set(case_names)) != len(case_names):
        raise ValueError("--cases must not contain duplicates")
    return case_names


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("routing metadata fusion prototype requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    source = _load_json(args.shape_json)
    observed_shape = _most_common_shape(source)
    capacities = _metadata_capacities(source)
    case_names = _parse_case_names(args.cases)
    source_rank = int(source.get("rank", 0)) % capacities.attention_sp

    correctness = _run_correctness_matrix(
        observed_shape, capacities, device=device
    )

    timed_results = {}
    prepared_operations: dict[
        str, tuple[Callable[[], None], Callable[[], None]]
    ] = {}
    retained_data = []
    for index, case_name in enumerate(case_names):
        shape = _timed_case_shape(
            case_name,
            observed_shape,
            sp_rank=source_rank,
            max_num_seqs=capacities.max_num_seqs,
        )
        _validate_case_shape(shape, capacities)
        base, sources = _make_case_data(
            capacities,
            shape,
            device=device,
            seed=91_000 + index,
        )
        reference_destinations = _clone_destinations(base)
        fused_destinations = _clone_destinations(base)

        def reference_operation(
            destinations: MetadataDestinations = reference_destinations,
            operation_sources: MetadataSources = sources,
            operation_shape: MetadataCaseShape = shape,
        ) -> None:
            _reference_routing_metadata_once(
                destinations, operation_sources, operation_shape
            )

        fused_operation, operation_device = _prepare_fused_operation(
            fused_destinations, sources, shape
        )
        if operation_device != device:
            raise RuntimeError("prepared fused operation changed CUDA device")
        prepared_operations[case_name] = (
            reference_operation,
            fused_operation,
        )
        retained_data.append(
            (
                base,
                sources,
                reference_destinations,
                fused_destinations,
            )
        )
        timed_results[case_name] = {
            "shape": asdict(shape),
            **_benchmark_operation_pair(
                reference_operation,
                fused_operation,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                device=device,
            ),
        }

    profiler_result = None
    if args.trace_output is not None:
        if "issue1_padded" not in prepared_operations:
            raise ValueError(
                "--trace-output requires issue1_padded in the timed cases"
            )
        reference_operation, fused_operation = prepared_operations[
            "issue1_padded"
        ]
        profiler_result = _profile_operations(
            reference_operation,
            fused_operation,
            trace_output=args.trace_output,
            device=device,
        )

    del retained_data
    return {
        "schema_version": 1,
        "manifest": _manifest(device),
        "source_shape_json": str(args.shape_json.resolve()),
        "observed_shape": observed_shape,
        "capacities": asdict(capacities),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "correctness": correctness,
        "timed_cases": timed_results,
        "profiler": profiler_result,
    }


def main() -> None:
    args = _parse_args()
    result = _run(args)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Benchmark-only fused prototype for no-padding SP Graph metadata injection."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from triton.runtime import driver as triton_driver


BLOCK_SIZE = 256


@triton.jit(
    do_not_specialize=[
        "actual_master_bs",
        "max_master_bs",
        "context_matrix_numel",
        "actual_attn_bs",
        "max_attn_bs",
        "actual_remote_rows",
        "max_remote_rows",
        "q_offsets_numel",
        "q_dst_numel",
    ]
)
def _copy_vector_metadata_kernel(
    input_ids_source_ptr,
    input_ids_destination_ptr,
    positions_source_ptr,
    positions_destination_ptr,
    slot_mapping_source_ptr,
    slot_mapping_destination_ptr,
    context_lens_source_ptr,
    context_lens_destination_ptr,
    global_context_lens_source_ptr,
    global_context_lens_destination_ptr,
    q_mask_source_ptr,
    q_mask_destination_ptr,
    res_lse_mask_source_ptr,
    res_lse_mask_destination_ptr,
    context_lens_for_attn_source_ptr,
    context_lens_for_attn_destination_ptr,
    q_slice_get_source_ptr,
    q_slice_get_destination_ptr,
    q_slice_fill_source_ptr,
    q_slice_fill_destination_ptr,
    q_copy_mask_source_ptr,
    q_copy_mask_destination_ptr,
    res_get_output_source_ptr,
    res_get_output_destination_ptr,
    res_fill_output_source_ptr,
    res_fill_output_destination_ptr,
    res_mask_output_source_ptr,
    res_mask_output_destination_ptr,
    res_get_input_source_ptr,
    res_get_input_destination_ptr,
    res_fill_input_source_ptr,
    res_fill_input_destination_ptr,
    res_mask_input_source_ptr,
    res_mask_input_destination_ptr,
    q_offsets_source_ptr,
    q_offsets_destination_ptr,
    q_dst_source_ptr,
    q_dst_destination_ptr,
    actual_attn_bs_destination_ptr,
    actual_master_bs,
    max_master_bs,
    context_matrix_numel,
    actual_attn_bs,
    max_attn_bs,
    actual_remote_rows,
    max_remote_rows,
    q_offsets_numel,
    q_dst_numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    master_destination_mask = offsets < max_master_bs
    active_master_mask = offsets < actual_master_bs
    tl.store(
        input_ids_destination_ptr + offsets,
        tl.load(input_ids_source_ptr + offsets, mask=active_master_mask, other=0),
        mask=master_destination_mask,
    )
    tl.store(
        positions_destination_ptr + offsets,
        tl.load(positions_source_ptr + offsets, mask=active_master_mask, other=0),
        mask=master_destination_mask,
    )
    tl.store(
        slot_mapping_destination_ptr + offsets,
        tl.load(slot_mapping_source_ptr + offsets, mask=active_master_mask, other=-1),
        mask=master_destination_mask,
    )

    context_matrix_mask = offsets < context_matrix_numel
    tl.store(
        context_lens_destination_ptr + offsets,
        tl.load(context_lens_source_ptr + offsets, mask=context_matrix_mask),
        mask=context_matrix_mask,
    )
    tl.store(
        global_context_lens_destination_ptr + offsets,
        tl.load(global_context_lens_source_ptr + offsets, mask=context_matrix_mask),
        mask=context_matrix_mask,
    )
    tl.store(
        q_mask_destination_ptr + offsets,
        tl.load(q_mask_source_ptr + offsets, mask=context_matrix_mask),
        mask=context_matrix_mask,
    )
    tl.store(
        res_lse_mask_destination_ptr + offsets,
        tl.load(res_lse_mask_source_ptr + offsets, mask=context_matrix_mask),
        mask=context_matrix_mask,
    )

    attention_destination_mask = offsets < max_attn_bs
    active_attention_mask = offsets < actual_attn_bs
    tl.store(
        context_lens_for_attn_destination_ptr + offsets,
        tl.load(
            context_lens_for_attn_source_ptr + offsets,
            mask=active_attention_mask,
            other=0,
        ),
        mask=attention_destination_mask,
    )

    tl.store(
        q_slice_get_destination_ptr + offsets,
        tl.load(q_slice_get_source_ptr + offsets, mask=active_master_mask, other=-1),
        mask=master_destination_mask,
    )
    tl.store(
        q_slice_fill_destination_ptr + offsets,
        tl.load(q_slice_fill_source_ptr + offsets, mask=active_master_mask, other=-1),
        mask=master_destination_mask,
    )
    tl.store(
        q_copy_mask_destination_ptr + offsets,
        tl.load(q_copy_mask_source_ptr + offsets, mask=active_master_mask, other=0),
        mask=master_destination_mask,
    )

    tl.store(
        res_get_output_destination_ptr + offsets,
        tl.load(res_get_output_source_ptr + offsets, mask=active_master_mask, other=-1),
        mask=master_destination_mask,
    )
    tl.store(
        res_fill_output_destination_ptr + offsets,
        tl.load(res_fill_output_source_ptr + offsets, mask=active_master_mask, other=-1),
        mask=master_destination_mask,
    )
    tl.store(
        res_mask_output_destination_ptr + offsets,
        tl.load(res_mask_output_source_ptr + offsets, mask=active_master_mask, other=0),
        mask=master_destination_mask,
    )

    remote_destination_mask = offsets < max_remote_rows
    active_remote_mask = offsets < actual_remote_rows
    tl.store(
        res_get_input_destination_ptr + offsets,
        tl.load(res_get_input_source_ptr + offsets, mask=active_remote_mask, other=-1),
        mask=remote_destination_mask,
    )
    tl.store(
        res_fill_input_destination_ptr + offsets,
        tl.load(res_fill_input_source_ptr + offsets, mask=active_remote_mask, other=-1),
        mask=remote_destination_mask,
    )
    tl.store(
        res_mask_input_destination_ptr + offsets,
        tl.load(res_mask_input_source_ptr + offsets, mask=active_remote_mask, other=0),
        mask=remote_destination_mask,
    )

    q_offsets_mask = offsets < q_offsets_numel
    tl.store(
        q_offsets_destination_ptr + offsets,
        tl.load(q_offsets_source_ptr + offsets, mask=q_offsets_mask),
        mask=q_offsets_mask,
    )

    q_dst_mask = offsets < q_dst_numel
    tl.store(
        q_dst_destination_ptr + offsets,
        tl.load(q_dst_source_ptr + offsets, mask=q_dst_mask),
        mask=q_dst_mask,
    )
    tl.store(
        actual_attn_bs_destination_ptr + offsets,
        actual_attn_bs,
        mask=offsets == 0,
    )


@triton.jit(
    do_not_specialize=[
        "source_rows",
        "source_columns",
        "destination_rows",
        "destination_columns",
        "source_stride_row",
        "source_stride_column",
        "destination_stride_row",
        "destination_stride_column",
    ]
)
def _copy_block_tables_kernel(
    source_ptr,
    destination_ptr,
    source_rows,
    source_columns,
    destination_rows,
    destination_columns,
    source_stride_row,
    source_stride_column,
    destination_stride_row,
    destination_stride_column,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    destination_work = destination_rows * destination_columns
    destination_mask = offsets < destination_work
    rows = offsets // destination_columns
    columns = offsets - rows * destination_columns
    source_mask = (
        destination_mask & (rows < source_rows) & (columns < source_columns)
    )
    values = tl.load(
        source_ptr
        + rows * source_stride_row
        + columns * source_stride_column,
        mask=source_mask,
        other=-1,
    )
    tl.store(
        destination_ptr
        + rows * destination_stride_row
        + columns * destination_stride_column,
        values,
        mask=destination_mask,
    )


_compiled_vector_kernel = None
_compiled_block_table_kernel = None
_vector_runners: dict[tuple[int, int, int], Any] = {}
_block_table_runners: dict[tuple[int, int, int], Any] = {}


def _launch_compiled(
    kernel: Any,
    arguments: tuple[Any, ...],
    grid_x: int,
    *,
    compiled_name: str,
) -> None:
    global _compiled_vector_kernel, _compiled_block_table_kernel

    if compiled_name == "vector":
        compiled = _compiled_vector_kernel
        runners = _vector_runners
    elif compiled_name == "block_table":
        compiled = _compiled_block_table_kernel
        runners = _block_table_runners
    else:
        raise ValueError(f"Unsupported compiled kernel name: {compiled_name}")

    if compiled is None:
        compiled = kernel.warmup(
            *arguments,
            grid=(grid_x,),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        if compiled_name == "vector":
            _compiled_vector_kernel = compiled
        else:
            _compiled_block_table_kernel = compiled

    device = torch.cuda.current_device()
    stream = triton_driver.active.get_current_stream(device)
    runner_key = (device, stream, grid_x)
    runner = runners.get(runner_key)
    if runner is None:
        runner = compiled[(grid_x, 1, 1)]
        runners[runner_key] = runner
    runner(*arguments, BLOCK_SIZE, stream=stream)


def copy_decode_context_to_graph_vars_fused_no_padding(
    graph_vars: dict[str, torch.Tensor | None],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    actual_master_bs: int,
    graph_master_bs: int,
    graph_attn_bs: int,
    context: Any,
) -> None:
    """Materialize the benchmark's no-padding Graph inputs in two launches."""

    actual_attn_bs = int(context.attention_compute_bs)
    if actual_master_bs != graph_master_bs or actual_attn_bs != graph_attn_bs:
        raise ValueError("fused prototype supports no-padding benchmark cases only")
    if not context.use_sp_a2a:
        raise ValueError("fused prototype requires SP all-to-all metadata")

    required_names = (
        "input_ids",
        "positions",
        "slot_mapping",
        "context_lens",
        "global_context_lens",
        "q_mask",
        "q_dst_row_indices",
        "actual_attn_bs",
        "res_lse_mask",
        "block_tables",
        "context_lens_for_attn",
        "q_slice_get",
        "q_slice_fill",
        "q_copy_mask",
        "res_slice_get_to_buffer_output",
        "res_slice_fill_to_buffer_output",
        "res_to_buffer_output_mask",
        "res_slice_get_to_buffer_input",
        "res_slice_fill_to_buffer_input",
        "res_to_buffer_input_mask",
        "q_offsets",
    )
    missing = [name for name in required_names if graph_vars.get(name) is None]
    if missing:
        raise ValueError(f"fused prototype is missing Graph tensors: {missing}")

    max_master_bs = graph_vars["slot_mapping"].numel()  # type: ignore[union-attr]
    max_attn_bs = graph_vars["context_lens_for_attn"].numel()  # type: ignore[union-attr]
    max_remote_rows = graph_vars["res_to_buffer_input_mask"].numel()  # type: ignore[union-attr]
    actual_remote_rows = context.res_to_buffer_input_mask.numel()
    context_matrix_numel = graph_vars["context_lens"].numel()  # type: ignore[union-attr]
    q_offsets_numel = graph_vars["q_offsets"].numel()  # type: ignore[union-attr]
    q_dst_numel = graph_vars["q_dst_row_indices"].numel()  # type: ignore[union-attr]

    if input_ids.numel() != actual_master_bs or positions.numel() != actual_master_bs:
        raise ValueError("input_ids and positions must match actual_master_bs")
    if context.slot_mapping.numel() != actual_master_bs:
        raise ValueError("slot_mapping must match actual_master_bs")
    for name in ("context_lens", "global_context_lens", "q_mask", "res_lse_mask"):
        if getattr(context, name).numel() != context_matrix_numel:
            raise ValueError(f"{name} source and destination sizes differ")
    if context.context_lens_for_attn.numel() != actual_attn_bs:
        raise ValueError("context_lens_for_attn must match actual_attn_bs")
    if context.q_offsets.numel() != q_offsets_numel:
        raise ValueError("q_offsets source and destination sizes differ")
    if context.q_dst_row_indices.numel() != q_dst_numel:
        raise ValueError("q_dst_row_indices source and destination sizes differ")

    vector_arguments = (
        input_ids,
        graph_vars["input_ids"],
        positions,
        graph_vars["positions"],
        context.slot_mapping,
        graph_vars["slot_mapping"],
        context.context_lens,
        graph_vars["context_lens"],
        context.global_context_lens,
        graph_vars["global_context_lens"],
        context.q_mask,
        graph_vars["q_mask"],
        context.res_lse_mask,
        graph_vars["res_lse_mask"],
        context.context_lens_for_attn,
        graph_vars["context_lens_for_attn"],
        context.q_slice_get,
        graph_vars["q_slice_get"],
        context.q_slice_fill,
        graph_vars["q_slice_fill"],
        context.q_copy_mask,
        graph_vars["q_copy_mask"],
        context.res_slice_get_to_buffer_output,
        graph_vars["res_slice_get_to_buffer_output"],
        context.res_slice_fill_to_buffer_output,
        graph_vars["res_slice_fill_to_buffer_output"],
        context.res_to_buffer_output_mask,
        graph_vars["res_to_buffer_output_mask"],
        context.res_slice_get_to_buffer_input,
        graph_vars["res_slice_get_to_buffer_input"],
        context.res_slice_fill_to_buffer_input,
        graph_vars["res_slice_fill_to_buffer_input"],
        context.res_to_buffer_input_mask,
        graph_vars["res_to_buffer_input_mask"],
        context.q_offsets,
        graph_vars["q_offsets"],
        context.q_dst_row_indices,
        graph_vars["q_dst_row_indices"],
        graph_vars["actual_attn_bs"],
        actual_master_bs,
        max_master_bs,
        context_matrix_numel,
        actual_attn_bs,
        max_attn_bs,
        actual_remote_rows,
        max_remote_rows,
        q_offsets_numel,
        q_dst_numel,
    )
    vector_work = max(
        max_master_bs,
        context_matrix_numel,
        max_attn_bs,
        max_remote_rows,
        q_offsets_numel,
        q_dst_numel,
        1,
    )
    _launch_compiled(
        _copy_vector_metadata_kernel,
        vector_arguments,
        triton.cdiv(vector_work, BLOCK_SIZE),
        compiled_name="vector",
    )

    source_block_tables = context.block_tables
    destination_block_tables = graph_vars["block_tables"]
    block_arguments = (
        source_block_tables,
        destination_block_tables,
        source_block_tables.size(0),
        source_block_tables.size(1),
        destination_block_tables.size(0),  # type: ignore[union-attr]
        destination_block_tables.size(1),  # type: ignore[union-attr]
        source_block_tables.stride(0),
        source_block_tables.stride(1),
        destination_block_tables.stride(0),  # type: ignore[union-attr]
        destination_block_tables.stride(1),  # type: ignore[union-attr]
    )
    destination_work = destination_block_tables.numel()  # type: ignore[union-attr]
    _launch_compiled(
        _copy_block_tables_kernel,
        block_arguments,
        triton.cdiv(destination_work, BLOCK_SIZE),
        compiled_name="block_table",
    )

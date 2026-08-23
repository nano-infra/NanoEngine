import torch
import triton
import triton.language as tl
from triton.runtime import driver as triton_driver


BLOCK_SIZE = 256


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
        "actual_block_stride_col",
        "context_lens_stride_sp",
        "context_lens_stride_row",
        "global_context_lens_stride_sp",
        "global_context_lens_stride_row",
        "context_lens_for_attn_stride",
        "graph_block_stride_row",
        "graph_block_stride_col",
        "res_get_stride",
        "res_fill_stride",
        "res_mask_stride",
    ]
)
def update_sp_graph_metadata_kernel(
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
        actual_block_tables_ptr + columns * actual_block_stride_col,
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


_compiled_kernel = None
_compiled_runners = {}


def _kernel_arguments(
    q_dst_source: torch.Tensor,
    graph_q_dst: torch.Tensor,
    graph_actual_attn_bs: torch.Tensor,
    actual_block_tables: torch.Tensor,
    graph_context_lens: torch.Tensor,
    graph_global_context_lens: torch.Tensor,
    graph_context_lens_for_attn: torch.Tensor,
    graph_block_tables: torch.Tensor,
    graph_res_get: torch.Tensor,
    graph_res_fill: torch.Tensor,
    graph_res_mask: torch.Tensor,
    actual_attn_bs: int,
    graph_attn_bs: int,
    actual_master_bs: int,
    graph_master_bs: int,
    sp_rank: int,
    max_num_seqs: int,
):
    return (
        q_dst_source,
        graph_q_dst,
        graph_actual_attn_bs,
        actual_block_tables,
        graph_context_lens,
        graph_global_context_lens,
        graph_context_lens_for_attn,
        graph_block_tables,
        graph_res_get,
        graph_res_fill,
        graph_res_mask,
        q_dst_source.numel(),
        actual_attn_bs,
        graph_attn_bs,
        actual_master_bs,
        graph_master_bs,
        actual_block_tables.size(1),
        sp_rank,
        max_num_seqs,
        actual_block_tables.stride(1),
        graph_context_lens.stride(0),
        graph_context_lens.stride(1),
        graph_global_context_lens.stride(0),
        graph_global_context_lens.stride(1),
        graph_context_lens_for_attn.stride(0),
        graph_block_tables.stride(0),
        graph_block_tables.stride(1),
        graph_res_get.stride(0),
        graph_res_fill.stride(0),
        graph_res_mask.stride(0),
    )


def update_sp_graph_metadata(
    q_dst_source: torch.Tensor,
    graph_q_dst: torch.Tensor,
    graph_actual_attn_bs: torch.Tensor,
    actual_block_tables: torch.Tensor,
    graph_context_lens: torch.Tensor,
    graph_global_context_lens: torch.Tensor,
    graph_context_lens_for_attn: torch.Tensor,
    graph_block_tables: torch.Tensor,
    graph_res_get: torch.Tensor,
    graph_res_fill: torch.Tensor,
    graph_res_mask: torch.Tensor,
    actual_attn_bs: int,
    graph_attn_bs: int,
    actual_master_bs: int,
    graph_master_bs: int,
    sp_rank: int,
    max_num_seqs: int,
) -> None:
    global _compiled_kernel

    arguments = _kernel_arguments(
        q_dst_source,
        graph_q_dst,
        graph_actual_attn_bs,
        actual_block_tables,
        graph_context_lens,
        graph_global_context_lens,
        graph_context_lens_for_attn,
        graph_block_tables,
        graph_res_get,
        graph_res_fill,
        graph_res_mask,
        actual_attn_bs,
        graph_attn_bs,
        actual_master_bs,
        graph_master_bs,
        sp_rank,
        max_num_seqs,
    )
    attention_work = (graph_attn_bs - actual_attn_bs) * actual_block_tables.size(1)
    master_work = graph_master_bs - actual_master_bs
    total_work = max(q_dst_source.numel(), attention_work, master_work, 1)
    grid_x = triton.cdiv(total_work, BLOCK_SIZE)

    if _compiled_kernel is None:
        _compiled_kernel = update_sp_graph_metadata_kernel.warmup(
            *arguments,
            grid=(grid_x,),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    device = torch.cuda.current_device()
    stream = triton_driver.active.get_current_stream(device)
    runner_key = (device, stream, grid_x)
    runner = _compiled_runners.get(runner_key)
    if runner is None:
        runner = _compiled_kernel[(grid_x, 1, 1)]
        _compiled_runners[runner_key] = runner
    runner(*arguments, BLOCK_SIZE, stream=stream)


def warmup_sp_graph_metadata(
    graph_q_dst: torch.Tensor,
    graph_actual_attn_bs: torch.Tensor,
    graph_context_lens: torch.Tensor,
    graph_global_context_lens: torch.Tensor,
    graph_context_lens_for_attn: torch.Tensor,
    graph_block_tables: torch.Tensor,
    graph_res_get: torch.Tensor,
    graph_res_fill: torch.Tensor,
    graph_res_mask: torch.Tensor,
    max_num_seqs: int,
) -> None:
    update_sp_graph_metadata(
        graph_q_dst,
        graph_q_dst,
        graph_actual_attn_bs,
        graph_block_tables[:1, :1],
        graph_context_lens,
        graph_global_context_lens,
        graph_context_lens_for_attn,
        graph_block_tables,
        graph_res_get,
        graph_res_fill,
        graph_res_mask,
        1,
        1,
        1,
        1,
        0,
        max_num_seqs,
    )

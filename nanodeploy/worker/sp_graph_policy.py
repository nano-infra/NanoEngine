from __future__ import annotations

import torch


def materialize_sp_graph_padding(
    graph_vars: dict[str, torch.Tensor | None],
    *,
    actual_block_tables: torch.Tensor,
    actual_attn_bs: int,
    graph_attn_bs: int,
    actual_master_bs: int,
    graph_master_bs: int,
    local_result_rows: int,
    sp_rank: int,
    max_num_seqs: int,
) -> None:
    """Make packed SP metadata safe for a padded CUDA Graph replay.

    Real attention rows remain packed in the leading prefix. Graph-only
    attention rows borrow one valid read-only KV page, while Graph-only local
    master slots reuse one finite attention partial and stay outside sampling.
    """

    if not 0 < actual_attn_bs <= graph_attn_bs:
        raise ValueError(
            "actual_attn_bs must be in [1, graph_attn_bs], got "
            f"actual={actual_attn_bs} graph={graph_attn_bs}"
        )
    if not 0 <= actual_master_bs <= graph_master_bs:
        raise ValueError(
            "actual_master_bs must be in [0, graph_master_bs], got "
            f"actual={actual_master_bs} graph={graph_master_bs}"
        )
    if local_result_rows != actual_master_bs:
        raise ValueError(
            "local result rows must match the actual master batch: "
            f"rows={local_result_rows} actual_master_bs={actual_master_bs}"
        )

    graph_context_lens = graph_vars["context_lens"]
    graph_global_context_lens = graph_vars["global_context_lens"]
    graph_context_lens_for_attn = graph_vars["context_lens_for_attn"]
    graph_block_tables = graph_vars["block_tables"]
    graph_res_get = graph_vars["res_slice_get_to_buffer_output"]
    graph_res_fill = graph_vars["res_slice_fill_to_buffer_output"]
    graph_res_mask = graph_vars["res_to_buffer_output_mask"]
    required = (
        graph_context_lens,
        graph_global_context_lens,
        graph_context_lens_for_attn,
        graph_block_tables,
        graph_res_get,
        graph_res_fill,
        graph_res_mask,
    )
    if any(tensor is None for tensor in required):
        raise ValueError("SP Graph padding requires complete Graph metadata")

    if graph_attn_bs > actual_attn_bs:
        if actual_block_tables.ndim != 2 or actual_block_tables.size(1) == 0:
            raise ValueError("SP Graph padding requires one valid KV block table")
        graph_context_lens_for_attn[
            actual_attn_bs:graph_attn_bs
        ].fill_(1)
        block_width = actual_block_tables.size(1)
        graph_block_tables[
            actual_attn_bs:graph_attn_bs, :block_width
        ].copy_(
            actual_block_tables[0:1].expand(
                graph_attn_bs - actual_attn_bs, -1
            )
        )

    graph_only_master_rows = graph_master_bs - actual_master_bs
    if graph_only_master_rows == 0:
        return

    graph_context_lens[
        sp_rank, actual_master_bs:graph_master_bs
    ].fill_(1)
    graph_global_context_lens[
        sp_rank, actual_master_bs:graph_master_bs
    ].fill_(1)

    mapping_begin = actual_master_bs
    mapping_end = mapping_begin + graph_only_master_rows
    if mapping_end > graph_res_get.numel():
        raise ValueError(
            "Graph-only local result mappings exceed the captured buffer: "
            f"required={mapping_end} capacity={graph_res_get.numel()}"
        )

    dummy_attention_row = (
        actual_attn_bs if actual_attn_bs < graph_attn_bs else 0
    )
    graph_res_get[mapping_begin:mapping_end].fill_(dummy_attention_row)
    torch.arange(
        sp_rank * max_num_seqs + actual_master_bs,
        sp_rank * max_num_seqs + graph_master_bs,
        dtype=graph_res_fill.dtype,
        device=graph_res_fill.device,
        out=graph_res_fill[mapping_begin:mapping_end],
    )
    graph_res_mask[mapping_begin:mapping_end].fill_(1)

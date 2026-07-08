from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def copy_tensor_to_graph_buffer(
    name: str,
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if source.ndim != target.ndim:
        raise RuntimeError(
            "Decode CUDA graph buffer rank mismatch for dynamic metadata: "
            f"{name} source ndim={source.ndim}, target ndim={target.ndim}."
        )
    if any(
        source_dim > target_dim
        for source_dim, target_dim in zip(source.shape, target.shape)
    ):
        raise RuntimeError(
            "Decode CUDA graph buffer is too small for dynamic metadata: "
            f"{name} requires shape={tuple(source.shape)}, "
            f"captured shape={tuple(target.shape)}."
        )

    target.zero_()
    if source.shape == target.shape:
        target.copy_(source)
        return

    target_slices = tuple(slice(0, dim) for dim in source.shape)
    target[target_slices].copy_(source)


def copy_mla_metadata_to_graph_vars(
    graph_vars: dict[str, torch.Tensor | None],
    graph_attention_bs: int,
    compute_metadata: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    num_key_value_heads: int,
) -> None:
    tile_scheduler_metadata = graph_vars.get("tile_scheduler_metadata")
    num_splits = graph_vars.get("num_splits")
    if tile_scheduler_metadata is None or num_splits is None:
        return
    if num_key_value_heads != 1:
        return

    context_lens = graph_vars["context_lens_for_attn"][:graph_attention_bs]
    current_tile_scheduler_metadata, current_num_splits = compute_metadata(context_lens)
    copy_tensor_to_graph_buffer(
        "tile_scheduler_metadata",
        current_tile_scheduler_metadata,
        tile_scheduler_metadata,
    )
    copy_tensor_to_graph_buffer("num_splits", current_num_splits, num_splits)


def select_decode_graph_master_bs(
    graph_master_rank_bs: list[int],
    sp_graph_map: dict[int, list[int]] | None,
    *,
    bs: int,
    use_sp_a2a: bool,
    sp_comm_bs: int | None,
    attention_compute_bs: int | None,
) -> int:
    required_master_bs = bs
    if use_sp_a2a and sp_comm_bs is not None:
        required_master_bs = max(required_master_bs, int(sp_comm_bs))

    if use_sp_a2a and sp_graph_map is not None:
        required_attn_bs = int(attention_compute_bs or required_master_bs)
        for candidate_bs in graph_master_rank_bs:
            if candidate_bs < required_master_bs:
                continue
            valid_attn_bs_list = sp_graph_map.get(candidate_bs)
            if valid_attn_bs_list and valid_attn_bs_list[-1] >= required_attn_bs:
                return candidate_bs
        raise RuntimeError(
            "SP decode CUDA graph capacity is insufficient: "
            f"required_master_bs={required_master_bs}, "
            f"required_attention_compute_bs={required_attn_bs}, "
            f"max_captured_master_bs={graph_master_rank_bs[-1]}"
        )

    try:
        return next(x for x in graph_master_rank_bs if x >= required_master_bs)
    except StopIteration:
        raise RuntimeError(
            f"Decode CUDA graph master batch {required_master_bs} exceeds "
            f"max captured master_bs ({graph_master_rank_bs[-1]})"
        )


def validate_decode_graph_copy_capacity(
    graph_vars: dict[str, torch.Tensor | None],
    context: Any,
) -> None:
    checks = [
        ("slot_mapping", context.slot_mapping, graph_vars["slot_mapping"], 0),
        ("block_tables", context.block_tables, graph_vars["block_tables"], 0),
        (
            "context_lens_for_attn",
            context.context_lens_for_attn,
            graph_vars["context_lens_for_attn"],
            0,
        ),
        ("q_slice_get", context.q_slice_get, graph_vars["q_slice_get"], 0),
        ("q_slice_fill", context.q_slice_fill, graph_vars["q_slice_fill"], 0),
        ("q_copy_mask", context.q_copy_mask, graph_vars["q_copy_mask"], 0),
        (
            "res_slice_get_to_buffer_output",
            context.res_slice_get_to_buffer_output,
            graph_vars["res_slice_get_to_buffer_output"],
            0,
        ),
        (
            "res_slice_fill_to_buffer_output",
            context.res_slice_fill_to_buffer_output,
            graph_vars["res_slice_fill_to_buffer_output"],
            0,
        ),
        (
            "res_to_buffer_output_mask",
            context.res_to_buffer_output_mask,
            graph_vars["res_to_buffer_output_mask"],
            0,
        ),
        (
            "res_slice_get_to_buffer_input",
            context.res_slice_get_to_buffer_input,
            graph_vars["res_slice_get_to_buffer_input"],
            0,
        ),
        (
            "res_slice_fill_to_buffer_input",
            context.res_slice_fill_to_buffer_input,
            graph_vars["res_slice_fill_to_buffer_input"],
            0,
        ),
        (
            "res_to_buffer_input_mask",
            context.res_to_buffer_input_mask,
            graph_vars["res_to_buffer_input_mask"],
            0,
        ),
    ]
    for name, source, target, dim in checks:
        if source is None or target is None:
            continue
        if source.shape[dim] > target.shape[dim]:
            raise RuntimeError(
                "Decode CUDA graph buffer is too small for dynamic SP metadata: "
                f"{name} requires dim{dim}={source.shape[dim]}, "
                f"captured dim{dim}={target.shape[dim]}. Increase "
                "max_num_recv_seqs/max_num_seqs or run this path eagerly."
            )

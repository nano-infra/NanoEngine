from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch

from nanodeploy.kernels.sp_graph_metadata import update_sp_graph_metadata


PhaseFactory = Callable[[str], AbstractContextManager[None]]


def select_decode_graph_master_bs(
    actual_master_bs: int,
    *,
    graph_master_rank_bs: Sequence[int],
    use_sp_a2a: bool,
    sp_backend: str,
    fixed_sp_size: int,
    sp_comm_bs: int | None,
) -> int:
    """Select the smallest valid master bucket for one decode replay."""

    required_master_bs = actual_master_bs
    if (
        use_sp_a2a
        and (sp_backend == "nccl" or fixed_sp_size > 0)
        and sp_comm_bs is not None
    ):
        required_master_bs = max(required_master_bs, sp_comm_bs)

    try:
        return next(
            bucket
            for bucket in graph_master_rank_bs
            if bucket >= required_master_bs
        )
    except StopIteration as exc:
        max_bucket = graph_master_rank_bs[-1] if graph_master_rank_bs else None
        raise RuntimeError(
            f"Required master batch {required_master_bs} exceeds "
            f"max captured master_bs ({max_bucket})"
        ) from exc


def select_decode_graph_bucket(
    actual_master_bs: int,
    actual_attn_bs: int | None,
    *,
    graph_master_rank_bs: Sequence[int],
    sp_graph_map: Mapping[int, Sequence[int]],
    use_sp_a2a: bool,
    sp_backend: str,
    fixed_sp_size: int,
    sp_comm_bs: int | None,
) -> tuple[int, int]:
    """Select the master and attention buckets used for Graph lookup."""

    master_bs = select_decode_graph_master_bs(
        actual_master_bs,
        graph_master_rank_bs=graph_master_rank_bs,
        use_sp_a2a=use_sp_a2a,
        sp_backend=sp_backend,
        fixed_sp_size=fixed_sp_size,
        sp_comm_bs=sp_comm_bs,
    )
    if not use_sp_a2a:
        return master_bs, master_bs

    required_attn_bs = (
        actual_master_bs if actual_attn_bs is None else actual_attn_bs
    )
    valid_attn_bs = sp_graph_map.get(master_bs)
    if not valid_attn_bs:
        raise RuntimeError(f"No SP graph map found for master_bs={master_bs}")
    try:
        graph_attn_bs = next(
            bucket for bucket in valid_attn_bs if bucket >= required_attn_bs
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"Input attention_compute_bs {required_attn_bs} exceeds "
            f"max captured attn_bs ({valid_attn_bs[-1]}) for "
            f"master_bs {master_bs}"
        ) from exc
    return master_bs, graph_attn_bs


def copy_decode_context_to_graph_vars(
    graph_vars: dict[str, torch.Tensor | None],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    actual_master_bs: int,
    graph_master_bs: int,
    graph_attn_bs: int,
    context: Any,
    *,
    sp_rank: int,
    max_num_seqs: int,
    phase_factory: PhaseFactory | None = None,
) -> None:
    """Copy decode state into persistent Graph tensors.

    ``phase_factory`` is disabled on the normal serving path. Runtime-overhead
    experiments use it to delimit the fused routing update.
    """

    if graph_vars.get("input_ids") is not None:
        graph_vars["input_ids"].zero_()  # type: ignore[union-attr]
        graph_vars["input_ids"][:actual_master_bs] = input_ids  # type: ignore[index]
    if graph_vars.get("positions") is not None:
        graph_vars["positions"].zero_()  # type: ignore[union-attr]
        graph_vars["positions"][:actual_master_bs] = positions  # type: ignore[index]

    graph_vars["slot_mapping"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["slot_mapping"][: context.slot_mapping.shape[0]] = (  # type: ignore[index]
        context.slot_mapping
    )

    graph_vars["context_lens"].zero_()  # type: ignore[union-attr]
    graph_vars["context_lens"].copy_(context.context_lens)  # type: ignore[union-attr]
    graph_vars["global_context_lens"].zero_()  # type: ignore[union-attr]
    graph_vars["global_context_lens"].copy_(  # type: ignore[union-attr]
        context.global_context_lens
    )
    graph_vars["q_mask"].zero_()  # type: ignore[union-attr]
    graph_vars["q_mask"].copy_(context.q_mask)  # type: ignore[union-attr]
    graph_vars["res_lse_mask"].zero_()  # type: ignore[union-attr]
    graph_vars["res_lse_mask"].copy_(context.res_lse_mask)  # type: ignore[union-attr]
    graph_vars["block_tables"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["block_tables"][  # type: ignore[index]
        : context.block_tables.size(0), : context.block_tables.size(1)
    ] = context.block_tables

    graph_vars["context_lens_for_attn"].zero_()  # type: ignore[union-attr]
    graph_vars["context_lens_for_attn"][  # type: ignore[index]
        : context.context_lens_for_attn.shape[0]
    ].copy_(context.context_lens_for_attn)

    graph_vars["q_slice_get"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["q_slice_fill"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["q_copy_mask"].zero_()  # type: ignore[union-attr]
    graph_vars["q_slice_get"][: context.q_slice_get.shape[0]] = (  # type: ignore[index]
        context.q_slice_get
    )
    graph_vars["q_slice_fill"][: context.q_slice_fill.shape[0]] = (  # type: ignore[index]
        context.q_slice_fill
    )
    graph_vars["q_copy_mask"][: context.q_copy_mask.shape[0]] = (  # type: ignore[index]
        context.q_copy_mask
    )

    graph_vars["res_slice_get_to_buffer_output"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["res_slice_fill_to_buffer_output"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["res_to_buffer_output_mask"].zero_()  # type: ignore[union-attr]
    graph_vars["res_slice_get_to_buffer_output"][  # type: ignore[index]
        : context.res_slice_get_to_buffer_output.shape[0]
    ] = context.res_slice_get_to_buffer_output
    graph_vars["res_slice_fill_to_buffer_output"][  # type: ignore[index]
        : context.res_slice_fill_to_buffer_output.shape[0]
    ] = context.res_slice_fill_to_buffer_output
    graph_vars["res_to_buffer_output_mask"][  # type: ignore[index]
        : context.res_to_buffer_output_mask.shape[0]
    ] = context.res_to_buffer_output_mask

    graph_vars["res_slice_get_to_buffer_input"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["res_slice_fill_to_buffer_input"].fill_(-1)  # type: ignore[union-attr]
    graph_vars["res_to_buffer_input_mask"].zero_()  # type: ignore[union-attr]
    graph_vars["res_slice_get_to_buffer_input"][  # type: ignore[index]
        : context.res_slice_get_to_buffer_input.shape[0]
    ].copy_(context.res_slice_get_to_buffer_input)
    graph_vars["res_slice_fill_to_buffer_input"][  # type: ignore[index]
        : context.res_slice_fill_to_buffer_input.shape[0]
    ].copy_(context.res_slice_fill_to_buffer_input)
    graph_vars["res_to_buffer_input_mask"][  # type: ignore[index]
        : context.res_to_buffer_input_mask.shape[0]
    ].copy_(context.res_to_buffer_input_mask)

    graph_vars["q_offsets"].zero_()  # type: ignore[union-attr]
    graph_vars["q_offsets"].copy_(context.q_offsets)  # type: ignore[union-attr]

    if not context.use_sp_a2a:
        return

    actual_attn_bs = int(context.attention_compute_bs)
    routing_args = (
        context.q_dst_row_indices,
        graph_vars["q_dst_row_indices"],
        graph_vars["actual_attn_bs"],
        context.block_tables,
        graph_vars["context_lens"],
        graph_vars["global_context_lens"],
        graph_vars["context_lens_for_attn"],
        graph_vars["block_tables"],
        graph_vars["res_slice_get_to_buffer_output"],
        graph_vars["res_slice_fill_to_buffer_output"],
        graph_vars["res_to_buffer_output_mask"],
        actual_attn_bs,
        graph_attn_bs,
        actual_master_bs,
        graph_master_bs,
        sp_rank,
        max_num_seqs,
    )
    if phase_factory is None:
        update_sp_graph_metadata(*routing_args)  # type: ignore[arg-type]
    else:
        with phase_factory("nanodeploy.graph.metadata.fused"):
            update_sp_graph_metadata(*routing_args)  # type: ignore[arg-type]

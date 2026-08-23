from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch


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


def copy_graph_q_dst_rows(
    graph_q_dst_row_indices: torch.Tensor | None,
    q_dst_row_indices: torch.Tensor | None,
) -> None:
    if graph_q_dst_row_indices is None:
        return
    graph_q_dst_row_indices.fill_(-1)
    if q_dst_row_indices is not None:
        graph_q_dst_row_indices.copy_(q_dst_row_indices)


def copy_graph_actual_attn_bs(
    graph_actual_attn_bs: torch.Tensor | None,
    actual_attn_bs: int,
) -> None:
    if graph_actual_attn_bs is not None:
        graph_actual_attn_bs.fill_(actual_attn_bs)


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
    experiments use it to delimit the three routing-specific updates without
    duplicating the production implementation in the benchmark.
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

    graph_q_dst_row_indices = graph_vars.get("q_dst_row_indices")
    if phase_factory is None:
        copy_graph_q_dst_rows(
            graph_q_dst_row_indices, context.q_dst_row_indices
        )
    else:
        with phase_factory("nanodeploy.graph.metadata.q_dst_rows"):
            copy_graph_q_dst_rows(
                graph_q_dst_row_indices, context.q_dst_row_indices
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
    graph_actual_attn_bs = graph_vars.get("actual_attn_bs")
    if phase_factory is None:
        copy_graph_actual_attn_bs(graph_actual_attn_bs, actual_attn_bs)
    else:
        with phase_factory("nanodeploy.graph.metadata.actual_attn_bs"):
            copy_graph_actual_attn_bs(graph_actual_attn_bs, actual_attn_bs)

    padding_kwargs = {
        "actual_block_tables": context.block_tables,
        "actual_attn_bs": actual_attn_bs,
        "graph_attn_bs": graph_attn_bs,
        "actual_master_bs": actual_master_bs,
        "graph_master_bs": graph_master_bs,
        "local_result_rows": context.res_slice_get_to_buffer_output.numel(),
        "sp_rank": sp_rank,
        "max_num_seqs": max_num_seqs,
    }
    if phase_factory is None:
        materialize_sp_graph_padding(graph_vars, **padding_kwargs)
    else:
        with phase_factory("nanodeploy.graph.metadata.padding"):
            materialize_sp_graph_padding(graph_vars, **padding_kwargs)


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

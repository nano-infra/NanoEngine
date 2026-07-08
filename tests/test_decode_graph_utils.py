from types import SimpleNamespace

import pytest
import torch

from nanodeploy.worker.decode_graph_utils import (
    copy_mla_metadata_to_graph_vars,
    copy_tensor_to_graph_buffer,
    select_decode_graph_master_bs,
    validate_decode_graph_copy_capacity,
)


def test_copy_tensor_to_graph_buffer_pads_dynamic_metadata():
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    target = torch.full((3, 2), -1, dtype=torch.int32)

    copy_tensor_to_graph_buffer("tile_scheduler_metadata", source, target)

    assert target.tolist() == [[1, 2], [3, 4], [0, 0]]


def test_copy_tensor_to_graph_buffer_rejects_oversized_metadata():
    source = torch.empty((3, 2), dtype=torch.int32)
    target = torch.empty((2, 2), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="tile_scheduler_metadata"):
        copy_tensor_to_graph_buffer("tile_scheduler_metadata", source, target)


def test_copy_mla_metadata_to_graph_vars_uses_graph_attention_bucket():
    graph_vars = {
        "context_lens_for_attn": torch.tensor([5, 4, 0, 0], dtype=torch.int32),
        "tile_scheduler_metadata": torch.full((3, 2), -1, dtype=torch.int32),
        "num_splits": torch.full((6,), -1, dtype=torch.int32),
    }
    seen_context_lens = []

    def compute_metadata(context_lens):
        seen_context_lens.append(context_lens.clone())
        return (
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        )

    copy_mla_metadata_to_graph_vars(
        graph_vars,
        graph_attention_bs=4,
        compute_metadata=compute_metadata,
        num_key_value_heads=1,
    )

    assert [x.tolist() for x in seen_context_lens] == [[5, 4, 0, 0]]
    assert graph_vars["tile_scheduler_metadata"].tolist() == [
        [10, 11],
        [12, 13],
        [0, 0],
    ]
    assert graph_vars["num_splits"].tolist() == [0, 1, 2, 3, 4, 0]


def test_copy_mla_metadata_to_graph_vars_skips_non_mla_metadata():
    graph_vars = {
        "context_lens_for_attn": torch.tensor([5, 4], dtype=torch.int32),
        "tile_scheduler_metadata": torch.full((2, 2), -1, dtype=torch.int32),
        "num_splits": torch.full((3,), -1, dtype=torch.int32),
    }

    def compute_metadata(_context_lens):
        raise AssertionError("metadata should not be computed for non-MLA attention")

    copy_mla_metadata_to_graph_vars(
        graph_vars,
        graph_attention_bs=2,
        compute_metadata=compute_metadata,
        num_key_value_heads=8,
    )

    assert graph_vars["tile_scheduler_metadata"].tolist() == [[-1, -1], [-1, -1]]
    assert graph_vars["num_splits"].tolist() == [-1, -1, -1]


def test_select_decode_graph_master_bs_accounts_for_dynamic_sp_attention_rows():
    selected = select_decode_graph_master_bs(
        [1, 2, 4, 8],
        {
            1: [1],
            2: [2],
            4: [4, 6],
            8: [8, 12],
        },
        bs=2,
        use_sp_a2a=True,
        sp_comm_bs=4,
        attention_compute_bs=7,
    )

    assert selected == 8


def test_select_decode_graph_master_bs_rejects_uncaptured_dynamic_sp_shape():
    with pytest.raises(RuntimeError, match="required_attention_compute_bs=13"):
        select_decode_graph_master_bs(
            [1, 2, 4, 8],
            {
                1: [1],
                2: [2],
                4: [4, 6],
                8: [8, 12],
            },
            bs=2,
            use_sp_a2a=True,
            sp_comm_bs=4,
            attention_compute_bs=13,
        )


def test_validate_decode_graph_copy_capacity_rejects_oversized_remote_metadata():
    graph_vars = {
        "slot_mapping": torch.empty(4, dtype=torch.int32),
        "block_tables": torch.empty(6, 2, dtype=torch.int32),
        "context_lens_for_attn": torch.empty(6, dtype=torch.int32),
        "q_slice_get": torch.empty(4, dtype=torch.int32),
        "q_slice_fill": torch.empty(4, dtype=torch.int32),
        "q_copy_mask": torch.empty(4, dtype=torch.int32),
        "res_slice_get_to_buffer_output": torch.empty(4, dtype=torch.int32),
        "res_slice_fill_to_buffer_output": torch.empty(4, dtype=torch.int32),
        "res_to_buffer_output_mask": torch.empty(4, dtype=torch.int32),
        "res_slice_get_to_buffer_input": torch.empty(2, dtype=torch.int32),
        "res_slice_fill_to_buffer_input": torch.empty(2, dtype=torch.int32),
        "res_to_buffer_input_mask": torch.empty(2, dtype=torch.int32),
    }
    context = SimpleNamespace(
        slot_mapping=torch.empty(2, dtype=torch.int32),
        block_tables=torch.empty(5, 2, dtype=torch.int32),
        context_lens_for_attn=torch.empty(5, dtype=torch.int32),
        q_slice_get=torch.empty(2, dtype=torch.int32),
        q_slice_fill=torch.empty(2, dtype=torch.int32),
        q_copy_mask=torch.empty(2, dtype=torch.int32),
        res_slice_get_to_buffer_output=torch.empty(2, dtype=torch.int32),
        res_slice_fill_to_buffer_output=torch.empty(2, dtype=torch.int32),
        res_to_buffer_output_mask=torch.empty(2, dtype=torch.int32),
        res_slice_get_to_buffer_input=torch.empty(3, dtype=torch.int32),
        res_slice_fill_to_buffer_input=torch.empty(3, dtype=torch.int32),
        res_to_buffer_input_mask=torch.empty(3, dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="res_slice_get_to_buffer_input"):
        validate_decode_graph_copy_capacity(graph_vars, context)

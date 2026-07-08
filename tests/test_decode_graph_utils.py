from types import SimpleNamespace

import pytest
import torch

from nanodeploy.worker.decode_graph_utils import (
    select_decode_graph_master_bs,
    validate_decode_graph_copy_capacity,
)


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

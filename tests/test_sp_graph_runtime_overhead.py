from types import SimpleNamespace

import pytest
import torch

from nanodeploy.config import Config
from nanodeploy.worker.sp_graph_policy import (
    copy_decode_context_to_graph_vars,
    select_decode_graph_bucket,
)


MASTER_BUCKETS = [1, 2, 4, 8, 16, 32]
SP_GRAPH_MAP = {
    1: [1, 17, 32],
    2: [2, 18, 32],
    4: [4, 20, 32],
    8: [8, 24, 40],
    16: [16, 32, 48],
    32: [32, 48, 64],
}


@pytest.mark.parametrize(
    ("actual_master_bs", "actual_attn_bs", "expected"),
    [
        (13, 29, (16, 32)),
        (16, 32, (16, 32)),
        (17, 33, (32, 48)),
        (32, 64, (32, 64)),
    ],
)
def test_select_decode_graph_bucket(
    actual_master_bs, actual_attn_bs, expected
):
    assert (
        select_decode_graph_bucket(
            actual_master_bs,
            actual_attn_bs,
            graph_master_rank_bs=MASTER_BUCKETS,
            sp_graph_map=SP_GRAPH_MAP,
            use_sp_a2a=True,
            sp_backend="hao_basic",
            fixed_sp_size=0,
            sp_comm_bs=None,
        )
        == expected
    )


def test_select_decode_graph_bucket_uses_communication_minimum_for_nccl():
    assert select_decode_graph_bucket(
        13,
        29,
        graph_master_rank_bs=MASTER_BUCKETS,
        sp_graph_map=SP_GRAPH_MAP,
        use_sp_a2a=True,
        sp_backend="nccl",
        fixed_sp_size=0,
        sp_comm_bs=17,
    ) == (32, 32)


@pytest.mark.parametrize(
    ("actual_master_bs", "actual_attn_bs", "message"),
    [
        (33, 33, "exceeds max captured master_bs"),
        (32, 65, "exceeds max captured attn_bs"),
    ],
)
def test_select_decode_graph_bucket_rejects_overflow(
    actual_master_bs, actual_attn_bs, message
):
    with pytest.raises(RuntimeError, match=message):
        select_decode_graph_bucket(
            actual_master_bs,
            actual_attn_bs,
            graph_master_rank_bs=MASTER_BUCKETS,
            sp_graph_map=SP_GRAPH_MAP,
            use_sp_a2a=True,
            sp_backend="hao_basic",
            fixed_sp_size=0,
            sp_comm_bs=None,
        )


def test_copy_decode_context_updates_routing_and_padding_metadata():
    graph_vars = {
        "input_ids": torch.zeros(4, dtype=torch.int64),
        "positions": torch.zeros(4, dtype=torch.int64),
        "slot_mapping": torch.zeros(4, dtype=torch.int32),
        "context_lens": torch.zeros(2, 4, dtype=torch.int32),
        "global_context_lens": torch.zeros(2, 4, dtype=torch.int32),
        "q_mask": torch.zeros(2, 4, dtype=torch.int32),
        "q_dst_row_indices": torch.zeros(2, 4, dtype=torch.int32),
        "actual_attn_bs": torch.zeros((), dtype=torch.int32),
        "res_lse_mask": torch.zeros(2, 4, dtype=torch.int32),
        "block_tables": torch.zeros(5, 3, dtype=torch.int32),
        "context_lens_for_attn": torch.zeros(5, dtype=torch.int32),
        "q_slice_get": torch.zeros(4, dtype=torch.int32),
        "q_slice_fill": torch.zeros(4, dtype=torch.int32),
        "q_copy_mask": torch.zeros(4, dtype=torch.int32),
        "res_slice_get_to_buffer_output": torch.zeros(4, dtype=torch.int32),
        "res_slice_fill_to_buffer_output": torch.zeros(4, dtype=torch.int32),
        "res_to_buffer_output_mask": torch.zeros(4, dtype=torch.int32),
        "res_slice_get_to_buffer_input": torch.zeros(2, dtype=torch.int32),
        "res_slice_fill_to_buffer_input": torch.zeros(2, dtype=torch.int32),
        "res_to_buffer_input_mask": torch.zeros(2, dtype=torch.int32),
        "q_offsets": torch.zeros(3, dtype=torch.int32),
    }
    context = SimpleNamespace(
        slot_mapping=torch.tensor([10, 11], dtype=torch.int32),
        q_dst_row_indices=torch.arange(8, dtype=torch.int32).reshape(2, 4),
        context_lens=torch.full((2, 4), 7, dtype=torch.int32),
        global_context_lens=torch.full((2, 4), 8, dtype=torch.int32),
        q_mask=torch.ones(2, 4, dtype=torch.int32),
        res_lse_mask=torch.ones(2, 4, dtype=torch.int32),
        block_tables=torch.tensor(
            [[10, 11], [20, 21], [30, 31]], dtype=torch.int32
        ),
        context_lens_for_attn=torch.tensor([10, 20, 30], dtype=torch.int32),
        q_slice_get=torch.tensor([0, 1], dtype=torch.int32),
        q_slice_fill=torch.tensor([0, 1], dtype=torch.int32),
        q_copy_mask=torch.ones(2, dtype=torch.int32),
        res_slice_get_to_buffer_output=torch.tensor([0, 1], dtype=torch.int32),
        res_slice_fill_to_buffer_output=torch.tensor([0, 1], dtype=torch.int32),
        res_to_buffer_output_mask=torch.ones(2, dtype=torch.int32),
        res_slice_get_to_buffer_input=torch.tensor([2], dtype=torch.int32),
        res_slice_fill_to_buffer_input=torch.tensor([3], dtype=torch.int32),
        res_to_buffer_input_mask=torch.ones(1, dtype=torch.int32),
        q_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        use_sp_a2a=True,
        attention_compute_bs=3,
    )

    copy_decode_context_to_graph_vars(
        graph_vars,
        torch.tensor([101, 102]),
        torch.tensor([201, 202]),
        2,
        4,
        5,
        context,
        sp_rank=0,
        max_num_seqs=4,
    )

    assert graph_vars["q_dst_row_indices"].tolist() == list(
        map(list, range_rows(2, 4))
    )
    assert graph_vars["actual_attn_bs"].item() == 3
    assert graph_vars["context_lens_for_attn"].tolist() == [10, 20, 30, 1, 1]
    assert graph_vars["block_tables"][3:5, :2].tolist() == [[10, 11], [10, 11]]
    assert graph_vars["res_slice_get_to_buffer_output"].tolist() == [0, 1, 3, 3]
    assert graph_vars["res_slice_fill_to_buffer_output"].tolist() == [0, 1, 2, 3]
    assert graph_vars["res_to_buffer_output_mask"].tolist() == [1, 1, 1, 1]


def range_rows(rows: int, columns: int):
    values = iter(range(rows * columns))
    return [[next(values) for _ in range(columns)] for _ in range(rows)]


@pytest.fixture
def fake_model_dir(tmp_path, monkeypatch):
    hf_config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        max_position_embeddings=16_384,
        model_type="qwen3",
    )
    monkeypatch.setattr(
        "nanodeploy.config.AutoConfig.from_pretrained",
        lambda *args, **kwargs: hf_config,
    )
    return tmp_path


def test_runtime_overhead_profiler_config(fake_model_dir):
    config = Config(
        model=str(fake_model_dir),
        enable_profiler=True,
        profiler_mode="runtime_overhead",
        profiler_ranks=(0,),
    )
    assert config.profiler_mode == "runtime_overhead"
    assert config.profiler_ranks == (0,)


def test_runtime_overhead_profiler_requires_profiler(fake_model_dir):
    with pytest.raises(ValueError, match="requires enable_profiler"):
        Config(
            model=str(fake_model_dir),
            profiler_mode="runtime_overhead",
        )


def test_runtime_overhead_timing_rejects_profiler(fake_model_dir):
    with pytest.raises(ValueError, match="cannot be combined"):
        Config(
            model=str(fake_model_dir),
            enable_profiler=True,
            runtime_overhead_timing=True,
        )


def test_profiler_ranks_rejects_rank_outside_world(fake_model_dir):
    with pytest.raises(ValueError, match="attn_world_size"):
        Config(
            model=str(fake_model_dir),
            enable_profiler=True,
            profiler_ranks=(1,),
        )

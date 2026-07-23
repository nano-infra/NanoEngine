from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.bench_ls_decode_static_layout import (
    ATTENTION_DP,
    ATTENTION_SP,
    BATCH_PER_DP,
    CASE_NAMES,
    OBSERVED_DOP_COUNTS,
    OBSERVED_MASTER_COUNTS,
    _allocate_case,
    _deallocate_case,
    _execution_sequences_by_dp,
    build_case_layout,
    calculate_2x2_effects,
    summarize_case_layout,
    summarize_forward_timings,
)


@pytest.mark.parametrize("case", CASE_NAMES)
def test_static_layout_cases_have_exact_factor_marginals(case: str) -> None:
    layout = build_case_layout(case, seed=17)
    summary = summarize_case_layout(layout)

    assert summary["total_batch_size"] == ATTENTION_DP * BATCH_PER_DP
    for dp_idx, dp_summary in enumerate(summary["per_dp"]):
        if case[2] == "1":
            assert (
                tuple(dp_summary["master_batch_by_sp"])
                == OBSERVED_MASTER_COUNTS[dp_idx]
            )
        else:
            assert dp_summary["master_batch_by_sp"] == [65] * ATTENTION_SP

        if case[1] == "1":
            assert dp_summary["dop_histogram"] == {
                str(dop): OBSERVED_DOP_COUNTS[dp_idx][dop - 1] for dop in (1, 2, 3)
            }
        else:
            assert dp_summary["dop_histogram"] == {"1": BATCH_PER_DP}

        assert max(dp_summary["master_batch_by_sp"]) <= 256
        assert max(dp_summary["receiver_batch_by_sp"]) <= 128
        assert sum(dp_summary["kv_prompt_tokens_by_sp"]) == BATCH_PER_DP * 800


def test_observed_cp_layout_reproduces_short_request_fragment_shape() -> None:
    layout = build_case_layout("T10", context_len=800, cp_shard_tokens=16)

    for seq in layout.sequences:
        positive = sorted(tokens for tokens in seq.dispatched_tokens if tokens > 0)
        assert seq.dispatched_tokens[seq.master_sp_idx] in (16, 800)
        if seq.dop == 1:
            assert positive == [800]
        elif seq.dop == 2:
            assert positive == [16, 784]
        else:
            assert seq.dop == 3
            assert positive == [16, 16, 768]


def test_layouts_are_deterministic_and_seed_controls_tie_breaking() -> None:
    first = build_case_layout("T11", seed=123)
    repeated = build_case_layout("T11", seed=123)
    another_seed = build_case_layout("T11", seed=124)

    assert first == repeated
    assert first != another_seed
    assert summarize_case_layout(first)["global_dop_histogram"] == {
        "1": 574,
        "2": 347,
        "3": 119,
    }
    assert summarize_case_layout(first)["attention_sequence_rank_work"] == 1625


def test_layout_rejects_insufficient_master_or_receiver_capacity() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        build_case_layout("T01", max_num_seqs=137)
    with pytest.raises(ValueError, match="receiver"):
        build_case_layout("T10", max_num_recv_seqs=1)


def test_static_layout_cpu_allocation_and_cleanup_leave_managers_empty() -> None:
    import torch

    from nanodeploy._cpp import (
        BlockContextSlot,
        SPStateManager,
        deserialize,
        prepare_decode_cpp,
        serialize,
    )

    engine_id = "static-layout-cpu-test"
    managers = [
        SPStateManager(
            engine_id=engine_id,
            attention_sp=ATTENTION_SP,
            num_kvcache_blocks=3_000,
            kvcache_block_size=64,
            max_num_seqs=256,
            max_num_batched_tokens=1_024_000,
            max_num_recv_seqs=128,
            reserved_blocks_per_req=0.0,
            enable_dynamic_sp_size=False,
            enable_non_uniform_split=False,
            sp_master_selector="RoundRobin",
        )
        for _ in range(ATTENTION_DP)
    ]
    engine = SimpleNamespace(
        engine_id=engine_id,
        config=SimpleNamespace(loop_count=16),
        scheduler=SimpleNamespace(worker_state=managers),
    )

    allocated = _allocate_case(engine, build_case_layout("T11"))
    for manager, sequences in zip(managers, allocated):
        assert manager.num_running_seqs == BATCH_PER_DP
        assert manager.num_running_tokens == BATCH_PER_DP * 801
        assert len(manager.running) == BATCH_PER_DP
        assert all(
            sum(
                seq.committed_context_len(BlockContextSlot.ACTIVE, rank)
                for rank in range(ATTENTION_SP)
            )
            == 800
            for seq in sequences
        )

    execution_by_dp = _execution_sequences_by_dp(engine, allocated)
    assert len(execution_by_dp[0]) == BATCH_PER_DP + 1
    assert len(execution_by_dp[1]) == BATCH_PER_DP
    assert managers[0].dummy_seqs[2] in execution_by_dp[0]
    for dp_sequences in execution_by_dp:
        for sp_idx in range(ATTENTION_SP):
            # This is the exact metadata operation that failed on the first GPU
            # run when DP0/SP2 had no real master and no allocated dummy.
            meta = prepare_decode_cpp(
                dp_sequences,
                sp_idx,
                ATTENTION_SP,
                64,
                256,
            )
            assert len(meta.input_ids) >= 1

    buffer = torch.empty(8 << 20, dtype=torch.int8)
    data_len = serialize(
        buffer.data_ptr(),
        buffer.numel(),
        execution_by_dp[0],
        False,
        2,
        ATTENTION_SP,
    )
    dp0_sp2_roundtrip = deserialize(buffer.data_ptr(), data_len)
    restored_dummy = next(
        seq
        for seq in dp0_sp2_roundtrip
        if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == 2
    )
    assert list(restored_dummy.block_table(BlockContextSlot.ACTIVE, 2))
    assert (
        len(
            prepare_decode_cpp(
                dp0_sp2_roundtrip,
                2,
                ATTENTION_SP,
                64,
                256,
            ).input_ids
        )
        == 1
    )

    _deallocate_case(engine, allocated)
    for manager in managers:
        assert manager.num_running_seqs == 0
        assert manager.num_running_tokens == 0
        assert len(manager.running) == 0


def test_forward_timing_summary_uses_per_loop_distributed_critical_path() -> None:
    summary = summarize_forward_timings(
        [
            [10.0, 13.0],
            [12.0, 11.0],
            [11.0, 12.0],
        ]
    )

    assert summary["critical_path_by_loop_ms"] == [12.0, 13.0]
    assert summary["critical_path_ms"] == 25.0
    assert summary["per_loop_ms"] == 12.5
    assert summary["rank_per_loop_ms"] == [11.5, 11.5, 11.5]


def test_2x2_effect_calculation_reports_main_and_interaction_terms() -> None:
    effects = calculate_2x2_effects(
        {
            "T00": 90.0,
            "T10": 100.0,
            "T01": 96.0,
            "T11": 120.0,
        }
    )

    assert effects["cp_effect_balanced_ms_T10_minus_T00"] == 10.0
    assert effects["imbalance_effect_dop1_ms_T01_minus_T00"] == 6.0
    assert effects["cp_effect_skewed_ms_T11_minus_T01"] == 24.0
    assert effects["imbalance_effect_observed_cp_ms_T11_minus_T10"] == 20.0
    assert effects["interaction_ms"] == 14.0
    assert effects["current_minus_ideal_ms_T11_minus_T00"] == 30.0
    assert effects["current_over_ideal_T11_div_T00"] == pytest.approx(4 / 3)

from __future__ import annotations

import pytest

from nanodeploy._cpp import (
    BlockContextSlot,
    Scheduler,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)


def _make_scheduler(
    *,
    attention_dp: int = 1,
    attention_sp: int = 4,
    block_size: int = 4,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 4096,
    max_num_recv_seqs: int = 16,
    num_blocks: int = 256,
    reserved_blocks_per_req: float = 0.0,
    initial_dop: int = 0,
    threshold: int = 2,
    memory_scale_up: bool = True,
    enable_ls: bool = True,
    future_kv_admission: bool = True,
) -> Scheduler:
    return Scheduler(
        "",
        1,
        max_num_seqs,
        max_num_batched_tokens,
        max_num_recv_seqs,
        -1,
        attention_dp,
        attention_sp,
        num_blocks,
        block_size,
        "decode",
        reserved_blocks_per_req,
        block_size,
        False,
        False,
        "legacy",
        100_000,
        0,
        False,
        "",
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1,
        1,
        1,
        False,
        "RoundRobin",
        False,
        0,
        enable_ls,
        initial_dop,
        threshold,
        memory_scale_up,
        "centralized",
        "off",
        0.50,
        0.80,
        32,
        64,
        8,
        0,
        future_kv_admission,
    )


def _admit_and_append_dummy(scheduler: Scheduler, prompt_lengths: list[int]):
    seqs = [
        Sequence(list(range(length)), 1.0, 32, True)
        for length in prompt_lengths
    ]
    for seq in seqs:
        scheduler.add(seq)
    result = scheduler.schedule()
    assert result.is_prefill is True

    _append_dummy_for_admission(scheduler, result)
    return seqs, result


def _append_dummy_for_admission(scheduler: Scheduler, result):
    assert result.is_prefill is True

    for dp_idx, admitted_seqs in enumerate(result.dp_seqs):
        worker = scheduler.worker_state[dp_idx]
        for seq in admitted_seqs:
            assert worker.may_append(seq, 1)
            seq.append_token(0, BlockContextSlot.ACTIVE)
            seq.mark_last_token_pending(BlockContextSlot.ACTIVE)
            worker.add_running_tokens(
                seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx, 1
            )


def _free_block_counts(scheduler: Scheduler) -> list[list[int]]:
    return [
        [manager.num_free_blocks for _, manager in scheduler.block_manager(dp).items()]
        for dp in range(len(scheduler.worker_state))
    ]


def _counter_snapshot(scheduler: Scheduler):
    snapshot = []
    for worker in scheduler.worker_state:
        ranks = sorted(worker.block_manager.keys())
        snapshot.append(
            (
                worker.num_running_seqs,
                worker.num_running_tokens,
                [worker.num_recv_seqs_per_sp(rank) for rank in ranks],
                [worker.master_seq_count(rank) for rank in ranks],
            )
        )
    return snapshot


def _finish_and_free(scheduler: Scheduler, dp_idx: int, seq: Sequence) -> None:
    seq.status = SequenceStatus.FINISHED
    worker = scheduler.worker_state[dp_idx]
    worker.deallocate(seq, BlockContextSlot.ACTIVE)
    worker.running.remove(seq)


def _token_ids_for(filtered_dp_sp_seqs):
    return [[[17] for _ in seqs] for seqs in filtered_dp_sp_seqs]


def test_forced_initial_dop_uses_batch_uniform_prompt_placement():
    scheduler = _make_scheduler(initial_dop=3)
    seqs, result = _admit_and_append_dummy(scheduler, [8, 9, 10])

    assert result.ls_initial_kv_dops == [3]
    assert result.ls_initial_kv_ranks == [[0, 1, 2]]
    assert result.ls_initial_prompt_kv_tokens == [
        [
            [3, 3, 2, 0],
            [3, 3, 3, 0],
            [4, 3, 3, 0],
        ]
    ]
    assert result.ls_initial_provisional_pending_targets == [[0, 0, 0]]
    assert len(result.ls_initial_sequence_ids[0]) == 3
    for seq, prompt_len in zip(seqs, [8, 9, 10]):
        committed = [
            seq.committed_context_len(BlockContextSlot.ACTIVE, rank)
            for rank in range(4)
        ]
        assert sum(committed) == prompt_len
        selected = committed[:3]
        assert max(selected) - min(selected) <= 1
        assert committed[3] == 0


def test_feature_flag_off_keeps_legacy_admission_path():
    scheduler = _make_scheduler(attention_sp=2, enable_ls=False)
    seq = Sequence(list(range(4)), 1.0, 16, True)
    scheduler.add(seq)

    result = scheduler.schedule()

    assert result.is_prefill is True
    assert result.ls_initial_batch_ids == []
    assert result.ls_group_ids == []
    assert seq.status == SequenceStatus.RUNNING
    assert scheduler.get_ls_pending_batch_ids() == []
    assert result.ls_pending_batch_count == 0


def test_auto_initial_dop_selects_first_feasible_and_forced_value_does_not_change():
    automatic = _make_scheduler(
        attention_sp=2,
        block_size=4,
        num_blocks=6,
        initial_dop=0,
        future_kv_admission=False,
    )
    _, result = _admit_and_append_dummy(automatic, [20])
    assert result.ls_initial_kv_dops == [2]

    forced = _make_scheduler(
        attention_sp=2,
        block_size=4,
        num_blocks=6,
        initial_dop=1,
        future_kv_admission=False,
    )
    seqs = [Sequence(list(range(20)), 1.0, 32, True)]
    for seq in seqs:
        forced.add(seq)
    with pytest.raises(RuntimeError, match="cannot fit an otherwise empty DP"):
        forced.schedule()

    assert forced.get_ls_pending_batch_ids() == []
    assert forced.get_total_waiting_migration_size() == 1


def test_initial_kv_dop_is_independent_from_first_decode_master_capacity():
    scheduler = _make_scheduler(
        attention_sp=4,
        max_num_batched_tokens=1,
        initial_dop=0,
        threshold=64,
    )
    _, admission = _admit_and_append_dummy(scheduler, [8, 8, 8])

    assert admission.ls_initial_kv_dops == [1]
    decode = scheduler.schedule()
    assert decode.ls_master_batch_sizes == [[1, 1, 1]]


def test_initial_reservation_headroom_follows_shadow_master_load():
    scheduler = _make_scheduler(
        attention_sp=2,
        block_size=4,
        num_blocks=4,
        reserved_blocks_per_req=1.0,
        initial_dop=0,
        threshold=1,
        future_kv_admission=False,
    )
    _, admission = _admit_and_append_dummy(scheduler, [1, 1])

    assert admission.ls_initial_kv_dops == [1]
    decode = scheduler.schedule()
    assert decode.ls_master_batch_sizes == [[1, 1]]


def test_future_kv_peak_keeps_merge_pending_before_prompt_capacity_is_exhausted():
    scheduler = _make_scheduler(
        attention_sp=1,
        block_size=4,
        num_blocks=10,
        initial_dop=1,
    )
    running = Sequence(list(range(4)), 1.0, 20, True)
    scheduler.add(running)
    first = scheduler.schedule()
    _append_dummy_for_admission(scheduler, first)

    waiting = Sequence(list(range(4)), 1.0, 20, True)
    scheduler.add(waiting)
    before_blocks = _free_block_counts(scheduler)

    blocked = scheduler.schedule()

    # Both prompts fit in the ten-block pool, but the LoongServe high-water
    # estimate is (5 + 5) + 2 * 18 = 46 tokens > 40-token capacity.
    assert blocked.ls_initial_batch_ids == []
    assert blocked.ls_atomic_admission_no_fit_count == 1
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[waiting.seq_id]]
    assert waiting.status == SequenceStatus.WAITING
    assert _free_block_counts(scheduler) == before_blocks

    _finish_and_free(scheduler, 0, running)
    admitted = scheduler.schedule()
    assert admitted.ls_initial_sequence_ids == [[waiting.seq_id]]
    assert waiting.status == SequenceStatus.RUNNING


def test_future_kv_peak_uses_completion_overlap_instead_of_summing_all_maxima():
    scheduler = _make_scheduler(
        attention_sp=1,
        block_size=4,
        num_blocks=30,
        initial_dop=1,
    )
    short_output = Sequence(list(range(100)), 1.0, 4, True)
    long_output = Sequence(list(range(4)), 1.0, 100, True)
    scheduler.add(short_output)
    scheduler.add(long_output)

    admitted = scheduler.schedule()

    # Naively summing both terminal KV sizes gives 206 tokens. LoongServe's
    # completion-boundary envelope is max(5 + 98, 5 + 101 + 2 * 2) = 110,
    # so the pair safely fits in the 120-token pool.
    assert admitted.ls_initial_sequence_ids == [[short_output.seq_id, long_output.seq_id]]
    assert all(seq.status == SequenceStatus.RUNNING for seq in (short_output, long_output))


def test_future_kv_peak_shrinks_only_the_unsealed_candidate():
    scheduler = _make_scheduler(
        attention_sp=1,
        block_size=4,
        num_blocks=10,
        initial_dop=1,
    )
    seqs = [Sequence(list(range(4)), 1.0, 20, True) for _ in range(2)]
    for seq in seqs:
        scheduler.add(seq)

    admitted = scheduler.schedule()

    assert admitted.ls_sealed_batch_sequence_ids == [[seqs[0].seq_id]]
    assert admitted.ls_initial_sequence_ids == [[seqs[0].seq_id]]
    assert scheduler.get_ls_pending_batch_ids() == []
    assert [seq.seq_id for seq in scheduler.waiting_migration] == [seqs[1].seq_id]
    assert seqs[0].status == SequenceStatus.RUNNING
    assert seqs[1].status == SequenceStatus.WAITING


def test_iteration_planner_uses_source_style_chunks_and_preserves_history():
    scheduler = _make_scheduler(initial_dop=0, threshold=2)
    seqs, admission = _admit_and_append_dummy(scheduler, [8] * 5)
    initial_history = [
        [
            seq.committed_context_len(BlockContextSlot.ACTIVE, rank)
            for rank in range(4)
        ]
        for seq in seqs
    ]
    initial_tables = [
        [
            list(seq.block_table(BlockContextSlot.ACTIVE, rank))[
                : (initial_history[seq_idx][rank] + 3) // 4
            ]
            for rank in range(4)
        ]
        for seq_idx, seq in enumerate(seqs)
    ]
    assert len({tables[0][0] for tables in initial_tables}) == len(seqs)

    result = scheduler.schedule()

    assert result.is_prefill is False
    assert admission.ls_initial_kv_dops == [1]
    assert result.ls_master_batch_sizes == [[2, 2, 1]]
    assert result.ls_master_dops == [3]
    assert result.ls_iteration_master_assignments == [[0, 0, 1, 1, 2]]
    assert result.ls_scale_reasons == ["compute"]
    assert result.ls_historical_kv_migration_bytes == [0]

    moved = []
    for seq, expected_history, old_tables in zip(seqs, initial_history, initial_tables):
        committed = [
            seq.committed_context_len(BlockContextSlot.ACTIVE, rank)
            for rank in range(4)
        ]
        assert committed == expected_history
        master = seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        assert seq.block_ctx(BlockContextSlot.ACTIVE).pending_token_target_sp == master
        if master != 0:
            moved.append(seq)
            assert committed[master] == 0
            assert len(seq.block_table(BlockContextSlot.ACTIVE, master)) == 1
            assert list(seq.block_table(BlockContextSlot.ACTIVE, 0)) == old_tables[0]

    assert moved
    moved_seq = moved[0]
    moved_master = moved_seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
    metadata = prepare_decode_cpp([*seqs], moved_master, 4, 4, 16)
    moved_index = [
        seq for seq in seqs
        if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == moved_master
    ].index(moved_seq)
    page_id = moved_seq.block_table(BlockContextSlot.ACTIVE, moved_master)[0]
    assert metadata.use_sp_a2a is True
    assert metadata.slot_mapping[moved_index] == page_id * 4

    scheduler.postprocess(
        result.filtered_dp_sp_seqs,
        _token_ids_for(result.filtered_dp_sp_seqs),
        False,
        1.0,
        1,
    )
    assert moved_seq.committed_context_len(
        BlockContextSlot.ACTIVE, moved_master
    ) == 1
    assert moved_seq.block_ctx(BlockContextSlot.ACTIVE).pending_token_present is True


def test_receiver_limit_shrinks_only_the_unsealed_candidate():
    scheduler = _make_scheduler(
        initial_dop=1,
        threshold=1,
        max_num_recv_seqs=1,
    )
    seqs = [Sequence(list(range(8)), 1.0, 32, True) for _ in range(4)]
    for seq in seqs:
        scheduler.add(seq)

    admission = scheduler.schedule()

    # Empty-system feasibility is applied before the batch gets an ID. Once B0
    # is sealed, its complete membership is exactly what admission commits.
    assert admission.ls_sealed_batch_sequence_ids == [
        [seqs[0].seq_id, seqs[1].seq_id]
    ]
    assert admission.ls_initial_sequence_ids == [
        [seqs[0].seq_id, seqs[1].seq_id]
    ]
    assert admission.ls_initial_batch_ids == admission.ls_sealed_batch_ids
    assert all(seq.status == SequenceStatus.RUNNING for seq in seqs[:2])
    assert all(seq.status == SequenceStatus.WAITING for seq in seqs[2:])
    assert [seq.seq_id for seq in scheduler.waiting_migration] == [
        seqs[2].seq_id,
        seqs[3].seq_id,
    ]

    second = scheduler.schedule()
    assert second.ls_initial_sequence_ids == [[seqs[2].seq_id, seqs[3].seq_id]]
    assert second.ls_initial_batch_ids == second.ls_sealed_batch_ids
    assert second.ls_initial_batch_ids != admission.ls_initial_batch_ids


def test_seal_balances_fifo_batches_and_keeps_no_fit_membership_stable():
    scheduler = _make_scheduler(
        attention_dp=4,
        attention_sp=1,
        max_num_seqs=4,
        num_blocks=4,
        initial_dop=1,
        future_kv_admission=False,
    )
    _, occupied = _admit_and_append_dummy(scheduler, [8, 8, 8, 8])
    assert len(occupied.ls_initial_batch_ids) == 4

    seqs = [Sequence([idx], 1.0, 16, True) for idx in range(10)]
    for seq in seqs:
        scheduler.add(seq)
    first = scheduler.schedule()

    expected = [
        [seq.seq_id for seq in seqs[:3]],
        [seq.seq_id for seq in seqs[3:6]],
        [seq.seq_id for seq in seqs[6:8]],
        [seq.seq_id for seq in seqs[8:]],
    ]
    assert first.ls_sealed_batch_sequence_ids == expected
    assert scheduler.get_ls_pending_batch_sequence_ids() == expected
    assert scheduler.get_ls_pending_batch_attempts() == [1, 1, 1, 1]
    assert first.ls_initial_batch_ids == []
    assert first.ls_pending_batch_count == 4
    assert first.ls_pending_request_count == 10
    assert first.ls_atomic_admission_no_fit_count == 4

    batch_ids = scheduler.get_ls_pending_batch_ids()
    second = scheduler.schedule()
    assert scheduler.get_ls_pending_batch_ids() == batch_ids
    assert scheduler.get_ls_pending_batch_sequence_ids() == expected
    assert scheduler.get_ls_pending_batch_attempts() == [2, 2, 2, 2]
    assert second.ls_oldest_pending_batch_age_steps == 1


def test_balanced_multi_dp_batches_are_each_admitted_whole_to_one_group():
    scheduler = _make_scheduler(
        attention_dp=4,
        attention_sp=1,
        max_num_seqs=4,
        num_blocks=16,
        initial_dop=1,
    )
    seqs = [Sequence([idx], 1.0, 16, True) for idx in range(10)]
    for seq in seqs:
        scheduler.add(seq)

    result = scheduler.schedule()

    expected = [
        [seq.seq_id for seq in seqs[:3]],
        [seq.seq_id for seq in seqs[3:6]],
        [seq.seq_id for seq in seqs[6:8]],
        [seq.seq_id for seq in seqs[8:]],
    ]
    assert result.ls_sealed_batch_sequence_ids == expected
    assert result.ls_initial_sequence_ids == expected
    assert len(set(result.ls_initial_group_ids)) == 4
    assert [[seq.seq_id for seq in dp_seqs] for dp_seqs in result.dp_seqs] == expected
    assert scheduler.get_ls_group_sequence_ids() == expected
    assert scheduler.get_ls_pending_batch_ids() == []


def test_seal_limit_leaves_tail_unsealed_without_reforming_old_batches():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=1,
        max_num_seqs=2,
        num_blocks=4,
        initial_dop=1,
        future_kv_admission=False,
    )
    _admit_and_append_dummy(scheduler, [8, 8])
    seqs = [Sequence([idx], 1.0, 16, True) for idx in range(6)]
    for seq in seqs:
        scheduler.add(seq)

    first = scheduler.schedule()
    first_pending_ids = scheduler.get_ls_pending_batch_ids()
    assert first.ls_sealed_batch_sequence_ids == [
        [seqs[0].seq_id, seqs[1].seq_id],
        [seqs[2].seq_id, seqs[3].seq_id],
    ]
    assert [seq.seq_id for seq in scheduler.waiting_migration] == [
        seqs[4].seq_id,
        seqs[5].seq_id,
    ]

    second = scheduler.schedule()
    assert second.ls_sealed_batch_sequence_ids == [
        [seqs[4].seq_id],
        [seqs[5].seq_id],
    ]
    assert scheduler.get_ls_pending_batch_ids()[:2] == first_pending_ids
    assert scheduler.get_ls_pending_batch_sequence_ids() == [
        [seqs[0].seq_id, seqs[1].seq_id],
        [seqs[2].seq_id, seqs[3].seq_id],
        [seqs[4].seq_id],
        [seqs[5].seq_id],
    ]


def test_no_fit_batch_admits_whole_after_capacity_is_released():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=5,
        initial_dop=1,
        future_kv_admission=False,
    )
    occupied, _ = _admit_and_append_dummy(scheduler, [8])
    waiting = Sequence(list(range(8)), 1.0, 32, True)
    scheduler.add(waiting)

    before_blocks = _free_block_counts(scheduler)
    no_fit = scheduler.schedule()
    batch_id = scheduler.get_ls_pending_batch_ids()[0]
    assert no_fit.ls_initial_batch_ids == []
    assert no_fit.ls_atomic_admission_no_fit_count == 1
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[waiting.seq_id]]
    assert scheduler.get_ls_pending_batch_attempts() == [1]
    assert waiting.status == SequenceStatus.WAITING
    assert _free_block_counts(scheduler) == before_blocks
    assert scheduler.get_total_waiting_migration_size() == 1
    assert scheduler.is_finished() is False

    _finish_and_free(scheduler, 0, occupied[0])
    admitted = scheduler.schedule()
    assert admitted.ls_initial_batch_ids == [batch_id]
    assert admitted.ls_initial_sequence_ids == [[waiting.seq_id]]
    assert admitted.ls_initial_admission_attempts == [2]
    assert waiting.status == SequenceStatus.RUNNING
    assert scheduler.get_ls_pending_batch_ids() == []


def test_permanently_unplaceable_singleton_fails_at_seal_instead_of_hanging():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=3,
        initial_dop=1,
    )
    seq = Sequence(list(range(8)), 1.0, 32, True)
    scheduler.add(seq)

    with pytest.raises(RuntimeError, match="cannot fit an otherwise empty DP"):
        scheduler.schedule()

    assert scheduler.get_ls_pending_batch_ids() == []
    assert [queued.seq_id for queued in scheduler.waiting_migration] == [seq.seq_id]
    assert seq.status == SequenceStatus.WAITING


def test_duplicate_arrival_is_rejected_before_any_batch_ownership_is_published():
    scheduler = _make_scheduler(attention_sp=1, initial_dop=1)
    seq = Sequence([1], 1.0, 16, True)
    scheduler.add(seq)
    scheduler.add(seq)

    with pytest.raises(RuntimeError, match="duplicate live sequence ownership"):
        scheduler.schedule()

    assert scheduler.get_ls_pending_batch_ids() == []
    assert scheduler.get_ls_active_batch_owners() == []
    assert len(scheduler.waiting_migration) == 2
    assert seq.status == SequenceStatus.WAITING

    scheduler.waiting_migration.popleft()
    admitted = scheduler.schedule()
    assert admitted.ls_initial_sequence_ids == [[seq.seq_id]]


def test_bounded_bypass_admits_a_later_complete_batch():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=8,
        initial_dop=1,
        future_kv_admission=False,
    )
    occupied, occupied_admission = _admit_and_append_dummy(scheduler, [8])

    blocked = [Sequence(list(range(8)), 1.0, 32, True) for _ in range(2)]
    for seq in blocked:
        scheduler.add(seq)
    first = scheduler.schedule()
    blocked_id = scheduler.get_ls_pending_batch_ids()[0]
    assert first.ls_initial_batch_ids == []
    assert scheduler.get_ls_pending_batch_sequence_ids() == [
        [seq.seq_id for seq in blocked]
    ]

    later = Sequence([1], 1.0, 16, True)
    scheduler.add(later)
    bypass = scheduler.schedule()
    assert bypass.ls_initial_sequence_ids == [[later.seq_id]]
    assert later.status == SequenceStatus.RUNNING
    assert scheduler.get_ls_pending_batch_ids() == [blocked_id]
    assert scheduler.get_ls_pending_batch_sequence_ids() == [
        [seq.seq_id for seq in blocked]
    ]
    assert scheduler.get_ls_pending_batch_attempts() == [2]
    assert all(seq.status == SequenceStatus.WAITING for seq in blocked)
    _append_dummy_for_admission(scheduler, bypass)

    later_id = bypass.ls_initial_batch_ids[0]
    _finish_and_free(scheduler, 0, occupied[0])
    delayed = scheduler.schedule()
    assert delayed.ls_initial_batch_ids == [blocked_id]
    assert delayed.ls_initial_sequence_ids == [[seq.seq_id for seq in blocked]]
    assert scheduler.get_ls_group_initial_batch_ids() == [[
        occupied_admission.ls_initial_batch_ids[0],
        later_id,
        blocked_id,
    ]]
    assert scheduler.get_ls_group_initial_admission_orders() == [[0, 1, 2]]
    assert scheduler.get_ls_group_sequence_ids() == [[
        later.seq_id,
        *(seq.seq_id for seq in blocked),
    ]]


def test_standalone_atomic_commit_failure_rolls_back_every_sequence():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=8,
        initial_dop=1,
        future_kv_admission=False,
    )
    seqs = [Sequence(list(range(4)), 1.0, 16, True) for _ in range(2)]
    for seq in seqs:
        scheduler.add(seq)
    before_blocks = _free_block_counts(scheduler)
    before_counters = _counter_snapshot(scheduler)
    scheduler.set_ls_admission_failure_after_publications_for_test(1)

    failed = scheduler.schedule()

    assert failed.ls_initial_batch_ids == []
    assert failed.ls_atomic_admission_no_fit_count == 0
    assert failed.ls_atomic_admission_rollback_count == 1
    assert _free_block_counts(scheduler) == before_blocks
    assert _counter_snapshot(scheduler) == before_counters
    assert scheduler.get_ls_group_ids() == []
    assert scheduler.get_ls_pending_batch_sequence_ids() == [
        [seq.seq_id for seq in seqs]
    ]
    assert len(scheduler.worker_state[0].running) == 0
    pending_id = scheduler.get_ls_pending_batch_ids()[0]
    assert scheduler.get_ls_active_batch_owners() == [
        (seq.seq_id, pending_id) for seq in seqs
    ]
    for seq in seqs:
        assert seq.status == SequenceStatus.WAITING
        assert all(
            not seq.block_table(BlockContextSlot.ACTIVE, rank)
            for rank in range(1)
        )
        assert list(seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens) == [0]

    admitted = scheduler.schedule()
    assert admitted.ls_initial_sequence_ids == [[seq.seq_id for seq in seqs]]
    assert admitted.ls_initial_admission_attempts == [2]


def test_merge_atomic_commit_failure_restores_existing_group():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=8,
        initial_dop=1,
        future_kv_admission=False,
    )
    running, initial = _admit_and_append_dummy(scheduler, [4])
    original_group_ids = scheduler.get_ls_group_ids()
    original_records = scheduler.get_ls_group_initial_batch_ids()
    original_blocks = _free_block_counts(scheduler)
    original_counters = _counter_snapshot(scheduler)

    seqs = [Sequence(list(range(4)), 1.0, 16, True) for _ in range(2)]
    for seq in seqs:
        scheduler.add(seq)
    scheduler.set_ls_admission_failure_after_publications_for_test(1)
    failed = scheduler.schedule()

    assert failed.ls_initial_batch_ids == []
    assert scheduler.get_ls_group_ids() == original_group_ids
    assert scheduler.get_ls_group_initial_batch_ids() == original_records
    assert scheduler.get_ls_group_sequence_ids() == [[running[0].seq_id]]
    assert _free_block_counts(scheduler) == original_blocks
    assert _counter_snapshot(scheduler) == original_counters
    assert failed.ls_atomic_admission_no_fit_count == 0
    assert failed.ls_atomic_admission_rollback_count == 1
    assert scheduler.get_ls_pending_batch_sequence_ids() == [
        [seq.seq_id for seq in seqs]
    ]
    assert all(seq.status == SequenceStatus.WAITING for seq in seqs)

    admitted = scheduler.schedule()
    assert admitted.ls_initial_group_ids == initial.ls_initial_group_ids
    assert admitted.ls_initial_sequence_ids == [[seq.seq_id for seq in seqs]]
    assert admitted.ls_atomic_admission_merge_count == 1


def test_sp_batch_allocation_failure_rolls_back_multi_rank_blocks_and_counters():
    scheduler = _make_scheduler(
        attention_sp=2,
        num_blocks=8,
        initial_dop=2,
    )
    seqs = [Sequence(list(range(5)), 1.0, 16, True) for _ in range(2)]
    for seq in seqs:
        scheduler.add(seq)
    before_blocks = _free_block_counts(scheduler)
    before_counters = _counter_snapshot(scheduler)
    scheduler.set_ls_admission_failure_after_allocations_for_test(1)

    failed = scheduler.schedule()

    assert failed.ls_atomic_admission_rollback_count == 1
    assert failed.ls_atomic_admission_no_fit_count == 0
    assert _free_block_counts(scheduler) == before_blocks
    assert _counter_snapshot(scheduler) == before_counters
    assert all(seq.status == SequenceStatus.WAITING for seq in seqs)
    for seq in seqs:
        assert all(
            not seq.block_table(BlockContextSlot.ACTIVE, rank)
            for rank in range(2)
        )


def test_admission_searches_all_groups_instead_of_only_the_oldest():
    scheduler = _make_scheduler(
        attention_sp=2,
        max_num_seqs=2,
        num_blocks=6,
        initial_dop=1,
        future_kv_admission=False,
    )
    _, first = _admit_and_append_dummy(scheduler, [12])
    _, second = _admit_and_append_dummy(scheduler, [4])
    assert first.ls_initial_group_ids != second.ls_initial_group_ids
    assert first.ls_initial_kv_ranks == [[0]]
    assert second.ls_initial_kv_ranks == [[1]]

    newcomer = Sequence(list(range(8)), 1.0, 32, True)
    scheduler.add(newcomer)
    admitted = scheduler.schedule()

    assert admitted.ls_initial_sequence_ids == [[newcomer.seq_id]]
    assert admitted.ls_initial_group_ids == second.ls_initial_group_ids
    assert admitted.ls_initial_group_ids != first.ls_initial_group_ids
    assert admitted.ls_atomic_admission_merge_count == 1


def test_kv_capacity_preemption_is_structured_and_frontier_safe():
    scheduler = _make_scheduler(
        attention_sp=1,
        block_size=4,
        num_blocks=2,
        initial_dop=1,
        threshold=64,
        future_kv_admission=False,
    )
    seqs, _ = _admit_and_append_dummy(scheduler, [3])

    result = scheduler.schedule()

    assert result.ls_group_ids == []
    assert result.ls_preempted_sequence_ids == [seqs[0].seq_id]
    assert "append capacity" in result.ls_preemption_reasons[0]
    assert seqs[0].status == SequenceStatus.WAITING
    assert seqs[0].block_ctx(BlockContextSlot.ACTIVE).pending_token_present is False
    assert len(result.dp_seqs[0]) == 1


def test_forced_initial_dop_validates_receiver_shape_and_plans_legal_chunks():
    scheduler = _make_scheduler(
        attention_sp=2,
        initial_dop=2,
        threshold=64,
        max_num_recv_seqs=2,
    )
    seqs, admission = _admit_and_append_dummy(scheduler, [8] * 4)

    assert admission.ls_initial_kv_dops == [2]
    result = scheduler.schedule()

    assert result.ls_master_batch_sizes == [[2, 2]]
    assert result.ls_iteration_master_assignments == [[0, 0, 1, 1]]
    assert all(seq.status == SequenceStatus.RUNNING for seq in seqs)


def test_empty_dp_uses_scheduler_owned_dummies_for_every_sp_rank():
    scheduler = _make_scheduler(attention_dp=2, attention_sp=2, initial_dop=1)
    _admit_and_append_dummy(scheduler, [8])

    result = scheduler.schedule()

    assert len(result.dp_seqs[1]) == 2
    assert [
        seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        for seq in result.dp_seqs[1]
    ] == [0, 1]
    for sp_rank in range(2):
        meta = prepare_decode_cpp(result.dp_seqs[1], sp_rank, 2, 4, 16)
        assert list(meta.input_ids) != []


def test_multi_group_compute_pressure_merges_into_oldest_group():
    scheduler = _make_scheduler(
        attention_sp=2,
        max_num_seqs=2,
        initial_dop=1,
        threshold=1,
    )
    first, first_admission = _admit_and_append_dummy(scheduler, [8, 8])

    second = [Sequence(list(range(8)), 1.0, 32, True) for _ in range(2)]
    for seq in second:
        scheduler.add(seq)
    second_admission = scheduler.schedule()
    assert second_admission.is_prefill is True
    for seq in second:
        assert scheduler.worker_state[0].may_append(seq, 1)
        seq.append_token(0, BlockContextSlot.ACTIVE)
        seq.mark_last_token_pending(BlockContextSlot.ACTIVE)
        scheduler.worker_state[0].add_running_tokens(
            seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx, 1
        )

    assert first_admission.ls_initial_group_ids == [0]
    assert second_admission.ls_initial_group_ids == [1]
    assert first_admission.ls_initial_kv_ranks == [[0]]
    assert second_admission.ls_initial_kv_ranks == [[1]]
    result = scheduler.schedule()

    assert result.ls_group_ids == [0]
    assert result.ls_group_rank_allocations == [[0, 1]]
    assert result.ls_master_batch_sizes == [[2, 2]]
    assert [seq.seq_id for seq in result.dp_seqs[0][:4]] == [
        *(seq.seq_id for seq in first),
        *(seq.seq_id for seq in second),
    ]


def test_kv_pressure_merges_with_highest_capacity_donor_instead_of_oldest():
    scheduler = _make_scheduler(
        attention_sp=3,
        block_size=4,
        num_blocks=3,
        max_num_seqs=2,
        max_num_recv_seqs=1,
        initial_dop=1,
        threshold=64,
        future_kv_admission=False,
    )
    first, first_admission = _admit_and_append_dummy(scheduler, [7])
    second, second_admission = _admit_and_append_dummy(scheduler, [3])
    third, third_admission = _admit_and_append_dummy(scheduler, [1])

    assert first_admission.ls_initial_group_ids == [0]
    assert second_admission.ls_initial_group_ids == [1]
    assert third_admission.ls_initial_group_ids == [2]
    assert scheduler.get_ls_group_sequence_ids() == [
        [first[0].seq_id],
        [second[0].seq_id],
        [third[0].seq_id],
    ]

    result = scheduler.schedule()

    # Group 0 is append-constrained. Group 2 has more usable append slack than
    # the older group 1, so the LoongServe-style capacity pass consumes rank 2
    # and leaves group 1 independent.
    assert result.ls_group_ids == [0, 1]
    assert result.ls_group_rank_allocations == [[0, 2], [1]]
    assert result.ls_iteration_sequence_ids == [
        [first[0].seq_id, third[0].seq_id],
        [second[0].seq_id],
    ]
    assert result.ls_preempted_sequence_ids == []


def test_preempt_readmit_then_merge_keeps_each_live_sequence_once():
    scheduler = _make_scheduler(
        attention_sp=2,
        max_num_seqs=4,
        initial_dop=1,
        threshold=1,
    )
    first_batch, first_admission = _admit_and_append_dummy(scheduler, [8, 8])
    re_admitted, survivor = first_batch
    scheduler.preempt(0, re_admitted)

    recovery_ids = scheduler.get_ls_pending_batch_ids()
    assert len(recovery_ids) == 1
    assert recovery_ids[0] != first_admission.ls_initial_batch_ids[0]
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[re_admitted.seq_id]]
    assert scheduler.get_ls_pending_batch_is_recovery() == [True]
    assert scheduler.get_ls_pending_batch_parent_batch_ids() == [
        first_admission.ls_initial_batch_ids[0]
    ]
    assert survivor.status == SequenceStatus.RUNNING

    readmission = scheduler.schedule()
    assert readmission.is_prefill is True
    assert readmission.ls_initial_batch_ids == recovery_ids
    assert readmission.ls_initial_is_recovery_batch == [True]
    assert readmission.ls_initial_parent_batch_ids == [
        first_admission.ls_initial_batch_ids[0]
    ]
    worker = scheduler.worker_state[0]
    assert worker.may_append(re_admitted, 1)
    re_admitted.append_token(0, BlockContextSlot.ACTIVE)
    re_admitted.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(
        re_admitted.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx, 1
    )

    newcomer = Sequence(list(range(8)), 1.0, 32, True)
    scheduler.add(newcomer)
    merged_admission = scheduler.schedule()
    assert merged_admission.is_prefill is True
    assert worker.may_append(newcomer, 1)
    newcomer.append_token(0, BlockContextSlot.ACTIVE)
    newcomer.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(
        newcomer.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx, 1
    )

    result = scheduler.schedule()
    live_ids = result.ls_iteration_sequence_ids[0]

    assert live_ids == [survivor.seq_id, re_admitted.seq_id, newcomer.seq_id]
    assert len(live_ids) == len(set(live_ids)) == 3
    assert scheduler.get_ls_group_initial_admission_orders() == [[0, 1, 2]]
    assert scheduler.get_ls_group_initial_sequence_ids() == [[
        [re_admitted.seq_id, survivor.seq_id],
        [re_admitted.seq_id],
        [newcomer.seq_id],
    ]]


def test_preempt_rejects_non_owner_dp_without_mutating_ls_state():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=2,
        max_num_seqs=4,
        initial_dop=1,
    )
    (seq,), _ = _admit_and_append_dummy(scheduler, [8])
    owner_dp = seq.block_ctx(BlockContextSlot.ACTIVE).dp_idx
    wrong_dp = 1 - owner_dp

    free_blocks = _free_block_counts(scheduler)
    counters = _counter_snapshot(scheduler)
    group_ids = scheduler.get_ls_group_ids()
    group_sequences = scheduler.get_ls_group_sequence_ids()
    batch_owners = scheduler.get_ls_active_batch_owners()
    running = [[item.seq_id for item in worker.running] for worker in scheduler.worker_state]

    with pytest.raises(RuntimeError, match="does not own"):
        scheduler.preempt(wrong_dp, seq)

    assert seq.status == SequenceStatus.RUNNING
    assert scheduler.get_ls_pending_batch_ids() == []
    assert scheduler.get_ls_group_ids() == group_ids
    assert scheduler.get_ls_group_sequence_ids() == group_sequences
    assert scheduler.get_ls_active_batch_owners() == batch_owners
    assert _free_block_counts(scheduler) == free_blocks
    assert _counter_snapshot(scheduler) == counters
    assert [
        [item.seq_id for item in worker.running]
        for worker in scheduler.worker_state
    ] == running


def test_batch_drop_immediately_reduces_active_masters_and_reclaims_empty_ranks():
    scheduler = _make_scheduler(initial_dop=1, threshold=2)
    seqs, _ = _admit_and_append_dummy(scheduler, [8] * 5)
    first = scheduler.schedule()
    assert first.ls_master_dops == [3]
    scheduler.postprocess(
        first.filtered_dp_sp_seqs,
        _token_ids_for(first.filtered_dp_sp_seqs),
        False,
        1.0,
        1,
    )

    worker = scheduler.worker_state[0]
    for seq in seqs[1:]:
        seq.status = SequenceStatus.FINISHED
        worker.deallocate(seq, BlockContextSlot.ACTIVE)
        worker.running.remove(seq)

    second = scheduler.schedule()

    assert second.ls_real_batch_sizes == [1]
    assert second.ls_master_dops == [1]
    assert second.ls_master_batch_sizes == [[1]]
    assert second.ls_group_rank_allocations == [[0]]

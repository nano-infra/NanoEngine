from __future__ import annotations

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

    for dp_idx, admitted_seqs in enumerate(result.dp_seqs):
        worker = scheduler.worker_state[dp_idx]
        for seq in admitted_seqs:
            assert worker.may_append(seq, 1)
            seq.append_token(0, BlockContextSlot.ACTIVE)
            seq.mark_last_token_pending(BlockContextSlot.ACTIVE)
            worker.add_running_tokens(
                seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx, 1
            )
    return seqs, result


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


def test_auto_initial_dop_selects_first_feasible_and_forced_value_does_not_change():
    automatic = _make_scheduler(
        attention_sp=2,
        block_size=4,
        num_blocks=6,
        initial_dop=0,
    )
    _, result = _admit_and_append_dummy(automatic, [20])
    assert result.ls_initial_kv_dops == [2]

    forced = _make_scheduler(
        attention_sp=2,
        block_size=4,
        num_blocks=6,
        initial_dop=1,
    )
    seqs = [Sequence(list(range(20)), 1.0, 32, True)]
    for seq in seqs:
        forced.add(seq)
    delayed = forced.schedule()

    assert delayed.ls_initial_kv_dops == []
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
    )
    _, admission = _admit_and_append_dummy(scheduler, [1, 1])

    assert admission.ls_initial_kv_dops == [1]
    decode = scheduler.schedule()
    assert decode.ls_master_batch_sizes == [[1, 1]]


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


def test_receiver_pressure_admits_only_a_stable_feasible_prefix():
    scheduler = _make_scheduler(
        initial_dop=1,
        threshold=1,
        max_num_recv_seqs=1,
    )
    seqs = [Sequence(list(range(8)), 1.0, 32, True) for _ in range(4)]
    for seq in seqs:
        scheduler.add(seq)

    admission = scheduler.schedule()

    assert admission.ls_initial_sequence_ids == [
        [seqs[0].seq_id, seqs[1].seq_id]
    ]
    assert all(seq.status == SequenceStatus.RUNNING for seq in seqs[:2])
    assert all(seq.status == SequenceStatus.WAITING for seq in seqs[2:])
    assert [seq.seq_id for seq in scheduler.waiting_migration] == [
        seqs[2].seq_id,
        seqs[3].seq_id,
    ]


def test_kv_capacity_preemption_is_structured_and_frontier_safe():
    scheduler = _make_scheduler(
        attention_sp=1,
        block_size=4,
        num_blocks=2,
        initial_dop=1,
        threshold=64,
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


def test_preempt_readmit_then_merge_keeps_each_live_sequence_once():
    scheduler = _make_scheduler(
        attention_sp=2,
        max_num_seqs=4,
        initial_dop=1,
        threshold=1,
    )
    first_batch, _ = _admit_and_append_dummy(scheduler, [8, 8])
    re_admitted, survivor = first_batch
    scheduler.preempt(0, re_admitted)

    readmission = scheduler.schedule()
    assert readmission.is_prefill is True
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

    assert live_ids == [re_admitted.seq_id, survivor.seq_id, newcomer.seq_id]
    assert len(live_ids) == len(set(live_ids)) == 3


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

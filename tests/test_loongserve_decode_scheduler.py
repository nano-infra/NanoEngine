from nanodeploy._cpp import (
    BlockContextSlot,
    Scheduler,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)


def _make_scheduler(
    attention_sp=4,
    block_size=4,
    min_batch=2,
    loop_count=1,
    scheduler_mode="centralized",
    max_recv=8,
):
    return Scheduler(
        "",
        loop_count,
        8,  # max_num_seqs
        1024,
        max_recv,
        -1,
        1,  # attention_dp
        attention_sp,
        128,
        block_size,
        "decode",
        0.0,
        block_size,
        False,
        False,
        "legacy",
        100000,
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
        scheduler_mode,
        True,
        False,
        "block",
        min_batch,
    )


def _running_seq(token_ids, attention_sp, master_sp):
    seq = Sequence(token_ids, 1.0, 32, False)
    seq.active("", attention_sp, 1)
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.dp_idx = 0
    ctx.master_sp_idx = master_sp
    ctx.num_dispatched_tokens = [
        len(token_ids) if sp_idx == master_sp else 0
        for sp_idx in range(attention_sp)
    ]
    seq.status = SequenceStatus.RUNNING
    return seq


def _distributed_running_seq(token_ids, attention_sp, master_sp, dispatched_tokens):
    seq = Sequence(token_ids, 1.0, 32, False)
    seq.active("", attention_sp, 1)
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.dp_idx = 0
    ctx.master_sp_idx = master_sp
    ctx.num_dispatched_tokens = dispatched_tokens
    seq.status = SequenceStatus.RUNNING
    return seq


def _append_target(seq):
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    return ctx.append_sp_idx if ctx.append_sp_idx >= 0 else ctx.master_sp_idx


def _token_ids_for(result):
    return [
        [[1000 + sp_idx * 100 + seq_idx] for seq_idx, _seq in enumerate(sp_seqs)]
        for sp_idx, sp_seqs in enumerate(result.filtered_dp_sp_seqs)
    ]


def test_loongserve_decode_scheduler_scales_out_new_kv_without_migration():
    scheduler = _make_scheduler(attention_sp=4, block_size=4, min_batch=2)
    worker = scheduler.worker_state[0]
    seqs = [_running_seq([1, 2, 3, 4], 4, 0) for _ in range(6)]
    for seq in seqs:
        worker.allocate(seq)
        worker.running.append(seq)

    result = scheduler.schedule()

    assert result.is_prefill is False
    masters = {seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx for seq in seqs}
    append_targets = {_append_target(seq) for seq in seqs}
    assert masters == {0}
    assert len(append_targets) >= 3
    for seq in seqs:
        target = _append_target(seq)
        assert len(seq.block_table(BlockContextSlot.ACTIVE, target)) >= 1

    scheduler.postprocess(result.filtered_dp_sp_seqs, _token_ids_for(result), False, 0.0, 1)

    masters_after_append = {seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx for seq in seqs}
    assert masters_after_append == append_targets
    for seq in seqs:
        assert seq.block_ctx(BlockContextSlot.ACTIVE).append_sp_idx == -1

    next_result = scheduler.schedule()
    assert next_result.is_prefill is False
    next_masters = {seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx for seq in seqs}
    assert len(next_masters) >= 3


def test_loongserve_decode_scheduler_scales_down_by_draining_append_targets():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=4)
    worker = scheduler.worker_state[0]
    seq_on_0 = _running_seq([1, 2, 3, 4], 2, 0)
    seq_on_1 = _running_seq([5, 6, 7, 8], 2, 1)
    for seq in [seq_on_0, seq_on_1]:
        worker.allocate(seq)
        worker.running.append(seq)

    result = scheduler.schedule()

    assert list(result.loongserve_occupied_instances[0]) == [0, 1]
    assert list(result.loongserve_append_instances[0]) == [0]
    assert list(result.loongserve_draining_instances[0]) == [1]
    assert _append_target(seq_on_0) == 0
    assert _append_target(seq_on_1) == 0

    scheduler.postprocess(result.filtered_dp_sp_seqs, _token_ids_for(result), False, 0.0, 1)

    assert seq_on_1.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == 0
    assert seq_on_1.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens == [1, 4]

    next_result = scheduler.schedule()
    next_append = list(next_result.loongserve_append_instances[0])
    next_draining = list(next_result.loongserve_draining_instances[0])
    assert len(next_append) == 1
    assert len(next_draining) == 1
    assert sorted(next_append + next_draining) == [0, 1]
    assert next_result.sp_q_matrix[0][0][1] + next_result.sp_q_matrix[0][1][0] >= 1
    assert next_result.sp_res_matrix[0][0][1] + next_result.sp_res_matrix[0][1][0] >= 1


def test_loongserve_draining_rank_decode_metadata_keeps_remote_kv_peer():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=4)
    worker = scheduler.worker_state[0]
    seq_on_0 = _running_seq([1, 2, 3, 4], 2, 0)
    seq_on_1 = _running_seq([5, 6, 7, 8], 2, 1)
    for seq in [seq_on_0, seq_on_1]:
        worker.allocate(seq)
        worker.running.append(seq)

    result = scheduler.schedule()
    scheduler.postprocess(result.filtered_dp_sp_seqs, _token_ids_for(result), False, 0.0, 1)

    assert seq_on_1.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == 0
    assert seq_on_1.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens == [1, 4]

    meta_master = prepare_decode_cpp([seq_on_0, seq_on_1], 0, 2, 4, 8)
    assert meta_master.use_sp_a2a is True
    assert meta_master.input_ids == [seq_on_0.last_token, seq_on_1.last_token]
    assert meta_master.context_lens_for_attn == [5, 1]
    assert meta_master.q_slice_get == [0, 1]
    assert meta_master.q_slice_fill == [0, 1]
    assert meta_master.q_offsets == [0, 2, 2]
    assert meta_master.global_context_lens_flat[:8] == [5, 1, 0, 0, 0, 0, 0, 0]
    assert meta_master.global_context_lens_flat[8:16] == [0, 4, 0, 0, 0, 0, 0, 0]

    meta_draining = prepare_decode_cpp([seq_on_0, seq_on_1], 1, 2, 4, 8)
    assert meta_draining.use_sp_a2a is True
    assert meta_draining.input_ids == []
    assert meta_draining.context_lens_for_attn == [4]
    assert meta_draining.q_slice_get == []
    assert meta_draining.q_slice_fill == []
    assert meta_draining.res_slice_get_to_buffer_input == [0]
    assert meta_draining.res_slice_fill_to_buffer_input == [1]
    assert meta_draining.q_offsets == [0, 1, 1]


def test_loongserve_draining_rank_releases_after_request_finish_without_migration():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=4)
    worker = scheduler.worker_state[0]
    seq_on_0 = _running_seq([1, 2, 3, 4], 2, 0)
    seq_on_1 = _running_seq([5, 6, 7, 8], 2, 1)
    seq_on_1.max_tokens = 1
    for seq in [seq_on_0, seq_on_1]:
        worker.allocate(seq)
        worker.running.append(seq)

    result = scheduler.schedule()
    scheduler.postprocess(result.filtered_dp_sp_seqs, _token_ids_for(result), False, 0.0, 1)

    assert seq_on_1.status == SequenceStatus.FINISHED
    assert seq_on_1.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens == [0, 0]

    next_result = scheduler.schedule()
    assert list(next_result.loongserve_occupied_instances[0]) == [0]
    assert list(next_result.loongserve_draining_instances[0]) == []


def test_loongserve_decode_scheduler_respects_remote_recv_capacity_for_full_graph():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=8, max_recv=1)
    worker = scheduler.worker_state[0]
    seqs = [
        _distributed_running_seq([1, 2, 3, 4, 5], 2, 0, [1, 4])
        for _ in range(3)
    ]
    for seq in seqs:
        worker.allocate(seq)
        worker.running.append(seq)

    result = scheduler.schedule()

    scheduled_real = [seq for seq in seqs if seq in list(result.dp_seqs[0])]
    assert len(scheduled_real) == 1
    assert result.sp_recv_counts[0][1] == 1

    meta_peer = prepare_decode_cpp(result.dp_seqs[0], 1, 2, 4, 8)
    assert len(meta_peer.res_slice_get_to_buffer_input) == 1
    assert len(meta_peer.res_slice_fill_to_buffer_input) == 1


def test_loongserve_decode_scheduler_requires_single_token_steps():
    try:
        _make_scheduler(attention_sp=2, block_size=4, min_batch=2, loop_count=2)
    except RuntimeError as exc:
        assert "loop_count=1" in str(exc)
    else:
        raise AssertionError("expected loop_count guard for LoongServe decode scheduler")


def test_loongserve_decode_scheduler_rejects_decentralized_mode():
    try:
        _make_scheduler(attention_sp=2, block_size=4, min_batch=2, scheduler_mode="decentralized")
    except RuntimeError as exc:
        assert "centralized" in str(exc)
    else:
        raise AssertionError("expected centralized scheduler_mode guard")


def test_loongserve_decode_scheduler_clears_state_for_empty_decode_batch():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=4)
    worker = scheduler.worker_state[0]
    seq = _running_seq([1, 2, 3, 4], 2, 0)
    worker.allocate(seq)
    worker.running.append(seq)

    result = scheduler.schedule()
    assert list(result.loongserve_occupied_instances[0]) == [0]

    worker.running.remove(seq)

    empty_result = scheduler.schedule()
    assert list(empty_result.loongserve_occupied_instances[0]) == []
    assert list(empty_result.loongserve_append_instances[0]) == []
    assert list(empty_result.loongserve_draining_instances[0]) == []


def test_decode_kv_accounting_uses_num_dispatched_tokens():
    scheduler = _make_scheduler(attention_sp=3, block_size=4, min_batch=16)
    worker = scheduler.worker_state[0]
    seq_a = _running_seq([1, 2, 3, 4], 3, 0)
    seq_b = _running_seq([5, 6, 7, 8, 9], 3, 1)
    worker.allocate(seq_a)
    worker.allocate(seq_b)
    worker.running.append(seq_a)
    worker.running.append(seq_b)

    assert worker.decode_total_used_tokens_per_sp() == [4, 5, 0]
    assert worker.decode_batch_used_tokens_per_sp([seq_b]) == [0, 5, 0]
    assert worker.decode_occupied_instances([0, 5, 0]) == [1]
    assert worker.select_decode_scale_up_ranks([4, 5, 0], [0, 1], 1) == [2]


def test_prepare_decode_accepts_master_with_preallocated_empty_kv_block():
    scheduler = _make_scheduler(attention_sp=2, block_size=4, min_batch=16)
    worker = scheduler.worker_state[0]
    seq = _running_seq([1, 2, 3, 4], 2, 0)
    worker.allocate(seq)

    worker.set_decode_master(seq, 1)
    assert worker.may_append_on_sp(seq, 1, 1)

    meta = prepare_decode_cpp([seq], 1, 2, 4, 8)
    block_id = seq.block_table(BlockContextSlot.ACTIVE, 1)[0]
    assert meta.slot_mapping == [block_id * 4]

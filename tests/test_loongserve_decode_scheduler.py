from nanodeploy._cpp import (
    BlockContextSlot,
    Scheduler,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)


def _make_scheduler(attention_sp=4, block_size=4, min_batch=2):
    return Scheduler(
        "",
        1,  # loop_count
        8,  # max_num_seqs
        1024,
        8,
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
        "centralized",
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

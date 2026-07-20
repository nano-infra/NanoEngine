from __future__ import annotations

from collections.abc import Callable

import pytest

from nanodeploy._cpp import (
    BlockContextSlot,
    LSAddError,
    LSAdmissionKind,
    LSAdmissionTargetKind,
    ScheduleAction,
    Scheduler,
    Sequence,
    SequenceMetric,
    SequenceStatus,
)


def _make_scheduler(
    *,
    attention_dp: int = 1,
    attention_sp: int = 2,
    block_size: int = 4,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 4096,
    max_num_recv_seqs: int = 16,
    num_blocks: int = 64,
    reserved_blocks_per_req: float = 0.0,
    initial_dop: int = 0,
    threshold: int = 2,
    memory_scale_up: bool = True,
    future_kv_admission: bool = True,
    max_num_ooe: int = 10,
    running_max_req_size: int = 1000,
    admission_max_tokens_per_pool: int = 4096,
    loop_count: int = 1,
) -> Scheduler:
    return Scheduler(
        "",
        loop_count,
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
        True,
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
        max_num_ooe,
        running_max_req_size,
        admission_max_tokens_per_pool,
        128,
    )


def _sequence(
    prompt_length: int,
    *,
    max_tokens: int = 16,
    ignore_eos: bool = True,
) -> Sequence:
    return Sequence(list(range(prompt_length)), 1.0, max_tokens, ignore_eos)


def _add(scheduler: Scheduler, sequence: Sequence, expected_dp: int) -> None:
    result = scheduler.add(sequence)
    assert result.accepted is True
    assert result.error == LSAddError.NONE
    assert result.assigned_dp == expected_dp
    assert sequence.assigned_dp == expected_dp


def _record_ids_by_dp(result, attention_dp: int) -> list[list[int]]:
    ids = [[] for _ in range(attention_dp)]
    for record in result.ls_admission_records:
        ids[record.dp_idx].append(record.sequence.seq_id)
    return ids


def _postprocess_decode(
    scheduler: Scheduler,
    result,
    token_id: int = 17,
) -> None:
    token_ids = [
        [
            [token_id + loop_idx for loop_idx in range(result.execution_loop_count)]
            for _ in sequences
        ]
        for sequences in result.filtered_dp_sp_seqs
    ]
    scheduler.postprocess(
        result.filtered_dp_sp_seqs,
        token_ids,
        False,
        1.0,
        result.execution_loop_count,
    )


def _free_block_counts(scheduler: Scheduler) -> list[list[int]]:
    return [
        [
            manager.num_free_blocks
            for _, manager in sorted(scheduler.block_manager(dp_idx).items())
        ]
        for dp_idx in range(len(scheduler.worker_state))
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


def _assert_no_persistent_admission_batch(scheduler: Scheduler) -> None:
    assert scheduler.get_ls_pending_batch_ids() == []
    assert scheduler.get_ls_pending_batch_sequence_ids() == []
    assert scheduler.get_ls_pending_batch_attempts() == []
    assert scheduler.get_ls_pending_batch_is_recovery() == []
    assert scheduler.get_ls_pending_batch_parent_batch_ids() == []
    assert scheduler.get_ls_active_batch_owners() == []


def test_typed_add_consumes_rr_only_for_structurally_new_attempts():
    scheduler = _make_scheduler(attention_dp=4, attention_sp=1)

    rejected_eos = _sequence(1, max_tokens=2, ignore_eos=False)
    result = scheduler.add(rejected_eos)
    assert result.accepted is False
    assert result.assigned_dp == 0
    assert result.error == LSAddError.IGNORE_EOS_REQUIRED
    assert rejected_eos.assigned_dp == 0
    assert scheduler.get_ls_arrival_order(rejected_eos.seq_id) is None

    first = _sequence(1, max_tokens=2)
    _add(scheduler, first, 1)
    assert scheduler.get_ls_arrival_order(first.seq_id) == 0

    duplicate = scheduler.add(first)
    assert duplicate.accepted is False
    assert duplicate.assigned_dp == 1
    assert duplicate.error == LSAddError.ALREADY_ADDED_OR_ASSIGNED

    reused_id = _sequence(1, max_tokens=2)
    reused_id.seq_id = first.seq_id
    duplicate_id = scheduler.add(reused_id)
    assert duplicate_id.accepted is False
    assert duplicate_id.assigned_dp == -1
    assert duplicate_id.error == LSAddError.ALREADY_ADDED_OR_ASSIGNED

    second = _sequence(1, max_tokens=2)
    _add(scheduler, second, 2)

    rejected_limit = _sequence(1, max_tokens=0)
    result = scheduler.add(rejected_limit)
    assert result.accepted is False
    assert result.assigned_dp == 3
    assert result.error == LSAddError.INVALID_MAX_TOKENS

    rejected_future = _sequence(1, max_tokens=10_000)
    result = scheduler.add(rejected_future)
    assert result.accepted is False
    assert result.assigned_dp == 0
    assert result.error == LSAddError.FUTURE_TOKEN_NO_FIT

    third = _sequence(1, max_tokens=2)
    _add(scheduler, third, 1)
    assert scheduler.get_ls_arrival_order(second.seq_id) == 1
    assert scheduler.get_ls_arrival_order(third.seq_id) == 2
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [
        [],
        [first.seq_id, third.seq_id],
        [second.seq_id],
        [],
    ]
    _assert_no_persistent_admission_batch(scheduler)


def test_typed_add_current_exact_rejection_is_detached_but_consumes_rr():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=1,
        num_blocks=6,
        initial_dop=1,
        future_kv_admission=False,
    )
    too_large = _sequence(20, max_tokens=2)

    rejected = scheduler.add(too_large)

    assert rejected.accepted is False
    assert rejected.assigned_dp == 0
    assert rejected.error == LSAddError.CURRENT_EXACT_NO_FIT
    assert too_large.assigned_dp == 0
    assert too_large.status == SequenceStatus.WAITING
    assert scheduler.get_ls_arrival_order(too_large.seq_id) is None
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[], []]

    accepted = _sequence(1, max_tokens=2)
    _add(scheduler, accepted, 1)
    _assert_no_persistent_admission_batch(scheduler)


def test_max_tokens_one_finishes_inside_cpp_bootstrap_without_decode_arrays():
    scheduler = _make_scheduler()
    sequence = _sequence(3, max_tokens=1)
    original_tokens = list(sequence.token_ids)
    _add(scheduler, sequence, 0)

    result = scheduler.schedule()

    assert result.action == ScheduleAction.ADMISSION
    assert result.is_prefill is False
    assert result.dp_seqs == []
    assert result.dp_sp_seqs == []
    assert result.filtered_dp_sp_seqs == []
    assert result.ls_real_decode_ids_by_dp == []
    assert result.ls_running_ids_by_dp_after_commit == [[]]
    assert result.kv_consolidation_plan is None
    assert len(result.ls_admission_records) == 1

    record = result.ls_admission_records[0]
    assert record.sequence is sequence
    assert record.dp_idx == 0
    assert record.admission_kind == LSAdmissionKind.FRESH
    assert record.target_kind == LSAdmissionTargetKind.STANDALONE
    assert record.group_id_after_commit is None
    assert record.planned_kv_dop == len(record.planned_kv_ranks) >= 1
    assert record.bootstrap_finished is True
    assert record.bootstrap_token_id == 0

    assert list(sequence.token_ids) == [*original_tokens, 0]
    assert sequence.num_completed_tokens == 1
    assert sequence.status == SequenceStatus.FINISHED
    assert scheduler.get_ls_group_ids() == []
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[]]
    assert scheduler.is_finished() is True
    _assert_no_persistent_admission_batch(scheduler)

def test_max_tokens_two_runs_exactly_one_real_decode_after_bootstrap():
    scheduler = _make_scheduler()
    sequence = _sequence(3, max_tokens=2)
    _add(scheduler, sequence, 0)

    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert admission.ls_admission_records[0].bootstrap_finished is False
    assert admission.ls_admission_records[0].group_id_after_commit is not None
    assert admission.ls_running_ids_by_dp_after_commit == [[sequence.seq_id]]
    assert list(sequence.token_ids)[-1] == 0
    assert sequence.num_completed_tokens == 1

    decode = scheduler.schedule()
    assert decode.action == ScheduleAction.DECODE
    assert decode.is_prefill is False
    assert decode.ls_admission_records == []
    assert decode.ls_real_decode_ids_by_dp == [[sequence.seq_id]]
    assert decode.ls_running_ids_by_dp_after_commit == [[sequence.seq_id]]
    assert sequence.seq_id in [item.seq_id for item in decode.dp_seqs[0]]

    _postprocess_decode(scheduler, decode)
    assert sequence.num_completed_tokens == 2
    assert sequence.status == SequenceStatus.FINISHED
    assert list(sequence.token_ids)[-2:] == [0, 17]
    assert scheduler.is_finished() is True


def test_chunked_16_reserves_kv_and_shortens_the_final_decode_step():
    scheduler = _make_scheduler(
        attention_sp=2,
        block_size=64,
        num_blocks=16,
        loop_count=16,
    )
    sequence = _sequence(63, max_tokens=18)
    _add(scheduler, sequence, 0)

    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert sequence.num_completed_tokens == 1

    first_decode = scheduler.schedule()
    assert first_decode.action == ScheduleAction.DECODE
    assert first_decode.execution_loop_count == 16

    _postprocess_decode(scheduler, first_decode)
    assert sequence.num_completed_tokens == 17
    assert sequence.status == SequenceStatus.RUNNING

    final_decode = scheduler.schedule()
    assert final_decode.execution_loop_count == 1
    _postprocess_decode(scheduler, final_decode, token_id=99)

    assert sequence.num_completed_tokens == 18
    assert sequence.status == SequenceStatus.FINISHED
    assert scheduler.is_finished() is True


def test_chunked_16_fills_an_idle_dp_with_kv_backed_rank_dummies():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=8,
        block_size=64,
        num_blocks=64,
        loop_count=16,
        threshold=128,
    )
    sequence = _sequence(221, max_tokens=698)
    _add(scheduler, sequence, 0)

    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION

    decode = scheduler.schedule()
    assert decode.action == ScheduleAction.DECODE
    assert decode.execution_loop_count == 16
    assert decode.ls_real_decode_ids_by_dp == [[sequence.seq_id], []]
    assert [len(sequences) for sequences in decode.dp_seqs] == [8, 8]

    idle_dp = decode.dp_seqs[1]
    assert sorted(
        dummy.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        for dummy in idle_dp
    ) == list(range(8))
    for dummy in idle_dp:
        master = dummy.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        assert dummy.last_block_page_id(BlockContextSlot.ACTIVE, master) >= 0


def test_pool_local_fifo_membership_then_stable_need_sort_one_batch_per_pool():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=2,
        max_num_seqs=2,
        future_kv_admission=False,
    )
    prompt_lengths = [4, 6, 8, 6, 2, 10]
    sequences = [_sequence(length) for length in prompt_lengths]
    for index, sequence in enumerate(sequences):
        _add(scheduler, sequence, index % 2)

    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [
        [sequences[0].seq_id, sequences[2].seq_id, sequences[4].seq_id],
        [sequences[1].seq_id, sequences[3].seq_id, sequences[5].seq_id],
    ]

    result = scheduler.schedule()

    # Membership is the first two requests from each pool. Sorting happens
    # only after membership is frozen; equal needs retain FIFO order.
    assert _record_ids_by_dp(result, 2) == [
        [sequences[2].seq_id, sequences[0].seq_id],
        [sequences[1].seq_id, sequences[3].seq_id],
    ]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [
        [sequences[4].seq_id],
        [sequences[5].seq_id],
    ]
    batch_ids_by_dp = [
        {record.batch_id for record in result.ls_admission_records if record.dp_idx == dp_idx}
        for dp_idx in range(2)
    ]
    assert all(len(batch_ids) == 1 for batch_ids in batch_ids_by_dp)
    assert batch_ids_by_dp[0] != batch_ids_by_dp[1]
    assert result.ls_pending_batch_count == 0
    _assert_no_persistent_admission_batch(scheduler)


def test_exact_no_fit_shrinks_fifo_tail_not_sorted_tail():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=6,
        initial_dop=1,
        future_kv_admission=False,
    )
    sequences = [
        _sequence(length, max_tokens=2)
        for length in [4, 8, 6, 2]
    ]
    for sequence in sequences:
        _add(scheduler, sequence, 0)

    result = scheduler.schedule()

    # Full membership cannot satisfy block-rounded bootstrap headroom. The
    # exact retry removes the FIFO tail [len=2, len=6], then sorts surviving
    # membership [len=4, len=8] into [len=8, len=4]. Removing the sorted tail
    # would have incorrectly discarded the first FIFO request instead.
    assert [record.sequence.seq_id for record in result.ls_admission_records] == [
        sequences[1].seq_id,
        sequences[0].seq_id,
    ]
    assert len({record.batch_id for record in result.ls_admission_records}) == 1
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[
        sequences[2].seq_id,
        sequences[3].seq_id,
    ]]
    assert scheduler.get_ls_group_sequence_ids() == [[
        sequences[1].seq_id,
        sequences[0].seq_id,
    ]]
    assert result.ls_atomic_admission_no_fit_count == 2
    assert result.ls_pending_batch_count == 0
    assert result.ls_pending_request_count == 2
    _assert_no_persistent_admission_batch(scheduler)


def test_pool_local_ooe_counter_and_limit_do_not_leak_between_pools():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=2,
        num_blocks=10,
        max_num_ooe=1,
    )
    running = [
        _sequence(4, max_tokens=40),
        _sequence(4, max_tokens=40),
    ]
    _add(scheduler, running[0], 0)
    _add(scheduler, running[1], 1)
    scheduler.schedule()

    blocker_dp0 = _sequence(4, max_tokens=40)
    prefix_dp1 = _sequence(1, max_tokens=2)
    bypass_dp0 = _sequence(1, max_tokens=2)
    second_dp1 = _sequence(1, max_tokens=2)
    for expected_dp, sequence in enumerate(
        [blocker_dp0, prefix_dp1, bypass_dp0, second_dp1]
    ):
        _add(scheduler, sequence, expected_dp % 2)

    bypass = scheduler.schedule()

    assert bypass.action == ScheduleAction.DECODE
    assert bypass.ls_real_decode_ids_by_dp == [
        [running[0].seq_id],
        [running[1].seq_id],
    ]
    assert _record_ids_by_dp(bypass, 2) == [
        [bypass_dp0.seq_id],
        [prefix_dp1.seq_id, second_dp1.seq_id],
    ]
    assert scheduler.get_ls_num_ooe() == [1, 0]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[blocker_dp0.seq_id], []]
    _assert_no_persistent_admission_batch(scheduler)

    _postprocess_decode(scheduler, bypass)
    capped_dp0 = _sequence(1, max_tokens=2)
    _add(scheduler, capped_dp0, 0)

    capped = scheduler.schedule()

    # DP0 has reached its bounded-OOE limit and cannot pass its blocker. DP1's
    # independent counter remains at zero while both pools continue Decode.
    assert _record_ids_by_dp(capped, 2) == [[], []]
    assert scheduler.get_ls_num_ooe() == [1, 0]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[
        blocker_dp0.seq_id,
        capped_dp0.seq_id,
    ], []]


def test_full_pool_never_borrows_idle_capacity_from_another_dp():
    scheduler = _make_scheduler(
        attention_dp=2,
        attention_sp=1,
        num_blocks=10,
    )
    running_dp0 = _sequence(4, max_tokens=20)
    finish_on_bootstrap_dp1 = _sequence(1, max_tokens=1)
    _add(scheduler, running_dp0, 0)
    _add(scheduler, finish_on_bootstrap_dp1, 1)
    first = scheduler.schedule()
    assert _record_ids_by_dp(first, 2) == [
        [running_dp0.seq_id],
        [finish_on_bootstrap_dp1.seq_id],
    ]
    assert first.ls_running_ids_by_dp_after_commit == [[running_dp0.seq_id], []]

    blocked_dp0 = _sequence(4, max_tokens=20)
    admitted_dp1 = _sequence(1, max_tokens=2)
    _add(scheduler, blocked_dp0, 0)
    _add(scheduler, admitted_dp1, 1)

    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert _record_ids_by_dp(result, 2) == [[], [admitted_dp1.seq_id]]
    assert result.ls_real_decode_ids_by_dp == [[running_dp0.seq_id], []]
    assert result.ls_running_ids_by_dp_after_commit == [
        [running_dp0.seq_id],
        [admitted_dp1.seq_id],
    ]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[blocked_dp0.seq_id], []]
    assert blocked_dp0.status == SequenceStatus.WAITING
    assert blocked_dp0.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 0
    assert admitted_dp1.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 1
    _assert_no_persistent_admission_batch(scheduler)


def test_admission_and_entry_snapshot_decode_share_typed_abi_without_overlap():
    scheduler = _make_scheduler(attention_sp=2)
    running = _sequence(4, max_tokens=4)
    _add(scheduler, running, 0)
    assert scheduler.schedule().action == ScheduleAction.ADMISSION

    newcomer = _sequence(2, max_tokens=4)
    _add(scheduler, newcomer, 0)
    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert result.is_prefill is False
    assert [record.sequence.seq_id for record in result.ls_admission_records] == [
        newcomer.seq_id
    ]
    assert result.ls_real_decode_ids_by_dp == [[running.seq_id]]
    assert result.ls_running_ids_by_dp_after_commit == [[running.seq_id, newcomer.seq_id]]
    assert newcomer.seq_id not in [sequence.seq_id for sequence in result.dp_seqs[0]]
    assert result.kv_consolidation_plan is None

    _postprocess_decode(scheduler, result)
    next_decode = scheduler.schedule()
    assert next_decode.action == ScheduleAction.DECODE
    assert next_decode.ls_admission_records == []
    assert next_decode.ls_real_decode_ids_by_dp == [[running.seq_id, newcomer.seq_id]]
    assert next_decode.ls_running_ids_by_dp_after_commit == [[
        running.seq_id,
        newcomer.seq_id,
    ]]


def test_capacity_append_admission_keeps_newcomer_out_of_entry_decode():
    scheduler = _make_scheduler(
        attention_sp=1,
        num_blocks=10,
        initial_dop=1,
    )
    running = _sequence(4, max_tokens=20)
    _add(scheduler, running, 0)
    assert scheduler.schedule().action == ScheduleAction.ADMISSION

    # With the only rank already owned, this request must capacity-append to
    # the existing group. Admission is a CPU/KV side effect; this model
    # iteration still uses the step-entry snapshot containing only the older
    # request.
    newcomer = _sequence(1, max_tokens=2)
    _add(scheduler, newcomer, 0)
    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert len(result.ls_admission_records) == 1
    record = result.ls_admission_records[0]
    assert record.sequence is newcomer
    assert record.target_kind == LSAdmissionTargetKind.CAPACITY_APPEND
    assert record.group_id_after_commit is not None
    assert result.ls_real_decode_ids_by_dp == [[running.seq_id]]
    assert result.ls_iteration_sequence_ids == [[running.seq_id]]
    assert result.ls_iteration_master_assignments == [[0]]
    assert set(result.ls_running_ids_by_dp_after_commit[0]) == {
        running.seq_id,
        newcomer.seq_id,
    }
    model_ids = [sequence.seq_id for sequence in result.dp_seqs[0]]
    assert running.seq_id in model_ids
    assert newcomer.seq_id not in model_ids
    _assert_no_persistent_admission_batch(scheduler)


@pytest.mark.parametrize(
    "inject_failure",
    [
        lambda scheduler: scheduler.set_ls_admission_failure_after_allocations_for_test(0),
        lambda scheduler: scheduler.set_ls_post_admission_component_failure_for_test(0),
    ],
    ids=["isolated-allocation", "post-admission-component"],
)
def test_admission_failure_rolls_back_and_falls_back_to_entry_decode(
    inject_failure: Callable[[Scheduler], None],
):
    scheduler = _make_scheduler(attention_sp=2, num_blocks=10)
    running = _sequence(4, max_tokens=20)
    _add(scheduler, running, 0)
    scheduler.schedule()

    newcomer = _sequence(4, max_tokens=20)
    _add(scheduler, newcomer, 0)
    before = (
        _free_block_counts(scheduler),
        _counter_snapshot(scheduler),
        scheduler.get_ls_group_ids(),
        scheduler.get_ls_group_sequence_ids(),
        scheduler.get_ls_waiting_sequence_ids_by_dp(),
        scheduler.get_ls_num_ooe(),
        list(newcomer.token_ids),
        newcomer.num_tokens,
        newcomer.status,
    )
    inject_failure(scheduler)

    result = scheduler.schedule()

    after = (
        _free_block_counts(scheduler),
        _counter_snapshot(scheduler),
        scheduler.get_ls_group_ids(),
        scheduler.get_ls_group_sequence_ids(),
        scheduler.get_ls_waiting_sequence_ids_by_dp(),
        scheduler.get_ls_num_ooe(),
        list(newcomer.token_ids),
        newcomer.num_tokens,
        newcomer.status,
    )
    assert result.action == ScheduleAction.DECODE
    assert result.ls_admission_records == []
    assert result.ls_real_decode_ids_by_dp == [[running.seq_id]]
    assert result.ls_running_ids_by_dp_after_commit == [[running.seq_id]]
    assert result.ls_atomic_admission_rollback_count == 1
    assert after == before
    assert all(
        not newcomer.block_table(BlockContextSlot.ACTIVE, rank)
        for rank in range(2)
    )
    _assert_no_persistent_admission_batch(scheduler)


@pytest.mark.parametrize(
    "inject_failure",
    [
        lambda scheduler: scheduler.set_ls_admission_failure_after_allocations_for_test(1),
        lambda scheduler: scheduler.set_ls_post_admission_component_failure_for_test(1),
    ],
    ids=["isolated-allocation", "post-admission-component"],
)
def test_pool_local_admission_failure_keeps_other_pool_prepared_until_global_decode_validation(
    inject_failure: Callable[[Scheduler], None],
):
    scheduler = _make_scheduler(attention_dp=2, attention_sp=2, num_blocks=12)
    running = [_sequence(4, max_tokens=20), _sequence(4, max_tokens=20)]
    _add(scheduler, running[0], 0)
    _add(scheduler, running[1], 1)
    scheduler.schedule()

    newcomers = [_sequence(2, max_tokens=20), _sequence(2, max_tokens=20)]
    _add(scheduler, newcomers[0], 0)
    _add(scheduler, newcomers[1], 1)
    before_blocks = _free_block_counts(scheduler)
    before_counters = _counter_snapshot(scheduler)
    before_ooe = scheduler.get_ls_num_ooe()
    inject_failure(scheduler)

    result = scheduler.schedule()

    # The countdown lets DP0 retain its prepared admission. The two variants
    # fail DP1 either during isolated allocation prepare or after its complete
    # admission overlay, at the required Decode-component boundary. In both
    # cases DP1 alone rolls back and rebuilds decode-only before global validate.
    assert result.action == ScheduleAction.DECODE
    assert _record_ids_by_dp(result, 2) == [[newcomers[0].seq_id], []]
    assert result.ls_real_decode_ids_by_dp == [
        [running[0].seq_id],
        [running[1].seq_id],
    ]
    assert result.ls_running_ids_by_dp_after_commit == [
        [running[0].seq_id, newcomers[0].seq_id],
        [running[1].seq_id],
    ]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [
        [],
        [newcomers[1].seq_id],
    ]
    assert scheduler.get_ls_num_ooe() == before_ooe
    assert newcomers[1].status == SequenceStatus.WAITING
    assert list(newcomers[1].token_ids) == [0, 1]
    assert all(
        not newcomers[1].block_table(BlockContextSlot.ACTIVE, rank)
        for rank in range(2)
    )
    assert result.ls_atomic_admission_rollback_count == 1
    # Only DP0's committed admission may change physical/counter state. DP1's
    # rollback is exact even though DP0 was already prepared.
    assert sum(_free_block_counts(scheduler)[1]) == sum(before_blocks[1])
    assert _counter_snapshot(scheduler)[1][0] == before_counters[1][0]


def test_decode_only_capacity_no_fit_offload_is_exclusive_and_never_commits_waiting_admission():
    scheduler = _make_scheduler(
        attention_sp=1,
        # One block is permanently owned by the rank-local cadence dummy; the
        # two requests each need two blocks including bootstrap headroom.
        num_blocks=5,
        initial_dop=1,
        future_kv_admission=False,
    )
    running = [_sequence(4, max_tokens=20), _sequence(4, max_tokens=20)]
    _add(scheduler, running[0], 0)
    _add(scheduler, running[1], 0)
    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert _record_ids_by_dp(admission, 1) == [[
        running[0].seq_id,
        running[1].seq_id,
    ]]

    # Both running requests exactly occupy the only rank. The next request is
    # valid in an empty pool but cannot be admitted beside them, so it must stay
    # waiting when their required Decode transaction reaches a block boundary.
    waiting = _sequence(5, max_tokens=2)
    _add(scheduler, waiting, 0)

    offload = None
    for _ in range(12):
        result = scheduler.schedule()
        if result.ls_preempted_sequence_ids:
            offload = result
            break
        assert result.action == ScheduleAction.DECODE
        assert result.ls_admission_records == []
        assert result.ls_real_decode_ids_by_dp == [[
            running[0].seq_id,
            running[1].seq_id,
        ]]
        _postprocess_decode(scheduler, result)

    assert offload is not None
    assert offload.action == ScheduleAction.ADMISSION
    assert offload.ls_admission_records == []
    assert offload.ls_real_decode_ids_by_dp == []
    assert offload.kv_consolidation_plan is None
    assert offload.ls_preempted_sequence_ids == [running[1].seq_id]
    assert len(offload.ls_preemption_reasons) == 1
    assert "OFFLOAD" in offload.ls_preemption_reasons[0]
    assert running[0].status == SequenceStatus.RUNNING
    assert running[1].status == SequenceStatus.PAUSED_OFFLOAD
    assert waiting.status == SequenceStatus.WAITING
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[
        running[1].seq_id,
        waiting.seq_id,
    ]]


def test_bypass_post_admission_component_rollback_restores_fifo_and_ooe_counter():
    scheduler = _make_scheduler(attention_sp=2, num_blocks=10)
    running = _sequence(4, max_tokens=40)
    _add(scheduler, running, 0)
    scheduler.schedule()

    blocker = _sequence(4, max_tokens=40)
    bypass_candidate = _sequence(1, max_tokens=2)
    _add(scheduler, blocker, 0)
    _add(scheduler, bypass_candidate, 0)
    before_blocks = _free_block_counts(scheduler)
    before_counters = _counter_snapshot(scheduler)
    scheduler.set_ls_post_admission_component_failure_for_test(0)

    rolled_back = scheduler.schedule()

    assert rolled_back.action == ScheduleAction.DECODE
    assert rolled_back.ls_admission_records == []
    assert rolled_back.ls_real_decode_ids_by_dp == [[running.seq_id]]
    assert rolled_back.ls_atomic_admission_rollback_count == 1
    assert scheduler.get_ls_num_ooe() == [0]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[
        blocker.seq_id,
        bypass_candidate.seq_id,
    ]]
    assert _free_block_counts(scheduler) == before_blocks
    assert _counter_snapshot(scheduler) == before_counters
    assert list(bypass_candidate.token_ids) == [0]
    assert bypass_candidate.status == SequenceStatus.WAITING
    _assert_no_persistent_admission_batch(scheduler)

    _postprocess_decode(scheduler, rolled_back)
    committed = scheduler.schedule()
    assert [record.sequence.seq_id for record in committed.ls_admission_records] == [
        bypass_candidate.seq_id
    ]
    assert scheduler.get_ls_num_ooe() == [1]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[blocker.seq_id]]


def test_prepared_offload_and_readmission_preserve_progress_and_assignment():
    scheduler = _make_scheduler(attention_sp=2, num_blocks=12, initial_dop=1)
    sequence = _sequence(4, max_tokens=3)
    sequence.metric = SequenceMetric(sequence.seq_id, sequence.num_prompt_tokens)
    _add(scheduler, sequence, 0)

    first = scheduler.schedule()
    assert first.action == ScheduleAction.ADMISSION
    assert first.ls_admission_records[0].admission_kind == LSAdmissionKind.FRESH
    tokens_after_first_bootstrap = list(sequence.token_ids)
    blocks_after_first_bootstrap = _free_block_counts(scheduler)
    first_scheduled_time = sequence.metric.first_scheduled_time
    first_token_time = sequence.metric.first_token_time
    last_token_time = sequence.metric.last_token_time
    epoch_after_admission = scheduler.get_ls_pool_resource_epochs()
    assert epoch_after_admission == [1]
    assert sequence.num_completed_tokens == 1
    assert sequence.metric.num_generated_tokens == 1

    scheduler.preempt(0, sequence)

    assert sequence.status == SequenceStatus.PAUSED_OFFLOAD
    assert sequence.assigned_dp == 0
    assert list(sequence.token_ids) == tokens_after_first_bootstrap
    assert sequence.metric.first_scheduled_time == first_scheduled_time
    assert sequence.metric.first_token_time == first_token_time
    assert sequence.metric.last_token_time == last_token_time
    assert sequence.metric.num_generated_tokens == 1
    assert scheduler.get_ls_pool_resource_epochs() == [2]
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[sequence.seq_id]]
    assert scheduler.get_ls_group_ids() == []
    assert _counter_snapshot(scheduler)[0][0] == 0
    assert all(
        not sequence.block_table(BlockContextSlot.ACTIVE, rank)
        for rank in range(2)
    )
    assert all(
        after >= before
        for after_row, before_row in zip(_free_block_counts(scheduler), blocks_after_first_bootstrap)
        for after, before in zip(after_row, before_row)
    )

    readmission = scheduler.schedule()

    assert readmission.action == ScheduleAction.ADMISSION
    assert len(readmission.ls_admission_records) == 1
    record = readmission.ls_admission_records[0]
    assert record.sequence is sequence
    assert record.admission_kind == LSAdmissionKind.OFFLOAD_READMIT
    assert record.bootstrap_finished is False
    assert record.group_id_after_commit is not None
    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.assigned_dp == 0
    assert list(sequence.token_ids) == [*tokens_after_first_bootstrap, 0]
    assert sequence.num_completed_tokens == 2
    assert sequence.metric.first_scheduled_time == first_scheduled_time
    assert sequence.metric.first_token_time == first_token_time
    assert sequence.metric.last_token_time > last_token_time
    assert sequence.metric.num_generated_tokens == 2
    assert len(sequence.metric.itl_samples) == 1
    assert scheduler.get_ls_waiting_sequence_ids_by_dp() == [[]]
    assert readmission.ls_running_ids_by_dp_after_commit == [[sequence.seq_id]]
    assert readmission.ls_pool_resource_epoch_before == [2]
    assert readmission.ls_pool_resource_epoch_after == [3]
    _assert_no_persistent_admission_batch(scheduler)

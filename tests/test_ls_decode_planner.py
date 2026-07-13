from __future__ import annotations

import pytest

from nanodeploy._cpp import (
    BlockContextSlot,
    SPStateManager,
    Sequence,
    SequenceStatus,
)


_SP_SIZE = 8
_BLOCK_SIZE = 256
_T_COMPUTE = 4


def _make_manager(
    *, max_num_seqs: int = 64, num_kvcache_blocks: int = 128
) -> SPStateManager:
    # Sequence::block_size defaults to 256 when constructing SPStateManager
    # directly instead of through Scheduler, so keep both block sizes aligned.
    return SPStateManager(
        engine_id="ls-planner-test",
        attention_sp=_SP_SIZE,
        num_kvcache_blocks=num_kvcache_blocks,
        kvcache_block_size=_BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=100_000,
        max_num_recv_seqs=max_num_seqs,
        reserved_blocks_per_req=0.0,
        enable_dynamic_sp_size=False,
        enable_non_uniform_split=False,
        sp_master_selector="RoundRobin",
    )


def _make_pending_sequence(
    seq_idx: int,
    *,
    committed_tokens: tuple[int, ...] = (1, 0, 0, 0, 0, 0, 0, 0),
    pending_target: int = 0,
) -> Sequence:
    assert len(committed_tokens) == _SP_SIZE
    dispatched_tokens = list(committed_tokens)
    dispatched_tokens[pending_target] += 1
    total_tokens = sum(dispatched_tokens)
    token_base = 10_000 * (seq_idx + 1)

    seq = Sequence(
        list(range(token_base, token_base + total_tokens)),
        1.0,
        64,
        False,
    )
    seq.active("ls-planner-test", _SP_SIZE, 1)
    seq.status = SequenceStatus.RUNNING
    seq.num_prompt_tokens = total_tokens - 1

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.master_sp_idx = pending_target
    ctx.num_dispatched_tokens = dispatched_tokens
    ctx.pending_token_present = True
    ctx.pending_token_target_sp = pending_target
    return seq


@pytest.mark.parametrize(
    ("batch_size", "expected_chunks"),
    [
        (0, []),
        (1, [1]),
        (_T_COMPUTE, [_T_COMPUTE]),
        (_T_COMPUTE + 1, [_T_COMPUTE, 1]),
        (2 * _T_COMPUTE, [_T_COMPUTE, _T_COMPUTE]),
        (2 * _T_COMPUTE + 1, [_T_COMPUTE, _T_COMPUTE, 1]),
    ],
)
def test_source_greedy_threshold_boundaries(
    batch_size: int, expected_chunks: list[int]
) -> None:
    manager = _make_manager()
    requests = [_make_pending_sequence(idx) for idx in range(batch_size)]

    plan = manager.plan_iteration_masters_source_greedy(
        requests,
        [0],
        list(range(1, _SP_SIZE)),
        _T_COMPUTE,
        True,
    )

    assert plan.success, plan.failure_reason
    assert list(plan.master_batch_sizes) == expected_chunks
    assert list(plan.master_ranks) == list(range(len(expected_chunks)))
    assert sum(plan.master_batch_sizes) == batch_size
    assert list(plan.sequence_master_ranks) == [
        rank
        for rank, chunk_size in zip(plan.master_ranks, expected_chunks, strict=True)
        for _ in range(chunk_size)
    ]
    assert list(plan.new_allocation_ranks) == list(range(1, len(expected_chunks)))
    assert plan.scale_reason == ("compute" if batch_size > _T_COMPUTE else "none")

    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_pack_full_first_prefers_rank_with_more_group_kv() -> None:
    manager = _make_manager()
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=(2, 0, 4, 0, 0, 0, 0, 0),
        )
        for idx in range(3)
    ]
    dispatched_before = [
        list(seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens)
        for seq in requests
    ]

    # Deliberately put rank 2 last in the allocation. Its 12 committed group
    # tokens must still make it the first (and only) master for this small batch.
    plan = manager.plan_iteration_masters_source_greedy(
        requests,
        [1, 0, 2],
        [],
        64,
        True,
    )

    assert plan.success, plan.failure_reason
    assert list(plan.group_used_kv_tokens[:3]) == [6, 0, 12]
    assert list(plan.master_ranks) == [2]
    assert list(plan.master_batch_sizes) == [3]
    assert list(plan.sequence_master_ranks) == [2, 2, 2]
    assert [
        list(seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens)
        for seq in requests
    ] == dispatched_before

    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_append_pressure_adds_extra_rank_and_reports_memory_scale_up() -> None:
    manager = _make_manager(num_kvcache_blocks=4)
    # One block per rank is permanently occupied by SPStateManager's dummy
    # sequence, so three more blocks exhaust this four-block manager.
    filler = Sequence(list(range(3 * _BLOCK_SIZE)))
    filler.active("ls-planner-test", _SP_SIZE, 1)
    filler.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [
        3 * _BLOCK_SIZE,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    manager.block_manager[0].allocate_uncached(filler)
    request = _make_pending_sequence(
        0,
        committed_tokens=(_BLOCK_SIZE - 1, 0, 0, 0, 0, 0, 0, 0),
    )

    plan = manager.plan_iteration_masters_source_greedy(
        [request], [0], [1], 64, True
    )

    assert plan.success, plan.failure_reason
    assert list(plan.master_ranks) == [1]
    assert list(plan.new_allocation_ranks) == [1]
    assert plan.scale_reason == "memory"

    disabled = manager.plan_iteration_masters_source_greedy(
        [request], [0], [1], 64, False
    )
    assert disabled.success is False
    assert "append capacity" in disabled.failure_reason


def test_duplicate_request_is_rejected_before_commit_mutates_frontier() -> None:
    manager = _make_manager()
    first = _make_pending_sequence(0)
    second = _make_pending_sequence(1)
    plan = manager.plan_iteration_masters_source_greedy(
        [first, second], [0], list(range(1, _SP_SIZE)), 64, True
    )
    assert plan.success, plan.failure_reason

    ctx = first.block_ctx(BlockContextSlot.ACTIVE)
    before = (
        list(ctx.num_dispatched_tokens),
        ctx.pending_token_present,
        ctx.pending_token_target_sp,
        ctx.master_sp_idx,
    )
    valid, error = manager.validate_iteration_master_plan([first, first], plan)

    assert valid is False
    assert "duplicate" in error
    assert manager.commit_iteration_master_plan([first, first], plan) is False
    assert (
        list(ctx.num_dispatched_tokens),
        ctx.pending_token_present,
        ctx.pending_token_target_sp,
        ctx.master_sp_idx,
    ) == before

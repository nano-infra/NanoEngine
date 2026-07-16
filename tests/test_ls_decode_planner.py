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
    *,
    max_num_seqs: int = 64,
    max_num_recv_seqs: int | None = None,
    num_kvcache_blocks: int = 128,
    reserved_blocks_per_req: float = 0.0,
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
        max_num_recv_seqs=(
            max_num_seqs if max_num_recv_seqs is None else max_num_recv_seqs
        ),
        reserved_blocks_per_req=reserved_blocks_per_req,
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


def _remote_receiver_counts(
    requests: list[Sequence], sequence_master_ranks: list[int]
) -> list[int]:
    counts = [0] * _SP_SIZE
    for seq, master in zip(requests, sequence_master_ranks, strict=True):
        for owner in range(_SP_SIZE):
            if (
                owner != master
                and seq.committed_context_len(BlockContextSlot.ACTIVE, owner) > 0
            ):
                counts[owner] += 1
    return counts


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


def test_receiver_aware_fallback_repairs_source_prefix_false_negative() -> None:
    manager = _make_manager(max_num_seqs=32, max_num_recv_seqs=4)
    committed_prefixes = [
        (0, 15, 0, 0),
        (0, 15, 0, 10),
        (0, 0, 0, 10),
        (30, 0, 0, 0),
        (30, 15, 0, 0),
        (0, 0, 30, 0),
        (0, 0, 30, 0),
        (0, 0, 30, 0),
        (30, 15, 0, 0),
        (0, 0, 30, 0),
        (0, 0, 30, 0),
    ]
    previous_masters = [1, 3, 3, 0, 0, 2, 2, 2, 1, 2, 2]
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=prefix + (0,) * (_SP_SIZE - len(prefix)),
            pending_target=previous_master,
        )
        for idx, (prefix, previous_master) in enumerate(
            zip(committed_prefixes, previous_masters, strict=True)
        )
    ]

    # The KV totals force candidate order [2, 0, 1, 3]. In admission order,
    # the old contiguous-prefix greedy exhausts receiver capacity and rejects
    # this batch. A receiver-aware fallback can preserve/recover the owner-local
    # assignments and fit all eleven requests under the same hard limit.
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2, 3], [4, 5, 6, 7], _T_COMPUTE, True
    )

    assert list(plan.group_used_kv_tokens[:4]) == [90, 60, 150, 20]
    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy == "owner_bucket_repair"
    assert list(plan.allocation) == [0, 1, 2, 3]
    assert list(plan.new_allocation_ranks) == []
    assert sum(plan.master_batch_sizes) == len(requests)
    assert len(plan.sequence_master_ranks) == len(requests)
    assert set(plan.sequence_master_ranks) <= {0, 1, 2, 3}
    receiver_counts = _remote_receiver_counts(
        requests, list(plan.sequence_master_ranks)
    )
    assert max(receiver_counts) <= 4

    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_receiver_aware_fallback_rejects_proven_infeasible_batch() -> None:
    manager = _make_manager(max_num_seqs=16, max_num_recv_seqs=1)
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=(1, 1, 1, 0, 0, 0, 0, 0),
            pending_target=idx,
        )
        for idx in range(2)
    ]

    # Each owner needs one of the two requests to remain local:
    # owner_count=2 and receiver_limit=1 imply three required local
    # assignments in total, but a request can be local to only one rank.
    assert all(
        manager.estimate_pending_append_capacity(rank, requests, requests)
        >= len(requests)
        for rank in (0, 1, 2)
    )
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2], [], 1, True
    )

    assert plan.success is False
    assert "receiver capacity proven infeasible" in plan.failure_reason
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid is False
    assert error


def test_receiver_quota_matching_handles_owner_bucket_hall_repair() -> None:
    manager = _make_manager(max_num_seqs=3, max_num_recv_seqs=2)
    owner_masks = [0b001, 0b011, 0b001, 0b001, 0b001]
    previous_masters = [2, 2, 0, 1, 2]
    requests = []
    for idx, (owner_mask, previous_master) in enumerate(
        zip(owner_masks, previous_masters, strict=True)
    ):
        committed = tuple(
            1 if owner_mask & (1 << rank) else 0 for rank in range(_SP_SIZE)
        )
        requests.append(
            _make_pending_sequence(
                idx,
                committed_tokens=committed,
                pending_target=previous_master,
            )
        )

    # Both the admission-order prefix and current-master buckets hit a local
    # ordering dead end, while the sticky plan overloads receiver 0. The exact
    # owner-quota matching must find the legal 3/2 split instead of preempting.
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2], [], 2, True
    )

    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy == "receiver_append_flow"
    assert sorted(plan.master_batch_sizes) == [2, 3]
    assert max(
        _remote_receiver_counts(requests, list(plan.sequence_master_ranks))
    ) <= 2
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_receiver_quota_matching_rejects_nontrivial_hall_violation() -> None:
    manager = _make_manager(max_num_seqs=3, max_num_recv_seqs=1)
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=(1, 1, 1, 0, 0, 0, 0, 0),
            pending_target=idx,
        )
        for idx in range(2)
    ]
    requests.append(
        _make_pending_sequence(
            2,
            committed_tokens=(0,) * _SP_SIZE,
            pending_target=0,
        )
    )

    # Three ranks each require one local-owner assignment, but all three quota
    # sets share only the first two requests. The max-flow fallback must expose
    # this Hall violation instead of looping or manufacturing a legal plan.
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2], [], 1, True
    )

    assert plan.success is False
    assert "owner-local quota matching failed" in plan.failure_reason


def test_receiver_quota_activates_required_owner_from_extra_ranks() -> None:
    manager = _make_manager(max_num_seqs=4, max_num_recv_seqs=1)
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=(0, 1, 0, 0, 0, 0, 0, 0),
            pending_target=0,
        )
        for idx in range(2)
    ]

    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0], [1], 64, True
    )

    assert plan.success, plan.failure_reason
    assert list(plan.new_allocation_ranks) == [1]
    assert list(plan.allocation) == [0, 1]
    assert "receiver" in plan.scale_reason
    assert 1 in plan.sequence_master_ranks
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_receiver_fallback_separates_decode_metadata_exhaustion() -> None:
    manager = _make_manager(max_num_seqs=2, max_num_recv_seqs=8)
    requests = [_make_pending_sequence(idx) for idx in range(3)]

    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0], [], 64, True
    )

    assert plan.success is False
    assert "decode metadata capacity" in plan.failure_reason


def test_receiver_append_flow_selects_complementary_zero_cost_subsets() -> None:
    manager = _make_manager(
        max_num_seqs=4,
        max_num_recv_seqs=2,
        num_kvcache_blocks=6,
    )
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=committed + (0,) * (_SP_SIZE - 2),
            pending_target=0,
        )
        for idx, committed in enumerate(
            [(255, 1), (255, 1), (1, 255), (1, 255)]
        )
    ]
    for seq in requests:
        manager.block_manager[0].allocate_uncached(seq)
        manager.block_manager[1].allocate_uncached(seq)
    assert manager.block_manager[0].num_free_blocks == 1
    assert manager.block_manager[1].num_free_blocks == 1

    # Each rank has room for only one cost-1 append. A worst-subset cardinality
    # bound would incorrectly reject the batch, while the exact flow can send
    # each rank the two requests whose append cost there is zero.
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1], [], 2, True
    )

    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy == "receiver_append_flow"
    assert list(plan.master_batch_sizes) == [2, 2]
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_receiver_append_flow_repairs_infeasible_nominal_load_vector() -> None:
    manager = _make_manager(
        max_num_seqs=4,
        max_num_recv_seqs=8,
        num_kvcache_blocks=6,
    )
    # B/C/D cost one append block on every candidate. A is the sole zero-cost
    # request on ranks 0 and 1, and still costs one block on rank 2. With one
    # free block per rank, the source-style nominal load [2, 2, 0] asks both
    # ranks 0 and 1 to consume A and is impossible. The neighboring load
    # [2, 1, 1] is feasible, so a fixed-load fallback must not preempt it.
    committed = [
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (253, 253, 0),
    ]
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=owners + (0,) * (_SP_SIZE - 3),
            pending_target=0,
        )
        for idx, owners in enumerate(committed)
    ]
    for seq in requests:
        for rank in range(3):
            manager.block_manager[rank].allocate_uncached(seq)
    rank2_filler = Sequence(list(range(_BLOCK_SIZE)))
    rank2_filler.active("ls-planner-test", _SP_SIZE, 1)
    rank2_filler.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [
        0,
        0,
        _BLOCK_SIZE,
        0,
        0,
        0,
        0,
        0,
    ]
    manager.block_manager[2].allocate_uncached(rank2_filler)
    assert [manager.block_manager[rank].num_free_blocks for rank in range(3)] == [
        1,
        1,
        1,
    ]

    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2], [], 2, True
    )

    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy == "receiver_append_flow_load_repair"
    assert sorted(plan.master_batch_sizes) == [1, 1, 2]
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_variable_load_flow_escapes_distant_nominal_load_basin() -> None:
    owner_masks = [
        15,
        11,
        7,
        13,
        12,
        10,
        12,
        5,
        4,
        13,
        11,
        13,
        8,
        4,
        6,
        10,
        10,
        8,
        12,
        8,
        13,
        13,
        2,
        15,
        7,
        15,
        10,
    ]
    cheap_masks = [
        7,
        2,
        5,
        13,
        8,
        0,
        0,
        5,
        4,
        4,
        9,
        13,
        0,
        4,
        2,
        8,
        2,
        0,
        8,
        0,
        8,
        5,
        2,
        4,
        7,
        2,
        8,
    ]
    previous_masters = [
        1,
        3,
        2,
        0,
        3,
        3,
        2,
        0,
        2,
        2,
        3,
        0,
        3,
        2,
        2,
        3,
        1,
        3,
        2,
        3,
        3,
        0,
        1,
        1,
        2,
        0,
        3,
    ]
    manager = _make_manager(
        max_num_seqs=len(owner_masks),
        max_num_recv_seqs=22,
        num_kvcache_blocks=130,
    )
    kv_multipliers = [8, 6, 4, 2]
    requests = []
    for idx, (owner_mask, cheap_mask, previous_master) in enumerate(
        zip(owner_masks, cheap_masks, previous_masters, strict=True)
    ):
        committed = []
        for rank, multiplier in enumerate(kv_multipliers):
            if cheap_mask & (1 << rank):
                committed.append(multiplier * _BLOCK_SIZE + 1)
            elif owner_mask & (1 << rank):
                committed.append(multiplier * _BLOCK_SIZE + _BLOCK_SIZE - 1)
            else:
                committed.append(0)
        requests.append(
            _make_pending_sequence(
                idx,
                committed_tokens=tuple(committed) + (0,) * (_SP_SIZE - 4),
                pending_target=previous_master,
            )
        )

    for seq in requests:
        for rank in range(4):
            manager.block_manager[rank].allocate_uncached(seq)
    target_free = [0, 1, 1, 3]
    for rank, target in enumerate(target_free):
        fill_blocks = manager.block_manager[rank].num_free_blocks - target
        assert fill_blocks >= 0
        if fill_blocks == 0:
            continue
        filler = Sequence(list(range(fill_blocks * _BLOCK_SIZE)))
        filler.active("ls-planner-test", _SP_SIZE, 1)
        dispatched = [0] * _SP_SIZE
        dispatched[rank] = fill_blocks * _BLOCK_SIZE
        filler.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = dispatched
        manager.block_manager[rank].allocate_uncached(filler)
    assert [manager.block_manager[rank].num_free_blocks for rank in range(4)] == target_free

    # The old bounded unit-transfer search exhausted 128 states from nominal
    # load [8, 8, 11, 0], despite a legal distant load [0, 6, 10, 11]. The
    # variable-load branch-and-flow must remain complete instead of preempting
    # when a heuristic search budget is exhausted.
    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1, 2, 3], [], 9, True
    )

    assert list(plan.group_used_kv_tokens[:4]) == [27907, 21505, 18949, 13308]
    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy == "receiver_append_flow_load_repair"
    assert sum(plan.master_batch_sizes) == len(requests)
    assert max(
        _remote_receiver_counts(requests, list(plan.sequence_master_ranks))
    ) <= 22
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error


def test_receiver_append_flow_with_reserved_headroom_commits() -> None:
    manager = _make_manager(
        max_num_seqs=4,
        max_num_recv_seqs=2,
        num_kvcache_blocks=8,
        reserved_blocks_per_req=1.0,
    )
    requests = [
        _make_pending_sequence(
            idx,
            committed_tokens=committed + (0,) * (_SP_SIZE - 2),
            pending_target=0,
        )
        for idx, committed in enumerate(
            [(255, 1), (255, 1), (1, 255), (1, 255)]
        )
    ]
    for seq in requests:
        manager.block_manager[0].allocate_uncached(seq)
        manager.block_manager[1].allocate_uncached(seq)
    assert manager.block_manager[0].num_free_blocks == 3
    assert manager.block_manager[1].num_free_blocks == 3

    plan = manager.plan_iteration_masters_source_greedy(
        requests, [0, 1], [], 2, True
    )

    assert plan.success, plan.failure_reason
    assert plan.assignment_strategy.startswith("receiver_append_flow")
    valid, error = manager.validate_iteration_master_plan(requests, plan)
    assert valid, error
    assert manager.commit_iteration_master_plan(requests, plan)
    assert [
        seq.block_ctx(BlockContextSlot.ACTIVE).pending_token_target_sp
        for seq in requests
    ] == list(plan.sequence_master_ranks)


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

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest
import torch

import nanodeploy.worker.kv_p2p as kv_p2p
from nanodeploy._cpp import (
    BlockContextSlot,
    ScheduleAction,
    Scheduler,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)
from nanodeploy.engine.kv_consolidation import (
    KVScaleDownAcknowledgementError,
    KVScaleDownPreDispatchAborted,
    KVScaleDownRejected,
    execute_planned_ls_kv_scale_down,
    execute_ls_kv_scale_down,
)
from nanodeploy.worker.kv_p2p import (
    KVCacheP2PMove,
    KVCacheP2PResult,
    KVCacheP2PTransport,
)


_SP_SIZE = 2
_BLOCK_SIZE = 4
_NUM_BLOCKS = 16
_MAX_NUM_SEQS = 16


class _PlanState(Enum):
    RESERVED = "RESERVED"
    DISPATCHED = "DISPATCHED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


def _make_scheduler(
    *,
    sp_size: int = _SP_SIZE,
    block_size: int = _BLOCK_SIZE,
    num_blocks: int = _NUM_BLOCKS,
    max_num_seqs: int = _MAX_NUM_SEQS,
    max_num_recv_seqs: int = _MAX_NUM_SEQS,
    initial_dop: int = 2,
    future_kv_admission: bool = True,
    consolidation_mode: str = "off",
    candidate_util: float = 0.50,
    target_high_watermark: float = 0.80,
    stable_steps: int = 32,
    cooldown_steps: int = 64,
    check_interval_steps: int = 8,
    max_source_blocks_per_event: int = 16,
    min_comp_bound: int = 1,
) -> Scheduler:
    return Scheduler(
        "scale-down-test",
        1,
        max_num_seqs,
        4096,
        max_num_recv_seqs,
        -1,
        1,
        sp_size,
        num_blocks,
        block_size,
        "decode",
        0.0,
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
        64,
        True,
        "centralized",
        consolidation_mode,
        candidate_util,
        target_high_watermark,
        stable_steps,
        cooldown_steps,
        check_interval_steps,
        max_source_blocks_per_event,
        future_kv_admission,
        10,
        1000,
        0,
        min_comp_bound,
    )


def _postprocess_decode(
    scheduler: Scheduler,
    result,
    *,
    tokens_per_sequence: int = 1,
) -> None:
    token_ids = [
        [
            [99 + offset for offset in range(tokens_per_sequence)]
            for _ in sequences
        ]
        for sequences in result.filtered_dp_sp_seqs
    ]
    scheduler.postprocess(
        result.filtered_dp_sp_seqs,
        token_ids,
        False,
        1.0,
        tokens_per_sequence,
    )


def _admit_scale_down_group(
    scheduler: Scheduler,
    *,
    source_committed_tokens: int = 1,
    include_unmoved: bool = False,
):
    """Build a two-rank group using only the source-aligned public lifecycle.

    Packed admission places every five-token prompt on rank 0. Four live
    requests make the first Decode plan split 2 + 2 across the two allocated
    ranks. The short requests then finish in normal postprocess, leaving the
    target's bootstrap token as historical KV on rank 1. Finally, the C++
    pending-frontier helper moves only the next input back to rank 0, making
    rank 1 a legal low-KV evacuation source without a Python-created dummy.
    """
    assert source_committed_tokens > 0
    fillers = [Sequence(list(range(5)), 1.0, 2, True) for _ in range(3)]
    sequence = Sequence(list(range(5)), 1.0, 32, True)
    unmoved = Sequence(list(range(5)), 1.0, 32, True) if include_unmoved else None
    batch = (
        [fillers[0], unmoved, fillers[1], sequence]
        if unmoved is not None
        else [*fillers, sequence]
    )
    for request in batch:
        assert scheduler.add(request).accepted is True

    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert admission.is_prefill is False
    assert admission.dp_seqs == []
    assert admission.ls_real_decode_ids_by_dp == []
    records = admission.ls_admission_records
    assert [record.sequence.seq_id for record in records] == [
        request.seq_id for request in batch
    ]
    assert len({record.batch_id for record in records}) == 1
    assert all(record.planned_kv_dop == 2 for record in records)
    assert all(record.planned_kv_ranks == [0, 1] for record in records)
    assert all(not record.bootstrap_finished for record in records)
    assert [request.block_ctx().pending_token_target_sp for request in batch] == [
        0,
        1,
        0,
        1,
    ]
    assert all(
        [
            request.committed_context_len(BlockContextSlot.ACTIVE, rank)
            for rank in range(_SP_SIZE)
        ]
        == [5, 0]
        for request in batch
    )

    group_ids = {record.group_id_after_commit for record in records}
    assert len(group_ids) == 1
    group_id = group_ids.pop()
    assert group_id is not None
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]

    decode = scheduler.schedule()
    assert decode.action == ScheduleAction.DECODE
    assert decode.is_prefill is False
    assert decode.ls_iteration_sequence_ids == [[request.seq_id for request in batch]]
    assert decode.ls_iteration_master_assignments == [[0, 0, 1, 1]]

    worker = scheduler.worker_state[0]
    assert sequence.block_ctx().pending_token_target_sp == 1
    assert worker.may_append_on_sp(sequence, 1, source_committed_tokens)
    _postprocess_decode(
        scheduler,
        decode,
        tokens_per_sequence=source_committed_tokens,
    )

    for filler in fillers[: 2 if include_unmoved else 3]:
        assert filler.status == SequenceStatus.FINISHED
    assert sequence.status == SequenceStatus.RUNNING
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [5, source_committed_tokens]
    assert sequence.block_ctx().pending_token_target_sp == 1

    assert worker.reassign_pending_append(sequence, 0)
    assert sequence.block_ctx().master_sp_idx == 0
    assert sequence.block_ctx().pending_token_target_sp == 0
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [5, source_committed_tokens]
    if unmoved is not None:
        assert unmoved.status == SequenceStatus.RUNNING
        assert unmoved.committed_context_len(BlockContextSlot.ACTIVE, 1) == 0
        assert unmoved.block_ctx().pending_token_target_sp == 0
    return sequence, group_id, unmoved


def _admit_scale_down_sequence(
    scheduler: Scheduler,
    *,
    source_committed_tokens: int = 1,
):
    sequence, group_id, unmoved = _admit_scale_down_group(
        scheduler,
        source_committed_tokens=source_committed_tokens,
    )
    assert unmoved is None
    return sequence, group_id


def _context_snapshot(sequence: Sequence):
    context = sequence.block_ctx(BlockContextSlot.ACTIVE)
    return (
        context.dp_idx,
        context.master_sp_idx,
        context.pending_token_present,
        context.pending_token_target_sp,
        list(context.num_dispatched_tokens),
        [list(context.sp_block_table[rank]) for rank in range(_SP_SIZE)],
        list(context.block_location),
    )


def _free_blocks(scheduler: Scheduler) -> list[int]:
    return [
        scheduler.block_manager(0)[rank].num_free_blocks for rank in range(_SP_SIZE)
    ]


class _CompletedWork:
    def wait(self):
        return True


class _InProcessP2PExecutor:
    def __init__(self, monkeypatch, caches: list[torch.Tensor], on_copy=None):
        self.config = SimpleNamespace(
            attention_dp=1,
            attention_sp=_SP_SIZE,
            attention_tp=1,
            attn_world_size=_SP_SIZE,
            ls_kv_consolidation_migration_chunk_tokens=2,
        )
        self.caches = caches
        self.on_copy = on_copy
        self.moves: list[KVCacheP2PMove] = []
        self._rank = {"value": 0}
        self._payloads: list[torch.Tensor] = []

        monkeypatch.setattr(kv_p2p.dist, "get_rank", lambda group: self._rank["value"])
        monkeypatch.setattr(kv_p2p.dist, "get_world_size", lambda group: _SP_SIZE)
        monkeypatch.setattr(
            kv_p2p.dist,
            "get_global_rank",
            lambda group, group_rank: group_rank,
        )

        def isend(tensor, dst, group):
            self._payloads.append(tensor.clone())
            return _CompletedWork()

        def irecv(tensor, src, group):
            tensor.copy_(self._payloads.pop(0))
            return _CompletedWork()

        monkeypatch.setattr(kv_p2p.dist, "isend", isend)
        monkeypatch.setattr(kv_p2p.dist, "irecv", irecv)

    def copy_kv_ranges_p2p(self, moves, timeout=None):
        del timeout
        self.moves = list(moves)
        if self.on_copy is not None:
            self.on_copy()
        source_rank = self.moves[0].src_sp_rank
        destinations = sorted({move.dst_sp_rank for move in self.moves})
        execution_order = [source_rank, *destinations]
        results = []
        for rank in execution_order:
            self._rank["value"] = rank
            transport = KVCacheP2PTransport(
                self.caches[rank], group="scale-down-test", chunk_tokens=2
            )
            results.append(transport.execute(self.moves, current_dp_idx=0))
        assert self._payloads == []
        return results


class _FailAfterPhysicalCopyExecutor:
    def __init__(self, delegate: _InProcessP2PExecutor):
        self.delegate = delegate

    def copy_kv_ranges_p2p(self, moves, timeout=None):
        self.delegate.copy_kv_ranges_p2p(moves, timeout=timeout)
        raise RuntimeError("injected worker completion failure")


def _make_caches() -> list[torch.Tensor]:
    shape = (2, 2, _NUM_BLOCKS, _BLOCK_SIZE, 1, 2)
    caches = []
    values_per_rank = torch.tensor(shape).prod().item()
    for rank in range(_SP_SIZE):
        cache = torch.arange(values_per_rank, dtype=torch.float32).reshape(shape)
        caches.append(cache + rank * values_per_rank)
    return caches


def test_scale_down_runs_p2p_then_commits_and_schedules_without_source(
    monkeypatch,
):
    scheduler = _make_scheduler()
    sequence, group_id = _admit_scale_down_sequence(
        scheduler,
        source_committed_tokens=4,
    )
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [5, 4]

    source_blocks = list(sequence.block_table(BlockContextSlot.ACTIVE, 1))
    source_context_before = _context_snapshot(sequence)
    free_before = _free_blocks(scheduler)
    epoch_before = scheduler.get_ls_pool_resource_epochs()
    caches = _make_caches()
    cache_before = [cache.clone() for cache in caches]

    def assert_copy_precedes_metadata_commit():
        assert _context_snapshot(sequence) == source_context_before
        assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
        assert _free_blocks(scheduler)[0] < free_before[0]
        assert all(
            block_id not in scheduler.block_manager(0)[1].free_block_ids
            for block_id in source_blocks
        )

    executor = _InProcessP2PExecutor(
        monkeypatch, caches, on_copy=assert_copy_precedes_metadata_commit
    )

    result = execute_ls_kv_scale_down(
        scheduler,
        executor,
        group_id=group_id,
        source_rank=1,
    )

    assert result.source_rank == 1
    assert result.retained_ranks == (0,)
    assert result.num_tokens == 4
    assert result.num_moves == len(executor.moves)
    assert {worker.role for worker in result.worker_results} == {
        "source",
        "destination",
    }
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0]
    assert scheduler.get_ls_pool_resource_epochs() == [epoch_before[0] + 1]
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [9, 0]
    assert list(sequence.block_table(BlockContextSlot.ACTIVE, 1)) == []
    assert _free_blocks(scheduler)[1] == free_before[1] + len(source_blocks)
    assert all(
        block_id in scheduler.block_manager(0)[1].free_block_ids
        for block_id in source_blocks
    )

    # The transaction must use the backend's physical ranges, preserve source
    # bytes, and copy every KV component/layer into the staged destination.
    assert torch.equal(caches[1], cache_before[1])
    for move in executor.moves:
        source = cache_before[move.src_sp_rank][
            :,
            :,
            move.src_block_id,
            move.src_token_offset : move.src_token_offset + move.num_tokens,
            :,
            :,
        ]
        destination = caches[move.dst_sp_rank][
            :,
            :,
            move.dst_block_id,
            move.dst_token_offset : move.dst_token_offset + move.num_tokens,
            :,
            :,
        ]
        assert torch.equal(destination, source)

    # This is the scale-down assertion missing from the backend-only tests:
    # the following Decode plan keeps rank 1 out of the group and Attention no
    # longer needs SP exchange for the real sequence.
    decode = scheduler.schedule()
    assert decode.is_prefill is False
    assert decode.ls_group_rank_allocations == [[0]]
    assert decode.ls_master_ranks == [[0]]
    assert decode.ls_kv_dops == [1]
    metadata = [
        prepare_decode_cpp([sequence], rank, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS)
        for rank in range(_SP_SIZE)
    ]
    assert all(meta.use_sp_a2a is False for meta in metadata)
    assert _context_snapshot(sequence) != source_context_before


def test_automatic_scale_down_waits_for_stability_and_executes_reserved_plan(
    monkeypatch,
):
    scheduler = _make_scheduler(
        consolidation_mode="execute",
        stable_steps=2,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    sequence, group_id = _admit_scale_down_sequence(scheduler)

    stable = scheduler.schedule()
    assert stable.action == ScheduleAction.DECODE
    assert stable.ls_kv_consolidation_candidate is True
    assert stable.ls_kv_consolidation_stable_steps == 1
    assert stable.ls_kv_consolidation_decision_reason == "stable_window"

    maintenance = scheduler.schedule()
    assert maintenance.action == ScheduleAction.KV_CONSOLIDATION
    assert maintenance.dp_seqs == []
    assert maintenance.kv_consolidation_plan.success is True
    assert maintenance.kv_consolidation_plan.state.name == "RESERVED"
    assert maintenance.ls_kv_consolidation_stable_steps == 2
    assert maintenance.ls_kv_consolidation_decision_reason == "execute"
    assert maintenance.ls_pool_resource_epoch_before == maintenance.ls_pool_resource_epoch_after
    epoch_before_commit = scheduler.get_ls_pool_resource_epochs()
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.schedule()

    executor = _InProcessP2PExecutor(monkeypatch, _make_caches())
    result = execute_planned_ls_kv_scale_down(
        scheduler,
        executor,
        maintenance.kv_consolidation_plan,
    )
    assert result.group_id == group_id
    assert result.source_rank == 1
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0]
    assert maintenance.kv_consolidation_plan.state.name == "COMMITTED"
    assert scheduler.get_ls_pool_resource_epochs() == [epoch_before_commit[0] + 1]


def test_committed_admission_precedes_low_kv_and_never_creates_pending_batch():
    scheduler = _make_scheduler(
        consolidation_mode="execute",
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    running, _ = _admit_scale_down_sequence(scheduler)
    one_token = Sequence([7], 1.0, 1, True)
    assert scheduler.add(one_token).accepted is True
    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert result.is_prefill is False
    assert result.kv_consolidation_plan is None
    assert result.ls_kv_consolidation_candidate is False
    assert [record.sequence.seq_id for record in result.ls_admission_records] == [
        one_token.seq_id
    ]
    assert result.ls_admission_records[0].bootstrap_finished is True
    assert result.ls_real_decode_ids_by_dp == [[running.seq_id]]
    assert scheduler.get_ls_pending_batch_ids() == []
    assert scheduler.get_ls_pending_batch_sequence_ids() == []


def test_automatic_candidate_util_threshold_is_strict():
    scheduler = _make_scheduler(
        consolidation_mode="execute",
        # The fixture owns two committed blocks on rank 0 and one on rank 1.
        # Source-aligned utilization uses the full allocator capacity.
        candidate_util=3 / (_SP_SIZE * _NUM_BLOCKS),
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_scale_down_sequence(scheduler)

    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is False
    assert result.ls_kv_consolidation_decision_reason == "no_candidate"
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_automatic_scale_down_honors_scale_up_cooldown():
    scheduler = _make_scheduler(
        consolidation_mode="execute",
        stable_steps=1,
        cooldown_steps=64,
        check_interval_steps=1,
    )
    _admit_scale_down_sequence(scheduler)

    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "cooldown"


def test_automatic_high_watermark_rejection_releases_reservations():
    control = _make_scheduler()
    _admit_scale_down_sequence(control)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="execute",
        target_high_watermark=0.05,
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_scale_down_sequence(scheduler)
    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "target_high_watermark"
    assert _free_blocks(scheduler) == expected_free_blocks
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_automatic_source_block_budget_releases_reservations():
    control = _make_scheduler()
    _admit_scale_down_sequence(control, source_committed_tokens=5)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="execute",
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
        max_source_blocks_per_event=1,
    )
    sequence, _ = _admit_scale_down_sequence(
        scheduler,
        source_committed_tokens=5,
    )
    assert len(sequence.block_table(BlockContextSlot.ACTIVE, 1)) == 2

    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "source_block_budget"
    assert _free_blocks(scheduler) == expected_free_blocks


def test_shadow_candidate_does_not_reserve_or_change_group_allocation():
    control = _make_scheduler()
    _admit_scale_down_sequence(control)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="shadow",
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_scale_down_sequence(scheduler)
    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "shadow_candidate"
    assert result.kv_consolidation_plan is None
    assert _free_blocks(scheduler) == expected_free_blocks
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_consolidation_feature_off_keeps_decode_and_never_reserves_plan():
    scheduler = _make_scheduler(consolidation_mode="off")
    sequence, group_id = _admit_scale_down_sequence(scheduler)
    free_before = _free_blocks(scheduler)

    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert result.is_prefill is False
    assert result.kv_consolidation_plan is None
    assert result.ls_kv_consolidation_candidate is False
    assert result.ls_kv_consolidation_decision_reason == "off"
    assert result.ls_real_decode_ids_by_dp == [[sequence.seq_id]]
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
    assert _free_blocks(scheduler) == free_before


def test_scale_down_worker_failure_is_dispatched_and_fail_closed(
    monkeypatch,
):
    scheduler = _make_scheduler()
    sequence, group_id = _admit_scale_down_sequence(
        scheduler,
        source_committed_tokens=4,
    )
    context_before = _context_snapshot(sequence)
    free_before = _free_blocks(scheduler)
    caches = _make_caches()
    source_cache_before = caches[1].clone()
    physical_executor = _InProcessP2PExecutor(monkeypatch, caches)

    with pytest.raises(RuntimeError, match="injected worker completion failure"):
        execute_ls_kv_scale_down(
            scheduler,
            _FailAfterPhysicalCopyExecutor(physical_executor),
            group_id=group_id,
            source_rank=1,
        )

    # Destination may contain copied bytes and the old source placement remains
    # visible, but completion is ambiguous after dispatch. Reservations and the
    # transaction guard must stay intact until process restart.
    assert _context_snapshot(sequence) == context_before
    assert _free_blocks(scheduler)[0] < free_before[0]
    assert _free_blocks(scheduler)[1] == free_before[1]
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
    assert torch.equal(caches[1], source_cache_before)

    with pytest.raises(RuntimeError, match="permanently fatal"):
        scheduler.schedule()


def test_automatic_copy_failure_keeps_reservation_fail_closed(monkeypatch):
    scheduler = _make_scheduler()
    _, group_id = _admit_scale_down_sequence(
        scheduler,
        source_committed_tokens=4,
    )
    plan = scheduler.plan_ls_kv_scale_down(group_id, 1)
    assert plan.success
    assert plan.state.name == "RESERVED"
    physical_executor = _InProcessP2PExecutor(monkeypatch, _make_caches())

    with pytest.raises(RuntimeError, match="injected worker completion failure"):
        execute_planned_ls_kv_scale_down(
            scheduler,
            _FailAfterPhysicalCopyExecutor(physical_executor),
            plan,
            abort_on_copy_error=False,
        )

    with pytest.raises(RuntimeError, match="permanently fatal"):
        scheduler.schedule()
    assert plan.state.name == "DISPATCHED"
    with pytest.raises(RuntimeError, match="cannot abort.*after worker dispatch"):
        scheduler.abort_ls_kv_scale_down(plan)


def test_scale_down_rejects_current_decode_master_without_calling_executor():
    scheduler = _make_scheduler()
    sequence, group_id = _admit_scale_down_sequence(scheduler)

    class _UnexpectedExecutor:
        def copy_kv_ranges_p2p(self, moves, timeout=None):
            pytest.fail("rejected scale-down invoked the physical backend")

    assert sequence.block_ctx().master_sp_idx == 0
    with pytest.raises(KVScaleDownRejected, match="active or pending Decode master"):
        execute_ls_kv_scale_down(
            scheduler,
            _UnexpectedExecutor(),
            group_id=group_id,
            source_rank=0,
        )


def test_reserved_scale_down_blocks_normal_decode_until_abort():
    scheduler = _make_scheduler()
    sequence, group_id = _admit_scale_down_sequence(
        scheduler,
        source_committed_tokens=4,
    )
    context_before = _context_snapshot(sequence)
    free_before = _free_blocks(scheduler)

    plan = scheduler.plan_ls_kv_scale_down(group_id, 1)
    assert plan.success, plan.failure_reason
    assert plan.state.name == "RESERVED"
    assert _context_snapshot(sequence) == context_before
    assert _free_blocks(scheduler)[0] < free_before[0]
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.schedule()
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.add(Sequence([7], 1.0, 4, True))
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.preempt(0, sequence)
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.postprocess([], [], False, 0.0, 1)
    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.free_to_be_migrated(sequence)

    scheduler.abort_ls_kv_scale_down(plan)
    assert plan.state.name == "ABORTED"
    assert _context_snapshot(sequence) == context_before
    assert _free_blocks(scheduler) == free_before
    resumed = scheduler.schedule()
    assert resumed.action == ScheduleAction.DECODE
    assert resumed.is_prefill is False


def test_scale_down_rejects_stale_change_on_unmoved_group_sequence(monkeypatch):
    scheduler = _make_scheduler()
    moved, group_id, unmoved = _admit_scale_down_group(
        scheduler,
        include_unmoved=True,
    )
    assert unmoved is not None
    worker = scheduler.worker_state[0]
    assert moved.committed_context_len(BlockContextSlot.ACTIVE, 1) > 0
    assert unmoved.committed_context_len(BlockContextSlot.ACTIVE, 1) == 0

    moved_before = _context_snapshot(moved)
    caches = _make_caches()

    def mutate_unmoved_sequence_after_plan():
        assert worker.reassign_pending_append(unmoved, 1)

    executor = _InProcessP2PExecutor(
        monkeypatch, caches, on_copy=mutate_unmoved_sequence_after_plan
    )
    with pytest.raises(RuntimeError, match="became stale before commit"):
        execute_ls_kv_scale_down(
            scheduler,
            executor,
            group_id=group_id,
            source_rank=1,
        )

    assert _context_snapshot(moved) == moved_before
    assert unmoved.block_ctx().pending_token_target_sp == 1
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_pre_dispatch_move_materialization_failure_aborts_exact_reservation():
    plan = SimpleNamespace(
        success=True,
        moves=None,
        state=_PlanState.RESERVED,
    )
    aborts = []

    class BrokenMoves:
        def __iter__(self):
            raise RuntimeError("injected move materialization failure")

    plan.moves = BrokenMoves()

    class SchedulerStub:
        def abort_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            assert plan.state == _PlanState.RESERVED
            plan.state = _PlanState.ABORTED
            aborts.append(reserved)

        def mark_ls_kv_scale_down_dispatched(self, _reserved):
            pytest.fail("pre-dispatch failure entered DISPATCHED")

    class ExecutorStub:
        def copy_kv_ranges_p2p(self, _moves, timeout=None):
            pytest.fail(f"pre-dispatch failure reached executor with timeout={timeout}")

    with pytest.raises(
        KVScaleDownPreDispatchAborted,
        match="move materialization failed before dispatch",
    ) as raised:
        execute_planned_ls_kv_scale_down(SchedulerStub(), ExecutorStub(), plan)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "injected move materialization failure" in str(raised.value.__cause__)
    assert aborts == [plan]
    assert plan.state == _PlanState.ABORTED


@pytest.mark.parametrize(
    "mismatch",
    [
        "empty",
        "dp",
        "source",
        "destination",
        "tokens",
        "negative-block",
        "source-overlap",
        "destination-overlap",
    ],
)
def test_materialized_move_mismatch_aborts_before_dispatch(mismatch):
    move = SimpleNamespace(
        dp_idx=0,
        src_sp_rank=1,
        dst_sp_rank=0,
        src_block_id=3,
        src_token_offset=0,
        dst_block_id=7,
        dst_token_offset=0,
        num_tokens=2,
    )
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=2,
        moves=[move],
    )
    if mismatch == "empty":
        plan.moves = []
    elif mismatch == "dp":
        move.dp_idx = 1
    elif mismatch == "source":
        move.src_sp_rank = 2
    elif mismatch == "destination":
        move.dst_sp_rank = 2
    elif mismatch == "tokens":
        plan.num_tokens = 3
    elif mismatch == "negative-block":
        move.src_block_id = -1
    elif mismatch == "source-overlap":
        plan.num_tokens = 4
        plan.moves.append(
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=1,
                dst_sp_rank=0,
                src_block_id=3,
                src_token_offset=1,
                dst_block_id=8,
                dst_token_offset=0,
                num_tokens=2,
            )
        )
    elif mismatch == "destination-overlap":
        plan.num_tokens = 4
        plan.moves.append(
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=1,
                dst_sp_rank=0,
                src_block_id=4,
                src_token_offset=0,
                dst_block_id=7,
                dst_token_offset=1,
                num_tokens=2,
            )
        )

    class SchedulerStub:
        def abort_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.ABORTED

        def mark_ls_kv_scale_down_dispatched(self, _reserved):
            pytest.fail("mismatched physical plan entered DISPATCHED")

    class ExecutorStub:
        def copy_kv_ranges_p2p(self, *_args, **_kwargs):
            pytest.fail("mismatched physical plan reached executor")

    with pytest.raises(KVScaleDownPreDispatchAborted) as raised:
        execute_planned_ls_kv_scale_down(SchedulerStub(), ExecutorStub(), plan)

    assert isinstance(raised.value.__cause__, ValueError)
    assert plan.state == _PlanState.ABORTED


def test_failed_pre_dispatch_abort_is_not_reported_as_safe_abort():
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        moves=[],
    )

    class SchedulerStub:
        def abort_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            raise RuntimeError("injected reservation abort failure")

        def mark_ls_kv_scale_down_dispatched(self, _reserved):
            pytest.fail("invalid plan entered DISPATCHED")

    with pytest.raises(RuntimeError, match="reservation abort failure") as raised:
        execute_planned_ls_kv_scale_down(SchedulerStub(), object(), plan)

    assert type(raised.value) is RuntimeError
    assert plan.state == _PlanState.RESERVED


@pytest.mark.parametrize("mark_result", [False, True])
def test_unconfirmed_dispatch_marker_safely_aborts_reserved_plan(mark_result):
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=2,
        moves=[
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=1,
                dst_sp_rank=0,
                src_block_id=3,
                src_token_offset=0,
                dst_block_id=7,
                dst_token_offset=0,
                num_tokens=2,
            )
        ],
    )
    aborts = []

    class SchedulerStub:
        def mark_ls_kv_scale_down_dispatched(self, reserved):
            assert reserved is plan
            return mark_result

        def abort_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            assert plan.state == _PlanState.RESERVED
            plan.state = _PlanState.ABORTED
            aborts.append(reserved)

    class ExecutorStub:
        def copy_kv_ranges_p2p(self, *_args, **_kwargs):
            pytest.fail("unconfirmed DISPATCHED state reached executor")

    with pytest.raises(KVScaleDownPreDispatchAborted):
        execute_planned_ls_kv_scale_down(SchedulerStub(), ExecutorStub(), plan)

    assert aborts == [plan]
    assert plan.state == _PlanState.ABORTED


def test_post_dispatch_commit_failure_latches_without_abort():
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=2,
        moves=[SimpleNamespace(
            dp_idx=0,
            src_sp_rank=1,
            dst_sp_rank=0,
            src_block_id=3,
            src_token_offset=0,
            dst_block_id=7,
            dst_token_offset=0,
            num_tokens=2,
        )],
    )
    latched = []

    class SchedulerStub:
        def mark_ls_kv_scale_down_dispatched(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.DISPATCHED
            return True

        def commit_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            return False

        def abort_ls_kv_scale_down(self, _reserved):
            pytest.fail("post-dispatch commit failure attempted abort")

        def latch_ls_fatal(self, code):
            latched.append(code)

    class ExecutorStub:
        config = SimpleNamespace(
            attention_dp=1,
            attention_sp=2,
            attention_tp=1,
            attn_world_size=2,
            ls_kv_consolidation_migration_chunk_tokens=2,
        )

        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert len(moves) == 1
            assert timeout is None
            return [
                KVCacheP2PResult(0, 0, "destination", 1, 1, 0, 16),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ]

    with pytest.raises(RuntimeError, match="dispatched.*failed to commit"):
        execute_planned_ls_kv_scale_down(SchedulerStub(), ExecutorStub(), plan)
    assert len(latched) == 1
    assert latched[0].name == "KV_CONSOLIDATION_FAILED"


def test_valid_worker_ack_covers_every_split_move_and_chunk_before_commit():
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        transaction_id=7,
        group_id=3,
        dp_idx=0,
        source_rank=2,
        retained_ranks=[0, 1],
        num_tokens=7,
        moves=[
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=2,
                dst_sp_rank=0,
                src_block_id=3,
                src_token_offset=0,
                dst_block_id=7,
                dst_token_offset=0,
                num_tokens=5,
            ),
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=2,
                dst_sp_rank=1,
                src_block_id=4,
                src_token_offset=0,
                dst_block_id=8,
                dst_token_offset=0,
                num_tokens=1,
            ),
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=2,
                dst_sp_rank=1,
                src_block_id=5,
                src_token_offset=0,
                dst_block_id=9,
                dst_token_offset=0,
                num_tokens=1,
            ),
        ],
    )
    commits = []

    class SchedulerStub:
        def mark_ls_kv_scale_down_dispatched(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.DISPATCHED
            return True

        def commit_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.COMMITTED
            commits.append(reserved)
            return True

        def abort_ls_kv_scale_down(self, _reserved):
            pytest.fail("valid dispatched transaction attempted abort")

        def latch_ls_fatal(self, _code):
            pytest.fail("valid acknowledgements latched fatal state")

    class ExecutorStub:
        config = SimpleNamespace(
            attention_dp=1,
            attention_sp=3,
            attention_tp=1,
            attn_world_size=3,
            ls_kv_consolidation_migration_chunk_tokens=2,
        )

        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert len(moves) == 3
            assert timeout is None
            return [
                KVCacheP2PResult(0, 0, "destination", 3, 3, 0, 40),
                KVCacheP2PResult(0, 1, "destination", 2, 1, 0, 16),
                KVCacheP2PResult(0, 2, "source", 5, 4, 56, 0),
            ]

    result = execute_planned_ls_kv_scale_down(
        SchedulerStub(), ExecutorStub(), plan
    )

    assert plan.state == _PlanState.COMMITTED
    assert commits == [plan]
    assert result.num_moves == 3
    assert len(result.worker_results) == 3


def test_worker_ack_census_includes_idle_dp_sp_and_tp_workers():
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        transaction_id=7,
        group_id=3,
        dp_idx=1,
        source_rank=2,
        retained_ranks=[0, 1],
        num_tokens=2,
        moves=[
            SimpleNamespace(
                dp_idx=1,
                src_sp_rank=2,
                dst_sp_rank=0,
                src_block_id=3,
                src_token_offset=0,
                dst_block_id=7,
                dst_token_offset=0,
                num_tokens=2,
            )
        ],
    )

    class SchedulerStub:
        def mark_ls_kv_scale_down_dispatched(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.DISPATCHED
            return True

        def commit_ls_kv_scale_down(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.COMMITTED
            return True

        def abort_ls_kv_scale_down(self, _reserved):
            pytest.fail("valid full worker census attempted abort")

        def latch_ls_fatal(self, _code):
            pytest.fail("valid full worker census latched fatal state")

    class ExecutorStub:
        config = SimpleNamespace(
            attention_dp=2,
            attention_sp=3,
            attention_tp=2,
            attn_world_size=12,
            ls_kv_consolidation_migration_chunk_tokens=2,
        )
        workers = [object() for _ in range(12)]

        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert len(moves) == 1
            assert timeout is None
            results = []
            for dp_idx in range(2):
                for sp_rank in range(3):
                    for _tp_rank in range(2):
                        if dp_idx == 1 and sp_rank == 2:
                            result = KVCacheP2PResult(
                                dp_idx, sp_rank, "source", 1, 1, 16, 0
                            )
                        elif dp_idx == 1 and sp_rank == 0:
                            result = KVCacheP2PResult(
                                dp_idx, sp_rank, "destination", 1, 1, 0, 16
                            )
                        else:
                            result = KVCacheP2PResult(
                                dp_idx, sp_rank, "idle", 0, 0, 0, 0
                            )
                        results.append(result)
            return results

    result = execute_planned_ls_kv_scale_down(
        SchedulerStub(), ExecutorStub(), plan
    )

    assert plan.state == _PlanState.COMMITTED
    assert len(result.worker_results) == 12
    assert sum(item.role == "idle" for item in result.worker_results) == 8


@pytest.mark.parametrize(
    "worker_results",
    [
        pytest.param(
            [KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0)],
            id="missing-destination-worker",
        ),
        pytest.param(
            [
                KVCacheP2PResult(0, 0, "idle", 0, 0, 0, 0),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ],
            id="wrong-role",
        ),
        pytest.param(
            [
                KVCacheP2PResult(0, 0, "destination", 1, 0, 0, 16),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ],
            id="incomplete-chunk-coverage",
        ),
        pytest.param(
            [
                KVCacheP2PResult(0, 0, "destination", 1, 1, 0, 8),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ],
            id="unbalanced-bytes",
        ),
        pytest.param(
            [
                KVCacheP2PResult(0, 0, "destination", -1, 1, 0, 16),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ],
            id="negative-dynamic-counter",
        ),
        pytest.param(
            [
                KVCacheP2PResult(1, 0, "destination", 1, 1, 0, 16),
                KVCacheP2PResult(0, 1, "source", 1, 1, 16, 0),
            ],
            id="wrong-dp-census",
        ),
    ],
)
def test_malformed_worker_ack_is_fatal_before_commit_without_abort(worker_results):
    plan = SimpleNamespace(
        success=True,
        state=_PlanState.RESERVED,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=2,
        moves=[SimpleNamespace(
            dp_idx=0,
            src_sp_rank=1,
            dst_sp_rank=0,
            src_block_id=3,
            src_token_offset=0,
            dst_block_id=7,
            dst_token_offset=0,
            num_tokens=2,
        )],
    )
    latched = []

    class SchedulerStub:
        def mark_ls_kv_scale_down_dispatched(self, reserved):
            assert reserved is plan
            plan.state = _PlanState.DISPATCHED
            return True

        def commit_ls_kv_scale_down(self, _reserved):
            pytest.fail("malformed acknowledgement reached metadata commit")

        def abort_ls_kv_scale_down(self, _reserved):
            pytest.fail("post-dispatch acknowledgement failure attempted abort")

        def latch_ls_fatal(self, code):
            latched.append(code)

    class ExecutorStub:
        config = SimpleNamespace(
            attention_dp=1,
            attention_sp=2,
            attention_tp=1,
            attn_world_size=2,
            ls_kv_consolidation_migration_chunk_tokens=2,
        )

        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert len(moves) == 1
            assert timeout is None
            return worker_results

    with pytest.raises(KVScaleDownAcknowledgementError):
        execute_planned_ls_kv_scale_down(SchedulerStub(), ExecutorStub(), plan)

    assert plan.state == _PlanState.DISPATCHED
    assert len(latched) == 1
    assert latched[0].name == "KV_CONSOLIDATION_FAILED"

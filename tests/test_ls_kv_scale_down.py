from __future__ import annotations

import pytest
import torch

import nanodeploy.worker.kv_p2p as kv_p2p
from nanodeploy._cpp import (
    BlockContextSlot,
    ScheduleAction,
    Scheduler,
    Sequence,
    prepare_decode_cpp,
)
from nanodeploy.engine.kv_consolidation import (
    KVScaleDownRejected,
    execute_planned_ls_kv_scale_down,
    execute_ls_kv_scale_down,
)
from nanodeploy.worker.kv_p2p import KVCacheP2PMove, KVCacheP2PTransport


_SP_SIZE = 2
_BLOCK_SIZE = 4
_NUM_BLOCKS = 16
_MAX_NUM_SEQS = 16


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
    )


def _admit_striped_sequence(scheduler: Scheduler):
    # Five committed tokens force a partial destination tail (3 + 2). Moving
    # rank 1 into rank 0 therefore crosses both source/destination block
    # boundaries and overwrites the old, not-yet-computed pending slot.
    sequence = Sequence(list(range(5)), 1.0, 32, True)
    scheduler.add(sequence)
    admission = scheduler.schedule()
    assert admission.is_prefill is True
    assert admission.action == ScheduleAction.ADMISSION
    assert admission.ls_initial_kv_dops == [2]

    worker = scheduler.worker_state[0]
    assert worker.may_append(sequence, 1)
    sequence.append_token(99, BlockContextSlot.ACTIVE)
    sequence.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(sequence.block_ctx().master_sp_idx, 1)
    return sequence, admission.ls_initial_group_ids[0]


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
    sequence, group_id = _admit_striped_sequence(scheduler)
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [3, 2]

    source_blocks = list(sequence.block_table(BlockContextSlot.ACTIVE, 1))
    source_context_before = _context_snapshot(sequence)
    free_before = _free_blocks(scheduler)
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
    assert result.num_tokens == 2
    assert result.num_moves == len(executor.moves)
    assert {worker.role for worker in result.worker_results} == {
        "source",
        "destination",
    }
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0]
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(_SP_SIZE)
    ] == [5, 0]
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
        check_interval_steps=3,
    )
    sequence, group_id = _admit_striped_sequence(scheduler)

    stable = scheduler.schedule()
    assert stable.action == ScheduleAction.DECODE
    assert stable.ls_kv_consolidation_candidate is True
    assert stable.ls_kv_consolidation_stable_steps == 1
    assert stable.ls_kv_consolidation_decision_reason == "stable_window"

    maintenance = scheduler.schedule()
    assert maintenance.action == ScheduleAction.KV_CONSOLIDATION
    assert maintenance.dp_seqs == []
    assert maintenance.kv_consolidation_plan.success is True
    assert maintenance.ls_kv_consolidation_stable_steps == 2
    assert maintenance.ls_kv_consolidation_decision_reason == "execute"
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


def test_pending_no_fit_executes_only_consolidation_that_proves_next_admission():
    scheduler = _make_scheduler(
        sp_size=3,
        block_size=4,
        num_blocks=3,
        max_num_seqs=1,
        max_num_recv_seqs=1,
        initial_dop=3,
        future_kv_admission=True,
        consolidation_mode="execute",
        target_high_watermark=1.0,
        # Admission pressure deliberately bypasses the opportunistic windows.
        stable_steps=32,
        cooldown_steps=64,
        check_interval_steps=8,
    )
    running = Sequence(list(range(14)), 1.0, 2, True)
    scheduler.add(running)
    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION

    worker = scheduler.worker_state[0]
    assert worker.may_append(running, 1)
    running.append_token(99, BlockContextSlot.ACTIVE)
    running.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(running.block_ctx().master_sp_idx, 1)
    assert scheduler.schedule().action == ScheduleAction.DECODE

    waiting = Sequence([7], 1.0, 2, True)
    scheduler.add(waiting)
    maintenance = scheduler.schedule()

    assert maintenance.action == ScheduleAction.KV_CONSOLIDATION
    assert maintenance.ls_kv_consolidation_decision_reason == "execute_pressure"
    assert maintenance.kv_consolidation_plan.success is True
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[waiting.seq_id]]

    # Metadata-only commit is sufficient for this scheduler unit test; the
    # physical P2P transaction is covered by the executor tests above.
    assert scheduler.commit_ls_kv_scale_down(maintenance.kv_consolidation_plan)
    readmission = scheduler.schedule()
    assert readmission.action == ScheduleAction.ADMISSION
    assert readmission.ls_initial_sequence_ids == [[waiting.seq_id]]
    assert readmission.ls_initial_admission_kinds == ["merge"]
    assert scheduler.get_ls_pending_batch_ids() == []


def test_pending_no_fit_rejects_consolidation_without_admission_benefit():
    scheduler = _make_scheduler(
        sp_size=3,
        block_size=4,
        num_blocks=3,
        max_num_seqs=1,
        max_num_recv_seqs=1,
        initial_dop=3,
        future_kv_admission=False,
        consolidation_mode="execute",
        target_high_watermark=1.0,
        stable_steps=32,
        cooldown_steps=64,
        check_interval_steps=8,
    )
    running = Sequence(list(range(14)), 1.0, 32, True)
    scheduler.add(running)
    assert scheduler.schedule().action == ScheduleAction.ADMISSION

    worker = scheduler.worker_state[0]
    assert worker.may_append(running, 1)
    running.append_token(99, BlockContextSlot.ACTIVE)
    running.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(running.block_ctx().master_sp_idx, 1)
    assert scheduler.schedule().action == ScheduleAction.DECODE

    # This prompt fits an empty three-rank DP, but not the capacity left by the
    # running request. Evacuating one rank only moves KV and cannot create the
    # additional blocks needed by this admission.
    waiting = Sequence(list(range(20)), 1.0, 32, True)
    scheduler.add(waiting)
    free_before = _free_blocks(scheduler)
    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert result.kv_consolidation_plan is None
    assert result.ls_kv_consolidation_decision_reason == "pending_no_benefit"
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[waiting.seq_id]]
    assert _free_blocks(scheduler) == free_before


def test_pending_pressure_shadow_mode_proves_candidate_without_reserving_blocks():
    scheduler = _make_scheduler(
        sp_size=3,
        block_size=4,
        num_blocks=3,
        max_num_seqs=1,
        max_num_recv_seqs=1,
        initial_dop=3,
        future_kv_admission=False,
        consolidation_mode="shadow",
        target_high_watermark=1.0,
        stable_steps=32,
        cooldown_steps=64,
        check_interval_steps=8,
        # A shadow proof observes the actual move but does not execute it, so
        # the execute-only transfer budget must not hide this telemetry.
        max_source_blocks_per_event=0,
    )
    running = Sequence(list(range(14)), 1.0, 32, True)
    scheduler.add(running)
    assert scheduler.schedule().action == ScheduleAction.ADMISSION

    worker = scheduler.worker_state[0]
    assert worker.may_append(running, 1)
    running.append_token(99, BlockContextSlot.ACTIVE)
    running.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(running.block_ctx().master_sp_idx, 1)
    assert scheduler.schedule().action == ScheduleAction.DECODE

    waiting = Sequence([7], 1.0, 32, True)
    scheduler.add(waiting)
    free_before = _free_blocks(scheduler)
    result = scheduler.schedule()

    assert result.action == ScheduleAction.DECODE
    assert result.kv_consolidation_plan is None
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "shadow_pressure_candidate"
    assert scheduler.get_ls_pending_batch_sequence_ids() == [[waiting.seq_id]]
    assert _free_blocks(scheduler) == free_before


def test_automatic_candidate_util_threshold_is_strict():
    scheduler = _make_scheduler(
        consolidation_mode="execute",
        candidate_util=2 / (_SP_SIZE * (_NUM_BLOCKS - 1)),
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_striped_sequence(scheduler)

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
    _admit_striped_sequence(scheduler)

    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "cooldown"


def test_automatic_high_watermark_rejection_releases_reservations():
    control = _make_scheduler()
    _admit_striped_sequence(control)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="execute",
        target_high_watermark=0.05,
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_striped_sequence(scheduler)
    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "target_high_watermark"
    assert _free_blocks(scheduler) == expected_free_blocks
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_automatic_source_block_budget_releases_reservations():
    control = _make_scheduler()
    control_sequence = Sequence(list(range(13)), 1.0, 32, True)
    control.add(control_sequence)
    control.schedule()
    control_worker = control.worker_state[0]
    assert control_worker.may_append(control_sequence, 1)
    control_sequence.append_token(99, BlockContextSlot.ACTIVE)
    control_sequence.mark_last_token_pending(BlockContextSlot.ACTIVE)
    control_worker.add_running_tokens(control_sequence.block_ctx().master_sp_idx, 1)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="execute",
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
        max_source_blocks_per_event=1,
    )
    sequence = Sequence(list(range(13)), 1.0, 32, True)
    scheduler.add(sequence)
    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert len(sequence.block_table(BlockContextSlot.ACTIVE, 1)) == 2

    worker = scheduler.worker_state[0]
    assert worker.may_append(sequence, 1)
    sequence.append_token(99, BlockContextSlot.ACTIVE)
    sequence.mark_last_token_pending(BlockContextSlot.ACTIVE)
    worker.add_running_tokens(sequence.block_ctx().master_sp_idx, 1)

    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "source_block_budget"
    assert _free_blocks(scheduler) == expected_free_blocks


def test_shadow_candidate_does_not_reserve_or_change_group_allocation():
    control = _make_scheduler()
    _admit_striped_sequence(control)
    control.schedule()
    expected_free_blocks = _free_blocks(control)

    scheduler = _make_scheduler(
        consolidation_mode="shadow",
        stable_steps=1,
        cooldown_steps=0,
        check_interval_steps=1,
    )
    _, group_id = _admit_striped_sequence(scheduler)
    result = scheduler.schedule()
    assert result.action == ScheduleAction.DECODE
    assert result.ls_kv_consolidation_candidate is True
    assert result.ls_kv_consolidation_decision_reason == "shadow_candidate"
    assert result.kv_consolidation_plan is None
    assert _free_blocks(scheduler) == expected_free_blocks
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]


def test_scale_down_worker_failure_aborts_metadata_and_block_reservations(
    monkeypatch,
):
    scheduler = _make_scheduler()
    sequence, group_id = _admit_striped_sequence(scheduler)
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

    # Destination may contain unreachable bytes, but the old source placement
    # and KV remain authoritative and all reserved blocks are returned.
    assert _context_snapshot(sequence) == context_before
    assert _free_blocks(scheduler) == free_before
    assert scheduler.get_ls_group_allocated_ranks(group_id) == [0, 1]
    assert torch.equal(caches[1], source_cache_before)

    decode = scheduler.schedule()
    assert decode.is_prefill is False
    assert decode.ls_group_rank_allocations == [[0, 1]]


def test_automatic_copy_failure_keeps_reservation_fail_closed(monkeypatch):
    scheduler = _make_scheduler()
    _, group_id = _admit_striped_sequence(scheduler)
    plan = scheduler.plan_ls_kv_scale_down(group_id, 1)
    assert plan.success
    physical_executor = _InProcessP2PExecutor(monkeypatch, _make_caches())

    with pytest.raises(RuntimeError, match="injected worker completion failure"):
        execute_planned_ls_kv_scale_down(
            scheduler,
            _FailAfterPhysicalCopyExecutor(physical_executor),
            plan,
            abort_on_copy_error=False,
        )

    with pytest.raises(RuntimeError, match="transaction is reserved"):
        scheduler.schedule()
    scheduler.abort_ls_kv_scale_down(plan)


def test_scale_down_rejects_current_decode_master_without_calling_executor():
    scheduler = _make_scheduler()
    sequence, group_id = _admit_striped_sequence(scheduler)

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
    sequence, group_id = _admit_striped_sequence(scheduler)
    context_before = _context_snapshot(sequence)
    free_before = _free_blocks(scheduler)

    plan = scheduler.plan_ls_kv_scale_down(group_id, 1)
    assert plan.success, plan.failure_reason
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
    assert _context_snapshot(sequence) == context_before
    assert _free_blocks(scheduler) == free_before
    assert scheduler.schedule().is_prefill is False


def test_scale_down_rejects_stale_change_on_unmoved_group_sequence(monkeypatch):
    scheduler = _make_scheduler()
    moved = Sequence(list(range(5)), 1.0, 32, True)
    unmoved = Sequence([17], 1.0, 32, True)
    scheduler.add(moved)
    scheduler.add(unmoved)
    admission = scheduler.schedule()
    assert admission.ls_initial_kv_dops == [2]

    worker = scheduler.worker_state[0]
    for sequence in (moved, unmoved):
        assert worker.may_append(sequence, 1)
        sequence.append_token(99, BlockContextSlot.ACTIVE)
        sequence.mark_last_token_pending(BlockContextSlot.ACTIVE)
        worker.add_running_tokens(sequence.block_ctx().master_sp_idx, 1)
    assert moved.committed_context_len(BlockContextSlot.ACTIVE, 1) > 0
    assert unmoved.committed_context_len(BlockContextSlot.ACTIVE, 1) == 0

    group_id = admission.ls_initial_group_ids[0]
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

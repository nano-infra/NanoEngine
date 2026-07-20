from __future__ import annotations

import importlib
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from nanodeploy._cpp import (
    LSAddError,
    LSAdmissionKind,
    LSAdmissionTargetKind,
    LSFatalCode,
    ScheduleAction,
    SequenceStatus,
)
from nanodeploy.worker.kv_p2p import KVCacheP2PResult


class _KVPlanState(Enum):
    RESERVED = 0
    DISPATCHED = 1
    COMMITTED = 2
    ABORTED = 3


def _add_result(*, accepted=True, assigned_dp=0, error=LSAddError.NONE, reason=""):
    return SimpleNamespace(
        accepted=accepted,
        assigned_dp=assigned_dp,
        error=error,
        reason=reason,
    )


def _load_llm_engine_without_model_runner(monkeypatch):
    fake_ray_executor = ModuleType("nanodeploy.engine.ray_executor")
    fake_ray_executor.RayExecutor = object
    monkeypatch.setitem(
        sys.modules, "nanodeploy.engine.ray_executor", fake_ray_executor
    )
    sys.modules.pop("nanodeploy.engine.llm_engine", None)
    return importlib.import_module("nanodeploy.engine.llm_engine").LLMEngine


def _empty_ls_action_abi(*, dp_size=1, epoch_after=0):
    return {
        "ls_pool_resource_epoch_before": [0] * dp_size,
        "ls_pool_resource_epoch_after": [epoch_after] * dp_size,
        "ls_preemption_reasons": [],
        "ls_group_ids": [],
        "ls_group_dp_indices": [],
        "ls_real_batch_sizes": [],
        "ls_master_dops": [],
        "ls_kv_dops": [],
        "ls_master_ranks": [],
        "ls_master_batch_sizes": [],
        "ls_group_rank_allocations": [],
        "ls_group_used_kv_tokens": [],
        "ls_group_used_kv_blocks": [],
        "ls_iteration_sequence_ids": [],
        "ls_iteration_master_assignments": [],
        "ls_pending_append_blocks_per_master": [],
        "ls_new_master_ranks": [],
        "ls_reused_passive_master_ranks": [],
        "ls_scale_reasons": [],
        "ls_historical_kv_migration_bytes": [],
    }


def _kv_consolidation_action(plan, *, epoch=0, **overrides):
    fields = _empty_ls_action_abi(dp_size=1)
    fields.update(
        action=ScheduleAction.KV_CONSOLIDATION,
        is_prefill=False,
        kv_consolidation_plan=plan,
        ls_pool_resource_epoch_before=[epoch],
        ls_pool_resource_epoch_after=[epoch],
        ls_admission_records=[],
        ls_real_decode_ids_by_dp=[],
        ls_running_ids_by_dp_after_commit=[],
        ls_preempted_sequence_ids=[],
        dp_seqs=[],
        dp_sp_seqs=[],
        filtered_dp_sp_seqs=[],
        sp_send_counts=[],
        sp_recv_counts=[],
        sp_size_hist_per_dp=[],
        sp_q_matrix=[],
        sp_res_matrix=[],
        waiting_head_blocks=[],
        waiting_total_blocks=[],
        ls_kv_consolidation_candidate=True,
        ls_kv_consolidation_group_id=plan.group_id,
        ls_kv_consolidation_source_rank=plan.source_rank,
        ls_kv_consolidation_target_dop=len(plan.retained_ranks),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _admission_action(record, *, dp_size=1, running_ids=None):
    if running_ids is None:
        running_ids = [[] for _ in range(dp_size)]
        if not record.bootstrap_finished:
            running_ids[record.dp_idx].append(record.sequence.seq_id)
    epochs = [0] * dp_size
    epochs[record.dp_idx] = 1
    fields = _empty_ls_action_abi(dp_size=dp_size)
    fields.update(
        action=ScheduleAction.ADMISSION,
        is_prefill=False,
        kv_consolidation_plan=None,
        ls_admission_records=[record],
        ls_real_decode_ids_by_dp=[],
        ls_running_ids_by_dp_after_commit=running_ids,
        ls_preempted_sequence_ids=[],
        ls_pool_resource_epoch_after=epochs,
        dp_seqs=[],
        dp_sp_seqs=[],
        filtered_dp_sp_seqs=[],
        sp_send_counts=[],
        sp_recv_counts=[],
        sp_size_hist_per_dp=[],
        sp_q_matrix=[],
        sp_res_matrix=[],
        waiting_head_blocks=[],
        waiting_total_blocks=[],
    )
    return SimpleNamespace(**fields)


def _decode_action(*, real_ids_by_dp, attention_sp=1, running_ids=None):
    dp_size = len(real_ids_by_dp)
    running_ids = (
        [list(ids) for ids in real_ids_by_dp]
        if running_ids is None
        else running_ids
    )
    sequences = {
        seq_id: SimpleNamespace(seq_id=seq_id)
        for ids in real_ids_by_dp
        for seq_id in ids
    }
    dp_seqs = [[sequences[seq_id] for seq_id in ids] for ids in real_ids_by_dp]
    group_dp_indices = [
        dp_idx for dp_idx, ids in enumerate(real_ids_by_dp) if ids
    ]
    group_sequence_ids = [
        list(ids) for ids in real_ids_by_dp if ids
    ]
    group_count = len(group_dp_indices)
    zero_rank_vector = [0] * attention_sp
    zero_rank_matrix = [
        [0] * attention_sp for _ in range(attention_sp)
    ]
    active_dps = {dp_idx for dp_idx, ids in enumerate(real_ids_by_dp) if ids}
    fields = _empty_ls_action_abi(dp_size=dp_size)
    fields.update(
        action=ScheduleAction.DECODE,
        is_prefill=False,
        kv_consolidation_plan=None,
        ls_admission_records=[],
        ls_real_decode_ids_by_dp=[list(ids) for ids in real_ids_by_dp],
        ls_running_ids_by_dp_after_commit=[list(ids) for ids in running_ids],
        ls_preempted_sequence_ids=[],
        ls_pool_resource_epoch_after=[
            int(dp_idx in active_dps) for dp_idx in range(dp_size)
        ],
        dp_seqs=dp_seqs,
        dp_sp_seqs=[
            list(dp_seqs[dp_idx])
            for dp_idx in range(dp_size)
            for _ in range(attention_sp)
        ],
        filtered_dp_sp_seqs=[
            list(dp_seqs[dp_idx]) if sp_idx == 0 else []
            for dp_idx in range(dp_size)
            for sp_idx in range(attention_sp)
        ],
        sp_send_counts=[list(zero_rank_vector) for _ in range(dp_size)],
        sp_recv_counts=[list(zero_rank_vector) for _ in range(dp_size)],
        sp_size_hist_per_dp=[
            [0] * (attention_sp + 1) for _ in range(dp_size)
        ],
        sp_q_matrix=[
            [list(row) for row in zero_rank_matrix] for _ in range(dp_size)
        ],
        sp_res_matrix=[
            [list(row) for row in zero_rank_matrix] for _ in range(dp_size)
        ],
        waiting_head_blocks=[0] * dp_size,
        waiting_total_blocks=[0] * dp_size,
        ls_group_ids=list(range(10, 10 + group_count)),
        ls_group_dp_indices=group_dp_indices,
        ls_real_batch_sizes=[len(ids) for ids in group_sequence_ids],
        ls_master_dops=[1] * group_count,
        ls_kv_dops=[1] * group_count,
        ls_master_ranks=[[0] for _ in range(group_count)],
        ls_master_batch_sizes=[[len(ids)] for ids in group_sequence_ids],
        ls_group_rank_allocations=[[0] for _ in range(group_count)],
        ls_group_used_kv_tokens=[
            list(zero_rank_vector) for _ in range(group_count)
        ],
        ls_group_used_kv_blocks=[
            list(zero_rank_vector) for _ in range(group_count)
        ],
        ls_iteration_sequence_ids=group_sequence_ids,
        ls_iteration_master_assignments=[
            [0] * len(ids) for ids in group_sequence_ids
        ],
        ls_pending_append_blocks_per_master=[
            list(zero_rank_vector) for _ in range(group_count)
        ],
        ls_new_master_ranks=[[] for _ in range(group_count)],
        ls_reused_passive_master_ranks=[[] for _ in range(group_count)],
        ls_scale_reasons=["none"] * group_count,
        ls_historical_kv_migration_bytes=[0] * group_count,
    )
    return SimpleNamespace(**fields)


def _validator_engine(llm_engine_type, *, attention_dp=1, attention_sp=1):
    latched = []
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
    )
    engine.scheduler = SimpleNamespace(latch_ls_fatal=latched.append)
    engine.fatal_error = None
    engine.fatal_code = None
    return engine, latched


def _make_actual_ls_scheduler(*, loop_count=1, block_size=4, num_blocks=64):
    from nanodeploy._cpp import Scheduler

    return Scheduler(
        "",
        loop_count,
        16,
        4096,
        16,
        -1,
        1,
        1,
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
        0,
        64,
        True,
        "centralized",
        "off",
        0.50,
        0.80,
        2,
        2,
        1,
        128,
        True,
        10,
        1000,
        4096,
        100,
    )


class _ActualSchedulerAdapter:
    """Mirror the production Python wrapper around the bound C++ scheduler."""

    def __init__(self, scheduler, *, attention_sp=1):
        self._scheduler = scheduler
        self.attention_sp = attention_sp

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def postprocess(
        self,
        dp_sp_seqs,
        sample_token_ids,
        metrics_manager,
        elapsed_time,
        loop_count,
    ):
        return self._scheduler.postprocess(
            dp_sp_seqs,
            sample_token_ids,
            metrics_manager is not None,
            elapsed_time,
            loop_count,
        )


def _make_engine_around_actual_scheduler(
    llm_engine_type, scheduler, *, loop_count=1, block_size=4, num_blocks=64
):
    from nanodeploy.metrics import MetricsManager

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
        loop_count=loop_count,
        mode="decode",
        dummy_prefill=True,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    engine.scheduler = _ActualSchedulerAdapter(scheduler)
    engine.metrics_manager = MetricsManager()
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0
    engine.last_schedule_action = None
    engine.last_execution_loop_count = 1
    engine.log_decode_step_detail = False
    return engine


def test_maintenance_copy_failure_marks_engine_fatal(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    plan = SimpleNamespace(
        success=True,
        moves=[
            SimpleNamespace(
                dp_idx=0,
                src_sp_rank=1,
                dst_sp_rank=0,
                src_block_id=3,
                src_token_offset=0,
                dst_block_id=7,
                dst_token_offset=0,
                num_tokens=4,
            )
        ],
        transaction_id=7,
        group_id=3,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=4,
        state=_KVPlanState.RESERVED,
    )
    schedule_result = _kv_consolidation_action(plan)

    class FakeScheduler:
        def __init__(self):
            self.schedule_calls = 0
            self.abort_calls = 0
            self.latched_codes = []

        def schedule(self):
            self.schedule_calls += 1
            return schedule_result

        def is_finished(self):
            return False

        def plan_ls_kv_scale_down(self, *_args):
            pytest.fail("automatic maintenance re-planned the reserved plan")

        def mark_ls_kv_scale_down_dispatched(self, reserved_plan):
            assert reserved_plan is plan
            plan.state = _KVPlanState.DISPATCHED
            return True

        def abort_ls_kv_scale_down(self, reserved_plan):
            assert reserved_plan is plan
            self.abort_calls += 1

        def latch_ls_fatal(self, code):
            self.latched_codes.append(code)

    class FailingExecutor:
        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert len(moves) == 1
            assert timeout is None
            raise RuntimeError("injected collective completion failure")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=2,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
    )
    engine.scheduler = FakeScheduler()
    engine.executor = FailingExecutor()
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0

    with pytest.raises(RuntimeError, match="injected collective completion failure"):
        engine.step()
    assert isinstance(engine.fatal_error, RuntimeError)
    assert engine.scheduler.schedule_calls == 1
    assert engine.scheduler.abort_calls == 0
    assert plan.state == _KVPlanState.DISPATCHED
    assert engine.fatal_code.name == "KV_CONSOLIDATION_FAILED"
    assert all(
        code.name == "KV_CONSOLIDATION_FAILED"
        for code in engine.scheduler.latched_codes
    )

    for operation in (
        engine.step,
        lambda: engine.add_request(object()),
        lambda: engine.free_to_be_migrated(object()),
        engine.is_finished,
    ):
        with pytest.raises(RuntimeError, match="restart required"):
            operation()
    assert engine.scheduler.schedule_calls == 1


def test_maintenance_pre_dispatch_abort_is_nonfatal_zero_token_step(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)

    class BrokenMoves:
        def __iter__(self):
            raise RuntimeError("injected move materialization failure")

    plan = SimpleNamespace(
        success=True,
        moves=BrokenMoves(),
        transaction_id=7,
        group_id=3,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=4,
        state=_KVPlanState.RESERVED,
    )
    schedule_result = _kv_consolidation_action(plan)

    class FakeScheduler:
        def __init__(self):
            self.abort_calls = 0

        def schedule(self):
            return schedule_result

        def is_finished(self):
            return False

        def abort_ls_kv_scale_down(self, reserved_plan):
            assert reserved_plan is plan
            assert plan.state == _KVPlanState.RESERVED
            plan.state = _KVPlanState.ABORTED
            self.abort_calls += 1

        def mark_ls_kv_scale_down_dispatched(self, _reserved_plan):
            pytest.fail("pre-dispatch failure entered DISPATCHED")

        def latch_ls_fatal(self, _code):
            pytest.fail("safe pre-dispatch abort latched scheduler fatal state")

    class ExecutorStub:
        def copy_kv_ranges_p2p(self, *_args, **_kwargs):
            pytest.fail("pre-dispatch failure reached physical executor")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=2,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
    )
    engine.scheduler = FakeScheduler()
    engine.executor = ExecutorStub()
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0

    outputs, num_tokens, batch_size, _sch_ms, post_ms = engine.step()

    assert outputs == []
    assert num_tokens == 0
    assert batch_size == 0
    assert post_ms == 0.0
    assert engine.fatal_error is None
    assert engine.fatal_code is None
    assert engine.scheduler.abort_calls == 1
    assert plan.state == _KVPlanState.ABORTED
    assert engine.pending_maintenance_stall_ms >= 0.0


def test_maintenance_success_logs_committed_plan_identity(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    module = sys.modules[llm_engine_type.__module__]
    log_records = []
    monkeypatch.setattr(
        module,
        "logger",
        SimpleNamespace(
            info=log_records.append,
            warning=log_records.append,
            exception=lambda *_args, **_kwargs: None,
        ),
    )
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
        moves=[move],
        transaction_id=7,
        group_id=3,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=2,
        state=_KVPlanState.RESERVED,
    )
    schedule_result = _kv_consolidation_action(
        plan,
        epoch=4,
        ls_kv_consolidation_group_util=0.25,
        ls_kv_consolidation_stable_steps=2,
    )

    class FakeScheduler:
        def schedule(self):
            return schedule_result

        def is_finished(self):
            return False

        def mark_ls_kv_scale_down_dispatched(self, reserved_plan):
            assert reserved_plan is plan
            plan.state = _KVPlanState.DISPATCHED
            return True

        def commit_ls_kv_scale_down(self, reserved_plan):
            assert reserved_plan is plan
            assert plan.state == _KVPlanState.DISPATCHED
            plan.state = _KVPlanState.COMMITTED
            return True

        def abort_ls_kv_scale_down(self, _reserved_plan):
            pytest.fail("successful dispatched transaction attempted abort")

        def latch_ls_fatal(self, _code):
            pytest.fail("successful transaction latched fatal state")

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

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=2,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
    )
    engine.scheduler = FakeScheduler()
    engine.executor = ExecutorStub()
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0

    outputs, num_tokens, batch_size, _sch_ms, post_ms = engine.step()

    assert outputs == []
    assert num_tokens == 0
    assert batch_size == 0
    assert post_ms == 0.0
    assert plan.state == _KVPlanState.COMMITTED
    assert engine.fatal_error is None
    maintenance_record = next(
        record
        for record in log_records
        if isinstance(record, dict) and record.get("mode") == "ls_kv_consolidation"
    )


def test_direct_step_on_empty_engine_is_nonfatal_api_error(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)

    class FakeScheduler:
        def is_finished(self):
            return True

        def schedule(self):
            pytest.fail("empty engine reached Scheduler.schedule")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    engine.scheduler = FakeScheduler()
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(RuntimeError, match="cannot step an empty"):
        engine.step()
    assert engine.fatal_error is None


def test_non_ls_empty_engine_delegates_to_legacy_scheduler(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)

    class LegacyScheduleReached(RuntimeError):
        pass

    class FakeScheduler:
        def is_finished(self):
            return True

        def schedule(self):
            raise LegacyScheduleReached("legacy scheduler reached")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        enable_ls_decode_core_scheduler=False,
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
    )
    engine.scheduler = FakeScheduler()
    engine.fatal_error = None

    with pytest.raises(LegacyScheduleReached, match="legacy scheduler reached"):
        engine.step()
    assert engine.fatal_error is None


def test_schedule_fatal_code_preserves_ordinal_zero(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)

    class OrdinalZeroFailure(RuntimeError):
        pass

    class FakeScheduler:
        def __init__(self):
            self.latched = []

        def is_finished(self):
            return False

        def schedule(self):
            raise OrdinalZeroFailure("injected ordinal-zero scheduler fatal")

        def ls_fatal_code(self):
            return LSFatalCode.NO_PROGRESS_INVARIANT

        def latch_ls_fatal(self, code):
            self.latched.append(code)

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
    )
    engine.scheduler = FakeScheduler()
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(OrdinalZeroFailure, match="ordinal-zero"):
        engine.step()

    assert engine.fatal_code == LSFatalCode.NO_PROGRESS_INVARIANT
    assert engine.scheduler.latched == [LSFatalCode.NO_PROGRESS_INVARIANT]


def test_unexpected_action_validator_exception_latches_postpublication(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    latched = []
    schedule_result = SimpleNamespace(action=ScheduleAction.ADMISSION)
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
    )
    engine.scheduler = SimpleNamespace(
        is_finished=lambda: False,
        schedule=lambda: schedule_result,
        latch_ls_fatal=latched.append,
    )
    engine.fatal_error = None
    engine.fatal_code = None
    engine._validate_ls_action_fields = lambda _result: (_ for _ in ()).throw(
        KeyError("injected validator implementation failure")
    )

    with pytest.raises(KeyError, match="validator implementation failure"):
        engine.step()

    assert isinstance(engine.fatal_error, KeyError)
    assert engine.fatal_code.name == "POST_PUBLICATION_INVARIANT"
    assert len(latched) == 1
    assert latched[0].name == "POST_PUBLICATION_INVARIANT"


def test_generate_keeps_zero_token_outputs_before_skipping_throughput(
    monkeypatch,
):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    progress_updates = []

    class FakeProgress:
        def __init__(self, **_kwargs):
            pass

        def update(self, amount):
            progress_updates.append(amount)

        def set_postfix(self, _values):
            pass

        def close(self):
            pass

    module = sys.modules[llm_engine_type.__module__]
    monkeypatch.setattr(module, "tqdm", FakeProgress)
    clock = iter([0.0, 1.0, 10.0, 11.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))
    decode_samples = []
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(loop_count=1)
    engine.scheduler = SimpleNamespace(get_total_waiting_size=lambda: 2)
    engine.metrics_manager = SimpleNamespace(
        server_metric=SimpleNamespace(
            record_prefill_throughput=lambda *_args: None,
            record_decode_throughput=lambda tokens, duration: (
                decode_samples.append((tokens, duration))
            ),
        ),
        log_server_metrics=lambda **_kwargs: None,
        get_server_summary=lambda: {},
    )
    steps = iter(
        [
            ([(17, [0])], 0, 0, 0.0, 0.0),
            ([(18, [0, 4])], -1, 1, 0.0, 0.0),
        ]
    )
    finished = iter([False, False, True])

    def fake_step():
        result = next(steps)
        engine.last_schedule_action = (
            ScheduleAction.ADMISSION
            if result[1] == 0
            else ScheduleAction.DECODE
        )
        return result

    engine.step = fake_step
    engine.is_finished = lambda: next(finished)

    result = engine.generate(use_tqdm=True)

    assert result is None
    assert progress_updates == [1, 1]
    assert decode_samples == [(1, 1.0)]


def test_ls_add_request_commits_metric_ticket_after_scheduler_accepts(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.metrics import MetricsManager

    seq = SimpleNamespace(seq_id=17, num_prompt_tokens=11, metric=None)

    class FakeScheduler:
        def precheck_add_identity(self, checked):
            assert checked is seq
            return _add_result()

        def add(self, added):
            assert added is seq
            assert added.metric is None
            return _add_result()

        def get_ls_arrival_order(self, seq_id):
            assert seq_id == seq.seq_id
            return 0

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    engine.scheduler = FakeScheduler()
    engine.metrics_manager = MetricsManager()
    engine.fatal_error = None
    engine.fatal_code = None

    engine.add_request([seq])

    assert seq.metric is engine.metrics_manager.get_sequence_metric(17)
    assert seq.metric.arrival_time is not None
    assert seq.metric.decode_arrival_time is not None
    assert engine.metrics_manager.server_metric.total_prompt_tokens == 11


def test_ls_add_request_aborts_metric_ticket_when_scheduler_rejects(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.metrics import MetricsManager

    seq = SimpleNamespace(seq_id=17, num_prompt_tokens=11, metric=None)

    class FakeScheduler:
        def precheck_add_identity(self, checked):
            assert checked is seq
            return _add_result()

        def add(self, _seq):
            raise ValueError("injected singleton rejection")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    engine.scheduler = FakeScheduler()
    engine.metrics_manager = MetricsManager()
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(ValueError, match="injected singleton rejection"):
        engine.add_request([seq])

    assert seq.metric is None
    assert engine.metrics_manager.get_sequence_metric(17) is None
    assert engine.metrics_manager.server_metric.total_prompt_tokens == 0

    # The aborted provisional entry must not poison a later independent retry.
    ticket = engine.metrics_manager.prepare_sequence_metric(17, 13)
    assert engine.metrics_manager.abort_sequence_metric(ticket) is True


def test_ls_add_request_identity_precheck_runs_before_metric_ticket(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.engine.scheduler import UnschedulableRequestError
    from nanodeploy.metrics import MetricsManager

    seq = SimpleNamespace(seq_id=17, num_prompt_tokens=11, metric=None)
    rejection_code = LSAddError.ALREADY_ADDED_OR_ASSIGNED

    class FakeScheduler:
        def precheck_add_identity(self, checked):
            assert checked is seq
            return SimpleNamespace(
                accepted=False,
                error=rejection_code,
                assigned_dp=-1,
                reason="duplicate sequence ID",
            )

        def add(self, _seq):
            pytest.fail("identity-rejected request reached Scheduler.add")

    manager = MetricsManager()
    manager.prepare_sequence_metric = lambda *_args: pytest.fail(
        "identity-rejected request created a metric ticket"
    )
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    engine.scheduler = FakeScheduler()
    engine.metrics_manager = manager
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(UnschedulableRequestError) as raised:
        engine.add_request([seq])

    assert raised.value.error is rejection_code
    assert raised.value.assigned_dp == -1
    assert raised.value.reason == "duplicate sequence ID"
    assert manager.get_server_summary()["ls_rejected_requests"] == 1


def test_ls_add_request_typed_singleton_rejection_aborts_ticket(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.engine.scheduler import UnschedulableRequestError
    from nanodeploy.metrics import MetricsManager

    seq = SimpleNamespace(seq_id=17, num_prompt_tokens=11, metric=None)
    rejection_code = LSAddError.FUTURE_TOKEN_NO_FIT
    rejection = SimpleNamespace(
        accepted=False,
        error=rejection_code,
        assigned_dp=3,
        reason="future KV does not fit",
    )
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    engine.scheduler = SimpleNamespace(
        precheck_add_identity=lambda _seq: _add_result(),
        add=lambda _seq: rejection,
    )
    engine.metrics_manager = MetricsManager()
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(UnschedulableRequestError) as raised:
        engine.add_request([seq])

    assert raised.value.error is rejection_code
    assert raised.value.assigned_dp == 3
    assert engine.metrics_manager.get_sequence_metric(17) is None
    assert engine.metrics_manager.server_metric.total_prompt_tokens == 0
    assert (
        engine.metrics_manager.get_server_summary()["ls_rejected_requests"] == 1
    )


def test_ls_add_request_metric_commit_failure_latches_engine(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.metrics import MetricsManager

    seq = SimpleNamespace(seq_id=17, num_prompt_tokens=11, metric=None)
    manager = MetricsManager()
    manager.commit_sequence_metric = lambda _ticket: (_ for _ in ()).throw(
        RuntimeError("injected metric commit failure")
    )
    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(enable_ls_decode_core_scheduler=True)
    latched = []
    engine.scheduler = SimpleNamespace(
        precheck_add_identity=lambda _seq: _add_result(),
        add=lambda _seq: _add_result(),
        get_ls_arrival_order=lambda _seq_id: 0,
        latch_ls_fatal=latched.append,
    )
    engine.metrics_manager = manager
    engine.fatal_error = None
    engine.fatal_code = None

    with pytest.raises(RuntimeError, match="injected metric commit failure"):
        engine.add_request([seq])
    assert isinstance(engine.fatal_error, RuntimeError)
    assert engine.fatal_code == LSFatalCode.METRIC_COMMIT_FAILED
    assert latched == [LSFatalCode.METRIC_COMMIT_FAILED]

    with pytest.raises(RuntimeError, match="restart required"):
        engine.add_request([seq])


def test_admission_only_bootstrap_completion_returns_zero_token_output(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.metrics import MetricsManager

    manager = MetricsManager()
    ticket = manager.prepare_sequence_metric(17, 11)
    metric = manager.commit_sequence_metric(ticket)
    metric.record_first_token()
    metric.num_generated_tokens = 1
    sequence = SimpleNamespace(
        seq_id=17,
        assigned_dp=0,
        status=SequenceStatus.FINISHED,
        completion_token_ids=[0],
    )
    admission = SimpleNamespace(
        sequence=sequence,
        dp_idx=0,
        batch_id=3,
        group_id_after_commit=None,
        admission_kind=LSAdmissionKind.FRESH,
        target_kind=LSAdmissionTargetKind.STANDALONE,
        planned_kv_dop=1,
        planned_kv_ranks=[0],
        bootstrap_finished=True,
        bootstrap_token_id=0,
    )
    schedule_result = _admission_action(admission)

    class FakeScheduler:
        def is_finished(self):
            return False

        def schedule(self):
            return schedule_result

        def get_total_waiting_size(self):
            return 0

        def get_total_waiting_migration_size(self):
            return 0

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
    )
    engine.scheduler = FakeScheduler()
    engine.metrics_manager = manager
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0

    outputs, num_tokens, batch_size, _sch_ms, post_ms = engine.step()

    assert outputs == [(17, [0])]
    assert num_tokens == 0
    assert batch_size == 0
    assert post_ms == 0.0
    assert manager.server_metric.num_completed_requests == 1
    assert manager.server_metric.total_generated_tokens == 1


def test_actual_scheduler_max_tokens_one_finishes_in_admission_action(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy._cpp import Sequence

    scheduler = _make_actual_ls_scheduler()
    engine = _make_engine_around_actual_scheduler(llm_engine_type, scheduler)
    sequence = Sequence([1, 2, 3, 4], 1.0, 1, True)

    engine.add_request([sequence])
    outputs, num_tokens, batch_size, _sch_ms, post_ms = engine.step()

    assert engine.last_schedule_action == ScheduleAction.ADMISSION
    assert outputs == [(sequence.seq_id, [0])]
    assert num_tokens == 0
    assert batch_size == 0
    assert post_ms == 0.0
    assert scheduler.is_finished()
    assert engine.metrics_manager.server_metric.num_completed_requests == 1


def test_actual_scheduler_max_tokens_two_decodes_exactly_once(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy._cpp import Sequence

    scheduler = _make_actual_ls_scheduler()
    engine = _make_engine_around_actual_scheduler(llm_engine_type, scheduler)
    sequence = Sequence([1, 2, 3, 4], 1.0, 2, True)
    engine.executor = SimpleNamespace(
        run=lambda dp_sp_tp_seqs, _is_prefill, **_kwargs: [
            [[7] for _sequence in sequences]
            for sequences in dp_sp_tp_seqs
        ]
    )

    engine.add_request([sequence])
    admission = engine.step()
    decode = engine.step()

    assert admission[0] == []
    assert admission[1:3] == (0, 0)
    assert decode[0] == [(sequence.seq_id, [0, 7])]
    assert decode[1:3] == (-1, 1)
    assert scheduler.is_finished()
    assert engine.metrics_manager.server_metric.num_completed_requests == 1


def test_actual_engine_chunked_16_executes_full_chunk_then_one_token_tail(
    monkeypatch,
):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy._cpp import Sequence

    scheduler = _make_actual_ls_scheduler(
        loop_count=16, block_size=64, num_blocks=16
    )
    engine = _make_engine_around_actual_scheduler(
        llm_engine_type,
        scheduler,
        loop_count=16,
        block_size=64,
        num_blocks=16,
    )
    observed_loop_counts = []

    def run(dp_sp_tp_seqs, _is_prefill, *, execution_loop_count):
        observed_loop_counts.append(execution_loop_count)
        return [
            [
                [7 + index for index in range(execution_loop_count)]
                for _sequence in sequences
            ]
            for sequences in dp_sp_tp_seqs
        ]

    engine.executor = SimpleNamespace(run=run)
    sequence = Sequence(list(range(63)), 1.0, 18, True)

    engine.add_request([sequence])
    admission = engine.step()
    first_decode = engine.step()
    final_decode = engine.step()

    assert admission[1:3] == (0, 0)
    assert first_decode[0] == []
    assert first_decode[1:3] == (-16, 1)
    assert final_decode[0][0][0] == sequence.seq_id
    assert len(final_decode[0][0][1]) == 18
    assert final_decode[1:3] == (-1, 1)
    assert observed_loop_counts == [16, 1]
    assert scheduler.is_finished()


def test_admission_action_rejects_consolidation_plan(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    schedule_result = SimpleNamespace(
        action=ScheduleAction.ADMISSION,
        is_prefill=False,
        kv_consolidation_plan=object(),
        ls_admission_records=[],
        ls_real_decode_ids_by_dp=[],
        ls_preempted_sequence_ids=[],
        ls_preemption_reasons=[],
    )
    engine, latched = _validator_engine(llm_engine_type)

    with pytest.raises(RuntimeError, match="must not carry a consolidation plan"):
        engine._validate_ls_action_fields(schedule_result)

    assert engine.fatal_code == LSFatalCode.POST_PUBLICATION_INVARIANT
    assert latched == [LSFatalCode.POST_PUBLICATION_INVARIANT]


def test_ls_gauge_update_exception_latches_postpublication(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    schedule_result = _decode_action(real_ids_by_dp=[[11]])
    latched = []

    class ServerMetric:
        def update_running_requests(self, _count):
            raise RuntimeError("injected LS gauge update failure")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
        loop_count=1,
        mode="decode",
        dummy_prefill=True,
    )
    engine.scheduler = SimpleNamespace(
        is_finished=lambda: False,
        schedule=lambda: schedule_result,
        get_total_waiting_size=lambda: 0,
        get_total_waiting_migration_size=lambda: 0,
        latch_ls_fatal=latched.append,
    )
    engine.executor = SimpleNamespace(
        run=lambda *_args: pytest.fail("gauge failure reached executor")
    )
    engine.metrics_manager = SimpleNamespace(server_metric=ServerMetric())
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0
    engine.log_decode_step_detail = False

    with pytest.raises(RuntimeError, match="injected LS gauge update failure"):
        engine.step()

    assert engine.fatal_code.name == "POST_PUBLICATION_INVARIANT"
    assert [code.name for code in latched] == ["POST_PUBLICATION_INVARIANT"]


def test_ls_executor_run_exception_latches_postpublication(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    schedule_result = _decode_action(real_ids_by_dp=[[11]])
    latched = []

    class ServerMetric:
        update_running_requests = staticmethod(lambda _count: None)
        update_waiting_requests = staticmethod(lambda _count: None)
        update_waiting_migration_requests = staticmethod(lambda _count: None)
        update_sp_stats = staticmethod(lambda *_args: None)
        update_waiting_blocks = staticmethod(lambda *_args: None)

    def fail_run(*_args, **_kwargs):
        raise RuntimeError("injected LS executor.run failure")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
        loop_count=1,
        mode="decode",
        dummy_prefill=True,
    )
    engine.scheduler = SimpleNamespace(
        is_finished=lambda: False,
        schedule=lambda: schedule_result,
        get_total_waiting_size=lambda: 0,
        get_total_waiting_migration_size=lambda: 0,
        latch_ls_fatal=latched.append,
    )
    engine.executor = SimpleNamespace(run=fail_run)
    engine.metrics_manager = SimpleNamespace(server_metric=ServerMetric())
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0
    engine.log_decode_step_detail = False

    with pytest.raises(RuntimeError, match="injected LS executor.run failure"):
        engine.step()

    assert engine.fatal_code.name == "POST_PUBLICATION_INVARIANT"
    assert [code.name for code in latched] == ["POST_PUBLICATION_INVARIANT"]


def test_decode_metrics_exclude_collective_dummies_and_new_admissions(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    from nanodeploy.metrics import MetricsManager

    class FakeSequence:
        def __init__(self, seq_id, length, *, finished=False):
            self.seq_id = seq_id
            self.assigned_dp = 0
            self.status = SequenceStatus.RUNNING
            self.length = length
            self.is_finished = finished
            self.completion_token_ids = [0, seq_id]

        def __len__(self):
            return self.length

    real = FakeSequence(17, 5, finished=True)
    dummy = FakeSequence(999, 23, finished=True)
    newly_admitted = FakeSequence(18, 7)
    admission = SimpleNamespace(
        sequence=newly_admitted,
        dp_idx=0,
        batch_id=3,
        group_id_after_commit=4,
        admission_kind=LSAdmissionKind.FRESH,
        target_kind=LSAdmissionTargetKind.STANDALONE,
        planned_kv_dop=1,
        planned_kv_ranks=[0],
        bootstrap_finished=False,
        bootstrap_token_id=0,
    )

    class FakeScheduleResult(SimpleNamespace):
        def __getattr__(self, _name):
            return []

    schedule_result = FakeScheduleResult(
        action=ScheduleAction.DECODE,
        is_prefill=False,
        kv_consolidation_plan=None,
        ls_admission_records=[admission],
        ls_real_decode_ids_by_dp=[[17]],
        ls_running_ids_by_dp_after_commit=[[17, 18]],
        ls_preempted_sequence_ids=[],
        dp_seqs=[[real, dummy]],
        dp_sp_seqs=[[real, dummy]],
        filtered_dp_sp_seqs=[[real, dummy]],
        sp_send_counts=[[0]],
        sp_recv_counts=[[0]],
        sp_size_hist_per_dp=[[0, 0]],
        sp_q_matrix=[[[0]]],
        sp_res_matrix=[[[0]]],
        waiting_head_blocks=[0],
        waiting_total_blocks=[0],
        **(
            _empty_ls_action_abi(epoch_after=1)
            | {
                "ls_group_ids": [4],
                "ls_group_dp_indices": [0],
                "ls_real_batch_sizes": [1],
                "ls_master_dops": [1],
                "ls_kv_dops": [1],
                "ls_master_ranks": [[0]],
                "ls_master_batch_sizes": [[1]],
                "ls_group_rank_allocations": [[0]],
                "ls_group_used_kv_tokens": [[5]],
                "ls_group_used_kv_blocks": [[1]],
                "ls_iteration_sequence_ids": [[17]],
                "ls_iteration_master_assignments": [[0]],
                "ls_pending_append_blocks_per_master": [[1]],
                "ls_new_master_ranks": [[]],
                "ls_reused_passive_master_ranks": [[]],
                "ls_scale_reasons": ["none"],
                "ls_historical_kv_migration_bytes": [0],
            }
        ),
    )

    class FakeScheduler:
        attention_sp = 1
        worker_state = [
            SimpleNamespace(
                block_manager=[SimpleNamespace(free_block_ids=[])],
            )
        ]

        def is_finished(self):
            return False

        def schedule(self):
            return schedule_result

        def postprocess(self, *_args):
            pass

        def get_total_waiting_size(self):
            return 0

        def get_total_waiting_migration_size(self):
            return 0

    manager = MetricsManager()
    for seq_id, prompt_tokens in ((17, 5), (18, 7)):
        ticket = manager.prepare_sequence_metric(seq_id, prompt_tokens)
        metric = manager.commit_sequence_metric(ticket)
        metric.record_first_token()
        metric.num_generated_tokens = 1

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        enable_ls_decode_core_scheduler=True,
        dummy_bootstrap_token_id=0,
        loop_count=1,
        mode="decode",
        dummy_prefill=True,
        kvcache_block_size=64,
        num_kvcache_blocks=10,
    )
    engine.scheduler = FakeScheduler()
    engine.executor = SimpleNamespace(
        run=lambda *_args, **_kwargs: [[[4], [5]]]
    )
    engine.metrics_manager = manager
    engine.fatal_error = None
    engine.fatal_code = None
    engine.pending_maintenance_stall_ms = 0.0
    engine.log_decode_step_detail = False

    outputs, num_tokens, batch_size, _sch_ms, _post_ms = engine.step()

    assert outputs == [(17, [0, 17])]
    assert num_tokens == -1
    assert batch_size == 1
    assert manager.server_metric.num_running_requests == 2
    assert manager.server_metric.num_completed_requests == 1
    assert manager.server_metric.token_usage_by_dp[0] == 5

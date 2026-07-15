from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from nanodeploy._cpp import ScheduleAction


def _load_llm_engine_without_model_runner(monkeypatch):
    fake_ray_executor = ModuleType("nanodeploy.engine.ray_executor")
    fake_ray_executor.RayExecutor = object
    monkeypatch.setitem(
        sys.modules, "nanodeploy.engine.ray_executor", fake_ray_executor
    )
    sys.modules.pop("nanodeploy.engine.llm_engine", None)
    return importlib.import_module("nanodeploy.engine.llm_engine").LLMEngine


def test_maintenance_copy_failure_marks_engine_fatal(monkeypatch):
    llm_engine_type = _load_llm_engine_without_model_runner(monkeypatch)
    plan = SimpleNamespace(
        success=True,
        moves=[],
        transaction_id=7,
        group_id=3,
        dp_idx=0,
        source_rank=1,
        retained_ranks=[0],
        num_tokens=4,
    )
    schedule_result = SimpleNamespace(
        action=ScheduleAction.KV_CONSOLIDATION,
        kv_consolidation_plan=plan,
    )

    class FakeScheduler:
        def __init__(self):
            self.schedule_calls = 0
            self.abort_calls = 0

        def schedule(self):
            self.schedule_calls += 1
            return schedule_result

        def plan_ls_kv_scale_down(self, *_args):
            pytest.fail("automatic maintenance re-planned the reserved plan")

        def abort_ls_kv_scale_down(self, reserved_plan):
            assert reserved_plan is plan
            self.abort_calls += 1

    class FailingExecutor:
        def copy_kv_ranges_p2p(self, moves, timeout=None):
            assert moves == []
            assert timeout is None
            raise RuntimeError("injected collective completion failure")

    engine = object.__new__(llm_engine_type)
    engine.config = SimpleNamespace(attention_dp=1, attention_sp=2, attention_tp=1)
    engine.scheduler = FakeScheduler()
    engine.executor = FailingExecutor()
    engine.fatal_error = None
    engine.pending_maintenance_stall_ms = 0.0

    with pytest.raises(RuntimeError, match="injected collective completion failure"):
        engine.step()
    assert isinstance(engine.fatal_error, RuntimeError)
    assert engine.scheduler.schedule_calls == 1
    assert engine.scheduler.abort_calls == 0

    for operation in (
        engine.step,
        lambda: engine.add_request(object()),
        lambda: engine.free_to_be_migrated(object()),
        engine.is_finished,
    ):
        with pytest.raises(RuntimeError, match="restart required"):
            operation()
    assert engine.scheduler.schedule_calls == 1

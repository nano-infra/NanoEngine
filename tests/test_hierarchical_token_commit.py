import importlib
import sys
from types import ModuleType

import pytest

from nanodeploy._cpp import Sequence
from nanodeploy.engine.hierarchical_contract import (
    FinishEvent,
    FrontendEventBatch,
    LoadSnapshot,
    TokenCommitEvent,
)


def _load_lightweight_llm_engine(monkeypatch):
    lightweight_imports = {
        "nanodeploy.engine.deployment_manager": "DeploymentManager",
        "nanodeploy.engine.ray_executor": "RayExecutor",
        "nanodeploy.engine.scheduler": "Scheduler",
        "nanodeploy.engine.sequence": "Sequence",
    }
    for module_name, symbol in lightweight_imports.items():
        module = ModuleType(module_name)
        setattr(module, symbol, type(symbol, (), {}))
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delitem(
        sys.modules, "nanodeploy.engine.llm_engine", raising=False
    )
    return importlib.import_module("nanodeploy.engine.llm_engine").LLMEngine


def test_frontend_applies_token_commits_exactly_once(monkeypatch):
    llm_engine_type = _load_lightweight_llm_engine(monkeypatch)
    engine = object.__new__(llm_engine_type)
    sequence = Sequence([1, 2], 0.1, 2, True)
    sequence.seq_id = 9
    engine._hierarchical_sequences = {9: sequence}
    engine._hierarchical_generation_epochs = {}
    engine._hierarchical_terminal_reasons = {}
    first = TokenCommitEvent(9, 0, 1, 3, 4, 1, (7,))

    engine._apply_token_commit_event(first)

    assert sequence.completion_token_ids == [7]
    assert sequence.num_completed_tokens == 1
    with pytest.raises(RuntimeError, match="offset mismatch"):
        engine._apply_token_commit_event(first)

    terminal = TokenCommitEvent(
        9,
        0,
        1,
        3,
        5,
        2,
        (8,),
        finish_reason="LENGTH",
    )
    engine._apply_token_commit_event(terminal)
    assert sequence.completion_token_ids == [7, 8]
    assert engine._hierarchical_terminal_reasons == {9: "LENGTH"}


def test_frontend_rejects_generation_epoch_change_after_first_commit(
    monkeypatch,
):
    llm_engine_type = _load_lightweight_llm_engine(monkeypatch)
    engine = object.__new__(llm_engine_type)
    sequence = Sequence([1], 0.1, 2, True)
    sequence.seq_id = 10
    engine._hierarchical_sequences = {10: sequence}
    engine._hierarchical_generation_epochs = {10: 2}
    engine._hierarchical_terminal_reasons = {}

    with pytest.raises(RuntimeError, match="generation epoch changed"):
        engine._apply_token_commit_event(
            TokenCommitEvent(10, 0, 3, 1, 0, 1, (7,))
        )

    assert sequence.completion_token_ids == []


def test_frontend_commits_terminal_token_before_finish(monkeypatch):
    llm_engine_type = _load_lightweight_llm_engine(monkeypatch)
    engine = object.__new__(llm_engine_type)
    sequence = Sequence([1], 0.1, 1, True)
    sequence.seq_id = 11
    token_event = TokenCommitEvent(
        11,
        0,
        0,
        1,
        0,
        1,
        (8,),
        finish_reason="LENGTH",
    )
    finish_event = FinishEvent(
        11,
        1,
        "FINISHED",
        0,
        finish_reason="LENGTH",
    )
    batch = FrontendEventBatch(
        engine_id=0,
        load=LoadSnapshot(
            engine_id=0,
            ready=True,
            waiting=0,
            running=0,
            free_blocks_min=1,
            wave_id=1,
            quantum_id=0,
        ),
        token_commit_events=(token_event,),
        finish_events=(finish_event,),
    )

    class Deployment:
        @staticmethod
        def poll_ready_frontend_events():
            return (batch,)

    class Router:
        @staticmethod
        def record_loads(_loads):
            return None

        @staticmethod
        def record_add_results(events):
            return tuple(events)

        @staticmethod
        def poll_ingress_acks():
            return ()

        @staticmethod
        def record_first_schedule_events(events):
            return tuple(events)

        @staticmethod
        def record_token_commit_events(events):
            return tuple(events)

        @staticmethod
        def record_finish_events(events):
            return tuple(events)

        @staticmethod
        def record_resource_release_events(events):
            return tuple(events)

        @staticmethod
        def last_loads():
            return {}

    engine.deployment = Deployment()
    engine.router = Router()
    engine._hierarchical_sequences = {11: sequence}
    engine._hierarchical_generation_epochs = {}
    engine._hierarchical_terminal_reasons = {}
    engine._frontend_ingress_acks = []
    engine._frontend_add_results = []
    engine._frontend_first_schedule_events = []
    engine._frontend_first_token_events = []
    engine._frontend_finish_events = []

    engine._poll_frontend_control_plane()

    assert sequence.completion_token_ids == [8]
    assert sequence.is_finished
    assert engine._frontend_finish_events == [finish_event]

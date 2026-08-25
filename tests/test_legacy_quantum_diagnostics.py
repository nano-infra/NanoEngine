import importlib
import sys
from collections import deque
from types import ModuleType, SimpleNamespace


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


def test_legacy_engine_emits_common_quantum_diagnostic(monkeypatch):
    LLMEngine = _load_lightweight_llm_engine(monkeypatch)

    class FakeSequence:
        seq_id = 7
        metric = None
        is_finished = False
        completion_token_ids = []

        def __len__(self):
            return 128

        @staticmethod
        def block_ctx(_slot):
            return SimpleNamespace(num_dispatched_tokens=(64,))

    sequence = FakeSequence()

    class FakeWorkerState:
        block_manager = [SimpleNamespace(free_block_ids=range(10))]

        @staticmethod
        def is_control_dummy(_sequence):
            return False

    class FakeScheduler:
        attention_sp = 1
        worker_state = [FakeWorkerState()]
        waiting_migration = []

        @staticmethod
        def schedule():
            return SimpleNamespace(
                dp_seqs=[[sequence]],
                is_prefill=False,
                dp_sp_seqs=[[sequence]],
                filtered_dp_sp_seqs=[[sequence]],
                sp_send_counts=[[0]],
                sp_recv_counts=[[0]],
                sp_size_hist_per_dp=[[0, 1]],
                sp_q_matrix=[[[0]]],
                sp_res_matrix=[[[0]]],
                waiting_head_blocks=[0],
                waiting_total_blocks=[0],
            )

        @staticmethod
        def get_total_waiting_size():
            return 3

        @staticmethod
        def get_total_waiting_migration_size():
            return 0

        @staticmethod
        def postprocess(*_args):
            return None

    class FakeServerMetric:
        def __getattr__(self, name):
            if name.startswith("update_"):
                return lambda *_args: None
            raise AttributeError(name)

    executor_diagnostic = {
        "gpu_loop_ms_max": 4.0,
        "worker_total_ms_max": 7.0,
        "worker_rank_timings": (
            {"global_rank": 0, "gpu_loop_ms": 4.0},
        ),
    }
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        scheduler_arch="legacy_global",
        hierarchical_quantum_diagnostics=True,
        attention_dp=1,
        attention_sp=1,
        attention_tp=1,
        loop_count=16,
        mode="decode",
        dummy_prefill=True,
        num_kvcache_blocks=32,
    )
    engine.scheduler = FakeScheduler()
    engine.executor = SimpleNamespace(
        run=lambda _sequences, _is_prefill: [[[1] * 16]],
        last_quantum_diagnostic=executor_diagnostic,
    )
    engine.metrics_manager = SimpleNamespace(
        server_metric=FakeServerMetric(),
        complete_sequence=lambda _seq_id: None,
    )
    engine.log_decode_step_detail = False
    engine._quantum_diagnostics = deque()
    engine._central_quantum_id = 0

    engine.step()
    samples = engine.drain_quantum_diagnostics()

    assert len(samples) == 1
    sample = samples[0]
    assert sample["schema_version"] == 2
    assert sample["scheduler_arch"] == "legacy_global"
    assert sample["useful_real_batch_size"] == 1
    assert sample["attention_work_tokens"] == 64
    assert sample["rank_loads_before"][0]["free_blocks"] == 10
    assert sample["executor"] == executor_diagnostic
    assert engine.drain_quantum_diagnostics() == ()

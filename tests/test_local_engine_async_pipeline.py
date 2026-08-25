from pathlib import Path
from threading import Thread
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace

import pytest

from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    HIERARCHICAL_LOOP_COUNT,
    WorkerDecodeResult,
)
from nanodeploy.engine.local_engine import LocalEngineCore
from nanodeploy.engine.sequence import Sequence
from nanodeploy.sampling_params import SamplingParams


DEEPSEEK_MODEL = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
)


pytestmark = pytest.mark.skipif(
    not (DEEPSEEK_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found at {DEEPSEEK_MODEL}",
)


class _FakeExecutor:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.operations: list[tuple[str, int]] = []
        self.last_quantum_diagnostic = None
        for name in (
            "ray_get_latency_ms_total",
            "ray_get_latency_ms_max",
            "worker_result_wait_latency_ms_total",
            "worker_result_wait_latency_ms_max",
            "result_rebuild_latency_ms_total",
            "result_rebuild_latency_ms_max",
            "result_rebuild_sample_count",
            "result_index_latency_ms_total",
            "result_validate_latency_ms_total",
            "result_pack_latency_ms_total",
        ):
            setattr(self, name, 0)

    def activate_worker_transport(self, _timeout: float) -> None:
        return None

    def submit(self, batch, timeout: float):
        assert timeout > 0
        self.operations.append(("submit", batch.quantum_id))
        return SimpleNamespace(executor_begin=perf_counter())

    def collect(self, flight):
        del flight
        batch = self.engine.scheduler._inflight_batches[0]
        self.operations.append(("collect", batch.quantum_id))
        return [
            WorkerDecodeResult(
                wave_id=batch.wave_id,
                quantum_id=batch.quantum_id,
                global_rank=global_rank,
                forward_count=HIERARCHICAL_LOOP_COUNT,
                mastered_request_ids=batch.expected_request_ids(global_rank),
                sampled_token_ids=tuple(
                    (100,)
                    for _ in batch.expected_request_ids(global_rank)
                ),
            )
            for global_rank in batch.per_rank_sequences
        ]

    def shutdown_worker_transport(
        self, *, timeout: float, failed: bool
    ) -> None:
        assert timeout > 0
        del failed


def _make_config(async_depth: int) -> Config:
    return Config(
        model=str(DEEPSEEK_MODEL),
        scheduler_arch="hierarchical",
        loop_count=HIERARCHICAL_LOOP_COUNT,
        mode="decode",
        dummy_prefill=True,
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        kvcache_block_size=64,
        num_kvcache_blocks=32,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        hierarchical_worker_transport="zmq",
        hierarchical_async_depth=async_depth,
    )


def _run_two_token_request(async_depth: int):
    config = _make_config(async_depth)
    actor_class = LocalEngineCore.__ray_metadata__.modified_class
    topology = config.hierarchical_topology.engine(0)
    engine = actor_class(config, topology, [object()] * topology.world_size)
    executor = _FakeExecutor(engine)
    engine.executor = executor
    sequence = Sequence(
        [1, 2],
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=2,
            ignore_eos=True,
        ),
    )
    sequence.seq_id = 1
    command = AddCommand(
        request_id=1,
        prompt_len=2,
        num_tokens=2,
        max_tokens=2,
        temperature=0.1,
        ignore_eos=True,
        wave_id=1,
        sequence_payload=b"unused",
    )
    assert engine.scheduler.add(command, sequence).accepted
    engine._wave_running = True
    engine._wave_id = 1
    thread = Thread(target=engine._event_loop, daemon=True)
    engine._loop_thread = thread
    thread.start()

    deadline = monotonic() + 3
    while monotonic() < deadline:
        with engine._state_cv:
            if not engine._wave_running:
                engine._stop = True
                engine._state_cv.notify_all()
                break
        sleep(0.005)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert engine._failure is None
    assert sequence.completion_token_ids == [100, 100]
    return engine, executor.operations


def test_depth_two_submits_lookahead_before_collecting_oldest_flight():
    engine, operations = _run_two_token_request(2)

    assert operations == [
        ("submit", 0),
        ("submit", 1),
        ("collect", 0),
        ("collect", 1),
    ]
    assert len(engine._terminal_events) == 1
    assert len(engine._resource_release_events) == 1


def test_depth_one_preserves_submit_collect_order():
    _engine, operations = _run_two_token_request(1)

    assert operations == [
        ("submit", 0),
        ("collect", 0),
        ("submit", 1),
        ("collect", 1),
    ]

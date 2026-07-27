from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nanodeploy

from nanodeploy.engine.hierarchical_contract import (
    AddResultEvent,
    DecodeITLSample,
    FinishEvent,
    IngressAck,
)
from nanodeploy.sampling_params import SamplingParams


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += max(1, round(seconds * 1_000_000_000))


class DelayedAckEngine:
    def __init__(self, clock: FakeClock, ack_delay_ms: int) -> None:
        self.clock = clock
        self.ack_delay_ns = ack_delay_ms * 1_000_000
        self.config = SimpleNamespace(scheduler_arch="hierarchical")
        self.router = self
        self.dispatch_times: list[int] = []
        self._pending_ingress: dict[int, int] = {}
        self._pending_add: list[int] = []
        self._finish: list[int] = []
        self._active: set[int] = set()

    @property
    def active_count(self) -> int:
        return len(self._active) + len(self._pending_ingress)

    @property
    def num_pending_ingress(self) -> int:
        return len(self._pending_ingress)

    @property
    def num_pending_adds(self) -> int:
        return len(self._pending_add)

    def submit_requests_async(self, sequences):
        request_ids = []
        for sequence in sequences:
            request_id = sequence.seq_id
            request_ids.append(request_id)
            self.dispatch_times.append(self.clock())
            self._pending_ingress[request_id] = self.clock() + self.ack_delay_ns
        return tuple(request_ids)

    def poll_ingress_acks(self):
        ready = [
            request_id
            for request_id, ready_ns in self._pending_ingress.items()
            if ready_ns <= self.clock()
        ]
        for request_id in ready:
            self._pending_ingress.pop(request_id)
            self._pending_add.append(request_id)
        return tuple(IngressAck(request_id, 0, True) for request_id in ready)

    def poll_add_results(self):
        ready = tuple(self._pending_add)
        self._pending_add.clear()
        self._active.update(ready)
        self._finish.extend(ready)
        return tuple(AddResultEvent(request_id, 0, True) for request_id in ready)

    def poll_first_token_events(self):
        return ()

    def poll(self):
        ready = tuple(self._finish)
        self._finish.clear()
        self._active.difference_update(ready)
        return tuple(FinishEvent(request_id, 16, "FINISHED", 0) for request_id in ready)

    def is_finished(self):
        return (
            not self._pending_ingress
            and not self._pending_add
            and not self._active
            and not self._finish
        )


def _load_benchmark_module(monkeypatch):
    # Importing the benchmark should not initialize the GPU model stack in a
    # CPU-only synthetic control-plane test.
    monkeypatch.setitem(nanodeploy.__dict__, "LLM", object)
    monkeypatch.setitem(nanodeploy.__dict__, "SamplingParams", SamplingParams)
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "sp_ablation" / "bench_serving_overhead.py"
    spec = importlib.util.spec_from_file_location(
        "hierarchical_bench_serving_overhead", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_step_plot_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "plot_step_log_timeseries.py"
    spec = importlib.util.spec_from_file_location(
        "hierarchical_plot_step_log_timeseries", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hierarchical_diag_parser_uses_runtime_rank_count():
    plotter = _load_step_plot_module()
    rank_loads = {
        "0": {
            "rank_loads": [
                {
                    "global_rank": 0,
                    "master_batch_size": 3,
                    "free_blocks": 90,
                    "total_blocks": 100,
                },
                {
                    "global_rank": 1,
                    "master_batch_size": 5,
                    "free_blocks": 80,
                    "total_blocks": 100,
                },
            ]
        },
        "1": {
            "rank_loads": [
                {
                    "global_rank": 2,
                    "master_batch_size": 7,
                    "free_blocks": 70,
                    "total_blocks": 100,
                },
                {
                    "global_rank": 3,
                    "master_batch_size": 11,
                    "free_blocks": 60,
                    "total_blocks": 100,
                },
            ]
        },
    }
    payload = {
        "elapsed_s": 5.0,
        "hierarchical": {
            "waiting_requests": 2,
            "per_engine": rank_loads,
        },
        "hierarchical_interval": {"decode_itl_ms_mean": 12.5},
    }

    df, batch, free, used, metadata = plotter.extract_step_records(
        "[BENCH_DIAG] " + json.dumps(payload),
        num_kvcache_blocks_override=None,
        default_block_size=64,
    )

    assert len(df) == 1
    assert batch.shape == (1, 4)
    assert batch.tolist() == [[3.0, 5.0, 7.0, 11.0]]
    assert free.tolist() == [[90.0, 80.0, 70.0, 60.0]]
    assert used.tolist() == [[10.0, 20.0, 30.0, 40.0]]
    assert df.iloc[0]["total_batch_size"] == 26.0
    assert df.iloc[0]["itl_ms"] == 12.5
    assert metadata["num_kvcache_blocks"] == 100


def test_request_metric_summary_uses_successful_request_scalars(
    monkeypatch,
):
    benchmark = _load_benchmark_module(monkeypatch)
    records = [
        {
            "is_error": False,
            "e2e_ms": 100.0,
            "ttft_ms": 20.0,
            "tpot_with_queue_ms": 10.0,
            "dispatch_lag_ms": 1.0,
            "ingress_ack_latency_ms": 2.0,
            "add_accept_latency_ms": 3.0,
        },
        {
            "is_error": False,
            "e2e_ms": 300.0,
            "ttft_ms": 40.0,
            "tpot_with_queue_ms": 30.0,
            "dispatch_lag_ms": 3.0,
            "ingress_ack_latency_ms": 4.0,
            "add_accept_latency_ms": 5.0,
        },
        {
            "is_error": True,
            "e2e_ms": 999.0,
            "ttft_ms": None,
            "tpot_with_queue_ms": None,
            "dispatch_lag_ms": 999.0,
            "ingress_ack_latency_ms": 999.0,
            "add_accept_latency_ms": None,
        },
    ]

    summary = benchmark.build_request_metrics_summary(
        records, slo_threshold_ms=25.0
    )

    assert summary["total_requests"] == 3
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["tpot_with_queue_ms"]["mean"] == 20.0
    assert summary["tpot_with_queue_ms"]["p50"] == 20.0
    assert summary["tpot_with_queue_ms"]["p90"] == 28.0
    assert summary["tpot_with_queue_ms"]["p99"] == 29.8
    assert summary["goodput"] == {
        "metric": "tpot_with_queue_ms",
        "threshold_ms": 25.0,
        "successful_requests": 1,
        "eligible_requests": 2,
        "attainment_percent": 50.0,
    }


def test_hierarchical_itl_summary_uses_runtime_engines_and_token_weights(
    monkeypatch,
):
    benchmark = _load_benchmark_module(monkeypatch)
    summary = benchmark.build_hierarchical_itl_summary(
        (
            DecodeITLSample(0, 1, 0, 10.0, 2),
            DecodeITLSample(0, 1, 1, 20.0, 2),
            DecodeITLSample(1, 1, 0, 30.0, 1),
        )
    )

    assert summary["mean"] == 18.0
    assert summary["p50"] == 20.0
    assert summary["p90"] == 26.0
    assert summary["p99"] == 29.6
    assert summary["token_intervals"] == 5
    assert summary["quantum_samples"] == 3
    assert set(summary["per_engine"]) == {"0", "1"}
    assert summary["per_engine"]["0"]["mean"] == 15.0
    assert summary["per_engine"]["1"]["mean"] == 30.0


def test_delayed_ingress_ack_does_not_throttle_fixed_rate_dispatch(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()
    engine = DelayedAckEngine(clock, ack_delay_ms=500)
    num_requests = 200
    arrival_times = np.arange(1, num_requests + 1) / 20.0

    def requests():
        for _ in range(num_requests):
            yield (
                [1, 2, 3, 4],
                SamplingParams(
                    temperature=0.6,
                    ignore_eos=True,
                    max_tokens=16,
                ),
            )

    request_metrics_path = tmp_path / "requests.jsonl"
    summary_path = tmp_path / "summary.json"
    total_time, seq_map = benchmark.run_benchmark(
        engine,
        iter(requests()),
        arrival_times,
        num_requests,
        request_metrics_log_path=str(request_metrics_path),
        metrics_summary_path=str(summary_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    assert len(seq_map) == num_requests
    assert len(engine.dispatch_times) == num_requests
    assert engine.dispatch_times[-1] == 10_000_000_000
    assert (
        max(
            later - earlier
            for earlier, later in zip(
                engine.dispatch_times, engine.dispatch_times[1:], strict=False
            )
        )
        <= 50_000_001
    )
    assert (
        max(
            abs(actual - scheduled)
            for actual, scheduled in zip(
                engine.dispatch_times,
                range(50_000_000, 10_000_000_001, 50_000_000),
                strict=True,
            )
        )
        <= 1
    )
    assert 10.5 <= total_time <= 10.502

    records = [
        json.loads(line)
        for line in request_metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == num_requests
    assert all(record["status"] == "FINISHED" for record in records)
    assert all(record["actual_output_tokens"] == 16 for record in records)
    assert all(record["ttft_ms"] is None for record in records)
    assert all(
        31.25 <= record["tpot_with_queue_ms"] <= 31.376
        for record in records
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_requests"] == num_requests
    assert summary["successful_requests"] == num_requests
    assert summary["failed_requests"] == 0
    assert summary["tpot_with_queue_ms"]["p50"] == 31.25
    assert summary["goodput"]["attainment_percent"] == 100.0

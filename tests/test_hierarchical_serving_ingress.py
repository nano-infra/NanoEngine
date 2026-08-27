from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nanodeploy

from nanodeploy.engine.hierarchical_contract import (
    AddResultEvent,
    DecodeITLSample,
    FirstScheduleEvent,
    FinishEvent,
    IngressAck,
)
from nanodeploy.metrics import MetricsManager
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
        self.first_forward_to_terminal_ms = 500.0
        self.finish_global_capacity_queue_ms = 0.0
        self.generated_count = 16
        self.final_quantum_execute_ms = None

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

    def poll_first_schedule_events(self):
        return ()

    def hierarchical_itl_samples(self):
        return ()

    def hierarchical_metrics(self, *, refresh, include_per_engine):
        return {}

    def execution_boundary_metrics(self):
        return {"mode": "synthetic"}

    def poll(self):
        ready = tuple(self._finish)
        self._finish.clear()
        self._active.difference_update(ready)
        return tuple(
            FinishEvent(
                request_id,
                self.generated_count,
                "FINISHED",
                0,
                first_forward_to_terminal_ms=(
                    self.first_forward_to_terminal_ms
                ),
                global_capacity_queue_ms=(
                    self.finish_global_capacity_queue_ms
                ),
                final_quantum_execute_ms=(
                    self.final_quantum_execute_ms
                ),
            )
            for request_id in ready
        )

    def is_finished(self):
        return (
            not self._pending_ingress
            and not self._pending_add
            and not self._active
            and not self._finish
        )


class SyntheticCentralEngine:
    def __init__(self) -> None:
        self.config = SimpleNamespace(scheduler_arch="legacy_global")
        self.metrics_manager = MetricsManager()
        self._active = []
        self._pending_ingress: list[int] = []
        self._pending_add: list[int] = []

    @property
    def num_pending_ingress(self) -> int:
        return len(self._pending_ingress)

    @property
    def num_pending_adds(self) -> int:
        return len(self._pending_add)

    def submit_requests_async(self, sequences):
        request_ids = []
        for sequence in sequences:
            request_ids.append(sequence.seq_id)
            sequence.metric = self.metrics_manager.create_sequence_metric(
                sequence.seq_id, sequence.num_prompt_tokens
            )
            sequence.metric.record_arrival()
            sequence.metric.record_first_scheduled()
            self._active.append(sequence)
            self._pending_ingress.append(sequence.seq_id)
            self._pending_add.append(sequence.seq_id)
        return tuple(request_ids)

    def poll_ingress_acks(self):
        ready = tuple(self._pending_ingress)
        self._pending_ingress.clear()
        return tuple(IngressAck(request_id, 0, True) for request_id in ready)

    def poll_add_results(self):
        ready = tuple(self._pending_add)
        self._pending_add.clear()
        return tuple(
            AddResultEvent(request_id, 0, True) for request_id in ready
        )

    def poll_first_token_events(self):
        return ()

    def poll_first_schedule_events(self):
        return ()

    def step(self):
        time.sleep(0.002)
        outputs = []
        for sequence in self._active:
            sequence.metric.record_first_token()
            sequence.metric.record_step_tokens(16, 1.0)
            sequence.metric.record_completion()
            outputs.append((sequence.seq_id, [0] * 16))
        self._active.clear()
        return outputs, 16, len(outputs), 0.0, 0.0

    def is_finished(self):
        return (
            not self._active
            and not self._pending_ingress
            and not self._pending_add
        )

    def execution_boundary_metrics(self):
        return {"mode": "synthetic_central"}


class SyntheticCentralPartialQuantumEngine(SyntheticCentralEngine):
    def __init__(self) -> None:
        super().__init__()
        self._decode_steps = 0

    def step(self):
        self._decode_steps += 1
        if self._decode_steps == 1:
            for sequence in self._active:
                sequence.metric.record_first_token()
                sequence.metric.record_step_tokens(16, 1.0)
            return [], 16, len(self._active), 0.0, 0.0

        outputs = []
        for sequence in self._active:
            sequence.metric.record_step_tokens(1, 10.0)
            sequence.metric.arrival_time = 0.0
            sequence.metric.first_scheduled_time = 0.1
            sequence.metric.completion_time = 0.42
            outputs.append((sequence.seq_id, [0] * 17))
        self._active.clear()
        return outputs, 1, len(outputs), 0.0, 0.0


def _load_benchmark_module(monkeypatch, script_dir="sp_ablation"):
    # Importing the benchmark should not initialize the GPU model stack in a
    # CPU-only synthetic control-plane test.
    monkeypatch.setitem(nanodeploy.__dict__, "LLM", object)
    monkeypatch.setitem(nanodeploy.__dict__, "SamplingParams", SamplingParams)
    root = Path(__file__).resolve().parents[1]
    benchmark_paths = {
        "sp_ablation": (
            root / "scripts" / "sp_ablation" / "bench_serving_overhead.py"
        ),
        "issue003": (
            root / "scripts" / "issue003" / "bench_serving_overhead.py"
        ),
        "example": root / "examples" / "bench_serving.py",
        "example_overhead": root / "examples" / "bench_serving_overhead.py",
    }
    path = benchmark_paths[script_dir]
    spec = importlib.util.spec_from_file_location(
        f"{script_dir}_bench_serving_overhead", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quantum_log_cli_accepts_both_scheduler_architectures(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    for scheduler_arch in ("legacy_global", "hierarchical"):
        quantum_path = tmp_path / f"{scheduler_arch}.jsonl"
        monkeypatch.setattr(
            "sys.argv",
            [
                "bench_serving_overhead.py",
                "--scheduler-arch",
                scheduler_arch,
                "--quantum-log-path",
                str(quantum_path),
            ],
        )
        args = benchmark.parse_args()
        assert args.quantum_log_path == str(quantum_path)


def test_uniform_random_moe_routing_cli_reaches_engine(monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)
    monkeypatch.setattr("sys.argv", ["bench_serving_overhead.py"])
    defaults = benchmark.parse_args()
    assert defaults.moe_routing_simulation_strategy == "perfect_eplb"
    assert defaults.moe_routing_seed == 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_serving_overhead.py",
            "--moe-routing-simulation-strategy",
            "uniform_random",
            "--moe-routing-seed",
            "17",
        ],
    )
    args = benchmark.parse_args()
    captured = {}

    def fake_llm(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return SimpleNamespace(
            config=SimpleNamespace(
                hierarchical_quantum_diagnostics=False
            ),
            reset_execution_boundary_metrics=lambda: None,
        )

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "LLM", fake_llm)
    monkeypatch.setattr(benchmark, "print_model_config", lambda engine: None)
    monkeypatch.setattr(
        benchmark,
        "run_warmup",
        lambda engine, max_num_seqs, world_size: None,
    )
    monkeypatch.setattr(
        benchmark, "get_dataset_generator", lambda parsed: iter(())
    )
    monkeypatch.setattr(
        benchmark,
        "generate_arrival_times",
        lambda num_requests, request_rate, burstiness: (),
    )
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda *call_args, **call_kwargs: (0.0, {}, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "calculate_and_print_metrics",
        lambda *call_args, **call_kwargs: None,
    )

    benchmark.main()

    assert captured["moe_routing_simulation_strategy"] == "uniform_random"
    assert captured["seed"] == 17
    assert captured["perfect_eplb"] is False


def test_csv_dataset_skips_requests_above_total_token_limit(
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "prompt_len,output_len\n"
        "6,4\n"
        "7,4\n"
        "8,2\n",
        encoding="utf-8",
    )

    for script_dir in (
        "sp_ablation",
        "issue003",
        "example",
        "example_overhead",
    ):
        benchmark = _load_benchmark_module(monkeypatch, script_dir)
        assert benchmark.DEFAULT_MAX_REQUEST_TOKENS == 910_000
        args = SimpleNamespace(
            dataset="csv",
            csv_path=str(csv_path),
            max_input_len=None,
            max_request_tokens=10,
            max_model_len=100,
            num_requests=4,
        )

        requests = list(benchmark.get_dataset_generator(args))

        assert [len(prompt) for prompt, _ in requests] == [6, 8, 6, 8]
        assert [params.max_tokens for _, params in requests] == [4, 2, 4, 2]
        assert all(
            len(prompt) + params.max_tokens <= args.max_request_tokens
            for prompt, params in requests
        )


def test_csv_dataset_reports_when_total_token_filter_removes_every_row(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    csv_path = tmp_path / "too_long.csv"
    csv_path.write_text(
        "prompt_len,output_len\n"
        "8,4\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset="csv",
        csv_path=str(csv_path),
        max_input_len=None,
        max_request_tokens=10,
        max_model_len=100,
        num_requests=1,
    )

    try:
        list(benchmark.get_dataset_generator(args))
    except ValueError as exc:
        assert str(exc) == "CSV dataset has no rows after applying length filters"
    else:
        raise AssertionError("expected an empty filtered CSV dataset to fail")


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
            "dispatch_tpot_ms": 11.0,
            "dispatch_lag_ms": 1.0,
            "ingress_ack_latency_ms": 2.0,
            "add_accept_latency_ms": 3.0,
        },
        {
            "is_error": False,
            "e2e_ms": 300.0,
            "ttft_ms": 40.0,
            "tpot_with_queue_ms": 30.0,
            "dispatch_tpot_ms": 31.0,
            "dispatch_lag_ms": 3.0,
            "ingress_ack_latency_ms": 4.0,
            "add_accept_latency_ms": 5.0,
        },
        {
            "is_error": True,
            "e2e_ms": 999.0,
            "ttft_ms": None,
            "tpot_with_queue_ms": None,
            "dispatch_tpot_ms": None,
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
    assert summary["dispatch_tpot_ms"]["mean"] == 21.0
    assert "LocalEngine ingress" in summary[
        "dispatch_tpot_ms_definition"
    ]
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
    total_time, seq_map, metrics_summary = benchmark.run_benchmark(
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
    assert all(
        record["tpot_with_queue_source"]
        == "hierarchical_real_token_execution_boundary"
        for record in records
    )
    assert all(
        record["first_forward_to_terminal_ms"] == 500.0
        for record in records
    )
    assert all(
        record["first_forward_to_terminal_real_token_ms"] == 500.0
        for record in records
    )
    assert all(
        record["final_quantum_unused_decode_ms"] == 0.0
        for record in records
    )
    assert all(
        31.25 <= record["dispatch_normalized_latency_ms"] <= 31.376
        for record in records
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == metrics_summary
    assert summary["total_requests"] == num_requests
    assert summary["successful_requests"] == num_requests
    assert summary["failed_requests"] == 0
    assert summary["tpot_with_queue_ms"]["p50"] == 31.25
    assert summary["goodput"]["attainment_percent"] == 100.0


def test_planned_add_result_records_bootstrap_ttft_and_commit_timing(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()

    class PlannedReceiptEngine(DelayedAckEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._first_schedule_emitted: set[int] = set()
            self.first_forward_to_terminal_ms = 470.0
            self.finish_global_capacity_queue_ms = 30.0

        def poll_ingress_acks(self):
            return tuple(
                IngressAck(
                    request_id=ack.request_id,
                    engine_id=ack.engine_id,
                    enqueued=ack.enqueued,
                    reason=ack.reason,
                    ingress_version=1,
                    router_pending_ms=25.0,
                    admission_rpc_ms=450.0,
                )
                for ack in super().poll_ingress_acks()
            )

        def poll_add_results(self):
            return tuple(
                AddResultEvent(
                    request_id=result.request_id,
                    engine_id=result.engine_id,
                    accepted=result.accepted,
                    reason=result.reason,
                    admission_version=1,
                    local_planned_queue_ms=100.0,
                    local_admission_ms=50.0,
                )
                for result in super().poll_add_results()
            )

        def poll_first_schedule_events(self):
            ready = self._active.difference(
                self._first_schedule_emitted
            )
            self._first_schedule_emitted.update(ready)
            return tuple(
                FirstScheduleEvent(
                    request_id=request_id,
                    engine_id=0,
                    local_scheduler_queue_ms=12.5,
                    global_capacity_queue_ms=30.0,
                )
                for request_id in ready
            )

    engine = PlannedReceiptEngine(clock, ack_delay_ms=500)
    request_metrics_path = tmp_path / "requests.jsonl"
    summary_path = tmp_path / "summary.json"
    benchmark.run_benchmark(
        engine,
        iter(
            (
                (
                    [1, 2, 3, 4],
                    SamplingParams(
                        temperature=0.6,
                        ignore_eos=True,
                        max_tokens=16,
                    ),
                ),
            )
        ),
        np.asarray([0.05]),
        1,
        request_metrics_log_path=str(request_metrics_path),
        metrics_summary_path=str(summary_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    record = json.loads(
        request_metrics_path.read_text(encoding="utf-8").strip()
    )
    assert record["schema_version"] == 3
    assert record["ttft_source"] == "planned_add_result"
    assert record["ttft_ms"] == 500.0
    assert record["bootstrap_ttft_ms"] == 500.0
    assert record["model_ttft_ms"] is None
    assert record["router_pending_ms"] == 25.0
    assert record["admission_rpc_ms"] == 450.0
    assert record["local_command_queue_ms"] is None
    assert record["local_admission_ms"] is None
    assert record["staged_queue_ms"] == 100.0
    assert record["planned_commit_ms"] == 50.0
    assert record["admission_rpc_residual_ms"] is None
    assert record["frontend_ack_overhead_ms"] == 25.0
    assert record["first_schedule_latency_ms"] == 42.5
    assert (
        record["first_schedule_latency_source"]
        == "hierarchical_first_forward_event"
    )
    assert record["global_capacity_queue_ms"] == 30.0
    assert record["local_scheduler_queue_ms"] == 12.5
    assert record["first_forward_to_terminal_ms"] == 470.0
    assert record["tpot_with_queue_ms"] == 31.25
    assert (
        record["tpot_with_queue_source"]
        == "hierarchical_real_token_execution_boundary"
    )
    assert record["first_forward_to_terminal_real_token_ms"] == 470.0
    assert record["final_quantum_real_tokens"] == 16
    assert record["final_quantum_unused_decode_ms"] == 0.0
    assert record["dispatch_normalized_latency_ms"] == 31.25

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["ttft_ms"]["mean"] == 500.0
    assert summary["bootstrap_ttft_ms"]["mean"] == 500.0
    assert summary["model_ttft_ms"] is None
    assert summary["router_pending_ms"]["mean"] == 25.0
    assert summary["admission_rpc_ms"]["mean"] == 450.0
    assert summary["local_command_queue_ms"] is None
    assert summary["local_admission_ms"] is None
    assert summary["staged_queue_ms"]["mean"] == 100.0
    assert summary["planned_commit_ms"]["mean"] == 50.0
    assert summary["admission_rpc_residual_ms"] is None
    assert summary["frontend_ack_overhead_ms"]["mean"] == 25.0
    assert summary["first_schedule_latency_ms"]["mean"] == 42.5
    assert summary["global_capacity_queue_ms"]["mean"] == 30.0
    assert summary["local_scheduler_queue_ms"]["mean"] == 12.5
    assert summary["first_forward_to_terminal_ms"]["mean"] == 470.0
    assert (
        summary["first_forward_to_terminal_real_token_ms"]["mean"]
        == 470.0
    )
    assert summary["final_quantum_unused_decode_ms"]["mean"] == 0.0
    assert summary["tpot_with_queue_ms"]["mean"] == 31.25
    assert summary["dispatch_normalized_latency_ms"]["mean"] == 31.25


def test_hierarchical_tpot_excludes_partial_final_quantum_tail(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()
    engine = DelayedAckEngine(clock, ack_delay_ms=0)
    engine.generated_count = 17
    engine.first_forward_to_terminal_ms = 320.0
    engine.finish_global_capacity_queue_ms = 20.0
    engine.final_quantum_execute_ms = 160.0
    request_metrics_path = tmp_path / "partial_hierarchical.jsonl"

    benchmark.run_benchmark(
        engine,
        iter(
            (
                (
                    [1, 2, 3, 4],
                    SamplingParams(
                        temperature=0.6,
                        ignore_eos=True,
                        max_tokens=17,
                    ),
                ),
            )
        ),
        np.asarray([0.05]),
        1,
        request_metrics_log_path=str(request_metrics_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    record = json.loads(
        request_metrics_path.read_text(encoding="utf-8").strip()
    )
    assert record["first_forward_to_terminal_ms"] == 320.0
    assert record["final_quantum_real_tokens"] == 1
    assert record["final_quantum_unused_decode_ms"] == 150.0
    assert record["first_forward_to_terminal_real_token_ms"] == 170.0
    assert record["global_capacity_queue_ms"] == 20.0
    assert record["tpot_with_queue_ms"] == round(190.0 / 17, 6)


def test_centralized_tpot_excludes_partial_final_quantum_tail(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()
    engine = SyntheticCentralPartialQuantumEngine()
    request_metrics_path = tmp_path / "partial_central.jsonl"

    benchmark.run_benchmark(
        engine,
        iter(
            (
                (
                    [1, 2, 3, 4],
                    SamplingParams(
                        temperature=0.6,
                        ignore_eos=True,
                        max_tokens=17,
                    ),
                ),
            )
        ),
        np.asarray([0.05]),
        1,
        request_metrics_log_path=str(request_metrics_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    record = json.loads(
        request_metrics_path.read_text(encoding="utf-8").strip()
    )
    assert record["first_forward_to_terminal_ms"] == 320.0
    assert record["final_quantum_real_tokens"] == 1
    assert record["final_quantum_unused_decode_ms"] == 150.0
    assert record["first_forward_to_terminal_real_token_ms"] == 170.0
    assert record["global_capacity_queue_ms"] == 100.0
    assert record["tpot_with_queue_ms"] == round(270.0 / 17, 6)


def test_centralized_default_tpot_uses_sequence_scheduler_boundary(
    monkeypatch,
    tmp_path,
    capsys,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()
    engine = SyntheticCentralEngine()
    request_metrics_path = tmp_path / "central_requests.jsonl"

    _, seq_map, metrics_summary = benchmark.run_benchmark(
        engine,
        iter(
            (
                (
                    [1, 2, 3, 4],
                    SamplingParams(
                        temperature=0.6,
                        ignore_eos=True,
                        max_tokens=16,
                    ),
                ),
            )
        ),
        np.asarray([0.05]),
        1,
        request_metrics_log_path=str(request_metrics_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    record = json.loads(
        request_metrics_path.read_text(encoding="utf-8").strip()
    )
    sequence = next(iter(seq_map.values()))
    expected_tpot = round(
        float(sequence.metric.avg_tpot_with_queueing), 6
    )
    expected_execution_ms = round(
        float(sequence.metric.avg_tpot_wo_queueing) * 16, 6
    )

    assert record["tpot_with_queue_ms"] == expected_tpot
    assert (
        record["tpot_with_queue_source"]
        == "sequence_metric_real_token_execution_boundary"
    )
    assert (
        record["first_forward_to_terminal_ms"]
        == expected_execution_ms
    )
    assert (
        record["first_forward_to_terminal_real_token_ms"]
        == expected_execution_ms
    )
    assert record["final_quantum_real_tokens"] == 16
    assert record["final_quantum_unused_decode_ms"] == 0.0
    assert record["global_capacity_queue_ms"] is not None
    assert record["local_scheduler_queue_ms"] == 0.0
    assert record["dispatch_normalized_latency_ms"] == 0.0
    assert metrics_summary["tpot_with_queue_ms"]["mean"] == round(
        expected_tpot, 3
    )
    benchmark.calculate_and_print_metrics(
        1.0,
        seq_map,
        1,
        metrics_summary,
    )
    output = capsys.readouterr().out
    assert "real-token execution-boundary time" in output
    assert "unused final-quantum slots excluded" in output
    assert "Legacy Dispatch-Normalized Latency" in output
    assert "GPU-capacity queue only" in output


def test_benchmark_persists_compact_quantum_diagnostics(
    monkeypatch,
    tmp_path,
):
    benchmark = _load_benchmark_module(monkeypatch)
    clock = FakeClock()

    class DiagnosticEngine(DelayedAckEngine):
        def __init__(self):
            super().__init__(clock, ack_delay_ms=1)
            self.quantum_samples = [
                {
                    "schema_version": 1,
                    "engine_id": 0,
                    "wave_id": 1,
                    "quantum_id": 0,
                    "started_at_unix_s": time.time(),
                    "execute_ms": 800.0,
                    "executor": {
                        "gpu_loop_ms_max": 790.0,
                    },
                }
            ]

        def drain_quantum_diagnostics(self):
            samples = tuple(self.quantum_samples)
            self.quantum_samples.clear()
            return samples

        def hierarchical_metrics(self, *, refresh, include_per_engine):
            assert refresh
            assert include_per_engine
            return {
                "decode_quantum_count": 1,
                "schedule_latency_ms_total": 2.5,
                "per_engine": {
                    "0": {
                        "decode_quantum_count": 1,
                        "rank_loads": [],
                    }
                },
            }

    engine = DiagnosticEngine()

    def requests():
        yield (
            [1, 2, 3, 4],
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=16,
            ),
        )

    quantum_path = tmp_path / "quantums.jsonl"
    summary_path = tmp_path / "summary.json"
    benchmark.run_benchmark(
        engine,
        iter(requests()),
        np.asarray([0.001]),
        1,
        quantum_log_path=str(quantum_path),
        metrics_summary_path=str(summary_path),
        clock_ns=clock,
        sleep_fn=clock.sleep,
        show_progress=False,
    )

    records = [
        json.loads(line)
        for line in quantum_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 1
    assert records[0]["engine_id"] == 0
    assert records[0]["executor"]["gpu_loop_ms_max"] == 790.0
    assert "benchmark_elapsed_s" in records[0]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["quantum_diagnostic_samples"] == 1
    assert (
        summary["quantum_diagnostics_jsonl"]
        == str(quantum_path)
    )
    assert summary["hierarchical_metrics"] == {
        "decode_quantum_count": 1,
        "schedule_latency_ms_total": 2.5,
        "per_engine": {
            "0": {
                "decode_quantum_count": 1,
                "rank_loads": [],
            }
        },
    }

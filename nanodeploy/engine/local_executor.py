from __future__ import annotations

import time
from time import perf_counter
from typing import Any, Iterable

import ray

from nanodeploy.config import Config
from nanodeploy.endpoint.rpc_endpoint import RPCServerEndpoint
from nanodeploy.engine.hierarchical_contract import (
    HIERARCHICAL_LOOP_COUNT,
    LocalDecodeBatch,
    WorkerDecodeResult,
)
from nanodeploy.engine.execution_boundary import ExecutionBoundaryRecorder
from nanodeploy.engine.topology import EngineTopology


class LocalExecutor:
    """Drives only the workers owned by one attention-DP engine."""

    _SERVER_BUFFER_BYTES = 8 * 32_000_000

    def __init__(
        self,
        config: Config,
        topology: EngineTopology,
        workers: Iterable[Any],
    ) -> None:
        self.config = config
        self.topology = topology
        self.workers = tuple(workers)
        if len(self.workers) != topology.world_size:
            raise ValueError(
                f"engine {topology.engine_id} expected {topology.world_size} "
                f"workers, got {len(self.workers)}"
            )
        self.endpoint = RPCServerEndpoint(
            self._SERVER_BUFFER_BYTES,
            topology.world_size,
            topology.attention_sp,
            topology.attention_tp,
            config.optimize_decode_block_table,
        )
        self.last_execution_traces: tuple[dict[str, Any], ...] = ()
        self._execution_trace_history: list[dict[str, Any]] = []
        self.result_fastpath_enabled = bool(
            getattr(config, "hierarchical_result_fastpath", False)
        )
        self.quantum_diagnostics_enabled = bool(
            getattr(config, "hierarchical_quantum_diagnostics", False)
        )
        self.last_quantum_diagnostic: dict[str, Any] | None = None
        self.ray_get_latency_ms_total = 0.0
        self.ray_get_latency_ms_max = 0.0
        self.result_rebuild_latency_ms_total = 0.0
        self.result_rebuild_latency_ms_max = 0.0
        self.result_rebuild_sample_count = 0
        self.result_index_latency_ms_total = 0.0
        self.result_validate_latency_ms_total = 0.0
        self.result_pack_latency_ms_total = 0.0
        self._execution_boundary = ExecutionBoundaryRecorder()

    def initialize_endpoint(self, timeout: float) -> None:
        server_info = self.endpoint.init_server_endpoint()
        # Ray 2.51 can deserialize a nested actor handle with generic
        # ``(**kwargs)`` method metadata. Named arguments remain valid for
        # both that handle and the concrete ModelRunner signature.
        try:
            futures = [
                worker.init_rpc_endpoint.remote(server_info=server_info)
                for worker in self.workers
            ]
        except TypeError as exc:
            signatures = tuple(
                repr(
                    getattr(worker, "_ray_method_signatures", {}).get(
                        "init_rpc_endpoint"
                    )
                )
                for worker in self.workers
            )
            raise TypeError(
                "Ray rejected ModelRunner.init_rpc_endpoint; "
                f"worker handle signatures={signatures}"
            ) from exc
        client_info = ray.get(futures, timeout=timeout)
        self.endpoint.connect(client_info)

    def worker_identities(self, timeout: float) -> tuple[dict[str, Any], ...]:
        return tuple(
            ray.get(
                [worker.get_worker_identity.remote() for worker in self.workers],
                timeout=timeout,
            )
        )

    def run(
        self, batch: LocalDecodeBatch, timeout: float
    ) -> list[WorkerDecodeResult]:
        if batch.engine_id != self.topology.engine_id:
            raise ValueError("LocalDecodeBatch belongs to another engine")
        executor_begin = perf_counter()
        ordered_sequences = [
            batch.per_rank_sequences[global_rank]
            for global_rank in self.topology.global_ranks
        ]
        send_timestamp = time.time()
        futures = []
        mastered_by_rank: dict[int, list[Any]] = {}
        for global_rank, worker in zip(
            self.topology.global_ranks, self.workers, strict=True
        ):
            mastered = [
                sequence
                for sequence in batch.per_rank_sequences[global_rank]
                if sequence.block_ctx().master_sp_idx
                == self.topology.engine_local_rank(global_rank)
            ]
            mastered_by_rank[global_rank] = mastered
            trace_context = None
            if self.config.hierarchical_execution_trace:
                trace_context = {
                    "wave_id": batch.wave_id,
                    "quantum_id": batch.quantum_id,
                    "global_rank": global_rank,
                    "real_batch_size": len(
                        batch.expected_request_ids(global_rank)
                    ),
                    "control_dummy_count": sum(
                        batch.is_control_dummy(sequence)
                        for sequence in mastered
                    ),
                    "batch_kind": (
                        "real_or_mixed"
                        if batch.engine_has_real
                        else "all_control_dummy"
                    ),
                }
            futures.append(
                # Keep this call keyword-only for the same nested actor-handle
                # compatibility required by initialize_endpoint().
                worker.run.remote(
                    dp_seqs=[],
                    is_prefill=False,
                    enable_rpc=True,
                    send_timestamp=send_timestamp,
                    hierarchical_trace=trace_context,
                    hierarchical_quantum_diagnostics=(
                        self.quantum_diagnostics_enabled
                    ),
                )
            )

        submit_latency_ms = (perf_counter() - executor_begin) * 1000
        send_begin = perf_counter()
        self.endpoint.send_seqs(ordered_sequences, is_prefill=False)
        send_seqs_latency_ms = (perf_counter() - send_begin) * 1000
        ray_get_begin = perf_counter()
        raw_results = ray.get(futures, timeout=timeout)
        ray_get_end = perf_counter()
        recv_timestamp = time.time()
        ray_get_latency_ms = (ray_get_end - ray_get_begin) * 1000
        self.ray_get_latency_ms_total += ray_get_latency_ms
        self.ray_get_latency_ms_max = max(
            self.ray_get_latency_ms_max, ray_get_latency_ms
        )
        expected_result_len = (
            2
            + int(self.config.hierarchical_execution_trace)
            + int(self.quantum_diagnostics_enabled)
        )
        if any(
            not isinstance(raw, tuple) or len(raw) != expected_result_len
            for raw in raw_results
        ):
            raise RuntimeError(
                "hierarchical worker returned an invalid result envelope: "
                f"expected tuple length {expected_result_len}"
            )
        worker_end_times = tuple(float(raw[1]) for raw in raw_results)
        last_worker_end = max(worker_end_times)
        boundary_metrics = {
            "actor_submit_latency_ms": submit_latency_ms,
            "send_seqs_latency_ms": send_seqs_latency_ms,
            "ray_get_latency_ms": ray_get_latency_ms,
            "executor_until_ray_get_ms": (
                ray_get_end - executor_begin
            )
            * 1000,
            "worker_observed_critical_ms": (
                last_worker_end - send_timestamp
            )
            * 1000,
            "worker_finish_to_ray_get_ms": (
                recv_timestamp - last_worker_end
            )
            * 1000,
            "worker_finish_skew_ms": (
                last_worker_end - min(worker_end_times)
            )
            * 1000,
        }
        self._execution_boundary.record(boundary_metrics)

        rebuild_begin = perf_counter()
        result_index_latency_ms = 0.0
        result_validate_latency_ms = 0.0
        result_pack_latency_ms = 0.0
        results: list[WorkerDecodeResult] = []
        traces: list[dict[str, Any]] = []
        worker_diagnostics: list[dict[str, Any]] = []
        for global_rank, raw in zip(
            self.topology.global_ranks, raw_results, strict=True
        ):
            token_rows, _worker_end_time = raw[:2]
            extra_index = 2
            trace = None
            if self.config.hierarchical_execution_trace:
                trace = raw[extra_index]
                extra_index += 1
            if self.quantum_diagnostics_enabled:
                worker_diagnostic = raw[extra_index]
                if (
                    not isinstance(worker_diagnostic, dict)
                    or worker_diagnostic.get("global_rank") != global_rank
                ):
                    raise RuntimeError(
                        f"hierarchical worker {global_rank} returned an "
                        "invalid quantum diagnostic"
                    )
                worker_diagnostics.append(dict(worker_diagnostic))
            mastered_sequences = mastered_by_rank[global_rank]
            if len(token_rows) != len(mastered_sequences):
                raise RuntimeError(
                    f"worker {global_rank} token row mismatch: "
                    f"expected={len(mastered_sequences)}, got={len(token_rows)}"
                )
            expected = batch.expected_request_ids(global_rank)
            index_begin = perf_counter()
            if self.result_fastpath_enabled:
                actual_request_ids: list[int] = []
                sampled_token_ids: list[tuple[int, ...]] = []
                for sequence, tokens in zip(
                    mastered_sequences, token_rows, strict=True
                ):
                    if batch.is_control_dummy(sequence):
                        continue
                    actual_request_ids.append(sequence.seq_id)
                    sampled_token_ids.append(tuple(tokens))
                actual_request_order = tuple(actual_request_ids)
                packed_token_ids = tuple(sampled_token_ids)
            else:
                token_by_request = {
                    sequence.seq_id: tuple(tokens)
                    for sequence, tokens in zip(
                        mastered_sequences, token_rows, strict=True
                    )
                    if not batch.is_control_dummy(sequence)
                }
                actual_request_order = tuple(token_by_request)
                packed_token_ids = ()
            result_index_latency_ms += (
                perf_counter() - index_begin
            ) * 1000

            validate_begin = perf_counter()
            if self.result_fastpath_enabled:
                valid_result = actual_request_order == expected
            else:
                valid_result = set(actual_request_order) == set(expected)
            result_validate_latency_ms += (
                perf_counter() - validate_begin
            ) * 1000
            if not valid_result:
                raise RuntimeError(
                    f"worker {global_rank} mastered request mismatch: "
                    f"expected={expected}, got={actual_request_order}"
                )
            if trace is not None:
                expected_trace_header = (
                    batch.wave_id,
                    batch.quantum_id,
                    global_rank,
                    HIERARCHICAL_LOOP_COUNT,
                    len(expected),
                    sum(
                        batch.is_control_dummy(sequence)
                        for sequence in mastered_sequences
                    ),
                    (
                        "real_or_mixed"
                        if batch.engine_has_real
                        else "all_control_dummy"
                    ),
                )
                actual_trace_header = (
                    trace.get("wave_id"),
                    trace.get("quantum_id"),
                    trace.get("global_rank"),
                    trace.get("forward_count"),
                    trace.get("real_batch_size"),
                    trace.get("control_dummy_count"),
                    trace.get("batch_kind"),
                )
                if actual_trace_header != expected_trace_header:
                    raise RuntimeError(
                        f"worker {global_rank} returned an invalid execution trace"
                    )
                inner_loops = tuple(
                    item.get("inner_loop_idx")
                    for item in trace.get("forwards", ())
                )
                if inner_loops != tuple(range(HIERARCHICAL_LOOP_COUNT)):
                    raise RuntimeError(
                        f"worker {global_rank} trace is missing inner forwards"
                    )
                traces.append(trace)
            pack_begin = perf_counter()
            if not self.result_fastpath_enabled:
                packed_token_ids = tuple(
                    token_by_request[request_id]
                    for request_id in expected
                )
            results.append(
                WorkerDecodeResult(
                    wave_id=batch.wave_id,
                    quantum_id=batch.quantum_id,
                    global_rank=global_rank,
                    forward_count=HIERARCHICAL_LOOP_COUNT,
                    mastered_request_ids=expected,
                    sampled_token_ids=packed_token_ids,
                )
            )
            result_pack_latency_ms += (
                perf_counter() - pack_begin
            ) * 1000

        if traces:
            sp_branches = {
                tuple(
                    bool(item.get("use_sp_a2a"))
                    for item in trace["forwards"]
                )
                for trace in traces
            }
            if len(sp_branches) != 1:
                raise RuntimeError(
                    f"engine {self.topology.engine_id} workers took "
                    "inconsistent SP branches"
                )
        self.last_execution_traces = tuple(traces)
        self._execution_trace_history.extend(traces)
        result_rebuild_latency_ms = (perf_counter() - rebuild_begin) * 1000
        self.result_rebuild_latency_ms_total += result_rebuild_latency_ms
        self.result_rebuild_latency_ms_max = max(
            self.result_rebuild_latency_ms_max,
            result_rebuild_latency_ms,
        )
        self.result_rebuild_sample_count += 1
        self.result_index_latency_ms_total += result_index_latency_ms
        self.result_validate_latency_ms_total += (
            result_validate_latency_ms
        )
        self.result_pack_latency_ms_total += result_pack_latency_ms
        if self.quantum_diagnostics_enabled:
            critical_worker = max(
                worker_diagnostics,
                key=lambda item: float(item["worker_total_ms"]),
            )
            gpu_loop_values = [
                float(item["gpu_loop_ms"])
                for item in worker_diagnostics
                if item.get("gpu_loop_ms") is not None
            ]
            self.last_quantum_diagnostic = {
                "engine_id": batch.engine_id,
                "wave_id": batch.wave_id,
                "quantum_id": batch.quantum_id,
                **boundary_metrics,
                "result_rebuild_ms": result_rebuild_latency_ms,
                "result_index_ms": result_index_latency_ms,
                "result_validate_ms": result_validate_latency_ms,
                "result_pack_ms": result_pack_latency_ms,
                "critical_worker_global_rank": int(
                    critical_worker["global_rank"]
                ),
                "worker_total_ms_min": min(
                    float(item["worker_total_ms"])
                    for item in worker_diagnostics
                ),
                "worker_total_ms_max": max(
                    float(item["worker_total_ms"])
                    for item in worker_diagnostics
                ),
                "gpu_loop_ms_min": (
                    min(gpu_loop_values) if gpu_loop_values else None
                ),
                "gpu_loop_ms_max": (
                    max(gpu_loop_values) if gpu_loop_values else None
                ),
                "worker_rank_timings": tuple(worker_diagnostics),
            }
        else:
            self.last_quantum_diagnostic = None
        return results

    def execution_boundary_metrics(self) -> dict[str, float | int]:
        return self._execution_boundary.snapshot()

    def reset_execution_boundary_metrics(self) -> None:
        self._execution_boundary.reset()

    def drain_execution_traces(self) -> tuple[dict[str, Any], ...]:
        traces = tuple(self._execution_trace_history)
        self._execution_trace_history.clear()
        return traces

from __future__ import annotations

import time
from typing import Any, Iterable

import ray

from nanodeploy.config import Config
from nanodeploy.endpoint.rpc_endpoint import RPCServerEndpoint
from nanodeploy.engine.hierarchical_contract import (
    HIERARCHICAL_LOOP_COUNT,
    LocalDecodeBatch,
    WorkerDecodeResult,
)
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

    def initialize_endpoint(self, timeout: float) -> None:
        server_info = self.endpoint.init_server_endpoint()
        futures = [
            worker.init_rpc_endpoint.remote(server_info)
            for worker in self.workers
        ]
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
                worker.run.remote(
                    [],
                    False,
                    True,
                    send_timestamp,
                    trace_context,
                )
            )

        self.endpoint.send_seqs(ordered_sequences, is_prefill=False)
        raw_results = ray.get(futures, timeout=timeout)

        results: list[WorkerDecodeResult] = []
        traces: list[dict[str, Any]] = []
        for global_rank, raw in zip(
            self.topology.global_ranks, raw_results, strict=True
        ):
            expected_result_len = (
                3 if self.config.hierarchical_execution_trace else 2
            )
            if not isinstance(raw, tuple) or len(raw) != expected_result_len:
                raise RuntimeError(
                    f"hierarchical worker {global_rank} returned an invalid result"
                )
            token_rows, _worker_end_time = raw[:2]
            trace = raw[2] if len(raw) == 3 else None
            mastered_sequences = mastered_by_rank[global_rank]
            if len(token_rows) != len(mastered_sequences):
                raise RuntimeError(
                    f"worker {global_rank} token row mismatch: "
                    f"expected={len(mastered_sequences)}, got={len(token_rows)}"
                )
            token_by_request = {
                sequence.seq_id: tuple(tokens)
                for sequence, tokens in zip(
                    mastered_sequences, token_rows, strict=True
                )
                if not batch.is_control_dummy(sequence)
            }
            expected = batch.expected_request_ids(global_rank)
            if set(token_by_request) != set(expected):
                raise RuntimeError(
                    f"worker {global_rank} mastered request mismatch: "
                    f"expected={expected}, got={tuple(token_by_request)}"
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
            results.append(
                WorkerDecodeResult(
                    wave_id=batch.wave_id,
                    quantum_id=batch.quantum_id,
                    global_rank=global_rank,
                    forward_count=HIERARCHICAL_LOOP_COUNT,
                    mastered_request_ids=expected,
                    sampled_token_ids=tuple(
                        token_by_request[request_id]
                        for request_id in expected
                    ),
                )
            )

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
        return results

    def drain_execution_traces(self) -> tuple[dict[str, Any], ...]:
        traces = tuple(self._execution_trace_history)
        self._execution_trace_history.clear()
        return traces

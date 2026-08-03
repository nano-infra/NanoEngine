import atexit
import os
import time
import uuid
from collections import deque
from dataclasses import fields
from time import perf_counter
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy._cpp import BlockContextSlot, SequenceStatus
from nanodeploy.config import Config
from nanodeploy.engine.deployment_manager import DeploymentManager
from nanodeploy.engine.hierarchical_contract import (
    AddResult,
    AddResultEvent,
    FirstScheduleEvent,
    FirstTokenEvent,
    FinishEvent,
    IngressAck,
)
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.metrics import MetricsManager
from nanodeploy.router.admission_planner import AdmissionPlannerConfig
from nanodeploy.router.request_router import RequestRouter

logger = get_logger()


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LLMEngine:
    def __init__(self, model, **kwargs):
        if "scheduler_mode" in kwargs:
            raise TypeError(
                "scheduler_mode was removed; use "
                "scheduler_arch='legacy_global' or 'hierarchical'"
            )
        self.engine_id = str(uuid.uuid4())

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.config = config
        self.config.engine_id = self.engine_id
        self.ps = []
        self.events = []
        self.metrics_manager = MetricsManager()
        self._frontend_ingress_acks: deque[IngressAck] = deque()
        self._frontend_add_results: deque[AddResultEvent] = deque()
        self._frontend_first_token_events: deque[
            FirstTokenEvent
        ] = deque()
        self._frontend_first_schedule_events: deque[
            FirstScheduleEvent
        ] = deque()
        self._frontend_finish_events: deque[FinishEvent] = deque()
        self._frontend_cycle_active = False
        self._closed = False
        self.log_decode_step_detail = _env_flag_enabled(
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL", default=False
        )

        if config.scheduler_arch == "hierarchical":
            self.tokenizer = None
            config.eos = getattr(config.hf_config, "eos_token_id", 1) or 1
            self.executor = None
            self.scheduler = None
            self.deployment = DeploymentManager(config)
            # LocalEngine ingress triggers wakeup only after the request is
            # visible in its queue, avoiding a START_WAVE/enqueue race.
            self.router = RequestRouter(
                self.deployment.engine_clients,
                router_policy=config.router_policy,
                kvcache_block_size=config.kvcache_block_size,
                admission_batch_size=config.max_ingress_batch_requests,
                poll_admission_batches=(
                    self.deployment.poll_admission_batches
                ),
                admission_planner_config=(
                    AdmissionPlannerConfig.from_config(config)
                ),
            )
            # LeastBatch also needs an authoritative running-count baseline
            # before draining its global admission queue.
            self.router.record_loads(
                self.deployment.load_snapshots()
            )
            self._hierarchical_sequences: dict[int, Sequence] = {}
            self._last_load_report_time = 0.0
            atexit.register(self.exit)
            return

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        if config.dummy_prefill:
            self.tokenizer = None
            config.eos = getattr(config.hf_config, "eos_token_id", 1) or 1
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model, use_fast=True, trust_remote_code=True
            )
            config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        logger.info(
            f"Initialized Scheduler with RoutingStrategy: {self.scheduler.routing_strategy}"
        )
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        if self.config.scheduler_arch == "hierarchical":
            self.deployment.close()
            return
        if self.executor is not None:
            del self.executor
            self.executor = None

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()
        self.executor.init_rpc_endpoint()

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        if self.config.scheduler_arch == "hierarchical":
            request_ids = set(self.submit_requests_async(seqs))
            results_by_id: dict[int, AddResult] = {}
            deferred_acks: list[IngressAck] = []
            deferred_results: list[AddResultEvent] = []
            deadline = perf_counter() + self.config.quantum_timeout_s
            try:
                while request_ids.difference(results_by_id):
                    for ack in self.poll_ingress_acks():
                        if ack.request_id not in request_ids:
                            deferred_acks.append(ack)
                        elif not ack.enqueued:
                            results_by_id[ack.request_id] = AddResult(
                                request_id=ack.request_id,
                                accepted=False,
                                engine_id=(
                                    ack.engine_id
                                    if ack.engine_id >= 0
                                    else None
                                ),
                                reason=ack.reason,
                            )
                    for event in self.poll_add_results():
                        if event.request_id not in request_ids:
                            deferred_results.append(event)
                        else:
                            results_by_id[event.request_id] = AddResult(
                                request_id=event.request_id,
                                accepted=event.accepted,
                                engine_id=event.engine_id,
                                reason=event.reason,
                            )
                    if request_ids.issubset(results_by_id):
                        break
                    if perf_counter() >= deadline:
                        raise TimeoutError(
                            "hierarchical ADD result timed out"
                        )
                    time.sleep(0.0005)
            finally:
                self._frontend_ingress_acks.extend(deferred_acks)
                self._frontend_add_results.extend(deferred_results)
            results = [results_by_id[seq.seq_id] for seq in seqs]
            return results[0] if len(results) == 1 else tuple(results)
        for seq in seqs:
            seq.metric = self.metrics_manager.create_sequence_metric(
                seq.seq_id, seq.num_prompt_tokens
            )
            self.scheduler.add(seq)

    def submit_requests_async(
        self, seqs: Sequence | list[Sequence]
    ) -> tuple[int, ...]:
        """Submit requests without waiting for hierarchical scheduler ADD."""
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        request_ids: list[int] = []
        if self.config.scheduler_arch != "hierarchical":
            for seq in seqs:
                self.add_request(seq)
                request_ids.append(seq.seq_id)
                self._frontend_ingress_acks.append(
                    IngressAck(
                        request_id=seq.seq_id,
                        engine_id=-1,
                        enqueued=True,
                    )
                )
                self._frontend_add_results.append(
                    AddResultEvent(
                        request_id=seq.seq_id,
                        engine_id=-1,
                        accepted=True,
                    )
                )
            return tuple(request_ids)

        for seq in seqs:
            if seq.metric is None:
                seq.metric = self.metrics_manager.create_sequence_metric(
                    seq.seq_id, seq.num_prompt_tokens
                )
                seq.metric.record_arrival()
                seq.metric.record_decode_arrival()
            self._hierarchical_sequences.setdefault(seq.seq_id, seq)
            request_ids.append(
                self.router.submit_async(
                    request_id=seq.seq_id,
                    prompt_token_ids=tuple(seq.prompt_token_ids),
                    max_tokens=seq.max_tokens,
                    temperature=seq.temperature,
                    ignore_eos=seq.ignore_eos,
                )
            )
        return tuple(request_ids)

    def poll_ingress_acks(self) -> tuple[IngressAck, ...]:
        if self.config.scheduler_arch == "hierarchical":
            # This is the first call in the serving loop's frontend poll
            # cycle. Force one new consolidated RPC; the other poll_* methods
            # consume buffers populated by the same response.
            self._ensure_frontend_cycle(force=True)
        events = tuple(self._frontend_ingress_acks)
        self._frontend_ingress_acks.clear()
        return events

    def poll_add_results(self) -> tuple[AddResultEvent, ...]:
        if self.config.scheduler_arch == "hierarchical":
            self._ensure_frontend_cycle()
        events = tuple(self._frontend_add_results)
        self._frontend_add_results.clear()
        return events

    def poll_first_token_events(self) -> tuple[FirstTokenEvent, ...]:
        if self.config.scheduler_arch != "hierarchical":
            return ()
        self._ensure_frontend_cycle()
        events = tuple(self._frontend_first_token_events)
        self._frontend_first_token_events.clear()
        return events

    def poll_first_schedule_events(
        self,
    ) -> tuple[FirstScheduleEvent, ...]:
        if self.config.scheduler_arch != "hierarchical":
            return ()
        self._ensure_frontend_cycle()
        events = tuple(self._frontend_first_schedule_events)
        self._frontend_first_schedule_events.clear()
        return events

    @property
    def num_pending_ingress(self) -> int:
        if self.config.scheduler_arch != "hierarchical":
            return 0
        return self.router.pending_ingress_count

    @property
    def num_pending_adds(self) -> int:
        if self.config.scheduler_arch != "hierarchical":
            return 0
        return self.router.pending_add_count

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        if self.config.scheduler_arch == "hierarchical":
            raise RuntimeError(
                "hierarchical dummy-decode mode does not support KV migration"
            )
        self.scheduler.free_to_be_migrated(seqs)

    def step(self):
        if self.config.scheduler_arch == "hierarchical":
            events = self.poll()
            outputs = [
                (event.request_id, [])
                for event in events
            ]
            return (
                outputs,
                -sum(event.generated_count for event in events),
                self.router.active_count,
                0.0,
                0.0,
            )
        step_start = time.perf_counter()
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.perf_counter()
        sch_res = self.scheduler.schedule()
        dp_seqs = sch_res.dp_seqs
        is_prefill = sch_res.is_prefill
        dp_sp_seqs = sch_res.dp_sp_seqs
        filtered_dp_sp_seqs = sch_res.filtered_dp_sp_seqs

        total_running = sum(len(seqs) for seqs in dp_seqs)
        total_waiting = self.scheduler.get_total_waiting_size()
        total_waiting_migration = self.scheduler.get_total_waiting_migration_size()
        self.metrics_manager.server_metric.update_running_requests(total_running)
        self.metrics_manager.server_metric.update_waiting_requests(total_waiting)
        self.metrics_manager.server_metric.update_waiting_migration_requests(
            total_waiting_migration
        )

        if (
            self.log_decode_step_detail
            and total_waiting_migration > 0
            and self.scheduler.waiting_migration
        ):
            logger.info(f"{self.scheduler.waiting_migration[0].num_tokens=}")

        dp_sp_tp_seqs = [seqs for seqs in dp_sp_seqs for _ in range(tp_size)]
        # dp_batch_sizes = [len(seqs) for seqs in dp_seqs]
        sp_batch_sizes = [
            [
                len(filtered_dp_sp_seqs[dp_idx * sp_size + sp_idx])
                for sp_idx in range(sp_size)
            ]
            for dp_idx in range(dp_size)
        ]

        sp_send_counts = sch_res.sp_send_counts
        sp_recv_counts = sch_res.sp_recv_counts
        sp_size_hist_per_dp_raw = sch_res.sp_size_hist_per_dp
        # sp_comm_matrix = sch_res.sp_comm_matrix
        sp_q_matrix = sch_res.sp_q_matrix
        sp_res_matrix = sch_res.sp_res_matrix

        sp_size_hist_per_dp = []
        sp_size_hist_global: dict[int, int] = {}
        for dp_hist in sp_size_hist_per_dp_raw:
            compact_hist = {}
            for sp_size, count in enumerate(dp_hist):
                if count > 0:
                    compact_hist[sp_size] = count
                    sp_size_hist_global[sp_size] = (
                        sp_size_hist_global.get(sp_size, 0) + count
                    )
            sp_size_hist_per_dp.append(compact_hist)
        
        # Update metrics with raw counts
        self.metrics_manager.server_metric.update_sp_stats(sp_send_counts, sp_recv_counts)
        
        waiting_head_blocks = sch_res.waiting_head_blocks
        waiting_total_blocks = sch_res.waiting_total_blocks
        self.metrics_manager.server_metric.update_waiting_blocks(waiting_head_blocks, waiting_total_blocks)

        sch_end = time.perf_counter()
        post_sch_begin = 0
        post_sch_end = 0
        if is_prefill and self.config.mode == "decode":
            if not self.config.dummy_prefill:
                logger.debug("perform migration")
                for seqs in filtered_dp_sp_seqs:
                    for seq in seqs:
                        logger.debug(
                            f"{seq.block_ctx().block_location}, "
                            f"{seq.block_ctx(BlockContextSlot.MIGRATE).block_location}"
                        )
                self.executor.migrate(dp_sp_seqs)
            else:
                for dp_idx, seqs in enumerate(dp_seqs):
                    for seq in seqs:
                        if self.scheduler.worker_state[
                            dp_idx
                        ].is_control_dummy(seq):
                            continue
                        if not self.scheduler.worker_state[dp_idx].may_append(seq, 1):
                            logger.error(
                                "Failed to allocate block for sequence %s during dummy prefill; skipping token append.",
                                getattr(seq, "seq_id", "<unknown>"),
                            )
                            continue
                        seq.append_token(0, BlockContextSlot.ACTIVE)
                for seqs in dp_seqs:
                    for seq in seqs:
                        if seq.metric and seq.metric.num_generated_tokens == 0:
                            seq.metric.record_first_token()
                            seq.metric.num_generated_tokens = 1
        else:
            model_runner_start = time.perf_counter()
            token_ids = self.executor.run(dp_sp_tp_seqs, is_prefill)[::tp_size]
            model_runner_duration_ms = (time.perf_counter() - model_runner_start) * 1000.0
            step_duration_ms = (time.perf_counter() - step_start) * 1000.0
            post_sch_begin = time.perf_counter()
            self.scheduler.postprocess(
                filtered_dp_sp_seqs, token_ids, self.metrics_manager,
                step_duration_ms, self.config.loop_count
            )
            post_sch_end = time.perf_counter()
        outputs = []
        num_tokens = 0

        for dp_idx, seqs in enumerate(dp_seqs):
            num_tokens_in_dp = sum(len(seq) for seq in seqs)
            self.metrics_manager.server_metric.update_token_usage(
                dp_idx, num_tokens_in_dp
            )

        for seqs in dp_seqs:
            num_tokens += (
                sum(len(seq) for seq in seqs)
                if is_prefill
                else -len(seqs) * self.config.loop_count
            )
            for seq in seqs:
                if seq.is_finished:
                    # Complete sequence metric and log
                    self.metrics_manager.complete_sequence(seq.seq_id)
                    outputs.append((seq.seq_id, seq.completion_token_ids))
        
        # Calculate and log ITL for this step
        if not is_prefill:
            itl = step_duration_ms / self.config.loop_count
            free_blocks = [
                [
                    len(worker_state.block_manager[i].free_block_ids)
                    for i in range(self.scheduler.attention_sp)
                ]
                for worker_state in self.scheduler.worker_state
            ]
            if self.log_decode_step_detail:
                # Per-SP-rank seq_lens: [dp_idx][sp_idx] -> list of seq lens on that GPU.
                sp_seq_lens = [
                    [
                        [
                            len(seq)
                            for seq in filtered_dp_sp_seqs[dp_idx * sp_size + sp_idx]
                        ]
                        for sp_idx in range(sp_size)
                    ]
                    for dp_idx in range(dp_size)
                ]
                logger.info(
                    {
                        "mode": "decode",
                        "itl": f"{itl:.2f}ms",
                        "sch_ovhd": f"{(sch_end - sch_begin) * 1000:.2f}ms",
                        "post_sch_ovhd": f"{(post_sch_end - post_sch_begin) * 1000:.2f}ms",
                        "waiting_reqs": total_waiting,
                        "sp_seq_lens": sp_seq_lens,
                        "sp_batch_sizes": sp_batch_sizes,
                        "sp_send_counts": sp_send_counts,
                        "sp_recv_counts": sp_recv_counts,
                        "sp_size_hist_global": sp_size_hist_global,
                        "sp_size_hist_per_dp": sp_size_hist_per_dp,
                        "waiting_head_blocks": waiting_head_blocks,
                        "waiting_total_blocks": waiting_total_blocks,
                        "sp_q_matrix": sp_q_matrix,
                        "sp_res_matrix": sp_res_matrix,
                        "free_blocks": free_blocks,
                    }
                )
            else:
                flat_sp_batch_sizes = [
                    batch_size
                    for dp_batch_sizes in sp_batch_sizes
                    for batch_size in dp_batch_sizes
                ]
                flat_free_blocks = [
                    num_free_blocks
                    for dp_free_blocks in free_blocks
                    for num_free_blocks in dp_free_blocks
                ]
                min_free_blocks = min(flat_free_blocks) if flat_free_blocks else 0
                max_kv_util_pct = (
                    100.0
                    * (self.config.num_kvcache_blocks - min_free_blocks)
                    / self.config.num_kvcache_blocks
                    if self.config.num_kvcache_blocks > 0
                    else 0.0
                )
                logger.info(
                    {
                        "mode": "decode",
                        "itl": f"{itl:.2f}ms",
                        "sch_ovhd": f"{(sch_end - sch_begin) * 1000:.2f}ms",
                        "post_sch_ovhd": f"{(post_sch_end - post_sch_begin) * 1000:.2f}ms",
                        "waiting_reqs": total_waiting,
                        "total_batch_size": sum(flat_sp_batch_sizes),
                        "max_sp_batch_size": (
                            max(flat_sp_batch_sizes) if flat_sp_batch_sizes else 0
                        ),
                        "min_free_blocks": min_free_blocks,
                        "max_kv_util_pct": f"{max_kv_util_pct:.2f}",
                        "sp_size_hist_global": sp_size_hist_global,
                        "waiting_head_blocks_sum": sum(waiting_head_blocks),
                        "waiting_total_blocks_sum": sum(waiting_total_blocks),
                    }
                )

        return (
            outputs,
            num_tokens,
            sum(len(seqs) for seqs in dp_seqs),
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        if self.config.scheduler_arch == "hierarchical":
            # A consolidated frontend poll can remove the final owners from
            # the router before the serving loop consumes their buffered
            # FinishEvents through step()/poll().
            return (
                self.router.is_idle
                and not self._frontend_finish_events
            )
        return self.scheduler.is_finished()

    def _ensure_frontend_cycle(self, *, force: bool = False) -> None:
        if force:
            self._frontend_cycle_active = False
        if self._frontend_cycle_active:
            return
        self._poll_frontend_control_plane()
        self._frontend_cycle_active = True

    def _poll_frontend_control_plane(self) -> None:
        batches = self.deployment.poll_frontend_events()
        snapshots = tuple(batch.load for batch in batches)
        self.router.record_loads(snapshots)

        raw_add_results = tuple(
            event
            for batch in batches
            for event in batch.add_results
        )
        ready_add_results = list(
            self.router.record_add_results(raw_add_results)
        )
        self._frontend_ingress_acks.extend(
            self.router.poll_ingress_acks()
        )
        # Admission ACKs can make ADD results fetched in the same consolidated
        # RPC routable, so flush the router's early-event buffer immediately.
        ready_add_results.extend(self.router.record_add_results(()))
        self._frontend_add_results.extend(ready_add_results)

        first_schedule_events = self.router.record_first_schedule_events(
            event
            for batch in batches
            for event in batch.first_schedule_events
        )
        self._frontend_first_schedule_events.extend(
            first_schedule_events
        )

        first_token_events = tuple(
            event
            for batch in batches
            for event in batch.first_token_events
        )
        self._frontend_first_token_events.extend(first_token_events)
        for event in first_token_events:
            sequence = self._hierarchical_sequences[event.request_id]
            metric = sequence.metric
            if metric is None:
                continue
            if metric.first_scheduled_time is None:
                metric.record_first_scheduled()
            if metric.decode_scheduled_time is None:
                metric.record_decode_scheduled()
            if metric.first_token_time is None:
                metric.record_first_token()

        finish_events = self.router.record_finish_events(
            event
            for batch in batches
            for event in batch.finish_events
        )
        for event in finish_events:
            sequence = self._hierarchical_sequences[event.request_id]
            metric = sequence.metric
            if metric is not None:
                metric.num_generated_tokens = event.generated_count
                if (
                    event.generated_count > 0
                    and metric.first_token_time is None
                ):
                    if metric.first_scheduled_time is None:
                        metric.record_first_scheduled()
                    if metric.decode_scheduled_time is None:
                        metric.record_decode_scheduled()
                    metric.record_first_token()
                self.metrics_manager.complete_sequence(event.request_id)
            sequence.status = SequenceStatus.FINISHED
        self._frontend_finish_events.extend(finish_events)

        now = time.monotonic()
        if (
            now - self._last_load_report_time
            >= self.config.load_report_interval_ms / 1000
        ):
            self._last_load_report_time = now
            self.metrics_manager.server_metric.update_waiting_requests(
                sum(snapshot.waiting for snapshot in snapshots)
            )
            self.metrics_manager.server_metric.update_running_requests(
                sum(snapshot.running for snapshot in snapshots)
            )

    def poll(self) -> tuple[FinishEvent, ...]:
        if self.config.scheduler_arch != "hierarchical":
            outputs, *_ = self.step()
            return tuple(
                FinishEvent(
                    request_id=seq_id,
                    generated_count=len(token_ids),
                    status="FINISHED",
                    engine_id=-1,
                )
                for seq_id, token_ids in outputs
            )

        self._ensure_frontend_cycle()
        events = tuple(self._frontend_finish_events)
        self._frontend_finish_events.clear()
        self._frontend_cycle_active = False
        return events

    def abort_request(self, request_id: int):
        if self.config.scheduler_arch != "hierarchical":
            raise RuntimeError(
                "abort_request is currently implemented for hierarchical mode"
            )
        return self.router.abort(request_id)

    def drain_execution_traces(self) -> tuple[dict, ...]:
        if self.config.scheduler_arch != "hierarchical":
            raise RuntimeError(
                "execution traces are only available in hierarchical mode"
            )
        return self.deployment.execution_traces()

    def hierarchical_itl_samples(self):
        if self.config.scheduler_arch != "hierarchical":
            raise RuntimeError(
                "hierarchical ITL samples are only available in "
                "hierarchical mode"
            )
        return self.deployment.decode_itl_samples()

    def drain_hierarchical_quantum_diagnostics(
        self,
    ) -> tuple[dict, ...]:
        if self.config.scheduler_arch != "hierarchical":
            raise RuntimeError(
                "quantum diagnostics are only available in hierarchical mode"
            )
        return self.deployment.quantum_diagnostics()

    def execution_boundary_metrics(self) -> dict:
        if self.config.scheduler_arch == "hierarchical":
            return {
                "mode": "hierarchical",
                "per_engine": {
                    str(engine_id): metrics
                    for engine_id, metrics in (
                        self.deployment.execution_boundary_metrics().items()
                    )
                },
            }
        return {
            "mode": "legacy_global",
            **self.executor.execution_boundary_metrics(),
        }

    def reset_execution_boundary_metrics(self) -> None:
        if self.config.scheduler_arch == "hierarchical":
            self.deployment.reset_execution_boundary_metrics()
            return
        self.executor.reset_execution_boundary_metrics()

    def hierarchical_metrics(
        self,
        *,
        refresh: bool = True,
        include_per_engine: bool = False,
    ) -> dict:
        if self.config.scheduler_arch != "hierarchical":
            raise RuntimeError(
                "hierarchical metrics are only available in hierarchical mode"
            )
        if refresh:
            snapshots = self.deployment.load_snapshots()
            self.router.record_loads(snapshots)
        else:
            snapshots = tuple(self.router.last_loads().values())
        if not snapshots:
            return {}

        summed_fields = (
            "useful_decode_tokens",
            "raw_token_slots",
            "control_dummy_slots",
            "total_rank_forwards",
            "all_dummy_rank_forwards",
            "preemption_count",
            "command_count",
            "command_queue_delay_ms_total",
            "decode_quantum_count",
            "admission_latency_ms_total",
            "schedule_latency_ms_total",
            "coordination_latency_ms_total",
            "execute_latency_ms_total",
            "ray_get_latency_ms_total",
            "worker_result_wait_latency_ms_total",
            "result_rebuild_latency_ms_total",
            "result_rebuild_sample_count",
            "result_index_latency_ms_total",
            "result_validate_latency_ms_total",
            "result_pack_latency_ms_total",
            "postprocess_latency_ms_total",
            "ingress_queue_delay_ms_total",
            "scheduler_add_ms_total",
            "decode_itl_ms_weighted_total",
            "decode_itl_token_count",
            "decode_itl_sample_count",
        )
        metrics = {
            field: sum(getattr(snapshot, field) for snapshot in snapshots)
            for field in summed_fields
        }
        metrics.update(
            {
                "waiting_requests": sum(
                    snapshot.waiting for snapshot in snapshots
                ),
                "running_requests": sum(
                    snapshot.running for snapshot in snapshots
                ),
                "pending_ingress": sum(
                    snapshot.pending_ingress for snapshot in snapshots
                ),
                "pending_add_results": sum(
                    snapshot.pending_add_results for snapshot in snapshots
                ),
                "reserved_slots": sum(
                    snapshot.reserved_slots for snapshot in snapshots
                ),
                "free_blocks_min": min(
                    snapshot.free_blocks_min for snapshot in snapshots
                ),
            }
        )
        admission_routing = self.router.admission_metrics()
        metrics["admission_routing"] = admission_routing
        metrics["global_pending_admission"] = admission_routing[
            "global_pending"
        ]
        metrics["waiting_requests_total"] = (
            metrics["waiting_requests"]
            + metrics["global_pending_admission"]
        )
        decode_itl_token_count = metrics["decode_itl_token_count"]
        metrics["decode_itl_ms_mean"] = (
            metrics["decode_itl_ms_weighted_total"]
            / decode_itl_token_count
            if decode_itl_token_count
            else None
        )
        result_rebuild_sample_count = metrics[
            "result_rebuild_sample_count"
        ]
        metrics["result_rebuild_latency_ms_mean"] = (
            metrics["result_rebuild_latency_ms_total"]
            / result_rebuild_sample_count
            if result_rebuild_sample_count
            else None
        )
        metrics["ray_get_latency_ms_mean"] = (
            metrics["ray_get_latency_ms_total"]
            / result_rebuild_sample_count
            if result_rebuild_sample_count
            else None
        )
        metrics["worker_result_wait_latency_ms_mean"] = (
            metrics["worker_result_wait_latency_ms_total"]
            / result_rebuild_sample_count
            if result_rebuild_sample_count
            else None
        )
        metrics["result_rebuild_latency_ms_max"] = max(
            snapshot.result_rebuild_latency_ms_max
            for snapshot in snapshots
        )
        metrics["ray_get_latency_ms_max"] = max(
            snapshot.ray_get_latency_ms_max for snapshot in snapshots
        )
        metrics["worker_result_wait_latency_ms_max"] = max(
            snapshot.worker_result_wait_latency_ms_max
            for snapshot in snapshots
        )
        if include_per_engine:
            per_engine_fields = (
                "waiting",
                "running",
                "free_blocks_min",
                "wave_id",
                "quantum_id",
                "useful_real_batch_size",
                "control_dummy_count",
                "all_dummy_engine_quantums",
                "useful_decode_tokens",
                "raw_token_slots",
                "control_dummy_slots",
                "total_rank_forwards",
                "all_dummy_rank_forwards",
                "preemption_count",
                "command_count",
                "command_queue_delay_ms_total",
                "decode_quantum_count",
                "admission_latency_ms_total",
                "schedule_latency_ms_total",
                "coordination_latency_ms_total",
                "execute_latency_ms_total",
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
                "postprocess_latency_ms_total",
                "pending_ingress",
                "pending_add_results",
                "reserved_slots",
                "ingress_queue_delay_ms_total",
                "scheduler_add_ms_total",
                "decode_itl_ms_weighted_total",
                "decode_itl_token_count",
                "decode_itl_sample_count",
            )
            per_engine = {}
            for snapshot in snapshots:
                engine_metrics = {
                    field: getattr(snapshot, field)
                    for field in per_engine_fields
                }
                engine_metrics["decode_itl_ms_mean"] = (
                    snapshot.decode_itl_ms_weighted_total
                    / snapshot.decode_itl_token_count
                    if snapshot.decode_itl_token_count
                    else None
                )
                rebuild_samples = snapshot.result_rebuild_sample_count
                engine_metrics["result_rebuild_latency_ms_mean"] = (
                    snapshot.result_rebuild_latency_ms_total
                    / rebuild_samples
                    if rebuild_samples
                    else None
                )
                engine_metrics["ray_get_latency_ms_mean"] = (
                    snapshot.ray_get_latency_ms_total / rebuild_samples
                    if rebuild_samples
                    else None
                )
                engine_metrics["worker_result_wait_latency_ms_mean"] = (
                    snapshot.worker_result_wait_latency_ms_total
                    / rebuild_samples
                    if rebuild_samples
                    else None
                )
                engine_metrics["rank_loads"] = [
                    {
                        "global_rank": rank_load.global_rank,
                        "sp_idx": rank_load.sp_idx,
                        "tp_idx": rank_load.tp_idx,
                        "master_batch_size": (
                            rank_load.master_batch_size
                        ),
                        "active_master_requests": (
                            rank_load.active_master_requests
                        ),
                        "active_receiver_requests": (
                            rank_load.active_receiver_requests
                        ),
                        "active_dispatched_tokens": (
                            rank_load.active_dispatched_tokens
                        ),
                        "free_blocks": rank_load.free_blocks,
                        "total_blocks": rank_load.total_blocks,
                        "control_dummy_blocks": (
                            rank_load.control_dummy_blocks
                        ),
                        "master_assignments": (
                            rank_load.master_assignments
                        ),
                        "mastered_decode_tokens": (
                            rank_load.mastered_decode_tokens
                        ),
                    }
                    for rank_load in snapshot.rank_loads
                ]
                per_engine[str(snapshot.engine_id)] = engine_metrics
            metrics["per_engine"] = per_engine
        raw_slots = metrics["raw_token_slots"]
        rank_forwards = metrics["total_rank_forwards"]
        metrics["dummy_slot_ratio"] = (
            metrics["control_dummy_slots"] / raw_slots
            if raw_slots
            else 0.0
        )
        metrics["dummy_rank_forward_ratio"] = (
            metrics["all_dummy_rank_forwards"] / rank_forwards
            if rank_forwards
            else 0.0
        )
        measured_ms = sum(
            metrics[name]
            for name in (
                "admission_latency_ms_total",
                "schedule_latency_ms_total",
                "coordination_latency_ms_total",
                "execute_latency_ms_total",
                "postprocess_latency_ms_total",
            )
        )
        metrics["coordination_overhead_ratio"] = (
            metrics["coordination_latency_ms_total"] / measured_ms
            if measured_ms
            else 0.0
        )
        return metrics

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ):
        if self.config.scheduler_arch == "hierarchical":
            raise RuntimeError(
                "hierarchical dummy-decode mode does not support P/D migration"
            )
        return self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

    def p2p_connect(
        self, remote_engine_name: str, remote_endpoints_info: list[list[dict]]
    ):
        if self.config.scheduler_arch == "hierarchical":
            raise RuntimeError(
                "hierarchical dummy-decode mode does not support P/D migration"
            )
        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
    ) -> None:
        if self.config.scheduler_arch == "hierarchical":
            return self._generate_hierarchical(use_tqdm=use_tqdm)
        num_reqs = self.scheduler.get_total_waiting_size()
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        step_count = 0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                    self.metrics_manager.server_metric.record_prefill_throughput(
                        num_tokens, (perf_counter() - t)
                    )
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                    self.metrics_manager.server_metric.record_decode_throughput(
                        -num_tokens, (perf_counter() - t)
                    )
                itl = (perf_counter() - t) * 1000 / self.config.loop_count
                pbar.set_postfix(
                    {
                        "bs": f"{bs}",
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                        "itl": f"{itl:.2f}ms",
                        "sch_ovhd": f"{sch_latency:.2f}ms",
                        "post_sch_ovhd": f"{post_sch_latency:.2f}ms",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()

        logger.info("=" * 60)
        logger.info("Final Server Metrics Summary")
        logger.info("=" * 60)
        self.metrics_manager.log_server_metrics(include_detailed=True)
        summary = self.metrics_manager.get_server_summary()
        for key, value in summary.items():
            if value is not None:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        return

    def _generate_hierarchical(self, *, use_tqdm: bool) -> None:
        total = self.router.active_count
        pbar = (
            tqdm(total=total, desc="Generating", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        while not self.is_finished():
            events = self.poll()
            if not events:
                time.sleep(0.001)
                continue
            if pbar is not None:
                pbar.update(len(events))
                pbar.set_postfix(
                    {
                        "active": self.router.active_count,
                        "wave": self.router.wave_id,
                    }
                )
        if pbar is not None:
            pbar.close()
        logger.info({"mode": "hierarchical", **self.hierarchical_metrics()})
        self.metrics_manager.log_server_metrics(include_detailed=True)

import atexit
import os
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy._cpp import BlockContextSlot, ScheduleAction
from nanodeploy.config import Config
from nanodeploy.engine.kv_consolidation import execute_planned_ls_kv_scale_down
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.metrics import MetricsManager

logger = get_logger()


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LLMEngine:
    def __init__(self, model, **kwargs):
        self.engine_id = str(uuid.uuid4())

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.config = config
        self.config.engine_id = self.engine_id
        self.ps = []
        self.events = []
        self.pending_maintenance_stall_ms = 0.0
        self.fatal_error: BaseException | None = None
        self.log_decode_step_detail = _env_flag_enabled(
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL", default=False
        )

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
        self.metrics_manager = MetricsManager()
        atexit.register(self.exit)

    def exit(self):
        del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()
        self.executor.init_rpc_endpoint()

    def add_request(self, seqs: Sequence | list[Sequence]):
        self._raise_if_fatal()
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq.metric = self.metrics_manager.create_sequence_metric(
                seq.seq_id, seq.num_prompt_tokens
            )
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self._raise_if_fatal()
        self.scheduler.free_to_be_migrated(seqs)

    def _raise_if_fatal(self):
        if self.fatal_error is not None:
            raise RuntimeError(
                "LLMEngine is fatal after a failed KV consolidation; restart required"
            ) from self.fatal_error

    def step(self):
        self._raise_if_fatal()
        step_start = time.perf_counter()
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.perf_counter()
        sch_res = self.scheduler.schedule()
        sch_end = time.perf_counter()
        if sch_res.action == ScheduleAction.KV_CONSOLIDATION:
            maintenance_begin = time.perf_counter()
            try:
                result = execute_planned_ls_kv_scale_down(
                    self.scheduler,
                    self.executor,
                    sch_res.kv_consolidation_plan,
                    abort_on_copy_error=False,
                )
            except BaseException as error:
                self.fatal_error = error
                logger.exception(
                    "KV consolidation failed; engine is fatal and must be restarted"
                )
                raise
            maintenance_ms = (time.perf_counter() - maintenance_begin) * 1000.0
            maintenance_stall_ms = (time.perf_counter() - step_start) * 1000.0
            self.pending_maintenance_stall_ms += maintenance_stall_ms
            logger.info(
                {
                    "mode": "ls_kv_consolidation",
                    "transaction_id": result.transaction_id,
                    "group_id": result.group_id,
                    "dp_idx": result.dp_idx,
                    "source_rank": result.source_rank,
                    "retained_ranks": result.retained_ranks,
                    "num_tokens": result.num_tokens,
                    "num_moves": result.num_moves,
                    "group_util": sch_res.ls_kv_consolidation_group_util,
                    "target_dop": sch_res.ls_kv_consolidation_target_dop,
                    "stable_steps": sch_res.ls_kv_consolidation_stable_steps,
                    "maintenance_ms": maintenance_ms,
                    "maintenance_stall_ms": maintenance_stall_ms,
                }
            )
            return (
                [],
                0,
                0,
                (sch_end - sch_begin) * 1000.0,
                0.0,
            )

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

        if self.log_decode_step_detail and total_waiting_migration > 0:
            # In decentralized mode, we can't directly access waiting_migration[0]
            # This is just for logging, so we skip it in decentralized mode
            if hasattr(self.scheduler, 'waiting_migration') and self.scheduler.waiting_migration:
                logger.info(f"{self.scheduler.waiting_migration[0].num_tokens=}")
            elif self.config.enable_ls_decode_core_scheduler:
                pending = self.scheduler.get_ls_pending_batch_sequence_ids()
                if pending:
                    logger.info(
                        "oldest LS pending logical batch sequence_ids=%s", pending[0]
                    )

        dp_sp_tp_seqs = [seqs for seqs in dp_sp_seqs for _ in range(tp_size)]

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
                        if not self.scheduler.worker_state[dp_idx].may_append(seq, 1):
                            if self.config.enable_ls_decode_core_scheduler:
                                raise RuntimeError(
                                    "validated LS admission lost its provisional "
                                    f"pending-token capacity for sequence {seq.seq_id}"
                                )
                            logger.error(
                                "Failed to allocate block for sequence %s during dummy prefill; skipping token append.",
                                getattr(seq, "seq_id", "<unknown>"),
                            )
                            continue
                        seq.append_token(0, BlockContextSlot.ACTIVE)
                        if self.config.enable_ls_decode_core_scheduler:
                            seq.mark_last_token_pending(BlockContextSlot.ACTIVE)
                            self.scheduler.worker_state[dp_idx].add_running_tokens(
                                seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx,
                                1,
                            )
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
            if not is_prefill:
                step_duration_ms += self.pending_maintenance_stall_ms
            post_sch_begin = time.perf_counter()
            self.scheduler.postprocess(
                filtered_dp_sp_seqs, token_ids, self.metrics_manager,
                step_duration_ms, self.config.loop_count
            )
            if not is_prefill:
                self.pending_maintenance_stall_ms = 0.0
            post_sch_end = time.perf_counter()

        if self.config.enable_ls_decode_core_scheduler:
            if sch_res.ls_sealed_batch_ids:
                logger.info(
                    {
                        "mode": "ls_decode_batch_seal",
                        "batch_ids": sch_res.ls_sealed_batch_ids,
                        "sequence_ids": sch_res.ls_sealed_batch_sequence_ids,
                    }
                )
            if is_prefill and sch_res.ls_initial_batch_ids:
                logger.info(
                    {
                        "mode": "ls_decode_admission",
                        "batch_ids": sch_res.ls_initial_batch_ids,
                        "group_ids": sch_res.ls_initial_group_ids,
                        "sequence_ids": sch_res.ls_initial_sequence_ids,
                        "initial_kv_dops": sch_res.ls_initial_kv_dops,
                        "initial_kv_ranks": sch_res.ls_initial_kv_ranks,
                        "prompt_kv_tokens": sch_res.ls_initial_prompt_kv_tokens,
                        "provisional_pending_targets": (
                            sch_res.ls_initial_provisional_pending_targets
                        ),
                        "admission_orders": sch_res.ls_initial_admission_orders,
                        "admission_attempts": sch_res.ls_initial_admission_attempts,
                        "is_recovery_batch": (
                            sch_res.ls_initial_is_recovery_batch
                        ),
                        "parent_batch_ids": sch_res.ls_initial_parent_batch_ids,
                        "admission_kinds": sch_res.ls_initial_admission_kinds,
                        "placement_strategy": "batch_uniform",
                        "block_size": self.config.kvcache_block_size,
                    }
                )
            elif not is_prefill and (
                sch_res.ls_group_ids or sch_res.ls_preempted_sequence_ids
            ):
                logger.info(
                    {
                        "mode": "ls_decode_iteration",
                        "group_ids": sch_res.ls_group_ids,
                        "group_dp_indices": sch_res.ls_group_dp_indices,
                        "real_batch_sizes": sch_res.ls_real_batch_sizes,
                        "master_dops": sch_res.ls_master_dops,
                        "kv_dops": sch_res.ls_kv_dops,
                        "master_ranks": sch_res.ls_master_ranks,
                        "master_batch_sizes": sch_res.ls_master_batch_sizes,
                        "rank_allocations": sch_res.ls_group_rank_allocations,
                        "iteration_sequence_ids": sch_res.ls_iteration_sequence_ids,
                        "iteration_master_assignments": (
                            sch_res.ls_iteration_master_assignments
                        ),
                        "group_used_kv_tokens": sch_res.ls_group_used_kv_tokens,
                        "group_used_kv_blocks": sch_res.ls_group_used_kv_blocks,
                        "pending_append_blocks_per_master": (
                            sch_res.ls_pending_append_blocks_per_master
                        ),
                        "scale_reasons": sch_res.ls_scale_reasons,
                        "new_master_ranks": sch_res.ls_new_master_ranks,
                        "reused_passive_master_ranks": (
                            sch_res.ls_reused_passive_master_ranks
                        ),
                        "source_greedy_target_chunks": (
                            sch_res.ls_master_batch_sizes
                        ),
                        "historical_kv_migration_bytes": (
                            sch_res.ls_historical_kv_migration_bytes
                        ),
                        "kv_consolidation_candidate": (
                            sch_res.ls_kv_consolidation_candidate
                        ),
                        "kv_consolidation_group_id": (
                            sch_res.ls_kv_consolidation_group_id
                        ),
                        "kv_consolidation_source_rank": (
                            sch_res.ls_kv_consolidation_source_rank
                        ),
                        "kv_consolidation_target_dop": (
                            sch_res.ls_kv_consolidation_target_dop
                        ),
                        "kv_consolidation_stable_steps": (
                            sch_res.ls_kv_consolidation_stable_steps
                        ),
                        "kv_consolidation_group_util": (
                            sch_res.ls_kv_consolidation_group_util
                        ),
                        "kv_consolidation_decision_reason": (
                            sch_res.ls_kv_consolidation_decision_reason
                        ),
                        "preempted_sequence_ids": sch_res.ls_preempted_sequence_ids,
                        "preemption_reasons": sch_res.ls_preemption_reasons,
                        "planning_latency_ms": sch_res.ls_planning_latency_ms,
                        "model_runner_duration_ms": model_runner_duration_ms,
                        "step_itl_ms": step_duration_ms,
                    }
                )
            if (
                sch_res.ls_pending_batch_count
                or sch_res.ls_atomic_admission_no_fit_count
                or sch_res.ls_atomic_admission_rollback_count
            ):
                logger.info(
                    {
                        "mode": "ls_decode_pending_batches",
                        "pending_batch_count": sch_res.ls_pending_batch_count,
                        "pending_request_count": sch_res.ls_pending_request_count,
                        "oldest_pending_batch_age_steps": (
                            sch_res.ls_oldest_pending_batch_age_steps
                        ),
                        "max_pending_batch_attempts": (
                            sch_res.ls_max_pending_batch_attempts
                        ),
                        "atomic_admission_no_fit_count": (
                            sch_res.ls_atomic_admission_no_fit_count
                        ),
                        "atomic_admission_merge_count": (
                            sch_res.ls_atomic_admission_merge_count
                        ),
                        "atomic_admission_rollback_count": (
                            sch_res.ls_atomic_admission_rollback_count
                        ),
                    }
                )
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
        self._raise_if_fatal()
        return self.scheduler.is_finished()

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ):
        self._raise_if_fatal()
        return self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

    def p2p_connect(
        self, remote_engine_name: str, remote_endpoints_info: list[list[dict]]
    ):
        self._raise_if_fatal()
        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
    ) -> None:
        num_reqs = self.scheduler.get_total_waiting_size()
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        pending_maintenance_sec = 0.0
        step_count = 0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
            step_duration_sec = perf_counter() - t
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / step_duration_sec
                    self.metrics_manager.server_metric.record_prefill_throughput(
                        num_tokens, step_duration_sec
                    )
                elif num_tokens < 0:
                    decode_duration_sec = (
                        pending_maintenance_sec + step_duration_sec
                    )
                    pending_maintenance_sec = 0.0
                    decode_throughput = -num_tokens / decode_duration_sec
                    self.metrics_manager.server_metric.record_decode_throughput(
                        -num_tokens, decode_duration_sec
                    )
                else:
                    # An exclusive maintenance step moves no user token and
                    # must not be recorded as a zero-throughput Decode sample.
                    pending_maintenance_sec += step_duration_sec
                    continue
                itl_duration_sec = (
                    decode_duration_sec if num_tokens < 0 else step_duration_sec
                    )
                itl = itl_duration_sec * 1000 / self.config.loop_count
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

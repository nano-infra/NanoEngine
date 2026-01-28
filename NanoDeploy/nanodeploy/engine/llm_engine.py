import atexit
import json
import threading
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

import etcd3
import flatbuffers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy._cpp import BlockContextSlot
from nanodeploy.config import Config
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence, SequenceStatus
from nanodeploy.fbs import EngineInfo as EngineInfoModule
from nanodeploy.fbs.EngineInfo import EngineInfo
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.metrics import MetricsManager

logger = get_logger()


class LLMEngine:
    def __init__(self, config: Config):
        self.engine_id = str(uuid.uuid4())

        self.config = config
        self.config.engine_id = self.engine_id

        # Set log level globally first
        if self.config.log_level:
            set_log_level(self.config.log_level)

        self.ps = []
        self.events = []

        from nanodeploy.engine.ray_executor import RayExecutor

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
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

    def get_engine_id(self):
        return self.engine_id

    def get_num_kv_blocks(self):
        return self.config.num_kvcache_blocks

    def get_attn_world_size(self):
        return self.config.attn_world_size

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq.metric = self.metrics_manager.create_sequence_metric(
                seq.seq_id, seq.num_prompt_tokens
            )
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self.scheduler.free_to_be_migrated(seqs)

    def step(self):
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.time()
        sch_res = self.scheduler.schedule()
        dp_seqs = sch_res.dp_seqs
        is_prefill = sch_res.is_prefill
        dp_sp_seqs = sch_res.dp_sp_seqs
        filtered_dp_sp_seqs = sch_res.filtered_dp_sp_seqs

        total_running = sum(len(seqs) for seqs in dp_seqs)
        total_waiting = len(self.scheduler.waiting)
        total_waiting_migration = len(self.scheduler.waiting_migration)
        self.metrics_manager.server_metric.update_running_requests(total_running)
        self.metrics_manager.server_metric.update_waiting_requests(total_waiting)
        self.metrics_manager.server_metric.update_waiting_migration_requests(
            total_waiting_migration
        )

        if self.scheduler.waiting_migration:
            logger.info(f"{self.scheduler.waiting_migration[0].num_tokens=}")

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
        # sp_comm_matrix = sch_res.sp_comm_matrix
        sp_q_matrix = sch_res.sp_q_matrix
        # sp_res_matrix = sch_res.sp_res_matrix

        # Update metrics with raw counts
        self.metrics_manager.server_metric.update_sp_stats(
            sp_send_counts, sp_recv_counts
        )

        waiting_head_blocks = sch_res.waiting_head_blocks
        waiting_total_blocks = sch_res.waiting_total_blocks
        self.metrics_manager.server_metric.update_waiting_blocks(
            waiting_head_blocks, waiting_total_blocks
        )

        logger.debug(
            {
                "mode": "prefill" if is_prefill else "decode",
                # "dp_batch_sizes": dp_batch_sizes,
                "sp_batch_sizes": sp_batch_sizes,
                "sp_send_counts": sp_send_counts,
                "sp_recv_counts": sp_recv_counts,
                "waiting_head_blocks": waiting_head_blocks,
                "waiting_total_blocks": waiting_total_blocks,
                # "sp_comm_matrix": sp_comm_matrix,
                "sp_q_matrix": sp_q_matrix,
                # "sp_res_matrix": sp_res_matrix,
                "free_blocks": [
                    [
                        len(worker_state.block_manager[i].free_block_ids)
                        for i in range(self.scheduler.attention_sp)
                    ]
                    for worker_state in self.scheduler.worker_state
                ],
            }
        )

        sch_end = time.time()
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
            token_ids = self.executor.run(dp_sp_tp_seqs, is_prefill)[::tp_size]
            post_sch_begin = time.time()
            self.scheduler.postprocess(
                filtered_dp_sp_seqs, token_ids, self.metrics_manager
            )
            post_sch_end = time.time()
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
                if seq.is_finished or seq.is_to_be_migrated:
                    if seq.is_finished or seq.is_to_be_migrated:
                        self.metrics_manager.complete_sequence(seq.seq_id)
                    outputs.append(seq)
        return (
            dp_seqs,
            outputs,
            num_tokens,
            sum(len(seqs) for seqs in dp_seqs),
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
    ) -> list[Sequence]:
        num_reqs = len(self.scheduler.waiting)
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        finished_seqs = []
        prefill_throughput = decode_throughput = 0.0
        step_count = 0

        while not self.is_finished():
            t = perf_counter()
            dp_seqs, output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
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
            for seq in output:
                finished_seqs.append(seq)
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

        return finished_seqs

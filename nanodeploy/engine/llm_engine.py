import atexit
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy.config import Config
from nanodeploy.engine._core import prepare_step_inputs
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.metrics import MetricsManager

logger = get_logger()


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

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.metrics_manager = MetricsManager()
        atexit.register(self.exit)

    def exit(self):
        del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq.set_engine_id(
                self.engine_id, self.config.attention_dp, self.config.attention_sp
            )
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
        dp_seqs, is_prefill = self.scheduler.schedule()
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
        # # TODO (JimyMa): For loop
        # filtered_dp_sp_seqs = [
        #     [
        #         seq
        #         for seq in seqs
        #         if seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
        #     ]
        #     for seqs in dp_seqs
        #     for sp_idx in range(self.config.attention_sp)
        # ]

        # dp_sp_seqs = [
        #     [seq for seq in seqs]
        #     for seqs in dp_seqs
        #     for _ in range(self.config.attention_sp)
        # ]

        # dp_sp_tp_seqs = [seqs for seqs in dp_sp_seqs for _ in range(tp_size)]
        # dp_batch_sizes = [len(seqs) for seqs in dp_seqs]
        # sp_batch_sizes = [
        #     [
        #         len(filtered_dp_sp_seqs[dp_idx * sp_size + sp_idx])
        #         for sp_idx in range(sp_size)
        #     ]
        #     for dp_idx in range(dp_size)
        # ]

        inputs = prepare_step_inputs(dp_seqs, self.engine_id, sp_size, tp_size)

        filtered_dp_sp_seqs = inputs.filtered_dp_sp_seqs
        dp_sp_seqs = inputs.dp_sp_seqs
        dp_sp_tp_seqs = inputs.dp_sp_tp_seqs
        dp_batch_sizes = inputs.dp_batch_sizes
        sp_batch_sizes = inputs.sp_batch_sizes

        logger.info(
            {
                "dp_batch_sizes": dp_batch_sizes,
                "sp_batch_sizes": sp_batch_sizes,
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
                            f"{seq.block_ctx(seq.backup_engine_id).block_location}, "
                            f"{seq.block_ctx(seq.active_engine_id).block_location}"
                        )
                self.executor.migrate(dp_sp_seqs)
            else:
                [[seq.append_token(0) for seq in seqs] for seqs in dp_seqs]
        else:
            model_begin = time.time()
            token_ids, worker_run_latencies, worker_model_latencies = self.executor.run(
                dp_sp_tp_seqs, is_prefill
            )[::tp_size]
            model_end = time.time()
            model_latency = (model_end - model_begin) * 1000

            logger.info(f"Call self.executor.run Time: {model_latency:.2f} ms")
            formatted_run_lats = [f"{x:.3f}" for x in worker_run_latencies]
            formatted_model_lats = [f"{x:.3f}" for x in worker_model_latencies]
            logger.info(f"Worker GPU Run Latencies (ms): {formatted_run_lats}")
            logger.info(f"Worker GPU Model Latencies (ms): {formatted_model_lats}")

            post_sch_begin = time.time()
            token_ids = [
                token_ids[i * sp_size : (i + 1) * sp_size] for i in range(0, dp_size)
            ]
            filtered_dp_sp_seqs = [
                filtered_dp_sp_seqs[i * sp_size : (i + 1) * sp_size]
                for i in range(0, dp_size)
            ]
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
            outputs.extend(
                [
                    (seq.seq_id, seq.completion_token_ids)
                    for seq in seqs
                    if seq.is_finished
                ]
            )
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
        return (
            outputs,
            num_tokens,
            [len(seqs) for seqs in dp_seqs],
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ):
        return self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

    def p2p_connect(
        self, remote_engine_name: str, remote_endpoints_info: list[list[dict]]
    ):
        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
    ) -> None:
        num_reqs = len(self.scheduler.waiting)
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

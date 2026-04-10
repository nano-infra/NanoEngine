import atexit
import json
import os
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

import flatbuffers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nanodeploy._cpp import BlockContextSlot, init_scheduler, Sequence, SequenceStatus

from nanodeploy.config import Config
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

        # Sync C++ Sequence.block_size with Python kvcache_block_size
        Sequence.set_block_size(config.kvcache_block_size)

        self.ps = []
        self.events = []

        from nanodeploy.engine.ray_executor import RayExecutor

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = init_scheduler(config)
        logger.info(
            f"Initialized Scheduler with RoutingStrategy: {self.scheduler.routing_strategy}"
        )
        self.metrics_manager = MetricsManager()

        atexit.register(self.exit)

    def exit(self):
        """Cleanup engine resources."""
        if hasattr(self, "executor"):
            del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()

    def get_engine_id(self):
        return self.engine_id

    def get_num_kv_blocks(self):
        return self.config.num_kvcache_blocks

    def get_attn_world_size(self):
        return self.config.attn_world_size

    def get_peer_agent_addrs(self) -> list[str]:
        """Get peer agent addresses from all workers."""
        return self.executor.get_peer_agent_addrs()

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            # Debug: log received sequence info
            if self.config.mode == "decode":
                logger.info(
                    f"[DEBUG] Decode engine received seq {seq.seq_id}: last_token={seq.last_token}, num_tokens={seq.num_tokens}, token_ids_len={len(seq.token_ids)}, token_ids_last10={seq.token_ids[-10:] if seq.token_ids else []}"
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
        sch_res = self.scheduler.schedule()
        dp_seqs = sch_res.dp_seqs
        is_prefill = sch_res.is_prefill
        dp_group_seqs = sch_res.dp_group_seqs
        filtered_dp_group_seqs = sch_res.filtered_dp_group_seqs

        # Build dummy seq id set for filtering
        dummy_seq_ids = set()
        for ws in self.scheduler.worker_state:
            for d in ws.dummy_seqs:
                dummy_seq_ids.add(d.seq_id)

        total_running = sum(
            sum(1 for seq in seqs if seq.seq_id not in dummy_seq_ids)
            for seqs in dp_seqs
        )
        total_waiting = len(self.scheduler.waiting)
        total_waiting_migration = len(self.scheduler.waiting_migration)
        self.metrics_manager.server_metric.update_running_requests(total_running)
        self.metrics_manager.server_metric.update_waiting_requests(total_waiting)
        self.metrics_manager.server_metric.update_waiting_migration_requests(
            total_waiting_migration
        )

        if self.scheduler.waiting_migration:
            logger.info(f"{self.scheduler.waiting_migration[0].num_tokens=}")

        dp_group_tp_seqs = [seqs for seqs in dp_group_seqs for _ in range(tp_size)]

        dp_group_tp_seqs = [seqs for seqs in dp_group_seqs for _ in range(tp_size)]
        # dp_batch_sizes = [len(seqs) for seqs in dp_seqs]
        group_batch_sizes = [
            [
                len(filtered_dp_group_seqs[dp_idx * sp_size + sp_idx])
                for sp_idx in range(sp_size)
            ]
            for dp_idx in range(dp_size)
        ]

        group_send_counts = sch_res.group_send_counts
        group_recv_counts = sch_res.group_recv_counts
        # group_comm_matrix = sch_res.group_comm_matrix
        group_q_matrix = sch_res.group_q_matrix
        # group_res_matrix = sch_res.group_res_matrix

        # Update metrics with raw counts
        self.metrics_manager.server_metric.update_group_stats(
            group_send_counts, group_recv_counts
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
                "group_batch_sizes": group_batch_sizes,
                "group_send_counts": group_send_counts,
                "group_recv_counts": group_recv_counts,
                "waiting_head_blocks": waiting_head_blocks,
                "waiting_total_blocks": waiting_total_blocks,
                # "group_comm_matrix": group_comm_matrix,
                "group_q_matrix": group_q_matrix,
                # "group_res_matrix": group_res_matrix,
                "free_blocks": [
                    [
                        worker_state.block_manager[i].num_free_blocks
                        for i in range(self.scheduler.group_size)
                    ]
                    for worker_state in self.scheduler.worker_state
                ],
            }
        )

        sch_end = time.time()
        post_sch_begin = 0
        post_sch_end = 0

        # Run prefill to populate KV cache (or skip for decode engine receiving prefill request)
        token_ids = None
        if not (is_prefill and self.config.mode == "decode"):
            # Normal execution: prefill engine runs prefill, or decode engine runs decode
            token_ids = self.executor.run(dp_group_tp_seqs, is_prefill)[::tp_size]
            post_sch_begin = time.time()
            self.scheduler.postprocess(filtered_dp_group_seqs, token_ids, True)
            post_sch_end = time.time()

        else:
            # PD disaggregation: decode engine receives prefill request
            # DO NOT run prefill on decode engine - KV cache will be migrated from prefill engine
            logger.info(
                f"Decode engine receiving prefill request, skipping local prefill execution"
            )
            post_sch_begin = time.time()
            post_sch_end = time.time()
            self.executor.migrate(dp_group_seqs)
        outputs = []
        num_tokens = 0

        for dp_idx, seqs in enumerate(dp_seqs):
            num_tokens_in_dp = sum(len(seq) for seq in seqs)
            self.metrics_manager.server_metric.update_token_usage(
                dp_idx, num_tokens_in_dp
            )

        if is_prefill:
            for seqs in dp_seqs:
                num_tokens += sum(
                    len(seq) for seq in seqs if seq.seq_id not in dummy_seq_ids
                )
        elif token_ids is not None:
            # Count actual generated tokens, excluding dummy sequences
            for dp_idx in range(dp_size):
                for sp_idx in range(sp_size):
                    group_idx = dp_idx * sp_size + sp_idx
                    group_seqs = filtered_dp_group_seqs[group_idx]
                    group_tokens = token_ids[group_idx]
                    for seq, seq_tokens in zip(group_seqs, group_tokens):
                        if seq.seq_id not in dummy_seq_ids:
                            num_tokens -= len(seq_tokens)
        else:
            for seqs in dp_seqs:
                num_real = sum(1 for seq in seqs if seq.seq_id not in dummy_seq_ids)
                num_tokens -= num_real * self.config.loop_count

        # Collect finished/migrated sequences after postprocess
        for seqs in dp_seqs:
            for seq in seqs:
                if seq.is_finished or seq.is_to_be_migrated:
                    self.metrics_manager.complete_sequence(seq.seq_id)
                    outputs.append(seq)
        real_bs = sum(
            sum(1 for seq in seqs if seq.seq_id not in dummy_seq_ids)
            for seqs in dp_seqs
        )
        return (
            dp_seqs,
            outputs,
            num_tokens,
            real_bs,
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
        return_serialized: bool = False,
    ) -> list[Sequence] | list[bytes]:
        num_reqs = len(self.scheduler.waiting)
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        finished_seqs = []
        prefill_throughput = decode_throughput = 0.0
        step_count = 0

        # Window-based throughput tracking
        throughput_window_start = perf_counter()
        throughput_window_tokens = 0
        throughput_report_interval = 5.0  # seconds
        last_tqdm_update = perf_counter()
        tqdm_update_interval = 1.0  # seconds

        while not self.is_finished():
            t = perf_counter()
            dp_seqs, output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
            step_count += 1

            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
                self.metrics_manager.server_metric.record_prefill_throughput(
                    num_tokens, (perf_counter() - t)
                )
            else:
                step_decode_tokens = -num_tokens
                step_duration = perf_counter() - t
                self.metrics_manager.server_metric.record_decode_throughput(
                    step_decode_tokens, step_duration
                )
                throughput_window_tokens += step_decode_tokens

            # Periodic throughput reporting
            now = perf_counter()
            window_elapsed = now - throughput_window_start
            if (
                window_elapsed >= throughput_report_interval
                and throughput_window_tokens > 0
            ):
                decode_throughput = throughput_window_tokens / window_elapsed
                logger.info(
                    f"[Throughput] {decode_throughput:.0f} tok/s "
                    f"({throughput_window_tokens} tokens in {window_elapsed:.1f}s, "
                    f"bs={bs}, step={step_count})"
                )
                throughput_window_start = now
                throughput_window_tokens = 0

            # Update tqdm periodically (not every step)
            if use_tqdm and (now - last_tqdm_update >= tqdm_update_interval):
                last_tqdm_update = now
                pbar.set_postfix(
                    {
                        "bs": f"{bs}",
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                        "step": f"{step_count}",
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

        # Per-sequence ITL summary (from sequence start/end timestamps)
        import numpy as np

        itl_values = []
        for seq_id, metric in self.metrics_manager.sequence_metrics.items():
            tpot = metric.avg_tpot_wo_queueing
            if tpot is not None:
                itl_values.append(tpot)
        if itl_values:
            itl_arr = np.array(itl_values)
            logger.info(
                f"  Per-Sequence ITL (from timestamps): "
                f"mean={np.mean(itl_arr):.2f}ms, "
                f"median={np.median(itl_arr):.2f}ms, "
                f"p99={np.percentile(itl_arr, 99):.2f}ms, "
                f"n={len(itl_arr)}"
            )

        # Per-token ITL w/o first token (from itl_samples collected in C++)
        all_itl_samples = []
        for seq_id, metric in self.metrics_manager.sequence_metrics.items():
            if metric.itl_samples:
                all_itl_samples.extend(metric.itl_samples)
        if all_itl_samples:
            itl_arr = np.array(all_itl_samples)
            logger.info(
                f"  ITL w/o first token (per-token samples): "
                f"mean={np.mean(itl_arr):.2f}ms, "
                f"median={np.median(itl_arr):.2f}ms, "
                f"p99={np.percentile(itl_arr, 99):.2f}ms, "
                f"n={len(itl_arr)}"
            )

        # Overall effective throughput (wall-clock based)
        total_uptime = summary.get("uptime_seconds", 0)
        total_gen = summary.get("total_generated_tokens", 0)
        if total_uptime and total_gen:
            logger.info(
                f"  Effective decode throughput (wall-clock): "
                f"{total_gen / total_uptime:.0f} tok/s "
                f"({total_gen} tokens / {total_uptime:.1f}s)"
            )
        logger.info("=" * 60)

        # Workaround for SIGSEGV during Ray serialization of migrated sequences
        # Use FlatBuffers serialization directly to avoid pickle issues
        if return_serialized:
            logger.info(
                "Serializing sequences using FlatBuffers to avoid Ray pickle issues..."
            )
            import numpy as np

            from nanodeploy._cpp import deserialize, serialize

            serialized_seqs = []
            for seq in finished_seqs:
                try:
                    # Allocate buffer for serialization
                    buffer_size = 1024 * 1024  # 1MB should be enough
                    buffer = np.zeros(buffer_size, dtype=np.uint8)
                    data_ptr = buffer.ctypes.data

                    # Serialize using FlatBuffers
                    actual_size = serialize(data_ptr, buffer_size, [seq], False)

                    # Extract the used portion
                    serialized_bytes = bytes(buffer[:actual_size])
                    serialized_seqs.append(serialized_bytes)
                    logger.debug(
                        f"Serialized sequence {seq.seq_id} ({actual_size} bytes)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to serialize sequence {seq.seq_id}: {e}", exc_info=True
                    )
                    raise
            logger.info(f"Successfully serialized {len(serialized_seqs)} sequences")
            return serialized_seqs

        return finished_seqs

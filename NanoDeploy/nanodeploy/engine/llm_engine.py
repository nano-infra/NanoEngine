import atexit
import json
import os
import time
import uuid
from dataclasses import dataclass, fields
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


@dataclass
class StepResult:
    dp_seqs: list
    outputs: list
    prefill_tokens: int
    decode_tokens: int
    real_bs: int
    schedule_latency_ms: float
    postprocess_latency_ms: float


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
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        # Prefer generation_config.json eos_token_id (may differ from tokenizer)
        try:
            from transformers import GenerationConfig

            gen_config = GenerationConfig.from_pretrained(config.model)
            gen_eos = gen_config.eos_token_id
            if isinstance(gen_eos, list):
                eos_ids.update(gen_eos)
            elif gen_eos is not None:
                eos_ids.add(gen_eos)
        except Exception:
            pass
        config.eos = sorted(eos_ids)

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
        prefill_tokens = 0
        decode_tokens = 0

        for dp_idx, seqs in enumerate(dp_seqs):
            num_tokens_in_dp = sum(len(seq) for seq in seqs)
            self.metrics_manager.server_metric.update_token_usage(
                dp_idx, num_tokens_in_dp
            )

        if is_prefill:
            for seqs in dp_seqs:
                prefill_tokens += sum(
                    len(seq) for seq in seqs if seq.seq_id not in dummy_seq_ids
                )
        elif token_ids is not None:
            for dp_idx in range(dp_size):
                for sp_idx in range(sp_size):
                    group_idx = dp_idx * sp_size + sp_idx
                    group_seqs = filtered_dp_group_seqs[group_idx]
                    group_tokens = token_ids[group_idx]
                    for seq, seq_tokens in zip(group_seqs, group_tokens):
                        if seq.seq_id not in dummy_seq_ids:
                            decode_tokens += len(seq_tokens)
        else:
            for seqs in dp_seqs:
                num_real = sum(1 for seq in seqs if seq.seq_id not in dummy_seq_ids)
                decode_tokens += num_real * self.config.loop_count

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
        return StepResult(
            dp_seqs=dp_seqs,
            outputs=outputs,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            real_bs=real_bs,
            schedule_latency_ms=(sch_end - sch_begin) * 1000,
            postprocess_latency_ms=(post_sch_end - post_sch_begin) * 1000,
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
        window_start = perf_counter()
        window_tokens = 0
        window_interval = 5.0  # seconds
        last_tqdm_update = perf_counter()
        tqdm_interval = 1.0  # seconds

        while not self.is_finished():
            t = perf_counter()
            result = self.step()
            step_count += 1
            step_duration = perf_counter() - t

            if result.prefill_tokens > 0:
                prefill_throughput = result.prefill_tokens / step_duration
                self.metrics_manager.server_metric.record_prefill_throughput(
                    result.prefill_tokens, step_duration
                )
            if result.decode_tokens > 0:
                self.metrics_manager.server_metric.record_decode_throughput(
                    result.decode_tokens, step_duration
                )
                window_tokens += result.decode_tokens

            # Periodic throughput reporting
            now = perf_counter()
            window_elapsed = now - window_start
            if window_elapsed >= window_interval and window_tokens > 0:
                decode_throughput = window_tokens / window_elapsed
                logger.info(
                    f"[Throughput] {decode_throughput:.0f} tok/s "
                    f"({window_tokens} tokens in {window_elapsed:.1f}s, "
                    f"bs={result.real_bs}, step={step_count})"
                )
                window_start = now
                window_tokens = 0

            # Update tqdm periodically (not every step)
            if use_tqdm and (now - last_tqdm_update >= tqdm_interval):
                last_tqdm_update = now
                pbar.set_postfix(
                    {
                        "bs": f"{result.real_bs}",
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                        "step": f"{step_count}",
                    }
                )
            for seq in result.outputs:
                finished_seqs.append(seq)
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()

        self.metrics_manager.log_final_summary()

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

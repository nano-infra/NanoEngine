import atexit
import json
import os
import pickle
import time
import uuid
from collections import deque
from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

import flatbuffers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nanodeploy._cpp import BlockContextSlot, init_scheduler, Sequence, SequenceStatus

from nanodeploy.config import Config
from nanodeploy.engine.weight_sync import WeightUpdateBarrier
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.metrics import MetricsManager

logger = get_logger()


class AdmissionPaused(RuntimeError):
    """Raised when streaming rollout tries to admit during weight update."""


def _split_run_result(per_dp_results):
    """Normalise the executor's per-DP result list.

    Each element is either ``list[list[int]]`` (legacy or logprobs
    disabled) or ``(list[list[int]], list[list[float]] | None)`` when
    SamplingParams.return_completion_logprobs is on for any seq in the
    batch. We return ``(token_ids, logprobs)`` where ``logprobs`` is None
    iff *no* DP shard shipped logprobs (matches the scheduler's empty
    fallback that skips Sequence.completion_logprobs population).
    """
    token_ids = []
    logprobs = []
    any_logprobs = False
    for r in per_dp_results:
        if isinstance(r, tuple):
            ids, lp = r
            token_ids.append(ids)
            logprobs.append(lp if lp is not None else [])
            if lp is not None:
                any_logprobs = True
        else:
            token_ids.append(r)
            logprobs.append([])
    return token_ids, (logprobs if any_logprobs else None)


def _build_executor(config: Config):
    if config.executor_backend == "ray":
        from nanodeploy.engine.ray_executor import RayExecutor

        return RayExecutor(config=config)
    if config.executor_backend == "dlslime":
        from nanodeploy.engine.dlslime_executor import DLSLimeExecutor

        return DLSLimeExecutor(config=config)
    raise ValueError(f"Unknown executor backend: {config.executor_backend}")


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

        self.executor = _build_executor(config)
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
        self.weight_barrier = WeightUpdateBarrier()
        self.weight_version: int = 0
        self.last_generate_weight_version: int = 0
        self._streaming_mode = False
        self._admission_paused = False
        self._stream_pending = deque()
        self._seq_policy_version: dict[int, int] = {}

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

    def pull_and_apply_weights(
        self, manifest_blob: bytes, train_alias: str
    ) -> list[dict]:
        """Fast path: each worker pulls its own copy from ``train_alias`` in
        parallel via RDMA, then applies in place.

        ``manifest_blob`` is a pickled ``WeightManifest`` (see
        ``nanorl.weights.transport``). The train side must have already
        registered the corresponding MRs.
        """
        manifest_version = None
        try:
            manifest_version = getattr(pickle.loads(manifest_blob), "version", None)
        except Exception:
            logger.warning("failed to decode weight manifest version", exc_info=True)

        barrier = (
            self.weight_barrier.update_streaming(self)
            if self._streaming_mode
            else self.weight_barrier.update()
        )
        with barrier as barrier_wait_s:
            stats = self.executor.collective_rpc(
                "pull_and_apply_weights",
                (manifest_blob, train_alias),
            )
            if manifest_version is not None:
                self.weight_version = int(manifest_version)
            else:
                self.weight_version += 1
            for row in stats:
                row.setdefault("version", self.weight_version)
                row.setdefault("barrier_wait_s", barrier_wait_s)
        logger.info(
            "engine.pull_and_apply_weights applied version=%s barrier_wait_s=%.3f",
            self.weight_version,
            barrier_wait_s,
        )
        return stats

    def get_weight_version(self) -> int:
        return self.weight_version

    def get_last_generate_weight_version(self) -> int:
        return self.last_generate_weight_version

    def enter_streaming_mode(self) -> None:
        self._streaming_mode = True

    def exit_streaming_mode(self) -> None:
        self._streaming_mode = False
        self._admission_paused = False
        set_paused = getattr(self.scheduler, "set_admission_paused", None)
        if set_paused is not None:
            set_paused(False)
        self._stream_pending.clear()
        self._seq_policy_version.clear()
        self.weight_barrier.notify_step()

    def is_streaming(self) -> bool:
        return self._streaming_mode

    def pause_admission(self) -> None:
        self._admission_paused = True
        set_paused = getattr(self.scheduler, "set_admission_paused", None)
        if set_paused is not None:
            set_paused(True)

    def resume_admission(self) -> None:
        self._admission_paused = False
        set_paused = getattr(self.scheduler, "set_admission_paused", None)
        if set_paused is not None:
            set_paused(False)
        self.weight_barrier.notify_step()

    def is_admission_paused(self) -> bool:
        return self._admission_paused

    def submit_request(self, seqs: Sequence | list[Sequence]) -> None:
        """Queue streaming requests without assigning a policy version yet.

        The version is stamped when the scheduler first selects the sequence
        for execution. A waiting request has not consumed model weights, so it
        should observe whatever version is current when it actually starts.
        """
        if self._admission_paused:
            raise AdmissionPaused("streaming admission is paused for weight update")
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        self._stream_pending.extend(seqs)

    def _add_to_scheduler(self, seq: Sequence) -> None:
        if self.config.mode == "decode":
            logger.info(
                f"[DEBUG] Decode engine received seq {seq.seq_id}: last_token={seq.last_token}, num_tokens={seq.num_tokens}, token_ids_len={len(seq.token_ids)}, token_ids_last10={seq.token_ids[-10:] if seq.token_ids else []}"
            )
        seq.metric = self.metrics_manager.create_sequence_metric(
            seq.seq_id, seq.num_prompt_tokens
        )
        self.scheduler.add(seq)

    def _admit_stream_pending(self) -> int:
        if self._admission_paused:
            return 0
        admitted = 0
        while self._stream_pending:
            self._add_to_scheduler(self._stream_pending.popleft())
            admitted += 1
        return admitted

    def _stamp_scheduled_policy_versions(self, dp_seqs: list) -> None:
        for seqs in dp_seqs:
            for seq in seqs:
                self._seq_policy_version.setdefault(seq.seq_id, self.weight_version)

    def step_once(self) -> StepResult:
        self._admit_stream_pending()
        return self.step()

    def num_pending(self) -> int:
        return len(self._stream_pending)

    def num_scheduler_waiting(self) -> int:
        return len(self.scheduler.waiting) + len(self.scheduler.waiting_migration)

    def num_prefilling(self) -> int:
        return len(getattr(self.scheduler, "prefilling", []))

    def num_running(self) -> int:
        return sum(
            len(self.scheduler.running(dp_idx))
            for dp_idx in range(self.config.attention_dp)
        )

    def num_inflight(self) -> int:
        return self.num_pending() + self.num_scheduler_waiting() + self.num_running()

    def num_active_for_update(self) -> int:
        # Python-side pending and scheduler waiting requests have not consumed
        # model weights yet. C++ admission pause prevents waiting from becoming
        # running during update; only active running/prefilling sequences drain.
        return self.num_prefilling() + self.num_running()

    def policy_version_for(self, seq_id: int) -> int | None:
        return self._seq_policy_version.get(seq_id)

    def ack_sequence(self, seq_id: int) -> None:
        self._seq_policy_version.pop(seq_id, None)

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            self._add_to_scheduler(seq)

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
        if self._streaming_mode:
            self._stamp_scheduled_policy_versions(dp_seqs)

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
        token_logprobs = None
        if not (is_prefill and self.config.mode == "decode"):
            # Normal execution: prefill engine runs prefill, or decode engine runs decode.
            # Each per-DP result is either ``list[list[int]]`` (legacy /
            # logprobs disabled) or ``(list[list[int]], list[list[float]])``
            # when SamplingParams.return_completion_logprobs is on.
            raw = self.executor.run(dp_group_tp_seqs, is_prefill)[::tp_size]
            token_ids, token_logprobs = _split_run_result(raw)
            post_sch_begin = time.time()
            if token_logprobs is not None:
                self.scheduler.postprocess(
                    filtered_dp_group_seqs,
                    token_ids,
                    True,
                    token_logprobs,
                )
            else:
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
        step_result = StepResult(
            dp_seqs=dp_seqs,
            outputs=outputs,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            real_bs=real_bs,
            schedule_latency_ms=(sch_end - sch_begin) * 1000,
            postprocess_latency_ms=(post_sch_end - post_sch_begin) * 1000,
        )
        if self._streaming_mode:
            self.weight_barrier.notify_step()
        return step_result

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
        return_serialized: bool = False,
    ) -> list[Sequence] | list[bytes]:
        with self.weight_barrier.generation():
            self.last_generate_weight_version = self.weight_version
            return self._generate_locked(
                use_tqdm=use_tqdm,
                log_metrics_interval=log_metrics_interval,
                return_serialized=return_serialized,
            )

    def _generate_locked(
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

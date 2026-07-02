import atexit
import time
import uuid
from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

import flatbuffers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dlengine._cpp import Sequence
from dlengine.config import Config
from dlengine.engine.scheduler import ensure_cache_plan, init_scheduler
from dlengine.logging import get_logger, set_log_level
from dlengine.metrics import MetricsManager
from dlengine.metrics.dump import EngineMetricDumper

logger = get_logger()


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
        from dlengine.engine.ray_executor import RayExecutor

        return RayExecutor(config=config)
    if config.executor_backend == "dlslime":
        from dlengine.engine.dlslime_executor import DLSLimeExecutor

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


@dataclass
class PendingStep:
    """In-flight step state between step_begin() and step_complete().

    Carries the schedule output plus the executor handle of the submitted
    forward so the driver can overlap bookkeeping of step N with the GPU
    forward of step N+1 (see run_engine_backend's pipelined loop).
    """

    dp_seqs: list
    schedule_result: Any
    is_prefill: bool
    filtered_dp_group_seqs: list
    scheduler_metric: Any
    sch_begin: float
    sch_end: float
    # executor run_async handle (None = migrate path)
    handle: Optional[dict] = None
    # Filled by step_finish():
    token_ids: Optional[list] = None
    post_sch_begin: float = 0.0
    post_sch_end: float = 0.0
    forward_tx_bytes: int = 0
    forward_rx_bytes: int = 0
    transfer_ms: float = 0.0
    wwi_ms: float = 0.0
    immrecv_ms: float = 0.0
    net_ms: float = 0.0
    serialize_ms: float = 0.0


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

        ensure_cache_plan(config)
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

        # Engine-side request/metric dumper (Redis). Lives here (not in the HTTP
        # server) so it also fires for offline ``generate()`` usage. Captures the
        # exact tokenized prompt at admission and per-request latency (incl.
        # chunk-prefill timing) at completion. No-op unless enabled.
        self._metric_dumper = EngineMetricDumper(
            model=config.model,
            setting=config.dump_requests_redis,
            stream=config.dump_requests_stream,
            maxlen=config.dump_requests_maxlen,
        )

        atexit.register(self.exit)

    def exit(self):
        """Cleanup engine resources."""
        dumper = getattr(self, "_metric_dumper", None)
        if dumper is not None:
            dumper.close()
        if hasattr(self, "executor"):
            del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()
        service_token_capacity = (
            self.config.num_kvcache_blocks * self.config.kvcache_block_size
        )
        if self.config.max_model_len > service_token_capacity:
            logger.warning(
                "max_model_len=%d exceeds KV cache capacity (%d blocks x %d "
                "tokens = %d); clamping effective max_model_len to %d.",
                self.config.max_model_len,
                self.config.num_kvcache_blocks,
                self.config.kvcache_block_size,
                service_token_capacity,
                service_token_capacity,
            )
            self.config.max_model_len = service_token_capacity

    def get_engine_id(self):
        return self.engine_id

    def get_num_kv_blocks(self):
        return self.config.num_kvcache_blocks

    def get_attn_world_size(self):
        return self.config.attn_world_size

    def get_peer_agent_addrs(self) -> list[str]:
        """Get peer agent addresses from all workers."""
        return self.executor.get_peer_agent_addrs()

    def update_weights(self, named_tensors: dict[str, "torch.Tensor"]) -> list[dict]:
        """Apply HF-named full tensors to the live model on every worker.

        Slow path: the dict is shipped to every worker via Ray RPC. For
        large models prefer ``pull_and_apply_weights`` which has each
        worker pull from the train side directly via RDMA.
        """
        from dlengine.engine.weight_sync import update_weights as _update_weights

        return _update_weights(self.executor, named_tensors)

    def pull_and_apply_weights(
        self, manifest_blob: bytes, train_alias: str
    ) -> list[dict]:
        """Fast path: each worker pulls its own copy from ``train_alias`` in
        parallel via RDMA, then applies in place.

        ``manifest_blob`` is a pickled ``WeightManifest`` (see
        ``nanorl.weights.transport``). The train side must have already
        registered the corresponding MRs.
        """
        return self.executor.collective_rpc(
            "pull_and_apply_weights",
            (manifest_blob, train_alias),
        )

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq.metric = self.metrics_manager.create_sequence_metric(
                seq.seq_id, len(seq.prompt_token_ids)
            )
            self._metric_dumper.record_request(seq, self.tokenizer)
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self.scheduler.free_to_be_migrated(seqs)

    def abort(self, seq_ids: int | list[int]) -> list[int]:
        """Stop generating for the given sequences and free their KV blocks.

        Returns the subset of ``seq_ids`` that were actually found and aborted.
        MUST be called between steps (no forward in flight); the backend loop
        enforces this by deferring aborts while a step is pending.
        """
        if isinstance(seq_ids, int):
            seq_ids = [seq_ids]
        aborted: list[int] = []
        for seq_id in seq_ids:
            seq_id = int(seq_id)
            if self.scheduler.abort(seq_id):
                self.scheduler.clear_finished_metric_state(seq_id)
                aborted.append(seq_id)
        return aborted

    def step_begin(self) -> PendingStep:
        """Schedule one step and submit its forward to the workers.

        Returns immediately after the (non-blocking) submit; pair with
        step_finish() + step_complete(). The PD-disagg migrate path (prefill
        request on a decode engine) runs synchronously here since it has no
        forward to overlap.
        """
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.time()
        sch_res = self.scheduler.schedule()
        dp_seqs = sch_res.dp_seqs
        is_prefill = sch_res.is_prefill
        dp_group_seqs = sch_res.dp_group_seqs
        filtered_dp_group_seqs = sch_res.filtered_dp_group_seqs

        scheduler_metric = self.scheduler.update_server_metric(
            self.metrics_manager.server_metric, sch_res
        )

        waiting_migration_head_num_tokens = (
            scheduler_metric.waiting_migration_head_tokens
        )
        if waiting_migration_head_num_tokens >= 0:
            logger.info(f"{waiting_migration_head_num_tokens=}")

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

        waiting_head_blocks = sch_res.waiting_head_blocks
        waiting_total_blocks = sch_res.waiting_total_blocks

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
                "free_blocks": scheduler_metric.free_blocks,
            }
        )

        sch_end = time.time()

        pending = PendingStep(
            dp_seqs=dp_seqs,
            schedule_result=sch_res,
            is_prefill=is_prefill,
            filtered_dp_group_seqs=filtered_dp_group_seqs,
            scheduler_metric=scheduler_metric,
            sch_begin=sch_begin,
            sch_end=sch_end,
        )

        if not (is_prefill and self.config.mode == "decode"):
            # Normal execution: prefill engine runs prefill, or decode engine
            # runs decode. Submit only; step_finish() waits for the replies.
            pending.handle = self.executor.run_async(dp_group_tp_seqs, is_prefill)
        else:
            # PD disaggregation: decode engine receives prefill request
            # DO NOT run prefill on decode engine - KV cache will be migrated from prefill engine
            logger.info(
                f"Decode engine receiving prefill request, skipping local prefill execution"
            )
            # TP-expand so every worker (all dp*sp*tp ranks) receives the
            # migrate batch. With attention_tp > 1 (GQA) each TP rank holds a
            # distinct KV-head shard and must run its own RDMA reads; sending
            # only dp_group_seqs would leave tp_idx > 0 ranks unmigrated.
            self.executor.migrate(dp_group_tp_seqs)
        return pending

    def step_finish(self, pending: PendingStep) -> StepResult:
        """Wait for the submitted forward and postprocess its results.

        Only the work that the next step_begin() depends on lives here (reply
        wait + token append); counting/metrics/heartbeat are deferred to
        step_complete() so they can be overlapped with the next forward.
        """
        tp_size = self.config.attention_tp
        token_ids = None
        token_logprobs = None
        if pending.handle is not None:
            # Each per-DP result is either ``list[list[int]]`` (legacy /
            # logprobs disabled) or ``(list[list[int]], list[list[float]])``
            # when SamplingParams.return_completion_logprobs is on.
            raw = self.executor.run_wait(pending.handle)[::tp_size]
            pending.forward_tx_bytes = getattr(
                self.executor, "last_run_request_bytes", 0
            )
            pending.forward_rx_bytes = getattr(self.executor, "last_run_reply_bytes", 0)
            pending.transfer_ms = getattr(self.executor, "last_run_transfer_ms", 0.0)
            pending.wwi_ms = getattr(self.executor, "last_run_wwi_ms", 0.0)
            pending.immrecv_ms = getattr(self.executor, "last_run_immrecv_ms", 0.0)
            pending.net_ms = getattr(self.executor, "last_run_net_ms", 0.0)
            pending.serialize_ms = getattr(self.executor, "last_run_serialize_ms", 0.0)
            token_ids, token_logprobs = _split_run_result(raw)
            pending.post_sch_begin = time.time()
            if token_logprobs is not None:
                self.scheduler.postprocess(
                    pending.filtered_dp_group_seqs,
                    token_ids,
                    True,
                    token_logprobs,
                )
            else:
                self.scheduler.postprocess(
                    pending.filtered_dp_group_seqs, token_ids, True
                )
            pending.post_sch_end = time.time()
        else:
            pending.post_sch_begin = time.time()
            pending.post_sch_end = pending.post_sch_begin
        pending.token_ids = token_ids

        return StepResult(
            dp_seqs=pending.dp_seqs,
            outputs=[],
            prefill_tokens=0,
            decode_tokens=0,
            real_bs=0,
            schedule_latency_ms=(pending.sch_end - pending.sch_begin) * 1000,
            postprocess_latency_ms=(pending.post_sch_end - pending.post_sch_begin)
            * 1000,
        )

    def step_complete(self, pending: PendingStep, result: StepResult) -> StepResult:
        """Token accounting, finished-seq collection, and heartbeat for one step.

        Pure bookkeeping over already-postprocessed sequences: safe to run
        after the next step has been scheduled and submitted, so the backend
        loop calls this in the shadow of the next GPU forward.
        """
        dp_seqs = pending.dp_seqs
        token_ids = pending.token_ids

        outputs = []
        metric_snapshot = self.scheduler.record_step_metric(
            self.metrics_manager.server_metric,
            pending.schedule_result,
            token_ids or [],
        )
        result.prefill_tokens = metric_snapshot.prefill_tokens
        result.decode_tokens = metric_snapshot.decode_tokens

        # Collect finished/migrated sequences after postprocess
        for seqs in dp_seqs:
            for seq in seqs:
                if seq.is_finished or seq.is_to_be_migrated:
                    self.metrics_manager.complete_sequence(seq.seq_id)
                    # complete_sequence() stamps completion_time; dump after it so
                    # e2e is populated. Only FINISHED requests have full metrics
                    # (migration handoffs are dumped by the decode engine).
                    if self._metric_dumper.enabled and seq.is_finished:
                        self._metric_dumper.record_completion(
                            seq,
                            cached_len=self.scheduler.prefix_cached_tokens(seq.seq_id),
                        )
                    self.scheduler.clear_finished_metric_state(seq.seq_id)
                    outputs.append(seq)
        result.outputs = outputs
        result.real_bs = metric_snapshot.real_bs

        # Periodic engine status report (throttling/accounting live in the
        # metrics manager; this path drives step() directly, bypassing
        # generate(), so it is what produces the serve-mode heartbeat).
        try:
            resource_metric = self.scheduler.metric_snapshot()
            blocks_per_dp = resource_metric.total_blocks_per_dp
            used_blocks_per_dp = resource_metric.used_blocks_per_dp
        except Exception:  # noqa: BLE001
            blocks_per_dp = self.config.num_kvcache_blocks
            used_blocks_per_dp = None
        # Latency breakdown for this step. forward = executor.run/migrate, i.e.
        # the gap between scheduling end and postprocess start.
        forward_latency_ms = (
            (pending.post_sch_begin - pending.sch_end) * 1000
            if pending.post_sch_begin
            else 0.0
        )

        self.metrics_manager.maybe_report_engine_status(
            engine_id=self.engine_id,
            mode=self.config.mode,
            running_per_dp=pending.scheduler_metric.running_per_dp,
            waiting=pending.scheduler_metric.total_waiting,
            waiting_migration=pending.scheduler_metric.total_waiting_migration,
            used_blocks_per_dp=used_blocks_per_dp,
            total_blocks=blocks_per_dp,
            prefill_tokens_per_dp=metric_snapshot.prefill_tokens_per_dp,
            decode_tokens_per_dp=metric_snapshot.decode_tokens_per_dp,
            prefix_cached_tokens_per_dp=metric_snapshot.prefix_cached_tokens_per_dp,
            prefix_prompt_tokens_per_dp=metric_snapshot.prefix_prompt_tokens_per_dp,
            schedule_ms=result.schedule_latency_ms,
            forward_ms=forward_latency_ms,
            postprocess_ms=result.postprocess_latency_ms,
            forward_tx_bytes=pending.forward_tx_bytes,
            forward_rx_bytes=pending.forward_rx_bytes,
            transfer_ms=pending.transfer_ms,
            wwi_ms=pending.wwi_ms,
            immrecv_ms=pending.immrecv_ms,
            net_ms=pending.net_ms,
            serialize_ms=pending.serialize_ms,
        )
        return result

    def step(self):
        """Synchronous one-step execution (schedule + forward + bookkeeping).

        Kept for generate() and other non-pipelined callers; the backend
        server loop uses step_begin/step_finish/step_complete directly to
        overlap driver bookkeeping with the next GPU forward.
        """
        pending = self.step_begin()
        result = self.step_finish(pending)
        return self.step_complete(pending, result)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
        return_serialized: bool = False,
    ) -> list[Sequence] | list[bytes]:
        num_reqs = self.scheduler.num_waiting()
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

            from dlengine._cpp import deserialize, serialize

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

import atexit
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Set

from dlengine.config import Config
from dlengine.engine.scheduler import ensure_cache_plan, init_scheduler
from dlengine.logging import get_logger, set_log_level
from dlengine.metrics import MetricsManager
from dlengine.metrics.dump import EngineMetricDumper
from dlengine.models.trait import load_tokenizer_and_eos

logger = get_logger()


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
    runner_outs: Optional[list] = None
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
        self.config = config
        self.engine_id = self.config.engine_id

        # Set log level globally first
        if self.config.log_level:
            set_log_level(self.config.log_level)

        ensure_cache_plan(config)
        self.executor = _build_executor(config)
        self.update_num_kvcache_blocks()

        self.tokenizer, config.eos = load_tokenizer_and_eos(config.model)

        self.scheduler = init_scheduler(config)
        logger.info(
            f"Initialized Scheduler with RoutingStrategy: {self.scheduler.routing_strategy}"
        )
        self.metrics_manager = MetricsManager()

        # Engine-side request/metric dumper (Redis). Lives here (not in the HTTP
        # server) so it also fires for offline generation usage. Captures the
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

    def add_request_payload(self, payload: bytes):
        added = self.scheduler.add_request_bytes(payload)
        self._register_added_requests(added)
        return added

    def _register_added_requests(self, added):
        for seq_id, prompt_len in added:
            metric = self.metrics_manager.create_sequence_metric(seq_id, prompt_len)
            self.scheduler.set_sequence_metric(seq_id, metric)

    def free_to_be_migrated_ids(self, seq_ids: int | list[int]):
        if isinstance(seq_ids, int):
            seq_ids = [seq_ids]
        self.scheduler.free_to_be_migrated_ids([int(seq_id) for seq_id in seq_ids])

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
        tp_size = self.config.attention_tp
        sch_begin = time.time()
        sch_res = self.scheduler.schedule()
        is_prefill = sch_res.is_prefill
        dp_group_seqs = sch_res.dp_group_seqs

        scheduler_metric = self.scheduler.update_server_metric(
            self.metrics_manager.server_metric, sch_res
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                sch_res.debug_summary(
                    self.config.attention_dp,
                    self.config.attention_sp,
                    scheduler_metric.free_blocks,
                )
            )

        sch_end = time.time()

        pending = PendingStep(
            dp_seqs=sch_res.dp_seqs,
            schedule_result=sch_res,
            is_prefill=is_prefill,
            filtered_dp_group_seqs=sch_res.filtered_dp_group_seqs,
            scheduler_metric=scheduler_metric,
            sch_begin=sch_begin,
            sch_end=sch_end,
        )

        if not (is_prefill and self.config.mode == "decode"):
            # Normal execution: prefill engine runs prefill, or decode engine
            # runs decode. Submit only; step_finish() waits for the replies.
            batch_bytes = self.scheduler.serialize_run_batches(
                dp_group_seqs, is_prefill, tp_size
            )
            pending.handle = self.executor.run_batch_bytes_async(
                batch_bytes, is_prefill
            )
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
            batch_bytes = self.scheduler.serialize_migrate_batches(
                dp_group_seqs, tp_size
            )
            self.executor.migrate_batch_bytes(batch_bytes)
        return pending

    def step_finish(self, pending: PendingStep) -> StepResult:
        """Wait for the submitted forward and postprocess its results.

        Only the work that the next step_begin() depends on lives here (reply
        wait + token append); counting/metrics/heartbeat are deferred to
        step_complete() so they can be overlapped with the next forward.
        """
        tp_size = self.config.attention_tp
        runner_outs = None
        if pending.handle is not None:
            # Each per-DP result is either ``list[list[int]]`` (legacy /
            # logprobs disabled) or ``(list[list[int]], list[list[float]])``
            # when SamplingParams.return_completion_logprobs is on.
            runner_outs = self.executor.run_wait_runner_outs(pending.handle)[::tp_size]
            pending.forward_tx_bytes = getattr(
                self.executor, "last_run_request_bytes", 0
            )
            pending.forward_rx_bytes = getattr(self.executor, "last_run_reply_bytes", 0)
            pending.transfer_ms = getattr(self.executor, "last_run_transfer_ms", 0.0)
            pending.wwi_ms = getattr(self.executor, "last_run_wwi_ms", 0.0)
            pending.immrecv_ms = getattr(self.executor, "last_run_immrecv_ms", 0.0)
            pending.net_ms = getattr(self.executor, "last_run_net_ms", 0.0)
            pending.serialize_ms = getattr(self.executor, "last_run_serialize_ms", 0.0)
            pending.post_sch_begin = time.time()
            self.scheduler.postprocess_runner_outs(
                pending.filtered_dp_group_seqs,
                runner_outs,
                True,
            )
            pending.post_sch_end = time.time()
        else:
            pending.post_sch_begin = time.time()
            pending.post_sch_end = pending.post_sch_begin
        pending.runner_outs = runner_outs
        pending.token_ids = (
            [out.token_ids for out in runner_outs] if runner_outs else None
        )

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

    def step_complete(
        self,
        pending: PendingStep,
        result: StepResult,
        track_running: bool = False,
        previous_running: Set[int] | None = None,
    ) -> StepResult:
        """Token accounting, finished-seq collection, and heartbeat for one step.

        Pure bookkeeping over already-postprocessed sequences: safe to run
        after the next step has been scheduled and submitted, so the backend
        loop calls this in the shadow of the next GPU forward.
        """
        dp_seqs = pending.dp_seqs
        if pending.runner_outs is not None:
            metric_snapshot = self.scheduler.record_step_metric_runner_outs(
                self.metrics_manager.server_metric,
                pending.schedule_result,
                pending.runner_outs,
            )
        else:
            metric_snapshot = self.scheduler.record_step_metric(
                self.metrics_manager.server_metric,
                pending.schedule_result,
                pending.token_ids or [],
            )
        result.prefill_tokens = metric_snapshot.prefill_tokens
        result.decode_tokens = metric_snapshot.decode_tokens

        events = self.scheduler.collect_sequence_events(
            dp_seqs, track_running, previous_running or set()
        )
        for event in events:
            seq_id = event["seq_id"]
            if event["is_finished"] or event["is_to_be_migrated"]:
                self.metrics_manager.complete_sequence(seq_id)
                self.scheduler.clear_finished_metric_state(seq_id)
        result.outputs = events
        result.real_bs = metric_snapshot.real_bs

        # Periodic engine status report (throttling/accounting live in the
        # metrics manager). The serve loop reaches this path directly, so this
        # is what produces the serve-mode heartbeat.
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

        Kept for offline generation and other non-pipelined callers; the
        backend server loop uses step_begin/step_finish/step_complete directly to
        overlap driver bookkeeping with the next GPU forward.
        """
        pending = self.step_begin()
        result = self.step_finish(pending)
        return self.step_complete(pending, result)

    def is_finished(self):
        return self.scheduler.is_finished()

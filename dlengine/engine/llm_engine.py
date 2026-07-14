import atexit
from typing import Set

from dlengine.config import Config
from dlengine.engine.scheduler import ensure_cache_plan, init_scheduler
from dlengine.logging import get_logger, set_log_level
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
            self.scheduler.register_sequence_metric(seq_id, prompt_len)

    def free_to_be_migrated_ids(self, seq_ids: int | list[int]):
        if isinstance(seq_ids, int):
            seq_ids = [seq_ids]
        self.scheduler.free_to_be_migrated_ids([int(seq_id) for seq_id in seq_ids])

    def abort(self, seq_ids: int | list[int]) -> list[int]:
        """Stop generating for the given sequences and free their KV blocks.

        Returns the subset of ``seq_ids`` that were actually found and aborted.
        The serial backend loop handles aborts between engine steps, so no
        in-flight forward can still touch the freed KV blocks.
        """
        if isinstance(seq_ids, int):
            seq_ids = [seq_ids]
        return self.scheduler.abort_many([int(seq_id) for seq_id in seq_ids])

    def _run_scheduled_step(self, schedule_result):
        tp_size = self.config.attention_tp
        pp_size = self.config.pp
        runner_outs = None
        self._run_host_swap_outs(schedule_result, tp_size)
        self._run_host_swap_ins(schedule_result, tp_size)

        if not (schedule_result.is_prefill and self.config.mode == "decode"):
            batch_bytes = self.scheduler.serialize_run_batches_for_result(
                schedule_result, tp_size
            )
            # Every pipeline stage runs the same batch (each stage forwards its
            # own layers); replicate the per-inner-rank bytes across stages so
            # the flat worker list [stage0 ranks..., stage1 ranks..., ...] lines
            # up with the batch list.
            inner = len(batch_bytes)
            batch_bytes = self._replicate_for_pp(batch_bytes, pp_size)
            handle = self.executor.run_batch_bytes_async(
                batch_bytes, schedule_result.is_prefill
            )
            all_outs = self.executor.run_wait_runner_outs(handle)
            # Tokens are produced only by the last pipeline stage; its workers
            # occupy the final ``inner`` slots of the worker list.
            last_stage_outs = all_outs[(pp_size - 1) * inner :]
            runner_outs = last_stage_outs[::tp_size]
            self.scheduler.postprocess_schedule_runner_outs(
                schedule_result,
                runner_outs,
                True,
            )
        else:
            logger.info("Decode engine receiving prefill request, migrating KV only")
            batch_bytes = self.scheduler.serialize_migrate_batches_for_result(
                schedule_result, tp_size
            )
            batch_bytes = self._replicate_for_pp(batch_bytes, pp_size)
            self.executor.migrate_batch_bytes(batch_bytes)

        return runner_outs

    @staticmethod
    def _replicate_for_pp(items, pp_size: int):
        """Replicate a per-inner-rank list across pipeline stages.

        Workers are ordered pp-major (all of stage 0's inner ranks, then stage
        1's, ...), so repeating the inner list ``pp_size`` times aligns each
        stage's workers with the same batch/task payloads.
        """
        if pp_size <= 1:
            return items
        return list(items) * pp_size

    def _run_host_swap_outs(self, schedule_result, tp_size: int) -> None:
        tasks = getattr(schedule_result, "swap_out_tasks", None) or []
        if not any(tasks):
            return
        num_seqs = sum(len(group_tasks) for group_tasks in tasks)
        num_blocks = sum(
            len(gpu_blocks)
            for group_tasks in tasks
            for _seq_id, gpu_blocks, _host_blocks in group_tasks
        )
        logger.info("host swap-out begin: seqs=%d blocks=%d", num_seqs, num_blocks)
        per_worker_tasks = []
        for group_tasks in tasks:
            for _ in range(max(1, tp_size)):
                per_worker_tasks.append(group_tasks)
        # Each pipeline stage holds its own KV shard for the same block ids, so
        # every stage must perform the swap.
        per_worker_tasks = self._replicate_for_pp(per_worker_tasks, self.config.pp)
        self.executor.swap_out_blocks_to_host(per_worker_tasks)
        self.scheduler.complete_host_swap_outs(tasks)
        logger.info("host swap-out done: seqs=%d blocks=%d", num_seqs, num_blocks)

    def _run_host_swap_ins(self, schedule_result, tp_size: int) -> None:
        tasks = getattr(schedule_result, "swap_in_tasks", None) or []
        if not any(tasks):
            return
        num_seqs = sum(len(group_tasks) for group_tasks in tasks)
        num_blocks = sum(
            len(host_blocks)
            for group_tasks in tasks
            for _seq_id, host_blocks, _gpu_blocks in group_tasks
        )
        logger.info("host swap-in begin: seqs=%d blocks=%d", num_seqs, num_blocks)
        per_worker_tasks = []
        for group_tasks in tasks:
            for _ in range(max(1, tp_size)):
                per_worker_tasks.append(group_tasks)
        per_worker_tasks = self._replicate_for_pp(per_worker_tasks, self.config.pp)
        self.executor.swap_in_blocks_from_host(per_worker_tasks)
        self.scheduler.complete_host_swap_ins(tasks)
        logger.info("host swap-in done: seqs=%d blocks=%d", num_seqs, num_blocks)

    def step(
        self,
        track_running: bool = False,
        previous_running: Set[int] | None = None,
    ):
        """Run one engine step serially: schedule, execute, postprocess, report."""
        schedule_result = self.scheduler.schedule()
        self.scheduler.record_schedule_metrics(schedule_result)

        runner_outs = self._run_scheduled_step(schedule_result)
        result = self.scheduler.record_complete_step(
            schedule_result,
            track_running,
            previous_running or set(),
            runner_outs,
        )
        if result.status_message:
            logger.info(result.status_message)
        for message in result.log_messages:
            logger.info(message)
        return result

    def is_finished(self):
        return self.scheduler.is_finished()

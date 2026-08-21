import base64
import os
import time

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.profiler as profiler

from nanodeploy._cpp import (
    BlockContextSlot,
    prepare_decode_cpp,
    prepare_prefill_cpp,
    update_seqs_inner_loop,
)
from nanodeploy.config import Config
from nanodeploy.endpoint.rpc_endpoint import RPCClientEndpoint
from nanodeploy.engine.sequence import Sequence
from nanodeploy.engine.worker_transport import (
    DecodeCommand,
    WorkerExecutionOutput,
    WorkerZmqConfig,
    ZmqWorkerClient,
)
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger
from nanodeploy.models.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.models.qwen3 import Qwen3ForCausalLM
from nanodeploy.models.qwen3_moe import Qwen3MoeForCausalLM
from nanodeploy.worker.cache import get_cache_context, set_cache_context
from nanodeploy.worker.context import get_context, reset_context, set_context
from nanodeploy.worker.distributed import (
    get_dist_context,
    get_local_ip,
    set_dist_context,
)
from nanodeploy.worker.decode_backend_compat import (
    resolve_decode_deepep_config,
    validate_decode_backend_compat,
    validate_deepseek_decode_contract,
)
from nanodeploy.worker.ep_context import (
    destroy_ep_context,
    set_ep_context,
)
from nanodeploy.worker.loader import load_model
from nanodeploy.worker.mla_metadata import prepare_decode_mla_metadata
from nanodeploy.worker.prefill_logits import compute_prefill_logits
from nanodeploy.worker.random_seed import set_random_seed
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config
from nanodeploy.worker.sp_context import set_sp_context
from nanodeploy.worker.sp_graph_policy import materialize_sp_graph_padding

logger = get_logger()


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


architectures = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(
        self,
        config: Config,
        rank: int,
        engine_local_rank: int | None = None,
    ):
        self.config = config
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.cuda_graph_mode = config.cuda_graph_mode
        self.world_size = config.attn_world_size
        self.rank = rank
        self.engine_local_rank = (
            rank if engine_local_rank is None else engine_local_rank
        )
        self.log_decode_a2a_masks = _env_flag_enabled(
            "NANODEPLOY_LOG_DECODE_A2A_MASKS", default=False
        )
        self._deepep_enabled = False
        self._deepep_destroyed = False
        self._exited = False

        logger.debug(f"init ModelRunner, {rank=}, {get_local_ip()=}")

        set_runner_config(
            max_num_seqs=config.max_num_seqs,
            dummy_weight=config.dummy_weight,
            perfect_eplb=config.perfect_eplb,
            moe_routing_simulation_strategy=(
                config.moe_routing_simulation_strategy
            ),
            seed=config.seed,
        )

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            f"tcp://{config.master_address}",
            world_size=self.world_size,
            rank=rank,
        )

        set_dist_context(
            rank=rank,
            world_size=config.attn_world_size,
            attention_dp=config.attention_dp,
            attention_sp=config.attention_sp,
            attention_tp=config.attention_tp,
            ffn_dp=config.ffn_dp,
            ffn_ep=config.ffn_ep,
            ffn_tp=config.ffn_tp,
        )

        torch.cuda.set_device(0)
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        sp_size = get_dist_context().attn_sp_world_size
        ep_size = get_dist_context().ffn_ep_world_size

        if ep_size > 1:
            self._configure_decode_deepep(ep_size)
            dist.barrier(group=get_dist_context().cuda_world_group)

        if sp_size > 1:
            sp_rank = get_dist_context().attn_sp_rank
            max_head_dim = 0
            if self.config.hf_config.num_key_value_heads > 1:
                max_head_dim = self.config.hf_config.head_dim
            else:
                max_head_dim = (
                    self.config.hf_config.kv_lora_rank
                    + self.config.hf_config.qk_rope_head_dim
                )
            set_sp_context(
                max_num_seqs=config.max_num_seqs,
                head_size=max_head_dim,
                num_attention_heads=hf_config.num_attention_heads,
                dtype=torch.get_default_dtype(),
                rank=sp_rank,
                sp_size=sp_size,
                backend=config.sp_backend,
            )

        self.run_count = 0
        self.profiler = None
        self.profiler_start_time = None
        self.profiler_use_time = False
        if getattr(config, "enable_profiler", False):
            # Check if using time-based profiling
            profiler_start_time = getattr(config, "profiler_start_time", None)
            profiling_duration = getattr(config, "profiling_duration", None)

            if profiler_start_time is not None and profiling_duration is not None:
                # Time-based profiling mode
                self.profiler_use_time = True
                self.profiler_start_time = profiler_start_time
                self.profiling_duration = profiling_duration
                self.profiler_start_timestamp = None  # Will be set when profiling starts
                self.profiler_end_timestamp = None  # Will be set when profiling starts
                self.profiler_stopped = False  # Track if profiler has been stopped
                self.profiler_step_count = 0  # Count profiler steps for logging
                logger.info(
                    f"Rank {rank}: Profiler enabled (time-based). Start after {self.profiler_start_time}s, duration {self.profiling_duration}s."
                )
            else:
                # Step-based profiling mode (original)
                self.profiler_use_time = False
                self.profiler_start_step = getattr(config, "profiler_start_step", 10)
                self.profiler_steps = getattr(config, "profiling_step", 10)
                self.profiler_end_step = self.profiler_start_step + self.profiler_steps
                self.profiler_stopped = False  # Track if profiler has been stopped
                self.profiler_step_count = 0  # Count profiler steps for logging
                logger.info(
                    f"Rank {rank}: Profiler enabled (step-based). Start at {self.profiler_start_step}, duration {self.profiler_steps} steps."
                )

            profiler_dir = getattr(config, "profiler_dir", "./profiler_logs")
            os.makedirs(profiler_dir, exist_ok=True)

            # Store profiler directory for logging
            self.profiler_dir = profiler_dir
            self.profiler_worker_name = f"{self.engine_id}_rank_{self.rank}"

            self.profiler = torch.profiler.profile(
                activities=[
                    # torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=profiler_dir,
                    worker_name=self.profiler_worker_name,
                    use_gzip=False,
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

            logger.info(
                f"Rank {rank}: Profiler initialized. Directory: {profiler_dir}, Worker name: {self.profiler_worker_name}"
            )

            # Record the start time for time-based profiling
            if self.profiler_use_time:
                self.profiler_init_time = time.time()
                logger.info(
                    f"Rank {rank}: Profiler init time recorded: {self.profiler_init_time:.2f}"
                )

        model_architecture = hf_config.architectures[0]
        self.model = architectures[model_architecture](hf_config)

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        dist.barrier()

        self.sampler = Sampler()
        # self.warmup_model()
        self.preallocate_kvcache()

        self.endpoint = RPCClientEndpoint(
            32 * 32_000_000, self.engine_local_rank
        )
        self._zmq_worker_config: WorkerZmqConfig | None = None

    def _configure_decode_deepep(self, ep_size: int) -> None:
        versions: dict[str, str] = {}
        effective = None
        buffer_settings = None
        setup_error = None
        try:
            effective = resolve_decode_deepep_config(
                self.config.max_num_seqs
            )
            os.environ.update(effective.worker_env())
            os.environ["DEEPEP_MODE"] = effective.mode

            versions = validate_decode_backend_compat(self.rank)
            validate_deepseek_decode_contract(self.config.hf_config, ep_size)
            num_experts = int(
                getattr(self.config.hf_config, "n_routed_experts", None)
                or getattr(self.config.hf_config, "num_experts", ep_size)
            )
            if num_experts <= 0 or num_experts % ep_size != 0:
                raise ValueError(
                    f"num_experts={num_experts} must be positive and "
                    f"divisible by ep_size={ep_size}"
                )
            hidden_size = int(self.config.hf_config.hidden_size)
            if hidden_size <= 0:
                raise ValueError(
                    f"hidden_size must be positive; got {hidden_size}"
                )
            local_experts = num_experts // ep_size
            buffer_settings = {
                "num_experts": num_experts,
                "local_experts": local_experts,
                "hidden_size": hidden_size,
                "num_qps_per_rank": max(
                    effective.num_sms, local_experts
                ),
            }
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        local_report = {
            "rank": self.rank,
            "versions": versions,
            "deepep": (
                effective.fingerprint_payload()
                if effective is not None
                else None
            ),
            "buffer": buffer_settings,
            "error": setup_error,
        }
        reports: list[dict[str, object] | None] = [
            None
        ] * get_dist_context().cpu_world_size
        dist.all_gather_object(
            reports,
            local_report,
            group=get_dist_context().cpu_world_group,
        )
        if any(report is None for report in reports):
            raise RuntimeError(
                f"rank {self.rank} received an incomplete decode backend report"
            )
        complete_reports = [report for report in reports if report is not None]
        failures = [
            report
            for report in complete_reports
            if report["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "decode backend compatibility failed across ranks: "
                + "; ".join(
                    f"rank {report['rank']}: {report['error']}"
                    for report in failures
                )
            )
        reference = {
            "versions": complete_reports[0]["versions"],
            "deepep": complete_reports[0]["deepep"],
            "buffer": complete_reports[0]["buffer"],
        }
        mismatches = [
            report
            for report in complete_reports[1:]
            if {
                "versions": report["versions"],
                "deepep": report["deepep"],
                "buffer": report["buffer"],
            }
            != reference
        ]
        if mismatches:
            raise RuntimeError(
                "decode backend versions or DeepEP settings differ across ranks: "
                f"{complete_reports}"
            )

        assert effective is not None
        assert buffer_settings is not None
        ep_context = set_ep_context(
            ep_group=get_dist_context().ffn_ep_group,
            ep_size=ep_size,
            num_experts=buffer_settings["num_experts"],
            hidden_size=buffer_settings["hidden_size"],
            max_tokens_per_rank=effective.max_tokens_per_rank,
            num_sms=effective.num_sms,
            allow_mnnvl=effective.enable_mnnvl,
            nvshmem_qp_depth=effective.nvshmem_qp_depth,
        )
        self._deepep_enabled = True
        self._deepep_config = effective
        logger.info(
            "Rank %d decode backend: deep_gemm=%s deep_ep=%s "
            "ep_size=%d local_experts=%d DEEPEP_SMS=%d "
            "DEEPEP_MAX_TOKENS_PER_RANK=%d DEEPEP_ENABLE_MNNVL=%d "
            "DEEPEP_MODE=%s NVSHMEM_QP_DEPTH=%d num_qps_per_rank=%d "
            "num_nvl_bytes=%d num_rdma_bytes=%d topk_idx_t=%s "
            "SLIME_QP_NUM=%s",
            self.rank,
            versions["deep_gemm"],
            versions["deep_ep"],
            ep_size,
            ep_context.num_local_experts,
            effective.num_sms,
            effective.max_tokens_per_rank,
            int(effective.enable_mnnvl),
            effective.mode,
            effective.nvshmem_qp_depth,
            ep_context.num_qps_per_rank,
            ep_context.num_nvl_bytes,
            ep_context.num_rdma_bytes,
            ep_context.topk_idx_t,
            os.getenv("SLIME_QP_NUM", "<unset>"),
        )

    def init_rpc_endpoint(self, server_info):
        client_info = self.endpoint.init_client_endpoint()
        self.endpoint.connect(server_info)
        logger.info("client endpoint initialized")
        return client_info

    def configure_zmq_worker_transport(
        self, *, transport_config: WorkerZmqConfig
    ) -> None:
        if self._zmq_worker_config is not None:
            raise RuntimeError("worker ZMQ transport is already configured")
        expected_engine_id = self.rank // (
            self.config.attention_sp * self.config.attention_tp
        )
        if transport_config.engine_id != expected_engine_id:
            raise ValueError(
                "worker ZMQ engine mismatch: "
                f"expected={expected_engine_id}, "
                f"got={transport_config.engine_id}"
            )
        if transport_config.global_rank != self.rank:
            raise ValueError(
                "worker ZMQ rank mismatch: "
                f"expected={self.rank}, got={transport_config.global_rank}"
            )
        self._zmq_worker_config = transport_config

    def run_zmq_loop(self) -> None:
        transport_config = self._zmq_worker_config
        if transport_config is None:
            raise RuntimeError("worker ZMQ transport is not configured")

        def execute(command: DecodeCommand) -> WorkerExecutionOutput:
            trace_enabled = command.hierarchical_trace is not None
            if trace_enabled != bool(
                self.config.hierarchical_execution_trace
            ):
                raise RuntimeError(
                    "worker ZMQ execution-trace configuration mismatch"
                )
            if command.hierarchical_quantum_diagnostics != bool(
                self.config.hierarchical_quantum_diagnostics
            ):
                raise RuntimeError(
                    "worker ZMQ quantum-diagnostic configuration mismatch"
                )
            raw = self.run(
                dp_seqs=[],
                is_prefill=False,
                enable_rpc=True,
                send_timestamp=command.send_timestamp,
                hierarchical_trace=command.hierarchical_trace,
                hierarchical_quantum_diagnostics=(
                    command.hierarchical_quantum_diagnostics
                ),
            )
            expected_len = (
                2
                + int(trace_enabled)
                + int(command.hierarchical_quantum_diagnostics)
            )
            if not isinstance(raw, tuple) or len(raw) != expected_len:
                raise RuntimeError(
                    "worker decode returned an invalid result envelope: "
                    f"expected tuple length {expected_len}"
                )
            token_rows, worker_end_time = raw[:2]
            extra_index = 2
            trace = None
            diagnostic = None
            if trace_enabled:
                trace = raw[extra_index]
                extra_index += 1
            if command.hierarchical_quantum_diagnostics:
                diagnostic = raw[extra_index]
            return WorkerExecutionOutput(
                token_rows=token_rows,
                worker_end_time=float(worker_end_time),
                hierarchical_trace=trace,
                diagnostic=diagnostic,
            )

        ZmqWorkerClient(transport_config).run(execute)

    def num_kvcache_blocks(self):
        return self.config.num_kvcache_blocks

    def get_worker_identity(self):
        return {
            "global_rank": self.rank,
            "engine_local_rank": self.engine_local_rank,
            "node_id": ray.get_runtime_context().get_node_id(),
            "gpu_ids": tuple(ray.get_gpu_ids()),
            "config_fingerprint": self.config.collective_fingerprint(),
        }

    def allocate_kvcache(self, num_kvcache_blocks: int):
        self.config.num_kvcache_blocks = num_kvcache_blocks
        cache_context = get_cache_context()
        cache_context.allocate_kvcache(num_kvcache_blocks)
        layer_id = 0
        for module in self.model.modules():
            allocated = False
            if hasattr(module, "k_cache"):
                module.k_cache = cache_context.kv_cache[0][layer_id]
                allocated = True
            if hasattr(module, "v_cache"):
                if cache_context.kv_cache.size(0) > 1:
                    module.v_cache = cache_context.kv_cache[1][layer_id]
                else:
                    module.v_cache = torch.tensor([], device=cache_context.device)
                allocated = True
            if allocated:
                layer_id += 1
        if self.config.moe_routing_simulation_strategy == "uniform_random":
            set_random_seed(self.config.seed)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(self.default_dtype)

    def zero_kvcache(self):
        cache_context = get_cache_context()
        cache_context.kv_cache.zero_()
        torch.cuda.synchronize()
        return cache_context.kv_cache.numel()

    def p2p_init(self, remote_engine_id, num_kv_blocks, remote_engine_world_size):
        return get_cache_context().p2p_init(
            remote_engine_id, num_kv_blocks, remote_engine_world_size
        )

    def p2p_connect(
        self, remote_engine_id: str, endpoints_info_list: list[dict[int, dict]]
    ):
        return get_cache_context().p2p_connect(remote_engine_id, endpoints_info_list)

    def exit(self):
        if self._exited:
            return
        if not self.enforce_eager:
            if self.cuda_graph_mode == "piecewise":
                graph_attributes = (
                    "piecewise_graphs",
                    "piecewise_graph_vars",
                    "graph_pool",
                )
            else:
                graph_attributes = (
                    "local_graphs",
                    "sp_graphs",
                    "sp_graph_map",
                    "graph_pool",
                )
            for name in graph_attributes:
                if hasattr(self, name):
                    delattr(self, name)

        torch.cuda.synchronize()
        cleanup_error: BaseException | None = None
        if self._deepep_enabled and not self._deepep_destroyed:
            try:
                destroyed = destroy_ep_context()
                if destroyed:
                    logger.info("Rank %d explicitly destroyed DeepEP", self.rank)
                else:
                    logger.info(
                        "Rank %d DeepEP buffer was already destroyed; "
                        "no runtime needed destruction",
                        self.rank,
                    )
                self._deepep_destroyed = True
            except BaseException as exc:
                cleanup_error = exc

        if dist.is_initialized():
            if self._deepep_enabled:
                try:
                    dist.barrier(group=get_dist_context().cuda_world_group)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            try:
                dist.destroy_process_group()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        self._exited = True
        if cleanup_error is not None:
            raise RuntimeError(
                f"rank {self.rank} failed to cleanly shut down DeepEP"
            ) from cleanup_error

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            min(self.config.max_num_batched_tokens, 16384),
            min(self.config.max_model_len, 8192),
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        seqs = []
        for _ in range(num_seqs):
            seq = Sequence(
                list(np.random.randint(low=0, high=10000, size=max_model_len))
            )
            seq.active(self.engine_id, sp_size, 1)
            seq.block_ctx().master_sp_idx = sp_rank

        self.run(seqs, True)
        torch.cuda.empty_cache()

    def preallocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config

        mode = "gqa" if hf_config.num_key_value_heads > 1 else "mla"
        kv_lora_rank = (
            hf_config.kv_lora_rank if hasattr(hf_config, "kv_lora_rank") else 0
        )
        qk_rope_head_dim = (
            hf_config.qk_rope_head_dim if hasattr(hf_config, "qk_rope_head_dim") else 0
        )

        cache_context = set_cache_context(
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            block_size=config.kvcache_block_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            attention_tp=config.attention_tp,
            gpu_memory_utilization=config.gpu_memory_utilization,
            gpu_memory_limit_gb=config.gpu_memory_limit_gb,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
            mode=mode,
        )
        config.num_kvcache_blocks = cache_context.num_local_kvcache_blocks

    def prepare_prefill(self, seqs: list[Sequence], is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        meta = prepare_prefill_cpp(
            seqs, sp_rank, sp_size, block_size, self.config.max_num_seqs
        )
        if not is_dummy and len(meta.input_ids) != len(meta.slot_mapping):
            raise RuntimeError(
                "Non-dummy prefill metadata has mismatched KV rows: "
                f"rank={self.rank}, sp_rank={sp_rank}, "
                f"input_ids={len(meta.input_ids)}, "
                f"slot_mapping={len(meta.slot_mapping)}"
            )

        input_ids = torch.tensor(
            meta.input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            meta.positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(
            meta.cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            meta.cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            meta.slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = None
        hf_config = self.config.hf_config
        is_mla = (
            getattr(hf_config, "num_key_value_heads", 0) == 1
            and hasattr(hf_config, "kv_lora_rank")
            and hasattr(hf_config, "qk_rope_head_dim")
        )
        num_prefill_seqs = len(meta.cu_seqlens_q) - 1
        prefill_has_prefix = any(
            meta.cu_seqlens_k[seq_idx + 1] - meta.cu_seqlens_k[seq_idx]
            != meta.cu_seqlens_q[seq_idx + 1] - meta.cu_seqlens_q[seq_idx]
            for seq_idx in range(num_prefill_seqs)
        )
        if meta.use_block_tables:
            all_block_tables = torch.tensor(
                meta.block_tables_flat, dtype=torch.int32, pin_memory=True
            ).reshape(sp_size, self.config.max_num_seqs, meta.max_num_blocks)
            if is_mla:
                # FlashMLA consumes the local packed batch rather than the
                # scheduler's dense [SP, max_num_seqs, blocks] layout.
                all_block_tables = all_block_tables[
                    sp_rank, :num_prefill_seqs
                ].contiguous()
            block_tables = all_block_tables.cuda(non_blocking=True)
        elif is_mla and not is_dummy:
            # A cache-less prefill still runs FlashMLA against the paged cache
            # after store_kcache. Reconstruct each sequence's block table from
            # its token-to-slot mapping; prefix-cache prefills use the dense
            # block tables supplied above instead.
            block_table_rows: list[list[int]] = []
            for seq_idx in range(num_prefill_seqs):
                start = meta.cu_seqlens_q[seq_idx]
                end = meta.cu_seqlens_q[seq_idx + 1]
                row: list[int] = []
                for slot in meta.slot_mapping[start:end]:
                    block_id = slot // block_size
                    if not row or row[-1] != block_id:
                        row.append(block_id)
                if not row:
                    raise RuntimeError(
                        "Non-dummy MLA prefill sequence has no KV-cache blocks: "
                        f"rank={self.rank}, sp_rank={sp_rank}, seq_idx={seq_idx}"
                    )
                block_table_rows.append(row)

            max_num_blocks = max(map(len, block_table_rows), default=0)
            padded_block_tables = [
                row + [-1] * (max_num_blocks - len(row))
                for row in block_table_rows
            ]
            block_tables = torch.tensor(
                padded_block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)

        set_context(
            True,
            self.config.max_num_seqs,
            cu_seqlens_q,
            cu_seqlens_k,
            meta.max_seqlen_q,
            meta.max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            None,
            is_dummy=is_dummy,
            prefill_cu_seqlens_q_host=tuple(meta.cu_seqlens_q),
            prefill_has_prefix=prefill_has_prefix,
        )
        return input_ids, positions

    def prepare_decode(self, dp_seqs: list[Sequence], is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        meta = prepare_decode_cpp(
            dp_seqs,
            sp_rank,
            sp_size,
            block_size,
            self.config.max_num_seqs,
        )
        sp_master_batch_sizes = [0 for _ in range(sp_size)]
        for seq in dp_seqs:
            master_sp_idx = seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
            if 0 <= master_sp_idx < sp_size:
                sp_master_batch_sizes[master_sp_idx] += 1
        sp_comm_bs = max(sp_master_batch_sizes, default=0)

        input_ids = torch.tensor(
            meta.input_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            meta.positions, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            meta.slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        context_lens = (
            torch.tensor(meta.context_lens_flat, dtype=torch.int32, pin_memory=True)
            .reshape(sp_size, self.config.max_num_seqs)
            .cuda(non_blocking=True)
        )

        global_context_lens = (
            torch.tensor(
                meta.global_context_lens_flat, dtype=torch.int32, pin_memory=True
            )
            .reshape(sp_size, self.config.max_num_seqs)
            .cuda(non_blocking=True)
        )

        if len(meta.block_tables_flat) == 0:
            block_tables = torch.empty((0, 0), dtype=torch.int32).cuda(non_blocking=True)
        else:
            block_tables = torch.tensor(
                meta.block_tables_flat, dtype=torch.int32, pin_memory=True
            ).reshape(-1, meta.max_num_blocks).cuda(non_blocking=True)

        q_mask = global_context_lens.clone()
        q_mask[sp_rank].fill_(0)
        q_mask[q_mask != 0] = 1
        res_lse_mask = context_lens.clone()
        res_lse_mask[sp_rank].fill_(0)
        res_lse_mask[res_lse_mask != 0] = 1

        use_sp_a2a = meta.use_sp_a2a
        context_lens_for_attn = torch.tensor(
            meta.context_lens_for_attn, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        q_slice_get = torch.tensor(
            meta.q_slice_get, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        q_slice_fill = torch.tensor(
            meta.q_slice_fill, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        q_copy_mask = torch.tensor(
            meta.q_copy_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_get_to_buffer_output = torch.tensor(
            meta.res_slice_get_to_buffer_output, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_fill_to_buffer_output = torch.tensor(
            meta.res_slice_fill_to_buffer_output, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_to_buffer_output_mask = torch.tensor(
            meta.res_to_buffer_output_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_get_to_buffer_input = torch.tensor(
            meta.res_slice_get_to_buffer_input, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_slice_fill_to_buffer_input = torch.tensor(
            meta.res_slice_fill_to_buffer_input, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        res_to_buffer_input_mask = torch.tensor(
            meta.res_to_buffer_input_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        q_offsets = torch.tensor(
            meta.q_offsets, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        use_hao_destination_rows = (
            sp_size > 1
            and self.config.sp_backend == "hao_basic"
        )
        q_dst_row_indices = None
        if use_hao_destination_rows:
            q_dst_row_indices = (
                torch.tensor(
                    meta.q_dst_row_indices_flat,
                    dtype=torch.int32,
                    pin_memory=True,
                )
                .reshape(sp_size, self.config.max_num_seqs)
                .cuda(non_blocking=True)
            )
        attention_compute_bs = (
            context_lens_for_attn.numel() if use_sp_a2a else input_ids.size(0)
        )

        set_context(
            is_prefill=False,
            max_bs=self.config.max_num_seqs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            q_mask=q_mask,
            res_lse_mask=res_lse_mask,
            use_sp_a2a=use_sp_a2a,
            is_dummy=is_dummy,
            context_lens_for_attn=context_lens_for_attn,
            attention_compute_bs=attention_compute_bs,
            sp_comm_bs=sp_comm_bs,
            q_slice_get=q_slice_get,
            q_slice_fill=q_slice_fill,
            q_copy_mask=q_copy_mask,
            res_slice_get_to_buffer_output=res_slice_get_to_buffer_output,
            res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output,
            res_to_buffer_output_mask=res_to_buffer_output_mask,
            res_slice_get_to_buffer_input=res_slice_get_to_buffer_input,
            res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input,
            res_to_buffer_input_mask=res_to_buffer_input_mask,
            q_offsets=q_offsets,
            q_dst_row_indices=q_dst_row_indices,
            tile_scheduler_metadata=None,
            num_splits=None,
        )

        return input_ids, positions

    def update_decode(
        self, input_ids: torch.Tensor, positions: torch.Tensor, dp_seqs: list[Sequence]
    ):
        # update position
        positions.add_(1)

        sp_rank = get_dist_context().attn_sp_rank
        num_sp_seqs = sum(
            1
            for seq in dp_seqs
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank
        )

        # update slot mapping
        slot_mapping = []
        for seq in dp_seqs:
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank:
                slot_mapping.append(
                    seq.last_block_page_id(BlockContextSlot.ACTIVE, sp_rank)
                    * get_cache_context().block_size
                    + seq.last_block_num_tokens(BlockContextSlot.ACTIVE, sp_rank)
                    - 1
                )

        # update context
        context = get_context()

        context.slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # update context length
        context.context_lens[sp_rank][:num_sp_seqs].add_(1)

        # update global context length
        context.global_context_lens[sp_rank][:num_sp_seqs].add_(1)

        # update context lens for attention
        context.context_lens_for_attn[context.q_slice_fill.long()] += 1

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            if (
                seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
                == get_dist_context().attn_sp_rank
            ):
                temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    def _pack_binary_mask(self, mask: torch.Tensor) -> list[str]:
        packed = np.packbits(mask.detach().cpu().numpy().astype(np.uint8), axis=1)
        return [base64.b64encode(row.tobytes()).decode("ascii") for row in packed]

    def _log_decode_a2a_masks(self, loop_idx: int, is_dummy: bool) -> None:
        context = get_context()
        if context.use_sp_a2a is not True:
            return
        if context.q_mask is None or context.res_lse_mask is None or context.q_offsets is None:
            return

        dist_context = get_dist_context()
        q_dst_row_indices = getattr(context, "q_dst_row_indices", None)
        logger.info(
            {
                "mode": "decode_a2a_masks",
                "global_run_count": self.run_count,
                "loop_idx": loop_idx,
                "global_rank": self.rank,
                "dp_rank": dist_context.attn_dp_rank,
                "sp_rank": dist_context.attn_sp_rank,
                "tp_rank": dist_context.attn_tp_rank,
                "cp_size": dist_context.attn_sp_world_size,
                "max_bs": int(context.q_mask.shape[1]),
                "sp_comm_bs": context.sp_comm_bs,
                "is_dummy": is_dummy,
                "mask_encoding": "bit_b64",
                "q_offsets": context.q_offsets.detach().cpu().tolist(),
                "q_dst_row_indices": (
                    q_dst_row_indices.detach().cpu().tolist()
                    if q_dst_row_indices is not None
                    else None
                ),
                "q_mask": self._pack_binary_mask(context.q_mask),
                "res_lse_mask": self._pack_binary_mask(context.res_lse_mask),
            }
        )

    def _select_decode_graph_master_bs(self, bs: int, context) -> int:
        master_bs = next(x for x in self.graph_master_rank_bs if x >= bs)
        if (
            context.use_sp_a2a
            and (
                self.config.sp_backend == "nccl"
                or self.config.fixed_sp_size > 0
            )
            and context.sp_comm_bs is not None
        ):
            comm_min_master_bs = max(bs, context.sp_comm_bs)
            try:
                master_bs = next(
                    x for x in self.graph_master_rank_bs if x >= comm_min_master_bs
                )
            except StopIteration:
                raise RuntimeError(
                    f"SP communication batch {comm_min_master_bs} exceeds "
                    f"max captured master_bs ({self.graph_master_rank_bs[-1]})"
                )
        return master_bs

    def _copy_decode_context_to_graph_vars(
        self,
        graph_vars: dict[str, torch.Tensor | None],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        bs: int,
        master_bs: int,
        graph_attn_bs: int,
        context,
    ) -> None:
        if graph_vars.get("input_ids") is not None:
            graph_vars["input_ids"].zero_()
            graph_vars["input_ids"][:bs] = input_ids
        if graph_vars.get("positions") is not None:
            graph_vars["positions"].zero_()
            graph_vars["positions"][:bs] = positions

        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][: context.slot_mapping.shape[0]] = context.slot_mapping  # type: ignore

        graph_q_dst_row_indices = graph_vars.get("q_dst_row_indices")
        if graph_q_dst_row_indices is not None:
            graph_q_dst_row_indices.fill_(-1)
            if context.q_dst_row_indices is not None:
                graph_q_dst_row_indices.copy_(context.q_dst_row_indices)

        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"].copy_(context.context_lens)  # type: ignore
        graph_vars["global_context_lens"].zero_()
        graph_vars["global_context_lens"].copy_(context.global_context_lens)  # type: ignore
        graph_vars["q_mask"].zero_()
        graph_vars["q_mask"].copy_(context.q_mask)  # type: ignore
        graph_vars["res_lse_mask"].zero_()
        graph_vars["res_lse_mask"].copy_(context.res_lse_mask)  # type: ignore
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][
            : context.block_tables.size(0), : context.block_tables.size(1)  # type: ignore
        ] = context.block_tables

        graph_vars["context_lens_for_attn"].zero_()
        graph_vars["context_lens_for_attn"][
            : context.context_lens_for_attn.shape[0]
        ].copy_(context.context_lens_for_attn)  # type: ignore

        graph_vars["q_slice_get"].fill_(-1)
        graph_vars["q_slice_fill"].fill_(-1)
        graph_vars["q_copy_mask"].zero_()
        graph_vars["q_slice_get"][: context.q_slice_get.shape[0]] = context.q_slice_get  # type: ignore
        graph_vars["q_slice_fill"][: context.q_slice_fill.shape[0]] = context.q_slice_fill  # type: ignore
        graph_vars["q_copy_mask"][: context.q_copy_mask.shape[0]] = context.q_copy_mask  # type: ignore

        graph_vars["res_slice_get_to_buffer_output"].fill_(-1)
        graph_vars["res_slice_fill_to_buffer_output"].fill_(-1)
        graph_vars["res_to_buffer_output_mask"].zero_()
        graph_vars["res_slice_get_to_buffer_output"][: context.res_slice_get_to_buffer_output.shape[0]] = context.res_slice_get_to_buffer_output  # type: ignore
        graph_vars["res_slice_fill_to_buffer_output"][: context.res_slice_fill_to_buffer_output.shape[0]] = context.res_slice_fill_to_buffer_output  # type: ignore
        graph_vars["res_to_buffer_output_mask"][: context.res_to_buffer_output_mask.shape[0]] = context.res_to_buffer_output_mask  # type: ignore

        graph_vars["res_slice_get_to_buffer_input"].fill_(-1)
        graph_vars["res_slice_fill_to_buffer_input"].fill_(-1)
        graph_vars["res_to_buffer_input_mask"].zero_()
        graph_vars["res_slice_get_to_buffer_input"][: context.res_slice_get_to_buffer_input.shape[0]].copy_(context.res_slice_get_to_buffer_input)  # type: ignore
        graph_vars["res_slice_fill_to_buffer_input"][: context.res_slice_fill_to_buffer_input.shape[0]].copy_(context.res_slice_fill_to_buffer_input)  # type: ignore
        graph_vars["res_to_buffer_input_mask"][: context.res_to_buffer_input_mask.shape[0]].copy_(context.res_to_buffer_input_mask)  # type: ignore

        graph_vars["q_offsets"].zero_()
        graph_vars["q_offsets"].copy_(context.q_offsets)  # type: ignore

        if not context.use_sp_a2a:
            return

        actual_attn_bs = int(context.attention_compute_bs)
        graph_actual_attn_bs = graph_vars.get("actual_attn_bs")
        if graph_actual_attn_bs is not None:
            graph_actual_attn_bs.fill_(actual_attn_bs)

        materialize_sp_graph_padding(
            graph_vars,
            actual_block_tables=context.block_tables,
            actual_attn_bs=actual_attn_bs,
            graph_attn_bs=graph_attn_bs,
            actual_master_bs=bs,
            graph_master_bs=master_bs,
            local_result_rows=context.res_slice_get_to_buffer_output.numel(),
            sp_rank=get_dist_context().attn_sp_rank,
            max_num_seqs=self.config.max_num_seqs,
        )

    def _build_graph_master_rank_bs(self, max_bs: int) -> list[int]:
        graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        graph_bs = [bs for bs in graph_bs if bs <= max_bs]
        if not graph_bs or graph_bs[-1] != max_bs:
            graph_bs.append(max_bs)
        return graph_bs

    def _build_sp_graph_attn_bs_candidates(
        self, master_bs: int, sp_world_size: int
    ) -> list[int]:
        config = self.config
        fixed_sp_graph = sp_world_size > 1 and config.fixed_sp_size > 0
        fixed_full_sp_graph = fixed_sp_graph and config.fixed_sp_size == sp_world_size

        if fixed_full_sp_graph:
            return [sp_world_size * master_bs]

        limit = master_bs + config.max_num_recv_seqs
        if fixed_sp_graph:
            limit = sp_world_size * master_bs
        elif config.sp_backend == "nccl":
            limit = min(limit, sp_world_size * master_bs)

        current_attn_bs_candidates = []
        curr = master_bs
        while curr <= limit:
            current_attn_bs_candidates.append(curr)
            curr += self.attn_bs_step
        if current_attn_bs_candidates[-1] != limit:
            current_attn_bs_candidates.append(limit)
        return current_attn_bs_candidates

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill:
            return compute_prefill_logits(self.model, input_ids, positions)

        context = get_context()
        if self.enforce_eager or input_ids.size(0) > 512:
            attention_compute_bs = context.attention_compute_bs or input_ids.size(0)
            if context.context_lens_for_attn is None:
                raise RuntimeError("Decode context is missing FlashMLA sequence lengths")
            (
                context.tile_scheduler_metadata,
                context.num_splits,
            ) = prepare_decode_mla_metadata(
                self.config.hf_config,
                context.context_lens_for_attn[:attention_compute_bs],
            )
            return self.model.compute_logits(self.model(input_ids, positions))

        if self.cuda_graph_mode == "piecewise":
            return self.run_model_piecewise_cudagraph(input_ids, positions)

        bs = input_ids.size(0)
        master_bs = self._select_decode_graph_master_bs(bs, context)

        if context.use_sp_a2a:
            ac_bs = context.attention_compute_bs
            if ac_bs is None:
                ac_bs = bs
            valid_attn_bs_list = self.sp_graph_map.get(master_bs)
            if valid_attn_bs_list is None:
                raise RuntimeError(f"No SP graph map found for master_bs={master_bs}")

            try:
                attn_bs = next(x for x in valid_attn_bs_list if x >= ac_bs)
            except StopIteration:
                raise RuntimeError(
                    f"Input attention_compute_bs {ac_bs} exceeds max captured attn_bs "
                    f"({valid_attn_bs_list[-1]}) for master_bs {master_bs}"
                )

            graph = self.sp_graphs[(master_bs, attn_bs)]
        else:
            attn_bs = master_bs
            graph = self.local_graphs[master_bs]

        graph_vars = self.graph_vars
        self._copy_decode_context_to_graph_vars(
            graph_vars,
            input_ids,
            positions,
            bs,
            master_bs,
            attn_bs,
            context,
        )
        prepare_decode_mla_metadata(
            self.config.hf_config,
            graph_vars["context_lens_for_attn"][:attn_bs],
            graph_vars["tile_scheduler_metadata"],
            graph_vars["num_splits"],
        )
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_model_piecewise_cudagraph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        bs = input_ids.size(0)
        context = get_context()
        master_bs = self._select_decode_graph_master_bs(bs, context)
        graph_set = self.piecewise_graphs[master_bs]
        graph_vars = self.piecewise_graph_vars[master_bs]

        attention_compute_bs = context.attention_compute_bs
        if attention_compute_bs is None:
            attention_compute_bs = bs

        self._copy_decode_context_to_graph_vars(
            graph_vars,
            input_ids,
            positions,
            bs,
            master_bs,
            attention_compute_bs,
            context,
        )
        tile_scheduler_metadata, num_splits = prepare_decode_mla_metadata(
            self.config.hf_config,
            graph_vars["context_lens_for_attn"][:attention_compute_bs],
        )

        temp_context_fields = {
            "slot_mapping": graph_vars["slot_mapping"][:master_bs],
            "context_lens": graph_vars["context_lens"],
            "block_tables": graph_vars["block_tables"],
            "global_context_lens": graph_vars["global_context_lens"],
            "q_mask": graph_vars["q_mask"],
            "q_dst_row_indices": (
                graph_vars["q_dst_row_indices"]
                if context.q_dst_row_indices is not None
                else None
            ),
            "res_lse_mask": graph_vars["res_lse_mask"],
            "tile_scheduler_metadata": tile_scheduler_metadata,
            "num_splits": num_splits,
            "q_slice_get": graph_vars["q_slice_get"][:master_bs],
            "q_slice_fill": graph_vars["q_slice_fill"][:master_bs],
            "q_copy_mask": graph_vars["q_copy_mask"][:master_bs],
            "res_slice_get_to_buffer_output": graph_vars[
                "res_slice_get_to_buffer_output"
            ][:master_bs],
            "res_slice_fill_to_buffer_output": graph_vars[
                "res_slice_fill_to_buffer_output"
            ][:master_bs],
            "res_to_buffer_output_mask": graph_vars["res_to_buffer_output_mask"][
                :master_bs
            ],
            "res_slice_get_to_buffer_input": graph_vars[
                "res_slice_get_to_buffer_input"
            ],
            "res_slice_fill_to_buffer_input": graph_vars[
                "res_slice_fill_to_buffer_input"
            ],
            "res_to_buffer_input_mask": graph_vars["res_to_buffer_input_mask"],
            "attention_compute_bs": attention_compute_bs,
            "sp_comm_bs": master_bs,
            "q_offsets": graph_vars["q_offsets"],
            "context_lens_for_attn": graph_vars["context_lens_for_attn"],
        }
        saved_context_fields = {
            name: getattr(context, name) for name in temp_context_fields
        }
        for name, value in temp_context_fields.items():
            setattr(context, name, value)

        try:
            layers = self.model.model.layers
            workspace = graph_set["workspace"]
            query_states = workspace["query_states"][:master_bs]
            key_states = workspace["key_states"][:master_bs]
            value_states = workspace["value_states"][:master_bs]
            attn_input = workspace["attn_output"][:master_bs]
            for layer_idx, layer_graph in enumerate(graph_set["layers"]):
                layer_graph["pre"].replay()
                layer = layers[layer_idx]
                attn_output = layer.self_attn.attention_core(
                    query_states,
                    key_states,
                    value_states,
                )
                if attn_output.size(0) > master_bs:
                    raise RuntimeError(
                        "Piecewise attention output exceeds graph master_bs: "
                        f"rows={attn_output.size(0)} master_bs={master_bs}"
                    )
                if attn_output.size(0) < master_bs:
                    attn_input.zero_()
                attn_input[: attn_output.size(0)].copy_(attn_output)
                layer_graph["post"].replay()

            graph_set["final"].replay()
        finally:
            for name, value in saved_context_fields.items():
                setattr(context, name, value)

        return self.model.compute_logits(graph_set["workspace"]["final_hidden"][:bs])

    def migrate(self, seqs: list[Sequence]) -> None:
        get_cache_context().migrate(seqs=seqs)

    def run(
        self,
        dp_seqs: list[Sequence],
        is_prefill: bool,
        enable_rpc: bool = False,
        send_timestamp: float = 0.0,
        hierarchical_trace: dict | None = None,
        hierarchical_quantum_diagnostics: bool = False,
    ) -> (
        tuple[list[list[int]], float]
        | tuple[list[list[int]], float, dict]
        | tuple[list[list[int]], float, dict, dict]
    ):
        diagnostics_enabled = bool(hierarchical_quantum_diagnostics)
        worker_begin = time.perf_counter() if diagnostics_enabled else 0.0
        recv_begin = time.perf_counter() if diagnostics_enabled else 0.0
        if enable_rpc:
            dp_seqs = self.endpoint.recv_seqs()
        recv_end = time.perf_counter() if diagnostics_enabled else 0.0

        if hierarchical_trace is not None:
            if is_prefill:
                raise RuntimeError(
                    "hierarchical execution trace is decode-only"
                )
            if hierarchical_trace.get("global_rank") != self.rank:
                raise RuntimeError(
                    "hierarchical trace/global worker rank mismatch"
                )
            execution_forwards: list[dict] = []

        if send_timestamp > 0:
            latency = (time.time() - send_timestamp) * 1000
            logger.info(f"[METRIC] Rank {self.rank} Input Transfer Latency: {latency:.4f} ms")
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size

        num_sp_seqs = sum(
            1
            for seq in dp_seqs
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank
        )
        is_dummy = num_sp_seqs == 0
        if is_dummy and not is_prefill:
            raise RuntimeError(
                "worker received no canonical Sequence for its SP rank; "
                "the scheduler must provide a persistent control dummy with "
                "reserved KV blocks"
            )
        if is_dummy:
            # Prefill collectives still need every rank, but decode-shaped
            # persistent dummies must not enter the prefill KV write path.
            seq = Sequence([0])
            seq.active(self.engine_id, sp_size, 1)
            seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx = sp_rank
            dp_seqs.append(seq)

        sp_seqs = [
            seq
            for seq in dp_seqs
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank
        ]

        loop_count = self.config.loop_count if not is_prefill else 1
        prepare_update_ms = 0.0
        forward_host_ms = 0.0
        loop_begin = time.perf_counter() if diagnostics_enabled else 0.0
        gpu_loop_begin = None
        gpu_loop_end = None
        if diagnostics_enabled and torch.cuda.is_available():
            gpu_loop_begin = torch.cuda.Event(enable_timing=True)
            gpu_loop_end = torch.cuda.Event(enable_timing=True)
            gpu_loop_begin.record()
        for i in range(loop_count):
            current_time = time.time()

            # Check if should start profiling (time-based or step-based)
            should_start_profiling = False
            if self.profiler:
                if self.profiler_use_time:
                    # Time-based: check if enough time has passed since initialization
                    elapsed_time = current_time - self.profiler_init_time
                    if self.profiler_start_timestamp is None and elapsed_time >= self.profiler_start_time:
                        should_start_profiling = True
                else:
                    # Step-based: check if reached start step
                    if self.run_count == self.profiler_start_step:
                        should_start_profiling = True

            if should_start_profiling:
                logger.info(
                    f"Rank {self.rank}: Starting profiler... (profiler_dir={self.profiler_dir}, worker_name={self.profiler_worker_name})"
                )
                try:
                    self.profiler.start()
                    if self.profiler_use_time:
                        self.profiler_start_timestamp = current_time
                        self.profiler_end_timestamp = current_time + self.profiling_duration
                        logger.info(
                            f"Rank {self.rank}: ✓ Profiler STARTED successfully at time {current_time:.2f} "
                            f"(elapsed {elapsed_time:.2f}s, will run for {self.profiling_duration}s until {self.profiler_end_timestamp:.2f})"
                        )
                    else:
                        logger.info(
                            f"Rank {self.rank}: ✓ Profiler STARTED successfully at step {self.run_count} "
                            f"(will run until step {self.profiler_end_step})"
                        )
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: ✗ Failed to start profiler: {e}", exc_info=True
                    )

            prepare_begin = (
                time.perf_counter() if diagnostics_enabled else 0.0
            )
            if is_prefill:
                input_ids, positions = self.prepare_prefill(dp_seqs, is_dummy)
            else:
                if i == 0:
                    input_ids, positions = self.prepare_decode(dp_seqs, is_dummy)
                else:
                    input_ids, positions = self.update_decode(
                        input_ids, positions, dp_seqs
                    )
                if self.log_decode_a2a_masks:
                    self._log_decode_a2a_masks(loop_idx=i, is_dummy=is_dummy)
            if diagnostics_enabled:
                prepare_update_ms += (
                    time.perf_counter() - prepare_begin
                ) * 1000

            if hierarchical_trace is not None or diagnostics_enabled:
                forward_begin = time.perf_counter()
                logits = self.run_model(input_ids, positions, is_prefill)
                forward_end = time.perf_counter()
                if diagnostics_enabled:
                    forward_host_ms += (
                        forward_end - forward_begin
                    ) * 1000
                if hierarchical_trace is not None:
                    execution_forwards.append(
                        {
                            "inner_loop_idx": i,
                            "use_sp_a2a": bool(get_context().use_sp_a2a),
                            "forward_begin": forward_begin,
                            "forward_end": forward_end,
                        }
                    )
            else:
                logits = self.run_model(input_ids, positions, is_prefill)

            tp_rank = get_dist_context().attn_tp_rank
            if tp_rank == 0:
                temperatures = (
                    self.prepare_sample(dp_seqs)
                    if tp_rank == 0
                    else [None] * len(sp_seqs)
                )
                input_ids = self.sampler(logits, temperatures)
            else:
                input_ids = torch.zeros_like(input_ids)
            dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)

            update_seqs_inner_loop(sp_seqs, sp_rank)

            # Check if should continue/stop profiling
            is_profiling = False
            if self.profiler and not self.profiler_stopped:
                if self.profiler_use_time:
                    # Time-based: check if profiling has started
                    is_profiling = self.profiler_start_timestamp is not None
                else:
                    # Step-based: check if reached start step
                    is_profiling = self.run_count >= self.profiler_start_step

            if is_profiling:
                should_continue = False
                should_stop = False

                if self.profiler_use_time:
                    # Time-based: check current time
                    current_time = time.time()
                    if current_time < self.profiler_end_timestamp:
                        should_continue = True
                    else:
                        should_stop = True
                else:
                    # Step-based: check step count
                    # Note: run_count increments at the end of the loop, so we check before incrementing
                    # We want to profile from start_step to (start_step + profiling_steps - 1)
                    # After the last step, run_count will be >= profiler_end_step
                    if self.run_count < self.profiler_end_step:
                        should_continue = True
                    else:
                        should_stop = True

                if should_continue:
                    try:
                        self.profiler.step()
                        self.profiler_step_count += 1
                        # Log every 10 steps or at key intervals
                        if self.profiler_use_time:
                            remaining_time = self.profiler_end_timestamp - current_time
                            if self.profiler_step_count % 10 == 0 or remaining_time < 1.0:
                                logger.info(
                                    f"Rank {self.rank}: Profiler step {self.profiler_step_count}, "
                                    f"remaining time: {remaining_time:.2f}s"
                                )
                        else:
                            remaining_steps = self.profiler_end_step - self.run_count
                            if self.profiler_step_count % 10 == 0 or remaining_steps <= 2:
                                logger.info(
                                    f"Rank {self.rank}: Profiler step {self.profiler_step_count}, "
                                    f"remaining steps: {remaining_steps}"
                                )
                    except RuntimeError as e:
                        logger.warning(f"Rank {self.rank}: Failed to step profiler: {e}")
                elif should_stop and not self.profiler_stopped:
                    logger.info(
                        f"Rank {self.rank}: Stopping profiler... (profiler_dir={self.profiler_dir}, worker_name={self.profiler_worker_name})"
                    )
                    try:
                        self.profiler.stop()
                        self.profiler_stopped = True
                        # Reset timestamps to prevent further profiling attempts
                        if self.profiler_use_time:
                            elapsed_profiling_time = current_time - self.profiler_start_timestamp
                            self.profiler_start_timestamp = None
                            self.profiler_end_timestamp = None
                            # Construct expected trace file path
                            trace_file = os.path.join(
                                self.profiler_dir,
                                self.profiler_worker_name,
                                "*.pt.trace.json"
                            )
                            logger.info(
                                f"Rank {self.rank}: ✓ Profiler STOPPED and saved successfully at time {current_time:.2f} "
                                f"(profiled for {elapsed_profiling_time:.2f}s, {self.profiler_step_count} steps). "
                                f"Trace files should be in: {os.path.join(self.profiler_dir, self.profiler_worker_name)}"
                            )
                        else:
                            trace_file = os.path.join(
                                self.profiler_dir,
                                self.profiler_worker_name,
                                "*.pt.trace.json"
                            )
                            logger.info(
                                f"Rank {self.rank}: ✓ Profiler STOPPED and saved successfully at step {self.run_count} "
                                f"({self.profiler_step_count} profiler steps). "
                                f"Trace files should be in: {os.path.join(self.profiler_dir, self.profiler_worker_name)}"
                            )
                    except RuntimeError as e:
                        logger.error(
                            f"Rank {self.rank}: ✗ Failed to stop profiler: {e}", exc_info=True
                        )
                        self.profiler_stopped = True  # Mark as stopped even if error occurred
                    except Exception as e:
                        logger.error(
                            f"Rank {self.rank}: ✗ Unexpected error stopping profiler: {e}", exc_info=True
                        )
                        self.profiler_stopped = True

            self.run_count += 1  # 每次调用计数+1
            get_context().token_ids.append(input_ids[None, ...])

        if gpu_loop_end is not None:
            gpu_loop_end.record()
        token_materialize_begin = (
            time.perf_counter() if diagnostics_enabled else 0.0
        )
        loop_count_token_ids = torch.cat(get_context().token_ids, dim=0).T.tolist()
        gpu_loop_ms = None
        if gpu_loop_begin is not None and gpu_loop_end is not None:
            # token materialization above already synchronizes the dependent
            # result. Synchronizing the end event also covers work enqueued
            # on the current stream without adding per-inner-loop barriers.
            gpu_loop_end.synchronize()
            gpu_loop_ms = gpu_loop_begin.elapsed_time(gpu_loop_end)
        worker_end = time.perf_counter() if diagnostics_enabled else 0.0
        token_materialize_ms = (
            (worker_end - token_materialize_begin) * 1000
            if diagnostics_enabled
            else 0.0
        )
        reset_context()
        worker_end_time = time.time()

        diagnostic = None
        if diagnostics_enabled:
            diagnostic = {
                "global_rank": int(self.rank),
                "recv_seqs_ms": (recv_end - recv_begin) * 1000,
                "prepare_update_host_ms": prepare_update_ms,
                "forward_host_ms": forward_host_ms,
                "gpu_loop_ms": gpu_loop_ms,
                "loop_host_ms": (worker_end - loop_begin) * 1000,
                "token_materialize_ms": token_materialize_ms,
                "worker_body_ms": (worker_end - recv_end) * 1000,
                "worker_total_ms": (worker_end - worker_begin) * 1000,
            }

        if hierarchical_trace is not None:
            trace = dict(hierarchical_trace)
            trace["forward_count"] = loop_count
            trace["forwards"] = tuple(execution_forwards)
            if diagnostic is not None:
                return (
                    loop_count_token_ids,
                    worker_end_time,
                    trace,
                    diagnostic,
                )
            return loop_count_token_ids, worker_end_time, trace
        if diagnostic is not None:
            return loop_count_token_ids, worker_end_time, diagnostic
        return loop_count_token_ids, worker_end_time

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self.cuda_graph_mode == "piecewise":
            return self.capture_piecewise_cudagraph()

        sp_world_size = get_dist_context().attn_sp_world_size
        config = self.config
        hf_config = config.hf_config
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        max_bs = min(self.config.max_num_seqs, 512)
        fixed_sp_graph = sp_world_size > 1 and config.fixed_sp_size > 0
        use_hao_destination_rows = (
            sp_world_size > 1
            and config.sp_backend == "hao_basic"
        )
        max_attention_comp_seqs = (
            sp_world_size * max_bs
            if fixed_sp_graph
            else max_bs + config.max_num_recv_seqs
        )
        max_remote_attention_comp_seqs = (
            max_attention_comp_seqs
            if fixed_sp_graph
            else config.max_num_recv_seqs
        )
        block_size = get_cache_context().block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens_for_attn = torch.zeros(
            max_attention_comp_seqs, dtype=torch.int32
        )
        context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        global_context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        q_mask = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        q_dst_row_indices = (
            torch.full(
                (sp_world_size, config.max_num_seqs),
                -1,
                dtype=torch.int32,
            )
            if use_hao_destination_rows
            else None
        )
        actual_attn_bs = (
            torch.zeros((), dtype=torch.int32)
            if use_hao_destination_rows
            else None
        )
        res_lse_mask = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            max_attention_comp_seqs, max_num_blocks, dtype=torch.int32
        )
        q_slice_get = torch.full((max_bs,), -1, dtype=torch.int32)
        q_slice_fill = torch.full((max_bs,), -1, dtype=torch.int32)
        q_copy_mask = torch.zeros(max_bs, dtype=torch.int32)
        res_slice_get_to_buffer_output = torch.full((max_bs,), -1, dtype=torch.int32)
        res_slice_fill_to_buffer_output = torch.full((max_bs,), -1, dtype=torch.int32)
        res_to_buffer_output_mask = torch.zeros(max_bs, dtype=torch.int32)
        res_slice_get_to_buffer_input = torch.full(
            (max_remote_attention_comp_seqs,), -1, dtype=torch.int32
        )
        res_slice_fill_to_buffer_input = torch.full(
            (max_remote_attention_comp_seqs,), -1, dtype=torch.int32
        )
        res_to_buffer_input_mask = torch.zeros(
            max_remote_attention_comp_seqs, dtype=torch.int32
        )
        q_offsets = torch.zeros(sp_world_size + 1, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        if hf_config.num_key_value_heads == 1:
            tile_scheduler_metadata_buffer, num_splits_buffer = (
                prepare_decode_mla_metadata(
                    hf_config,
                    torch.ones(
                        max_attention_comp_seqs,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
            )
        else:
            tile_scheduler_metadata_buffer, num_splits_buffer = None, None

        self.graph_master_rank_bs = self._build_graph_master_rank_bs(max_bs)
        self.local_graphs = {}
        self.sp_graphs = {}
        self.graph_pool = None
        self.sp_graph_map = {}  # store master_bs -> [available_attn_bs...]
        self.attn_bs_step = 16

        def capture_graph(master_bs: int, attn_bs: int, use_sp_a2a: bool):
            graph = torch.cuda.CUDAGraph()
            context_lens_for_attn.zero_()
            context_lens_for_attn[:attn_bs].fill_(1)
            tile_scheduler_metadata, num_splits = prepare_decode_mla_metadata(
                hf_config,
                context_lens_for_attn[:attn_bs],
                tile_scheduler_metadata_buffer,
                num_splits_buffer,
            )
            set_context(
                is_prefill=False,
                max_bs=self.config.max_num_seqs,
                slot_mapping=slot_mapping[:master_bs],
                context_lens=context_lens,
                block_tables=block_tables,
                global_context_lens=global_context_lens,
                q_mask=q_mask,
                q_dst_row_indices=(
                    q_dst_row_indices
                    if use_sp_a2a and use_hao_destination_rows
                    else None
                ),
                actual_attn_bs=(
                    actual_attn_bs
                    if use_sp_a2a and use_hao_destination_rows
                    else None
                ),
                res_lse_mask=res_lse_mask,
                use_sp_a2a=use_sp_a2a,
                q_slice_get=q_slice_get[:master_bs],
                q_slice_fill=q_slice_fill[:master_bs],
                q_copy_mask=q_copy_mask[:master_bs],
                res_slice_get_to_buffer_output=res_slice_get_to_buffer_output[
                    :master_bs
                ],
                res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output[
                    :master_bs
                ],
                res_to_buffer_output_mask=res_to_buffer_output_mask[:master_bs],
                res_slice_get_to_buffer_input=res_slice_get_to_buffer_input,
                res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input,
                res_to_buffer_input_mask=res_to_buffer_input_mask,
                attention_compute_bs=attn_bs,
                sp_comm_bs=master_bs,
                context_lens_for_attn=context_lens_for_attn,
                q_offsets=q_offsets,
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
            )

            outputs[:master_bs] = self.model(
                input_ids[:master_bs], positions[:master_bs]
            )  # warmup

            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:master_bs] = self.model(
                    input_ids[:master_bs], positions[:master_bs]
                )  # capture

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            reset_context()
            return graph

        logger.info("Starting CUDAGraph capture...")
        total_graphs = 0

        for master_bs in reversed(self.graph_master_rank_bs):
            logger.info(f"正在捕获本地图 - (master_bs={master_bs})")
            self.local_graphs[master_bs] = capture_graph(
                master_bs=master_bs,
                attn_bs=master_bs,
                use_sp_a2a=False,
            )
            total_graphs += 1

        if sp_world_size > 1:
            for master_bs in reversed(self.graph_master_rank_bs):
                self.sp_graph_map[master_bs] = []

                current_attn_bs_candidates = self._build_sp_graph_attn_bs_candidates(
                    master_bs, sp_world_size
                )
                for attn_bs in reversed(current_attn_bs_candidates):
                    logger.info(
                        f"正在捕获 SP 图 - (master_bs={master_bs}, attn_bs={attn_bs})"
                    )
                    self.sp_graphs[(master_bs, attn_bs)] = capture_graph(
                        master_bs=master_bs,
                        attn_bs=attn_bs,
                        use_sp_a2a=True,
                    )
                    self.sp_graph_map[master_bs].append(attn_bs)
                    total_graphs += 1

                self.sp_graph_map[master_bs].sort()

        logger.info(
            "Finished capturing all graphs. "
            f"Successfully captured {len(self.local_graphs)} local graphs, "
            f"{len(self.sp_graphs)} SP graphs, for a total of {total_graphs} graphs."
        )

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            outputs=outputs,
            q_mask=q_mask,
            q_dst_row_indices=q_dst_row_indices,
            actual_attn_bs=actual_attn_bs,
            res_lse_mask=res_lse_mask,
            tile_scheduler_metadata=tile_scheduler_metadata_buffer,
            num_splits=num_splits_buffer,
            context_lens_for_attn=context_lens_for_attn,
            q_slice_get=q_slice_get,
            q_slice_fill=q_slice_fill,
            q_copy_mask=q_copy_mask,
            res_slice_get_to_buffer_output=res_slice_get_to_buffer_output,
            res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output,
            res_to_buffer_output_mask=res_to_buffer_output_mask,
            res_slice_get_to_buffer_input=res_slice_get_to_buffer_input,
            res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input,
            res_to_buffer_input_mask=res_to_buffer_input_mask,
            q_offsets=q_offsets,
        )

    @torch.inference_mode()
    def capture_piecewise_cudagraph(self):
        sp_world_size = get_dist_context().attn_sp_world_size
        config = self.config
        hf_config = config.hf_config
        if hf_config.architectures[0] != "DeepseekV3ForCausalLM":
            raise RuntimeError("Piecewise CUDA Graph only supports DeepseekV3ForCausalLM")

        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        max_bs = min(self.config.max_num_seqs, 512)
        fixed_sp_graph = sp_world_size > 1 and config.fixed_sp_size > 0
        use_hao_destination_rows = (
            sp_world_size > 1
            and config.sp_backend == "hao_basic"
        )
        max_attention_comp_seqs = (
            sp_world_size * max_bs
            if fixed_sp_graph
            else max_bs + config.max_num_recv_seqs
        )
        max_remote_attention_comp_seqs = (
            max_attention_comp_seqs
            if fixed_sp_graph
            else config.max_num_recv_seqs
        )
        block_size = get_cache_context().block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size

        self.graph_master_rank_bs = self._build_graph_master_rank_bs(max_bs)
        self.piecewise_graphs = {}
        self.piecewise_graph_vars = {}
        self.graph_pool = None

        def capture_one_graph(fn):
            graph = torch.cuda.CUDAGraph()
            fn()
            with torch.cuda.graph(graph, self.graph_pool):
                fn()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            return graph

        def make_decode_graph_vars():
            return dict(
                input_ids=torch.zeros(max_bs, dtype=torch.int64),
                positions=torch.zeros(max_bs, dtype=torch.int64),
                slot_mapping=torch.full((max_bs,), -1, dtype=torch.int32),
                context_lens=torch.zeros(sp_world_size, max_bs, dtype=torch.int32),
                block_tables=torch.zeros(
                    max_attention_comp_seqs, max_num_blocks, dtype=torch.int32
                ),
                global_context_lens=torch.zeros(
                    sp_world_size, max_bs, dtype=torch.int32
                ),
                q_mask=torch.zeros(sp_world_size, max_bs, dtype=torch.int32),
                q_dst_row_indices=(
                    torch.full(
                        (sp_world_size, config.max_num_seqs),
                        -1,
                        dtype=torch.int32,
                    )
                    if use_hao_destination_rows
                    else None
                ),
                res_lse_mask=torch.zeros(sp_world_size, max_bs, dtype=torch.int32),
                tile_scheduler_metadata=None,
                num_splits=None,
                context_lens_for_attn=torch.zeros(
                    max_attention_comp_seqs, dtype=torch.int32
                ),
                q_slice_get=torch.full((max_bs,), -1, dtype=torch.int32),
                q_slice_fill=torch.full((max_bs,), -1, dtype=torch.int32),
                q_copy_mask=torch.zeros(max_bs, dtype=torch.int32),
                res_slice_get_to_buffer_output=torch.full(
                    (max_bs,), -1, dtype=torch.int32
                ),
                res_slice_fill_to_buffer_output=torch.full(
                    (max_bs,), -1, dtype=torch.int32
                ),
                res_to_buffer_output_mask=torch.zeros(max_bs, dtype=torch.int32),
                res_slice_get_to_buffer_input=torch.full(
                    (max_remote_attention_comp_seqs,), -1, dtype=torch.int32
                ),
                res_slice_fill_to_buffer_input=torch.full(
                    (max_remote_attention_comp_seqs,), -1, dtype=torch.int32
                ),
                res_to_buffer_input_mask=torch.zeros(
                    max_remote_attention_comp_seqs, dtype=torch.int32
                ),
                q_offsets=torch.zeros(sp_world_size + 1, dtype=torch.int32),
            )

        def set_piecewise_capture_context(master_bs: int, graph_vars: dict):
            set_context(
                is_prefill=False,
                max_bs=self.config.max_num_seqs,
                slot_mapping=graph_vars["slot_mapping"][:master_bs],
                context_lens=graph_vars["context_lens"],
                block_tables=graph_vars["block_tables"],
                global_context_lens=graph_vars["global_context_lens"],
                q_mask=graph_vars["q_mask"],
                q_dst_row_indices=(
                    graph_vars["q_dst_row_indices"]
                    if use_hao_destination_rows
                    else None
                ),
                res_lse_mask=graph_vars["res_lse_mask"],
                use_sp_a2a=False,
                q_slice_get=graph_vars["q_slice_get"][:master_bs],
                q_slice_fill=graph_vars["q_slice_fill"][:master_bs],
                q_copy_mask=graph_vars["q_copy_mask"][:master_bs],
                res_slice_get_to_buffer_output=graph_vars[
                    "res_slice_get_to_buffer_output"
                ][:master_bs],
                res_slice_fill_to_buffer_output=graph_vars[
                    "res_slice_fill_to_buffer_output"
                ][:master_bs],
                res_to_buffer_output_mask=graph_vars["res_to_buffer_output_mask"][
                    :master_bs
                ],
                res_slice_get_to_buffer_input=graph_vars[
                    "res_slice_get_to_buffer_input"
                ],
                res_slice_fill_to_buffer_input=graph_vars[
                    "res_slice_fill_to_buffer_input"
                ],
                res_to_buffer_input_mask=graph_vars["res_to_buffer_input_mask"],
                attention_compute_bs=master_bs,
                sp_comm_bs=master_bs,
                context_lens_for_attn=graph_vars["context_lens_for_attn"],
                q_offsets=graph_vars["q_offsets"],
                tile_scheduler_metadata=graph_vars["tile_scheduler_metadata"],
                num_splits=graph_vars["num_splits"],
            )

        logger.info("Starting piecewise CUDAGraph capture...")
        total_graphs = 0
        model = self.model.model
        graph_vars = make_decode_graph_vars()
        input_ids = graph_vars["input_ids"]
        positions = graph_vars["positions"]
        workspace = dict(
            hidden_states=torch.zeros(max_bs, hf_config.hidden_size),
            residual=torch.zeros(max_bs, hf_config.hidden_size),
            layer_residual=torch.zeros(max_bs, hf_config.hidden_size),
            query_states=torch.zeros(
                max_bs,
                hf_config.num_attention_heads,
                hf_config.kv_lora_rank + hf_config.qk_rope_head_dim,
            ),
            key_states=torch.zeros(
                max_bs,
                getattr(hf_config, "num_key_value_heads", 1),
                hf_config.kv_lora_rank + hf_config.qk_rope_head_dim,
            ),
            value_states=torch.zeros(
                max_bs,
                getattr(hf_config, "num_key_value_heads", 1),
                hf_config.kv_lora_rank,
            ),
            attn_output=torch.zeros(
                max_bs,
                hf_config.num_attention_heads,
                hf_config.kv_lora_rank,
            ),
            final_hidden=torch.zeros(max_bs, hf_config.hidden_size),
        )

        for master_bs in reversed(self.graph_master_rank_bs):
            logger.info(f"正在捕获 Piecewise 图 - (master_bs={master_bs})")
            layers = []
            set_piecewise_capture_context(master_bs, graph_vars)
            input_ids_view = input_ids[:master_bs]
            positions_view = positions[:master_bs]
            hidden_states = workspace["hidden_states"][:master_bs]
            residual = workspace["residual"][:master_bs]
            layer_residual_workspace = workspace["layer_residual"][:master_bs]
            query_workspace = workspace["query_states"][:master_bs]
            key_workspace = workspace["key_states"][:master_bs]
            value_workspace = workspace["value_states"][:master_bs]
            attn_output_workspace = workspace["attn_output"][:master_bs]
            final_hidden = workspace["final_hidden"][:master_bs]

            for layer_idx, layer in enumerate(model.layers):
                if layer_idx == 0:
                    def pre_fn(layer=layer):
                        embedded = model.embed_tokens(input_ids_view)
                        query_states, key_states, value_states, layer_residual = (
                            layer.piecewise_pre_attention(
                                embedded, positions_view, residual=None
                            )
                        )
                        query_workspace.copy_(query_states)
                        key_workspace.copy_(key_states)
                        value_workspace.copy_(value_states)
                        layer_residual_workspace.copy_(layer_residual)
                else:
                    def pre_fn(layer=layer):
                        query_states, key_states, value_states, layer_residual = (
                            layer.piecewise_pre_attention(
                                hidden_states, positions_view, residual
                            )
                        )
                        query_workspace.copy_(query_states)
                        key_workspace.copy_(key_states)
                        value_workspace.copy_(value_states)
                        layer_residual_workspace.copy_(layer_residual)

                pre_graph = capture_one_graph(pre_fn)

                def post_fn(
                    layer=layer,
                ):
                    next_hidden_states, next_residual = (
                        layer.piecewise_post_attention(
                            attn_output_workspace,
                            layer_residual_workspace,
                        )
                    )
                    hidden_states.copy_(next_hidden_states)
                    residual.copy_(next_residual)

                post_graph = capture_one_graph(post_fn)
                layers.append(
                    dict(
                        pre=pre_graph,
                        post=post_graph,
                    )
                )
                total_graphs += 2

            def final_fn():
                final_hidden.copy_(model.piecewise_finalize(hidden_states, residual))

            final_graph = capture_one_graph(final_fn)
            total_graphs += 1
            self.piecewise_graphs[master_bs] = dict(
                layers=layers,
                final=final_graph,
                workspace=workspace,
            )
            self.piecewise_graph_vars[master_bs] = graph_vars

        logger.info(
            "Finished capturing piecewise graphs. "
            f"Successfully captured {len(self.piecewise_graphs)} master_bs buckets, "
            f"{total_graphs} graphs."
        )
        reset_context()

import os
import threading

import ray
import torch
import torch.distributed as dist
from nanodeploy._cpp import (
    extract_aux_from_bytes,
    extract_vision_slots_from_bytes,
    serialize_run_batch,
)
from nanodeploy.config import Config
from nanodeploy.context.cache import get_cache_context, set_cache_context
from nanodeploy.context.context import get_context, reset_context
from nanodeploy.context.distributed import (
    get_dist_context,
    get_local_ip,
    set_dist_context,
)
from nanodeploy.context.expert_context import ExpertContext
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.models.deepseek_v2.deepseek_v2_mtp import DeepSeekMTP
from nanodeploy.models.deepseek_v4.deepseek_v4 import DeepseekV4ForCausalLM
from nanodeploy.models.qwen3.qwen3 import Qwen3ForCausalLM
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe_mtp import Qwen3_5MTP
from nanodeploy.models.qwen3_moe.qwen3_moe import Qwen3MoeForCausalLM
from nanodeploy.worker.graph_runner import DecodeGraphRunner
from nanodeploy.worker.input_preparer import InputPreparer, prepare_sample_from_aux
from nanodeploy.worker.loader import load_model, load_mtp_model
from nanodeploy.worker.mtp_worker import MTPWorker
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config
from nanodeploy.worker.vision_embed import VisionEmbedManager

logger = get_logger("NANODEPLOY")


architectures = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV32ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLM,
    "GlmMoeDsaForCausalLM": DeepseekV2ForCausalLM,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
}

architectures_mtp = {
    "DeepseekV3ForCausalLM": DeepSeekMTP,
    "DeepseekV32ForCausalLM": DeepSeekMTP,
    "GlmMoeDsaForCausalLM": DeepSeekMTP,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MTP,
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(
        self,
        config: Config,
        rank: int,
        defer_dist_init: bool = False,
        debug_env: dict[str, str] | None = None,
    ):
        if debug_env:
            os.environ.update({key: str(value) for key, value in debug_env.items()})

        # Set log level
        if config.log_level:
            set_log_level(config.log_level)

        self.config = config
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.attn_world_size
        self.rank = rank
        self._dist_initialized = False
        self._dlslime_agent = None
        self._dlslime_alias = None
        self._dlslime_thread = None
        self._dlslime_peer = None

        # Sync C++ Sequence.block_size with Python kvcache_block_size
        from nanodeploy._cpp import Sequence as _Seq

        _Seq.set_block_size(config.kvcache_block_size)

        logger.debug(f"init ModelRunner, {rank=}, {get_local_ip()=}")

        set_runner_config(
            max_num_seqs=config.max_num_seqs,
            dummy_weight=config.dummy_weight,
            dummy_eplb=config.dummy_eplb,
            enable_eplb=config.enable_eplb,
        )

        if defer_dist_init:
            # NanoOps mode: skip heavy init here; caller will invoke
            # init_dist(master_address) after probing the worker node IP.
            logger.info(f"Deferring dist init for rank {rank} (NanoOps mode)")
            return

        self._complete_dist_init()

    # ------------------------------------------------------------------
    # Node probing (called before dist init in NanoOps mode)
    # ------------------------------------------------------------------

    def get_node_info(self):
        """Return (ip, free_port) of the node this worker runs on."""
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("", 0))
            port = s2.getsockname()[1]
        return ip, port

    def init_dist(self, master_address: str):
        """Complete deferred distributed initialization.

        Called by RayExecutor after probing the worker-node IP.
        Must be invoked on ALL ranks simultaneously (collective call).
        """
        self.config.master_address = master_address
        self._complete_dist_init()

    # ------------------------------------------------------------------

    def _complete_dist_init(self):
        """Phase-2 init: process group, contexts, CUDA, model, etc."""
        config = self.config
        hf_config = config.hf_config
        rank = self.rank

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        torch.cuda.set_device(0)

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            f"tcp://{config.master_address}",
            world_size=self.world_size,
            rank=rank,
        )
        self._dist_initialized = True

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

        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        assert config.attention_sp == 1, "attention_sp > 1 (SP) not supported"
        ep_size = get_dist_context().ffn_ep_world_size

        self.run_count = 0
        self.profiler = None
        if getattr(config, "enable_profiler", False):
            self.profiler_start_step = getattr(config, "profiler_start_step", 10)
            self.profiler_steps = getattr(config, "profiling_step", 10)
            self.profiler_end_step = self.profiler_start_step + self.profiler_steps
            profiler_dir = getattr(config, "profiler_dir", "./profiler_logs")

            os.makedirs(profiler_dir, exist_ok=True)

            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=profiler_dir,
                    worker_name=f"{self.engine_id}_rank_{rank}",
                    use_gzip=False,
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            logger.info(
                f"Rank {rank}: Profiler enabled. Start at {self.profiler_start_step}, duration {self.profiler_steps} steps."
            )

        # Initialise the hardware backend before constructing the model so that
        # all layer factories are available when model __init__ runs.
        from nanodeploy.backends import init_backend
        from nanodeploy.models.quant_config import QuantizationConfig as _QC

        _quant_cfg_dict = getattr(hf_config, "quantization_config", None) or {}
        if not isinstance(_quant_cfg_dict, dict):
            _quant_cfg_dict = {}
        init_backend(quant_config=_QC(**_quant_cfg_dict))

        model_architecture = hf_config.architectures[0]
        self.model = architectures[model_architecture](hf_config)

        # Warmup ExpertContext for MoE models
        num_total_experts = getattr(hf_config, "num_experts", 0) or getattr(
            hf_config, "n_routed_experts", 0
        )
        if num_total_experts > 0:
            ep_rank = get_dist_context().ffn_ep_rank

            # Check FP8 quantization configs
            quant_config = getattr(hf_config, "quantization_config", None)
            is_fp8 = False
            if quant_config is not None:
                is_fp8 = quant_config.get("quant_method", "") == "fp8"

            num_local_experts = num_total_experts // ep_size
            ExpertContext.get_instance().warmup(
                ep_group=get_dist_context().ffn_ep_group,
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                hidden_size=hf_config.hidden_size,
                max_num_sequence=config.max_num_seqs,
                is_fp8=is_fp8,
            )

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        # --- MTP model initialization ---
        mtp_model = None
        if config.num_speculative_tokens > 0:
            mtp_cls = architectures_mtp.get(model_architecture)
            if mtp_cls is None:
                raise ValueError(
                    f"MTP not supported for architecture {model_architecture}"
                )
            mtp_model = mtp_cls(hf_config)
            target_embed = self.model.model.embed_tokens
            mtp_model.embed_tokens = target_embed
            if hasattr(mtp_model, "lm_head") and getattr(
                hf_config, "tie_word_embeddings", False
            ):
                mtp_model.lm_head = self.model.lm_head
            if not get_runner_config().dummy_weight:
                load_mtp_model(mtp_model, config.model)
            # Share lm_head weights with MTP shared_head.head —
            # checkpoints either store identical copies or omit the head entirely.
            for _layer in mtp_model.layers.values():
                if hasattr(_layer, "shared_head") and hasattr(
                    _layer.shared_head, "head"
                ):
                    _layer.shared_head.head.weight = self.model.lm_head.weight
            logger.info(
                f"MTP model loaded: {mtp_cls.__name__}, "
                f"num_speculative_tokens={config.num_speculative_tokens}"
            )

        dist.barrier()

        self.sampler = Sampler()
        self.input_preparer = InputPreparer(config)
        self.vision_manager = VisionEmbedManager(hf_config)
        self.mtp_worker = (
            MTPWorker(config, mtp_model, self.sampler) if mtp_model else None
        )

        self.preallocate_kvcache()

    def num_kvcache_blocks(self):
        return self.config.num_kvcache_blocks

    def allocate_kvcache(self, num_kvcache_blocks: int):
        self.config.num_kvcache_blocks = num_kvcache_blocks
        cache_context = get_cache_context()
        cache_context.allocate_kvcache(num_kvcache_blocks)

        if cache_context.mode == "dsv4":
            self._wire_dsv4_caches(cache_context)
        else:
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

        # Allocate NSA indexer cache (V3.2 only)
        if cache_context.index_head_dim > 0:
            cache_context.allocate_indexer_cache(self.config.hf_config)
            # Wire indexer cache to each layer's Indexer module
            for module in self.model.modules():
                if hasattr(module, "indexer") and module.indexer is not None:
                    module.indexer.indexer_cache = cache_context.indexer_cache

        # Start PeerAgent AFTER kv_cache (and GDN states) are allocated,
        # so that all tensors exist for RDMA memory region registration.
        # In hybrid mode, PeerAgent is started but KV/GDN MR is skipped.
        cache_context.start_peer_agent(mode=self.config.mode)

        if not self.enforce_eager:
            self._init_graph_runners()
        torch.set_default_device("cpu")
        torch.set_default_dtype(self.default_dtype)
        self.warmup_model()

    def _wire_dsv4_caches(self, cache_context):
        """Wire DSv4 FP8 paged SWA cache + compressed caches to attention layers."""
        from nanodeploy.models.deepseek_v4.deepseek_v4 import DeepseekV4Attention

        # Collect compress_ratios from model layers
        compress_ratios = []
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, DeepseekV4Attention):
                compress_ratios.append(getattr(module, "compress_ratio", 0))
                # Wire SWA paged cache (per layer slice)
                module.swa_cache = cache_context.kv_cache[layer_id]
                # Keep old k_cache/v_cache as empty for backward compat
                module.k_cache = torch.tensor([], device=cache_context.device)
                module.v_cache = torch.tensor([], device=cache_context.device)
                layer_id += 1

        # Pool sizes from config (0 = derive worst case)
        pool_pages_per_ratio = {}
        if self.config.dsv4_compressed_pool_pages_ratio4 > 0:
            pool_pages_per_ratio[4] = self.config.dsv4_compressed_pool_pages_ratio4
        if self.config.dsv4_compressed_pool_pages_ratio128 > 0:
            pool_pages_per_ratio[128] = self.config.dsv4_compressed_pool_pages_ratio128

        # Allocate compressed caches for layers with compress_ratio > 0
        cache_context.allocate_dsv4_compressed_caches(
            compress_ratios,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            pool_pages_per_ratio=pool_pages_per_ratio,
        )

        # S2.2: allocate flat compressor scratch state (per ratio, all layers).
        # Layers will use views into these buffers for RDMA-friendly migration.
        cache_context.allocate_dsv4_compressor_state(
            compress_ratios=compress_ratios,
            head_dim=512,  # DSv4 fixed head dim
            max_num_seqs=self.config.max_num_seqs,
        )

        # Wire compressed caches to layers and initialize tensorized compressor state
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, DeepseekV4Attention):
                if layer_id in cache_context.dsv4_compressed_caches:
                    module.compressed_cache = cache_context.dsv4_compressed_caches[
                        layer_id
                    ]
                else:
                    module.compressed_cache = None
                # Initialize tensorized compressor state — pass views into the
                # per-ratio flat tensors when available.
                if hasattr(module, "compressor") and module.compress_ratio > 0:
                    ratio = module.compress_ratio
                    ratio_layer_idx = cache_context.dsv4_layer_to_ratio_idx.get(
                        layer_id
                    )
                    kv_view = score_view = counts_view = None
                    if (
                        ratio_layer_idx is not None
                        and ratio in cache_context.dsv4_compressor_kv_flat
                    ):
                        kv_view = cache_context.dsv4_compressor_kv_flat[ratio][
                            ratio_layer_idx
                        ]
                        score_view = cache_context.dsv4_compressor_score_flat[ratio][
                            ratio_layer_idx
                        ]
                        counts_view = cache_context.dsv4_compressor_counts_flat[ratio][
                            ratio_layer_idx
                        ]
                    module.compressor.init_tensorized_state(
                        max_slots=self.config.max_num_seqs,
                        device=cache_context.device,
                        kv_view=kv_view,
                        score_view=score_view,
                        counts_view=counts_view,
                    )
                layer_id += 1

    def get_peer_agent_addr(self) -> str | None:
        """Return the peer agent address for this rank."""
        return get_cache_context().get_peer_agent_addr()

    def start_dlslime_server(self, driver_alias: str) -> str:
        """Start a dedicated DLSLime server for executor transport."""
        if self._dlslime_alias is not None:
            if self._dlslime_peer != driver_alias:
                raise RuntimeError(
                    "DLSLime server already initialized for a different driver "
                    f"({self._dlslime_peer} != {driver_alias})"
                )
            return self._dlslime_alias

        if not self.config.nanoctrl_address:
            raise RuntimeError(
                "executor_backend='dlslime' requires nanoctrl_address to be set"
            )

        try:
            import dlslime
            from dlslime.rpc import serve
        except ImportError as exc:
            raise RuntimeError(
                "DLSLime transport requires the optional 'dlslime' dependency"
            ) from exc

        from nanodeploy.engine.dlslime_protocol import ModelRunnerRpcService

        available_nics = dlslime.available_nic()
        if not available_nics:
            raise RuntimeError("No available NICs found for DLSLime worker agent")

        agent_alias = f"{self.engine_id}:rpc:{self.rank}"
        device = available_nics[get_dist_context().local_rank % len(available_nics)]
        self._dlslime_agent = dlslime.start_peer_agent(
            alias=agent_alias,
            server_url=self.config.nanoctrl_address,
            device=device,
            ib_port=1,
            link_type="RoCE",
            qp_num=int(os.environ.get("SLIME_QP_NUM", 1)),
            scope=self.config.nanoctrl_scope,
        )
        service = ModelRunnerRpcService(self)

        def _serve_loop():
            self._dlslime_agent.set_desired_topology([driver_alias])
            self._dlslime_agent.wait_for_peers([driver_alias], timeout_sec=30)
            serve(self._dlslime_agent, service, driver_alias)

        self._dlslime_thread = threading.Thread(
            target=_serve_loop,
            daemon=True,
            name=f"dlslime-rank-{self.rank}",
        )
        self._dlslime_thread.start()
        self._dlslime_alias = agent_alias
        self._dlslime_peer = driver_alias
        logger.info(f"Rank {self.rank}: DLSLime server ready at {self._dlslime_alias}")
        return self._dlslime_alias

    def p2p_disconnect(self, remote_engine_id: str):
        return get_cache_context().p2p_disconnect(remote_engine_id)

    def get_num_connected_peers(self):
        return len(get_cache_context().endpoints)

    def exit(self):
        if not self.enforce_eager:
            del self.decode_graph_runner
            if self.mtp_worker is not None:
                self.mtp_worker.cleanup()
        torch.cuda.synchronize()
        dist.destroy_process_group()

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
        # empty for warmup — serialize empty batch into bytes
        warmup_data = serialize_run_batch([], True)
        self.run_from_bytes(warmup_data, True)
        torch.cuda.empty_cache()

    def preallocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config

        # Detect cache mode: dsv4, mla, or gqa
        is_dsv4 = hf_config.architectures[0] == "DeepseekV4ForCausalLM"
        if is_dsv4:
            mode = "dsv4"
        elif getattr(hf_config, "kv_lora_rank", 0) > 0:
            mode = "mla"
        else:
            mode = "gqa"
        kv_lora_rank = (
            hf_config.kv_lora_rank if hasattr(hf_config, "kv_lora_rank") else 0
        )
        qk_rope_head_dim = (
            hf_config.qk_rope_head_dim if hasattr(hf_config, "qk_rope_head_dim") else 0
        )

        # For mixed attention models (Qwen3.5-MoE), only full_attention layers
        # need KV cache. Count the number of full_attention layers.
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types is not None:
            num_kv_layers = sum(1 for lt in layer_types if lt == "full_attention")
        else:
            num_kv_layers = hf_config.num_hidden_layers

        # If nanoctrl_address is provided, fetch engine_id from NanoCtrl
        engine_id = config.engine_id
        if config.nanoctrl_address and not engine_id:
            engine_id = _get_engine_id_from_nanoctrl(
                config.nanoctrl_address, config.host, config.port
            )

        # Enable FP8 KV cache for sparse attention (V3.2)
        # On by default when model has NSA indexer; --disable_nsa turns it off
        index_head_dim = getattr(hf_config, "index_head_dim", 0)
        is_fp8_kvcache = (mode == "mla" and index_head_dim > 0) and not getattr(
            config, "disable_nsa", False
        )

        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        cache_context = set_cache_context(
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            block_size=config.kvcache_block_size,
            num_hidden_layers=num_kv_layers,
            attention_tp=config.attention_tp,
            gpu_memory_utilization=config.gpu_memory_utilization,
            gpu_memory_limit_gb=config.gpu_memory_limit_gb,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            is_fp8_kvcache=is_fp8_kvcache,
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
            mode=mode,
            nanoctrl_address=config.nanoctrl_address,
            nanoctrl_scope=config.nanoctrl_scope,
            engine_id=engine_id,
        )
        config.num_kvcache_blocks = cache_context.num_local_kvcache_blocks

        # Allocate GDN state buffers for linear_attention layers
        if layer_types is not None:
            cache_context.allocate_gdn_states(
                hf_config,
                layer_types,
                config.max_num_seqs,
                need_backup=config.num_speculative_tokens > 0,
            )

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            context = get_context()
            inputs_embeds = None
            if is_prefill and self.vision_manager.has_embeds:
                logger.info(
                    f"[RUN_MODEL] Injecting vision embeds for prefill, "
                    f"input_ids.shape={input_ids.shape}"
                )
                inputs_embeds = self.vision_manager.inject(
                    input_ids, self.model.model.embed_tokens
                )
                self.vision_manager.clear()
            elif is_prefill and not context.is_dummy:
                logger.debug(
                    f"[RUN_MODEL] Prefill WITHOUT vision embeds, "
                    f"input_ids.shape={input_ids.shape}"
                )
            if inputs_embeds is not None:
                hidden = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
            else:
                hidden = self.model(input_ids, positions)
            if is_prefill:
                ExpertContext.get_instance().transition_to_low_latency()
            if not is_prefill and self.mtp_worker is not None:
                self.mtp_worker.last_hidden = hidden
            return self.model.compute_logits(hidden)
        else:
            context = get_context()

            # Lazy verify path (seqlen_q=2): dedicated graph runner
            if (
                context.num_tokens_per_seq == 2
                and self.mtp_worker is not None
                and self.mtp_worker.lv_graph_runner is not None
            ):
                outputs = self.mtp_worker.lv_graph_runner.run(
                    input_ids, positions, context
                )
                if outputs is not None:
                    if self.mtp_worker is not None:
                        self.mtp_worker.last_hidden = outputs.clone()
                    return self.model.compute_logits(outputs)
                # No graph for this bs — eager fallback
                hidden = self.model(input_ids, positions)
                if self.mtp_worker is not None:
                    self.mtp_worker.last_hidden = hidden
                return self.model.compute_logits(hidden)

            # Normal decode (seqlen_q=1)
            outputs = self.decode_graph_runner.run(input_ids, positions, context)
            if self.mtp_worker is not None:
                self.mtp_worker.last_hidden = outputs.clone()
            return self.model.compute_logits(outputs)

    def migrate_from_bytes(self, data: bytes) -> None:
        """Migrate using lean MigrateBatchInput bytes (no Sequence objects)."""
        get_cache_context().migrate_from_bytes(data=data)

    def _standard_sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        aux,
        num_seqs: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Standard sampling path (prefill or normal decode without lazy verify)."""
        tp_rank = get_dist_context().attn_tp_rank
        if tp_rank == 0:
            temperatures = prepare_sample_from_aux(aux)
            context = get_context()
            if is_prefill and context.sampling_seq_indices is not None:
                temps_filtered = temperatures[context.sampling_seq_indices]
                sampled = self.sampler(logits, temps_filtered)
                input_ids = input_ids.new_zeros(num_seqs)
                input_ids[context.sampling_seq_indices] = sampled
            else:
                input_ids = self.sampler(logits, temperatures)
        else:
            input_ids = input_ids.new_zeros([num_seqs])
        dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)
        return input_ids

    @torch.inference_mode()
    def run_from_bytes(self, data: bytes, is_prefill: bool) -> list[list[int]]:
        """Run model from lean RunBatchInput bytes (completely Sequence-free)."""
        sp_rank = get_dist_context().attn_sp_rank
        aux = extract_aux_from_bytes(data, sp_rank)
        num_seqs = aux.num_group_seqs

        is_dummy = False
        if num_seqs == 0:
            is_dummy = True
            from nanodeploy._cpp import SamplingParams, Sequence as _Seq

            dummy_seq = _Seq([0], SamplingParams())
            dummy_seq.block_ctx().reset(
                self.engine_id,
                1,
                1,
                get_cache_context().num_local_kvcache_blocks,
            )
            dummy_seq.block_ctx().master_group_id = 0
            data = serialize_run_batch([dummy_seq], is_prefill)
            aux = extract_aux_from_bytes(data, sp_rank)
            num_seqs = aux.num_group_seqs

        # Determine loop count
        if is_prefill:
            loop_count = 1
            if self.mtp_worker is not None:
                self.mtp_worker.reset_lazy_verify_state()
        elif hasattr(self.config, "_mtp_original_loop_count"):
            loop_count = self.config._mtp_original_loop_count
        else:
            loop_count = self.config.loop_count

        for i in range(loop_count):
            # --- Profiler start ---
            if self.profiler and self.run_count == self.profiler_start_step:
                self.profiler.start()
                logger.info(
                    f"Rank {self.rank}: Profiler started at step {self.run_count}"
                )

            # --- Prepare inputs ---
            has_lazy_verify = False
            if is_prefill:
                if i == 0 and not self.vision_manager.has_embeds:
                    vision_slots = extract_vision_slots_from_bytes(data)
                    if vision_slots:
                        self.vision_manager.fetch_rdma(
                            vision_slots, self.model.model.embed_tokens.weight.dtype
                        )
                input_ids, positions = self.input_preparer.prepare_prefill_bytes(
                    data, aux, is_dummy
                )
            else:
                if i == 0:
                    input_ids, positions = self.input_preparer.prepare_decode_bytes(
                        data, aux, is_dummy
                    )
                else:
                    input_ids, positions = self.input_preparer.update_decode_inplace(
                        input_ids, positions, num_seqs
                    )

                if (
                    self.mtp_worker is not None
                    and self.mtp_worker.has_drafts
                    and not is_dummy
                ):
                    has_lazy_verify = True
                    input_ids, positions = self.mtp_worker.prepare_lazy_verify_decode(
                        input_ids, positions, num_seqs
                    )

            if input_ids.numel() == 0:
                logger.critical(
                    "EMPTY input_ids before run_model! rank=%s is_prefill=%s "
                    "is_dummy=%s input_ids.shape=%s positions.shape=%s num_seqs=%s",
                    self.rank,
                    is_prefill,
                    is_dummy,
                    input_ids.shape,
                    positions.shape,
                    num_seqs,
                )

            # --- Forward ---
            logits = self.run_model(input_ids, positions, is_prefill)
            if is_prefill and self.vision_manager.has_embeds:
                self.vision_manager.clear()

            # --- Sampling ---
            num_accepted = None
            if not is_prefill and has_lazy_verify:
                num_accepted = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")
                input_ids = self.mtp_worker.lazy_verify_sample(
                    logits, aux, num_seqs, num_accepted
                )
            else:
                input_ids = self._standard_sample(
                    logits, input_ids, aux, num_seqs, is_prefill
                )

            # --- MTP draft generation ---
            if self.mtp_worker is not None and not is_prefill and not is_dummy:
                self.mtp_worker.generate_and_store(
                    input_ids, positions, aux, num_seqs, has_lazy_verify, num_accepted
                )

            # --- Profiler step ---
            if self.profiler and self.run_count >= self.profiler_start_step:
                if self.run_count < self.profiler_end_step:
                    self.profiler.step()
                if self.run_count == self.profiler_end_step - 1:
                    self.profiler.stop()
                    logger.info(
                        f"Rank {self.rank}: Profiler stopped and saved at step {self.run_count}"
                    )

            self.run_count += 1
            get_context().token_ids.append(input_ids[None, ...])

        # --- Build output ---
        if self.mtp_worker is not None:
            result = self.mtp_worker.build_output_tokens(self.rank)
        else:
            result = torch.cat(get_context().token_ids, dim=0).T.tolist()
        reset_context()
        return result

    def _init_graph_runners(self):
        """Initialize CUDAGraph runners for decode, MTP, and lazy verify."""
        config = self.config
        hf_config = config.hf_config
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        cache_ctx = get_cache_context()

        self.decode_graph_runner = DecodeGraphRunner(config, hf_config, cache_ctx)
        graph_pool = self.decode_graph_runner.capture(self.model, cache_ctx)

        if self.mtp_worker is not None:
            self.mtp_worker.init_graph_runners(self.model, graph_pool, cache_ctx)

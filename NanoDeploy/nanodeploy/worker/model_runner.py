import os

import flash_mla
import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.profiler as profiler
from nanodeploy._cpp import (
    extract_aux_from_bytes,
    parse_migrate_batch,
    prepare_decode_from_bytes,
    prepare_prefill_from_bytes,
    serialize_run_batch,
)
from nanodeploy.config import Config
from nanodeploy.context.cache import get_cache_context, set_cache_context
from nanodeploy.context.context import get_context, reset_context, set_context
from nanodeploy.context.distributed import (
    get_dist_context,
    get_local_ip,
    set_dist_context,
)
from nanodeploy.context.expert_context import ExpertContext
from nanodeploy.context.sp_context import set_sp_context
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.models.qwen3.qwen3 import Qwen3ForCausalLM
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from nanodeploy.models.qwen3_moe.qwen3_moe import Qwen3MoeForCausalLM
from nanodeploy.worker.loader import load_model
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config

logger = get_logger("NANODEPLOY")


architectures = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(self, config: Config, rank: int, defer_dist_init: bool = False):
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

        # Sync C++ Sequence.block_size with Python kvcache_block_size
        from nanodeploy._cpp import Sequence as _Seq

        _Seq.set_block_size(config.kvcache_block_size)

        # Propagate scope to actor environment: the Config object carries
        # scope from the driver (set from NANOCTRL_SCOPE env var), but
        # actor processes may not inherit the job's env vars.  Libraries
        # like dlslime read NANOCTRL_SCOPE from os.environ, so we must
        # set it here to ensure correct scoped registration in Redis.
        if config.nanoctrl_scope and not os.getenv("NANOCTRL_SCOPE"):
            os.environ["NANOCTRL_SCOPE"] = config.nanoctrl_scope
            logger.info(
                f"Set NANOCTRL_SCOPE={config.nanoctrl_scope} in actor environment"
            )

        logger.debug(f"init ModelRunner, {rank=}, {get_local_ip()=}")

        set_runner_config(
            max_num_seqs=config.max_num_seqs,
            dummy_weight=config.dummy_weight,
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

        sp_size = get_dist_context().attn_sp_world_size
        ep_size = get_dist_context().ffn_ep_world_size

        if sp_size > 1:
            sp_rank = get_dist_context().attn_sp_rank
            max_head_dim = 0
            if config.hf_config.num_key_value_heads > 1:
                max_head_dim = config.hf_config.head_dim
            else:
                max_head_dim = (
                    config.hf_config.kv_lora_rank + config.hf_config.qk_rope_head_dim
                )
            set_sp_context(
                config.max_num_seqs,
                max_head_dim,
                hf_config.num_attention_heads,
                torch.get_default_dtype(),
                sp_size,
                sp_rank,
            )

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

        dist.barrier()

        self.sampler = Sampler()
        self.preallocate_kvcache()

        # Vision embedding side-channel for VL inference
        self._vision_embeds: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Vision embedding side-channel (VL inference)
    # ------------------------------------------------------------------

    def set_vision_embeds(self, embeds: dict[str, torch.Tensor]) -> None:
        """Store vision embeddings for injection during prefill.

        Called by VLEngine via RayExecutor before a prefill step that
        contains vision tokens.

        Args:
            embeds: Dict with optional keys ``"image"`` and ``"video"``,
                each mapping to a CPU tensor of shape ``[num_tokens, hidden]``.
        """
        self._vision_embeds = {
            k: v.to(device="cuda", dtype=torch.get_default_dtype(), non_blocking=True)
            for k, v in embeds.items()
        }

    def clear_vision_embeds(self) -> None:
        """Clear stored vision embeddings after prefill."""
        self._vision_embeds = None

    def _inject_vision_embeds(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Build ``inputs_embeds`` by merging text + vision embeddings.

        If no vision embeddings are stored, returns ``None`` so that the
        model falls back to its normal ``embed_tokens(input_ids)`` path.
        """
        if self._vision_embeds is None:
            return None

        # Get text embeddings from the model's embedding layer
        embed_tokens = self.model.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        hf_config = self.config.hf_config

        # Inject image embeddings
        if "image" in self._vision_embeds:
            image_token_id = getattr(hf_config, "image_token_id", None)
            if image_token_id is not None:
                image_embeds = self._vision_embeds["image"].to(
                    dtype=inputs_embeds.dtype
                )
                mask = input_ids == image_token_id
                n_tokens = mask.sum().item()
                if n_tokens > 0 and n_tokens == image_embeds.shape[0]:
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        mask_expanded, image_embeds
                    )

        # Inject video embeddings
        if "video" in self._vision_embeds:
            video_token_id = getattr(hf_config, "video_token_id", None)
            if video_token_id is not None:
                video_embeds = self._vision_embeds["video"].to(
                    dtype=inputs_embeds.dtype
                )
                mask = input_ids == video_token_id
                n_tokens = mask.sum().item()
                if n_tokens > 0 and n_tokens == video_embeds.shape[0]:
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        mask_expanded, video_embeds
                    )

        return inputs_embeds

    def num_kvcache_blocks(self):
        return self.config.num_kvcache_blocks

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

        # Start PeerAgent AFTER kv_cache (and GDN states) are allocated,
        # so that all tensors exist for RDMA memory region registration.
        cache_context.start_peer_agent()

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(self.default_dtype)
        self.warmup_model()

    def get_peer_agent_addr(self) -> str | None:
        """Return the peer agent address for this rank."""
        return get_cache_context().get_peer_agent_addr()

    def p2p_disconnect(self, remote_engine_id: str):
        return get_cache_context().p2p_disconnect(remote_engine_id)

    def get_num_connected_peers(self):
        return len(get_cache_context().endpoints)

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
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
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        # empty for warmup — serialize empty batch into bytes
        warmup_data = serialize_run_batch([], True)
        self.run_from_bytes(warmup_data, True)
        torch.cuda.empty_cache()

    def preallocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config

        # Detect MLA by presence of kv_lora_rank
        mode = "mla" if getattr(hf_config, "kv_lora_rank", 0) > 0 else "gqa"
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

        cache_context = set_cache_context(
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            block_size=config.kvcache_block_size,
            num_hidden_layers=num_kv_layers,
            attention_tp=config.attention_tp,
            gpu_memory_utilization=config.gpu_memory_utilization,
            gpu_memory_limit_gb=config.gpu_memory_limit_gb,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
            mode=mode,
            nanoctrl_address=config.nanoctrl_address,
            engine_id=engine_id,
        )
        config.num_kvcache_blocks = cache_context.num_local_kvcache_blocks

        # Allocate GDN state buffers for linear_attention layers
        if layer_types is not None:
            cache_context.allocate_gdn_states(
                hf_config, layer_types, config.max_num_seqs
            )

    def prepare_prefill_bytes(self, data: bytes, aux, is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        meta = prepare_prefill_from_bytes(
            data,
            sp_rank,
            sp_size,
            block_size,
            self.config.max_num_seqs,
            self.config.num_kvcache_blocks,
        )

        if len(meta.input_ids) == 0:
            logger.critical(
                "prepare_prefill_from_bytes returned empty input_ids! "
                "is_dummy=%s sp_rank=%s sp_size=%s block_size=%s max_num_seqs=%s",
                is_dummy,
                sp_rank,
                sp_size,
                block_size,
                self.config.max_num_seqs,
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
        if meta.use_block_tables:
            block_tables = (
                torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
                .reshape(sp_size, self.config.max_num_seqs, meta.max_num_blocks)
                .cuda(non_blocking=True)
            )

        cache_ctx = get_cache_context()
        gdn_state_slots = None
        if cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            gdn_state_slots = torch.tensor(
                [
                    s if 0 <= s < dummy_gdn_slot else dummy_gdn_slot
                    for s in aux.state_slots
                ],
                dtype=torch.int64,
                pin_memory=True,
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
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
        )
        return input_ids, positions

    def prepare_decode_bytes(self, data: bytes, aux, is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        try:
            meta = prepare_decode_from_bytes(
                data,
                sp_rank,
                sp_size,
                block_size,
                self.config.max_num_seqs,
                self.config.num_kvcache_blocks,
            )
        except (IndexError, ValueError, RuntimeError) as e:
            logger.error(
                "prepare_decode_from_bytes failed: %s (sp_rank=%s sp_size=%s block_size=%s max_num_seqs=%s)",
                str(e),
                sp_rank,
                sp_size,
                block_size,
                self.config.max_num_seqs,
            )
            raise

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
            block_tables = torch.empty((0, 0), dtype=torch.int32).cuda(
                non_blocking=True
            )
        else:
            block_tables = (
                torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
                .reshape(-1, meta.max_num_blocks)
                .cuda(non_blocking=True)
            )

        q_mask = global_context_lens.clone()
        q_mask[sp_rank].fill_(0)
        q_mask[q_mask != 0] = 1
        res_lse_mask = context_lens.clone()
        res_lse_mask[sp_rank].fill_(0)
        res_lse_mask[res_lse_mask != 0] = 1

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
        attention_compute_bs = context_lens_for_attn.numel()

        config = self.config
        hf_config = config.hf_config
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            new_tile_scheduler_metadata, new_num_splits = flash_mla.get_mla_metadata(
                context_lens_for_attn.view(-1),
                hf_config.num_attention_heads // mla_num_kv_heads,
                mla_num_kv_heads,
            )
        else:
            new_tile_scheduler_metadata, new_num_splits = None, None

        cache_ctx = get_cache_context()
        gdn_state_slots = None
        if cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = cache_ctx.gdn_conv_states.shape[1] - 1
            gdn_state_slots = torch.tensor(
                [
                    s if 0 <= s < dummy_gdn_slot else dummy_gdn_slot
                    for s in aux.state_slots
                ],
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True)

        set_context(
            is_prefill=False,
            max_bs=self.config.max_num_seqs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            q_mask=q_mask,
            res_lse_mask=res_lse_mask,
            is_dummy=is_dummy,
            context_lens_for_attn=context_lens_for_attn,
            attention_compute_bs=attention_compute_bs,
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
            tile_scheduler_metadata=new_tile_scheduler_metadata,
            num_splits=new_num_splits,
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
        )

        return input_ids, positions

    def update_decode_inplace(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_sp_seqs: int
    ):
        """Update decode metadata in-place for multi-step decode (no Sequence needed)."""
        positions.add_(1)
        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        context = get_context()

        # Update context length (now reflects the NEW token count)
        context.context_lens[sp_rank][:num_sp_seqs].add_(1)
        # Update global context length
        context.global_context_lens[sp_rank][:num_sp_seqs].add_(1)
        # Update context lens for attention
        context.context_lens_for_attn[context.q_slice_fill.long()] += 1

        # Recalculate slot_mapping from context_lens and block_tables.
        # Simply doing slot_mapping.add_(1) is WRONG when a sequence's new
        # token crosses a block boundary, because the page_id changes.
        new_ctx = context.context_lens[sp_rank][:num_sp_seqs]  # already incremented
        block_idx = (new_ctx - 1) // block_size  # which block the new token falls in
        offset_in_block = (new_ctx - 1) % block_size  # offset within that block
        # block_tables is packed by sp_rank order; use q_offsets to find the
        # starting row for this sp_rank's sequences.
        bt_offset = context.q_offsets[sp_rank]
        row_indices = bt_offset + torch.arange(num_sp_seqs, device=block_idx.device)
        page_ids = context.block_tables[row_indices, block_idx.long()]
        context.slot_mapping[:num_sp_seqs] = page_ids * block_size + offset_in_block

        return input_ids, positions

    def prepare_sample_from_aux(self, aux):
        """Build temperature tensor from BatchAuxData (no Sequence needed)."""
        temperatures = torch.tensor(
            aux.temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            context = get_context()
            # Inject vision embeddings during prefill if available
            inputs_embeds = None
            if is_prefill and self._vision_embeds is not None:
                inputs_embeds = self._inject_vision_embeds(input_ids)
            return self.model.compute_logits(
                self.model(input_ids, positions, inputs_embeds=inputs_embeds)
            )
        else:
            bs = input_ids.size(0)
            context = get_context()
            master_bs = next(x for x in self.graph_master_rank_bs if x >= bs)

            ac_bs = context.attention_compute_bs
            if ac_bs is None:
                ac_bs = bs
            valid_attn_bs_list = self.graph_map.get(master_bs)
            if valid_attn_bs_list is None:
                raise RuntimeError(f"No graph map found for master_bs={master_bs}")

            try:
                attn_bs = next(x for x in valid_attn_bs_list if x >= ac_bs)
            except StopIteration:
                raise RuntimeError(
                    f"Input attention_compute_bs {ac_bs} exceeds max captured attn_bs "
                    f"({valid_attn_bs_list[-1]}) for master_bs {master_bs}"
                )

            graph = self.graphs[(master_bs, attn_bs)]

            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping  # type: ignore
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"].copy_(context.context_lens)  # type: ignore
            graph_vars["global_context_lens"].zero_()
            graph_vars["global_context_lens"].copy_(context.global_context_lens)  # type: ignore
            graph_vars["q_mask"].zero_()
            graph_vars["q_mask"].copy_(context.q_mask)  # type: ignore
            graph_vars["res_lse_mask"].zero_()
            graph_vars["res_lse_mask"].copy_(context.res_lse_mask)  # type: ignore
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][
                : context.block_tables.size(0), : context.block_tables.size(1)  # type: ignore
            ] = context.block_tables

            config = self.config
            hf_config = config.hf_config
            is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
            if is_mla:
                graph_vars["tile_scheduler_metadata"].zero_()
                graph_vars["num_splits"].zero_()
                graph_vars["tile_scheduler_metadata"].copy_(context.tile_scheduler_metadata)  # type: ignore
                graph_vars["num_splits"][: context.num_splits.shape[0]].copy_(context.num_splits)  # type: ignore

            graph_vars["context_lens_for_attn"].zero_()
            graph_vars["context_lens_for_attn"][: context.context_lens_for_attn.shape[0]].copy_(context.context_lens_for_attn)  # type: ignore

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

            if graph_vars.get("gdn_state_slots") is not None:
                dummy_gdn_slot = get_cache_context().gdn_conv_states.shape[1] - 1
                graph_vars["gdn_state_slots"].fill_(dummy_gdn_slot)
                if context.gdn_state_slots is not None:
                    graph_vars["gdn_state_slots"][:bs].copy_(context.gdn_state_slots)

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def migrate_from_bytes(self, data: bytes) -> None:
        """Migrate using lean MigrateBatchInput bytes (no Sequence objects)."""
        get_cache_context().migrate_from_bytes(data=data)

    @torch.inference_mode()
    def run_from_bytes(self, data: bytes, is_prefill: bool) -> list[list[int]]:
        """Run model from lean RunBatchInput bytes (completely Sequence-free)."""
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size

        # Extract auxiliary data (temperatures, state_slots, master_sp_indices)
        aux = extract_aux_from_bytes(data, sp_rank)
        num_sp_seqs = aux.num_sp_seqs

        is_dummy = False
        if num_sp_seqs == 0:
            is_dummy = True
            # Create a minimal dummy RunBatchInput with one dummy sequence
            from nanodeploy._cpp import SamplingParams, Sequence as _Seq

            dummy_seq = _Seq([0], SamplingParams())
            dummy_seq.block_ctx().reset(
                self.engine_id,
                sp_size,
                1,
                get_cache_context().num_local_kvcache_blocks,
            )
            dummy_seq.block_ctx().master_sp_idx = sp_rank
            data = serialize_run_batch([dummy_seq], is_prefill)
            aux = extract_aux_from_bytes(data, sp_rank)
            num_sp_seqs = aux.num_sp_seqs

        loop_count = self.config.loop_count if not is_prefill else 1
        for i in range(loop_count):
            if self.profiler and self.run_count == self.profiler_start_step:
                self.profiler.start()
                logger.info(
                    f"Rank {self.rank}: Profiler started at step {self.run_count}"
                )

            if is_prefill:
                input_ids, positions = self.prepare_prefill_bytes(data, aux, is_dummy)
            else:
                if i == 0:
                    input_ids, positions = self.prepare_decode_bytes(
                        data, aux, is_dummy
                    )
                else:
                    input_ids, positions = self.update_decode_inplace(
                        input_ids, positions, num_sp_seqs
                    )

            if input_ids.numel() == 0:
                logger.critical(
                    "EMPTY input_ids before run_model! rank=%s is_prefill=%s "
                    "is_dummy=%s input_ids.shape=%s positions.shape=%s "
                    "num_sp_seqs=%s",
                    self.rank,
                    is_prefill,
                    is_dummy,
                    input_ids.shape,
                    positions.shape,
                    num_sp_seqs,
                )
            logits = self.run_model(input_ids, positions, is_prefill)

            tp_rank = get_dist_context().attn_tp_rank
            if tp_rank == 0:
                temperatures = self.prepare_sample_from_aux(aux)
                input_ids = self.sampler(logits, temperatures)
            else:
                input_ids = input_ids.new_zeros([num_sp_seqs])
            dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)

            # No update_seqs_inner_loop needed — metadata already updated in-place

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

        loop_count_token_ids = torch.cat(get_context().token_ids, dim=0).T.tolist()
        reset_context()

        return loop_count_token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        sp_world_size = get_dist_context().attn_sp_world_size
        config = self.config
        hf_config = config.hf_config
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        max_bs = min(self.config.max_num_seqs, 512)
        max_attention_comp_seqs = max_bs + config.max_num_recv_seqs
        block_size = get_cache_context().block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens_for_attn = torch.zeros(max_attention_comp_seqs, dtype=torch.int32)
        context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        global_context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        q_mask = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
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
            (config.max_num_recv_seqs,), -1, dtype=torch.int32
        )
        res_slice_fill_to_buffer_input = torch.full(
            (config.max_num_recv_seqs,), -1, dtype=torch.int32
        )
        res_to_buffer_input_mask = torch.zeros(
            config.max_num_recv_seqs, dtype=torch.int32
        )
        q_offsets = torch.zeros(sp_world_size + 1, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            tile_scheduler_metadata_buffer, num_splits_buffer = (
                flash_mla.get_mla_metadata(
                    torch.ones(
                        max_attention_comp_seqs, dtype=torch.int32, device="cuda"
                    ),
                    hf_config.num_attention_heads // mla_num_kv_heads,
                    mla_num_kv_heads,
                )
            )
        else:
            tile_scheduler_metadata_buffer, num_splits_buffer = None, None

        # GDN state slot indices for CUDAGraph (maps batch position -> buffer slot)
        _cache_ctx = get_cache_context()
        gdn_state_slots_buf = None
        if _cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = _cache_ctx.gdn_conv_states.shape[1] - 1
            gdn_state_slots_buf = torch.full(
                (max_bs,), dummy_gdn_slot, dtype=torch.int64
            )

        self.graph_master_rank_bs = [x for x in [1, 2, 4, 8] if x <= max_bs] + list(
            range(16, max_bs + 1, 16)
        )
        self.graphs = {}
        self.graph_pool = None
        self.graph_map = {}  # store master_bs -> [available_attn_bs...]

        self.attn_bs_step = 16  # 定义 attn_bs 的步长

        total_graphs = 0

        logger.info(f"开始捕获 CUDAGraph...")
        completed_graphs = 0

        for master_bs in reversed(self.graph_master_rank_bs):
            self.graph_map[master_bs] = []

            current_attn_bs_candidates = []
            curr = master_bs
            limit = master_bs + config.max_num_recv_seqs
            while curr <= limit:
                current_attn_bs_candidates.append(curr)
                curr += self.attn_bs_step

            for attn_bs in reversed(current_attn_bs_candidates):

                completed_graphs += 1
                logger.info(f"正在捕获图 - (master_bs={master_bs}, attn_bs={attn_bs})")
                graph = torch.cuda.CUDAGraph()
                set_context(
                    is_prefill=False,
                    max_bs=self.config.max_num_seqs,
                    slot_mapping=slot_mapping[:master_bs],
                    context_lens=context_lens,
                    block_tables=block_tables,
                    global_context_lens=global_context_lens,
                    q_mask=q_mask,
                    res_lse_mask=res_lse_mask,
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
                    context_lens_for_attn=context_lens_for_attn,
                    q_offsets=q_offsets,
                    tile_scheduler_metadata=tile_scheduler_metadata_buffer,
                    num_splits=num_splits_buffer,
                    gdn_conv_states=_cache_ctx.gdn_conv_states,
                    gdn_recurrent_states=_cache_ctx.gdn_recurrent_states,
                    gdn_state_slots=(
                        gdn_state_slots_buf[:master_bs]
                        if gdn_state_slots_buf is not None
                        else None
                    ),
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

                self.graphs[(master_bs, attn_bs)] = graph
                self.graph_map[master_bs].append(attn_bs)

                torch.cuda.synchronize()
                dist.barrier(group=get_dist_context().cuda_world_group)
                reset_context()

            self.graph_map[master_bs].sort()

        logger.info(f"完成所有 graph 的捕获，成功捕获 {len(self.graphs)} 个图")

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            outputs=outputs,
            q_mask=q_mask,
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
            gdn_state_slots=gdn_state_slots_buf,
        )

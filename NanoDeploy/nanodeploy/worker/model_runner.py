import os

import flash_mla
import numpy as np
import ray
import torch
import torch.distributed as dist
from nanodeploy._cpp import (
    extract_aux_from_bytes,
    extract_vision_slots_from_bytes,
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
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.models.deepseek_v2.deepseek_v2_mtp import DeepSeekMTP
from nanodeploy.models.qwen3.qwen3 import Qwen3ForCausalLM
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe_mtp import Qwen3_5MTP
from nanodeploy.models.qwen3_moe.qwen3_moe import Qwen3MoeForCausalLM
from nanodeploy.worker.loader import load_model, load_mtp_model
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config

logger = get_logger("NANODEPLOY")


architectures = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
}

architectures_mtp = {
    "DeepseekV3ForCausalLM": DeepSeekMTP,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MTP,
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
        self.mtp_model = None
        self._last_hidden = None
        self.mtp_graphs = None
        self.mtp_graph_vars = None
        # Lazy verify state: drafts and sampled token from previous decode step
        self._prev_drafts: list[torch.Tensor] | None = None
        self._prev_sampled_token: torch.Tensor | None = None
        if config.num_speculative_tokens > 0:
            mtp_cls = architectures_mtp.get(model_architecture)
            if mtp_cls is None:
                raise ValueError(
                    f"MTP not supported for architecture {model_architecture}"
                )
            self.mtp_model = mtp_cls(hf_config)
            # Share embeddings from target model
            target_embed = self.model.model.embed_tokens
            self.mtp_model.embed_tokens = target_embed
            # For Qwen3.5: share lm_head if tie_word_embeddings
            if hasattr(self.mtp_model, "lm_head") and getattr(
                hf_config, "tie_word_embeddings", False
            ):
                self.mtp_model.lm_head = self.model.lm_head
            if not get_runner_config().dummy_weight:
                load_mtp_model(self.mtp_model, config.model)
            logger.info(
                f"MTP model loaded: {mtp_cls.__name__}, "
                f"num_speculative_tokens={config.num_speculative_tokens}"
            )

        dist.barrier()

        self.sampler = Sampler()
        self.preallocate_kvcache()

        # Vision embeddings fetched via RDMA from encoder (EP-separated mode)
        self._vision_embeds: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Vision embedding injection (EP-separated mode)
    # ------------------------------------------------------------------

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
                    logger.debug(f"Injected {n_tokens} image tokens")
                else:
                    logger.warning(
                        f"Image token count mismatch: input has {n_tokens}, "
                        f"embeds has {image_embeds.shape[0]} — skipping injection"
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

    # ------------------------------------------------------------------
    # Vision embedding RDMA fetch (EP-separated mode)
    # ------------------------------------------------------------------

    def _fetch_vision_embeds_rdma(self, vision_slot_views: list) -> None:
        """RDMA-fetch vision embeddings from remote encoder(s).

        Reads embeddings from encoder EmbeddingPool into a local receive
        buffer via dlslime, then stores them in ``self._vision_embeds``
        for injection during model forward.

        Args:
            vision_slot_views: List of VisionSlotView from
                ``extract_vision_slots_from_bytes``.
        """
        if not vision_slot_views:
            return

        cache_ctx = get_cache_context()
        peer_agent = cache_ctx._peer_agent
        if peer_agent is None:
            logger.warning("PeerAgent not available, cannot RDMA-fetch vision embeds")
            return

        # Group vision slots by encoder_engine_id for batched reads
        from collections import defaultdict

        from nanodeploy.context.embedding_pool import _VISION_EMBED_BUFFER_ID

        by_encoder: dict[str, list] = defaultdict(list)
        for v in vision_slot_views:
            by_encoder[v.encoder_engine_id].append(v)

        # Compute total tokens for local receive buffer
        total_tokens = sum(v.num_tokens for v in vision_slot_views)
        hidden_size = vision_slot_views[0].hidden_size
        # Use model's embedding dtype to match encoder EmbeddingPool dtype
        # (RDMA copies raw bytes, so local buffer dtype must match remote)
        dtype = self.model.model.embed_tokens.weight.dtype
        itemsize = torch.tensor([], dtype=dtype).element_size()

        # Allocate local receive buffer on GPU
        recv_buf = torch.zeros(total_tokens, hidden_size, dtype=dtype, device="cuda")
        # Register receive buffer as a temporary MR
        recv_buf_size = recv_buf.nelement() * recv_buf.element_size()
        recv_mr = peer_agent.register_memory_region(
            "vision_recv",
            recv_buf.data_ptr(),
            int(recv_buf.storage_offset()),
            recv_buf_size,
        )

        # Look up peer_addrs for all encoders via NanoCtrl (cached, single request per engine)
        encoder_info_map = cache_ctx._fetch_engine_info_from_nanoctrl(
            set(by_encoder.keys())
        )

        token_offset = 0
        for encoder_id, slots in by_encoder.items():
            # Resolve peer alias from control plane; fall back to legacy convention
            encoder_info = encoder_info_map.get(encoder_id, {})
            peer_addrs = encoder_info.get("peer_addrs", [])
            if peer_addrs:
                peer_alias = peer_addrs[0]
            else:
                logger.warning(
                    f"No peer_addrs for encoder {encoder_id} in NanoCtrl, "
                    "falling back to legacy alias"
                )
                peer_alias = f"{encoder_id}:0"

            # Ensure connection
            if peer_alias not in cache_ctx._connected_peers:
                cache_ctx._peer_agent.set_desired_topology(
                    target_peers=list(cache_ctx._connected_peers | {peer_alias}),
                    symmetric=True,
                )
                cache_ctx._peer_agent.wait_for_peers([peer_alias], timeout_sec=30)
                cache_ctx._connected_peers.add(peer_alias)

            # Get remote MR info for vision_embed buffer
            remote_mr_info = peer_agent.get_mr_info(peer_alias, _VISION_EMBED_BUFFER_ID)
            if remote_mr_info is None:
                logger.error(
                    f"Failed to get MR info for vision_embed from {peer_alias}"
                )
                continue

            remote_mr = peer_agent.register_remote_memory_region(
                peer_alias, _VISION_EMBED_BUFFER_ID, remote_mr_info
            )
            endpoint = peer_agent.get_endpoint(peer_alias)
            if endpoint is None:
                logger.error(f"Failed to get endpoint for {peer_alias}")
                continue

            # Build RDMA read ops for all slots from this encoder
            rdma_ops = []
            for v in slots:
                # Encoder EmbeddingPool layout: [num_slots, max_tokens_per_slot, hidden_size]
                # Remote offset for slot = slot_idx * max_tokens_per_slot * hidden_size * itemsize
                slot_stride = v.max_tokens_per_slot * v.hidden_size * itemsize
                remote_off = v.slot_idx * slot_stride
                # Read only num_tokens * hidden_size (actual data, not full slot)
                read_len = v.num_tokens * v.hidden_size * itemsize
                # Local offset in recv buffer
                local_off = token_offset * hidden_size * itemsize

                rdma_ops.append(
                    (
                        recv_mr,  # local MR
                        remote_mr,  # remote MR
                        remote_off,  # remote offset
                        local_off,  # local offset
                        read_len,  # bytes to read
                    )
                )
                token_offset += v.num_tokens

            # Execute batched RDMA reads
            if rdma_ops:
                endpoint.read(rdma_ops)
                logger.debug(
                    f"RDMA-fetched {len(rdma_ops)} vision slots from encoder "
                    f"{encoder_id} ({sum(v.num_tokens for v in slots)} tokens)"
                )

        # Store as vision embeds for _inject_vision_embeds
        # The recv_buf contains all vision tokens concatenated
        logger.info(
            f"[VISION_RDMA] Stored vision embeds: shape={recv_buf.shape}, dtype={recv_buf.dtype}, norm={recv_buf.norm().item():.4f}, nonzero={recv_buf.count_nonzero().item()}/{recv_buf.numel()}"
        )
        self._vision_embeds = {"image": recv_buf}

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
        # In hybrid mode, PeerAgent is started but KV/GDN MR is skipped.
        cache_context.start_peer_agent(mode=self.config.mode)

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
            nanoctrl_scope=config.nanoctrl_scope,
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
                "is_dummy=%s block_size=%s max_num_seqs=%s",
                is_dummy,
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

        # Chunked prefill: selective lm_head — only compute logits for final-chunk seqs.
        # sampling_token_indices: Q-tensor indices of the last token for each final-chunk seq.
        # sampling_seq_indices: which seq (0-based) each index corresponds to.
        # If all sequences are final chunks, set to None to use the fast default path.
        sampling_token_indices = None
        sampling_seq_indices = None
        num_seqs = aux.num_group_seqs
        if len(meta.sampling_token_indices) < num_seqs:
            sampling_token_indices = torch.tensor(
                meta.sampling_token_indices, dtype=torch.int64, pin_memory=True
            ).cuda(non_blocking=True)
            sampling_seq_indices = torch.tensor(
                meta.sampling_seq_indices, dtype=torch.int64, pin_memory=True
            ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            max_bs=self.config.max_num_seqs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q,
            max_seqlen_k=meta.max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            is_dummy=is_dummy,
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
            sampling_token_indices=sampling_token_indices,
            sampling_seq_indices=sampling_seq_indices,
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
                "prepare_decode_from_bytes failed: %s (block_size=%s max_num_seqs=%s)",
                str(e),
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

        if len(meta.block_tables_flat) == 0:
            block_tables = torch.empty((1, 0, 0), dtype=torch.int32).cuda(
                non_blocking=True
            )
        else:
            block_tables = (
                torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
                .reshape(sp_size, -1, meta.max_num_blocks)
                .cuda(non_blocking=True)
            )

        config = self.config
        hf_config = config.hf_config
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            context_lens_for_mla = context_lens[sp_rank, : aux.num_group_seqs]
            new_tile_scheduler_metadata, new_num_splits = flash_mla.get_mla_metadata(
                context_lens_for_mla.view(-1),
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
            is_dummy=is_dummy,
            tile_scheduler_metadata=new_tile_scheduler_metadata,
            num_splits=new_num_splits,
            gdn_conv_states=cache_ctx.gdn_conv_states,
            gdn_recurrent_states=cache_ctx.gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
        )

        return input_ids, positions

    def update_decode_inplace(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int
    ):
        """Update decode metadata in-place for multi-step decode (no Sequence needed)."""
        positions.add_(1)
        block_size = self.config.kvcache_block_size
        context = get_context()

        sp_rank = get_dist_context().attn_sp_rank

        # Update context length (now reflects the NEW token count)
        context.context_lens[sp_rank, :num_seqs].add_(1)

        # Recalculate slot_mapping from context_lens and block_tables.
        # Simply doing slot_mapping.add_(1) is WRONG when a sequence's new
        # token crosses a block boundary, because the page_id changes.
        new_ctx = context.context_lens[sp_rank, :num_seqs]  # already incremented
        block_idx = (new_ctx - 1) // block_size  # which block the new token falls in
        offset_in_block = (new_ctx - 1) % block_size  # offset within that block
        row_indices = torch.arange(num_seqs, device=block_idx.device)
        page_ids = context.block_tables[sp_rank, row_indices, block_idx.long()]
        context.slot_mapping[:num_seqs] = page_ids * block_size + offset_in_block

        return input_ids, positions

    def _prepare_lazy_verify_decode(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand decode from seqlen_q=1 to seqlen_q=2 for lazy verification.

        Transforms:
            input_ids:  [bs] (scheduler's token for this step, i.e. prev_sampled)
            positions:  [bs] (positions from prepare_decode_bytes)

        Into:
            input_ids:  [bs*2] interleaved [prev_sampled_0, draft_0, prev_sampled_1, draft_1, ...]
            positions:  [bs*2] interleaved [pos_0, pos_0+1, pos_1, pos_1+1, ...]

        Also updates context: slot_mapping expanded to [bs*2], context_lens += 1,
        num_tokens_per_seq = 2, MLA metadata recomputed.
        """
        sp_rank = get_dist_context().attn_sp_rank
        block_size = self.config.kvcache_block_size
        context = get_context()

        prev_sampled = self._prev_sampled_token  # [bs] token from prev step
        prev_draft = self._prev_drafts[0]  # [bs] draft from prev step

        # Build interleaved input_ids: [prev_sampled_0, draft_0, prev_sampled_1, ...]
        new_input_ids = torch.empty(
            num_seqs * 2, dtype=input_ids.dtype, device=input_ids.device
        )
        new_input_ids[0::2] = prev_sampled[:num_seqs]
        new_input_ids[1::2] = prev_draft[:num_seqs]

        # Build interleaved positions: [pos_0, pos_0+1, ...]
        new_positions = torch.empty(
            num_seqs * 2, dtype=positions.dtype, device=positions.device
        )
        new_positions[0::2] = positions[:num_seqs]
        new_positions[1::2] = positions[:num_seqs] + 1

        # Expand slot_mapping: 2 slots per seq
        # context_lens uses post-write semantic (already includes 1 slot for
        # the standard decode token).  In lazy verify we replace that single
        # token with 2 (prev_sampled + draft), so the first slot is at
        # old_ctx - 1 (same position standard decode would use).
        old_ctx = context.context_lens[sp_rank, :num_seqs]
        new_slot_mapping = torch.empty(
            num_seqs * 2, dtype=torch.int32, device=old_ctx.device
        )

        for offset in range(2):
            pos = old_ctx - 1 + offset
            blk_idx = (pos // block_size).long()
            off_in_block = (pos % block_size).int()
            row_idx = torch.arange(num_seqs, device=pos.device)
            max_blocks = context.block_tables.shape[2]
            blk_idx = torch.clamp(blk_idx, max=max_blocks - 1)
            page_ids = context.block_tables[sp_rank, row_idx, blk_idx]
            new_slot_mapping[offset::2] = (page_ids * block_size + off_in_block).int()

        # Update context_lens: old_ctx already accounted for 1 token, we add
        # 1 more for the draft → total 2 new KV entries in cache.
        context.context_lens[sp_rank, :num_seqs] += 1

        # Recompute MLA metadata for seqlen_q=2
        config = self.config
        hf_config = config.hf_config
        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            new_ctx = context.context_lens[sp_rank, :num_seqs]
            new_tile_sched, new_num_splits = flash_mla.get_mla_metadata(
                new_ctx.view(-1),
                2 * hf_config.num_attention_heads // mla_num_kv_heads,
                mla_num_kv_heads,
            )
        else:
            new_tile_sched, new_num_splits = (
                context.tile_scheduler_metadata,
                context.num_splits,
            )

        # Update context for seqlen_q=2
        context.slot_mapping = new_slot_mapping
        context.num_tokens_per_seq = 2
        if is_mla:
            context.tile_scheduler_metadata = new_tile_sched
            context.num_splits = new_num_splits

        return new_input_ids, new_positions

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
            inputs_embeds = None
            if is_prefill and self._vision_embeds is not None:
                logger.info(
                    f"[RUN_MODEL] Injecting vision embeds for prefill, input_ids.shape={input_ids.shape}, _vision_embeds keys={list(self._vision_embeds.keys())}"
                )
                inputs_embeds = self._inject_vision_embeds(input_ids)
                self._vision_embeds = None
            elif is_prefill and not context.is_dummy:
                logger.debug(
                    f"[RUN_MODEL] Prefill WITHOUT vision embeds, input_ids.shape={input_ids.shape}"
                )
            if inputs_embeds is not None:
                hidden = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
            else:
                hidden = self.model(input_ids, positions)
            if is_prefill:
                # Clean low-latency RDMA buffer here, as part of the prefill stream,
                # so the nvshmemx_barrier_all_block() inside completes before any rank
                # calls graph.replay().  Calling it right before graph.replay() races:
                # the barrier's RDMA writes arrive at peer GPUs while they are already
                # executing the graph, corrupting NVSHMEM symmetric memory → SIGSEGV.
                ExpertContext.get_instance().transition_to_low_latency()
            # Save hidden states for MTP draft generation (decode only)
            if not is_prefill and self.mtp_model is not None:
                self._last_hidden = hidden
            return self.model.compute_logits(hidden)
        else:
            n_tokens = input_ids.size(0)
            context = get_context()
            ntps = context.num_tokens_per_seq

            # Lazy verify path: use dedicated seqlen_q=2 CUDAGraph
            if (
                ntps == 2
                and hasattr(self, "lazy_verify_graphs")
                and self.lazy_verify_graphs is not None
            ):
                bs = n_tokens // 2
                master_bs = next(
                    (x for x in self.lazy_verify_graph_bs_list if x >= bs), None
                )
                if master_bs is not None and master_bs in self.lazy_verify_graphs:
                    gv = self.lazy_verify_graph_vars
                    n = bs * 2
                    gv["lv_input_ids"][:n] = input_ids
                    gv["lv_positions"][:n] = positions
                    gv["lv_slot_mapping"].fill_(-1)
                    gv["lv_slot_mapping"][:n] = context.slot_mapping
                    gv["lv_context_lens"].zero_()
                    gv["lv_context_lens"][:, : context.context_lens.shape[1]].copy_(
                        context.context_lens
                    )
                    gv["lv_block_tables"].zero_()
                    gv["lv_block_tables"][
                        :,
                        : context.block_tables.size(1),
                        : context.block_tables.size(2),
                    ] = context.block_tables

                    config = self.config
                    hf_config = config.hf_config
                    is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
                    if is_mla:
                        gv["lv_tile_sched"].zero_()
                        gv["lv_tile_sched"].copy_(context.tile_scheduler_metadata)
                        gv["lv_num_splits"].zero_()
                        gv["lv_num_splits"][: context.num_splits.shape[0]].copy_(
                            context.num_splits
                        )

                    if gv.get("lv_gdn_state_slots") is not None:
                        dummy_gdn_slot = (
                            get_cache_context().gdn_conv_states.shape[1] - 1
                        )
                        gv["lv_gdn_state_slots"].fill_(dummy_gdn_slot)
                        if context.gdn_state_slots is not None:
                            gv["lv_gdn_state_slots"][:bs].copy_(context.gdn_state_slots)

                    self.lazy_verify_graphs[master_bs].replay()
                    outputs = gv["lv_outputs"][:n]
                    if self.mtp_model is not None:
                        self._last_hidden = outputs.clone()
                    return self.model.compute_logits(outputs)

            # Normal decode (seqlen_q=1) CUDAGraph path
            bs = n_tokens
            master_bs = next(x for x in self.graph_master_rank_bs if x >= bs)

            # Without SP, attention_compute_bs == bs
            attn_bs = bs
            valid_attn_bs_list = self.graph_map.get(master_bs)
            if valid_attn_bs_list is None:
                raise RuntimeError(f"No graph map found for master_bs={master_bs}")

            try:
                attn_bs = next(x for x in valid_attn_bs_list if x >= attn_bs)
            except StopIteration:
                raise RuntimeError(
                    f"Input bs {bs} exceeds max captured attn_bs "
                    f"({valid_attn_bs_list[-1]}) for master_bs {master_bs}"
                )

            graph = self.graphs[(master_bs, attn_bs)]

            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping  # type: ignore
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:, : context.context_lens.shape[1]].copy_(context.context_lens)  # type: ignore
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][
                :, : context.block_tables.size(1), : context.block_tables.size(2)  # type: ignore
            ] = context.block_tables

            config = self.config
            hf_config = config.hf_config
            is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
            if is_mla:
                graph_vars["tile_scheduler_metadata"].zero_()
                graph_vars["num_splits"].zero_()
                graph_vars["tile_scheduler_metadata"].copy_(context.tile_scheduler_metadata)  # type: ignore
                graph_vars["num_splits"][: context.num_splits.shape[0]].copy_(context.num_splits)  # type: ignore

            if graph_vars.get("gdn_state_slots") is not None:
                dummy_gdn_slot = get_cache_context().gdn_conv_states.shape[1] - 1
                graph_vars["gdn_state_slots"].fill_(dummy_gdn_slot)
                if context.gdn_state_slots is not None:
                    graph_vars["gdn_state_slots"][:bs].copy_(context.gdn_state_slots)

            graph.replay()
            outputs = graph_vars["outputs"][:bs]
            # Save hidden states for MTP (need clone since graph_vars are reused)
            if self.mtp_model is not None:
                self._last_hidden = outputs.clone()
            return self.model.compute_logits(outputs)

    def migrate_from_bytes(self, data: bytes) -> None:
        """Migrate using lean MigrateBatchInput bytes (no Sequence objects)."""
        get_cache_context().migrate_from_bytes(data=data)

    @torch.inference_mode()
    def run_from_bytes(self, data: bytes, is_prefill: bool) -> list[list[int]]:
        """Run model from lean RunBatchInput bytes (completely Sequence-free)."""
        # Extract auxiliary data (temperatures, state_slots)
        sp_rank = get_dist_context().attn_sp_rank
        aux = extract_aux_from_bytes(data, sp_rank)
        num_seqs = aux.num_group_seqs

        is_dummy = False
        if num_seqs == 0:
            is_dummy = True
            # Create a minimal dummy RunBatchInput with one dummy sequence
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

        # Use original loop_count for decode iterations (config.loop_count may
        # be inflated to pre-allocate KV blocks for MTP speculative tokens).
        if is_prefill:
            loop_count = 1
            # Reset lazy verify state on prefill (new sequences have no drafts)
            self._prev_drafts = None
            self._prev_sampled_token = None
        elif hasattr(self.config, "_mtp_original_loop_count"):
            loop_count = self.config._mtp_original_loop_count
        else:
            loop_count = self.config.loop_count
        for i in range(loop_count):
            if self.profiler and self.run_count == self.profiler_start_step:
                self.profiler.start()
                logger.info(
                    f"Rank {self.rank}: Profiler started at step {self.run_count}"
                )

            if is_prefill:
                # RDMA-fetch vision embeddings from encoder (EP-separated mode)
                if i == 0 and self._vision_embeds is None:
                    vision_slots = extract_vision_slots_from_bytes(data)
                    if vision_slots:
                        self._fetch_vision_embeds_rdma(vision_slots)

                input_ids, positions = self.prepare_prefill_bytes(data, aux, is_dummy)
            else:
                if i == 0:
                    input_ids, positions = self.prepare_decode_bytes(
                        data, aux, is_dummy
                    )
                else:
                    input_ids, positions = self.update_decode_inplace(
                        input_ids, positions, num_seqs
                    )

                # Lazy verify: if we have drafts from the previous step,
                # expand decode to seqlen_q=2 with [prev_sampled, prev_draft]
                has_lazy_verify = (
                    self.mtp_model is not None
                    and self._prev_drafts is not None
                    and not is_dummy
                )
                if has_lazy_verify:
                    input_ids, positions = self._prepare_lazy_verify_decode(
                        input_ids, positions, num_seqs
                    )

            if input_ids.numel() == 0:
                logger.critical(
                    "EMPTY input_ids before run_model! rank=%s is_prefill=%s "
                    "is_dummy=%s input_ids.shape=%s positions.shape=%s "
                    "num_seqs=%s",
                    self.rank,
                    is_prefill,
                    is_dummy,
                    input_ids.shape,
                    positions.shape,
                    num_seqs,
                )
            logits = self.run_model(input_ids, positions, is_prefill)

            # Clear RDMA-fetched vision embeddings after prefill forward
            if is_prefill and self._vision_embeds is not None:
                self._vision_embeds = None

            tp_rank = get_dist_context().attn_tp_rank
            temperatures = None

            # Lazy verify: logits has shape [bs*2, vocab]. Do verification +
            # sampling together, then output accepted tokens.
            if not is_prefill and has_lazy_verify:
                # logits[0::2] = verify logits (after token_{K-1}, predicts next)
                # logits[1::2] = bonus logits (after draft, predicts bonus token)
                verify_logits = logits[0::2]  # [bs, vocab]
                bonus_logits = logits[1::2]  # [bs, vocab]

                verified_tokens = torch.zeros(
                    1, num_seqs, dtype=torch.int64, device="cuda"
                )
                num_accepted = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")

                if tp_rank == 0:
                    temperatures = self.prepare_sample_from_aux(aux)
                    target_pred = self.sampler(verify_logits, temperatures)
                    bonus_pred = self.sampler(bonus_logits, temperatures)
                    prev_draft_0 = self._prev_drafts[0]

                    accepted_mask = target_pred == prev_draft_0
                    # accepted: output draft + bonus, input_ids = bonus
                    # rejected: output target_pred, input_ids = target_pred
                    verified_tokens[0] = prev_draft_0
                    num_accepted = accepted_mask.long()
                    # The "new token" for this step:
                    # accept → sample(bonus_logits), reject → sample(verify_logits)
                    input_ids = torch.where(accepted_mask, bonus_pred, target_pred)
                else:
                    input_ids = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")

                dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)
                dist.all_reduce(verified_tokens, group=get_dist_context().attn_tp_group)
                dist.all_reduce(num_accepted, group=get_dist_context().attn_tp_group)

                self._mtp_verified_tokens = verified_tokens
                self._mtp_num_accepted = num_accepted

                # Update context_lens: +1 for token_{K-1}, +1 more if accepted
                context = get_context()
                sp_rank = get_dist_context().attn_sp_rank
                # context_lens was set for seqlen_q=2 (already includes both slots)
                # For rejected seqs: roll back context_lens by 1
                rejected_mask = num_accepted == 0
                if rejected_mask.any():
                    context.context_lens[sp_rank, :num_seqs] -= rejected_mask.int()

                    # Rollback GDN states for rejected sequences.
                    # During lazy verify forward, each GDN layer saved the
                    # intermediate state (after token_0 only) into backup slots.
                    # For rejected seqs, copy backup → active to discard the
                    # effect of the draft token.
                    # NOTE: In attention_dp mode, only the DP rank that owns a
                    # sequence has a real state slot (0..max_active-1). Other
                    # ranks carry the dummy slot (= pool_size-1) and must be
                    # excluded from rollback to avoid out-of-bounds indexing.
                    _cache_ctx = get_cache_context()
                    if _cache_ctx.gdn_conv_states is not None:
                        active_slots = context.gdn_state_slots[:num_seqs]
                        # Combine rejected mask with real-slot mask
                        real_slot_mask = active_slots < _cache_ctx.gdn_max_active_slots
                        rollback_mask = rejected_mask & real_slot_mask
                        if rollback_mask.any():
                            backup_offset = _cache_ctx.gdn_max_active_slots
                            rej_active = active_slots[rollback_mask]
                            rej_backup = rej_active + backup_offset
                            _cache_ctx.gdn_conv_states[:, rej_active] = (
                                _cache_ctx.gdn_conv_states[:, rej_backup]
                            )
                            _cache_ctx.gdn_recurrent_states[:, rej_active] = (
                                _cache_ctx.gdn_recurrent_states[:, rej_backup]
                            )
            else:
                if tp_rank == 0:
                    temperatures = self.prepare_sample_from_aux(aux)
                    context = get_context()
                    if is_prefill and context.sampling_seq_indices is not None:
                        # Sparse prefill: logits has shape [n_final, vocab_size].
                        temps_filtered = temperatures[context.sampling_seq_indices]
                        sampled = self.sampler(logits, temps_filtered)
                        input_ids = input_ids.new_zeros(num_seqs)
                        input_ids[context.sampling_seq_indices] = sampled
                    else:
                        input_ids = self.sampler(logits, temperatures)
                else:
                    input_ids = input_ids.new_zeros([num_seqs])
                dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)

            # MTP lazy verify: generate drafts for the NEXT step
            if self.mtp_model is not None and not is_prefill and not is_dummy:
                decode_context = get_context()
                saved_token_ids = decode_context.token_ids

                # Generate N draft tokens from MTP using current hidden states
                # For lazy verify: use hidden from the correct position
                if has_lazy_verify:
                    # _last_hidden has shape [bs*2, hidden_size]
                    # For accepted seqs: use hidden[1::2] (after draft pos)
                    # For rejected seqs: use hidden[0::2] (after token_{K-1} pos)
                    h_verify = self._last_hidden[0::2]  # [bs, hidden_size]
                    h_bonus = self._last_hidden[1::2]  # [bs, hidden_size]
                    accepted_mask = num_accepted > 0
                    mtp_hidden = torch.where(
                        accepted_mask.unsqueeze(-1), h_bonus, h_verify
                    )
                    # positions for MTP: depends on accept/reject
                    # accepted: pos of bonus token = original_pos + 2
                    # rejected: pos of verify token = original_pos + 1
                    mtp_positions = positions[0::2] + 1 + num_accepted
                else:
                    mtp_hidden = self._last_hidden
                    mtp_positions = positions

                drafts = self._generate_mtp_drafts(
                    input_ids,
                    mtp_positions,
                    mtp_hidden,
                    temperatures,
                    num_seqs,
                )

                # Store drafts + sampled token for lazy verification in the NEXT step
                self._prev_drafts = drafts
                self._prev_sampled_token = input_ids.clone()

                # Restore decode context (MTP draft gen overwrites it)
                set_context(
                    is_prefill=decode_context.is_prefill,
                    max_bs=decode_context.max_bs,
                    slot_mapping=decode_context.slot_mapping,
                    context_lens=decode_context.context_lens,
                    block_tables=decode_context.block_tables,
                    is_dummy=decode_context.is_dummy,
                    tile_scheduler_metadata=decode_context.tile_scheduler_metadata,
                    num_splits=decode_context.num_splits,
                    gdn_conv_states=decode_context.gdn_conv_states,
                    gdn_recurrent_states=decode_context.gdn_recurrent_states,
                    gdn_state_slots=decode_context.gdn_state_slots,
                )
                get_context().token_ids = saved_token_ids

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

        # Build output: for lazy verify, verified draft tokens come BEFORE
        # the newly sampled token. base has [new_token] per seq per step.
        if (
            self.mtp_model is not None
            and hasattr(self, "_mtp_verified_tokens")
            and self._mtp_verified_tokens is not None
        ):
            base = torch.cat(get_context().token_ids, dim=0)  # [loop_count, num_seqs]
            verified = self._mtp_verified_tokens  # [N, num_seqs]
            accepted = self._mtp_num_accepted  # [num_seqs]
            loop_count_token_ids = []
            for s in range(base.shape[1]):
                n_acc = int(accepted[s].item())
                seq_tokens = []
                # Insert accepted draft tokens BEFORE the new token
                for j in range(n_acc):
                    seq_tokens.append(int(verified[j, s].item()))
                # Then the newly sampled token(s) from normal decode
                for t in range(base.shape[0]):
                    seq_tokens.append(int(base[t, s].item()))
                loop_count_token_ids.append(seq_tokens)
            if self.rank == 0:
                logger.debug(f"OUTPUT tokens[0]: {loop_count_token_ids[0]}")
            self._mtp_verified_tokens = None
            self._mtp_num_accepted = None
        else:
            loop_count_token_ids = torch.cat(get_context().token_ids, dim=0).T.tolist()

        reset_context()
        return loop_count_token_ids

    # ------------------------------------------------------------------
    # MTP (Multi-Token Prediction) draft generation
    # ------------------------------------------------------------------

    def _set_mtp_context(self, num_seqs: int):
        """Set context for MTP forward (prefill mode, seq_len=1, no KV cache).

        Uses low-latency EP for CUDAGraph compatibility.
        """
        cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
        set_context(
            is_prefill=True,
            max_bs=self.config.max_num_seqs,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=1,
            max_seqlen_k=1,
            slot_mapping=None,
            block_tables=None,
            is_dummy=False,
            use_low_latency_ep=True,
        )

    def _run_mtp_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int,
        bs: int,
    ) -> torch.Tensor:
        """Run one MTP speculative step. Uses CUDAGraph if captured."""
        if self.mtp_graphs is not None:
            master_bs = next((x for x in self.mtp_graph_bs_list if x >= bs), None)
            if master_bs is not None and master_bs in self.mtp_graphs:
                gv = self.mtp_graph_vars
                gv["mtp_input_ids"][:bs] = input_ids
                gv["mtp_positions"][:bs] = positions
                gv["mtp_hidden_states"][:bs] = hidden_states
                self.mtp_graphs[master_bs].replay()
                return gv["mtp_outputs"][:bs]

        # Eager fallback
        self._set_mtp_context(bs)
        return self.mtp_model(
            input_ids,
            positions,
            hidden_states,
            spec_step_idx=spec_step_idx,
        )

    @torch.inference_mode()
    def _generate_mtp_drafts(
        self,
        sampled_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        temperatures: torch.Tensor | None,
        num_seqs: int,
    ) -> list[torch.Tensor]:
        """Generate draft tokens using MTP layers (decode only)."""
        drafts = []
        current_hidden = hidden_states
        current_ids = sampled_ids
        current_pos = positions + 1

        for step in range(self.config.num_speculative_tokens):
            # _run_mtp_step handles context setup for eager path;
            # for CUDAGraph path, the context was baked in during capture.
            mtp_hidden = self._run_mtp_step(
                current_ids, current_pos, current_hidden, step, num_seqs
            )

            # Set MTP context for compute_logits (ParallelLMHead reads is_prefill)
            self._set_mtp_context(num_seqs)
            mtp_logits = self.mtp_model.compute_logits(mtp_hidden, spec_step_idx=step)

            # Sample on tp_rank 0, broadcast
            tp_rank = get_dist_context().attn_tp_rank
            if tp_rank == 0:
                draft_ids = self.sampler(mtp_logits, temperatures)
            else:
                draft_ids = current_ids.new_zeros(num_seqs)
            dist.all_reduce(draft_ids, group=get_dist_context().attn_tp_group)

            drafts.append(draft_ids)
            current_hidden = mtp_hidden
            current_ids = draft_ids
            current_pos = current_pos + 1

        return drafts

    def _capture_mtp_cudagraphs(self, max_bs: int, hf_config):
        """Capture separate CUDAGraphs for MTP forward per batch size.

        All context tensors (cu_seqlens) are persisted as instance attrs
        so they stay alive during graph replay.
        """
        mtp_input_ids = torch.zeros(max_bs, dtype=torch.int64)
        mtp_positions = torch.zeros(max_bs, dtype=torch.int64)
        mtp_hidden_states = torch.zeros(max_bs, hf_config.hidden_size)
        mtp_outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # Persistent cu_seqlens — must outlive graph lifetime
        cu_seqlens = torch.arange(max_bs + 1, dtype=torch.int32, device="cuda")

        # Fewer graph sizes to reduce memory pressure during capture
        self.mtp_graph_bs_list = [1, 2, 4, 8]
        self.mtp_graphs = {}

        logger.info(f"Capturing MTP CUDAGraphs...")

        # Must transition to low-latency before MTP forward
        ExpertContext.get_instance().transition_to_low_latency()

        # Persist per-bs cu_seqlens so they survive graph lifetime
        self._mtp_cu_seqlens_per_bs = {}

        for bs in reversed(self.mtp_graph_bs_list):
            logger.info(f"Capturing MTP graph - bs={bs}")

            # Each graph gets its OWN cu_seqlens matching its batch size
            bs_cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
            self._mtp_cu_seqlens_per_bs[bs] = bs_cu_seqlens

            set_context(
                is_prefill=True,
                max_bs=self.config.max_num_seqs,
                cu_seqlens_q=bs_cu_seqlens,
                cu_seqlens_k=bs_cu_seqlens,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=None,
                block_tables=None,
                is_dummy=False,
                use_low_latency_ep=True,
            )

            # Warmup
            mtp_outputs[:bs] = self.mtp_model(
                mtp_input_ids[:bs], mtp_positions[:bs], mtp_hidden_states[:bs]
            )

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                mtp_outputs[:bs] = self.mtp_model(
                    mtp_input_ids[:bs],
                    mtp_positions[:bs],
                    mtp_hidden_states[:bs],
                )

            self.mtp_graphs[bs] = graph

        reset_context()

        self.mtp_graph_vars = dict(
            mtp_input_ids=mtp_input_ids,
            mtp_positions=mtp_positions,
            mtp_hidden_states=mtp_hidden_states,
            mtp_outputs=mtp_outputs,
        )
        # Keep cu_seqlens alive for graph replay
        self._mtp_cu_seqlens = cu_seqlens

        logger.info(f"Finished capturing {len(self.mtp_graphs)} MTP CUDAGraphs")

    def _capture_lazy_verify_cudagraphs(self, max_bs: int, hf_config):
        """Capture CUDAGraphs for lazy verify decode (seqlen_q=2 per seq).

        Input buffer sizes are max_bs*2 for input_ids, positions, slot_mapping,
        outputs. The attention layer uses num_tokens_per_seq=2 with causal=True.
        """
        block_size = get_cache_context().block_size
        max_num_blocks = (self.config.max_model_len + block_size - 1) // block_size
        _cache_ctx = get_cache_context()

        # Buffers sized for max_bs sequences * 2 tokens each
        lv_input_ids = torch.zeros(max_bs * 2, dtype=torch.int64)
        lv_positions = torch.zeros(max_bs * 2, dtype=torch.int64)
        lv_slot_mapping = torch.full((max_bs * 2,), -1, dtype=torch.int32)
        lv_context_lens = torch.zeros(1, max_bs, dtype=torch.int32)
        lv_block_tables = torch.zeros(1, max_bs, max_num_blocks, dtype=torch.int32)
        lv_outputs = torch.zeros(max_bs * 2, hf_config.hidden_size)

        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            # For seqlen_q=2: num_q_heads_per_kv = 2 * num_attention_heads // kv_heads
            lv_tile_sched, lv_num_splits = flash_mla.get_mla_metadata(
                torch.ones(max_bs, dtype=torch.int32, device="cuda"),
                2 * hf_config.num_attention_heads // mla_num_kv_heads,
                mla_num_kv_heads,
            )
        else:
            lv_tile_sched, lv_num_splits = None, None

        # GDN state slots (Qwen3.5)
        lv_gdn_state_slots = None
        if _cache_ctx.gdn_conv_states is not None:
            dummy_gdn_slot = _cache_ctx.gdn_conv_states.shape[1] - 1
            lv_gdn_state_slots = torch.full(
                (max_bs,), dummy_gdn_slot, dtype=torch.int64
            )

        self.lazy_verify_graph_bs_list = [1, 2, 4, 8]
        self.lazy_verify_graphs = {}

        logger.info("Capturing lazy verify CUDAGraphs (seqlen_q=2)...")

        for bs in reversed(self.lazy_verify_graph_bs_list):
            n_tokens = bs * 2
            logger.info(f"Capturing lazy verify graph - bs={bs} (n_tokens={n_tokens})")
            set_context(
                is_prefill=False,
                max_bs=self.config.max_num_seqs,
                slot_mapping=lv_slot_mapping[:n_tokens],
                context_lens=lv_context_lens,
                block_tables=lv_block_tables,
                is_dummy=False,
                tile_scheduler_metadata=lv_tile_sched,
                num_splits=lv_num_splits,
                num_tokens_per_seq=2,
                gdn_conv_states=_cache_ctx.gdn_conv_states,
                gdn_recurrent_states=_cache_ctx.gdn_recurrent_states,
                gdn_state_slots=(
                    lv_gdn_state_slots[:bs] if lv_gdn_state_slots is not None else None
                ),
            )

            # Warmup
            lv_outputs[:n_tokens] = self.model(
                lv_input_ids[:n_tokens], lv_positions[:n_tokens]
            )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                lv_outputs[:n_tokens] = self.model(
                    lv_input_ids[:n_tokens], lv_positions[:n_tokens]
                )

            self.lazy_verify_graphs[bs] = graph
            reset_context()

        self.lazy_verify_graph_vars = dict(
            lv_input_ids=lv_input_ids,
            lv_positions=lv_positions,
            lv_slot_mapping=lv_slot_mapping,
            lv_context_lens=lv_context_lens,
            lv_block_tables=lv_block_tables,
            lv_outputs=lv_outputs,
            lv_tile_sched=lv_tile_sched,
            lv_num_splits=lv_num_splits,
            lv_gdn_state_slots=lv_gdn_state_slots,
        )

        logger.info(
            f"Finished capturing {len(self.lazy_verify_graphs)} lazy verify CUDAGraphs"
        )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        max_bs = min(self.config.max_num_seqs, 512)
        block_size = get_cache_context().block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(1, max_bs, dtype=torch.int32)
        block_tables = torch.zeros(1, max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        is_mla = getattr(hf_config, "kv_lora_rank", 0) > 0
        if is_mla:
            mla_num_kv_heads = 1
            tile_scheduler_metadata_buffer, num_splits_buffer = (
                flash_mla.get_mla_metadata(
                    torch.ones(max_bs, dtype=torch.int32, device="cuda"),
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

        self.attn_bs_step = 16

        logger.info(f"Capturing CUDAGraphs...")
        completed_graphs = 0

        for master_bs in reversed(self.graph_master_rank_bs):
            # Without SP, attn_bs == master_bs (no recv seqs from other SP ranks)
            attn_bs = master_bs
            self.graph_map[master_bs] = [attn_bs]

            completed_graphs += 1
            logger.info(f"Capturing graph - (master_bs={master_bs}, attn_bs={attn_bs})")
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                max_bs=self.config.max_num_seqs,
                slot_mapping=slot_mapping[:master_bs],
                context_lens=context_lens,
                block_tables=block_tables,
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

            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            reset_context()

        logger.info(f"Finished capturing {len(self.graphs)} CUDAGraphs")

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            tile_scheduler_metadata=tile_scheduler_metadata_buffer,
            num_splits=num_splits_buffer,
            gdn_state_slots=gdn_state_slots_buf,
        )

        # MTP lazy verify CUDAGraph support:
        # Both non-GDN (DeepSeek) and GDN (Qwen3.5) models use CUDAGraph for
        # MTP drafts and lazy verify.  GDN lazy verify uses two sequential
        # decode passes per layer (fixed-shape, CUDAGraph-friendly) with
        # intermediate state saved to backup slots for rollback.
        if self.mtp_model is not None:
            self._capture_mtp_cudagraphs(max_bs, hf_config)
            self._capture_lazy_verify_cudagraphs(max_bs, hf_config)

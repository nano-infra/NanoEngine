import os

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.profiler as profiler

# import flash_mla


from nanodeploy._cpp import (
    BlockContextSlot,
    prepare_decode_cpp,
    prepare_prefill_cpp,
    update_seqs_inner_loop,
)
from nanodeploy.config import Config
from nanodeploy.endpoint.rpc_endpoint import RPCClientEndpoint
from nanodeploy.engine.sequence import Sequence
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger, set_log_level

# from nanodeploy.models.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.models.qwen3 import Qwen3ForCausalLM
from nanodeploy.models.qwen3_moe import Qwen3MoeForCausalLM
from nanodeploy.worker.cache import get_cache_context, set_cache_context
from nanodeploy.worker.context import get_context, reset_context, set_context
from nanodeploy.worker.distributed import (
    get_dist_context,
    get_local_ip,
    set_dist_context,
)
from nanodeploy.worker.loader import load_model
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config
from nanodeploy.worker.sp_context import set_sp_context

logger = get_logger()


architectures = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    # "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(self, config: Config, rank: int):
        # Set log level
        if config.log_level:
            set_log_level(config.log_level)

        self.config = config
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.attn_world_size
        self.rank = rank

        self.rank = rank

        # Helper to setup CPATH for DeepGEMM/CUTLASS in Ray worker
        # Attempt to locate DeepGEMM relative to NanoDeploy location or CWD
        import os

        # Try to find DeepGEMM in common locations
        candidates = [
            os.path.abspath(os.path.join(os.getcwd(), "../DeepGEMM")),
            "/mnt/nvme1n1/ml_research/majinming/src/DeepGEMM",
        ]

        cutlass_include = None
        for path in candidates:
            candidate_include = os.path.join(path, "third-party/cutlass/include")
            if os.path.exists(candidate_include):
                cutlass_include = candidate_include
                break

        if cutlass_include:
            current_cpath = os.environ.get("CPATH", "")
            if cutlass_include not in current_cpath:
                logger.info(f"Adding CUTLASS include path to CPATH: {cutlass_include}")
                os.environ["CPATH"] = (
                    f"{cutlass_include}:{current_cpath}"
                    if current_cpath
                    else cutlass_include
                )
        else:
            logger.warning(
                "DeepGEMM/CUTLASS include path not found. DeepGEMM JIT might fail."
            )

        logger.debug(f"init ModelRunner, {rank=}, {get_local_ip()=}")

        set_runner_config(
            max_num_seqs=config.max_num_seqs,
            dummy_weight=config.dummy_weight,
            perfect_eplb=config.perfect_eplb,
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
            import deep_ep

            deep_ep.Buffer.num_sms = 16
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
                    worker_name=f"{self.engine_id}_rank_{self.rank}",
                    use_gzip=False,
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            logger.info(
                f"Rank {rank}: Profiler enabled. Start at {self.profiler_start_step}, duration {self.profiler_steps} steps."
            )

        model_architecture = hf_config.architectures[0]
        self.model = architectures[model_architecture](hf_config)

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        dist.barrier()

        self.sampler = Sampler()
        # self.warmup_model()
        self.preallocate_kvcache()

        self.endpoint = RPCClientEndpoint(32_000_000, get_dist_context().rank)

    def init_rpc_endpoint(self, server_info):
        client_info = self.endpoint.init_client_endpoint()
        self.endpoint.connect(server_info)
        logger.info("client endpoint initialized")
        return client_info

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
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(self.default_dtype)

    def p2p_init(self, remote_engine_id, num_kv_blocks, remote_engine_world_size):
        return get_cache_context().p2p_init(
            remote_engine_id, num_kv_blocks, remote_engine_world_size
        )

    def p2p_connect(
        self, remote_engine_id: str, endpoints_info_list: list[dict[int, dict]]
    ):
        return get_cache_context().p2p_connect(remote_engine_id, endpoints_info_list)

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
        seqs = []
        for _ in range(num_seqs):
            seq = Sequence(
                list(np.random.randint(low=0, high=10000, size=max_model_len))
            )
            # seq = Sequence(
            #     list(np.zeros(max_model_len, dtype=int))
            # )
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

        logger.info(f"ModelRunner block_tables.shape: {block_tables.shape}")

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
        if hf_config.num_key_value_heads == 1:
            # new_tile_scheduler_metadata, new_num_splits = flash_mla.get_mla_metadata(
            #     context_lens_for_attn.view(-1),
            #     hf_config.num_attention_heads // hf_config.num_key_value_heads,
            #     hf_config.num_key_value_heads,
            # )
            new_tile_scheduler_metadata, new_num_splits = None, None
        else:
            new_tile_scheduler_metadata, new_num_splits = None, None

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

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        # [DEBUG] Log input_ids for verification (len > 10)
        mask = positions > 9
        if mask.any():
            subset_ids = input_ids[mask]
            subset_pos = positions[mask]
            # print(
            #     f"[DEBUG] Python Input IDs (len > 10): IDs={subset_ids[:20].tolist()}, Pos={subset_pos[:20].tolist()}"
            # )

        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            context = get_context()
            return self.model.compute_logits(self.model(input_ids, positions))
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
            if hf_config.num_key_value_heads == 1:
                graph_vars["tile_scheduler_metadata"].zero_()
                graph_vars["num_splits"].zero_()
                # graph_vars["tile_scheduler_metadata"].copy_(context.tile_scheduler_metadata)  # type: ignore
                # graph_vars["num_splits"][:context.num_splits.shape[0]].copy_(context.num_splits)  # type: ignore

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
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def migrate(self, seqs: list[Sequence]) -> None:
        get_cache_context().migrate(seqs=seqs)

    def run(
        self, dp_seqs: list[Sequence], is_prefill: bool, enable_rpc: bool = False
    ) -> list[list[int]]:

        if enable_rpc:
            dp_seqs = self.endpoint.recv_seqs()
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size

        num_sp_seqs = sum(
            1
            for seq in dp_seqs
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank
        )
        is_dummy = False
        if num_sp_seqs == 0:
            is_dummy = True
            # seq = Sequence([np.random.randint(self.config.hf_config.vocab_size - 1)])
            seq = Sequence([0])  # Zero init for determinism
            seq.block_ctx().reset(self.engine_id, sp_size, 1)
            seq.block_ctx().master_sp_idx = sp_rank
            dp_seqs.append(seq)

        sp_seqs = [
            seq
            for seq in dp_seqs
            if seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == sp_rank
        ]

        loop_count = self.config.loop_count if not is_prefill else 1
        for i in range(loop_count):
            if self.profiler and self.run_count == self.profiler_start_step:
                self.profiler.start()
                logger.info(
                    f"Rank {self.rank}: Profiler started at step {self.run_count}"
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

            logits = self.run_model(input_ids, positions, is_prefill)

            tp_rank = get_dist_context().attn_tp_rank
            if tp_rank == 0:
                temperatures = (
                    self.prepare_sample(dp_seqs)
                    if tp_rank == 0
                    else [None] * len(sp_seqs)
                )

                # Logging Logits
                if len(dp_seqs) > 0 and len(dp_seqs[0].token_ids) > 0:
                    import logging

                    if logger.isEnabledFor(logging.INFO):
                        # Log first sequence's logits stats
                        log_logits = logits[0]
                        logger.info(
                            f"Python Logits: [{log_logits[:10].tolist()}...], Max: {log_logits.max().item()}, Max Index: {log_logits.argmax().item()}, Sum: {log_logits.sum().item()}"
                        )

                input_ids = self.sampler(logits, temperatures)
            else:
                input_ids = torch.zeros_like(input_ids)
            dist.all_reduce(input_ids, group=get_dist_context().attn_tp_group)

            update_seqs_inner_loop(sp_seqs, sp_rank)

            if self.profiler and self.run_count >= self.profiler_start_step:
                if self.run_count < self.profiler_end_step:
                    self.profiler.step()

                if self.run_count == self.profiler_end_step - 1:
                    self.profiler.stop()
                    logger.info(
                        f"Rank {self.rank}: Profiler stopped and saved at step {self.run_count}"
                    )

            self.run_count += 1  # 每次调用计数+1
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

        if hf_config.num_key_value_heads == 1:
            tile_scheduler_metadata_buffer, num_splits_buffer = (
                flash_mla.get_mla_metadata(
                    torch.ones(
                        max_attention_comp_seqs, dtype=torch.int32, device="cuda"
                    ),
                    hf_config.num_attention_heads // hf_config.num_key_value_heads,
                    hf_config.num_key_value_heads,
                )
            )
        else:
            tile_scheduler_metadata_buffer, num_splits_buffer = None, None

        self.graph_master_rank_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
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
        )

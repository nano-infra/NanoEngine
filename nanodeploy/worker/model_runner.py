import os
import time

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.profiler as profiler
import flash_mla


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
from nanodeploy.worker.loader import load_model
from nanodeploy.worker.runner_config import get_runner_config, set_runner_config
from nanodeploy.worker.sp_context import set_sp_context

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
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.cuda_graph_mode = config.cuda_graph_mode
        self.world_size = config.attn_world_size
        self.rank = rank
        self.log_decode_a2a_masks = _env_flag_enabled(
            "NANODEPLOY_LOG_DECODE_A2A_MASKS", default=False
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

        self.endpoint = RPCClientEndpoint(32*32_000_000, get_dist_context().rank)

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
            if self.cuda_graph_mode == "piecewise":
                del self.piecewise_graphs, self.piecewise_graph_vars, self.graph_pool
            else:
                del self.local_graphs, self.sp_graphs, self.sp_graph_map, self.graph_pool
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
        attention_compute_bs = (
            context_lens_for_attn.numel() if use_sp_a2a else input_ids.size(0)
        )
        
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

    def _log_decode_a2a_masks(self, loop_idx: int, is_dummy: bool) -> None:
        context = get_context()
        if context.use_sp_a2a is not True:
            return
        if context.q_mask is None or context.res_lse_mask is None or context.q_offsets is None:
            return

        dist_context = get_dist_context()
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
                "q_offsets": context.q_offsets.detach().cpu().tolist(),
                "q_mask": context.q_mask.detach().cpu().tolist(),
                "res_lse_mask": context.res_lse_mask.detach().cpu().tolist(),
            }
        )

    def _select_decode_graph_master_bs(self, bs: int, context) -> int:
        master_bs = next(x for x in self.graph_master_rank_bs if x >= bs)
        if (
            context.use_sp_a2a
            and (self.config.sp_backend == "nccl" or self.config.fixed_sp_size > 0)
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
        if hf_config.num_key_value_heads == 1 and graph_vars.get("tile_scheduler_metadata") is not None:
            graph_vars["tile_scheduler_metadata"].zero_()
            graph_vars["num_splits"].zero_()

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
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))

        if self.cuda_graph_mode == "piecewise":
            return self.run_model_piecewise_cudagraph(input_ids, positions)

        bs = input_ids.size(0)
        context = get_context()
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
            graph = self.local_graphs[master_bs]

        graph_vars = self.graph_vars
        self._copy_decode_context_to_graph_vars(
            graph_vars, input_ids, positions, bs, context
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

        self._copy_decode_context_to_graph_vars(
            graph_vars, input_ids, positions, bs, context
        )

        temp_context_fields = {
            "slot_mapping": graph_vars["slot_mapping"][:master_bs],
            "context_lens": graph_vars["context_lens"],
            "block_tables": graph_vars["block_tables"],
            "global_context_lens": graph_vars["global_context_lens"],
            "q_mask": graph_vars["q_mask"],
            "res_lse_mask": graph_vars["res_lse_mask"],
            "tile_scheduler_metadata": graph_vars["tile_scheduler_metadata"],
            "num_splits": graph_vars["num_splits"],
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
            "attention_compute_bs": context.attention_compute_bs,
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

    def migrate_decode_kv_blocks(self, migration_plans: list[dict]) -> None:
        get_cache_context().migrate_decode_kv_blocks(migration_plans)

    def run(
        self, dp_seqs: list[Sequence], is_prefill: bool, enable_rpc: bool = False, send_timestamp: float = 0.0
    ) -> list[list[int]]:

        if enable_rpc:
            dp_seqs = self.endpoint.recv_seqs()

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
        is_dummy = False
        if num_sp_seqs == 0:
            is_dummy = True
            seq = Sequence([np.random.randint(self.config.hf_config.vocab_size - 1)])
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

        loop_count_token_ids = torch.cat(get_context().token_ids, dim=0).T.tolist()
        reset_context()
        worker_end_time = time.time()

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

        self.graph_master_rank_bs = self._build_graph_master_rank_bs(max_bs)
        self.local_graphs = {}
        self.sp_graphs = {}
        self.graph_pool = None
        self.sp_graph_map = {}  # store master_bs -> [available_attn_bs...]
        self.attn_bs_step = 16

        def capture_graph(master_bs: int, attn_bs: int, use_sp_a2a: bool):
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
        max_attention_comp_seqs = max_bs + config.max_num_recv_seqs
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
                    (config.max_num_recv_seqs,), -1, dtype=torch.int32
                ),
                res_slice_fill_to_buffer_input=torch.full(
                    (config.max_num_recv_seqs,), -1, dtype=torch.int32
                ),
                res_to_buffer_input_mask=torch.zeros(
                    config.max_num_recv_seqs, dtype=torch.int32
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

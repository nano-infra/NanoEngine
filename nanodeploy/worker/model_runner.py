import os
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
from nanodeploy.kernels.copy import warmup_copy_kernel
from nanodeploy.engine.sequence import Sequence
from nanodeploy.layers.sampler import Sampler
from nanodeploy.logging import get_logger
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
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class ModelRunner:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.engine_id = self.config.engine_id
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.attn_world_size
        self.rank = rank
        self.graph_master_rank_bs = []
        self.graph_attn_compute_bs = []

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
        import time

        time.sleep(10)

        if sp_size > 1:
            sp_rank = get_dist_context().attn_sp_rank
            set_sp_context(
                config.max_num_seqs,
                hf_config.head_dim,
                hf_config.num_attention_heads,
                torch.get_default_dtype(),
                sp_size,
                sp_rank,
            )

        model_architecture = hf_config.architectures[0]
        self.model = architectures[model_architecture](hf_config)

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

        sp_size = get_dist_context().attn_sp_world_size
        ep_size = get_dist_context().ffn_ep_world_size

        if ep_size > 1:
            import deep_ep

            deep_ep.Buffer.num_sms = 16
            dist.barrier(group=get_dist_context().cuda_world_group)

        if sp_size > 1:
            sp_rank = get_dist_context().attn_sp_rank
            set_sp_context(
                config.max_num_seqs,
                hf_config.head_dim,
                hf_config.num_attention_heads,
                torch.get_default_dtype(),
                sp_size,
                sp_rank,
            )

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        dist.barrier()

        logger.info("Warming up copy kernels...")
        warmup_copy_kernel()
        torch.cuda.synchronize()
        logger.info("Finish warm up copy kernels...")

        self.sampler = Sampler()
        self.warmup_model()
        self.preallocate_kvcache()

    def num_kvcache_blocks(self):
        return self.config.num_kvcache_blocks

    def allocate_kvcache(self, num_kvcache_blocks: int):
        self.config.num_kvcache_blocks = num_kvcache_blocks
        cache_context = get_cache_context()
        cache_context.allocate_kvcache(num_kvcache_blocks)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = cache_context.kv_cache[0, layer_id]
                module.v_cache = cache_context.kv_cache[1, layer_id]
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
            seq.active(self.engine_id, sp_size, 1)
            seq.block_ctx().master_sp_idx = sp_rank

        self.run(seqs, True)
        torch.cuda.empty_cache()

    def preallocate_kvcache(self):
        config = self.config
        hf_config = config.hf_config

        cache_context = set_cache_context(
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            block_size=config.kvcache_block_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            attention_tp=config.attention_tp,
            gpu_memory_utilization=config.gpu_memory_utilization,
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
            mode="gqa",
        )
        config.num_kvcache_blocks = cache_context.num_local_kvcache_blocks

    def prepare_block_tables(self, dp_seqs: list[Sequence]):
        sp_size = get_dist_context().attn_sp_world_size
        sp_rank = get_dist_context().attn_sp_rank

        valid_block_tables = []

        for sp_idx in range(sp_size):
            group_seqs = [
                seq
                for seq in dp_seqs
                if seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
            ]

            for seq in group_seqs:
                bt = seq.block_table(self.engine_id, sp_rank)
                if len(bt) > 0:
                    valid_block_tables.append(bt)

        if not valid_block_tables:
            max_num_blocks = 0
        else:
            max_num_blocks = max(len(bt) for bt in valid_block_tables)

        final_table_data = []
        if max_num_blocks > 0:
            for bt in valid_block_tables:
                padding = [-1] * (max_num_blocks - len(bt))
                final_table_data.append(list(bt) + padding)

        if not final_table_data:
            block_tables = torch.empty((0, 0), dtype=torch.int32).cuda(
                non_blocking=True
            )
        else:
            block_tables = torch.tensor(
                final_table_data, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)

        return block_tables

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
        if _USING_CPP_UTILS:
            return self._prepare_decode_cpp(dp_seqs, is_dummy)
        else:
            return self._prepare_decode_py(dp_seqs, is_dummy)

    def _prepare_decode_cpp(self, dp_seqs: list[Sequence], is_dummy: bool = False):
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        block_size = self.config.kvcache_block_size

        # 调用 C++ 扩展获取元数据
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

        block_tables = (
            torch.tensor(meta.block_tables_flat, dtype=torch.int32, pin_memory=True)
            .reshape(sp_size, self.config.max_num_seqs, meta.max_num_blocks)
            .cuda(non_blocking=True)
        )

        # 4. 辅助掩码计算 (虽然 C++ 可以算，但保留 Python 计算 mask 逻辑通常更灵活，
        # 不过为了与 _prepare_decode_py 保持一致，这里使用 global/context_lens 计算)
        q_mask = global_context_lens.clone()
        q_mask[sp_rank].fill_(0)
        q_mask[q_mask != 0] = 1
        
        res_lse_mask = context_lens.clone()
        res_lse_mask[sp_rank].fill_(0)
        res_lse_mask[res_lse_mask != 0] = 1

        # 5. 新增字段转换 (Slices & Masks)
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

        # 需要 padding 的字段
        max_num_send_recv_seqs = max(
            self.config.max_num_send_seqs, self.config.max_num_recv_seqs
        )

        def pad_tensor(data_list, size, pad_val, dtype=torch.int32):
            if len(data_list) >= size:
                t_data = data_list[:size]
            else:
                t_data = data_list + [pad_val] * (size - len(data_list))
            return torch.tensor(t_data, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

        res_slice_get_to_buffer_input = pad_tensor(
            meta.res_slice_get_to_buffer_input, max_num_send_recv_seqs, -1
        )
        
        res_slice_fill_to_buffer_input = pad_tensor(
            meta.res_slice_fill_to_buffer_input, max_num_send_recv_seqs, -1
        )
        
        res_to_buffer_input_mask = pad_tensor(
            meta.res_to_buffer_input_mask, max_num_send_recv_seqs, 0
        )

        q_output_stride = torch.tensor(
            meta.q_output_stride, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        q_offsets = torch.tensor(
            meta.q_offsets, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # 6. 设置 Context
        set_context(
            False, # is_prefill
            self.config.max_num_seqs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            q_mask=q_mask,
            res_lse_mask=res_lse_mask,
            is_dummy=is_dummy,
            # 新增参数
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
            attention_compute_bs=meta.attention_compute_bs,
            q_output_stride=q_output_stride,
            q_offsets=q_offsets,
        )

        return input_ids, positions

    def _prepare_decode_py(self, dp_seqs: list[Sequence], is_dummy: bool = False):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        sp_rank = get_dist_context().attn_sp_rank

        for seq in dp_seqs:
            if seq.block_ctx(self.engine_id).master_sp_idx == sp_rank:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                slot_mapping.append(
                    seq.last_block_page_id(self.engine_id, sp_rank)
                    * get_cache_context().block_size
                    + seq.last_block_num_tokens(self.engine_id, sp_rank)
                    - 1
                )

        sp_size = get_dist_context().attn_sp_world_size
        sp_seqs = [
            [
                seq
                for seq in dp_seqs
                if seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
            ]
            for sp_idx in range(sp_size)
        ]
        sp_num_seqs = [len(seqs) for seqs in sp_seqs]
        context_lens = [
            [
                (
                    sp_seqs[sp_idx][seq_id].context_len(self.engine_id, sp_rank)
                    if seq_id < sp_num_seqs[sp_idx]
                    else 0
                )
                for seq_id in range(self.config.max_num_seqs)
            ]
            for sp_idx in range(sp_size)
        ]

        global_context_lens = [
            [
                (
                    sp_seqs[sp_rank][seq_id].context_len(self.engine_id, sp_idx)
                    if seq_id < sp_num_seqs[sp_rank]
                    else 0
                )
                for seq_id in range(self.config.max_num_seqs)
            ]
            for sp_idx in range(sp_size)
        ]

        # logger.info(f"{sp_rank=},{context_lens=},{global_context_lens=}")

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        global_context_lens = torch.tensor(
            global_context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(dp_seqs)
        q_mask = global_context_lens.clone()
        q_mask[sp_rank].fill_(0)
        q_mask[q_mask != 0] = 1
        res_lse_mask = context_lens.clone()
        res_lse_mask[sp_rank].fill_(0)
        res_lse_mask[res_lse_mask != 0] = 1
        set_context(
            False,
            self.config.max_num_seqs,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            q_mask=q_mask,
            res_lse_mask=res_lse_mask,
            is_dummy=is_dummy,
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
        context.context_lens_for_attn[:num_sp_seqs].add_(1)

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
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            context = get_context()
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            attention_compute_bs = context.attention_compute_bs
            selected_master_bs = next(x for x in self.graph_master_rank_bs if x >= bs)
            selected_attn_bs = next(
                x for x in self.graph_attn_compute_bs if x >= attention_compute_bs
            )
            sp_size = get_dist_context().attn_sp_world_size
            if sp_size == 1:
                assert selected_master_bs == selected_attn_bs
            graph_key = (selected_master_bs, selected_attn_bs)
            # print(f"use graph_key={graph_key}, selected_master_bs={selected_master_bs}, selected_attn_bs={selected_attn_bs}",flush=True)
            # print(f"context_lens_for_attn.shape: {context.context_lens_for_attn.shape}, context.context_lens_for_attn={context.context_lens_for_attn}",flush=True)
            graph = self.graphs[graph_key]

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
            graph_vars["res_slice_get_to_buffer_input"].copy_(context.res_slice_get_to_buffer_input)  # type: ignore
            graph_vars["res_slice_fill_to_buffer_input"].copy_(context.res_slice_fill_to_buffer_input)  # type: ignore
            graph_vars["res_to_buffer_input_mask"].copy_(context.res_to_buffer_input_mask)  # type: ignore

            if context.q_output_stride is not None:
                graph_vars["q_output_stride"].copy_(context.q_output_stride)  # type: ignore

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def migrate(self, seqs: list[Sequence]) -> None:
        get_cache_context().migrate(seqs=seqs)

    def run(self, dp_seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:

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
        max_attention_comp_seqs = config.max_num_seqs + config.max_num_recv_seqs
        hf_config.max_position_embeddings = max(
            config.max_model_len, hf_config.max_position_embeddings
        )
        max_bs = min(self.config.max_num_seqs, 512)
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
        max_num_send_recv_seqs = max(config.max_num_send_seqs, config.max_num_recv_seqs)
        q_slice_get = torch.full((max_bs,), -1, dtype=torch.int32)
        q_slice_fill = torch.full((max_bs,), -1, dtype=torch.int32)
        q_copy_mask = torch.zeros(max_bs, dtype=torch.int32)
        res_slice_get_to_buffer_output = torch.full((max_bs,), -1, dtype=torch.int32)
        res_slice_fill_to_buffer_output = torch.full((max_bs,), -1, dtype=torch.int32)
        res_to_buffer_output_mask = torch.zeros(max_bs, dtype=torch.int32)
        res_slice_get_to_buffer_input = torch.full(
            (max_num_send_recv_seqs,), -1, dtype=torch.int32
        )
        res_slice_fill_to_buffer_input = torch.full(
            (max_num_send_recv_seqs,), -1, dtype=torch.int32
        )
        res_to_buffer_input_mask = torch.zeros(
            max_num_send_recv_seqs, dtype=torch.int32
        )
        q_output_stride = torch.zeros(sp_world_size, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_master_rank_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_attn_compute_bs = [1, 2, 4, 8] + list(
            range(16, max_attention_comp_seqs + 1, 16)
        )
        self.graphs = {}
        self.graph_pool = None

        total_graphs = len(self.graph_master_rank_bs) * len(self.graph_attn_compute_bs)
        completed_graphs = 0
        skipped_graphs = 0
        sp_size = get_dist_context().attn_sp_world_size
        logger.info(f"开始捕获 CUDAGraph，总共需要捕获 {total_graphs} 个图")

        for master_bs in reversed(self.graph_master_rank_bs):
            for attn_bs in reversed(self.graph_attn_compute_bs):
                if (attn_bs < master_bs - config.max_num_send_seqs) or (
                    attn_bs > master_bs + config.max_num_recv_seqs
                ):
                    skipped_graphs += 1
                    logger.info(
                        f"跳过无效图组合 - (master_bs={master_bs}, attn_bs={attn_bs}) "
                        f"原因: attn_bs({attn_bs}) > master_bs×sp_size({master_bs}×{sp_size}={master_bs*sp_size})"
                    )
                    continue
                if sp_size == 1 and attn_bs != master_bs:
                    skipped_graphs += 1
                    logger.info(
                        f"跳过无效图组合 - (master_bs={master_bs}, attn_bs={attn_bs}) "
                        f"原因: SP Size = 1 下 attn_bs({attn_bs}) != master_bs({master_bs})"
                    )
                    continue

                completed_graphs += 1
                logger.info(
                    f"正在捕获图 {completed_graphs}/{total_graphs} - (master_bs={master_bs}, attn_bs={attn_bs})"
                )
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
                    q_output_stride=q_output_stride,
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
            q_slice_get=q_slice_get,
            q_slice_fill=q_slice_fill,
            q_copy_mask=q_copy_mask,
            res_slice_get_to_buffer_output=res_slice_get_to_buffer_output,
            res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output,
            res_to_buffer_output_mask=res_to_buffer_output_mask,
            res_slice_get_to_buffer_input=res_slice_get_to_buffer_input,
            res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input,
            res_to_buffer_input_mask=res_to_buffer_input_mask,
            attention_compute_bs=attn_bs,
            context_lens_for_attn=context_lens_for_attn,
            q_output_stride=q_output_stride,
        )
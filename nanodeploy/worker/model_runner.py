import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.profiler as profiler
from nanodeploy.config import Config
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
            set_sp_context(
                config.max_num_seqs,
                hf_config.head_dim,
                hf_config.num_attention_heads,
                hf_config.num_key_value_heads,
                torch.get_default_dtype(),
            )

        if ep_size > 1:
            import deep_ep

            deep_ep.Buffer.num_sms = 16
            dist.barrier(group=get_dist_context().cuda_world_group)

        model_architecture = hf_config.architectures[0]
        self.model = architectures[model_architecture](hf_config)

        self.run_count = 0
        self.prof_start = 0
        self.prof_end = 50
        self.profiler = None

        self.prof_kwargs = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "schedule": profiler.schedule(wait=1, warmup=1, active=30),
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(
                dir_name="/mnt/nvme1n1/ml_research/majinming/src/nano-deploy/",
                worker_name=f"trace_rank_{dist.get_rank()}",
            ),
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": True,
        }

        sp_size = get_dist_context().attn_sp_world_size
        ep_size = get_dist_context().ffn_ep_world_size

        if not get_runner_config().dummy_weight:
            load_model(self.model, config.model)

        dist.barrier()

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
        seqs = [
            Sequence(
                list(np.random.randint(low=0, high=10000, size=max_model_len)),
                engine_id=self.engine_id,
                master_sp_rank=sp_rank,
            )
            for _ in range(num_seqs)
        ]
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

        dp_sp_seqs = [
            [
                seq
                for seq in dp_seqs
                if seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
            ]
            for sp_idx in range(sp_size)
        ]

        max_num_blocks = max(
            [
                max([len(seq.block_table(self.engine_id, sp_rank)) for seq in seqs])
                for seqs in dp_sp_seqs
            ]
        )

        sp_num_seqs = [len(seqs) for seqs in dp_sp_seqs]

        block_tables = [
            [
                (
                    dp_sp_seqs[sp_idx][seq_id].block_table(self.engine_id, sp_rank)
                    + [-1]
                    * (
                        max_num_blocks
                        - len(
                            dp_sp_seqs[sp_idx][seq_id].block_table(
                                self.engine_id, sp_rank
                            )
                        )
                    )
                    if seq_id < sp_num_seqs[sp_idx]
                    else [-1] * max_num_blocks
                )
                for seq_id in range(self.config.max_num_seqs)
            ]
            for sp_idx in range(sp_size)
        ]
        # logger.info(f"{sp_rank=}, {block_tables=}")

        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], is_dummy: bool = False):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        sp_idx = get_dist_context().attn_sp_rank
        for seq in seqs:
            assert (
                seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
            ), f"{sp_idx=}, {seq.block_ctx(self.engine_id).master_sp_idx=}"
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table(self.engine_id, sp_idx):  # warmup
                continue
            num_blocks = seq.num_blocks(self.engine_id, sp_idx)
            for i in range(seq.num_cached_blocks, num_blocks):
                start = (
                    seq.block_table(self.engine_id, sp_idx)[i]
                    * get_cache_context().block_size
                )
                if i != seq.num_blocks(self.engine_id, sp_idx) - 1:
                    end = start + get_cache_context().block_size
                else:
                    end = start + seq.last_block_num_tokens(self.engine_id, sp_idx)
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            self.config.max_num_seqs,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            None,
            is_dummy=is_dummy,
        )
        return input_ids, positions

    def prepare_decode(self, dp_seqs: list[Sequence], is_dummy: bool = False):
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

        q_mask = [
            [
                (
                    sp_seqs[sp_rank][seq_id].context_len(self.engine_id, sp_idx)
                    if seq_id < sp_num_seqs[sp_rank] and sp_idx != sp_rank
                    else 0
                )
                for seq_id in range(self.config.max_num_seqs)
            ]
            for sp_idx in range(sp_size)
        ]

        res_lse_mask = [
            [
                (
                    sp_seqs[sp_idx][seq_id].context_len(self.engine_id, sp_rank)
                    if seq_id < sp_num_seqs[sp_idx] and sp_idx != sp_rank
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
        q_mask = torch.tensor(q_mask, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        res_lse_mask = torch.tensor(
            res_lse_mask, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(dp_seqs)
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

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            if (
                seq.block_ctx(self.engine_id).master_sp_idx
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
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

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
            graph_vars["block_tables"][
                :, :, : context.block_tables.size(2)  # type: ignore
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def migrate(self, seqs: list[Sequence]) -> None:
        get_cache_context().migrate(seqs=seqs)

    def run(self, dp_seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # """封装 run_model 的调用，加入 profiler 控制"""

        # # # 判断是否在目标范围内（50~100 次）
        # in_prof_range = (self.run_count >= self.prof_start) and (
        #     self.run_count <= self.prof_end
        # )

        # if self.run_count == self.prof_start and self.profiler is None:
        #     # 进入范围时启动 profiler
        #     self.profiler = profiler.profile(**self.prof_kwargs)
        #     self.profiler.start()
        #     print(f"开始 profiling（第 {self.run_count} 次）")
        # start_event.record()

        sp_rank = get_dist_context().attn_sp_rank

        num_sp_seqs = sum(
            1
            for seq in dp_seqs
            if seq.block_ctx(self.engine_id).master_sp_idx == sp_rank
        )
        is_dummy = False
        if num_sp_seqs == 0:
            is_dummy = True
            seq = Sequence(
                [np.random.randint(self.config.hf_config.vocab_size - 1)],
                engine_id=self.engine_id,
                master_sp_rank=get_dist_context().attn_sp_rank,
            )

            seq.block_ctx(self.engine_id).sp_block_table[sp_rank] = [0]
            dp_seqs.append(seq)

        sp_seqs = [
            seq
            for seq in dp_seqs
            if seq.block_ctx(self.engine_id).master_sp_idx == sp_rank
        ]

        loop_count = self.config.loop_count if not is_prefill else 1

        loop_count_token_ids = [[] for _ in sp_seqs]

        for i in range(loop_count):
            input_ids, positions = (
                self.prepare_prefill(dp_seqs, is_dummy)
                if is_prefill
                else self.prepare_decode(dp_seqs, is_dummy)
            )
            logits = self.run_model(input_ids, positions, is_prefill)
            tp_rank = get_dist_context().attn_tp_rank
            temperatures = (
                self.prepare_sample(dp_seqs) if tp_rank == 0 else [None] * len(sp_seqs)
            )
            token_ids = (
                self.sampler(logits, temperatures).tolist()
                if tp_rank == 0
                else [None] * len(sp_seqs)
            )
            for i, (seq, token_id) in enumerate(zip(sp_seqs, token_ids)):
                loop_count_token_ids[i].append(token_id)
                seq.num_tokens += 1
                seq.last_token = token_id
                seq.block_ctx(self.engine_id).num_dispatched_tokens[sp_rank] += 1
            self.run_count += 1  # 每次调用计数+1
            reset_context()
        # if in_prof_range and self.profiler is not None:
        #     self.profiler.step()
        #     # 在范围内时，每次调用结束后停止并记录（配合 schedule=active=1）
        #     # self.profiler.stop()
        #     print(f"记录第 {self.run_count} 次调用的性能数据")

        # if self.run_count > self.prof_end and self.profiler is not None:
        #     self.profiler.stop()
        #     # 超出范围后关闭 profiler
        #     self.profiler = None
        #     print(f"结束 profiling（共记录 {self.prof_end - self.prof_start + 1} 次）")
        # end_event.record()
        # torch.cuda.synchronize()
        # cuda_elapse_ms = start_event.elapsed_time(end_event)
        # cuda_time = torch.tensor(cuda_elapse_ms)
        # dist.all_reduce(cuda_time)
        # if dist.get_rank() == 0:
        #     print(
        #         f"model run latency: {(float(cuda_time) / get_dist_context().attn_dp_world_size):.2f} ms\n"
        #     )
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
        block_size = get_cache_context().block_size
        max_num_blocks = (config.max_model_len + block_size - 1) // block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        global_context_lens = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        q_mask = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        res_lse_mask = torch.zeros(sp_world_size, max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            sp_world_size, max_bs, max_num_blocks, dtype=torch.int32
        )
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                self.config.max_num_seqs,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens,
                block_tables=block_tables,
                global_context_lens=global_context_lens,
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            dist.barrier(group=get_dist_context().cuda_world_group)
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            global_context_lens=global_context_lens,
            q_mask=q_mask,
            res_lse_mask=res_lse_mask,
            outputs=outputs,
        )

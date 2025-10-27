import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler

from nanovllm.utils.loader import load_model

from nanovllm.engine.model_runner import ModelRunner as NanoVLLMModelRunner

from nanolmdeploy.utils.context import set_context, get_context, reset_context

from nanodeploy.models.deepseek_v2 import DeepseekV2ForCausalLM
from nanodeploy.worker.distributed import set_parallel_context, get_parallel_context
from nanodeploy.config import Config

from flash_mla import get_mla_metadata


class ModelRunner(NanoVLLMModelRunner):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    # def allocate_kv_cache(self):
    #     config = self.config
    #     hf_config = config.hf_config
    #     free, total = torch.cuda.mem_get_info()
    #     used = total - free
    #     peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    #     current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    #     num_kv_heads = hf_config.num_key_value_heads // dist.get_world_size(get_parallel_context().attn_tp_group)
    #     k_dim = hf_config.head_dim + hf_config.kv_lora_rank
    #     block_bytes = hf_config.num_hidden_layers * self.block_size * num_kv_heads * k_dim * hf_config.torch_dtype.itemsize
    #     print(hf_config.num_hidden_layers, self.block_size, num_kv_heads, k_dim, hf_config.torch_dtype.itemsize)
    #     config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        
    #     assert config.num_kvcache_blocks > 0
    #     self.kv_cache = torch.empty(hf_config.num_hidden_layers, 1, config.num_kvcache_blocks, self.block_size, num_kv_heads, k_dim, dtype=torch.bfloat16)
    #     print(f"{self.kv_cache.shape=}")
    #     layer_id = 0
    #     for module in self.model.modules():
    #         if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
    #             module.k_cache = self.kv_cache[0, layer_id]
    #             module.v_cache = self.kv_cache[1, layer_id]
    #             layer_id += 1

    # def init_cudagraph_buffer(self):
    #     """初始化所有需要的缓冲区变量"""

    #     max_num_blocks = (self.config.max_model_len + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
    #     max_num_blocks = min(max_num_blocks, self.config.num_kvcache_blocks)

    #     input_ids = torch.ones((1, self.config.max_num_seqs), dtype=torch.int64, device="cuda")
    #     position_ids = torch.zeros((1, self.config.max_num_seqs), dtype=torch.int64, device="cuda")
    #     # tile_scheduler_metadata is fix shape, num_splits is (bs+1,)
    #     tile_scheduler_metadata, num_splits = get_mla_metadata(
    #         torch.ones(self.config.max_num_seqs, dtype=torch.int32, device="cuda"), 
    #         self.hf_config.num_attention_heads // self.hf_config.num_key_value_heads, 
    #         self.hf_config.num_key_value_heads
    #     )
    #     cache_seqlens = torch.zeros(self.config.max_num_seqs, dtype=torch.int32, device="cuda")
    #     block_table = torch.zeros((self.config.max_num_seqs, max_num_blocks), dtype=torch.int32, device="cuda")
    #     outputs = torch.ones((1, self.config.max_num_seqs, self.hf_config.hidden_size), dtype=torch.bfloat16, device="cuda")

    #     self.graph_vars = dict(
    #         input_ids=input_ids,
    #         position_ids=position_ids,
    #         tile_scheduler_metadata=tile_scheduler_metadata,
    #         num_splits=num_splits,
    #         cache_seqlens=cache_seqlens,
    #         block_table=block_table,
    #         outputs=outputs,
    #     )

    #     bs = 4
    #     set_context(is_decoding=True, cache_seqlens=cache_seqlens[:bs], block_table=block_table[:bs, :], tile_scheduler_metadata=tile_scheduler_metadata, num_splits=num_splits[:bs+1])

    #     torch.cuda.synchronize()
    #     dist.barrier(group=get_parallel_context().cpu_world_group)

    # @torch.inference_mode()
    # def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    #     if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    #         return self.model.compute_logits(self.model(input_ids, positions, self.kv_cache))
    #     else:
    #         bs = input_ids.size(0)
    #         context = get_context()
    #         graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    #         graph_vars = self.graph_vars
    #         graph_vars["input_ids"][:bs] = input_ids
    #         graph_vars["positions"][:bs] = positions
    #         graph_vars["slot_mapping"].fill_(-1)
    #         graph_vars["slot_mapping"][:bs] = context.slot_mapping
    #         graph_vars["context_lens"].zero_()
    #         graph_vars["context_lens"][:bs] = context.context_lens
    #         graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
    #         graph.replay()
    #         return self.model.compute_logits(graph_vars["outputs"][:bs])

    # def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    #     input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    #     temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    #     logits = self.run_model(input_ids, positions, is_prefill)
    #     token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    #     reset_context()
    #     return token_ids

    # def warmup_model(self):
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     self.init_cudagraph_buffer()

    #     max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    #     num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    #     seqs = [Sequence([0] * 4) for _ in range(1)]
    #     print(len(seqs))
    #     self.run(seqs, True)
    #     torch.cuda.empty_cache()


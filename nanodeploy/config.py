import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 64
    num_kvcache_blocks: int = 3000

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 64 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.num_key_value_heads = 1
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

    @property
    def world_size(self):
        attn_world_size = self.data_parallel_size
        ffn_world_size = self.expert_parallel_size

        assert attn_world_size == ffn_world_size
        world_size = (attn_world_size + ffn_world_size) // 2
        return world_size


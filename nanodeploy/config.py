import os
from dataclasses import dataclass
from typing import Literal

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 128
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    attention_tp: int = 1
    attention_sp: int = 1
    attention_dp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1
    ffn_dp: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 15000

    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"

    master_addr: str | None = None
    master_port: int | None = None

    ray_address: str | None = "10.102.207.84:6379"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.attention_tp <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

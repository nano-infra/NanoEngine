import os
from dataclasses import dataclass
from typing import Any, Literal

from transformers import AutoConfig


@dataclass
class Config:
    model: str

    # scheduler config
    loop_count: int = 16
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9

    # parallel config
    attention_tp: int = 1
    attention_sp: int = 1
    attention_dp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1
    ffn_dp: int = 1

    # runner config
    enforce_eager: bool = False
    hf_config: Any = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 15000

    # deployment config
    engine_id: str | None = None
    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"

    dummy_prefill: bool | None = False
    dummy_weight: bool | None = False
    perfect_eplb: bool | None = False

    # dist config
    master_address: str = "127.0.0.1:6006"
    ray_address: str = "127.0.0.1:6379"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.attention_tp <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # self.max_model_len = max(
        #     self.max_model_len, self.hf_config.max_position_embeddings
        # )
        self.hf_config.max_position_embeddings = max(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

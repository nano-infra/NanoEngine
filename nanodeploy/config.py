import os
from dataclasses import dataclass
from typing import Any, Literal

from transformers import AutoConfig

# ==================== C++ Backend Configuration ====================
# Control whether to use C++ implementation for components
# Set to True to use C++ version, False to use Python version

USE_CPP_SEQUENCE: bool = True       # Sequence and BlockContext classes
USE_CPP_METRIC: bool = True         # SequenceMetric class
USE_CPP_BLOCK_MANAGER: bool = True  # Block and BlockManager classes
USE_CPP_SP_STATE_MANAGER: bool = True  # SPStateManager class
USE_CPP_MODEL_RUNNER: bool = True   # ModelRunner utils (prepare_prefill/decode)

# One-click switch for all components
USE_CPP_BACKEND: bool = True

def get_use_cpp_sequence() -> bool:
    return USE_CPP_BACKEND or USE_CPP_SEQUENCE

def get_use_cpp_metric() -> bool:
    return USE_CPP_BACKEND or USE_CPP_METRIC

def get_use_cpp_block_manager() -> bool:
    return USE_CPP_BACKEND or USE_CPP_BLOCK_MANAGER

def get_use_cpp_sp_state_manager() -> bool:
    return USE_CPP_BACKEND or USE_CPP_SP_STATE_MANAGER

def get_use_cpp_model_runner() -> bool:
    return USE_CPP_BACKEND or USE_CPP_MODEL_RUNNER
# ===================================================================

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

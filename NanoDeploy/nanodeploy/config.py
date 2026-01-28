import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig

from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class Config(BaseModel):
    model: str = Field(..., description="Path to the model")

    # scheduler config
    loop_count: int = 16
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_num_recv_seqs: int = 32
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    gpu_memory_limit_gb: Optional[float] = None
    routing_strategy: Literal["RoundRobin", "LeastBatch", "LeastCache"] = "RoundRobin"

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
    engine_id: Optional[str] = None
    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"
    host: str = "0.0.0.0"
    port: int = 5000

    dummy_prefill: Optional[bool] = False
    dummy_weight: Optional[bool] = False
    perfect_eplb: Optional[bool] = False

    # dist config
    master_address: str = "127.0.0.1:6006"
    ray_address: str = "127.0.0.1:6379"

    # profiler
    enable_profiler: bool = False
    profiler_start_step: int = 40
    profiling_step: int = 16
    profiler_dir: str = "/mnt/nvme1n1/ml_research/linbinbin1/profiler_res"

    # performance optimization
    use_dlslime_rpc: bool = True

    # logging config
    log_level: str = "CRITICAL"

    # etcd config
    enable_etcd: bool = False
    etcd_address: str = "127.0.0.1:2379"
    cluster_id: str = "default"

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # Remove isdir check to support HF Hub IDs
        # assert os.path.isdir(self.model)

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)

        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            assert self.kvcache_block_size == 64
            assert self.attention_tp == 1
        else:
            assert self.kvcache_block_size % 256 == 0
            assert 1 <= self.attention_tp <= 8

        if self.attention_sp == 1:
            self.max_num_recv_seqs = 0

        # Update hf_config max_position_embeddings
        if hasattr(self.hf_config, "max_position_embeddings"):
            self.hf_config.max_position_embeddings = max(
                self.max_model_len, self.hf_config.max_position_embeddings
            )
        else:
            # Fallback if attribute doesn't exist? or set it?
            # Usually causal LMs have it.
            self.hf_config.max_position_embeddings = self.max_model_len

        assert self.max_num_batched_tokens >= self.max_model_len

        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            # MLA requires num_kv_heads == 1
            if hasattr(self.hf_config, "num_key_value_heads"):
                self.hf_config.num_key_value_heads = 1

        return self

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

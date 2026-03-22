import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig, PretrainedConfig

from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class Config(BaseModel):
    model: str = Field(..., description="Path to the model")

    # scheduler config
    loop_count: int = 1
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 16
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
    trust_remote_code: bool = False
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
    enable_eplb: Optional[bool] = False

    # control plane config – enabled when nanoctrl_address is provided
    nanoctrl_scope: Optional[str] = None
    nanoctrl_address: Optional[str] = None

    # dist config
    master_address: str = "127.0.0.1:6006"
    ray_address: str = "127.0.0.1:6379"

    # profiler
    enable_profiler: bool = False
    profiler_start_step: int = 40
    profiling_step: int = 16
    profiler_dir: str = "./profiler_res"

    # logging config – override via NANODEPLOY_LOG_LEVEL env var
    log_level: str = os.getenv("NANODEPLOY_LOG_LEVEL", "INFO")

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # Normalise nanoctrl_address (add scheme if missing)
        if (
            self.nanoctrl_address
            and not self.nanoctrl_address.startswith("http://")
            and not self.nanoctrl_address.startswith("https://")
        ):
            self.nanoctrl_address = f"http://{self.nanoctrl_address}"

        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )

        # For VLM models with nested text_config (e.g. Qwen3.5-MoE),
        # flatten text_config attributes into hf_config for uniform access.
        if hasattr(self.hf_config, "text_config"):
            text_cfg = self.hf_config.text_config
            for attr in dir(text_cfg):
                if attr.startswith("_"):
                    continue
                if not hasattr(self.hf_config, attr):
                    try:
                        setattr(self.hf_config, attr, getattr(text_cfg, attr))
                    except Exception as e:
                        logger.warning(
                            f"Could not flatten attribute '{attr}' from text_config: {e}"
                        )
            # Explicitly propagate dtype/torch_dtype from text_config
            # (top-level config may have dtype=None while text_config has bfloat16)
            if getattr(text_cfg, "dtype", None) is not None:
                if getattr(self.hf_config, "dtype", None) is None:
                    self.hf_config.__dict__["dtype"] = text_cfg.dtype

        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            assert self.kvcache_block_size == 64
            assert self.attention_tp == 1
        else:
            assert self.kvcache_block_size % 64 == 0
            assert 1 <= self.attention_tp <= 8

        if self.attention_sp == 1:
            self.max_num_recv_seqs = 0

        if hasattr(self.hf_config, "max_position_embeddings"):
            self.hf_config.max_position_embeddings = max(
                self.max_model_len, self.hf_config.max_position_embeddings
            )
        else:
            self.hf_config.max_position_embeddings = self.max_model_len

        # With chunked prefill, max_num_batched_tokens may be smaller than max_model_len.
        assert self.max_num_batched_tokens >= 1

        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            if hasattr(self.hf_config, "num_key_value_heads"):
                self.hf_config.num_key_value_heads = 1

        # Convert dynamic trust_remote_code config class (from transformers_modules.*)
        # to a standard PretrainedConfig so Ray can serialize it across workers.
        if self.trust_remote_code and self.hf_config.__class__.__module__.startswith(
            "transformers_modules"
        ):
            _dtype = getattr(self.hf_config, "dtype", None)
            config_dict = self.hf_config.to_dict()
            self.hf_config = PretrainedConfig(**config_dict)
            # Preserve torch dtype (to_dict() may stringify it)
            if _dtype is not None:
                self.hf_config.dtype = _dtype

        return self

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

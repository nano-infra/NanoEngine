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
    max_num_recv_seqs: int = 32
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.85
    gpu_memory_limit_gb: float | None = None
    routing_strategy: Literal["RoundRobin", "LeastBatch", "LeastCache", "VLLMLoadBalance"] = "RoundRobin"
    scheduler_mode: Literal["centralized", "decentralized"] = "centralized"

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

    # profiler
    enable_profiler: bool = False
    profiler_start_step: int = 40
    profiling_step: int = 16
    profiler_dir: str = "./profiler_res"
    # Time-based profiling (in seconds). If set, will use time instead of steps.
    profiler_start_time: float | None = None  # Start profiling after N seconds
    profiling_duration: float | None = None  # Profile for N seconds

    # performance optimization
    use_dlslime_rpc: bool = True
    # Optimize Block Table transmission in Decode phase: if True, only send BlockTable
    # for sequences that have KVCache on the target rank; if False, send all BlockTables
    optimize_decode_block_table: bool = True

    # reserve for decode
    reserved_blocks_per_req: float = 1.0
    segment_size: int = 65536

    # Dynamic SP Size Knob
    enable_dynamic_sp_size: bool = False

    # Enable non-uniform KVCache partitioning for load balancing
    enable_non_uniform_split: bool = False

    # Strategy for how to select 
    sp_master_selector: Literal["RoundRobin", "LeastBatch", "LeastCache"] = "RoundRobin"

    # Debug mode for SP allocation (uses simplified RoundRobin + segment-based allocation)
    sp_debug: bool = False

    # Fixed number of SP segments per request (overrides segment_size calculation)
    # When set to a value > 0, all requests will be split into exactly this many segments
    fixed_sp_segments: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        # Convert custom config classes (e.g., kimi_k2 which maps to DeepseekV3ForCausalLM)
        # to the equivalent standard transformers config so Ray can pickle/unpickle without
        # needing the dynamic transformers_modules module on worker processes.
        if type(hf_config).__module__.startswith("transformers_modules"):
            config_dict = hf_config.to_dict()
            arch = (config_dict.get("architectures") or [""])[0]
            arch_to_model_type = {
                "DeepseekV3ForCausalLM": "deepseek_v3",
            }
            std_model_type = arch_to_model_type.get(arch)
            if std_model_type is None:
                raise ValueError(
                    f"Unsupported architecture '{arch}' with trust_remote_code config. "
                    f"Supported: {list(arch_to_model_type.keys())}"
                )
            config_dict.pop("model_type", None)
            hf_config = AutoConfig.for_model(std_model_type, **config_dict)
        self.hf_config = hf_config
        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            assert self.kvcache_block_size == 64
            assert self.attention_tp == 1
        else:
            assert self.kvcache_block_size % 256 == 0
            assert 1 <= self.attention_tp <= 8
        # self.max_model_len = max(
        #     self.max_model_len, self.hf_config.max_position_embeddings
        # )
        self.hf_config.max_position_embeddings = max(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.hf_config.architectures[0] == "DeepseekV3ForCausalLM":
            # MLA requires num_kv_heads == 1

            if hasattr(self.hf_config, "num_key_value_heads"):
                self.hf_config.num_key_value_heads = 1

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

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
    gpu_memory_utilization: float = 0.9
    gpu_memory_limit_gb: float | None = None
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
    profiler_dir: str = "/mnt/nvme1n1/ml_research/linbinbin1/profiler_res"

    # performance optimization
    use_dlslime_rpc: bool = True

    # reserve for decode
    reserved_blocks_per_req: float = 1.0
    segment_size: int = 65536

    # Dynamic SP Size Knob
    enable_dynamic_sp_size: bool = False

    # Enable non-uniform KVCache partitioning for load balancing
    enable_non_uniform_split: bool = False

    # Strategy for how to select 
    sp_master_selector: Literal["RoundRobin", "LeastBatch", "LeastCache"] = "RoundRobin"

    # === SP Size Policy Configuration ===
    # Mode: "segment" (original segment_size based) or "load_aware" (new adaptive strategy)
    # 
    # In "load_aware" mode, the system automatically:
    # - Learns long_req_threshold from historical traces
    # - Estimates expected waiting requests from arrival patterns
    # - Detects KVCache imbalance and balances Attention computation
    # - Uses LeastBatch strategy for master rank selection
    sp_size_mode: Literal["segment", "load_aware"] = "segment"
    
    # Initial statistics (optional, will be learned from runtime traces)
    # Can be set from offline analysis of dataset for faster warm-up
    initial_avg_prompt_length: float = 1024.0
    initial_avg_output_length: float = 256.0
    
    # Sliding window size for runtime statistics update
    stats_window_size: int = 1000
    
    # Hyperparameter learning/loading paths for offline trace-driven optimization
    export_hyperparams_path: str | None = None  # Path to export learned hyperparams (JSON)
    load_hyperparams_path: str | None = None     # Path to load pre-learned hyperparams (JSON)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
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

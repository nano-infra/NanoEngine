import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoConfig


DEEPSEEK_V3_BUCKET_POLICY = (
    "1:1024-104448;"
    "5:104449-174080;"
    "6:174081-194560;"
    "7:194561-436224;"
    "8:436225-1048576"
)


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
    sp_backend: Literal["legacy_ll", "hao_basic"] = "legacy_ll"
    # Optimize Block Table transmission in Decode phase: if True, only send BlockTable
    # for sequences that have KVCache on the target rank; if False, send all BlockTables
    optimize_decode_block_table: bool = True

    # reserve for decode
    reserved_blocks_per_req: float = 1.0
    segment_size: int = 65536

    # Dynamic SP Size Knob
    enable_dynamic_sp_size: bool = False
    # Decode-only scheduler implementation selector for dynamic SP:
    # False -> legacy can_allocate-based path
    # True  -> new batch planner path
    use_new_decode_dynamic_sp_scheduler: bool = False
    # SP size selection policy for the legacy dynamic-SP path.
    # "legacy": keep the current segment-based SP size search.
    # "long_short_sp8": prompt_len > dynamic_sp_long_request_threshold -> SP=attention_sp,
    #                   otherwise SP=1. Master selection and KV placement stay unchanged.
    # "bucket": choose CP size directly from a configured seq-len bucket policy.
    dynamic_sp_size_strategy: Literal["legacy", "long_short_sp8", "bucket"] = "legacy"
    dynamic_sp_long_request_threshold: int = 100000
    enable_dynamic_sp_bucket_policy: bool = False
    dynamic_sp_bucket_policy: str = ""
    dynamic_sp_bucket_preset: Literal["none", "deepseek_v3"] = "none"

    # Linear attention latency model for the new decode dynamic SP scheduler.
    # Current defaults come from:
    # mla_decode_latency_suite/mla_decode_cost_model/models/flashmla_axb_fit_20260405_160308.json
    # which was fit on CUDA Graph measurements with batch_size=64.
    dynamic_sp_attention_cost_a: float = 0.000444280970
    dynamic_sp_attention_cost_b: float = 9.862626316559
    # Q/Res/LSE defaults are calibrated from the current DLSlime hao_basic path
    # used by NanoDeploy-new. Keep this note here so later updates do not
    # accidentally mix old all_to_all_ll coefficients with hao_basic ones.
    dynamic_sp_q_cost_a: float = 0.000002768410
    dynamic_sp_q_cost_b: float = 5.924184585189
    dynamic_sp_res_cost_a: float = 0.000002764389
    dynamic_sp_res_cost_b: float = 5.639326242620
    dynamic_sp_lse_cost_a: float = 0.000026547019
    dynamic_sp_lse_cost_b: float = 4.263571143096
    # Leave these at 0 to auto-derive bytes-per-edge for DeepSeek-V3 MLA in __post_init__().
    dynamic_sp_q_bytes_per_edge: int = 0
    dynamic_sp_res_bytes_per_edge: int = 0
    dynamic_sp_lse_bytes_per_edge: int = 0

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
        if self.dynamic_sp_size_strategy not in {"legacy", "long_short_sp8", "bucket"}:
            raise ValueError(
                "dynamic_sp_size_strategy must be one of: legacy, long_short_sp8, bucket"
            )
        if self.dynamic_sp_bucket_preset not in {"none", "deepseek_v3"}:
            raise ValueError(
                "dynamic_sp_bucket_preset must be one of: none, deepseek_v3"
            )
        preset_policy = ""
        if self.dynamic_sp_bucket_preset == "deepseek_v3":
            preset_policy = DEEPSEEK_V3_BUCKET_POLICY
        if preset_policy:
            if (
                self.dynamic_sp_bucket_policy.strip()
                and self.dynamic_sp_bucket_policy.strip() != preset_policy
            ):
                raise ValueError(
                    "dynamic_sp_bucket_policy conflicts with dynamic_sp_bucket_preset"
                )
            self.dynamic_sp_bucket_policy = preset_policy
        bucket_requested = (
            self.dynamic_sp_size_strategy == "bucket"
            or self.enable_dynamic_sp_bucket_policy
            or bool(self.dynamic_sp_bucket_policy.strip())
            or self.dynamic_sp_bucket_preset != "none"
        )
        if self.dynamic_sp_size_strategy != "bucket" and bucket_requested:
            raise ValueError(
                "bucket policy is an independent scheduling strategy; use "
                "dynamic_sp_size_strategy='bucket' instead of combining it with "
                "legacy/long_short_sp8"
            )
        if self.dynamic_sp_size_strategy == "bucket":
            self.enable_dynamic_sp_bucket_policy = True
        if self.enable_dynamic_sp_bucket_policy and not self.dynamic_sp_bucket_policy.strip():
            raise ValueError(
                "dynamic_sp_bucket_policy must be non-empty when "
                "enable_dynamic_sp_bucket_policy is True"
            )
        if (
            self.enable_dynamic_sp_bucket_policy
            and self.use_new_decode_dynamic_sp_scheduler
        ):
            raise ValueError(
                "dynamic_sp_bucket_policy only applies to the legacy can_allocate path "
                "and must not be combined with "
                "use_new_decode_dynamic_sp_scheduler=True"
            )
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
        if self.dynamic_sp_bucket_preset == "deepseek_v3":
            if self.hf_config.architectures[0] != "DeepseekV3ForCausalLM":
                raise ValueError(
                    "dynamic_sp_bucket_preset=deepseek_v3 only supports "
                    "DeepseekV3ForCausalLM"
                )
            if self.attention_sp < 8:
                raise ValueError(
                    "dynamic_sp_bucket_preset=deepseek_v3 requires attention_sp >= 8"
                )
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

            def _dtype_size_bytes(torch_dtype: Any) -> int:
                if torch_dtype is None:
                    return 2
                if isinstance(torch_dtype, str):
                    mapping = {
                        "torch.float16": 2,
                        "float16": 2,
                        "torch.bfloat16": 2,
                        "bfloat16": 2,
                        "torch.float32": 4,
                        "float32": 4,
                    }
                    return mapping.get(torch_dtype, 2)
                try:
                    return torch.tensor([], dtype=torch_dtype).element_size()
                except Exception:
                    return 2

            dtype_size = _dtype_size_bytes(getattr(self.hf_config, "torch_dtype", None))
            num_heads = int(getattr(self.hf_config, "num_attention_heads"))
            kv_lora_rank = int(getattr(self.hf_config, "kv_lora_rank"))
            qk_rope_head_dim = int(getattr(self.hf_config, "qk_rope_head_dim"))

            if self.dynamic_sp_q_bytes_per_edge <= 0:
                self.dynamic_sp_q_bytes_per_edge = num_heads * (kv_lora_rank + qk_rope_head_dim) * dtype_size
            if self.dynamic_sp_res_bytes_per_edge <= 0:
                self.dynamic_sp_res_bytes_per_edge = num_heads * kv_lora_rank * dtype_size
            if self.dynamic_sp_lse_bytes_per_edge <= 0:
                # LSE is explicitly converted to bfloat16 before communication.
                self.dynamic_sp_lse_bytes_per_edge = num_heads * 2

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

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
    cuda_graph_mode: Literal["full", "piecewise"] = "full"
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
    sp_backend: Literal["legacy_ll", "hao_basic", "nccl", "nccl_compact"] = "legacy_ll"
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
    # 0 is normalized in __post_init__ to preserve the historical behavior:
    # long requests use attention_sp.
    dynamic_sp_long_request_size: int = 0
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

    # Fixed number of participating SP ranks per request.
    # 0 keeps the segment-size / dynamic-SP scheduling behavior.
    fixed_sp_size: int = 0

    # LoongServe-style Decode-only multi-master scheduler.
    enable_ls_decode_core_scheduler: bool = False
    # 0 selects the smallest feasible admission-time KV DoP automatically.
    ls_decode_initial_kv_dop: int = 0
    ls_decode_batch_per_master: int = 64
    ls_decode_enable_memory_scale_up: bool = True
    # Automatic KV consolidation is opt-in. Shadow mode evaluates the
    # utilization/stability gates without reserving blocks or moving KV.
    ls_kv_consolidation_mode: Literal["off", "shadow", "execute"] = "off"
    ls_kv_consolidation_candidate_util: float = 0.50
    ls_kv_consolidation_target_high_watermark: float = 0.80
    ls_kv_consolidation_stable_steps: int = 32
    ls_kv_consolidation_cooldown_steps: int = 64
    ls_kv_consolidation_check_interval_steps: int = 8
    # Hard execution budget. 0 means uncalibrated and forbids automatic execute.
    ls_kv_consolidation_max_source_blocks_per_event: int = 0
    # 0 keeps the physical KV-consolidation P2P transport disabled.  A positive
    # value reserves a fixed token-major scratch buffer before sizing KV blocks.
    ls_kv_consolidation_migration_chunk_tokens: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if not 0 <= self.ls_decode_initial_kv_dop <= self.attention_sp:
            raise ValueError(
                "ls_decode_initial_kv_dop must be in [0, attention_sp]"
            )
        if self.ls_decode_batch_per_master <= 0:
            raise ValueError("ls_decode_batch_per_master must be > 0")
        if self.ls_kv_consolidation_mode not in {"off", "shadow", "execute"}:
            raise ValueError(
                "ls_kv_consolidation_mode must be one of: off, shadow, execute"
            )
        if not 0.0 < self.ls_kv_consolidation_candidate_util <= 1.0:
            raise ValueError(
                "ls_kv_consolidation_candidate_util must be in (0, 1]"
            )
        if not 0.0 < self.ls_kv_consolidation_target_high_watermark <= 1.0:
            raise ValueError(
                "ls_kv_consolidation_target_high_watermark must be in (0, 1]"
            )
        if self.ls_kv_consolidation_stable_steps <= 0:
            raise ValueError("ls_kv_consolidation_stable_steps must be > 0")
        if self.ls_kv_consolidation_cooldown_steps < 0:
            raise ValueError("ls_kv_consolidation_cooldown_steps must be >= 0")
        if self.ls_kv_consolidation_check_interval_steps <= 0:
            raise ValueError(
                "ls_kv_consolidation_check_interval_steps must be > 0"
            )
        if self.ls_kv_consolidation_max_source_blocks_per_event < 0:
            raise ValueError(
                "ls_kv_consolidation_max_source_blocks_per_event must be >= 0"
            )
        if self.ls_kv_consolidation_migration_chunk_tokens < 0:
            raise ValueError(
                "ls_kv_consolidation_migration_chunk_tokens must be >= 0"
            )
        if (
            self.ls_kv_consolidation_mode != "off"
            and not self.enable_ls_decode_core_scheduler
        ):
            raise ValueError(
                "ls_kv_consolidation_mode requires "
                "enable_ls_decode_core_scheduler=True"
            )
        if (
            self.ls_kv_consolidation_mode == "execute"
            and self.ls_kv_consolidation_migration_chunk_tokens == 0
        ):
            raise ValueError(
                "ls_kv_consolidation_mode='execute' requires "
                "ls_kv_consolidation_migration_chunk_tokens > 0"
            )
        if (
            self.ls_kv_consolidation_mode == "execute"
            and self.ls_kv_consolidation_max_source_blocks_per_event == 0
        ):
            raise ValueError(
                "ls_kv_consolidation_mode='execute' requires "
                "ls_kv_consolidation_max_source_blocks_per_event > 0"
            )
        if self.fixed_sp_size < 0:
            raise ValueError("fixed_sp_size must be >= 0")
        if self.fixed_sp_size > self.attention_sp:
            raise ValueError("fixed_sp_size must be in [0, attention_sp]")
        if self.fixed_sp_size > 0 and (
            self.enable_dynamic_sp_size
            or self.use_new_decode_dynamic_sp_scheduler
            or self.dynamic_sp_size_strategy != "legacy"
            or self.enable_dynamic_sp_bucket_policy
            or bool(self.dynamic_sp_bucket_policy.strip())
            or self.dynamic_sp_bucket_preset != "none"
            or self.sp_debug
        ):
            raise ValueError(
                "fixed_sp_size is a baseline scheduling mode and cannot be "
                "combined with dynamic SP size strategies"
            )
        if self.dynamic_sp_size_strategy not in {"legacy", "long_short_sp8", "bucket"}:
            raise ValueError(
                "dynamic_sp_size_strategy must be one of: legacy, long_short_sp8, bucket"
            )
        if self.dynamic_sp_bucket_preset not in {"none", "deepseek_v3"}:
            raise ValueError(
                "dynamic_sp_bucket_preset must be one of: none, deepseek_v3"
            )
        if self.dynamic_sp_long_request_size < 0:
            raise ValueError("dynamic_sp_long_request_size must be >= 0")
        if self.dynamic_sp_long_request_size == 0:
            self.dynamic_sp_long_request_size = self.attention_sp
        if (
            self.dynamic_sp_long_request_size < 1
            or self.dynamic_sp_long_request_size > self.attention_sp
        ):
            raise ValueError(
                "dynamic_sp_long_request_size must be in [1, attention_sp]"
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
        if self.enable_ls_decode_core_scheduler:
            unsupported = []
            if self.mode != "decode":
                unsupported.append("mode must be 'decode'")
            if self.dummy_prefill is not True:
                unsupported.append("dummy_prefill must be True")
            if self.scheduler_mode != "centralized":
                unsupported.append("scheduler_mode must be 'centralized'")
            if self.loop_count != 1:
                unsupported.append("loop_count must be 1")
            ls_parallel_topology = (
                self.attention_dp,
                self.attention_sp,
                self.attention_tp,
                self.ffn_ep,
                self.ffn_dp,
                self.ffn_tp,
            )
            supported_ls_parallel_topologies = {
                (1, 8, 1, 8, 1, 1),   # single-node functional preflight
                (4, 8, 1, 32, 1, 1),  # target 4-DP experiment
            }
            if ls_parallel_topology not in supported_ls_parallel_topologies:
                unsupported.append(
                    "parallel topology must be either the single-node preflight "
                    "(attention_dp=1, attention_sp=8, attention_tp=1, "
                    "ffn_ep=8, ffn_dp=1, ffn_tp=1) or the target experiment "
                    "(attention_dp=4, attention_sp=8, attention_tp=1, "
                    "ffn_ep=32, ffn_dp=1, ffn_tp=1)"
                )
            if not self.use_dlslime_rpc:
                unsupported.append("use_dlslime_rpc must be True")
            if self.sp_backend != "hao_basic":
                unsupported.append("sp_backend must be 'hao_basic'")
            if self.fixed_sp_size != 0:
                unsupported.append("fixed_sp_size must be 0")
            if self.enable_dynamic_sp_size:
                unsupported.append("enable_dynamic_sp_size must be False")
            if self.dynamic_sp_size_strategy != "legacy":
                unsupported.append("dynamic_sp_size_strategy must be 'legacy'")
            if self.use_new_decode_dynamic_sp_scheduler:
                unsupported.append(
                    "use_new_decode_dynamic_sp_scheduler must be False"
                )
            if self.enable_non_uniform_split:
                unsupported.append("enable_non_uniform_split must be False")
            if self.sp_debug:
                unsupported.append("sp_debug must be False")
            if unsupported:
                raise ValueError(
                    "LS-Decode-Core unsupported configuration: "
                    + "; ".join(unsupported)
                )
        hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if self.cuda_graph_mode not in {"full", "piecewise"}:
            raise ValueError("cuda_graph_mode must be one of: full, piecewise")
        if (
            self.sp_backend == "nccl_compact"
            and not self.enforce_eager
            and self.cuda_graph_mode != "piecewise"
        ):
            raise ValueError(
                "sp_backend='nccl_compact' uses variable split-size NCCL and "
                "requires enforce_eager=True or cuda_graph_mode='piecewise'"
            )
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
        if (
            self.enable_ls_decode_core_scheduler
            and self.hf_config.architectures[0] != "DeepseekV3ForCausalLM"
        ):
            raise ValueError(
                "LS-Decode-Core only supports the DeepseekV3ForCausalLM model family"
            )
        if (
            self.cuda_graph_mode == "piecewise"
            and self.hf_config.architectures[0] != "DeepseekV3ForCausalLM"
        ):
            raise ValueError(
                "cuda_graph_mode='piecewise' currently supports DeepseekV3ForCausalLM only"
            )
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

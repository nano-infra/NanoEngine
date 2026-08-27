import os
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoConfig

from nanodeploy.engine.hierarchical_contract import (
    CONTROL_DUMMY_SCHEMA_VERSION,
    HIERARCHICAL_LOOP_COUNT,
)
from nanodeploy.engine.topology import (
    HierarchicalTopology,
    build_hierarchical_topology,
)
from nanodeploy.worker.decode_backend_compat import (
    resolve_decode_deepep_config,
)


DEEPSEEK_V3_BUCKET_POLICY = (
    "1:1024-63488;"
    "5:63489-210944;"
    "6:210945-399360;"
    "7:399361-428032;"
    "8:428033-1048576"
)

KIMI_K2_BUCKET_POLICY = (
    "1:1024-10240;"
    "2:10241-22528;"
    "3:22529-190464;"
    "4:190465-210944;"
    "5:210945-354304;"
    "6:354305-624640;"
    "7:624641-673792;"
    "8:673793-1000000"
)

DYNAMIC_SP_BUCKET_POLICIES = {
    "deepseek_v3": DEEPSEEK_V3_BUCKET_POLICY,
    "kimi_k2": KIMI_K2_BUCKET_POLICY,
}


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
    routing_strategy: Literal["RoundRobin", "LeastBatch", "LeastCache"] = "RoundRobin"
    scheduler_arch: Literal["legacy_global", "hierarchical"] = "legacy_global"
    router_policy: Literal[
        "round_robin",
        "least_batch",
        "least_cache",
        "least_projected_load",
    ] = "least_batch"
    load_report_interval_ms: int = 100
    hierarchical_queue_capacity: int = 4096
    max_ingress_batch_requests: int = 256
    # 0 disables the wall-clock ingress budget. The per-drain request cap
    # remains the fairness boundary that prevents decode starvation.
    max_ingress_drain_ms: float = 0.0
    startup_timeout_s: float = 600.0
    quantum_timeout_s: float = 120.0
    hierarchical_execution_trace: bool = False
    # Capture one compact timing/load record per decode quantum. The legacy
    # field name is retained for configuration/fingerprint compatibility;
    # both scheduler architectures may enable it.
    hierarchical_quantum_diagnostics: bool = False
    # Positional result path reuses the scheduler's frozen row layout. Disable
    # only for diagnostic A/B against the request-id dict/set rebuild path.
    hierarchical_result_fastpath: bool = True
    # Ray remains the lifecycle/placement plane. RequestRouter<->LocalEngine
    # ingress always uses ZMQ. This option selects LocalEngine<->ModelRunner
    # quantum control/results; DLSlime continues to carry Sequence payloads.
    hierarchical_worker_transport: Literal["ray", "zmq"] = "ray"

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
    moe_routing_simulation_strategy: Literal[
        "model", "uniform_random", "perfect_eplb"
    ] = "model"
    seed: int = 0

    # dist config
    master_address: str = "127.0.0.1:6006"
    ray_address: str = "127.0.0.1:6379"
    hierarchical_control_address: str | None = None

    # profiler
    enable_profiler: bool = False
    profiler_start_step: int = 40
    profiling_step: int = 16
    profiler_dir: str = "./profiler_res"
    # Time-based profiling (in seconds). If set, will use time instead of steps.
    profiler_start_time: float | None = None  # Start profiling after N seconds
    profiling_duration: float | None = None  # Profile for N seconds
    profiler_mode: Literal["default", "runtime_overhead"] = "default"
    profiler_ranks: tuple[int, ...] | None = None
    runtime_overhead_timing: bool = False

    # performance optimization
    use_dlslime_rpc: bool = True
    sp_backend: Literal["hao_basic", "nccl"] = "hao_basic"
    # Optimize Block Table transmission in Decode phase: if True, only send BlockTable
    # for sequences that have KVCache on the target rank; if False, send all BlockTables
    optimize_decode_block_table: bool = True

    # reserve for decode
    reserved_blocks_per_req: float = 1.0
    segment_size: int = 65536

    # SP size selection policy for dynamic SP placement.
    # "legacy": keep the current segment-based SP size selection.
    # "bucket": choose CP size directly from a configured seq-len bucket policy.
    dynamic_sp_size_strategy: Literal["legacy", "bucket"] = "legacy"
    enable_dynamic_sp_bucket_policy: bool = False
    dynamic_sp_bucket_policy: str = ""
    dynamic_sp_bucket_preset: Literal[
        "none", "deepseek_v3", "kimi_k2"
    ] = "none"

    # Enable non-uniform KVCache partitioning for load balancing
    enable_non_uniform_split: bool = False

    # Strategy for how to select 
    sp_master_selector: Literal[
        "RoundRobin", "LeastBatch", "LeastCache"
    ] = "LeastBatch"

    # Fixed number of participating SP ranks per request.
    # 0 keeps the segment-size / dynamic-SP scheduling behavior.
    fixed_sp_size: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if self.profiler_mode not in {"default", "runtime_overhead"}:
            raise ValueError(
                "profiler_mode must be one of: default, runtime_overhead"
            )
        if self.profiler_mode == "runtime_overhead" and not self.enable_profiler:
            raise ValueError(
                "profiler_mode='runtime_overhead' requires enable_profiler=True"
            )
        if self.runtime_overhead_timing and self.enable_profiler:
            raise ValueError(
                "runtime_overhead_timing cannot be combined with enable_profiler"
            )
        if self.profiler_ranks is not None:
            self.profiler_ranks = tuple(self.profiler_ranks)
            if not self.enable_profiler:
                raise ValueError("profiler_ranks requires enable_profiler=True")
            if not self.profiler_ranks:
                raise ValueError("profiler_ranks must not be empty")
            if any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                for rank in self.profiler_ranks
            ):
                raise ValueError(
                    "profiler_ranks must contain non-negative integers"
                )
            if len(set(self.profiler_ranks)) != len(self.profiler_ranks):
                raise ValueError("profiler_ranks must not contain duplicates")
            if any(rank >= self.attn_world_size for rank in self.profiler_ranks):
                raise ValueError(
                    "profiler_ranks entries must be smaller than "
                    f"attn_world_size ({self.attn_world_size})"
                )
        if self.profiler_start_step < 0:
            raise ValueError("profiler_start_step must be non-negative")
        if self.profiling_step <= 0:
            raise ValueError("profiling_step must be positive")
        valid_moe_routing_strategies = {
            "model",
            "uniform_random",
            "perfect_eplb",
        }
        if self.moe_routing_simulation_strategy not in valid_moe_routing_strategies:
            raise ValueError(
                "moe_routing_simulation_strategy must be one of: "
                "model, uniform_random, perfect_eplb"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.perfect_eplb:
            if self.moe_routing_simulation_strategy == "uniform_random":
                raise ValueError(
                    "perfect_eplb=True conflicts with "
                    "moe_routing_simulation_strategy='uniform_random'"
                )
            self.moe_routing_simulation_strategy = "perfect_eplb"
        if self.scheduler_arch not in {"legacy_global", "hierarchical"}:
            raise ValueError(
                "scheduler_arch must be one of: legacy_global, hierarchical"
            )
        if self.routing_strategy not in {
            "RoundRobin",
            "LeastBatch",
            "LeastCache",
        }:
            raise ValueError(
                "routing_strategy must be one of: "
                "RoundRobin, LeastBatch, LeastCache"
            )
        if self.hierarchical_worker_transport not in {"ray", "zmq"}:
            raise ValueError(
                "hierarchical_worker_transport must be one of: ray, zmq"
            )
        if (
            self.hierarchical_worker_transport == "zmq"
            and self.scheduler_arch != "hierarchical"
        ):
            raise ValueError(
                "hierarchical_worker_transport='zmq' requires "
                "scheduler_arch='hierarchical'"
            )
        if self.router_policy not in {
            "round_robin",
            "least_batch",
            "least_cache",
            "least_projected_load",
        }:
            raise ValueError(
                "router_policy must be one of: "
                "round_robin, least_batch, least_cache, "
                "least_projected_load"
            )
        if self.load_report_interval_ms <= 0:
            raise ValueError("load_report_interval_ms must be positive")
        if self.hierarchical_queue_capacity <= 0:
            raise ValueError("hierarchical_queue_capacity must be positive")
        if self.max_ingress_batch_requests <= 0:
            raise ValueError("max_ingress_batch_requests must be positive")
        if self.max_ingress_drain_ms < 0:
            raise ValueError("max_ingress_drain_ms must be non-negative")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.quantum_timeout_s <= 0:
            raise ValueError("quantum_timeout_s must be positive")
        if self.scheduler_arch == "hierarchical":
            if self.mode != "decode":
                raise ValueError(
                    "hierarchical scheduler requires mode='decode'"
                )
            if self.dummy_prefill is not True:
                raise ValueError(
                    "hierarchical scheduler requires dummy_prefill=True"
                )
            if self.loop_count != HIERARCHICAL_LOOP_COUNT:
                raise ValueError(
                    "hierarchical scheduler requires loop_count="
                    f"{HIERARCHICAL_LOOP_COUNT}"
                )
            if not self.use_dlslime_rpc:
                raise ValueError(
                    "hierarchical scheduler MVP requires use_dlslime_rpc=True"
                )
            build_hierarchical_topology(
                attention_dp=self.attention_dp,
                attention_sp=self.attention_sp,
                attention_tp=self.attention_tp,
                ffn_dp=self.ffn_dp,
                ffn_ep=self.ffn_ep,
                ffn_tp=self.ffn_tp,
            )
        if self.fixed_sp_size < 0:
            raise ValueError("fixed_sp_size must be >= 0")
        if self.fixed_sp_size > self.attention_sp:
            raise ValueError("fixed_sp_size must be in [0, attention_sp]")
        if self.fixed_sp_size > 0 and (
            self.dynamic_sp_size_strategy != "legacy"
            or self.enable_dynamic_sp_bucket_policy
            or bool(self.dynamic_sp_bucket_policy.strip())
            or self.dynamic_sp_bucket_preset != "none"
        ):
            raise ValueError(
                "fixed_sp_size is a baseline scheduling mode and cannot be "
                "combined with dynamic SP size strategies"
            )
        if self.dynamic_sp_size_strategy not in {"legacy", "bucket"}:
            raise ValueError(
                "dynamic_sp_size_strategy must be one of: legacy, bucket"
            )
        if self.dynamic_sp_bucket_preset not in {
            "none",
            *DYNAMIC_SP_BUCKET_POLICIES,
        }:
            raise ValueError(
                "dynamic_sp_bucket_preset must be one of: "
                "none, deepseek_v3, kimi_k2"
            )
        preset_policy = DYNAMIC_SP_BUCKET_POLICIES.get(
            self.dynamic_sp_bucket_preset, ""
        )
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
                "legacy"
            )
        if self.dynamic_sp_size_strategy == "bucket":
            self.enable_dynamic_sp_bucket_policy = True
        if self.enable_dynamic_sp_bucket_policy and not self.dynamic_sp_bucket_policy.strip():
            raise ValueError(
                "dynamic_sp_bucket_policy must be non-empty when "
                "enable_dynamic_sp_bucket_policy is True"
            )
        hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        source_model_type = getattr(hf_config, "model_type", "")
        if self.cuda_graph_mode not in {"full", "piecewise"}:
            raise ValueError("cuda_graph_mode must be one of: full, piecewise")
        if self.sp_backend not in {"hao_basic", "nccl"}:
            raise ValueError("sp_backend must be one of: hao_basic, nccl")
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
        if self.dynamic_sp_bucket_preset != "none":
            if self.hf_config.architectures[0] != "DeepseekV3ForCausalLM":
                raise ValueError(
                    "dynamic SP bucket presets only support "
                    "DeepseekV3ForCausalLM"
                )
            if source_model_type != self.dynamic_sp_bucket_preset:
                raise ValueError(
                    "dynamic_sp_bucket_preset="
                    f"{self.dynamic_sp_bucket_preset} requires model_type="
                    f"{self.dynamic_sp_bucket_preset}, got {source_model_type!r}"
                )
            if self.attention_sp < 8:
                raise ValueError(
                    "dynamic_sp_bucket_preset="
                    f"{self.dynamic_sp_bucket_preset} requires attention_sp >= 8"
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

    @property
    def attn_world_size(self):
        return self.attention_dp * self.attention_sp * self.attention_tp

    @property
    def ffn_world_size(self):
        return self.ffn_dp * self.ffn_ep * self.ffn_tp

    @property
    def hierarchical_topology(self) -> HierarchicalTopology:
        if self.scheduler_arch != "hierarchical":
            raise ValueError(
                "hierarchical_topology is only available for "
                "scheduler_arch='hierarchical'"
            )
        return build_hierarchical_topology(
            attention_dp=self.attention_dp,
            attention_sp=self.attention_sp,
            attention_tp=self.attention_tp,
            ffn_dp=self.ffn_dp,
            ffn_ep=self.ffn_ep,
            ffn_tp=self.ffn_tp,
        )

    def collective_fingerprint(self) -> str:
        """Hash fields that must agree before hierarchical workers become READY."""

        hf_dtype = getattr(self.hf_config, "dtype", None)
        if hf_dtype is None:
            hf_dtype = getattr(self.hf_config, "torch_dtype", None)
        communication_env_names = (
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_P2P_DISABLE",
            "NCCL_IB_DISABLE",
            "SLIME_QP_NUM",
        )
        deepep_config = (
            resolve_decode_deepep_config(self.max_num_seqs)
            if self.ffn_ep > 1
            else None
        )
        payload = {
            "model": os.path.realpath(self.model),
            "dtype": str(hf_dtype),
            "attention": [
                self.attention_dp,
                self.attention_sp,
                self.attention_tp,
            ],
            "ffn": [self.ffn_dp, self.ffn_ep, self.ffn_tp],
            "sp_backend": self.sp_backend,
            "cuda_graph": {
                "enforce_eager": self.enforce_eager,
                "mode": self.cuda_graph_mode,
            },
            "batch_limits": [
                self.max_num_seqs,
                self.max_num_batched_tokens,
                self.max_num_recv_seqs,
            ],
            "kv": [
                self.kvcache_block_size,
                self.num_kvcache_blocks,
                self.max_model_len,
            ],
            "moe_routing": {
                "strategy": self.moe_routing_simulation_strategy,
                "seed": self.seed,
                "legacy_perfect_eplb": self.perfect_eplb,
            },
            "dynamic_sp": {
                "strategy": self.dynamic_sp_size_strategy,
                "bucket": self.dynamic_sp_bucket_policy,
                "fixed_sp_size": self.fixed_sp_size,
            },
            "dummy_schema_version": CONTROL_DUMMY_SCHEMA_VERSION,
            "loop_count": self.loop_count,
            "hierarchical_quantum_diagnostics": (
                self.hierarchical_quantum_diagnostics
            ),
            "hierarchical_worker_transport": (
                self.hierarchical_worker_transport
            ),
            "communication_env": {
                name: os.getenv(name) for name in communication_env_names
            },
            "deepep": (
                deepep_config.fingerprint_payload()
                if deepep_config is not None
                else None
            ),
            "hierarchical_control_address": (
                self.hierarchical_control_address
                or f"derived-from:{self.master_address}"
            ),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

import json
import os
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig, PretrainedConfig

from dlengine.logging import get_logger
from dlengine.models.trait import apply_hf_config_compatibility_fixes

logger = get_logger("dlengine")


class Config(BaseModel):

    model_config = {
        "arbitrary_types_allowed": True,
    }

    model: str = Field(..., description="Path to the model")

    # scheduler config
    # Maximum tokens processed by one model forward. With pipeline parallel
    # prefill, the scheduler admits up to ``max_num_batched_tokens * pp``
    # tokens and splits that window into microbatches of this size.
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 16
    max_num_recv_seqs: int = 32
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    gpu_memory_limit_gb: Optional[float] = None
    # CPU KV spill tier for preempted decode requests. GiB budget per local
    # worker/rank; 0 disables it.
    host_utilization_per_device: float = 0.0
    routing_strategy: Literal[
        "RoundRobin", "LeastBatch", "LeastCache", "Affinity", "SessionPrefix"
    ] = "RoundRobin"
    # HBM KV prefix cache for attention KV blocks. Enabled by default for
    # attention-only models; GDN/linear-attention cache plans force it off
    # because recurrent state cannot be reused from KV prefix blocks alone.
    enable_prefix_cache: bool = True

    # Legacy no-op kept for config compatibility. Session-scoped parked cache
    # was removed; prefix reuse is now handled by the block hash prefix cache.
    gdn_state_cache_slots: int = 0

    # Debug: dump per-request data to a Redis stream (engine-side, so it works
    # for ``dlengine serve`` AND offline generation). Two record kinds keyed
    # on seq_id: ``kind="request"`` (tokenized prompt, for prefix-cache
    # divergence inspection) and ``kind="complete"`` (latency: ttft/tpot/e2e,
    # queue/prefill time, per-chunk prefill latencies, ITL avg/p50/p99). See
    # dlengine.metrics.dump. None/empty disables (zero overhead). "1"/"true" ->
    # redis://127.0.0.1:6379/0; any other value is the Redis URL verbatim. Falls
    # back to env DLENGINE_DUMP_REQUESTS_REDIS.
    dump_requests_redis: Optional[str] = None
    dump_requests_stream: str = "dlengine:requests"
    dump_requests_maxlen: int = 200000

    # parallel config
    attention_tp: int = 1
    attention_sp: int = 1
    attention_dp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1
    ffn_dp: int = 1
    # Pipeline parallelism. Splits the decoder layers into ``pp`` contiguous
    # stages. Each stage owns one full attn/ffn parallel group
    # (dp*sp*tp / dp*ep*tp), so the total number of GPU workers is
    # ``pp * attn_world_size``. Stages exchange hidden states with
    # point-to-point send/recv along the pipeline dimension.
    pp: int = 1
    # Static forward-only prefill pipeline. ``max_num_batched_tokens`` is the
    # per-stage microbatch size; the scheduler admits up to ``pp`` such
    # microbatches per step so adjacent stages can overlap.
    # Maximum queued microbatch RPCs per worker. Zero defaults to min(pp, 16),
    # matching DLSLime's default RPC slot count while filling a PP16 pipeline.
    pp_prefill_pipeline_depth: int = 0

    # runner config
    enforce_eager: bool = False
    use_flashinfer_decode: bool = Field(
        default_factory=lambda: os.environ.get("DLENGINE_USE_FLASHINFER_DECODE", "1")
        == "1"
    )
    use_flashinfer_prefill: bool = Field(
        default_factory=lambda: os.environ.get("DLENGINE_USE_FLASHINFER_PREFILL", "1")
        == "1"
    )
    # Globally disable ``torch.compile`` (run all compiled paths eagerly).
    # ``enforce_eager`` only skips CUDAGraph capture; several layers
    # (rotary embedding, activation, sampler, ...) still wrap their forward
    # with ``torch.compile``, which invokes the inductor/triton backend.
    # On platforms where that backend is not adapted (e.g. PPU) the first
    # compiled call raises ``BackendCompilerFailed``. Set this to run those
    # paths in plain eager mode. Threaded into the worker via the Config
    # object so it reliably reaches Ray actors.
    disable_compile: bool = False
    trust_remote_code: bool = False
    # ``repr=False``: the HF config stores ``dtype`` as a real ``torch.dtype``
    # object, which transformers' ``to_json_string()`` (used by its ``__repr__``)
    # cannot JSON-serialize on transformers <= 4.51.x (only ``torch_dtype`` is
    # stringified there). Including it in the pydantic repr makes any
    # ``repr(Config)`` — e.g. Ray's actor-error formatting — crash with
    # "Object of type dtype is not JSON serializable", masking the real error.
    hf_config: Any = Field(default=None, repr=False)
    cache_plan: Any = Field(default=None, repr=False)
    eos: List[int] = []
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 15000
    num_host_kvcache_blocks: int = 0

    # deployment config
    engine_id: Optional[str] = None
    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"
    host: str = "0.0.0.0"
    # 0 asks the OS to allocate an available service port at bind time.
    port: int = 0

    # Monitoring. When enabled, ``dlengine serve`` exposes Prometheus metrics
    # at /metrics. If docker CLI is available it also starts a local
    # Prometheus/Grafana stack. Prometheus listens on 9090 and Grafana on 3000.
    enable_monitor: bool = False
    monitor_dir: str = "examples/.monitors"
    monitor_scrape_interval: str = "5s"

    dummy_prefill: Optional[bool] = False
    dummy_weight: Optional[bool] = False
    dummy_eplb: Optional[bool] = False

    enable_eplb: Optional[bool] = False

    # control plane config – enabled when ctrl_address is provided
    ctrl_scope: Optional[str] = None
    ctrl_address: Optional[str] = None

    # dist config
    # Ray is the cluster entry point. ``auto`` connects to the local Ray
    # cluster (or the address in RAY_ADDRESS). The torch distributed rendezvous
    # address is discovered from rank 0 after Ray has placed the workers.
    # ``master_address`` remains as an optional legacy override.
    master_address: Optional[str] = None
    ray_address: str = "auto"
    executor_backend: Literal["ray", "dlslime"] = "ray"

    # MTP (Multi-Token Prediction) speculative decoding
    num_speculative_tokens: int = 0  # 0 = disabled, >0 = number of draft tokens

    # NSA sparse attention (V3.2) — enabled by default for models with index_head_dim > 0
    disable_nsa: bool = False
    enable_hisparse: bool = False
    # Number of hot token slots reserved for each active sequence. Total
    # HiSparse device capacity is this value multiplied by max_num_seqs.
    hisparse_device_buffer_size: int = 4096
    hisparse_swap_in_block_size: int = 960

    # Correctness-only fallback for MLA shapes that current FlashMLA wheels do
    # not instantiate (for example 256/256). Disabled by default because it
    # uses slow PyTorch reference attention and disables CUDA graph/FP8 KV cache.
    enable_mla_reference_fallback: bool = False

    # DSv4 compressed-cache pool sizes (tokens per pool, per ratio).
    # 0 means "derive worst case = max_num_seqs * max_model_len / ratio".
    # Set explicitly to a smaller value to save memory when seqs are short.
    dsv4_compressed_pool_pages_ratio4: int = 0
    dsv4_compressed_pool_pages_ratio128: int = 0

    # ------------------------------------------------------------------ #
    # L3 (3FS) tiered KV cache — persists evicted KV blocks to a 3FS mount
    # keyed by block hash, and loads them back on a prefix miss instead of
    # recomputing prefill. Disabled by default (fully inert when off).
    # PoC scope: mode="hybrid", attention_sp == attention_tp == 1.
    # ------------------------------------------------------------------ #
    l3_enable: bool = False
    l3_mountpoint: str = "/3fs/mnt"
    # Directory under the mount for per-block files; defaults to
    # "<mountpoint>/dlengine_l3" when None.
    l3_dir: Optional[str] = None
    # Skip L3 entirely for prompts shorter than this many full blocks
    # (short prefills are cheap to recompute; avoids tiny-IO overhead).
    l3_min_prefix_blocks: int = 1
    # Host staging buffer / ioring depth, in blocks, per worker.
    l3_staging_blocks: int = 8

    # MoE: opt into deep_gemm.fp8_fp4_mega_moe (one-kernel dispatch +
    # per-expert GEMM + activation + combine). Off by default — gated
    # so production can stay on the existing deep_ep low-latency path
    # while we burn in the new path. Requires FP8 weights and Hopper
    # (sm_90+); the experts layer falls back to the old path otherwise.
    use_mega_moe: bool = False
    # Cap on tokens-per-rank for the mega-MoE symmetric buffer. Each
    # routed-expert layer pre-allocates a SymmBuffer sized for this
    # cap; bench/decode num_tokens must stay <= this value or the call
    # raises with a helpful message.
    mega_moe_max_tokens_per_rank: int = 256

    # Per-step host-critical-path timing. Driver-side flag — threaded
    # into RunnerConfig at worker init so each Ray actor sees the same
    # value (env vars don't propagate through Ray runtime_env by
    # default). When ``step_timing=True``, model_runner.run_from_bytes
    # logs a phase breakdown (rpc_in / prep / forward / sample / tail)
    # every ``step_timing_interval`` steps. Off → zero overhead.
    step_timing: bool = False
    step_timing_interval: int = 16
    step_timing_rank: int = 0  # -1 = all ranks

    # Worker-side DLSlime RPC handler timing ("[dlslime worker]" lines:
    # decode_req / forward / encode per decode step). Threaded into
    # RunnerConfig like step_timing. Off by default.
    dlslime_timing: bool = False

    # GPU-idle probe: measures the inter-step GPU-idle gap (end of the
    # previous step's GPU work → start of the next step) with CUDA events,
    # independent of the sync-heavy ``step_timing`` host timer. Logs
    # ``gap``/``busy``/``duty`` every ``step_timing_interval`` steps on
    # ``step_timing_rank``. Off → zero overhead.
    gpu_idle_probe: bool = False

    # Output root for traces delimited by the runtime profiler API.
    profiler_dir: str = "./profiler_res"

    # logging config – override via DLENGINE_LOG_LEVEL env var
    log_level: str = os.getenv("DLENGINE_LOG_LEVEL", "INFO")

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        if not self.engine_id:
            self.engine_id = str(uuid.uuid4())

        # Normalise ctrl_address (add scheme if missing)
        if (
            self.ctrl_address
            and not self.ctrl_address.startswith("http://")
            and not self.ctrl_address.startswith("https://")
        ):
            self.ctrl_address = f"http://{self.ctrl_address}"

        # Register deepseek_v32 model type so that AutoConfig can load
        # DeepSeek-V3.2 checkpoints even when the installed transformers
        # version does not natively support it.
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
                DeepseekV3Config,
            )

            CONFIG_MAPPING.register("deepseek_v32", DeepseekV3Config, exist_ok=True)
        except Exception:
            pass  # transformers version too old for DeepseekV3Config; let it fall through

        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            from dlengine.models.deepseek_v4.configuration_deepseek_v4 import (
                DeepseekV4Config,
            )

            for register in (CONFIG_MAPPING.register, AutoConfig.register):
                try:
                    register("deepseek_v4", DeepseekV4Config, exist_ok=True)
                except TypeError:
                    register("deepseek_v4", DeepseekV4Config)
        except Exception:
            pass

        try:
            self.hf_config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except ValueError:
            config_path = Path(self.model) / "config.json"
            if not config_path.exists():
                raise
            with config_path.open() as f:
                config_dict = json.load(f)
            if config_dict.get("model_type") == "qwen3_5":
                text_config = config_dict.get("text_config")
                self.hf_config = PretrainedConfig(**config_dict)
                if isinstance(text_config, dict):
                    self.hf_config.text_config = PretrainedConfig(**text_config)
            elif config_dict.get("model_type") != "deepseek_v4":
                raise
            else:
                from dlengine.models.deepseek_v4.configuration_deepseek_v4 import (
                    DeepseekV4Config,
                )

                self.hf_config = DeepseekV4Config(**config_dict)

        # Read the unmodified JSON as well as the instantiated HF config. Some
        # Transformers config classes apply attribute aliases while loading and
        # can lose a model-specific value when both alias names are present.
        config_path = Path(self.model) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                raw_config = json.load(f)
        else:
            raw_config, _ = PretrainedConfig.get_config_dict(
                self.model, trust_remote_code=self.trust_remote_code
            )
        apply_hf_config_compatibility_fixes(self.hf_config, raw_config)

        # For VLM models with nested text_config (e.g. Qwen3.5-MoE),
        # flatten text_config attributes into hf_config for uniform access.
        if hasattr(self.hf_config, "text_config"):
            text_cfg = self.hf_config.text_config
            if isinstance(text_cfg, dict):
                text_config_dict = text_cfg
            elif hasattr(text_cfg, "to_dict"):
                text_config_dict = text_cfg.to_dict()
            else:
                text_config_dict = vars(text_cfg)
            for attr, value in text_config_dict.items():
                if attr.startswith("_"):
                    continue
                if not hasattr(self.hf_config, attr):
                    try:
                        setattr(self.hf_config, attr, value)
                    except Exception as e:
                        logger.warning(
                            f"Could not flatten attribute '{attr}' from text_config: {e}"
                        )
            # Explicitly propagate dtype/torch_dtype from text_config
            # (top-level config may have dtype=None while text_config has bfloat16)
            text_dtype = (
                text_cfg.get("dtype")
                if isinstance(text_cfg, dict)
                else getattr(text_cfg, "dtype", None)
            )
            if text_dtype is not None:
                if getattr(self.hf_config, "dtype", None) is None:
                    self.hf_config.__dict__["dtype"] = text_dtype

        for attr in (
            "enable_hisparse",
            "hisparse_device_buffer_size",
            "hisparse_swap_in_block_size",
        ):
            setattr(self.hf_config, attr, getattr(self, attr, None))

        if self.hf_config.architectures[0] in (
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV4ForCausalLM",
            "GlmMoeDsaForCausalLM",
        ):
            if self.hf_config.architectures[0] == "DeepseekV4ForCausalLM":
                assert self.attention_sp == 1
                assert self.ffn_tp == 1
                n_experts = getattr(self.hf_config, "n_routed_experts", None)
                if n_experts is not None:
                    assert n_experts % self.ffn_ep == 0
                # NOTE: flash_mla batched decode path supports CUDAGraph for
                # non-compressed layers. Compressed layers still have per-seq
                # compressor loops that block graph capture. The overall forward
                # is CUDAGraph-safe only when ALL layers' compressor loops are
                # vectorized or when using the eager fallback.
                if not self.enforce_eager:
                    logger.info(
                        "DeepSeek-V4 flash_mla path: CUDAGraph enabled. "
                        "Compressor loops are NOT yet fully vectorized — "
                        "graph capture may fail for compressed layers."
                    )
            else:
                self.kvcache_block_size = 64
            assert self.attention_tp == 1
        else:
            assert self.kvcache_block_size % 64 == 0
            if self.kvcache_block_size % 256 != 0:
                adjusted_block_size = ((self.kvcache_block_size + 255) // 256) * 256
                logger.warning(
                    "kvcache_block_size=%s is incompatible with flash-attn "
                    "release wheels for paged KV decode; adjusting to %s.",
                    self.kvcache_block_size,
                    adjusted_block_size,
                )
                self.kvcache_block_size = adjusted_block_size
            assert 1 <= self.attention_tp <= 8

        if self.attention_sp == 1:
            self.max_num_recv_seqs = 0

        if hasattr(self.hf_config, "max_position_embeddings"):
            self.hf_config.max_position_embeddings = max(
                self.max_model_len, self.hf_config.max_position_embeddings
            )
        else:
            self.hf_config.max_position_embeddings = self.max_model_len

        dtype = getattr(self.hf_config, "dtype", None) or getattr(
            self.hf_config, "torch_dtype", None
        )
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(dtype, None)
        if dtype is not None:
            self.hf_config.dtype = dtype
            self.hf_config.torch_dtype = dtype

        # With chunked prefill, max_num_batched_tokens may be smaller than max_model_len.
        assert self.max_num_batched_tokens >= 1
        if self.host_utilization_per_device < 0:
            raise ValueError("host_utilization_per_device must be >= 0")

        if self.enable_hisparse:
            arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
            if self.attention_sp != 1:
                raise ValueError(
                    "enable_hisparse rejects attention_sp > 1 until "
                    "per-rank block-table and slot semantics are validated."
                )
            if self.attention_tp != 1:
                raise ValueError("enable_hisparse requires attention_tp == 1")
            if self.num_speculative_tokens != 0:
                raise ValueError("enable_hisparse does not support MTP")
            if self.hisparse_device_buffer_size <= 0:
                raise ValueError("hisparse_device_buffer_size must be positive")
            if self.hisparse_swap_in_block_size <= 0:
                raise ValueError("hisparse_swap_in_block_size must be positive")
            if arch in ("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"):
                if self.mode != "decode":
                    raise ValueError(
                        "enable_hisparse NSA/MLA requires mode='decode'; "
                        f"got {self.mode!r}"
                    )
                # A standalone decode engine has no source for its cold tier,
                # so retain the synthetic-cache guard.  A PD decode engine is
                # populated by migration from the ordinary prefill engine.
                if not self.dummy_prefill and not self.ctrl_address:
                    raise ValueError(
                        "enable_hisparse decode requires either dummy_prefill=True "
                        "or ctrl_address for PD cold-cache migration"
                    )
                if self.ctrl_address and self.host_utilization_per_device <= 0:
                    raise ValueError(
                        "PD NSA/MLA HiSparse requires host_utilization_per_device > 0"
                    )
                if self.disable_nsa:
                    raise ValueError(
                        "enable_hisparse requires NSA/indexer to be enabled"
                    )
                if getattr(self.hf_config, "index_head_dim", 0) <= 0:
                    raise ValueError(
                        "enable_hisparse requires DSV3.2 index_head_dim > 0"
                    )
                if getattr(self.hf_config, "index_topk", 0) <= 0:
                    raise ValueError("enable_hisparse requires DSV3.2 index_topk > 0")
            elif arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
                # Gemma4 SWA is a bounded ring by construction. Unlike NSA/MLA,
                # its prefill path can populate the HiSparse hot buffer directly,
                # so do not apply the decode-only guard from the branch above.
                layer_types = getattr(self.hf_config, "layer_types", None) or []
                if "sliding_attention" not in layer_types:
                    raise ValueError(
                        "enable_hisparse for Gemma4 requires sliding_attention layers"
                    )
                if not self.enforce_eager:
                    logger.warning("Gemma4 HiSparse CUDA Graph is experimental")
            else:
                raise ValueError(
                    "enable_hisparse currently supports DeepseekV32ForCausalLM, "
                    "GlmMoeDsaForCausalLM, or Gemma4ForCausalLM; "
                    f"got {arch!r}"
                )
        else:
            arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
            if arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
                head_dim = getattr(self.hf_config, "head_dim", None)
                global_head_dim = getattr(self.hf_config, "global_head_dim", None)
                layer_types = getattr(self.hf_config, "layer_types", None) or []
                has_sliding = "sliding_attention" in layer_types
                has_full = "full_attention" in layer_types
                if (
                    has_sliding
                    and has_full
                    and global_head_dim is not None
                    and head_dim is not None
                    and int(global_head_dim) != int(head_dim)
                ):
                    raise ValueError(
                        "Gemma4 mixes sliding/global attention head_dim "
                        f"({head_dim} vs {global_head_dim}); enable_hisparse is "
                        "required so sliding-window KV uses the hot buffer instead "
                        "of the uniform paged KV cache."
                    )

        # MTP validation
        if self.num_speculative_tokens > 0:
            has_mtp = (
                getattr(self.hf_config, "num_nextn_predict_layers", 0) > 0
                or getattr(self.hf_config, "mtp_num_hidden_layers", 0) > 0
            )
            if not has_mtp:
                raise ValueError(
                    f"num_speculative_tokens={self.num_speculative_tokens} but "
                    f"model does not have MTP layers "
                    f"(num_nextn_predict_layers / mtp_num_hidden_layers not found)"
                )
            # KV-cache reservation for the extra MTP tokens per decode step is
            # handled in the Rust scheduler directly from num_speculative_tokens;
            # nothing to inflate here. The decode loop always runs a single
            # iteration — MTP produces its extra tokens within that one step.

        if self.hf_config.architectures[0] in (
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV4ForCausalLM",
            "GlmMoeDsaForCausalLM",
        ):
            if hasattr(self.hf_config, "num_key_value_heads"):
                self.hf_config.num_key_value_heads = 1

        # L3 (3FS) tiered KV cache — PoC scope guard. The driver-side block
        # manager marks L3 hits as "cached" (recompute skipped), so the
        # worker-side load MUST be guaranteed; restrict to topologies where
        # block_id -> worker routing is unambiguous (single SP group, no KV
        # head sharding). Disable rather than risk loading garbage KV.
        if self.l3_enable:
            if self.attention_sp != 1 or self.attention_tp != 1:
                logger.warning(
                    "l3_enable requires attention_sp==1 and attention_tp==1 "
                    "(PoC scope); got sp=%s tp=%s — disabling L3.",
                    self.attention_sp,
                    self.attention_tp,
                )
                self.l3_enable = False
            elif self.mode not in ("hybrid", "prefill"):
                logger.warning(
                    "l3_enable PoC supports mode in {hybrid, prefill}; got "
                    "%s — disabling L3.",
                    self.mode,
                )
                self.l3_enable = False

        # Pipeline parallelism validation and constraints.
        if self.pp < 1:
            raise ValueError("pp must be >= 1")
        if self.pp_prefill_pipeline_depth < 0:
            raise ValueError("pp_prefill_pipeline_depth must be >= 0")
        if self.pp > 1:
            arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
            supported_pp_archs = (
                "Qwen3ForCausalLM",
                "Qwen3MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "DeepseekV2ForCausalLM",
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
                "GlmMoeDsaForCausalLM",
                "Gemma4ForCausalLM",
                "Gemma4ForConditionalGeneration",
                "DeepseekV4ForCausalLM",
            )
            if arch not in supported_pp_archs:
                raise ValueError(
                    "pp > 1 is currently only supported for "
                    f"{supported_pp_archs}; got {arch!r}"
                )
            num_hidden_layers = getattr(self.hf_config, "num_hidden_layers", None)
            if num_hidden_layers is None or num_hidden_layers < self.pp:
                raise ValueError(
                    f"pp cannot exceed num_hidden_layers ({num_hidden_layers})"
                )
            if arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
                from dlengine.models.pp_utils import get_gemma4_shared_kv_source_start

                source_start = get_gemma4_shared_kv_source_start(self.hf_config)
                if source_start is not None and source_start < self.pp - 1:
                    raise ValueError(
                        "Gemma4 PP cannot split its shared-KV source suffix and "
                        f"needs at least {self.pp - 1} independent prefix layers; "
                        f"got {source_start}"
                    )
            if self.num_speculative_tokens > 0:
                raise ValueError(
                    "pp > 1 does not support MTP (num_speculative_tokens must be 0)"
                )
            if self.enable_hisparse and arch not in (
                "Gemma4ForCausalLM",
                "Gemma4ForConditionalGeneration",
            ):
                raise ValueError(
                    "pp > 1 only supports enable_hisparse for Gemma4; " f"got {arch!r}"
                )
            # A hybrid DLSLime executor still needs ctrl_address for RPC agent
            # discovery; ctrl_address alone does not imply PD disaggregation.
            # The prefill/decode roles are what make an engine disaggregated.
            # PD disaggregation supports a PP prefill engine paired with pp=1
            # decode engines: the decode side maps each global layer to the
            # prefill stage owning it during KV migration. A PP decode engine
            # is not supported yet.
            if self.mode == "decode":
                raise ValueError(
                    "pp > 1 is not supported for mode='decode' "
                    "(use PP prefill + pp=1 decode)"
                )
            # Stage boundaries send/recv hidden states eagerly; CUDA graph
            # capture across a P2P boundary is not wired yet.
            if not self.enforce_eager:
                logger.info(
                    "pp > 1: forcing enforce_eager=True (CUDA graph capture "
                    "across pipeline stages is not supported)"
                )
                self.enforce_eager = True

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

    @property
    def world_size(self):
        """Total number of GPU workers across all pipeline stages."""
        return self.pp * self.attn_world_size

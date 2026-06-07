import json
import os
from pathlib import Path
from typing import Any, List, Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig, PretrainedConfig

from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class Config(BaseModel):

    model_config = {
        "arbitrary_types_allowed": True,
    }

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
    eos: List[int] = []
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 15000

    # deployment config
    engine_id: Optional[str] = None
    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"
    host: str = "0.0.0.0"
    port: int = 5000

    dummy_prefill: Optional[bool] = False
    dummy_weight: Optional[bool] = False
    dummy_eplb: Optional[bool] = False

    enable_eplb: Optional[bool] = False

    # control plane config – enabled when ctrl_address is provided
    ctrl_scope: Optional[str] = None
    ctrl_address: Optional[str] = None

    # dist config
    master_address: str = "127.0.0.1:6006"
    ray_address: str = "127.0.0.1:6379"
    executor_backend: Literal["ray", "dlslime"] = "ray"

    # MTP (Multi-Token Prediction) speculative decoding
    num_speculative_tokens: int = 0  # 0 = disabled, >0 = number of draft tokens

    # NSA sparse attention (V3.2) — enabled by default for models with index_head_dim > 0
    disable_nsa: bool = False

    # DSv4 compressed-cache pool sizes (tokens per pool, per ratio).
    # 0 means "derive worst case = max_num_seqs * max_model_len / ratio".
    # Set explicitly to a smaller value to save memory when seqs are short.
    dsv4_compressed_pool_pages_ratio4: int = 0
    dsv4_compressed_pool_pages_ratio128: int = 0

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

    # profiler
    enable_profiler: bool = False
    profiler_start_step: int = 34
    profiling_step: int = 8
    profiler_forward_per_step: int = 2
    profiler_dir: str = "./profiler_res"

    # logging config – override via NANODEPLOY_LOG_LEVEL env var
    log_level: str = os.getenv("NANODEPLOY_LOG_LEVEL", "INFO")

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
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

            from nanodeploy.models.deepseek_v4.configuration_deepseek_v4 import (
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
            if config_dict.get("model_type") != "deepseek_v4":
                raise
            from nanodeploy.models.deepseek_v4.configuration_deepseek_v4 import (
                DeepseekV4Config,
            )

            self.hf_config = DeepseekV4Config(**config_dict)

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

        if self.hf_config.architectures[0] in (
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
                assert self.kvcache_block_size == 64
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
            if self.loop_count != 1:
                raise ValueError(
                    f"MTP requires loop_count=1, got loop_count={self.loop_count}"
                )
            # Inflate loop_count so the scheduler pre-allocates enough KV cache
            # blocks for the extra MTP tokens per decode step.
            # The actual decode loop still runs only the original loop_count
            # iterations; MTP generates the extra tokens within a single step.
            self._mtp_original_loop_count = self.loop_count
            self.loop_count = self.loop_count + self.num_speculative_tokens + 1

        if self.hf_config.architectures[0] in (
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV4ForCausalLM",
            "GlmMoeDsaForCausalLM",
        ):
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

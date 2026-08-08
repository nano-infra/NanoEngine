from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Callable, Mapping


DEEPEP_ENV_NAMES = (
    "DEEPEP_SMS",
    "DEEPEP_MAX_TOKENS_PER_RANK",
    "DEEPEP_ENABLE_MNNVL",
    "NVSHMEM_QP_DEPTH",
)

DECODE_WORKER_PASSTHROUGH_ENV_NAMES = (
    "DG_PRINT_CONFIGS",
    "DG_JIT_DEBUG",
    "NANODEPLOY_MOE_GEMM_DEBUG",
    "NANODEPLOY_MOE_GEMM_DEBUG_RANKS",
    "NANODEPLOY_MOE_GEMM_DEBUG_LAYERS",
    "NANODEPLOY_MOE_GEMM_DEBUG_GEMMS",
    "NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS",
    "NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS",
)

EXPECTED_BACKEND_VERSIONS = {
    "deep_gemm": "2.3.0+477618c",
    "deep_ep": "1.2.1+73b6ea4",
}

DEEP_GEMM_REQUIRED_SYMBOLS = (
    "ceil_div",
    "fp8_gemm_nt",
    "get_m_alignment_for_contiguous_layout",
    "get_mk_alignment_for_contiguous_layout",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "fp8_m_grouped_gemm_nt_masked",
    "set_num_sms",
    "transform_sf_into_required_layout",
)

DEEP_EP_REQUIRED_SYMBOLS = (
    "Buffer",
    "Config",
    "EventOverlap",
    "topk_idx_t",
)

DEEP_EP_BUFFER_REQUIRED_SYMBOLS = (
    "set_num_sms",
    "capture",
    "destroy",
    "get_low_latency_rdma_size_hint",
    "get_dispatch_config",
    "get_combine_config",
    "get_dispatch_layout",
    "dispatch",
    "combine",
    "clean_low_latency_buffer",
    "low_latency_dispatch",
    "low_latency_combine",
)

DEFAULT_NVSHMEM_QP_DEPTH = 1024


@dataclass(frozen=True)
class DecodeDeepEPConfig:
    num_sms: int
    max_tokens_per_rank: int
    enable_mnnvl: bool
    nvshmem_qp_depth: int
    mode: str = "auto"

    def worker_env(self) -> dict[str, str]:
        return {
            "DEEPEP_SMS": str(self.num_sms),
            "DEEPEP_MAX_TOKENS_PER_RANK": str(self.max_tokens_per_rank),
            "DEEPEP_ENABLE_MNNVL": "1" if self.enable_mnnvl else "0",
            "NVSHMEM_QP_DEPTH": str(self.nvshmem_qp_depth),
        }

    def fingerprint_payload(self) -> dict[str, str]:
        return {**self.worker_env(), "DEEPEP_MODE": self.mode}


def _parse_integer(name: str, raw_value: str) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}") from exc


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _resolve_nvshmem_qp_depth(
    max_tokens_per_rank: int, source: Mapping[str, str]
) -> int:
    minimum_depth = 2 * (max_tokens_per_rank + 1)
    raw_depth = source.get("NVSHMEM_QP_DEPTH")
    if raw_depth is None:
        return max(
            DEFAULT_NVSHMEM_QP_DEPTH,
            _next_power_of_two(minimum_depth),
        )

    depth = _parse_integer("NVSHMEM_QP_DEPTH", raw_depth)
    if depth <= 0:
        raise ValueError(
            f"NVSHMEM_QP_DEPTH must be a positive integer; got {depth}"
        )
    if depth < minimum_depth:
        raise ValueError(
            "NVSHMEM_QP_DEPTH must be at least "
            "2 * (DEEPEP_MAX_TOKENS_PER_RANK + 1); "
            f"got {depth} < {minimum_depth} for "
            f"max_tokens_per_rank={max_tokens_per_rank}"
        )
    return depth


def resolve_decode_deepep_config(
    max_num_seqs: int,
    environ: Mapping[str, str] | None = None,
) -> DecodeDeepEPConfig:
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")

    source = os.environ if environ is None else environ
    num_sms = _parse_integer("DEEPEP_SMS", source.get("DEEPEP_SMS", "16"))
    if num_sms <= 0 or num_sms % 2 != 0:
        raise ValueError(
            f"DEEPEP_SMS must be a positive even integer; got {num_sms}"
        )

    max_tokens_per_rank = _parse_integer(
        "DEEPEP_MAX_TOKENS_PER_RANK",
        source.get("DEEPEP_MAX_TOKENS_PER_RANK", str(max_num_seqs)),
    )
    if max_tokens_per_rank <= 0:
        raise ValueError(
            "DEEPEP_MAX_TOKENS_PER_RANK must be positive; "
            f"got {max_tokens_per_rank}"
        )
    if max_tokens_per_rank < max_num_seqs:
        raise ValueError(
            "DEEPEP_MAX_TOKENS_PER_RANK must cover max_num_seqs; "
            f"got {max_tokens_per_rank} < {max_num_seqs}"
        )

    enable_mnnvl_value = _parse_integer(
        "DEEPEP_ENABLE_MNNVL", source.get("DEEPEP_ENABLE_MNNVL", "0")
    )
    if enable_mnnvl_value not in {0, 1}:
        raise ValueError(
            "DEEPEP_ENABLE_MNNVL must be 0 or 1; "
            f"got {enable_mnnvl_value}"
        )

    nvshmem_qp_depth = _resolve_nvshmem_qp_depth(
        max_tokens_per_rank, source
    )

    mode = source.get("DEEPEP_MODE", "auto").strip().lower()
    if mode != "auto":
        raise ValueError(
            "native DeepEP decode backend requires DEEPEP_MODE=auto; "
            f"got {mode!r}"
        )

    return DecodeDeepEPConfig(
        num_sms=num_sms,
        max_tokens_per_rank=max_tokens_per_rank,
        enable_mnnvl=bool(enable_mnnvl_value),
        nvshmem_qp_depth=nvshmem_qp_depth,
    )


def build_decode_backend_worker_env(
    max_num_seqs: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    worker_env = resolve_decode_deepep_config(max_num_seqs, source).worker_env()
    for name in DECODE_WORKER_PASSTHROUGH_ENV_NAMES:
        if name in source:
            worker_env[name] = source[name]
    return worker_env


def _read_version(
    distribution: str,
    version_getter: Callable[[str], str],
) -> str:
    try:
        return version_getter(distribution)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def _missing_symbols(owner: object, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not hasattr(owner, name)]


def validate_decode_backend_compat(
    rank: int,
    *,
    version_getter: Callable[[str], str] = metadata.version,
    module_importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, str]:
    actual_versions = {
        distribution: _read_version(distribution, version_getter)
        for distribution in EXPECTED_BACKEND_VERSIONS
    }
    errors = [
        f"{distribution}: expected {expected}, got "
        f"{actual_versions[distribution]}"
        for distribution, expected in EXPECTED_BACKEND_VERSIONS.items()
        if actual_versions[distribution] != expected
    ]

    modules: dict[str, ModuleType] = {}
    for name in EXPECTED_BACKEND_VERSIONS:
        try:
            modules[name] = module_importer(name)
        except Exception as exc:
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")

    deep_gemm = modules.get("deep_gemm")
    if deep_gemm is not None:
        missing = _missing_symbols(deep_gemm, DEEP_GEMM_REQUIRED_SYMBOLS)
        if missing:
            errors.append(f"deep_gemm missing symbols: {', '.join(missing)}")

    deep_ep = modules.get("deep_ep")
    if deep_ep is not None:
        missing = _missing_symbols(deep_ep, DEEP_EP_REQUIRED_SYMBOLS)
        if missing:
            errors.append(f"deep_ep missing symbols: {', '.join(missing)}")

        buffer_type = getattr(deep_ep, "Buffer", None)
        if buffer_type is not None:
            missing = _missing_symbols(
                buffer_type, DEEP_EP_BUFFER_REQUIRED_SYMBOLS
            )
            if missing:
                errors.append(
                    f"deep_ep.Buffer missing symbols: {', '.join(missing)}"
                )

    if errors:
        details = "; ".join(errors)
        raise RuntimeError(
            f"rank {rank} decode backend compatibility check failed: {details}"
        )
    return actual_versions


def validate_deepseek_decode_contract(hf_config: object, ep_size: int) -> None:
    architectures = getattr(hf_config, "architectures", ()) or ()
    if not architectures or architectures[0] != "DeepseekV3ForCausalLM":
        return

    num_experts = int(getattr(hf_config, "n_routed_experts"))
    if num_experts % ep_size != 0:
        raise ValueError(
            f"n_routed_experts={num_experts} must be divisible by ep_size={ep_size}"
        )
    top_k = int(getattr(hf_config, "num_experts_per_tok"))
    if top_k != 8:
        raise ValueError(
            f"DeepSeek/Kimi decode backend requires num_experts_per_tok=8; got {top_k}"
        )
    quantization_config = getattr(hf_config, "quantization_config", None) or {}
    block_size = list(quantization_config.get("weight_block_size", ()))
    if block_size != [128, 128]:
        raise ValueError(
            "DeepSeek/Kimi decode backend requires "
            f"weight_block_size=[128, 128]; got {block_size}"
        )

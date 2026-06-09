"""Hopper-specific (sm_90) kernel re-exports."""

from nanodeploy_kernel.gpu_generic.moe import *  # noqa: F401, F403
from nanodeploy_kernel.hopper.block_gemm_fp8 import deep_gemm_fp8, quant_fp8_tma
from nanodeploy_kernel.hopper.fp8 import (
    per_token_group_quant_fp8,
    silu_and_mul_masked_post_quant_fwd,
)
from nanodeploy_kernel.hopper.fused_moe_v3 import fused_moe_v3, fused_moe_v3_bf16

__all__ = [
    "deep_gemm_fp8",
    "quant_fp8_tma",
    "per_token_group_quant_fp8",
    "silu_and_mul_masked_post_quant_fwd",
    "fused_moe_v3",
    "fused_moe_v3_bf16",
]

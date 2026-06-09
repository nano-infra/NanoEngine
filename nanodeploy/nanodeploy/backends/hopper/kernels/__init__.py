"""Hopper-specific kernel re-exports.

The kernel implementations live in ``nanodeploy/kernels/`` for now.
This package makes them accessible via the ``backends.hopper.kernels``
namespace so that Hopper layer code can import from here and the
namespace clearly communicates hardware ownership.
"""

from nanodeploy.backends.hopper.kernels.block_gemm_fp8 import (
    deep_gemm_fp8,
    quant_fp8_tma,
)
from nanodeploy.backends.hopper.kernels.fp8 import (
    per_token_group_quant_fp8,
    silu_and_mul_masked_post_quant_fwd,
)
from nanodeploy.backends.hopper.kernels.fused_moe_v3 import (
    fused_moe_v3,
    fused_moe_v3_bf16,
)
from nanodeploy.backends.gpu_generic.kernels.moe import *  # noqa: F401, F403

__all__ = [
    "deep_gemm_fp8",
    "quant_fp8_tma",
    "per_token_group_quant_fp8",
    "silu_and_mul_masked_post_quant_fwd",
    "fused_moe_v3",
    "fused_moe_v3_bf16",
]

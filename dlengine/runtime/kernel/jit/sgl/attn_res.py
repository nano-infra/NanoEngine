"""Blackwell fused attention-residual aggregation for Kimi K3."""

from __future__ import annotations

import os

import torch

from .utils import cache_once, load_jit, make_cpp_args

_DIM = 7168
_MAX_BANK_ROWS = 8
_CONFIG = {
    1: (4, 1, 200),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


@cache_once
def _jit_module(chunk_rows: int, occupancy: int, consumer_regs: int):
    args = make_cpp_args(_DIM, _MAX_BANK_ROWS, chunk_rows, occupancy, consumer_regs)
    cls = f"AttnResFusedTmaKernel<{args}>"
    # tcgen05/TMEM instructions require the current GPU's architecture-specific
    # target, matching the SGL_CUDA_ARCH macro injected by load_jit().
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())

    key = "TVM_FFI_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = f"{major}.{minor}a"
    try:
        return load_jit(
            "kimi_k3_attn_res_fused_tma",
            *args,
            cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
            cuda_wrappers=[("run", f"{cls}::run")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def fused_attention_residual_tma(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    *,
    write_prefix: bool = False,
) -> torch.Tensor:
    if torch.cuda.get_device_capability(prefix.device)[0] < 10:
        raise RuntimeError("K3 fused attention residual requires SM100 or newer")
    if prefix.shape[1] != _DIM or not 1 <= nvb <= _MAX_BANK_ROWS:
        raise ValueError("unsupported K3 attention-residual shape")
    output = torch.empty_like(prefix)
    _jit_module(*_CONFIG[nvb]).run(
        prefix,
        bank,
        score_weight,
        output_weight,
        output,
        nvb,
        float(eps),
        write_prefix,
    )
    return output

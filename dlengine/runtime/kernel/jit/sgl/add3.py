"""CUDA JIT elementwise three-way BF16 add."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_VEC_ELEMS = 16


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "add3_bf16",
        *args,
        cuda_files=["elementwise/add3.cuh"],
        cuda_wrappers=[("run", f"sglang::Add3Kernel<{args}>::launch")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.shape == b.shape == c.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
        and a.is_cuda
        and a.numel() > 0
        and a.numel() % _MAX_VEC_ELEMS == 0
    )


def add3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    prefetch_bc: bool = False,
) -> torch.Tensor:
    """Return ``bf16(bf16(a + b) + c)`` with the unfused rounding order."""
    if not covered(a, b, c):
        raise ValueError("add3 requires same-shape contiguous CUDA BF16 tensors")
    if out is None:
        out = torch.empty_like(a)
    _jit_module().run(a.view(-1), b.view(-1), c.view(-1), out.view(-1), prefetch_bc)
    return out

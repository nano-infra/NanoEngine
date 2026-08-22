"""Fused sigmoid gate multiplication for Kimi K3 MLA output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_mul_kernel(x, gate, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    xv = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, xv * tl.sigmoid(gv), mask=mask)


def sigmoid_mul_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not gate.is_cuda:
        raise RuntimeError("fused sigmoid multiply requires CUDA tensors")
    if x.shape != gate.shape or x.device != gate.device:
        raise ValueError("x and gate must have matching shapes and devices")
    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    block = 256
    _sigmoid_mul_kernel[(triton.cdiv(x.numel(), block),)](
        x, gate, out, n=x.numel(), BLOCK=block, num_warps=4
    )
    return out

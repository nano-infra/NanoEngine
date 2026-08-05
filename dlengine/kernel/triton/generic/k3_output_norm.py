"""Strided Kimi K3 sigmoid-gated RMSNorm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_output_norm_kernel(
    x, gate, weight, out,
    sx_b: tl.constexpr, sx_h: tl.constexpr,
    sg_b: tl.constexpr, sg_h: tl.constexpr,
    so_b: tl.constexpr, so_h: tl.constexpr,
    heads: tl.constexpr, dim: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // heads
    head = row % heads
    cols = tl.arange(0, BLOCK)
    mask = cols < dim
    xv = tl.load(x + batch * sx_b + head * sx_h + cols, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(gate + batch * sg_b + head * sg_h + cols, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(xv * xv, axis=0) / dim + eps)
    tl.store(out + batch * so_b + head * so_h + cols, xv * inv * wv * tl.sigmoid(gv), mask=mask)


def k3_output_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.ndim != 3 or gate.shape != x.shape or x.shape[-1] != weight.numel():
        raise ValueError("K3 output norm expects matching [B,H,D] inputs")
    out = torch.empty_like(x)
    heads, dim = x.shape[1:]
    block = triton.next_power_of_2(dim)
    _k3_output_norm_kernel[(x.shape[0] * heads,)](
        x, gate, weight, out,
        x.stride(0), x.stride(1), gate.stride(0), gate.stride(1),
        out.stride(0), out.stride(1),
        heads, dim, eps, BLOCK=block, num_warps=4,
    )
    return out

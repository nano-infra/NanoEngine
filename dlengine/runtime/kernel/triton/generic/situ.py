"""Fused SiTU-GLU activation used by Kimi K3."""

import torch
import triton
import triton.language as tl


@triton.jit
def _situ_and_mul_kernel(
    x,
    out,
    n_elements: tl.constexpr,
    width: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets // width
    cols = offsets - rows * width
    input_offsets = rows * (2 * width) + cols
    gate = tl.load(x + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + input_offsets + width, mask=mask, other=0.0).to(tl.float32)
    gate = beta * tl.extra.libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_CLAMP:
        up = linear_beta * tl.extra.libdevice.tanh(up / linear_beta)
    tl.store(out + offsets, gate * up, mask=mask)


def situ_and_mul_triton(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("fused SiTU requires a CUDA tensor")
    if not x.is_contiguous() or x.shape[-1] % 2:
        raise ValueError("fused SiTU requires contiguous input with an even last dim")
    width = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    n_elements = out.numel()
    block = 256
    _situ_and_mul_kernel[(triton.cdiv(n_elements, block),)](
        x,
        out,
        n_elements=n_elements,
        width=width,
        beta=float(beta),
        linear_beta=0.0 if linear_beta is None else float(linear_beta),
        HAS_LINEAR_CLAMP=linear_beta is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out

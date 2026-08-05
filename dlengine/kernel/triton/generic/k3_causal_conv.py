"""Packed Kimi K3 one-token causal-convolution state update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_causal_conv_update_kernel(
    x, state, weight, slots, out,
    stride_x_b: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_d: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    block = tl.program_id(1)
    d = block * BLOCK + tl.arange(0, BLOCK)
    mask = d < D
    slot = tl.load(slots + b).to(tl.int64)
    state_base = state + slot * stride_state_slot + d * stride_state_d
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in tl.static_range(0, W - 1):
        value = tl.load(state_base + (i + 1), mask=mask, other=0.0)
        tl.store(state_base + i, value, mask=mask)
        coeff = tl.load(weight + d * stride_weight_d + i, mask=mask, other=0.0)
        acc += value.to(tl.float32) * coeff.to(tl.float32)
    value = tl.load(x + b * stride_x_b + d, mask=mask, other=0.0)
    tl.store(state_base + (W - 1), value, mask=mask)
    coeff = tl.load(weight + d * stride_weight_d + (W - 1), mask=mask, other=0.0)
    acc += value.to(tl.float32) * coeff.to(tl.float32)
    activated = acc * tl.sigmoid(acc)
    tl.store(out + b * stride_out_b + d, activated, mask=mask)


def k3_causal_conv_update(
    x: torch.Tensor,
    state_pool: torch.Tensor,
    weight: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Update indexed states in-place and return SiLU convolution output."""
    if x.ndim != 2 or state_pool.ndim != 3 or weight.ndim != 2:
        raise ValueError("expected x [B,D], state_pool [S,D,W], weight [D,W]")
    batch, dim = x.shape
    if state_pool.shape[1:] != weight.shape or weight.shape[0] != dim:
        raise ValueError("causal convolution shapes do not match")
    out = torch.empty_like(x)
    block = 256
    _k3_causal_conv_update_kernel[(batch, triton.cdiv(dim, block))](
        x, state_pool, weight, slots, out,
        x.stride(0), state_pool.stride(0), state_pool.stride(1),
        weight.stride(0), out.stride(0),
        D=dim, W=weight.shape[1], BLOCK=block,
        num_warps=4,
    )
    return out

"""Triton Kimi-K3 attention-residual score and mixture kernels.

This is the graph-safe local equivalent of SGLang's two-stage fallback: score
all frozen residual rows in parallel, then softmax and combine by hidden
chunks.  The following NanoDeploy RMSNorm remains a separate fused kernel.
"""

import torch
import triton
import triton.language as tl

_BLOCK_H = 1024
_MAX_ROWS = 16


@triton.jit
def _score_kernel(
    prefix,
    bank,
    cw,
    scores,
    nvb: tl.constexpr,
    eps: tl.constexpr,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    row = tl.program_id(1)
    sumsq = 0.0
    dot = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        if row < nvb:
            value = tl.load(bank + token * stride_bm + row * stride_bb + h).to(
                tl.float32
            )
        else:
            value = tl.load(prefix + token * stride_pm + h).to(tl.float32)
        weight = tl.load(cw + h)
        sumsq += tl.sum(value * value)
        dot += tl.sum(value * weight)
    tl.store(scores + token * stride_sm + row, dot * tl.rsqrt(sumsq / H + eps))


@triton.jit
def _combine_kernel(
    prefix,
    bank,
    scores,
    output,
    nvb: tl.constexpr,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = tl.arange(0, MAX_ROWS)
    mask = rows <= nvb
    logits = tl.load(scores + token * stride_sm + rows, mask=mask, other=float("-inf"))
    logits = logits - tl.max(logits, axis=0)
    probs = tl.where(mask, tl.exp(logits), 0.0)
    probs = probs / tl.sum(probs, axis=0)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for row in tl.static_range(0, nvb + 1):
        if row < nvb:
            value = tl.load(bank + token * stride_bm + row * stride_bb + h).to(
                tl.float32
            )
        else:
            value = tl.load(prefix + token * stride_pm + h).to(tl.float32)
        probability = tl.sum(tl.where(rows == row, probs, 0.0), axis=0)
        acc += probability * value
    tl.store(output + token * stride_om + h, acc)


def fused_attention_residual(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    combined_score_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return the pre-output-norm mixture for ``bank[:nvb] + prefix``."""
    tokens, hidden = prefix.shape
    if hidden % _BLOCK_H:
        raise ValueError(f"K3 fused residual requires hidden multiple of {_BLOCK_H}")
    if not 1 <= nvb < _MAX_ROWS:
        raise ValueError(f"K3 residual bank rows must be in [1, {_MAX_ROWS - 1}]")
    scores = torch.empty((tokens, _MAX_ROWS), dtype=torch.float32, device=prefix.device)
    _score_kernel[(tokens, nvb + 1)](
        prefix,
        bank,
        combined_score_weight,
        scores,
        nvb,
        eps,
        prefix.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=hidden,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )
    output = torch.empty_like(prefix)
    _combine_kernel[(tokens, hidden // _BLOCK_H)](
        prefix,
        bank,
        scores,
        output,
        nvb,
        prefix.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        output.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return output

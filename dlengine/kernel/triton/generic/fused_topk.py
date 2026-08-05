"""Fused MoE gating: softmax + top-k + (optional) renormalisation.

Replaces the eager chain

    probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    weights /= weights.sum(dim=-1, keepdim=True)        # norm_topk_prob
    weights = weights.to(hidden_dtype)                  # + fp32 cast in dispatcher

(softmax_warp + reduce + gatherTopK + bitonicSort + div + 2 casts, ~5-6
kernel launches per MoE layer) with a single Triton kernel that emits the
int64 ids and float32 weights the DeepEP dispatchers consume directly.

CUDA-Graph safe: static shapes, no host sync.
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_EXPERTS = 1024
MAX_FUSED_TOP_K = 32


@triton.jit
def _fused_softmax_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_lm,
    stride_wm,
    stride_im,
    num_experts,
    RENORM: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < num_experts

    logits = tl.load(
        logits_ptr + row * stride_lm + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)

    # Full softmax over all experts (matches the eager reference, which
    # renormalises *softmax probabilities* over the selected k).
    row_max = tl.max(logits, axis=0)
    exp = tl.exp(logits - row_max)
    exp = tl.where(mask, exp, 0.0)
    probs = exp / tl.sum(exp, axis=0)

    # Iterative top-k: k is small (8 for Qwen3-MoE), so an unrolled
    # argmax loop beats a full sort. Ties resolve to the lowest index,
    # like torch.topk on contiguous CUDA tensors.
    sel_p = ()
    work = probs
    for k in tl.static_range(TOP_K):
        p = tl.max(work, axis=0)
        idx = tl.argmax(work, axis=0)
        sel_p = sel_p + (p,)
        tl.store(ids_ptr + row * stride_im + k, idx.to(tl.int64))
        work = tl.where(offs == idx, -1.0, work)

    denom = 1.0
    if RENORM:
        denom = sel_p[0]
        for k in tl.static_range(1, TOP_K):
            denom += sel_p[k]

    for k in tl.static_range(TOP_K):
        w = sel_p[k] / denom if RENORM else sel_p[k]
        tl.store(weights_ptr + row * stride_wm + k, w)


def can_use_fused_softmax_topk(router_logits: torch.Tensor, top_k: int) -> bool:
    if not router_logits.is_cuda:
        return False
    if router_logits.dim() != 2:
        return False
    if router_logits.stride(-1) != 1:
        return False
    if router_logits.shape[-1] > MAX_FUSED_EXPERTS:
        return False
    if not (0 < top_k <= min(MAX_FUSED_TOP_K, router_logits.shape[-1])):
        return False
    if router_logits.shape[0] == 0:
        return False
    return True


def fused_softmax_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax + top-k gating in one launch.

    Returns ``(topk_weights float32 [T, k], topk_ids int64 [T, k])`` —
    the dtypes DeepEP dispatch/combine require, so the downstream
    ``.to(torch.int64)`` / ``.to(torch.float32)`` casts become no-ops.
    """
    num_tokens, num_experts = router_logits.shape
    topk_weights = torch.empty(
        num_tokens, top_k, dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        num_tokens, top_k, dtype=torch.int64, device=router_logits.device
    )

    _fused_softmax_topk_kernel[(num_tokens,)](
        router_logits,
        topk_weights,
        topk_ids,
        router_logits.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        num_experts,
        RENORM=renormalize,
        TOP_K=top_k,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=4,
        num_stages=1,
    )
    return topk_weights, topk_ids


@triton.jit
def _fused_sigmoid_biased_topk_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_lm,
    stride_wm,
    stride_im,
    num_experts,
    RENORM: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < num_experts
    logits = tl.load(
        logits_ptr + row * stride_lm + offs, mask=mask, other=0.0
    ).to(tl.float32)
    probs = tl.where(mask, tl.sigmoid(logits), 0.0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    selection_scores = probs + bias
    # CUDA-graph warmup may feed dummy values containing NaNs. Triton's
    # argmax does not promise that a NaN comparison cannot select one of the
    # padded lanes, which would turn into an out-of-range MegaMoE expert id.
    # Keep routing ids valid even for such dummy inputs; real finite inputs are
    # unchanged.
    valid_score = selection_scores == selection_scores
    work = tl.where(mask & valid_score, selection_scores, float("-inf"))

    selected = ()
    for k in tl.static_range(TOP_K):
        idx = tl.argmax(work, axis=0)
        probability = tl.sum(tl.where(offs == idx, probs, 0.0), axis=0)
        selected = selected + (probability,)
        tl.store(ids_ptr + row * stride_im + k, idx.to(tl.int64))
        work = tl.where(offs == idx, float("-inf"), work)

    denom = 1.0
    if RENORM:
        denom = selected[0]
        for k in tl.static_range(1, TOP_K):
            denom += selected[k]
    for k in tl.static_range(TOP_K):
        value = selected[k] / denom if RENORM else selected[k]
        tl.store(weights_ptr + row * stride_wm + k, value)


def fused_sigmoid_biased_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K3 router: sigmoid + correction-biased selection + weight gather."""
    if not can_use_fused_softmax_topk(router_logits, top_k):
        raise ValueError("K3 fused router requires a non-empty contiguous CUDA matrix")
    if correction_bias.shape != (router_logits.shape[1],):
        raise ValueError("correction bias must have one value per expert")
    num_tokens, num_experts = router_logits.shape
    weights = torch.empty(
        num_tokens, top_k, dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty(
        num_tokens, top_k, dtype=torch.int64, device=router_logits.device
    )
    _fused_sigmoid_biased_topk_kernel[(num_tokens,)](
        router_logits,
        correction_bias,
        weights,
        ids,
        router_logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        num_experts,
        RENORM=renormalize,
        TOP_K=top_k,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=4,
        num_stages=1,
    )
    return weights, ids

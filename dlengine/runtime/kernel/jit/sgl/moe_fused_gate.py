"""Vendored SGLang fused MoE gate for single-group decode routing."""

from __future__ import annotations

import torch

from .utils import cache_once, load_jit

_SCORING_FUNC_MAP = {"sigmoid": 0, "sqrtsoftplus": 1}


@cache_once
def _jit_moe_fused_gate_module():
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


def moe_fused_gate(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select and normalize routed experts in one CUDA kernel."""
    scoring_func_id = _SCORING_FUNC_MAP.get(scoring_func.lower())
    if scoring_func_id is None:
        raise ValueError(f"unsupported scoring_func: {scoring_func}")
    if router_logits.dtype != torch.float32 or correction_bias.dtype != torch.float32:
        raise TypeError("fused MoE gate expects float32 logits and correction bias")
    if router_logits.ndim != 2 or correction_bias.shape != router_logits.shape[1:]:
        raise ValueError(
            "fused MoE gate expects logits [tokens, experts] and bias [experts]"
        )

    weights = torch.empty(
        router_logits.shape[0], topk, dtype=torch.float32, device=router_logits.device
    )
    indices = torch.empty(
        router_logits.shape[0], topk, dtype=torch.int32, device=router_logits.device
    )
    _jit_moe_fused_gate_module().moe_fused_gate(
        router_logits,
        correction_bias,
        weights,
        indices,
        topk,
        scoring_func_id,
        0,  # shared experts remain on their overlapped dense path
        renormalize,
        float(routed_scaling_factor),
        True,  # match DLEngine: routed scaling is baked into topk weights
    )
    return indices, weights

#!/usr/bin/env python3
"""Reproduce the K3 shape-derived capacity, FLOP, and traffic estimates."""

from __future__ import annotations

import json
from pathlib import Path

HIDDEN = 7168
HEADS = 96
KDA_WIDTH = 12288
KDA_HEAD_DIM = 128
MLA_QK_DIM = 192
MLA_V_DIM = 128
MLA_LATENT = 512
MLA_LAYERS = 24
KDA_LAYERS = 69
MOE_LAYERS = 92
TOP_K = 16
EXPERT_LATENT = 3584
EXPERT_INTERMEDIATE = 3072
TOTAL_CONTEXT = 1_048_576
FRESH = 16_384


def main() -> None:
    kda_elements = (
        5 * KDA_WIDTH * HIDDEN + HEADS * HIDDEN + 128 * HIDDEN + KDA_WIDTH * 128
    )
    mla_elements = (
        2 * KDA_WIDTH * HIDDEN
        + 576 * HIDDEN
        + 24576 * MLA_LATENT
        + 1536 * HIDDEN
        + 18432 * 1536
    )
    routed_flops = 2 * TOP_K * 3 * EXPERT_LATENT * EXPERT_INTERMEDIATE
    shared_flops = 2 * 3 * HIDDEN * (2 * EXPERT_INTERMEDIATE)
    latent_flops = 2 * 2 * HIDDEN * EXPERT_LATENT
    router_flops = 2 * 896 * HIDDEN
    attention_flops = 2 * HEADS * (MLA_QK_DIM + MLA_V_DIM) * FRESH * TOTAL_CONTEXT
    result = {
        "shape": {
            "hidden": HIDDEN,
            "heads": HEADS,
            "total_context": TOTAL_CONTEXT,
            "fresh_tokens": FRESH,
        },
        "gflop_per_active_token": {
            "kda_projection": 2 * kda_elements / 1e9,
            "kda_recurrence_approx": 4 * HEADS * KDA_HEAD_DIM**2 / 1e9,
            "mla_projection": 2 * mla_elements / 1e9,
            "moe_top16_and_shared": (
                routed_flops + shared_flops + latent_flops + router_flops
            )
            / 1e9,
        },
        "cached_prefill_1m_16k": {
            "mla_attention_pflop_per_layer": attention_flops / 1e15,
            "mla_attention_pflop_all_layers": attention_flops * MLA_LAYERS / 1e15,
            "expanded_kv_gib_per_layer": (
                TOTAL_CONTEXT * HEADS * (MLA_QK_DIM + MLA_V_DIM) * 2 / 2**30
            ),
        },
        "persistent_cache_at_1m_gib": {
            "mla_raw_fp8_all_layers": TOTAL_CONTEXT * 576 * MLA_LAYERS / 2**30,
            "mla_bf16_all_layers": TOTAL_CONTEXT * 1152 * MLA_LAYERS / 2**30,
            "kda_slot_tp1": (
                (
                    HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 2
                    + 3 * HEADS * KDA_HEAD_DIM * 4 * 2
                )
                * KDA_LAYERS
                / 2**30
            ),
        },
    }
    output = Path(__file__).parent / "results" / "model_costs.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

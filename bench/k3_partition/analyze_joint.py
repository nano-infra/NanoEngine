#!/usr/bin/env python3
"""Reproduce Chapter 8 capacity and algorithmic-work tables (estimates, not OOM tests)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent
GB, GIB = 10**9, 2**30
HBM = 197897748480
RESERVE = 20 * GIB
KDA = 887800832 * 69
KDA_REP = 1835520 * 69
MLA = 464392192 * 24
MLA_REP = 30281728 * 24
ROUTED = 1446.456 * GB
SHARED = 24.310 * GB
INFRA = 10.637 * GB
DENSE = 1.453 * GB
SHELL = 5.601 * GB
EMBED_HEAD = 2 * 163840 * 7168 * 2
STATE = 69 * 3440640


def weights(tk, tm, ep=16, tf=1, shell_sharded=True):
    # Shared experts and routed down/up remain replicated in current KimiMoE.
    return (ROUTED / (ep * tf) + SHARED + INFRA + DENSE / tf
            + KDA_REP + (KDA - KDA_REP) / tk
            + MLA_REP + (MLA - MLA_REP) / tm
            + SHELL - (EMBED_HEAD * (1 - 1 / tk) if shell_sharded else 0))


def main():
    layouts = []
    for tp, cp in [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (8, 2), (8, 4), (16, 4)]:
        # cp is nested in projection TP; effective attention heads are wider by cp.
        weight = weights(tp, tp)
        conservative = weights(tp, tp, shell_sharded=False)
        cache_per_seq = 24 * 576 * 2**20 / cp + STATE / tp
        remaining = HBM - weight - RESERVE
        slots = max(0, int(remaining // cache_per_seq))
        layouts.append(dict(attn_tp=tp, mla_cp=cp, dp=16 // tp,
            weights_gb=weight / GB, weights_gib=weight / GIB,
            shell_replicated_weights_gib=conservative / GIB,
            reserve_gib=RESERVE / GIB, cache_per_1m_sequence_gib=cache_per_seq / GIB,
            sequences_per_dp=slots, total_1m_sequences=slots * (16 // tp),
            status="current mesh" if cp == 1 else "nested CP design estimate"))
    costs = dict(cache_dtype="fp8_e4m3", cache_bytes_per_token_layer=576, hbm_bytes=HBM, reserve_bytes=RESERVE, layout_estimates=layouts)
    fixed_gflops = 69 * (0.887160832 + 0.006291456) + 24 * 0.464388096 + 92 * 1.436811264 + 1.453
    costs['algorithmic_work'] = dict(fixed_gflop_per_token=fixed_gflops)
    for chunk in (8192, 16384):
        length = 1048576
        pairs = chunk * (length - chunk) + chunk * (chunk + 1) / 2
        costs['algorithmic_work'][f'prefill_1m_{chunk}'] = dict(
            fixed_pflop=fixed_gflops * chunk / 1e6,
            mla_attention_pflop=24 * 2 * 96 * (192 + 128) * pairs / 1e15,
            cached_kv_expansion_pflop=24 * 2 * (length - chunk) * 96 * 256 * 512 / 1e15)
    costs['algorithmic_work']['absorbed_decode_mla_gflop_per_layer_at_1m'] = 2 * 96 * (576 + 512) * 1048576 / 1e9
    costs['algorithmic_work']['expanded_decode_equivalent_mla_gflop_per_layer_at_1m'] = 2 * 96 * (192 + 128) * 1048576 / 1e9
    output = ROOT / 'results/gb200/joint_model.json'
    output.write_text(json.dumps(costs, indent=2) + '\n')
    print(json.dumps(costs, indent=2))


if __name__ == '__main__':
    main()

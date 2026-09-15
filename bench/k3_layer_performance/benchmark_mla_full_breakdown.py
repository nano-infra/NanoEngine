#!/usr/bin/env python3
"""Profile K3 MLA cached-prefix kernels and the complete attention layer."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HIDDEN, HEADS = 7168, 96
Q_LORA, KV_LORA = 1536, 512
NOPE, ROPE, VALUE = 128, 64, 128
Q_DIM = HEADS * (NOPE + ROPE)
V_DIM = HEADS * VALUE
PACKED = 656


def event_pair():
    return torch.cuda.Event(True), torch.cuda.Event(True)


def elapsed(pairs):
    return sum(start.elapsed_time(end) for start, end in pairs)


def rms_norm(x, weight, eps=1e-6):
    variance = x.float().square().mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


class MLAHarness:
    def __init__(self, context: int, chunk: int, split: int):
        from flash_attn.cute import flash_attn_varlen_func
        from dlengine.runtime.kernel.triton.hopper.fp8_utils import (
            restore_mla_fp8_cache_rows,
            store_kcache_fp8,
        )
        from dlengine.runtime.layers.backends.attention.mla_utils import (
            merge_attention_states,
        )

        self.attention = flash_attn_varlen_func
        self.restore = restore_mla_fp8_cache_rows
        self.store = store_kcache_fp8
        self.merge = merge_attention_states
        self.context, self.chunk, self.cached, self.split = (
            context,
            chunk,
            context - chunk,
            split,
        )
        self.hidden = torch.zeros(chunk, HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.positions = torch.arange(chunk, device="cuda", dtype=torch.int32)
        self.cu_q = torch.tensor([0, chunk], device="cuda", dtype=torch.int32)
        self.packed = torch.zeros(self.cached, PACKED, device="cuda", dtype=torch.uint8)
        self.cache_out = torch.empty(
            (chunk + 63) // 64, 64, 1, PACKED,
            device="cuda", dtype=torch.float8_e4m3fn,
        )
        self.slots = torch.arange(chunk, device="cuda", dtype=torch.int32)
        self.weights = {
            "qa": torch.zeros(Q_LORA, HIDDEN, device="cuda", dtype=torch.bfloat16),
            "qb": torch.zeros(Q_DIM, Q_LORA, device="cuda", dtype=torch.bfloat16),
            "kva": torch.zeros(KV_LORA + ROPE, HIDDEN, device="cuda", dtype=torch.bfloat16),
            "gate": torch.zeros(V_DIM, HIDDEN, device="cuda", dtype=torch.bfloat16),
            "out": torch.zeros(HIDDEN, V_DIM, device="cuda", dtype=torch.bfloat16),
            "kc": torch.zeros(KV_LORA, HEADS * NOPE, device="cuda", dtype=torch.bfloat16),
            "vc": torch.zeros(KV_LORA, HEADS * VALUE, device="cuda", dtype=torch.bfloat16),
            "qn": torch.ones(Q_LORA, device="cuda", dtype=torch.bfloat16),
            "kvn": torch.ones(KV_LORA, device="cuda", dtype=torch.bfloat16),
        }

    def run(self):
        full_start, full_end = event_pair()
        full_start.record()
        events = {name: [] for name in (
            "qkv_gate_projection", "latent_norm_cache_write", "fresh_kv_expansion",
            "cache_restore", "fresh_attention", "prefix_kv_expansion",
            "prefix_attention", "lse_merge", "gate_output_projection",
        )}

        def measured(name, fn):
            start, end = event_pair(); start.record(); value = fn(); end.record()
            events[name].append((start, end)); return value

        def projections():
            qa = F.linear(self.hidden, self.weights["qa"])
            q = F.linear(rms_norm(qa, self.weights["qn"]), self.weights["qb"])
            kv = F.linear(self.hidden, self.weights["kva"])
            gate = F.linear(self.hidden, self.weights["gate"])
            return q.view(self.chunk, HEADS, NOPE + ROPE), kv, gate

        q, kv, gate = measured("qkv_gate_projection", projections)

        def norm_store():
            compressed = rms_norm(kv[:, :KV_LORA], self.weights["kvn"])
            key = torch.cat((compressed, kv[:, KV_LORA:]), dim=-1)
            self.store(key.unsqueeze(1), self.cache_out, self.slots)
            return compressed, kv[:, KV_LORA:]

        compressed, rope = measured("latent_norm_cache_write", norm_store)

        def fresh_expand():
            k_nope = (compressed @ self.weights["kc"]).view(self.chunk, HEADS, NOPE)
            k = torch.cat((k_nope, rope[:, None, :].expand(-1, HEADS, -1)), dim=-1)
            v = (compressed @ self.weights["vc"]).view(self.chunk, HEADS, VALUE)
            return k, v

        k_fresh, v_fresh = measured("fresh_kv_expansion", fresh_expand)
        restored = measured("cache_restore", lambda: self.restore(self.packed))

        def fresh_attn():
            value = self.attention(
                q, k_fresh, v_fresh, cu_seqlens_q=self.cu_q,
                cu_seqlens_k=self.cu_q, max_seqlen_q=self.chunk,
                max_seqlen_k=self.chunk, softmax_scale=(NOPE + ROPE) ** -0.5,
                causal=True, return_lse=True,
            )
            return value

        output, lse = measured("fresh_attention", fresh_attn)
        for offset in range(0, self.cached, self.split):
            rows = restored[offset:min(self.cached, offset + self.split)]

            def prefix_expand():
                comp, r = rows[:, :KV_LORA], rows[:, KV_LORA:]
                k_nope = (comp @ self.weights["kc"]).view(-1, HEADS, NOPE)
                k = torch.cat((k_nope, r[:, None, :].expand(-1, HEADS, -1)), dim=-1)
                v = (comp @ self.weights["vc"]).view(-1, HEADS, VALUE)
                return k, v

            k, v = measured("prefix_kv_expansion", prefix_expand)
            cu_k = torch.tensor([0, rows.shape[0]], device="cuda", dtype=torch.int32)

            def prefix_attn():
                return self.attention(
                    q, k, v, cu_seqlens_q=self.cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=self.chunk, max_seqlen_k=rows.shape[0],
                    softmax_scale=(NOPE + ROPE) ** -0.5, causal=False,
                    return_lse=True,
                )

            part, part_lse = measured("prefix_attention", prefix_attn)
            output, lse = measured(
                "lse_merge", lambda: self.merge(output, lse, part, part_lse)
            )

        def gate_out():
            gated = output.reshape(self.chunk, V_DIM) * torch.sigmoid(gate)
            return F.linear(gated, self.weights["out"])

        result = measured("gate_output_projection", gate_out)
        full_end.record()
        torch.cuda.synchronize()
        times = {name: elapsed(pairs) for name, pairs in events.items()}
        times["kernel_pipeline"] = sum(times[name] for name in (
            "cache_restore", "fresh_attention", "prefix_kv_expansion",
            "prefix_attention", "lse_merge",
        ))
        times["layer_stage_sum"] = sum(times[name] for name in events)
        times["full_layer"] = full_start.elapsed_time(full_end)
        assert result.shape == (self.chunk, HIDDEN)
        return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=1_048_576)
    parser.add_argument("--chunks", default="1024,2048,4096,8192,16384")
    parser.add_argument("--prefix-split", type=int, default=131_072)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/mla_full_breakdown.csv")
    args = parser.parse_args()
    rows = []
    for chunk in map(int, args.chunks.split(",")):
        harness = MLAHarness(args.context, chunk, args.prefix_split)
        for _ in range(args.warmup): harness.run()
        trials = [harness.run() for _ in range(args.repeats)]
        row = {"context": args.context, "chunk": chunk, "prefix_split": args.prefix_split}
        for key in trials[0]: row[f"{key}_ms"] = statistics.median(t[key] for t in trials)
        rows.append(row); print(row, flush=True)
        del harness; torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()

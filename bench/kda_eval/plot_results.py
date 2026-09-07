#!/usr/bin/env python3
"""Create KDA evaluation plots and a compact Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    numeric = {
        "logical_length": int,
        "requested_hit_rate": float,
        "effective_hit_rate": float,
        "cached_tokens_per_seq": int,
        "fresh_tokens_per_seq": int,
        "batch_size": int,
        "p50_ms": float,
        "p90_ms": float,
        "p99_ms": float,
        "fresh_tokens_per_s": float,
        "logical_tokens_per_s": float,
    }
    for row in rows:
        for key, cast in numeric.items():
            row[key] = cast(row[key])
    return rows


def save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir
    rows = load_rows(root / "kda_results.csv")
    meta = json.loads((root / "metadata.json").read_text())
    prefill = [r for r in rows if r["mode"] == "prefill_core"]
    decode = [r for r in rows if r["mode"] == "decode_core"]

    grouped = defaultdict(list)
    for row in prefill:
        grouped[(row["batch_size"], row["requested_hit_rate"])].append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (batch, hit), points in sorted(grouped.items()):
        points.sort(key=lambda r: r["logical_length"])
        ax.plot([p["logical_length"] for p in points], [p["p50_ms"] for p in points], marker="o", label=f"hit={hit:.0%}, B={batch}")
    ax.set(xscale="log", yscale="log", xlabel="Logical sequence length", ylabel="P50 latency (ms)", title="KDA chunk prefill latency")
    ax.grid(True, which="both", alpha=.25); ax.legend(ncol=2, fontsize=8)
    save(fig, root / "prefill_latency.png")

    grouped.clear()
    baseline = {(r["batch_size"], r["logical_length"]): r["p50_ms"] for r in prefill if r["requested_hit_rate"] == 0}
    for row in prefill:
        row["speedup"] = baseline[(row["batch_size"], row["logical_length"])] / row["p50_ms"]
        grouped[(row["batch_size"], row["logical_length"])].append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (batch, length), points in sorted(grouped.items()):
        points.sort(key=lambda r: r["effective_hit_rate"])
        ax.plot([p["effective_hit_rate"] * 100 for p in points], [p["speedup"] for p in points], marker="o", label=f"L={length}, B={batch}")
    ideal_x = [0, 25, 50, 75, 90, 95, 99]
    ax.plot(ideal_x, [1 / (1 - x / 100) for x in ideal_x], "k--", label="ideal 1/(1-hit)")
    ax.set(xlabel="Effective hit rate (%)", ylabel="Speedup vs 0% hit", yscale="log", title="KDA effective prefix-hit speedup")
    ax.grid(True, which="both", alpha=.25); ax.legend(ncol=2, fontsize=7)
    save(fig, root / "hit_rate_speedup.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    by_hit = defaultdict(list)
    for row in prefill: by_hit[row["requested_hit_rate"]].append(row)
    for hit, points in sorted(by_hit.items()):
        points.sort(key=lambda r: r["fresh_tokens_per_seq"])
        ax.plot([p["fresh_tokens_per_seq"] for p in points], [p["fresh_tokens_per_s"] for p in points], marker="o", label=f"hit={hit:.0%}")
    ax.set(xscale="log", xlabel="Fresh tokens per sequence", ylabel="Fresh tokens/s", title="KDA computed-token throughput")
    ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=8)
    save(fig, root / "fresh_token_throughput.png")

    if decode:
        decode.sort(key=lambda r: r["batch_size"])
        fig, ax1 = plt.subplots(figsize=(8.5, 5.2)); ax2 = ax1.twinx()
        xs = [r["batch_size"] for r in decode]
        ax1.plot(xs, [r["p50_ms"] for r in decode], "o-", color="tab:blue", label="P50 latency")
        ax2.plot(xs, [r["fresh_tokens_per_s"] for r in decode], "s-", color="tab:orange", label="Throughput")
        ax1.set(xscale="log", xlabel="Decode batch size", ylabel="P50 latency (ms)", title="KDA recurrent decode scaling")
        ax2.set_ylabel("Tokens/s"); ax1.grid(True, which="both", alpha=.25)
        save(fig, root / "decode_scaling.png")

    best = max(prefill, key=lambda r: r["speedup"])
    report = f"""# KDA performance evaluation

- GPU: {meta['gpu']}
- Torch/CUDA: {meta['torch']} / {meta['cuda']}
- Shape: H={meta['heads']}, K={meta['kdim']}, V={meta['vdim']}, BF16
- Timing: {meta['warmup']} warmups, {meta['repeats']} measured iterations (CUDA Events)
- Prefix lengths are aligned down to block size {meta['block_size']}.

## Scope

This is a kernel-level evaluation of the exact chunk-prefill and packed-decode
functions called by `FlashInferKda`. A hit means that the recurrent prefix state
already exists; only the fresh suffix is timed. Current production scheduling
disables cross-request prefix caching for cache plans containing GDN/KDA, so the
hit-rate sweep is a controlled what-if evaluation rather than current end-to-end
request behavior.

## Summary

- Cases: {len(prefill)} prefill, {len(decode)} decode.
- Maximum measured speedup: {best['speedup']:.2f}x at logical length {best['logical_length']}, effective hit {best['effective_hit_rate']:.2%}.

See `kda_results.csv` for all percentiles and throughput values.
"""
    (root / "REPORT.md").write_text(report)


if __name__ == "__main__": main()

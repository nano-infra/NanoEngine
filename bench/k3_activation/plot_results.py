#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def load(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


rows = load(RESULTS / "activation_peaks_kda_mla.csv")
rows += load(RESULTS / "activation_peaks_moe.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
colors = {"kda": "#5B8FF9", "mla": "#61DDAA", "moe": "#F6BD16"}
labels = {
    "kda": "KDA recurrence core",
    "mla": "MLA Prefill attention core",
    "moe": "Local BF16 routed experts",
}
for target in ("kda", "mla", "moe"):
    data = [r for r in rows if r["target"] == target and int(r["batch"]) == 1]
    data.sort(key=lambda r: int(r["active_tokens"]))
    x = [int(r["active_tokens"]) for r in data]
    gib = [int(r["peak_allocated_bytes"]) / 2**30 for r in data]
    kib_per_token = [float(r["allocated_bytes_per_token"]) / 2**10 for r in data]
    axes[0].plot(
        x, gib, marker="o", linewidth=2, label=labels[target], color=colors[target]
    )
    axes[1].plot(
        x,
        kib_per_token,
        marker="o",
        linewidth=2,
        label=labels[target],
        color=colors[target],
    )
for ax in axes:
    ax.grid(alpha=0.25)
    ax.set_xlabel("Active tokens in Prefill chunk")
    ax.legend(frameon=False)
axes[0].set_ylabel("Incremental peak allocated memory (GiB)")
axes[1].set_ylabel("Incremental peak memory (KiB/token)")
axes[0].set_title("Peak memory scales with active tokens")
axes[1].set_title("Per-token slope")
fig.suptitle("K3-shaped operator-core activation memory on NVIDIA B300")
fig.tight_layout()
fig.savefig(RESULTS / "prefill_component_peak_memory.png", dpi=180, bbox_inches="tight")

stages = load(RESULTS / "mla_cached_total_1048576_fresh_16384_stages.csv")
names = [r["stage"] for r in stages[1:]]
live = [int(r["allocated_bytes"]) / 2**30 for r in stages[1:]]
peak = [int(r["peak_allocated_bytes"]) / 2**30 for r in stages[1:]]
fig, ax = plt.subplots(figsize=(9, 4.8))
x = range(len(names))
ax.bar(x, peak, color="#5B8FF9", alpha=0.35, label="Peak so far")
ax.plot(x, live, color="#F4664A", marker="o", linewidth=2.5, label="Live after stage")
for i, value in enumerate(peak):
    ax.text(i, value + 3, f"{value:.2f} GiB", ha="center", fontsize=9)
ax.set_xticks(list(x), names)
ax.set_ylabel("Incremental allocated memory (GiB)")
ax.set_title("MLA cached-prefix Prefill: 16K fresh tokens over 1M total context")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
ax.set_ylim(0, max(peak) * 1.15)
fig.tight_layout()
fig.savefig(RESULTS / "mla_cached_1m_16k_peak.png", dpi=180, bbox_inches="tight")

context_lengths = [16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576]
context_peaks = []
for context_length in context_lengths:
    path = RESULTS / f"mla_cached_total_{context_length}_fresh_16384_stages.csv"
    final = [r for r in load(path) if r["stage"] == "FlashAttention"][0]
    context_peaks.append(int(final["peak_allocated_bytes"]) / 2**30)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(
    [value / 1024 for value in context_lengths],
    context_peaks,
    marker="o",
    linewidth=2.5,
    color="#5B8FF9",
)
for x_value, y_value in zip(context_lengths, context_peaks):
    ax.text(x_value / 1024, y_value + 3, f"{y_value:.2f}", ha="center", fontsize=8)
ax.set_xlabel("Total context length (K tokens)")
ax.set_ylabel("Incremental peak allocated memory (GiB)")
ax.set_title("MLA cached-prefix Prefill peak with a fixed 16K fresh chunk")
ax.grid(alpha=0.25)
ax.set_ylim(0, max(context_peaks) * 1.15)
fig.tight_layout()
fig.savefig(RESULTS / "mla_peak_vs_context_16k_chunk.png", dpi=180, bbox_inches="tight")

split_sizes = [16_384, 32_768, 65_536, 131_072, 262_144]
split_rows = []
for split_size in split_sizes:
    split_rows.extend(load(RESULTS / f"mla_prefix_split_{split_size}.csv"))
unsplit = load(RESULTS / "mla_prefix_split_unsplit.csv")[0]
labels = ["16K", "32K", "64K", "128K\n(default)", "256K", "Unsplit"]
plot_rows = split_rows + [unsplit]
peaks = [int(row["incremental_peak_bytes"]) / 2**30 for row in plot_rows]
latencies = [float(row["steady_forward_ms"]) for row in plot_rows]
fig, memory_ax = plt.subplots(figsize=(9.2, 5.0))
x = list(range(len(labels)))
bars = memory_ax.bar(x, peaks, color="#5B8FF9", alpha=0.82)
memory_ax.set_ylabel("Incremental peak allocated memory (GiB)", color="#3568B8")
memory_ax.tick_params(axis="y", labelcolor="#3568B8")
memory_ax.set_xticks(x, labels)
memory_ax.grid(axis="y", alpha=0.22)
latency_ax = memory_ax.twinx()
latency_ax.plot(x, latencies, color="#F4664A", marker="o", linewidth=2.5)
latency_ax.set_ylabel("Steady forward latency (ms)", color="#C84432")
latency_ax.tick_params(axis="y", labelcolor="#C84432")
for bar, value in zip(bars, peaks):
    memory_ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2.5,
        f"{value:.1f}",
        ha="center",
        fontsize=9,
    )
for index, value in enumerate(latencies):
    latency_ax.annotate(
        f"{value:.0f}",
        (index, value),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        color="#A83B2D",
        fontsize=8,
    )
memory_ax.set_ylim(0, max(peaks) * 1.14)
latency_ax.set_ylim(min(latencies) * 0.92, max(latencies) * 1.08)
memory_ax.set_title("MLA prefix chunking: 1M context, 16K fresh tokens")
fig.tight_layout()
fig.savefig(RESULTS / "mla_prefix_split_tradeoff.png", dpi=180, bbox_inches="tight")

mega = load(RESULTS / "activation_peaks_megamoe_ws1.csv")
reference = load(RESULTS / "activation_peaks_moe.csv")
mega_by_tokens = {int(row["tokens"]): row for row in mega}
reference_by_tokens = {int(row["active_tokens"]): row for row in reference}
tokens = sorted(set(mega_by_tokens) & set(reference_by_tokens))
workspace_gib = int(mega[0]["workspace_device_bytes_at_capacity"]) / 2**30
reference_gib = [
    int(reference_by_tokens[value]["peak_allocated_bytes"]) / 2**30
    for value in tokens
]
mega_dynamic_gib = [
    int(mega_by_tokens[value]["steady_forward_peak_bytes"]) / 2**30
    for value in tokens
]
mega_total_gib = [workspace_gib + value for value in mega_dynamic_gib]
fig, ax = plt.subplots(figsize=(8.8, 4.9))
ax.plot(tokens, reference_gib, marker="o", linewidth=2.4, label="BF16 reference dynamic peak", color="#F4664A")
ax.plot(tokens, mega_dynamic_gib, marker="o", linewidth=2.4, label="MegaMoE steady dynamic peak", color="#61DDAA")
ax.plot(tokens, mega_total_gib, marker="o", linewidth=2.4, label="MegaMoE persistent buffer + dynamic", color="#5B8FF9")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Active tokens")
ax.set_ylabel("Memory (GiB, log scale)")
ax.set_title("K3 routed experts: reference vs NanoDeploy MegaMoE (EP=1)")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RESULTS / "megamoe_ws1_activation.png", dpi=180, bbox_inches="tight")

# Chapter 5 summary: production-relevant activation/workspace reservations.
labels = ["KDA\nrecurrence", "MLA 1M/16K\n128K split", "MegaMoE\n16K capacity"]
dynamic = [4429185024 / 2**30, 18673041920 / 2**30, 117440512 / 2**30]
persistent = [0, 0, 6377439232 / 2**30]
fig, ax = plt.subplots(figsize=(8.8, 4.9))
x = list(range(len(labels)))
ax.bar(x, persistent, label="Persistent reusable buffer", color="#607D8B")
ax.bar(x, dynamic, bottom=persistent, label="Incremental activation peak", color="#4C78A8")
for i, (reserved, transient) in enumerate(zip(persistent, dynamic)):
    total = reserved + transient
    ax.text(i, total + 0.35, f"{total:.2f} GiB", ha="center", fontweight="bold")
ax.set_xticks(x, labels)
ax.set_ylabel("Device memory (GiB)")
ax.set_ylim(0, max(p + d for p, d in zip(persistent, dynamic)) * 1.16)
ax.set_title("K3 measured activation and workspace capacity")
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RESULTS / "activation_capacity_summary.png", dpi=180, bbox_inches="tight")

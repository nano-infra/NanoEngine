#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).parent / "results"
with (root / "kda_results.csv").open() as f:
    kda = list(csv.DictReader(f))
prefill = [r for r in kda if r["mode"] == "prefill_core"]

# Match the MLA presentation: fixed logical context, latency and speedup axes.
kda_fixed = sorted(
    (r for r in prefill if int(r["logical_length"]) == 16384),
    key=lambda r: float(r["effective_hit_rate"]),
)
kda_hit = [100 * float(r["effective_hit_rate"]) for r in kda_fixed]
kda_latency = [float(r["p50_ms"]) for r in kda_fixed]
kda_baseline = kda_latency[0]
fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax2 = ax1.twinx()
ax1.plot(kda_hit, kda_latency, "o-", linewidth=2.4, color="tab:blue")
ax2.plot(kda_hit, [kda_baseline / value for value in kda_latency], "s-", linewidth=2.0, color="tab:orange")
ax1.set(xlabel="Effective prefix-cache hit (%)", ylabel="P50 latency (ms)", title="K3 KDA cache hit at fixed 16K logical context")
ax2.set_ylabel("Speedup vs no cache hit")
ax1.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(root / "kda_cache_hit_latency.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for logical in sorted({int(r["logical_length"]) for r in prefill}):
    points = sorted((r for r in prefill if int(r["logical_length"]) == logical), key=lambda r: int(r["fresh_tokens_per_seq"]))
    ax.plot([int(p["fresh_tokens_per_seq"]) for p in points], [float(p["p50_ms"]) for p in points], "o-", label=f"logical={logical // 1024}K" if logical >= 1024 else f"logical={logical}")
ax.set(xscale="log", yscale="log", xlabel="Fresh suffix tokens", ylabel="P50 latency (ms)", title="K3 KDA cache hit: latency follows the fresh suffix")
ax.grid(True, which="both", alpha=0.25)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(root / "kda_latency_vs_fresh_suffix.png", dpi=180, bbox_inches="tight")
plt.close(fig)

decode = sorted((r for r in kda if r["mode"] == "decode_core"), key=lambda r: int(r["batch_size"]))
fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax2 = ax1.twinx()
batch = [int(r["batch_size"]) for r in decode]
ax1.plot(batch, [float(r["p50_ms"]) for r in decode], "o-", color="tab:blue")
ax2.plot(batch, [float(r["fresh_tokens_per_s"]) for r in decode], "s-", color="tab:orange")
ax1.set(xscale="log", xlabel="Decode batch size", ylabel="P50 latency (ms)", title="K3 KDA recurrent Decode scaling")
ax2.set_ylabel("Tokens/s")
ax1.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(root / "kda_decode_scaling.png", dpi=180, bbox_inches="tight")
plt.close(fig)

mla = [json.loads(path.read_text()) for path in root.glob("mla_cache_total_65536_fresh_*_split_131072.json")]
mla.sort(key=lambda r: r["cached_prefix"] / r["total_context"])
hit = [100 * r["cached_prefix"] / r["total_context"] for r in mla]
latency = [r["steady_forward_ms"] for r in mla]
baseline = latency[0]
fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax2 = ax1.twinx()
ax1.plot(hit, latency, "o-", linewidth=2.4, color="tab:blue")
ax2.plot(hit, [baseline / value for value in latency], "s-", linewidth=2.0, color="tab:orange")
ax1.set(xlabel="Effective prefix-cache hit (%)", ylabel="Steady latency (ms)", title="K3 MLA cache hit at fixed 64K logical context")
ax2.set_ylabel("Speedup vs no cache hit")
ax1.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(root / "mla_cache_hit_latency.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Primary serving result: 1M logical context processed in 16K chunks.
with (root / "cache_1m_serving.csv").open() as f:
    serving = list(csv.DictReader(f))
for component, stem, label in (
    ("KDA recurrence", "kda_cache_1m_serving", "KDA"),
    ("MLA cached attention", "mla_cache_1m_serving", "MLA"),
):
    points = [r for r in serving if r["component"] == component]
    hit = [100 * float(r["effective_hit_rate"]) for r in points]
    latency = [float(r["remaining_prefill_ms"]) for r in points]
    speedup = [float(r["speedup_vs_no_hit"]) for r in points]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(hit, latency, "o-", linewidth=2.4, color="tab:blue")
    ax2.plot(hit, speedup, "s-", linewidth=2.0, color="tab:orange")
    ax1.set(xlabel="Effective prefix-cache hit (%)", ylabel="Remaining Prefill latency (ms)", title=f"K3 {label} cache reuse: 1M context, 16K Prefill chunks")
    ax2.set_ylabel("Speedup vs no cache hit")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

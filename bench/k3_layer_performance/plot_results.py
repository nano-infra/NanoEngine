#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root = Path(__file__).parent / "results"
with (root / "kda_full_prefill_breakdown.csv").open() as f:
    kda_matrix = list(csv.DictReader(f))
with (root / "mla_full_breakdown.csv").open() as f:
    mla_full = list(csv.DictReader(f))
with (root / "megamoe_chunk.csv").open() as f:
    mega = list(csv.DictReader(f))
with (root / "dense_ffn.csv").open() as f:
    dense = list(csv.DictReader(f))
mla = [json.loads(p.read_text()) for p in root.glob("mla_cache_total_*_split_131072.json")]


def scientific_axis(ax, axis="y", power=(0, 0)):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(power)
    getattr(ax, f"{axis}axis").set_major_formatter(formatter)


def token_axis(ax):
    scientific_axis(ax, axis="x", power=(3, 3))


# KDA Prefill: one scaling subplot per internal stage. Context is aggregated
# because the measured curves overlap; the band shows the full context range.
line_stages = [
    ("full_ms", "Complete forward", 0.893),
    ("input_projections_ms", "Input projections", 0.711),
    ("causal_conv_ms", "Causal convolution", 0.000295),
    ("recurrence_ms", "KDA recurrence", 0.006),
    ("gated_norm_ms", "Gated normalization", 0.0006),
    ("output_projection_ms", "Output projection", 0.176),
]
fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0), sharex=True)
chunks = sorted({int(r["chunk"]) for r in kda_matrix})
contexts = sorted({int(r["context"]) for r in kda_matrix})
markers = ["o", "s", "^", "D"]
for panel_idx, (ax, (key, title, gflops_per_token)) in enumerate(zip(axes.flat, line_stages)):
    for context, marker in zip(contexts, markers):
        rows = sorted((r for r in kda_matrix if int(r["context"]) == context), key=lambda r: int(r["chunk"]))
        ax.plot([int(r["chunk"]) for r in rows], [float(r[key]) for r in rows], marker=marker, linewidth=1.8, markersize=4.5, label=f"{context // 1024}K context")
    reference = sorted((r for r in kda_matrix if int(r["context"]) == max(contexts)), key=lambda r: int(r["chunk"]))
    right = ax.twinx()
    right.plot(chunks, [gflops_per_token * int(r["chunk"]) * 1000 / float(r[key]) for r in reference], "k--", linewidth=1.35, alpha=.7)
    if panel_idx % 3 == 2:
        right.set_ylabel("Achieved throughput (GFLOP/s)", fontsize=8)
    right.tick_params(axis="y", labelsize=7)
    scientific_axis(right)
    ax.set_title(title); ax.set_xlabel("Chunk size (tokens)"); ax.set_ylabel("Latency (ms)")
    token_axis(ax); ax.grid(True, alpha=.24)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=4, frameon=False)
fig.suptitle("KDA Prefill stage scaling on one B300", y=.98, fontsize=14)
fig.tight_layout(rect=(0, .08, 1, .93)); fig.savefig(root / "kda_prefill_stage_scaling.svg", bbox_inches="tight"); plt.close(fig)

# KDA Prefill stage breakdown at the representative 1M serving context.
stages = [("input_projections_ms", "Input projections"), ("causal_conv_ms", "Causal convolution"), ("recurrence_ms", "KDA recurrence"), ("gated_norm_ms", "Gated normalization"), ("output_projection_ms", "Output projection")]
context = 1048576
rows = sorted((r for r in kda_matrix if int(r["context"]) == context), key=lambda r: int(r["chunk"]))
chunks = [int(r["chunk"]) for r in rows]
positions = list(range(len(chunks)))
tick_labels = [f"{chunk // 1024}K" for chunk in chunks]
fig, ax = plt.subplots(figsize=(9.4, 5.4))
bottom = [0.0] * len(rows)
for key, label in stages:
    values = [float(r[key]) for r in rows]
    ax.bar(positions, values, bottom=bottom, width=.72, label=label)
    bottom = [a + b for a, b in zip(bottom, values)]
ax.plot(positions, [float(r["full_ms"]) for r in rows], "ko--", linewidth=1.3, markersize=4, label="Complete forward")
ax.set_title("Complete KDA Prefill latency breakdown · 1M context · one B300")
ax.set_xlabel("Chunk size"); ax.set_ylabel("Latency (ms)")
ax.set_xticks(positions, tick_labels); ax.grid(True, axis="y", alpha=.22)
right = ax.twinx()
average_gflops = [0.893 * chunk * 1000 / float(row["full_ms"]) for chunk, row in zip(chunks, rows)]
right.plot(positions, average_gflops, "D-", color="tab:red", linewidth=2.0, markersize=4.5, label="Average achieved throughput")
right.axhline(4.5e6, color="tab:purple", linestyle=":", linewidth=2.0, label="B300 BF16 Tensor Core peak")
right.set_ylabel("Average achieved throughput (GFLOP/s)")
scientific_axis(right)
left_handles, left_labels = ax.get_legend_handles_labels()
right_handles, right_labels = right.get_legend_handles_labels()
fig.legend(left_handles + right_handles, left_labels + right_labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
fig.tight_layout(rect=(0, .14, 1, 1)); fig.savefig(root / "kda_prefill_breakdown.svg", bbox_inches="tight"); plt.close(fig)

# MLA cached-prefix kernel pipeline breakdown.
mla_rows = sorted(mla_full, key=lambda r: int(r["chunk"]))
mla_chunks = [int(r["chunk"]) for r in mla_rows]
mla_positions = list(range(len(mla_chunks)))
mla_labels = [f"{chunk // 1024}K" for chunk in mla_chunks]
kernel_stages = [("cache_restore_ms", "FP8 cache restore"), ("fresh_attention_ms", "Fresh attention"), ("prefix_kv_expansion_ms", "Prefix KV expansion"), ("prefix_attention_ms", "Prefix attention"), ("lse_merge_ms", "LSE merge")]
fig, ax = plt.subplots(figsize=(9.6, 5.6))
bottom = [0.0] * len(mla_rows)
for key, label in kernel_stages:
    values = [float(r[key]) for r in mla_rows]
    ax.bar(mla_positions, values, bottom=bottom, width=.72, label=label)
    bottom = [a + b for a, b in zip(bottom, values)]
right = ax.twinx()
kernel_rates = []
for chunk, row in zip(mla_chunks, mla_rows):
    cached = int(row["context"]) - chunk
    expansion = cached * 2 * 512 * 96 * (128 + 128)
    pairs = chunk * cached + chunk * (chunk + 1) // 2
    attention = 2 * 96 * (192 + 128) * pairs
    kernel_rates.append((expansion + attention) / 1e9 * 1000 / float(row["kernel_pipeline_ms"]))
right.plot(mla_positions, kernel_rates, "D-", color="tab:red", linewidth=2.0, markersize=4.5, label="Kernel average throughput")
right.axhline(4.5e6, color="tab:purple", linestyle=":", linewidth=2.0, label="B300 BF16 Tensor Core peak")
right.set_ylabel("Average achieved throughput (GFLOP/s)")
scientific_axis(right)
ax.set_title("MLA cached-prefix kernel pipeline · 1M context · 128K split · one B300")
ax.set_xlabel("Fresh chunk size")
ax.set_ylabel("Kernel latency (ms)")
ax.set_xticks(mla_positions, mla_labels)
ax.grid(True, axis="y", alpha=.22)
lh, ll = ax.get_legend_handles_labels()
fig.legend(lh, ll, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
fig.tight_layout(rect=(0, .16, 1, 1))
fig.savefig(root / "mla_kernel_breakdown.svg", bbox_inches="tight")
plt.close(fig)

# MLA kernel scaling at fixed 16K fresh chunk across context lengths.
mla_context_rows = []
for path in root.glob("mla_context_*_chunk_16384.csv"):
    with path.open() as f:
        mla_context_rows.extend(csv.DictReader(f))
mla_context_rows.sort(key=lambda r: int(r["context"]))
context_values = [int(r["context"]) for r in mla_context_rows]
context_positions = list(range(len(context_values)))
context_labels = [f"{value // 1024}K" if value < 1048576 else "1M" for value in context_values]
fig, ax = plt.subplots(figsize=(9.6, 5.6))
bottom = [0.0] * len(mla_context_rows)
for key, label in kernel_stages:
    values = [float(r[key]) for r in mla_context_rows]
    ax.bar(context_positions, values, bottom=bottom, width=.72, label=label)
    bottom = [a + b for a, b in zip(bottom, values)]
right = ax.twinx()
context_rates = []
for total, row in zip(context_values, mla_context_rows):
    chunk = 16384
    cached = total - chunk
    expansion = cached * 2 * 512 * 96 * (128 + 128)
    pairs = chunk * cached + chunk * (chunk + 1) // 2
    attention = 2 * 96 * (192 + 128) * pairs
    context_rates.append((expansion + attention) / 1e9 * 1000 / float(row["kernel_pipeline_ms"]))
right.plot(context_positions, context_rates, "D-", color="tab:cyan", linewidth=2.0, markersize=4.5)
right.axhline(4.5e6, color="tab:purple", linestyle=":", linewidth=2.0)
right.set_ylabel("Average achieved throughput (GFLOP/s)")
scientific_axis(right)
ax.set_title("MLA cached-prefix kernel scaling · 16K fresh chunk · 128K split · one B300")
ax.set_xlabel("Visible context length")
ax.set_ylabel("Kernel latency (ms)")
ax.set_xticks(context_positions, context_labels)
ax.grid(True, axis="y", alpha=.22)
lh, ll = ax.get_legend_handles_labels()
fig.legend(lh, ll, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
fig.tight_layout(rect=(0, .16, 1, 1))
fig.savefig(root / "mla_kernel_context_breakdown.svg", bbox_inches="tight")
plt.close(fig)

# Paired kernel breakdown: chunk and context sweeps.
fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.7))
for panel, rows_for_panel, x_values, x_labels, panel_title in (
    (axes[0], mla_rows, mla_chunks, mla_labels, "(a) Fixed 1M context · vary fresh chunk"),
    (axes[1], mla_context_rows, context_values, context_labels, "(b) Fixed 16K fresh chunk · vary context"),
):
    pos = list(range(len(rows_for_panel)))
    base = [0.0] * len(rows_for_panel)
    for key, label in kernel_stages:
        vals = [float(r[key]) for r in rows_for_panel]
        panel.bar(pos, vals, bottom=base, width=.72, label=label)
        base = [a + b for a, b in zip(base, vals)]
    panel.set_title(panel_title)
    panel.set_xticks(pos, x_labels)
    panel.set_xlabel("Fresh chunk size" if panel is axes[0] else "Visible context length")
    panel.set_ylabel("Kernel latency (ms)")
    panel.grid(True, axis="y", alpha=.22)
    rate_axis = panel.twinx()
    rates = []
    for row in rows_for_panel:
        chunk = int(row["chunk"]); total = int(row["context"]); cached = total - chunk
        expansion = cached * 2 * 512 * 96 * (128 + 128)
        pairs = chunk * cached + chunk * (chunk + 1) // 2
        attention = 2 * 96 * (192 + 128) * pairs
        rates.append((expansion + attention) / 1e9 * 1000 / float(row["kernel_pipeline_ms"]))
    rate_axis.plot(pos, rates, "D-", color="tab:cyan", linewidth=1.8, markersize=4)
    rate_axis.axhline(4.5e6, color="tab:purple", linestyle=":", linewidth=1.7)
    rate_axis.set_ylabel("Average achieved throughput (GFLOP/s)")
    scientific_axis(rate_axis)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncol=5, frameon=False)
fig.suptitle("MLA cached-prefix kernel pipeline · 128K split · one B300", y=.98)
fig.tight_layout(rect=(0, .12, 1, .94))
fig.savefig(root / "mla_kernel_ab_breakdown.svg", bbox_inches="tight")
plt.close(fig)

# Complete K3 MLA attention-layer cumulative breakdown.
layer_stages = [("qkv_gate_projection_ms", "Q/KV/G projections"), ("latent_norm_cache_write_ms", "Latent norm + cache write"), ("fresh_kv_expansion_ms", "Fresh K/V expansion"), ("kernel_pipeline_ms", "Cached-prefix kernel pipeline"), ("gate_output_projection_ms", "Gate + output projection")]

# Paired cumulative context sweep: cached-prefix kernels and complete MLA layer.
# Prefix legend labels make clear that the right panel contains the whole left
# pipeline as one stacked stage; both panels intentionally reuse the same sweep.
pipeline_legend_stages = [(key, f"Pipeline · {label}") for key, label in kernel_stages]
layer_legend_stages = [
    ("qkv_gate_projection_ms", "Layer · Q/KV/G projections"),
    ("latent_norm_cache_write_ms", "Layer · latent norm + cache write"),
    ("fresh_kv_expansion_ms", "Layer · fresh K/V expansion"),
    ("kernel_pipeline_ms", "Layer · cached-prefix pipeline"),
    ("gate_output_projection_ms", "Layer · gate + output projection"),
]
fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.7), sharey=True)
for panel, stages_for_panel, title, total_key in (
    (axes[0], pipeline_legend_stages, "(a) Pipeline only · fixed 16K chunk", "kernel_pipeline_ms"),
    (axes[1], layer_legend_stages, "(b) Complete layer · pipeline included", "full_layer_ms"),
):
    base = [0.0] * len(mla_context_rows)
    for key, label in stages_for_panel:
        values = [float(row[key]) for row in mla_context_rows]
        panel.bar(context_positions, values, bottom=base, width=.72, label=label)
        base = [a + b for a, b in zip(base, values)]
    panel.plot(context_positions, [float(row[total_key]) for row in mla_context_rows], "ko--", linewidth=1.3, markersize=4, label="Measured total")
    panel.set_title(title)
    panel.set_xlabel("Visible context length")
    panel.set_xticks(context_positions, context_labels)
    panel.grid(True, axis="y", alpha=.22)
axes[0].set_ylabel("Cumulative latency (ms)")
for panel in axes:
    panel.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.suptitle("MLA context scaling · fixed 16K fresh chunk · 128K split · one B300", y=.98)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(root / "mla_context_kernel_layer_ab.svg", bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(figsize=(9.6, 5.6))
bottom = [0.0] * len(mla_rows)
for key, label in layer_stages:
    values = [float(r[key]) for r in mla_rows]
    ax.bar(mla_positions, values, bottom=bottom, width=.72, label=label)
    bottom = [a + b for a, b in zip(bottom, values)]
ax.plot(mla_positions, [float(r["full_layer_ms"]) for r in mla_rows], "ko--", linewidth=1.4, markersize=4, label="Complete layer forward")
right = ax.twinx()
layer_rates = []
for chunk, row in zip(mla_chunks, mla_rows):
    cached = int(row["context"]) - chunk
    proj = 2 * chunk * (7168 * 1536 + 1536 * (96 * 192) + 7168 * 576 + 7168 * (96 * 128) + (96 * 128) * 7168)
    expansion = 2 * (cached + chunk) * 512 * 96 * (128 + 128)
    pairs = chunk * cached + chunk * (chunk + 1) // 2
    attention = 2 * 96 * (192 + 128) * pairs
    total_gflop = (proj + expansion + attention) / 1e9
    layer_rates.append(total_gflop * 1000 / float(row["full_layer_ms"]))
right.plot(mla_positions, layer_rates, "D-", color="tab:cyan", linewidth=2.0, markersize=4.5, label="Layer average throughput")
right.axhline(4.5e6, color="tab:purple", linestyle=":", linewidth=2.0, label="B300 BF16 Tensor Core peak")
right.set_ylabel("Average achieved throughput (GFLOP/s)")
scientific_axis(right)
ax.set_title("Complete K3 MLA attention layer · 1M context · 128K split · one B300")
ax.set_xlabel("Fresh chunk size")
ax.set_ylabel("Cumulative latency (ms)")
ax.set_xticks(mla_positions, mla_labels)
ax.grid(True, axis="y", alpha=.22)
lh, ll = ax.get_legend_handles_labels(); rh, rl = right.get_legend_handles_labels()
fig.legend(lh + rh, ll + rl, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
fig.tight_layout(rect=(0, .16, 1, 1))
fig.savefig(root / "mla_layer_breakdown.svg", bbox_inches="tight")
plt.close(fig)

# Dense FFN Prefill/Decode scaling.
dense_prefill = sorted((r for r in dense if r["mode"] == "prefill"), key=lambda r: int(r["tokens"]))
dense_decode = sorted((r for r in dense if r["mode"] == "decode"), key=lambda r: int(r["tokens"]))
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
for ax, rows, title in ((axes[0], dense_prefill, "(a) Prefill · one request"), (axes[1], dense_decode, "(b) Decode · active batch")):
    x = [int(r["tokens"]) for r in rows]
    ax.plot(x, [float(r["latency_ms"]) for r in rows], "o-", linewidth=2.2, color="tab:blue", label="Latency")
    right = ax.twinx()
    right.plot(x, [float(r["gflops"]) for r in rows], "s--", linewidth=1.8, color="tab:orange", label="Achieved GFLOP/s")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(x, [f"{v // 1024}K" if v >= 1024 else str(v) for v in x])
    ax.set_xlabel("Fresh tokens" if rows is dense_prefill else "Decode batch size")
    ax.set_ylabel("Latency (ms)"); right.set_ylabel("Achieved GFLOP/s")
    ax.set_title(title); ax.grid(True, which="both", alpha=.23); scientific_axis(right)
fig.suptitle("K3 Dense SwiGLU FFN scaling · one B300 · BF16", y=.99)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(root / "dense-ffn-scaling.svg", bbox_inches="tight")
plt.close(fig)

# MegaMoE Prefill: perfectly balanced routed rows.
mega.sort(key=lambda r: int(r["tokens"]))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
mega_tokens = [int(r["tokens"]) for r in mega]
mega_latency = [float(r["steady_latency_ms"]) for r in mega]
ax.plot(mega_tokens, mega_latency, "o-", linewidth=2.3)
right = ax.twinx()
right.plot(mega_tokens, [1.056964608 * n * 1000 / ms for n, ms in zip(mega_tokens, mega_latency)], "s--", color="tab:orange", linewidth=1.8)
ax.set(xlabel="Chunk size (tokens)", ylabel="Latency (ms)", title="Balanced MegaMoE Prefill scaling on one B300")
right.set_ylabel("Achieved throughput (GFLOP/s)")
scientific_axis(right)
token_axis(ax); ax.grid(True, alpha=.24)
fig.tight_layout(); fig.savefig(root / "megamoe_prefill_scaling.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# KDA Decode batch scaling.
with (root / "kda_chunk_decode.csv").open() as f:
    kda = list(csv.DictReader(f))
kd = sorted((r for r in kda if r["mode"] == "decode_core"), key=lambda r: int(r["batch_size"]))
fig, ax1 = plt.subplots(figsize=(8.5, 4.8)); ax2 = ax1.twinx()
b = [int(r["batch_size"]) for r in kd]
lat = [float(r["p50_ms"]) for r in kd]
tps = [0.006 * batch * 1000 / ms for batch, ms in zip(b, lat)]
ax1.plot(b, lat, "o-", linewidth=2.3, color="tab:blue")
ax2.plot(b, tps, "s-", linewidth=2.0, color="tab:orange")
ax1.set(xscale="log", xlabel="Decode batch size", ylabel="P50 latency (ms)", title="K3 KDA Decode scaling on one B300")
ax2.set_ylabel("Achieved throughput (GFLOP/s)"); scientific_axis(ax2); ax1.grid(True, which="both", alpha=.24)
fig.tight_layout(); fig.savefig(root / "kda_decode_scaling.png", dpi=180, bbox_inches="tight"); plt.close(fig)

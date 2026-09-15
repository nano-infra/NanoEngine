#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
COLORS = {
    "all_reduce": "#5B8FF9",
    "reduce_scatter": "#61DDAA",
    "all_gather": "#F6BD16",
    "all_to_all": "#F4664A",
}
LABELS = {
    "all_reduce": "All-reduce",
    "reduce_scatter": "Reduce-scatter",
    "all_gather": "All-gather",
    "all_to_all": "All-to-all",
}


def load(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, world in zip(axes, (2, 4)):
    rows = load(RESULTS / f"collectives_tp{world}.csv")
    for collective in COLORS:
        data = [row for row in rows if row["collective"] == collective]
        payload_mib = [int(row["logical_payload_bytes"]) / 2**20 for row in data]
        latency_us = [float(row["latency_ms"]) * 1000 for row in data]
        ax.plot(
            payload_mib,
            latency_us,
            marker="o",
            linewidth=2,
            color=COLORS[collective],
            label=LABELS[collective],
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Logical BF16 hidden-state payload (MiB)")
    ax.set_title(f"{world} GPUs")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Average collective latency (µs)")
axes[1].legend(frameon=False)
fig.suptitle("K3 boundary collectives on NVIDIA B300 NVLink")
fig.tight_layout()
fig.savefig(RESULTS / "collective_latency.png", dpi=180, bbox_inches="tight")

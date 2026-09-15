#!/usr/bin/env python3
"""Plot the SGLang HiSparse ratio-driven capacity model.

The formulas and defaults come from ``dlengine-hisparse-capacity.md``. The
configured ``R_host`` scales the logical pool from the device Buffer, and the
plot solves their joint Buffer-plus-Indexer HBM budget. The x-axis is the
host/device logical ratio. Each run evaluates one model, so pass that model's
Indexer layer-sharing factor (``--shared-ratios``) and context length
(``--max-model-len``). The capacity curve is the maximum worker-wide logical
capacity allowed by the GPU inequality:

    C_Batch_gpu = N_bs_max * R_host * M_cache
            / (((R_host * B_indexer / R_share) + B_mla)
               * N_layer * N_bs_max)

Feasibility additionally requires:

    N_buffer >= N_topk
    N_buffer <= GPU upper bound
    L_max_model <= C_Batch
    C_Batch = N_bs_max * R_host * N_buffer

Examples (GLM5.1: per-layer Indexer, R_share=1, 256K context;
GLM5.2: Indexer shared across layers 78/21, R_share=3.714, 1M context):

    python -m pip install numpy matplotlib

    python docs/site/blogs/plot_hisparse_capacity.py \
        --config-name glm51_h100_dp16_ep_16 \
        --shared-ratios 1.0 --max-model-len 262144 \
        --output docs/imgs/glm51_h100_dp16_ep_16.png
    python docs/site/blogs/plot_hisparse_capacity.py \
        --config-name glm51_h100_dp32_ep_32 \
        --shared-ratios 1.0 --max-model-len 262144 \
        --weights-gb 43.42 --memory-fraction 0.82 \
        --output docs/imgs/glm51_h100_dp32_ep_32.png
    python docs/site/blogs/plot_hisparse_capacity.py \
        --config-name glm52_h100_dp16_ep_16 \
        --shared-ratios 3.714 --max-model-len 1048576 \
        --output docs/imgs/glm52_h100_dp16_ep_16.png
    python docs/site/blogs/plot_hisparse_capacity.py \
        --config-name glm52_h100_dp32_ep_32 \
        --shared-ratios 3.714 --max-model-len 1048576 \
        --weights-gb 43.42 --memory-fraction 0.82 \
        --output docs/imgs/glm52_h100_dp32_ep_32.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import (
    AutoMinorLocator,
    FuncFormatter,
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
)


_PREFERRED_FONT = "Linux Biolinum O"
if any(font.name == _PREFERRED_FONT for font in font_manager.fontManager.ttflist):
    plt.rcParams.update(
        {
            "font.family": _PREFERRED_FONT,
            "mathtext.fontset": "custom",
            "mathtext.rm": _PREFERRED_FONT,
            "mathtext.it": f"{_PREFERRED_FONT}:italic",
            "mathtext.bf": f"{_PREFERRED_FONT}:bold",
        }
    )
plt.rcParams.update(
    {
        "font.size": 35,
        "axes.titlesize": 42,
        "axes.labelsize": 42,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
    }
)

_BS_COLORS = (
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
)


def power_of_two_formatter(value: float, _position: int) -> str:
    """Format positive base-2 log ticks as powers of two."""
    if value <= 0:
        return ""
    exponent = int(round(np.log2(value)))
    return rf"$2^{{{exponent}}}$"


def decimal_power_of_two_formatter(value: float, _position: int) -> str:
    """Format base-2 log ticks as ordinary decimal integers."""
    if value < 1:
        return ""
    return f"{int(round(value))}"


def power_of_two_ticks(
    minimum: float, maximum: float, *, include_zero: bool
) -> list[float]:
    """Return visible zero/power-of-two ticks for a ratio axis."""
    ticks = [0] if include_zero and minimum <= 0 <= maximum else []
    minimum_exponent = -1
    if maximum < 2**minimum_exponent:
        return ticks
    max_exponent = int(np.floor(np.log2(maximum)))
    ticks.extend(
        2**exponent
        for exponent in range(minimum_exponent, max_exponent + 1)
        if minimum <= 2**exponent <= maximum
    )
    return ticks


def academic_config_title(config_name: str) -> str:
    """Turn a config key into a concise publication-style figure title."""
    parts = config_name.split("_")
    if len(parts) >= 4:
        model = {
            "glm51": "GLM-5.1",
            "glm52": "GLM-5.2",
        }.get(parts[0].lower(), parts[0].upper())
        accelerator = parts[1].upper()
        dp = parts[2].upper()
        ep = "".join(parts[3:]).upper()
        return f"{model} · {accelerator} · {dp}/{ep}"
    return config_name.replace("_", " ")


@dataclass(frozen=True)
class CapacityConfig:
    config_name: str = "h100_dp16_ep_16"
    hbm_gb: float = 77.47
    memory_fraction: float = 0.88
    weights_gb: float = 64.52
    indexer_bytes: float = 132.0
    mla_bytes: float = 656.0
    num_layers: int = 78
    max_num_seqs: int = 8
    topk: int = 2048
    max_model_len: int = 1_048_576

    @property
    def available_cache_bytes(self) -> float:
        # The document's worked example uses decimal GB (1 GB = 1e9 bytes).
        cache_gb = self.hbm_gb * self.memory_fraction - self.weights_gb
        return cache_gb * 1e9


def annotate_ratio_boundary(
    ax, ratio: float, color: str, text: str | None = None
) -> None:
    """Write a vertical boundary's ratio directly against the x-axis."""
    ax.annotate(
        text or f"{ratio:.3g}",
        xy=(ratio, 0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=90,
        color=color,
        fontsize=29,
        fontweight="bold",
        clip_on=True,
        zorder=10,
    )


def gpu_buffer_upper(
    host_ratio: np.ndarray | float, shared_ratio: float, config: CapacityConfig
) -> np.ndarray:
    """Maximum N_T_Buffer allowed by the GPU-memory inequality."""
    host_ratio = np.asarray(host_ratio, dtype=float)
    bytes_per_slot = (
        host_ratio * config.indexer_bytes / shared_ratio + config.mla_bytes
    )
    return config.available_cache_bytes / (
        bytes_per_slot * config.num_layers * config.max_num_seqs
    )


def host_capacity_upper(
    host_ratio: np.ndarray | float, shared_ratio: float, config: CapacityConfig
) -> np.ndarray:
    """GPU-limited per-sequence host capacity, C_host_seq."""
    host_ratio = np.asarray(host_ratio, dtype=float)
    return host_ratio * gpu_buffer_upper(host_ratio, shared_ratio, config)


def topk_ratio_upper(shared_ratio: float, config: CapacityConfig) -> float:
    """Largest R_host for which the GPU buffer can still hold N_topk."""
    slots = config.num_layers * config.max_num_seqs
    return (
        shared_ratio
        * (
            config.available_cache_bytes / (slots * config.topk)
            - config.mla_bytes
        )
        / config.indexer_bytes
    )


def feasible_ratio_interval(
    shared_ratio: float, config: CapacityConfig
) -> tuple[float, float] | None:
    """Return the exact R_host interval where a feasible capacity exists."""
    available = config.available_cache_bytes
    slots = config.num_layers * config.max_num_seqs

    # Worker-wide C_gpu approaches this value as R_host -> infinity.  If it is
    # no larger than the model context, the capacity lower bound is unreachable.
    asymptotic_capacity = (
        available
        * shared_ratio
        / (config.num_layers * config.indexer_bytes)
    )
    if asymptotic_capacity <= config.max_model_len:
        return None

    # Solve C_Batch_gpu(R_host) = L_max_model for the lower endpoint.  Since
    # C_Batch_gpu = N_bs * C_host_seq_gpu, the per-sequence target here is
    # L_max_model / N_bs rather than L_max_model.
    per_sequence_target = config.max_model_len / config.max_num_seqs
    lower_denominator = available - (
        per_sequence_target
        * slots
        * config.indexer_bytes
        / shared_ratio
    )
    lower = (
        per_sequence_target
        * slots
        * config.mla_bytes
        / lower_denominator
    )

    # N_buffer_gpu >= N_topk gives one upper bound.  The logical namespace
    # upper bound R_host*N_topk <= L_max_model*N_bs_max gives the other.
    upper_sparse = topk_ratio_upper(shared_ratio, config)
    upper_namespace = (
        config.max_model_len * config.max_num_seqs / config.topk
    )
    upper = min(upper_sparse, upper_namespace)

    lower = max(0.0, lower)
    return (lower, upper) if upper >= lower else None


def plot_capacity(
    config: CapacityConfig,
    shared_ratios: tuple[float, ...],
    max_num_seqs_values: tuple[int, ...],
    ratio_min: float,
    ratio_max: float | None,
    output: Path,
    show: bool,
) -> None:
    if config.available_cache_bytes <= 0:
        raise ValueError("available cache memory must be positive")
    if ratio_min <= 0:
        raise ValueError("require ratio_min > 0 for the base-2 log axis")
    if ratio_max is not None and ratio_max <= ratio_min:
        raise ValueError("require ratio_max > ratio_min")
    if not shared_ratios:
        raise ValueError("shared_ratios must not be empty")
    if len(shared_ratios) != 1:
        raise ValueError(
            "the combined capacity plot requires exactly one shared ratio"
        )
    if not max_num_seqs_values or any(value < 1 for value in max_num_seqs_values):
        raise ValueError("max_num_seqs values must be positive integers")
    if len(max_num_seqs_values) > len(_BS_COLORS):
        raise ValueError(
            f"at most {len(_BS_COLORS)} max_num_seqs values are supported"
        )

    shared_ratio = shared_ratios[0]
    reference_config = replace(config, max_num_seqs=1)
    max_reachable_ratio = topk_ratio_upper(shared_ratio, reference_config)
    auto_ratio_max = max_reachable_ratio * 1.25
    plot_ratio_max = (
        auto_ratio_max
        if ratio_max is None
        else min(ratio_max, auto_ratio_max)
    )
    if plot_ratio_max <= ratio_min:
        plot_ratio_max = ratio_max if ratio_max is not None else ratio_min * 2
    topk_limits = {
        max_num_seqs: topk_ratio_upper(
            shared_ratio,
            replace(config, max_num_seqs=max_num_seqs),
        )
        for max_num_seqs in max_num_seqs_values
    }
    visible_topk_limits = [
        limit
        for limit in topk_limits.values()
        if ratio_min <= limit <= plot_ratio_max
    ]
    ratios = np.geomspace(ratio_min, plot_ratio_max, 4000)
    if visible_topk_limits:
        ratios = np.unique(
            np.concatenate((ratios, np.asarray(visible_topk_limits)))
        )
    boundary_indices = [
        int(np.searchsorted(ratios, limit)) for limit in visible_topk_limits
    ]

    fig, (ax, hbm_ax) = plt.subplots(1, 2, figsize=(36, 10.5))
    capacity_legend_entries: dict[str, object] = {}
    memory_legend_entries: dict[str, object] = {}

    model_handle = ax.axhline(
        config.max_model_len,
        color="tab:gray",
        linestyle="--",
        linewidth=2.8,
        alpha=0.95,
    )
    capacity_legend_entries[r"$L_{max,model}$"] = model_handle

    # C_Batch_gpu is independent of N_bs because the N_bs factor in worker
    # capacity cancels the N_bs factor in the GPU buffer denominator.
    gpu_capacity = host_capacity_upper(
        ratios, shared_ratio, reference_config
    )
    gpu_handle = ax.plot(
        ratios,
        gpu_capacity,
        color="tab:blue",
        marker="P",
        markevery=boundary_indices,
        markersize=12,
        linewidth=3.2,
        label=r"$C^{GPU}_{Batch}$",
    )[0]
    capacity_legend_entries[r"$C^{GPU}_{Batch}$"] = gpu_handle

    ax.set_xlabel(r"$R_{host}$")
    ax.set_ylabel("#tokens", labelpad=20)
    ax.set_title("Capacity and Request Count")
    ax.set_xlim(ratio_min, plot_ratio_max)
    ax.set_xscale("log", base=2)
    ratio_ticks = power_of_two_ticks(
        ratio_min,
        plot_ratio_max,
        include_zero=False,
    )
    ax.set_xticks(ratio_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(power_of_two_formatter))
    capacity_bottom = min(float(np.min(gpu_capacity)), config.max_model_len) * 0.75
    capacity_top = max(float(np.max(gpu_capacity)), config.max_model_len) * 1.8
    ax.set_yscale("log")
    ax.set_ylim(capacity_bottom, capacity_top)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(
        LogFormatterSciNotation(
            base=10,
            labelOnlyBase=False,
            minor_thresholds=(np.inf, np.inf),
        )
    )
    ax.xaxis.set_minor_locator(
        LogLocator(base=2, subs=(1.25, 1.5, 1.75))
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9))
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.14)
    total_buffer_upper = gpu_buffer_upper(
        ratios, shared_ratio, reference_config
    )
    request_ax = ax.twinx()
    request_styles = (
        (2048, "#2A9D8F", "-", "o"),
        (4096, "#E76F51", "--", "s"),
        (6144, "#6D597A", "-.", "D"),
    )
    max_request_curve = None
    for buffer_size, color, linestyle, marker in request_styles:
        request_curve = total_buffer_upper / buffer_size
        request_handle = request_ax.plot(
            ratios,
            request_curve,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=boundary_indices,
            markersize=12,
            linewidth=2.6,
            alpha=0.95,
        )[0]
        capacity_legend_entries[
            rf"$N_{{T,Buffer}}={buffer_size}$"
        ] = request_handle
        max_request_curve = (
            request_curve
            if max_request_curve is None
            else np.maximum(max_request_curve, request_curve)
        )
    request_ax.set_yscale("log", base=2)
    request_ax.set_ylim(
        bottom=1,
        top=2 ** np.ceil(np.log2(float(np.max(max_request_curve)) * 1.1)),
    )
    request_ax.set_ylabel("#requests", labelpad=14)
    request_ax.yaxis.set_major_locator(LogLocator(base=2, numticks=20))
    request_ax.yaxis.set_major_formatter(
        FuncFormatter(decimal_power_of_two_formatter)
    )
    request_ax.yaxis.set_minor_locator(
        LogLocator(base=2, subs=(1.25, 1.5, 1.75))
    )
    request_ax.yaxis.set_minor_formatter(NullFormatter())

    indexer_cost = ratios * config.indexer_bytes / shared_ratio
    total_cost = config.mla_bytes + indexer_cost
    buffer_share = 100.0 * config.mla_bytes / total_cost
    indexer_share = 100.0 * indexer_cost / total_cost
    cache_gb = config.available_cache_bytes / 1e9
    buffer_hbm_gb = cache_gb * buffer_share / 100.0
    indexer_hbm_gb = cache_gb * indexer_share / 100.0
    hbm_areas = hbm_ax.stackplot(
        ratios,
        buffer_hbm_gb,
        indexer_hbm_gb,
        colors=("tab:blue", "tab:gray"),
        alpha=0.55,
    )
    memory_legend_entries["Buffer HBM"] = hbm_areas[0]
    memory_legend_entries["Indexer HBM"] = hbm_areas[1]
    host_memory = gpu_capacity * config.num_layers * config.mla_bytes / 1e9
    memory_ax = hbm_ax.twinx()
    host_memory_handle = memory_ax.plot(
        ratios,
        host_memory,
        color="#264653",
        linestyle="-",
        marker="^",
        markevery=boundary_indices,
        markersize=12,
        linewidth=2.8,
        alpha=0.95,
    )[0]
    memory_legend_entries["Host memory"] = host_memory_handle
    memory_ax.set_yscale("log")
    memory_ax.set_ylim(
        bottom=float(np.min(host_memory)) * 0.8,
        top=float(np.max(host_memory)) * 1.8,
    )
    memory_ax.set_ylabel("HostMemory (GB)", labelpad=14)
    hbm_ax.set_xlabel(r"$R_{host}$")
    hbm_ax.set_ylabel("HBM (GB)", labelpad=12)
    hbm_ax.set_title("Memory Composition")
    hbm_ax.set_xlim(ratio_min, plot_ratio_max)
    hbm_ax.set_ylim(0, cache_gb)
    hbm_ax.set_xscale("log", base=2)
    hbm_ax.set_xticks(ratio_ticks)
    hbm_ax.xaxis.set_major_formatter(
        FuncFormatter(power_of_two_formatter)
    )
    hbm_ax.xaxis.set_minor_locator(
        LogLocator(base=2, subs=(1.25, 1.5, 1.75))
    )
    hbm_ax.xaxis.set_minor_formatter(NullFormatter())
    hbm_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    hbm_ax.set_axisbelow(True)
    hbm_ax.grid(True, which="major", alpha=0.28)
    hbm_ax.grid(True, which="minor", linestyle=":", alpha=0.14)

    fig.suptitle(academic_config_title(config.config_name), fontsize=34, y=0.975)
    legend_options = dict(
        fontsize=26,
        handlelength=1.7,
        handleheight=1.0,
        columnspacing=0.55,
        labelspacing=0.45,
        frameon=True,
    )
    ax.legend(
        capacity_legend_entries.values(),
        capacity_legend_entries.keys(),
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        ncol=1,
        **legend_options,
    )
    hbm_ax.legend(
        memory_legend_entries.values(),
        memory_legend_entries.keys(),
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        ncol=1,
        **legend_options,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8)
    fig.subplots_adjust(wspace=0.26, top=0.91)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")

    for shared_ratio in shared_ratios:
        for max_num_seqs in max_num_seqs_values:
            curve_config = replace(config, max_num_seqs=max_num_seqs)
            interval = feasible_ratio_interval(shared_ratio, curve_config)
            prefix = f"R_Share={shared_ratio:g}, max_num_seqs={max_num_seqs}"
            if interval is None:
                print(f"{prefix}: C_Batch >= L_max R_host interval = empty")
            else:
                print(
                    f"{prefix}: C_Batch >= L_max R_host interval = "
                    f"[{interval[0]:.6g}, {interval[1]:.6g}]"
                )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_buffer(
    config: CapacityConfig,
    shared_ratios: tuple[float, ...],
    max_num_seqs_values: tuple[int, ...],
    ratio_min: float,
    ratio_max: float,
    output: Path,
    show: bool,
) -> None:
    """Plot worker-wide total device-buffer capacity on one axis."""
    if len(shared_ratios) != 1:
        raise ValueError(
            "the combined buffer plot requires exactly one shared ratio"
        )
    ratios = np.geomspace(ratio_min, ratio_max, 4000)
    shared_ratio = shared_ratios[0]
    fig, ax = plt.subplots(figsize=(22, 12))
    colors = _BS_COLORS
    legend_entries: dict[str, object] = {}

    reference_config = replace(config, max_num_seqs=1)
    total_buffer_upper = gpu_buffer_upper(
        ratios, shared_ratio, reference_config
    )
    buffer_handle = ax.plot(
        ratios,
        total_buffer_upper,
        color="tab:blue",
        linewidth=3.4,
        label=rf"total buffer upper, $R_{{Share}}={shared_ratio:g}$",
    )
    legend_entries[
        rf"total buffer upper, $R_{{Share}}={shared_ratio:g}$"
    ] = buffer_handle[0]
    topk_limit_legend_added = False

    for curve_index, max_num_seqs in enumerate(max_num_seqs_values):
        curve_config = replace(config, max_num_seqs=max_num_seqs)
        color = colors[curve_index % len(colors)]
        topk_total = config.topk * max_num_seqs
        topk_label = rf"top-k need, $N_{{bs}}={max_num_seqs}$"
        topk_handle = ax.axhline(
            topk_total,
            color=color,
            linestyle="--",
            linewidth=2.2,
            label=topk_label,
        )
        legend_entries[topk_label] = topk_handle

        topk_limit = topk_ratio_upper(shared_ratio, curve_config)
        if ratio_min <= topk_limit <= ratio_max:
            limit_handle = ax.axvline(
                topk_limit,
                color=color,
                linestyle=":",
                linewidth=2.2,
                alpha=0.9,
                label=(
                    "top-k limits" if not topk_limit_legend_added else None
                ),
            )
            if not topk_limit_legend_added:
                legend_entries["top-k limits"] = limit_handle
                topk_limit_legend_added = True
            annotate_ratio_boundary(
                ax,
                topk_limit,
                color,
                text=rf"$N_{{bs}}={max_num_seqs}$: {topk_limit:.3g}",
            )

    ax.set_xlabel(r"Host/device ratio $R_{host}$")
    ax.set_ylabel(
        "Total device buffer upper\n"
        r"$N^{total}_{T,Buffer}$ (tokens)",
        labelpad=20,
    )
    ax.set_xlim(ratio_min, ratio_max)
    ax.set_xscale("log", base=2)
    ratio_ticks = power_of_two_ticks(
        ratio_min,
        ratio_max,
        include_zero=False,
    )
    ax.set_xticks(ratio_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(power_of_two_formatter))
    ax.set_yscale("log", base=2)
    max_topk_total = config.topk * max(max_num_seqs_values)
    ax.set_ylim(
        bottom=config.topk,
        top=max(total_buffer_upper[0], max_topk_total) * 1.25,
    )
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_major_formatter(FuncFormatter(power_of_two_formatter))
    ax.xaxis.set_minor_locator(
        LogLocator(base=2, subs=(1.25, 1.5, 1.75))
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(
        LogLocator(base=2, subs=(1.25, 1.5, 1.75))
    )
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.14)

    fig.suptitle(academic_config_title(config.config_name), fontsize=48, y=0.98)
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        fontsize=29,
        handlelength=2.5,
        handleheight=1.5,
        columnspacing=0.8,
        labelspacing=0.8,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parents[1] / "imgs" / "glm52_h100_dp16_ep_16.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="glm52_h100_dp16_ep_16")
    parser.add_argument("--hbm-gb", type=float, default=77.47)
    parser.add_argument("--memory-fraction", type=float, default=0.88)
    parser.add_argument("--weights-gb", type=float, default=64.52)
    parser.add_argument("--ratio-min", type=float, default=2**-1)
    parser.add_argument("--ratio-max", type=float)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1_048_576,
        help="Model context length (GLM5.2: 1048576, GLM5.1: 262144).",
    )
    parser.add_argument(
        "--shared-ratios",
        type=float,
        nargs="+",
        default=(3.714,),
        help=(
            "Indexer layer-sharing factor(s); one value per model "
            "(GLM5.2: 3.714 = 78/21 shared Indexer layers, GLM5.1: 1.0)."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--buffer-output", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CapacityConfig(
        config_name=args.config_name,
        hbm_gb=args.hbm_gb,
        memory_fraction=args.memory_fraction,
        weights_gb=args.weights_gb,
        max_model_len=args.max_model_len,
    )
    plot_capacity(
        config=config,
        shared_ratios=tuple(args.shared_ratios),
        max_num_seqs_values=tuple(args.max_num_seqs),
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        output=args.output,
        show=args.show,
    )
    if args.buffer_output:
        plot_buffer(
            config=config,
            shared_ratios=tuple(args.shared_ratios),
            max_num_seqs_values=tuple(args.max_num_seqs),
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            output=args.buffer_output,
            show=args.show,
        )


if __name__ == "__main__":
    main()

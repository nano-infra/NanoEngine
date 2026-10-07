#!/usr/bin/env python3
"""Shared external FlashMLA decoding microbenchmark for Fig. 3 and Fig. 5.

仅保留 MLA 计算路径，不涉及 SP/all2all/copy 相关逻辑。
默认按 total_token_num x batch_size 生成测试用例，输出 CSV 列：
seq_len,batch_size,total_token_num,time_us,time_us_p10,time_us_p90

支持从同格式 CSV 读 cache，只补测缺失点，并把 cache + 新结果合并写入一个新 CSV。
默认会同时生成一张总图（PNG/PDF）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from triton.testing import do_bench_cudagraph

flash_mla = None
flash_mla_with_kvcache = None
get_mla_metadata = None


CSV_FIELDNAMES = [
    "seq_len",
    "batch_size",
    "total_token_num",
    "time_us",
    "time_us_p10",
    "time_us_p90",
]
PLOT_COLORS = ["#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#9C755F", "#B07AA1", "#76B7B2"]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
PLOT_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_flash_mla() -> None:
    global flash_mla, flash_mla_with_kvcache, get_mla_metadata
    try:
        import flash_mla as loaded_flash_mla
    except ImportError as exc:
        raise ImportError(
            "The external flash_mla package is required and must match the "
            "active PyTorch/CUDA environment."
        ) from exc
    try:
        flash_mla_with_kvcache = loaded_flash_mla.flash_mla_with_kvcache
        get_mla_metadata = loaded_flash_mla.get_mla_metadata
    except AttributeError as exc:
        raise ImportError(
            "The imported flash_mla package does not expose the dense decode API."
        ) from exc
    flash_mla = loaded_flash_mla


def row_key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["seq_len"]),
        int(row["batch_size"]),
        int(row["total_token_num"]),
    )


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["total_token_num"]),
            int(row["batch_size"]),
            int(row["seq_len"]),
        ),
    )


def build_benchmark_cases(total_tokens, batch_sizes, seq_lens=None):
    if seq_lens is not None:
        cases = []
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                cases.append(
                    {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "total_token_num": seq_len * batch_size,
                    }
                )
        return dedupe_cases(cases)

    cases = []
    for total_token_num in total_tokens:
        if total_token_num < 1:
            raise ValueError("total_token_num must be >= 1")
        for batch_size in batch_sizes:
            if total_token_num % batch_size != 0:
                raise ValueError(
                    f"total_token_num={total_token_num} is not divisible by batch_size={batch_size}"
                )
            cases.append(
                {
                    "seq_len": total_token_num // batch_size,
                    "batch_size": batch_size,
                    "total_token_num": total_token_num,
                }
            )
    return dedupe_cases(cases)


def dedupe_cases(cases: list[dict]) -> list[dict]:
    seen = set()
    unique_cases = []
    for case in cases:
        key = row_key(case)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)
    return unique_cases


def load_cached_results(cache_csv: str | None) -> dict[tuple[int, int, int], dict]:
    if cache_csv is None:
        return {}

    cache_path = Path(cache_csv).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache CSV not found: {cache_path}")

    cache = {}
    with cache_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in CSV_FIELDNAMES if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{cache_path.name} missing required columns: {missing}")

        for raw_row in reader:
            row = {
                "seq_len": int(float(raw_row["seq_len"])),
                "batch_size": int(float(raw_row["batch_size"])),
                "total_token_num": int(float(raw_row["total_token_num"])),
                "time_us": float(raw_row["time_us"]),
                "time_us_p10": float(raw_row["time_us_p10"]),
                "time_us_p90": float(raw_row["time_us_p90"]),
            }
            cache[row_key(row)] = row

    return cache


def merge_results(cached_rows: dict[tuple[int, int, int], dict], fresh_rows: list[dict]) -> list[dict]:
    merged = dict(cached_rows)
    for row in fresh_rows:
        merged[row_key(row)] = row
    return sort_rows(list(merged.values()))


def write_results_csv(output_path: Path, rows: list[dict]) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def format_token_axis(value, _pos):
    value = int(round(value))
    if value >= 1024**2:
        scaled = value / (1024**2)
        return f"{int(scaled)}M" if float(scaled).is_integer() else f"{scaled:.1f}M"
    if value >= 1024:
        scaled = value / 1024
        return f"{int(scaled)}k" if float(scaled).is_integer() else f"{scaled:.1f}k"
    return str(value)


def plot_results(rows: list[dict], output_path: Path, title: str, use_log_x: bool) -> tuple[Path, Path]:
    mpl_config_dir = Path("/tmp/matplotlib-codex")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")

    from ae_utils.plotting import get_plot_font_family

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": get_plot_font_family(),
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})

    for idx, batch_size in enumerate(batch_sizes):
        sub_rows = [row for row in rows if int(row["batch_size"]) == batch_size]
        sub_rows = sort_rows(sub_rows)
        x_values = [int(row["total_token_num"]) for row in sub_rows]
        y_values = [float(row["time_us"]) for row in sub_rows]
        ax.plot(
            x_values,
            y_values,
            marker=PLOT_MARKERS[idx % len(PLOT_MARKERS)],
            markersize=5,
            linewidth=1.8,
            linestyle=PLOT_LINESTYLES[idx % len(PLOT_LINESTYLES)],
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            label=f"BS={batch_size}",
            markerfacecolor="white",
            markeredgewidth=0.9,
        )

    token_values = sorted({int(row["total_token_num"]) for row in rows})
    if use_log_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(token_values)
    ax.xaxis.set_major_formatter(FuncFormatter(format_token_axis))
    ax.set_xlim(min(token_values), max(token_values))

    time_values = [float(row["time_us"]) for row in rows]
    y_min = min(time_values)
    y_max = max(time_values)
    ax.set_ylim(max(0.0, y_min * 0.95), y_max * 1.08)

    ax.set_title(title)
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Median Time (us)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Batch Size", frameon=False, ncol=2)

    plt.tight_layout()

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def summarize_latency_samples(latency_us_samples: list[float]) -> dict[str, float]:
    if not latency_us_samples:
        raise ValueError("latency_us_samples must not be empty")

    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    samples = torch.tensor(latency_us_samples, dtype=torch.float64)
    p10, median, p90 = torch.quantile(samples, quantiles).tolist()
    return {
        "time_us": float(median),
        "time_us_p10": float(p10),
        "time_us_p90": float(p90),
    }


def build_inputs(
    seq_len: int,
    batch_size: int,
    num_heads: int = 128,
    head_dim: int = 576,
    num_kv_heads: int = 1,
    v_head_dim: int = 512,
    block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if head_dim != 576:
        raise ValueError("MLA decoding in this benchmark assumes head_dim == 576")
    if v_head_dim != 512:
        raise ValueError("MLA decoding in this benchmark assumes v_head_dim == 512")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype, device=device)
    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    block_table = torch.arange(
        batch_size * num_blocks_per_seq,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, num_blocks_per_seq)

    blocked_k = torch.randn(
        batch_size * num_blocks_per_seq,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    tail = seq_len % block_size
    if tail != 0:
        for i in range(batch_size):
            blk_idx = i * num_blocks_per_seq + (num_blocks_per_seq - 1)
            blocked_k[blk_idx, tail:block_size] = float("nan")

    num_query_heads_per_kv = num_heads // num_kv_heads
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        num_query_heads_per_kv,
        num_kv_heads,
    )
    scale = 1.0 / (head_dim**0.5)

    return {
        "q": q,
        "blocked_k": blocked_k,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "tile_scheduler_metadata": tile_scheduler_metadata,
        "num_splits": num_splits,
        "v_head_dim": v_head_dim,
        "scale": scale,
    }


@torch.inference_mode()
def benchmark_with_cudagraph(
    inputs: dict,
    rep_ms: int = 200,
    num_warmup: int = 10,
    num_bench_repeats: int = 10,
) -> dict[str, float]:
    """
    Run multiple independent CUDA Graph benchmarks and summarize latency in us.
    """
    if rep_ms < 1:
        raise ValueError("rep_ms must be >= 1")
    if num_warmup < 0:
        raise ValueError("num_warmup must be >= 0")
    if num_bench_repeats < 1:
        raise ValueError("num_bench_repeats must be >= 1")

    def run_kernel():
        return flash_mla_with_kvcache(
            inputs["q"],
            inputs["blocked_k"],
            inputs["block_table"],
            inputs["cache_seqlens"],
            inputs["v_head_dim"],
            inputs["tile_scheduler_metadata"],
            inputs["num_splits"],
            inputs["scale"],
            causal=True,
        )

    for _ in range(num_warmup):
        run_kernel()
    torch.cuda.synchronize()

    latency_us_samples = []
    for _ in range(num_bench_repeats):
        torch.cuda.synchronize()
        latency_ms = do_bench_cudagraph(
            run_kernel,
            rep=rep_ms,
            return_mode="median",
        )
        latency_us_samples.append(float(latency_ms) * 1000.0)

    return summarize_latency_samples(latency_us_samples)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "External flash_mla decoding benchmark with CUDA Graph timing "
            "(no vLLM, SP, all2all, or copy)"
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--total_tokens",
        type=str,
        help="Comma-separated total token counts; derive seq_len = total_tokens / batch_size",
    )
    input_group.add_argument(
        "--seq_lens",
        type=str,
        help="Explicit sequence lengths; run their Cartesian product with batch_sizes",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        required=True,
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--cache-csv",
        type=str,
        default=None,
        help="Existing CSV to use as cache; cached cases are skipped and merged into the new output.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=128,
        help="Number of query heads",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=576,
        help="MLA Q/K head dim (kv_lora_rank + rope_dim)",
    )
    parser.add_argument(
        "--v_head_dim",
        type=int,
        default=512,
        help="MLA V head dim (kv_lora_rank)",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=200,
        help="Target timing window in milliseconds for each independent CUDA Graph benchmark",
    )
    parser.add_argument(
        "--bench-repeats",
        type=int,
        default=10,
        help="Number of independent CUDA Graph benchmarks per case",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Eager warmup iterations before CUDA Graph benchmarking",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default auto-generated)",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not generate PNG/PDF plots.",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear X axis instead of log2 for plotting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_flash_mla()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(
            "benchmark_results",
            f"external_flashmla_cudagraph_total_tokens_{ts}.csv",
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = (
        parse_int_list(args.total_tokens) if args.total_tokens is not None else []
    )
    seq_lens = parse_int_list(args.seq_lens) if args.seq_lens is not None else None
    batch_sizes = parse_int_list(args.batch_sizes)

    if args.seq_lens is None and not total_tokens:
        raise ValueError("total_tokens is empty")
    if args.seq_lens is not None and not seq_lens:
        raise ValueError("seq_lens is empty")
    if not batch_sizes:
        raise ValueError("batch_sizes is empty")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(0)

    module_path = Path(flash_mla.__file__).resolve()
    print(f"External FlashMLA module: {module_path}")

    dtype = torch.bfloat16
    cases = build_benchmark_cases(
        total_tokens=total_tokens,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
    )

    cached_rows = load_cached_results(args.cache_csv)
    pending_cases = [case for case in cases if row_key(case) not in cached_rows]

    print(
        f"Planned cases: {len(cases)} | cached: {len(cases) - len(pending_cases)} | "
        f"to run: {len(pending_cases)}"
    )
    if args.cache_csv is not None:
        print(f"Using cache CSV: {Path(args.cache_csv).resolve()}")

    fresh_rows = []
    for idx, case in enumerate(pending_cases, start=1):
        print(
            f"[{idx}/{len(pending_cases)}] Running benchmark: "
            f"total_token_num={case['total_token_num']}, "
            f"batch_size={case['batch_size']}, seq_len={case['seq_len']}"
        )
        inputs = build_inputs(
            seq_len=case["seq_len"],
            batch_size=case["batch_size"],
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_kv_heads=1,
            v_head_dim=args.v_head_dim,
            dtype=dtype,
        )
        latency_stats = benchmark_with_cudagraph(
            inputs=inputs,
            rep_ms=args.rep_ms,
            num_warmup=args.warmup,
            num_bench_repeats=args.bench_repeats,
        )
        row = {
            "seq_len": case["seq_len"],
            "batch_size": case["batch_size"],
            "total_token_num": case["total_token_num"],
            **latency_stats,
        }
        print(
            f"  median={row['time_us']:.3f} us | "
            f"p10={row['time_us_p10']:.3f} us | "
            f"p90={row['time_us_p90']:.3f} us"
        )
        fresh_rows.append(row)

    merged_rows = merge_results(cached_rows, fresh_rows)
    write_results_csv(output_path, merged_rows)
    print(f"Saved merged results to {output_path}")

    if not args.skip_plot:
        plot_output_path = output_path.with_name(f"{output_path.stem}_plot")
        plot_title = "External FlashMLA CUDA Graph Median Latency vs Total Tokens"
        pdf_path, png_path = plot_results(
            rows=merged_rows,
            output_path=plot_output_path,
            title=plot_title,
            use_log_x=not args.linear_x,
        )
        print(f"Saved plot PDF: {pdf_path}")
        print(f"Saved plot PNG: {png_path}")


if __name__ == "__main__":
    main()

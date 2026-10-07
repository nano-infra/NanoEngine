#!/usr/bin/env python3
"""Run, validate, and plot Fig. 18 with DLSlime Hao Basic."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig18")

import matplotlib

matplotlib.use("Agg")

from ae_utils.plotting import get_plot_font_family

import matplotlib.pyplot as plt


PAPER_BATCH_SIZES = (2, 4, 8, 16, 32, 64, 128)
PAPER_FEATURE_SIZE = 73_728
PAPER_WORLD_SIZE = 8
PAPER_DTYPE = "bfloat16"
DLSLIME_IMPLEMENTATION = "dlslime_hao_basic"
NCCL_IMPLEMENTATION = "nccl"
IMPLEMENTATION_ORDER = (DLSLIME_IMPLEMENTATION, NCCL_IMPLEMENTATION)
EXPECTED_BACKENDS = {
    DLSLIME_IMPLEMENTATION: ("dlslime", "basic"),
    NCCL_IMPLEMENTATION: ("pytorch_nccl", "all_to_all_single"),
}
BENCHMARK_SCRIPT = Path(__file__).resolve().with_name(
    "benchmark_fig18_dlslime.py"
)
IMPLEMENTATION_LABELS = {
    "intra": "Ours",
    "nccl": "NCCL",
}
IMPLEMENTATION_COLORS = {
    "intra": "#C85A54",
    "nccl": "#4C78A8",
}
FIG_WIDTH = 7.0
FIG_HEIGHT = 2.5
FONT_SIZE = 14
TICK_FONT_SIZE = 12
XTICK_FONT_SIZE = 13
LABEL_FONT_SIZE = 16
Y_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 11


@dataclass(frozen=True)
class Measurement:
    batch_size: int
    implementation: str
    latency_us: float
    bandwidth_gbps: float


def parse_args() -> argparse.Namespace:
    artifact_dir = Path(__file__).resolve().parent
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Run the eight-GPU Fig. 18 benchmark with DLSlime Hao Basic and "
            "NCCL, strictly validate the fresh result, and plot it."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=(
            artifact_dir
            / "results"
            / "dlslime_hao_basic"
            / f"run_{run_id}_pid{os.getpid()}"
        ),
        help="Fresh CSV/JSON output directory for this run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "fig18_dlslime",
        help="Figure base path, or an explicit .pdf/.png path.",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        help="Skip the GPU benchmark and plot this existing DLSlime CSV.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the distributed rank processes.",
    )
    parser.add_argument(
        "--require-dlslime-root",
        type=Path,
        help=(
            "Require the imported dlslime package to resolve under this "
            "source checkout."
        ),
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29518)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    for name in ("warmup_iters", "iters", "rounds", "dpi"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def resolve_python(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise FileNotFoundError(f"Python executable not found: {candidate}")
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Python executable not found on PATH: {value}")
    return resolved


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(f"\n+ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def query_cuda_environment(python: str) -> dict[str, object]:
    code = """
import json
import torch
print(json.dumps({
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
"""
    completed = subprocess.run(
        [python, "-c", code],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    info = json.loads(completed.stdout.strip().splitlines()[-1])
    gpu_count = int(info["gpu_count"])
    if gpu_count != PAPER_WORLD_SIZE:
        raise RuntimeError(
            f"Fig. 18 requires exactly {PAPER_WORLD_SIZE} visible GPUs; "
            f"PyTorch reports {gpu_count}."
        )
    print(
        "Environment: "
        f"PyTorch {info['pytorch']}, CUDA {info['cuda']}, "
        f"{gpu_count} GPUs ({', '.join(str(gpu) for gpu in info['gpus'])})",
        flush=True,
    )
    return info


def prepare_result_root(path: Path) -> Path:
    result_root = path.expanduser().resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty result directory: {result_root}. "
            "Choose a new --result-root."
        )
    result_root.mkdir(parents=True, exist_ok=True)
    return result_root


def parse_int(row: dict[str, str], key: str, input_path: Path) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {key!r} in {input_path}: {row}") from error


def parse_float(row: dict[str, str], key: str, input_path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric {key!r} in {input_path}: {row}") from error
    if value <= 0:
        raise ValueError(f"Metric {key!r} must be positive in {input_path}: {value}")
    return value


def query_dlslime_environment(
    python: str, required_root: Path | None
) -> dict[str, object]:
    code = """
import json
from pathlib import Path
import dlslime

dlslime_file = Path(dlslime.__file__).resolve()
required_root = Path(__import__('sys').argv[1]).resolve() if __import__('sys').argv[1] else None
if required_root is not None:
    if not required_root.is_dir():
        raise FileNotFoundError(f"DLSlime source root not found: {required_root}")
    try:
        dlslime_file.relative_to(required_root)
    except ValueError as error:
        raise RuntimeError(
            f"Resolved dlslime to {dlslime_file}, outside required root {required_root}"
        ) from error

kernel_impl = getattr(dlslime, "KernelImpl", None)
print(json.dumps({
    "dlslime_file": str(dlslime_file),
    "has_all_to_all_buffer": hasattr(dlslime, "AllToAllBuffer"),
    "has_kernel_impl_basic": (
        kernel_impl is not None and hasattr(kernel_impl, "Basic")
    ),
}))
"""
    root_arg = "" if required_root is None else str(required_root.expanduser().resolve())
    completed = subprocess.run(
        [python, "-c", code, root_arg],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    info = json.loads(completed.stdout.strip().splitlines()[-1])
    if not info["has_all_to_all_buffer"] or not info["has_kernel_impl_basic"]:
        raise RuntimeError(
            "The selected DLSlime build does not expose AllToAllBuffer and "
            "KernelImpl.Basic. Rebuild it with CUDA intra-node ops enabled."
        )
    print(f"DLSlime: {info['dlslime_file']} (AllToAllBuffer/Basic available)")
    return info


def run_benchmark(
    args: argparse.Namespace,
    python: str,
) -> Path:
    if not BENCHMARK_SCRIPT.is_file():
        raise FileNotFoundError(f"Benchmark script not found: {BENCHMARK_SCRIPT}")

    result_root = prepare_result_root(args.result_root)
    csv_path = result_root / "fig18_dlslime.csv"
    json_path = result_root / "fig18_dlslime.json"
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        f"--nproc-per-node={PAPER_WORLD_SIZE}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        str(BENCHMARK_SCRIPT),
        "--batch-sizes",
        ",".join(str(value) for value in PAPER_BATCH_SIZES),
        "--feature-size",
        str(PAPER_FEATURE_SIZE),
        "--dtype",
        "bf16",
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters),
        "--rounds",
        str(args.rounds),
        "--check",
        "--csv-output",
        str(csv_path),
        "--json-output",
        str(json_path),
    ]
    if args.require_dlslime_root is not None:
        command.extend(
            [
                "--require-dlslime-root",
                str(args.require_dlslime_root.expanduser().resolve()),
            ]
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run_command(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )
    if not csv_path.is_file() or not json_path.is_file():
        raise RuntimeError(
            f"Expected fresh outputs {csv_path} and {json_path}, but one is missing."
        )
    print(f"Fresh benchmark CSV: {csv_path}")
    print(f"Fresh benchmark JSON: {json_path}")
    return csv_path


def load_measurements(input_path: Path) -> list[Measurement]:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "mode",
            "implementation",
            "backend",
            "kernel_impl",
            "dlslime_file",
            "world_size",
            "batch_size",
            "feature_size",
            "dtype",
            "checked",
            "e2e_p50_us",
            "effective_gbps_per_rank_p50",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Missing CSV columns in {input_path}: {missing}")
        rows = list(reader)

    measurements: list[Measurement] = []
    seen: set[tuple[int, str]] = set()
    dlslime_files: set[str] = set()
    for row in rows:
        implementation = row["implementation"]
        if implementation not in IMPLEMENTATION_ORDER:
            raise ValueError(
                f"Unexpected implementation {implementation!r} in {input_path}"
            )
        expected_backend, expected_kernel = EXPECTED_BACKENDS[implementation]
        if row["backend"] != expected_backend or row["kernel_impl"] != expected_kernel:
            raise ValueError(
                f"Unexpected backend identity for {implementation}: "
                f"backend={row['backend']!r}, kernel_impl={row['kernel_impl']!r}"
            )
        if not row["dlslime_file"].strip():
            raise ValueError("The CSV must record the imported dlslime module path")
        dlslime_files.add(row["dlslime_file"])
        if row["mode"] != "alltoall":
            raise ValueError(f"Expected mode=alltoall, found {row['mode']!r}")

        world_size = parse_int(row, "world_size", input_path)
        feature_size = parse_int(row, "feature_size", input_path)
        if world_size != PAPER_WORLD_SIZE:
            raise ValueError(
                f"Expected world_size={PAPER_WORLD_SIZE}, "
                f"found {world_size}"
            )
        if feature_size != PAPER_FEATURE_SIZE:
            raise ValueError(
                f"Expected feature_size={PAPER_FEATURE_SIZE}, "
                f"found {feature_size}"
            )
        if row["dtype"] != PAPER_DTYPE:
            raise ValueError(
                f"Expected dtype={PAPER_DTYPE}, found {row['dtype']!r}"
            )
        if row["checked"].strip().lower() not in {"true", "1", "yes"}:
            raise ValueError("The benchmark CSV must come from a correctness-check run")

        batch_size = parse_int(row, "batch_size", input_path)
        key = (batch_size, implementation)
        if key in seen:
            raise ValueError(f"Duplicate benchmark row: BS{batch_size}/{implementation}")
        seen.add(key)
        normalized_implementation = (
            "intra" if implementation == DLSLIME_IMPLEMENTATION else "nccl"
        )
        measurements.append(
            Measurement(
                batch_size=batch_size,
                implementation=normalized_implementation,
                latency_us=parse_float(row, "e2e_p50_us", input_path),
                bandwidth_gbps=parse_float(
                    row, "effective_gbps_per_rank_p50", input_path
                ),
            )
        )

    expected = {
        (batch_size, implementation)
        for batch_size in PAPER_BATCH_SIZES
        for implementation in IMPLEMENTATION_ORDER
    }
    missing_rows = sorted(expected - seen)
    unexpected_rows = sorted(seen - expected)
    if missing_rows or unexpected_rows:
        raise ValueError(
            "Incomplete or unexpected DLSlime Fig. 18 rows: "
            f"missing={missing_rows}, unexpected={unexpected_rows}"
        )
    if len(dlslime_files) != 1:
        raise ValueError(
            f"Expected one dlslime module path across the CSV; found {dlslime_files}"
        )
    print(f"Validated DLSlime provenance: {next(iter(dlslime_files))}")
    return measurements


def configure_matplotlib() -> str:
    font_name = get_plot_font_family()
    matplotlib.rcParams.update(
        {
            "font.family": font_name,
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.linewidth": 0.9,
            "xtick.labelsize": XTICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "hatch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font_name


def resolve_output_base(output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() in {".pdf", ".png"}:
        return output.with_suffix("")
    return output


def draw_panel(
    ax: plt.Axes,
    batch_sizes: Sequence[int],
    ours: Sequence[float],
    nccl: Sequence[float],
    y_label: str,
    font_name: str,
    speedups: Sequence[float],
) -> None:
    x = list(range(len(batch_sizes)))
    width = 0.36
    ax.bar(
        [value - width / 2 for value in x],
        ours,
        width=width,
        color=IMPLEMENTATION_COLORS["intra"],
        edgecolor="#303030",
        linewidth=0.35,
        label=IMPLEMENTATION_LABELS["intra"],
        zorder=3,
    )
    ax.bar(
        [value + width / 2 for value in x],
        nccl,
        width=width,
        color=IMPLEMENTATION_COLORS["nccl"],
        edgecolor="#303030",
        linewidth=0.35,
        hatch="///",
        label=IMPLEMENTATION_LABELS["nccl"],
        zorder=3,
    )

    y_max = max(max(ours), max(nccl))
    annotation_offset = y_max * 0.022
    for group_x, ours_value, nccl_value, speedup in zip(
        x, ours, nccl, speedups
    ):
        ax.text(
            group_x,
            max(ours_value, nccl_value) + annotation_offset,
            f"{speedup:.2f}$\\times$",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            fontfamily=font_name,
            color="#202020",
            clip_on=False,
        )

    ax.set_xlabel(
        "Batch Size",
        fontsize=LABEL_FONT_SIZE,
        fontweight="semibold",
        fontfamily=font_name,
        labelpad=2,
    )
    ax.set_ylabel(
        y_label,
        fontsize=Y_LABEL_FONT_SIZE,
        fontweight="semibold",
        fontfamily=font_name,
        labelpad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in batch_sizes])
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(0, y_max * 1.20)
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        color="#E6E9EF",
        alpha=1.0,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=XTICK_FONT_SIZE, pad=4)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily(font_name)


def plot_measurements(
    measurements: Sequence[Measurement],
    output: Path,
    dpi: int,
) -> tuple[Path, Path]:
    by_key = {
        (measurement.batch_size, measurement.implementation): measurement
        for measurement in measurements
    }
    batch_sizes = list(PAPER_BATCH_SIZES)
    ours_latency = [by_key[batch_size, "intra"].latency_us for batch_size in batch_sizes]
    nccl_latency = [by_key[batch_size, "nccl"].latency_us for batch_size in batch_sizes]
    ours_bandwidth = [
        by_key[batch_size, "intra"].bandwidth_gbps for batch_size in batch_sizes
    ]
    nccl_bandwidth = [
        by_key[batch_size, "nccl"].bandwidth_gbps for batch_size in batch_sizes
    ]
    speedups = [
        nccl_value / ours_value
        for ours_value, nccl_value in zip(ours_latency, nccl_latency)
    ]

    print("\nFig. 18 validated P50 results:")
    print(f"{'Batch':>5} {'Ours (us)':>11} {'NCCL (us)':>11} {'Speedup':>9}")
    for batch_size, ours_value, nccl_value, speedup in zip(
        batch_sizes, ours_latency, nccl_latency, speedups
    ):
        print(
            f"{batch_size:>5} {ours_value:>11.2f} {nccl_value:>11.2f} "
            f"{speedup:>8.2f}x"
        )

    font_name = configure_matplotlib()
    fig, axes = plt.subplots(ncols=2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.92, bottom=0.18, wspace=0.30)
    draw_panel(
        axes[0],
        batch_sizes,
        ours_latency,
        nccl_latency,
        "Latency ($\\mu$s)",
        font_name,
        speedups,
    )
    draw_panel(
        axes[1],
        batch_sizes,
        ours_bandwidth,
        nccl_bandwidth,
        "Bandwidth (GB/s)",
        font_name,
        speedups,
    )
    axes[0].legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        prop={"family": font_name, "size": LEGEND_FONT_SIZE},
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    output_base = resolve_output_base(output)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    try:
        if args.plot_only is not None:
            csv_path = args.plot_only
        else:
            python = resolve_python(args.python)
            query_dlslime_environment(python, args.require_dlslime_root)
            query_cuda_environment(python)
            csv_path = run_benchmark(args, python)
        measurements = load_measurements(csv_path)
        plot_measurements(measurements, args.output, args.dpi)
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

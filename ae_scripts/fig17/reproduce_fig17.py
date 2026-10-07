#!/usr/bin/env python3
"""Run the four-node NanoDeploy experiments and reproduce Fig. 17."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PAPER_BATCH_SIZES = (32, 64, 96, 128, 160, 192, 256)
PAPER_WORLD_SIZE = 32
PAPER_NODE_COUNT = 4
DEFAULT_RAY_ADDRESS = "10.102.252.174:6380"
FIG17_DIR = Path(__file__).resolve().parent
AE_ROOT = FIG17_DIR.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

BENCHMARK_SCRIPT = FIG17_DIR / "bench_serving_overhead.py"
DEFAULT_MODEL = Path(require_path("AE_DPSK_MODEL"))
DEFAULT_RDMA_DEVICES = ",".join(f"mlx5_{index}" for index in range(8))
NANODEPLOY_ENVIRONMENT_DEFAULTS = {
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_IB_HCA": f"={DEFAULT_RDMA_DEVICES}",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TC": "186",
    "SLIME_VISIBLE_DEVICES": DEFAULT_RDMA_DEVICES,
    "SLIME_GID_INDEX": "3",
    "SLIME_QP_NUM": "4",
    "DEEPEP_SMS": "16",
    "DEEPEP_MAX_TOKENS_PER_RANK": "256",
    "DEEPEP_ENABLE_MNNVL": "0",
    "DEEPEP_MODE": "auto",
    "NVSHMEM_QP_DEPTH": "1024",
}


@dataclass(frozen=True)
class Parallelism:
    name: str
    dp: int
    sp: int


@dataclass(frozen=True)
class CaseResult:
    strategy: str
    batch_size: int
    log_path: str
    json_path: str
    command: list[str]


@dataclass(frozen=True)
class LogIdentity:
    strategy: str
    batch_size: int
    loop_count: int


@dataclass(frozen=True)
class Measurement:
    strategy: str
    batch_size: int
    loop_count: int
    log_path: Path
    itl_p50_ms: float
    schedule_p50_ms: float
    post_schedule_p50_ms: float
    transfer_p50_ms: float

    @property
    def schedule_per_step_ms(self) -> float:
        return (self.schedule_p50_ms + self.post_schedule_p50_ms) / self.loop_count

    @property
    def transfer_per_step_ms(self) -> float:
        return self.transfer_p50_ms / self.loop_count

    @property
    def model_exec_ms(self) -> float:
        return max(
            0.0,
            self.itl_p50_ms
            - self.schedule_per_step_ms
            - self.transfer_per_step_ms,
        )

    @property
    def overhead_pct(self) -> float:
        if self.itl_p50_ms <= 0:
            return 0.0
        overhead = self.schedule_per_step_ms + self.transfer_per_step_ms
        return overhead / self.itl_p50_ms * 100.0


PARALLELISMS = {
    "DP32": Parallelism("DP32", dp=32, sp=1),
    "DP4SP8": Parallelism("DP4SP8", dp=4, sp=8),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"'itl':\s*'[\d.]+ms'.*'sch_ovhd':\s*'[\d.]+ms'.*"
    r"'post_sch_ovhd':\s*'[\d.]+ms'"
)
TRANSFER_RE = re.compile(r"Input Transfer Latency:\s*[\d.]+\s*ms")
DP_RE = re.compile(r"(?:^|_)DP(\d+)(?:_|$)")
SP_RE = re.compile(r"(?:^|_)SP(\d+)(?:_|$)")
BS_RE = re.compile(r"(?:^|_)BS(\d+)(?:_|$)")
LOOP_RE = re.compile(r"(?:^|_)loop(\d+)(?:_|$)", re.IGNORECASE)
RANK_TRANSFER_RE = re.compile(
    r"Rank \d+ Input Transfer Latency:\s*([\d.]+)\s*ms"
)
GLOBAL_TRANSFER_RE = re.compile(
    r"\[METRIC\] Input Transfer Latency:\s*([\d.]+)\s*ms"
)
SUMMARY_P50_RE = re.compile(r"^\s*P50:\s*([\d.]+)")
STRATEGY_ORDER = ("DP32", "DP4SP8")
STRATEGY_CONFIGS = {
    (32, 1): "DP32",
    (4, 8): "DP4SP8",
}
COLORS = {
    "DP32": {"bar": "#AEBBD6", "line": "#264653", "marker": "o"},
    "DP4SP8": {"bar": "#F4A261", "line": "#E76F51", "marker": "^"},
}
HATCHES = {
    "Model Exec.": "",
    "Schedule": "//////",
    "Data Transfer": "....",
}


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    unsupported = sorted(set(parsed) - set(PAPER_BATCH_SIZES))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported Fig. 17 batch sizes: {unsupported}"
        )
    return parsed


def comma_separated_strategies(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unsupported = [item for item in parsed if item not in PARALLELISMS]
    if not parsed or unsupported:
        raise argparse.ArgumentTypeError(
            f"strategies must be selected from {','.join(PARALLELISMS)}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    artifact_dir = Path(__file__).resolve().parent
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("FIG17_RAY_ADDRESS", DEFAULT_RAY_ADDRESS),
        help=(
            "Address of an already running four-node Ray cluster "
            f"(default: {DEFAULT_RAY_ADDRESS}); override with FIG17_RAY_ADDRESS."
        ),
    )
    parser.add_argument(
        "--master-address",
        help="Torch distributed address, HOST:PORT (default: Ray head HOST:27817).",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=artifact_dir / "results" / f"run_{run_id}_pid{os.getpid()}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Figure base path (default: RESULT_ROOT/fig17).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=comma_separated_ints,
        default=PAPER_BATCH_SIZES,
        help="Comma-separated subset; default: 32,64,96,128,160,192,256.",
    )
    parser.add_argument(
        "--strategies",
        type=comma_separated_strategies,
        default=tuple(PARALLELISMS),
        help="Comma-separated subset of DP32,DP4SP8.",
    )
    parser.add_argument("--loop-count", type=int, default=16)
    parser.add_argument("--segment-size", type=int, default=65536)
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=141.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=1_000_000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    args = parser.parse_args()

    if args.master_address is None:
        ray_host = args.ray_address.rsplit(":", 1)[0]
        args.master_address = f"{ray_host}:27817"
    if args.loop_count <= 0:
        parser.error("--loop-count must be positive")
    return args


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    benchmark = require_file(BENCHMARK_SCRIPT, "benchmark")
    model = args.model_path.expanduser().resolve()
    require_file(model / "config.json", "model config")
    return benchmark, model


def validate_ray_cluster(address: str) -> list[str]:
    import ray

    ray.init(address=address)
    try:
        active = [node for node in ray.nodes() if node["Alive"]]
    finally:
        ray.shutdown()

    gpu_nodes = [node for node in active if float(node["Resources"].get("GPU", 0)) > 0]
    total_gpus = int(sum(float(node["Resources"].get("GPU", 0)) for node in gpu_nodes))
    node_ips = [str(node["NodeManagerAddress"]) for node in gpu_nodes]
    if len(gpu_nodes) != PAPER_NODE_COUNT or total_gpus != PAPER_WORLD_SIZE:
        raise RuntimeError(
            "Fig. 17 requires exactly four active GPU nodes and 32 GPUs; "
            f"found {len(gpu_nodes)} nodes, {total_gpus} GPUs: {node_ips}"
        )
    print(f"Ray topology: {len(gpu_nodes)} nodes, {total_gpus} GPUs: {node_ips}")
    return node_ips


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def setting_name(
    parallelism: Parallelism,
    batch_size: int,
    args: argparse.Namespace,
) -> str:
    num_requests = batch_size * PAPER_WORLD_SIZE
    scheduling_policy = (
        "bucket_deepseek_v3" if parallelism.sp > 1 else "pure_dp"
    )
    return (
        f"DP{parallelism.dp}_SP{parallelism.sp}_EP32_TP1_"
        f"Seg{args.segment_size}_R#{num_requests}_"
        f"Burst_BS{batch_size}_"
        f"LeastBatch_centralized_{format_number(args.gpu_memory_limit_gb)}GB_"
        f"MEM{format_number(args.gpu_memory_utilization * 10)}_"
        f"LEN{args.max_model_len}_"
        f"loop{args.loop_count}_nonuniform_fixedsp0_{scheduling_policy}"
    )


def build_command(
    *,
    args: argparse.Namespace,
    benchmark: Path,
    model: Path,
    parallelism: Parallelism,
    batch_size: int,
    json_path: Path,
) -> list[str]:
    command = [
        args.python,
        "-u",
        str(benchmark),
        "--num-requests",
        str(batch_size * PAPER_WORLD_SIZE),
        "--sp",
        str(parallelism.sp),
        "--dp",
        str(parallelism.dp),
        "--ep",
        str(PAPER_WORLD_SIZE),
        "--tp",
        "1",
        "--max-num-seqs",
        str(batch_size),
        "--gpu-memory-limit-gb",
        format_number(args.gpu_memory_limit_gb),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--ray-address",
        args.ray_address,
        "--master-address",
        args.master_address,
        "--loop-count",
        str(args.loop_count),
        "--model-path",
        str(model),
        "--routing-strategy",
        "LeastBatch",
        "--itl-log-path",
        str(json_path),
        "--segment-size",
        str(args.segment_size),
    ]
    if parallelism.sp > 1:
        command.extend(
            [
                "--dynamic-sp-size-strategy",
                "bucket",
                "--dynamic-sp-bucket-preset",
                "deepseek_v3",
            ]
        )
    return command


def stream_command(
    command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]
) -> None:
    printable = shlex.join(command)
    print(f"\n+ {printable}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("================= Reproduce Command =================\n")
        log_file.write(printable + "\n")
        log_file.write("=====================================================\n\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def validate_case(log_path: Path, json_path: Path, expected_requests: int) -> None:
    warmup_done = False
    step_count = 0
    transfer_count = 0
    completed_requests: int | None = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            if "Warmup completed in" in line:
                warmup_done = True
                continue
            if warmup_done and STEP_RE.search(line):
                step_count += 1
            if warmup_done and TRANSFER_RE.search(line):
                transfer_count += 1
            if line.startswith("Requests completed:"):
                completed_requests = int(line.split(":", 1)[1].strip())

    json_rows = 0
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as handle:
            json_rows = sum(1 for line in handle if line.strip())
    failures = []
    if completed_requests != expected_requests:
        failures.append(
            f"completed_requests={completed_requests}, expected={expected_requests}"
        )
    if json_rows != expected_requests:
        failures.append(f"JSON rows={json_rows}, expected={expected_requests}")
    if step_count == 0:
        failures.append("no post-warmup ITL/schedule samples")
    if transfer_count == 0:
        failures.append("no post-warmup input-transfer samples")
    if failures:
        raise RuntimeError(f"Invalid benchmark output {log_path}: {'; '.join(failures)}")
    print(
        f"Validated {log_path.name}: requests={completed_requests}, "
        f"steps={step_count}, transfer_samples={transfer_count}"
    )


def run_case(
    *,
    args: argparse.Namespace,
    benchmark: Path,
    model: Path,
    result_root: Path,
    parallelism: Parallelism,
    batch_size: int,
) -> CaseResult:
    case_dir = result_root / "logs" / setting_name(parallelism, batch_size, args)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = case_dir / f"{timestamp}.log"
    json_path = case_dir / f"{timestamp}.json"
    command = build_command(
        args=args,
        benchmark=benchmark,
        model=model,
        parallelism=parallelism,
        batch_size=batch_size,
        json_path=json_path,
    )
    print(f"[{parallelism.name}/BS{batch_size}] {shlex.join(command)}")
    if not args.dry_run:
        case_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        for proxy_name in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ):
            env.pop(proxy_name, None)
        for name, value in NANODEPLOY_ENVIRONMENT_DEFAULTS.items():
            env.setdefault(name, value)
        env.update(
            {
                "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "1",
                "PYTHONUNBUFFERED": "1",
                "RAY_DEDUP_LOGS": "0",
                "TORCHDYNAMO_DISABLE": "1",
            }
        )
        stream_command(command, cwd=AE_ROOT, log_path=log_path, env=env)
        validate_case(log_path, json_path, batch_size * PAPER_WORLD_SIZE)
    return CaseResult(
        strategy=parallelism.name,
        batch_size=batch_size,
        log_path=str(log_path),
        json_path=str(json_path),
        command=command,
    )


def parse_log_identity(log_path: Path) -> LogIdentity | None:
    setting = log_path.parent.name
    dp_match = DP_RE.search(setting)
    sp_match = SP_RE.search(setting)
    batch_match = BS_RE.search(setting)
    if dp_match is None or sp_match is None or batch_match is None:
        return None

    strategy = STRATEGY_CONFIGS.get(
        (int(dp_match.group(1)), int(sp_match.group(1)))
    )
    if strategy is None:
        return None
    batch_size = int(batch_match.group(1))
    if batch_size not in PAPER_BATCH_SIZES:
        return None
    loop_match = LOOP_RE.search(setting)
    loop_count = int(loop_match.group(1)) if loop_match is not None else 16
    if loop_count <= 0:
        raise ValueError(f"invalid loop count in {setting}")
    return LogIdentity(strategy, batch_size, loop_count)


def numeric_ms(value: object) -> float | None:
    if not isinstance(value, str) or not value.endswith("ms"):
        return None
    try:
        return float(value[:-2])
    except ValueError:
        return None


def parse_measurement(log_path: Path, identity: LogIdentity) -> Measurement:
    itl_samples: list[float] = []
    schedule_samples: list[float] = []
    post_schedule_samples: list[float] = []
    transfer_samples: list[float] = []
    warmup_completed = False
    in_tpot_summary = False
    summary_itl_p50_ms: float | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            if "Warmup completed in" in line:
                warmup_completed = True
                continue
            if not warmup_completed:
                continue

            if "--- TPOT without Queueing Time (ms/token) ---" in line:
                in_tpot_summary = True
                continue
            if in_tpot_summary:
                summary_match = SUMMARY_P50_RE.search(line)
                if summary_match is not None:
                    summary_itl_p50_ms = float(summary_match.group(1))
                    in_tpot_summary = False
                    continue
                if line.strip().startswith("---"):
                    in_tpot_summary = False

            if "step -" in line and "{'mode':" in line:
                start = line.find("{'mode':")
                end = line.rfind("}")
                if start >= 0 and end > start:
                    try:
                        step_data = ast.literal_eval(line[start : end + 1])
                    except (SyntaxError, ValueError):
                        step_data = {}
                    for key, destination in (
                        ("itl", itl_samples),
                        ("sch_ovhd", schedule_samples),
                        ("post_sch_ovhd", post_schedule_samples),
                    ):
                        parsed = numeric_ms(step_data.get(key))
                        if parsed is not None:
                            destination.append(parsed)

            if "Input Transfer Latency:" in line:
                transfer_match = RANK_TRANSFER_RE.search(line)
                if transfer_match is None:
                    transfer_match = GLOBAL_TRANSFER_RE.search(line)
                if transfer_match is not None:
                    transfer_samples.append(float(transfer_match.group(1)))

    missing = [
        name
        for name, values in (
            ("schedule overhead", schedule_samples),
            ("post-schedule overhead", post_schedule_samples),
            ("input-transfer latency", transfer_samples),
        )
        if not values
    ]
    if summary_itl_p50_ms is None and not itl_samples:
        missing.insert(0, "ITL")
    if missing:
        raise ValueError(
            f"{log_path}: no post-warmup samples for {', '.join(missing)}"
        )

    itl_p50_ms = (
        summary_itl_p50_ms
        if summary_itl_p50_ms is not None
        else float(statistics.median(itl_samples))
    )
    return Measurement(
        strategy=identity.strategy,
        batch_size=identity.batch_size,
        loop_count=identity.loop_count,
        log_path=log_path,
        itl_p50_ms=itl_p50_ms,
        schedule_p50_ms=float(statistics.median(schedule_samples)),
        post_schedule_p50_ms=float(
            statistics.median(post_schedule_samples)
        ),
        transfer_p50_ms=float(statistics.median(transfer_samples)),
    )


def load_measurements(
    log_root: Path,
    expected: set[tuple[str, int]],
) -> list[Measurement]:
    selected: dict[tuple[str, int], tuple[Path, LogIdentity]] = {}
    for log_path in sorted(log_root.rglob("*.log")):
        identity = parse_log_identity(log_path)
        if identity is None:
            continue
        key = (identity.strategy, identity.batch_size)
        previous = selected.get(key)
        if previous is None or log_path.stem < previous[0].stem:
            selected[key] = (log_path, identity)

    missing = sorted(expected - set(selected), key=lambda item: (item[1], item[0]))
    unexpected = sorted(
        set(selected) - expected,
        key=lambda item: (item[1], item[0]),
    )
    if missing or unexpected:
        raise ValueError(
            "invalid Fig. 17 log matrix: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [
        parse_measurement(*selected[key])
        for key in sorted(expected, key=lambda item: (item[1], item[0]))
    ]


def print_breakdown(measurements: list[Measurement]) -> None:
    print("\n=== Fig. 17 component breakdown (post-warmup P50) ===")
    print(
        f"{'Batch':<6} {'Strategy':<9} {'ITL(ms)':<9} {'Model':<9} "
        f"{'Schedule':<9} {'Transfer':<9} {'Ovhd%':<7} Log"
    )
    for measurement in sorted(
        measurements,
        key=lambda item: (item.batch_size, STRATEGY_ORDER.index(item.strategy)),
    ):
        print(
            f"{measurement.batch_size:<6} {measurement.strategy:<9} "
            f"{measurement.itl_p50_ms:<9.2f} "
            f"{measurement.model_exec_ms:<9.2f} "
            f"{measurement.schedule_per_step_ms:<9.2f} "
            f"{measurement.transfer_per_step_ms:<9.2f} "
            f"{measurement.overhead_pct:<7.2f} {measurement.log_path}"
        )


def plot_measurements(measurements: list[Measurement], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig17")
    if str(AE_ROOT) not in sys.path:
        sys.path.insert(0, str(AE_ROOT))

    import matplotlib

    matplotlib.use("Agg")

    from ae_utils.plotting import get_plot_font_family
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": get_plot_font_family(),
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "hatch.linewidth": 0.5,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    by_key = {
        (measurement.strategy, measurement.batch_size): measurement
        for measurement in measurements
    }
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if any(key[0] == strategy for key in by_key)
    ]
    batch_sizes = [
        batch_size
        for batch_size in PAPER_BATCH_SIZES
        if any(key[1] == batch_size for key in by_key)
    ]

    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.5, zorder=0)
    bar_width = 0.16
    group_gap = 0.5
    cursor = 0.0
    x_centers: list[float] = []
    x_tick_labels: list[str] = []
    line_x = {strategy: [] for strategy in strategies}
    line_y = {strategy: [] for strategy in strategies}

    for batch_size in batch_sizes:
        for strategy_index, strategy in enumerate(strategies):
            measurement = by_key.get((strategy, batch_size))
            if measurement is None:
                continue
            position = cursor + strategy_index * bar_width
            line_x[strategy].append(position)
            line_y[strategy].append(measurement.overhead_pct)
            bottom = 0.0
            for component, value in (
                ("Model Exec.", measurement.model_exec_ms),
                ("Schedule", measurement.schedule_per_step_ms),
                ("Data Transfer", measurement.transfer_per_step_ms),
            ):
                ax.bar(
                    position,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=COLORS[strategy]["bar"],
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=HATCHES[component] or None,
                    zorder=2,
                    alpha=0.95,
                )
                bottom += value
        x_centers.append(cursor + (len(strategies) - 1) * bar_width / 2.0)
        x_tick_labels.append(str(batch_size))
        cursor += len(strategies) * bar_width + group_gap

    ax2 = ax.twinx()
    ax2.set_ylabel("Overhead/ITL (%)", fontweight="bold")
    for strategy in strategies:
        style = COLORS[strategy]
        ax2.plot(
            line_x[strategy],
            line_y[strategy],
            color=style["line"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.8,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=10,
        )
    max_overhead = max(
        (max(values) for values in line_y.values() if values),
        default=0.0,
    )
    ax2.set_ylim(0, max(4, math.ceil(max_overhead * 1.15)))
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Batch Size", fontweight="bold", labelpad=1)
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_xlim(-0.3, cursor - group_gap + 0.3)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    strategy_handles = [
        (
            Patch(
                facecolor=COLORS[strategy]["bar"],
                edgecolor="black",
                linewidth=0.5,
            ),
            Line2D(
                [0],
                [0],
                color=COLORS[strategy]["line"],
                marker=COLORS[strategy]["marker"],
                linewidth=1.8,
                markersize=6,
                markeredgecolor="white",
            ),
        )
        for strategy in strategies
    ]
    strategy_legend = ax.legend(
        handles=strategy_handles,
        labels=strategies,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    )
    ax.add_artist(strategy_legend)
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="Model Exec."),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=HATCHES["Schedule"],
                label="Schedule",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=HATCHES["Data Transfer"],
                label="Data Transfer",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
        handletextpad=0.6,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.20, top=0.78)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output_path = output.with_suffix(f".{extension}")
        fig.savefig(
            output_path,
            format=extension,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"Saved {extension.upper()}: {output_path}")
    plt.close(fig)


def run_postprocessing(
    *,
    args: argparse.Namespace,
    result_root: Path,
    output: Path,
) -> tuple[str, str]:
    expected = {
        (strategy, batch_size)
        for strategy in args.strategies
        for batch_size in args.batch_sizes
    }
    measurements = load_measurements(result_root / "logs", expected)
    print_breakdown(measurements)
    plot_measurements(measurements, output)
    return str(output.with_suffix(".pdf")), str(output.with_suffix(".png"))


def main() -> int:
    args = parse_args()
    benchmark, model = validate_inputs(args)
    result_root = args.result_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_root / "fig17"
    )
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty result root: {result_root}")

    if not args.dry_run:
        result_root.mkdir(parents=True)
        node_ips = validate_ray_cluster(args.ray_address)
    else:
        node_ips = []

    results = []
    for strategy in args.strategies:
        parallelism = PARALLELISMS[strategy]
        for batch_size in args.batch_sizes:
            results.append(
                run_case(
                    args=args,
                    benchmark=benchmark,
                    model=model,
                    result_root=result_root,
                    parallelism=parallelism,
                    batch_size=batch_size,
                )
            )

    if args.dry_run:
        print(f"\nDry run complete: {len(results)} case(s).")
        return 0

    pdf_path, png_path = run_postprocessing(
        args=args,
        result_root=result_root,
        output=output,
    )
    manifest = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "ray_address": args.ray_address,
        "master_address": args.master_address,
        "node_ips": node_ips,
        "model_path": str(model),
        "benchmark_script": str(benchmark),
        "scheduling": {
            "DP32": "pure DP (SP=1)",
            "DP4SP8": "bucket dynamic SP (deepseek_v3 preset)",
        },
        "measured_request_submission": (
            "all requests are added before the first measured engine step"
        ),
        "cases": [asdict(result) for result in results],
        "figure_pdf": pdf_path,
        "figure_png": png_path,
    }
    manifest_path = result_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nCompleted {len(results)} Fig. 17 cases.")
    print(f"Raw logs: {result_root / 'logs'}")
    print(f"Figure: {pdf_path}, {png_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

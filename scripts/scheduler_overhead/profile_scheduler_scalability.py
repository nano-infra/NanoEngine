#!/usr/bin/env python3
"""Profile the production C++ scheduler with larger logical topologies.

This benchmark does not launch Ray, workers, or GPU kernels.  It constructs the
same ``nanodeploy._cpp.Scheduler`` used by the legacy global control plane and
measures steady-state decode scheduling after requests have been admitted.

The four built-in scenarios preserve the same total GPU and request counts:

* ``no_sp``: DP equals the logical GPU count and SP is 1.
* ``fixed_sp8``: DP equals the logical node count and every request uses SP8.
* ``dynamic_sp8_1pct``: one percent of requests use SP8; the rest use SP1.
* ``dynamic_sp8_5pct``: five percent of requests use SP8; the rest use SP1.

One logical node contains eight logical GPUs.  No physical cluster resources
are emulated: increasing ``--logical-nodes`` expands only the scheduler's
in-memory topology and state.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import socket
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence as TypingSequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanodeploy._cpp import Scheduler, Sequence, update_seqs_inner_loop  # noqa: E402
from nanodeploy.config import DEEPSEEK_V3_BUCKET_POLICY  # noqa: E402


GPUS_PER_LOGICAL_NODE = 8
SP8_DEGREE = 8
DEFAULT_SCENARIOS = (
    "no_sp",
    "fixed_sp8",
    "dynamic_sp8_1pct",
    "dynamic_sp8_5pct",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    use_sp8_topology: bool
    fixed_sp_size: int
    sp8_ratio: float

    @property
    def uses_dynamic_policy(self) -> bool:
        return self.fixed_sp_size == 0 and self.sp8_ratio > 0.0


SCENARIOS = {
    "no_sp": Scenario(
        name="no_sp",
        use_sp8_topology=False,
        fixed_sp_size=1,
        sp8_ratio=0.0,
    ),
    "fixed_sp8": Scenario(
        name="fixed_sp8",
        use_sp8_topology=True,
        fixed_sp_size=SP8_DEGREE,
        sp8_ratio=1.0,
    ),
    "dynamic_sp8_1pct": Scenario(
        name="dynamic_sp8_1pct",
        use_sp8_topology=True,
        fixed_sp_size=0,
        sp8_ratio=0.01,
    ),
    "dynamic_sp8_5pct": Scenario(
        name="dynamic_sp8_5pct",
        use_sp8_topology=True,
        fixed_sp_size=0,
        sp8_ratio=0.05,
    ),
}


@dataclass(frozen=True)
class ProfileCase:
    scenario: Scenario
    logical_nodes: int
    batch_size_per_gpu: int
    short_context_len: int
    long_context_len: int
    block_size: int
    loop_count: int
    seed: int
    # Mixed admission/decode profilers may keep BS/GPU as the public workload
    # label while adding a small admission cohort to the decode cohort.  The
    # default preserves the original bulk-only case exactly.
    request_count_override: int | None = None

    @property
    def logical_gpus(self) -> int:
        return self.logical_nodes * GPUS_PER_LOGICAL_NODE

    @property
    def attention_dp(self) -> int:
        if self.scenario.use_sp8_topology:
            return self.logical_nodes
        return self.logical_gpus

    @property
    def attention_sp(self) -> int:
        return SP8_DEGREE if self.scenario.use_sp8_topology else 1

    @property
    def total_requests(self) -> int:
        if self.request_count_override is not None:
            if self.request_count_override <= 0:
                raise ValueError("request_count_override must be positive")
            return self.request_count_override
        return self.logical_gpus * self.batch_size_per_gpu

    @property
    def requests_per_dp(self) -> int:
        return math.ceil(self.total_requests / self.attention_dp)

    @property
    def expected_sp8_requests(self) -> int:
        return round(self.total_requests * self.scenario.sp8_ratio)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return parsed


def _parse_scenarios(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(names) - set(SCENARIOS))
    if not names:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown scenarios {unknown}; choose from {sorted(SCENARIOS)}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("scenarios must not contain duplicates")
    return names


def _bucket_sp_degree(policy: str, context_len: int) -> int | None:
    for item in policy.split(";"):
        degree_text, interval_text = item.split(":", maxsplit=1)
        low_text, high_text = interval_text.split("-", maxsplit=1)
        if int(low_text) <= context_len <= int(high_text):
            return int(degree_text)
    return None


def _percentile(samples: TypingSequence[float], percentile: float) -> float:
    if not samples:
        raise ValueError("cannot calculate a percentile of no samples")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _long_request_indices(
    total_requests: int,
    long_requests: int,
    seed: int,
) -> set[int]:
    if not 0 <= long_requests <= total_requests:
        raise ValueError("long request count must be within the total request count")
    if long_requests == 0:
        return set()
    return set(random.Random(seed).sample(range(total_requests), long_requests))


def _profile_sequence(context_len: int, request_index: int, loop_count: int) -> Sequence:
    # Full token storage is intentional. BlockManager::allocate() hashes block
    # contents through Sequence::block_view(), so changing only num_tokens would
    # create an invalid out-of-bounds view. A per-request token value also avoids
    # accidentally deduplicating every request through the prefix-block cache.
    token_value = request_index + 1
    return Sequence(
        [token_value] * context_len,
        1.0,
        context_len + loop_count + 1,
        True,
    )


def _scheduler_capacity_blocks(
    case: ProfileCase,
    profiled_decode_iterations: int,
) -> int:
    requests_per_dp = case.requests_per_dp
    master_requests_per_rank = math.ceil(
        requests_per_dp / case.attention_sp
    )
    short_blocks = math.ceil(case.short_context_len / case.block_size)
    if not case.scenario.use_sp8_topology:
        initial_blocks_per_rank = requests_per_dp * short_blocks
    elif case.scenario.fixed_sp_size == SP8_DEGREE:
        per_request_rank_tokens = math.ceil(
            case.short_context_len / SP8_DEGREE
        )
        per_request_rank_blocks = math.ceil(
            per_request_rank_tokens / case.block_size
        )
        initial_blocks_per_rank = (
            requests_per_dp
            * SP8_DEGREE
            * per_request_rank_blocks
        )
    else:
        # Round-robin DP routing maps request i to i % attention_dp. Derive the
        # largest long-request count assigned to any SP8 group, then reserve a
        # per-rank share of each long request. Non-uniform placement balances
        # free capacity, while the 10% margin below covers small rank skew.
        long_indices = _long_request_indices(
            case.total_requests,
            case.expected_sp8_requests,
            case.seed,
        )
        long_per_dp = [0] * case.attention_dp
        for request_index in long_indices:
            long_per_dp[request_index % case.attention_dp] += 1
        max_long_per_dp = max(long_per_dp, default=0)
        long_rank_tokens = math.ceil(case.long_context_len / SP8_DEGREE)
        long_rank_blocks = math.ceil(long_rank_tokens / case.block_size)
        initial_blocks_per_rank = (
            master_requests_per_rank * short_blocks
            + max_long_per_dp * long_rank_blocks
        )
    generated_master_blocks = (
        master_requests_per_rank
        * math.ceil(
            profiled_decode_iterations * case.loop_count / case.block_size
        )
    )
    return (
        math.ceil(initial_blocks_per_rank * 1.1)
        + case.batch_size_per_gpu
        + generated_master_blocks
        + 64
    )


def _new_scheduler(
    case: ProfileCase,
    profiled_decode_iterations: int = 0,
) -> Scheduler:
    dynamic = case.scenario.uses_dynamic_policy
    bucket_policy = DEEPSEEK_V3_BUCKET_POLICY if dynamic else ""
    return Scheduler(
        f"scheduler-profile-{case.scenario.name}",
        case.loop_count,
        math.ceil(case.requests_per_dp / case.attention_sp),
        case.requests_per_dp * case.long_context_len + 1,
        case.requests_per_dp * SP8_DEGREE + SP8_DEGREE,
        -1,
        case.attention_dp,
        case.attention_sp,
        _scheduler_capacity_blocks(case, profiled_decode_iterations),
        case.block_size,
        "decode",
        1.0,
        65_536,
        "bucket" if dynamic else "legacy",
        dynamic,
        bucket_policy,
        True,
        "LeastBatch",
        case.scenario.fixed_sp_size,
    )


@contextmanager
def _native_stderr(suppress: bool) -> Iterator[None]:
    """Optionally silence verbose native scheduler construction messages."""

    if not suppress:
        yield
        return
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def _sp_histogram(schedule_result: object) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for per_dp in schedule_result.sp_size_hist_per_dp:
        for degree, count in enumerate(per_dp):
            if count:
                histogram[degree] = histogram.get(degree, 0) + int(count)
    return histogram


def _validate_decode_result(case: ProfileCase, schedule_result: object) -> dict[int, int]:
    if schedule_result.is_prefill:
        raise RuntimeError("expected steady-state decode, got an admission step")
    histogram = _sp_histogram(schedule_result)
    real_requests = sum(histogram.values())
    if real_requests != case.total_requests:
        raise RuntimeError(
            f"scheduled {real_requests} real requests, expected {case.total_requests}"
        )
    actual_sp8 = histogram.get(SP8_DEGREE, 0)
    if actual_sp8 != case.expected_sp8_requests:
        raise RuntimeError(
            f"scheduled {actual_sp8} SP8 requests, expected "
            f"{case.expected_sp8_requests} for {case.scenario.name}"
        )
    actual_sp1 = histogram.get(1, 0)
    if actual_sp1 != case.total_requests - case.expected_sp8_requests:
        raise RuntimeError(
            f"scheduled {actual_sp1} SP1 requests, expected "
            f"{case.total_requests - case.expected_sp8_requests}"
        )
    unexpected = set(histogram) - {1, SP8_DEGREE}
    if unexpected:
        raise RuntimeError(f"unexpected SP degrees in schedule: {sorted(unexpected)}")
    return histogram


def _populate_waiting_queue(
    case: ProfileCase,
    *,
    profiled_decode_iterations: int,
    suppress_native_setup_logs: bool,
) -> tuple[Scheduler, float]:
    setup_begin = time.perf_counter_ns()
    with _native_stderr(suppress_native_setup_logs):
        scheduler = _new_scheduler(case, profiled_decode_iterations)
    long_indices = _long_request_indices(
        case.total_requests,
        case.expected_sp8_requests if case.scenario.uses_dynamic_policy else 0,
        case.seed,
    )
    for request_index in range(case.total_requests):
        context_len = (
            case.long_context_len
            if request_index in long_indices
            else case.short_context_len
        )
        scheduler.add(
            _profile_sequence(context_len, request_index, case.loop_count)
        )
    setup_ms = (time.perf_counter_ns() - setup_begin) / 1_000_000.0
    return scheduler, setup_ms


def _validate_admission_result(case: ProfileCase, schedule_result: object) -> None:
    if not schedule_result.is_prefill:
        raise RuntimeError("initial scheduler call did not admit the profiling workload")
    admitted = sum(len(per_dp) for per_dp in schedule_result.dp_seqs)
    if admitted != case.total_requests:
        raise RuntimeError(
            f"admitted {admitted} requests, expected {case.total_requests}; "
            "increase profiling scheduler capacity"
        )


def _advance_decode_state(schedule_result: object, loop_count: int) -> None:
    """Advance scheduler-owned sequence lengths outside the timed interval.

    Production workers advance their deserialized Sequence copies during each
    inner decode step, and Scheduler.postprocess() applies the generated tokens
    to the authoritative scheduler copies. For this CPU-only benchmark, the
    existing C++ helper performs the equivalent count update without building
    fake model outputs or including postprocessing in the scheduler timing.
    """

    attention_sp = len(schedule_result.sp_size_hist_per_dp[0]) - 1
    for flat_index, sequences in enumerate(schedule_result.filtered_dp_sp_seqs):
        sp_rank = flat_index % attention_sp
        for _ in range(loop_count):
            update_seqs_inner_loop(sequences, sp_rank)


def run_case(
    case: ProfileCase,
    *,
    warmup_iterations: int,
    measured_iterations: int,
    admission_iterations: int = 1,
    suppress_native_setup_logs: bool = True,
) -> dict[str, object]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measured_iterations <= 0:
        raise ValueError("measured_iterations must be positive")
    if admission_iterations <= 0:
        raise ValueError("admission_iterations must be positive")

    profiled_decode_iterations = 1 + warmup_iterations + measured_iterations
    admission_samples_ms: list[float] = []
    setup_samples_ms: list[float] = []
    scheduler: Scheduler | None = None
    for admission_index in range(admission_iterations):
        candidate, setup_ms = _populate_waiting_queue(
            case,
            profiled_decode_iterations=profiled_decode_iterations,
            suppress_native_setup_logs=suppress_native_setup_logs,
        )
        setup_samples_ms.append(setup_ms)
        admission_begin = time.perf_counter_ns()
        admission_result = candidate.schedule()
        admission_samples_ms.append(
            (time.perf_counter_ns() - admission_begin) / 1_000_000.0
        )
        _validate_admission_result(case, admission_result)
        del admission_result
        if admission_index == admission_iterations - 1:
            scheduler = candidate
        else:
            del candidate
    if scheduler is None:
        raise RuntimeError("admission profiling did not retain a scheduler")

    validation_result = scheduler.schedule()
    histogram = _validate_decode_result(case, validation_result)
    _advance_decode_state(validation_result, case.loop_count)
    del validation_result

    for _ in range(warmup_iterations):
        warmup_result = scheduler.schedule()
        if warmup_result.is_prefill:
            raise RuntimeError("warmup unexpectedly returned an admission step")
        _advance_decode_state(warmup_result, case.loop_count)
        del warmup_result

    gc_was_enabled = gc.isenabled()
    gc.disable()
    samples_ms: list[float] = []
    try:
        for _ in range(measured_iterations):
            begin = time.perf_counter_ns()
            result = scheduler.schedule()
            elapsed_ms = (time.perf_counter_ns() - begin) / 1_000_000.0
            if result.is_prefill:
                raise RuntimeError("timed iteration unexpectedly returned admission")
            samples_ms.append(elapsed_ms)
            _advance_decode_state(result, case.loop_count)
            # Destroy the result outside the measured interval. This matches the
            # production timer around Scheduler.schedule(), which excludes later
            # LLMEngine processing and ScheduleResult destruction.
            del result
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_ms = statistics.fmean(samples_ms)
    stddev_ms = statistics.pstdev(samples_ms)
    admission_mean_ms = statistics.fmean(admission_samples_ms)
    return {
        "scenario": case.scenario.name,
        "logical_nodes": case.logical_nodes,
        "logical_gpus": case.logical_gpus,
        "attention_dp": case.attention_dp,
        "attention_sp": case.attention_sp,
        "batch_size_per_gpu": case.batch_size_per_gpu,
        "total_requests": case.total_requests,
        "expected_sp1_requests": case.total_requests - case.expected_sp8_requests,
        "expected_sp8_requests": case.expected_sp8_requests,
        "actual_sp1_requests": histogram.get(1, 0),
        "actual_sp8_requests": histogram.get(SP8_DEGREE, 0),
        "actual_sp8_ratio": histogram.get(SP8_DEGREE, 0) / case.total_requests,
        "short_context_len": case.short_context_len,
        "long_context_len": case.long_context_len,
        "dynamic_sp_bucket_policy": (
            DEEPSEEK_V3_BUCKET_POLICY
            if case.scenario.uses_dynamic_policy
            else ""
        ),
        "block_size": case.block_size,
        "loop_count": case.loop_count,
        "scheduler_thread_pool_workers": case.attention_dp,
        "setup_mean_ms_excluded": statistics.fmean(setup_samples_ms),
        "admission_iterations": admission_iterations,
        "admission_mean_ms": admission_mean_ms,
        "admission_stddev_ms": statistics.pstdev(admission_samples_ms),
        "admission_min_ms": min(admission_samples_ms),
        "admission_p50_ms": _percentile(admission_samples_ms, 50.0),
        "admission_p95_ms": _percentile(admission_samples_ms, 95.0),
        "admission_p99_ms": _percentile(admission_samples_ms, 99.0),
        "admission_max_ms": max(admission_samples_ms),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "min_ms": min(samples_ms),
        "p50_ms": _percentile(samples_ms, 50.0),
        "p95_ms": _percentile(samples_ms, 95.0),
        "p99_ms": _percentile(samples_ms, 99.0),
        "max_ms": max(samples_ms),
    }


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "bench_logs" / "scheduler_overhead" / timestamp


def _write_results(output_dir: Path, metadata: dict[str, object], records: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scheduler_overhead.json"
    csv_path = output_dir / "scheduler_overhead.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "records": records},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logical-nodes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("4,8,16,32"),
        help="Comma-separated logical 8-GPU node counts (default: 4,8,16,32).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("32,64,128"),
        help="Comma-separated active request counts per logical GPU.",
    )
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=DEFAULT_SCENARIOS,
        help=f"Comma-separated scenarios (default: {','.join(DEFAULT_SCENARIOS)}).",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--admission-iterations",
        type=int,
        default=10,
        help="Fresh bulk-admission repetitions per case (default: 10).",
    )
    parser.add_argument(
        "--short-context-len",
        type=int,
        default=1_024,
        help="SP1 request length (default: 1024).",
    )
    parser.add_argument(
        "--long-context-len",
        type=int,
        default=428_033,
        help="Long request length in the DeepSeek-V3 SP8 bucket (default: 428033).",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--loop-count",
        type=int,
        default=16,
        help="Decode steps reserved by each scheduling quantum (default: 16).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: bench_logs/scheduler_overhead/<UTC timestamp>).",
    )
    parser.add_argument(
        "--show-native-setup-logs",
        action="store_true",
        help="Show verbose C++ scheduler construction messages.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.admission_iterations <= 0:
        parser.error("--admission-iterations must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.loop_count <= 0:
        parser.error("--loop-count must be positive")
    if args.short_context_len < SP8_DEGREE:
        parser.error(f"--short-context-len must be at least {SP8_DEGREE}")
    if args.long_context_len <= args.short_context_len:
        parser.error("--long-context-len must exceed --short-context-len")
    short_degree = _bucket_sp_degree(
        DEEPSEEK_V3_BUCKET_POLICY,
        args.short_context_len,
    )
    if short_degree != 1:
        parser.error(
            "--short-context-len must fall in the DeepSeek-V3 SP1 bucket; "
            f"got degree {short_degree}"
        )
    long_degree = _bucket_sp_degree(
        DEEPSEEK_V3_BUCKET_POLICY,
        args.long_context_len,
    )
    if long_degree != SP8_DEGREE:
        parser.error(
            "--long-context-len must fall in the DeepSeek-V3 SP8 bucket; "
            f"got degree {long_degree}"
        )


def main(argv: TypingSequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    output_dir = args.output_dir or _default_output_dir()
    records: list[dict[str, object]] = []
    total_cases = len(args.logical_nodes) * len(args.batch_sizes) * len(args.scenarios)
    case_index = 0
    for logical_nodes in args.logical_nodes:
        for batch_size in args.batch_sizes:
            for scenario_name in args.scenarios:
                case_index += 1
                case = ProfileCase(
                    scenario=SCENARIOS[scenario_name],
                    logical_nodes=logical_nodes,
                    batch_size_per_gpu=batch_size,
                    short_context_len=args.short_context_len,
                    long_context_len=args.long_context_len,
                    block_size=args.block_size,
                    loop_count=args.loop_count,
                    seed=args.seed,
                )
                print(
                    f"[{case_index}/{total_cases}] scenario={scenario_name} "
                    f"nodes={logical_nodes} gpus={case.logical_gpus} "
                    f"bs_per_gpu={batch_size} requests={case.total_requests}",
                    flush=True,
                )
                record = run_case(
                    case,
                    warmup_iterations=args.warmup_iterations,
                    measured_iterations=args.iterations,
                    admission_iterations=args.admission_iterations,
                    suppress_native_setup_logs=not args.show_native_setup_logs,
                )
                records.append(record)
                print(
                    f"  admission mean={record['admission_mean_ms']:.3f} ms "
                    f"p99={record['admission_p99_ms']:.3f} ms | "
                    f"decode mean={record['mean_ms']:.3f} ms "
                    f"p99={record['p99_ms']:.3f} ms",
                    flush=True,
                )

    metadata = {
        "benchmark": "nanodeploy-cpu-scheduler-scalability",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "repo_root": str(REPO_ROOT),
        "gpus_per_logical_node": GPUS_PER_LOGICAL_NODE,
        "dynamic_sp_bucket_policy": DEEPSEEK_V3_BUCKET_POLICY,
        "timed_scope": (
            "bulk waiting-request admission Scheduler.schedule() and "
            "steady-state decode Scheduler.schedule()"
        ),
        "excluded_scope": (
            "scheduler construction, Sequence creation and queue insertion, "
            "ScheduleResult destruction, Ray, RDMA, and GPU execution"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scenarios": [asdict(SCENARIOS[name]) for name in args.scenarios],
    }
    _write_results(output_dir, metadata, records)
    print(f"Wrote {output_dir / 'scheduler_overhead.json'}")
    print(f"Wrote {output_dir / 'scheduler_overhead.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

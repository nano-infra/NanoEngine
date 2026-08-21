"""Poisson serving benchmark for one prefill and one decode engine.

The two engines run on separate eight-GPU Ray nodes.  Requests are admitted to
prefill according to an open-loop Poisson process, handed to decode after
prefill, and drained after the arrival window closes.  Decode always advances
one token per scheduler iteration (``loop_count=1``).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections.abc import Iterable, Iterator, Sequence as SequenceABC
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import chain

import numpy as np
import ray
from tqdm.auto import tqdm

from nanodeploy import LLM, SamplingParams
from nanodeploy._cpp import SequenceStatus
from nanodeploy.engine.sequence import Sequence

import bench_serving as serving
import pd_disagg_deepseek_v3 as pd_example
import pd_disagg_deepseek_v3_parallel as pd_parallel


DECODE_LOOP_COUNT = 1
DEFAULT_DURATION_S = 300.0
DEFAULT_REQUEST_RATE = 8.0
DEFAULT_WARMUP_REQUESTS = 8
WARMUP_PROMPT_LEN = 512
WARMUP_OUTPUT_LEN = 32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a two-node 1-prefill + 1-decode deployment with "
            "Poisson request arrivals. Decode loop_count is fixed at 1."
        )
    )

    serving_group = parser.add_argument_group("serving workload")
    serving_group.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Poisson arrival window in seconds; drain time is additional.",
    )
    serving_group.add_argument(
        "--request-rate",
        type=float,
        default=DEFAULT_REQUEST_RATE,
        help="Mean Poisson request rate in requests/second.",
    )
    serving_group.add_argument("--seed", type=int, default=serving.SEED)
    serving_group.add_argument(
        "--dataset",
        choices=("random", "csv"),
        default="random",
    )
    serving_group.add_argument(
        "--csv-path",
        help="CSV containing prompt_len and output_len columns.",
    )
    serving_group.add_argument(
        "--max-request-tokens",
        type=int,
        default=serving.DEFAULT_MAX_REQUEST_TOKENS,
        help=(
            "Drop CSV rows whose prompt_len + output_len exceeds this limit; "
            "0 disables the filter."
        ),
    )
    serving_group.add_argument(
        "--itl-log-path",
        default="pd_itl_samples.jsonl",
        help="JSONL output for per-request ITL samples; empty disables it.",
    )
    serving_group.add_argument(
        "--warmup-requests",
        type=int,
        default=DEFAULT_WARMUP_REQUESTS,
        help="P/D requests to run before measurement; 0 disables warmup.",
    )
    serving_group.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable progress bars for cleaner redirected logs.",
    )

    engine_group = parser.add_argument_group("P/D engines")
    engine_group.add_argument(
        "--model-path",
        default=pd_example.DEFAULT_MODEL_PATH,
    )
    engine_group.add_argument(
        "--ray-address",
        default="10.102.252.174:6380",
    )
    engine_group.add_argument(
        "--prefill-master-address",
        default="10.102.252.174:6006",
    )
    engine_group.add_argument(
        "--decode-master-address",
        default="10.102.206.14:6006",
    )
    engine_group.add_argument(
        "--decode-topology",
        choices=("dp8", "sp8", "bucket-sp8"),
        default="sp8",
    )
    engine_group.add_argument(
        "--dynamic-sp-bucket-policy",
        default="",
        help="Required for --decode-topology=bucket-sp8.",
    )
    engine_group.add_argument(
        "--non-uniform-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    engine_group.add_argument("--dummy-weight", action="store_true")
    engine_group.add_argument(
        "--cuda-graph-mode",
        choices=("full", "piecewise"),
        default="full",
    )
    engine_group.add_argument("--decode-eager", action="store_true")
    engine_group.add_argument("--max-model-len", type=int, default=4096)
    engine_group.add_argument("--max-num-seqs", type=int, default=8)
    engine_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    engine_group.add_argument(
        "--sp-backend",
        choices=("hao_basic", "nccl"),
        default="hao_basic",
    )
    engine_group.add_argument(
        "--optimize-decode-block-table",
        action="store_true",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be positive")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    if args.max_request_tokens < 0:
        raise ValueError("--max-request-tokens must be non-negative")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.dataset == "csv":
        if not args.csv_path:
            raise ValueError("--csv-path is required with --dataset=csv")
        if not os.path.isfile(args.csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    pd_example.decode_dynamic_sp_kwargs(args)


def parse_args() -> argparse.Namespace:
    args = build_arg_parser().parse_args()
    validate_args(args)
    # The existing P/D builders consume this compatibility attribute.  It is
    # deliberately not a CLI option, so serving cannot accidentally use 16.
    args.decode_loop_count = DECODE_LOOP_COUNT
    return args


def generate_poisson_arrival_times(
    duration_s: float,
    request_rate: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate all arrivals inside a fixed-duration Poisson window."""
    arrival_times: list[float] = []
    arrival_time = 0.0
    while True:
        arrival_time += float(rng.exponential(1.0 / request_rate))
        if arrival_time > duration_s:
            return arrival_times
        arrival_times.append(arrival_time)


def _attach_end_to_end_metric(decode: LLM, sequence: Sequence) -> None:
    """Add a migrated sequence to decode without resetting its P/D metric."""
    metric = sequence.metric
    if metric is None:
        raise RuntimeError(
            f"Prefill request {sequence.seq_id} has no sequence metric"
        )

    decode.add_request(sequence)
    decode.metrics_manager.sequence_metrics[sequence.seq_id] = metric
    sequence.metric = metric
    metric.record_decode_arrival()


def _warmup_requests(count: int) -> list[tuple[list[int], SamplingParams]]:
    sampling_params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=WARMUP_OUTPUT_LEN,
    )
    return [
        (
            np.random.randint(0, 10_000, size=WARMUP_PROMPT_LEN).tolist(),
            sampling_params,
        )
        for _ in range(count)
    ]


def _completed_output_ids(step_result: tuple) -> tuple[int, ...]:
    outputs = step_result[0]
    return tuple(seq_id for seq_id, _ in outputs)


def run_pd_benchmark(
    prefill: LLM,
    decode: LLM,
    requests: Iterable[tuple[list[int], SamplingParams]],
    arrival_times: SequenceABC[float],
    *,
    description: str = "P/D requests",
    show_progress: bool = True,
) -> tuple[float, dict[int, Sequence]]:
    """Drive prefill and decode concurrently until every request completes."""
    request_iterator: Iterator[tuple[list[int], SamplingParams]] = iter(requests)
    num_requests = len(arrival_times)
    sequences: dict[int, Sequence] = {}
    prefill_pending: dict[int, Sequence] = {}
    decode_ready: deque[Sequence] = deque()
    source_kv_held: dict[int, Sequence] = {}
    source_release_ready: dict[int, Sequence] = {}
    completed_ids: set[int] = set()
    requests_sent = 0
    benchmark_start = time.perf_counter()

    prefill_future: Future | None = None
    decode_future: Future | None = None

    def record_completion(seq_id: int, progress: tqdm) -> None:
        if seq_id in completed_ids:
            return
        completed_ids.add(seq_id)
        progress.update(1)
        sequence = sequences[seq_id]
        if sequence.metric and sequence.metric.e2e_latency:
            progress.set_postfix(
                {"E2E": f"{sequence.metric.e2e_latency / 1000:.2f}s"}
            )

    with (
        ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="pd-serving-step",
        ) as step_pool,
        tqdm(
            total=num_requests,
            desc=description,
            disable=not show_progress,
        ) as progress,
    ):
        while len(completed_ids) < num_requests:
            made_progress = False

            if prefill_future is not None and prefill_future.done():
                prefill_result = prefill_future.result()
                prefill_future = None
                made_progress = True
                prefill_output_ids = set(_completed_output_ids(prefill_result))

                for seq_id, sequence in tuple(prefill_pending.items()):
                    if sequence.status == SequenceStatus.TO_BE_MIGRATED:
                        decode_ready.append(sequence)
                        source_kv_held[seq_id] = sequence
                        del prefill_pending[seq_id]
                    elif sequence.is_finished:
                        if seq_id not in prefill_output_ids:
                            raise RuntimeError(
                                "Prefill finished a request without returning it: "
                                f"seq_id={seq_id}"
                            )
                        del prefill_pending[seq_id]
                        record_completion(seq_id, progress)

            if decode_future is not None and decode_future.done():
                decode_result = decode_future.result()
                decode_future = None
                made_progress = True
                for seq_id in _completed_output_ids(decode_result):
                    record_completion(seq_id, progress)

                for seq_id, sequence in tuple(source_kv_held.items()):
                    if sequence.status != SequenceStatus.TO_BE_MIGRATED:
                        source_release_ready[seq_id] = sequence
                        del source_kv_held[seq_id]

            if prefill_future is None:
                if source_release_ready:
                    prefill.free_to_be_migrated(
                        list(source_release_ready.values())
                    )
                    source_release_ready.clear()
                    made_progress = True

                elapsed = time.perf_counter() - benchmark_start
                while (
                    requests_sent < num_requests
                    and elapsed >= arrival_times[requests_sent]
                ):
                    try:
                        prompt, sampling_params = next(request_iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Workload ended before all scheduled arrivals: "
                            f"expected={num_requests}, got={requests_sent}"
                        ) from exc
                    sequence = Sequence(
                        token_ids=prompt,
                        sampling_params=sampling_params,
                    )
                    prefill.add_request(sequence)
                    sequences[sequence.seq_id] = sequence
                    prefill_pending[sequence.seq_id] = sequence
                    requests_sent += 1
                    made_progress = True

                if not prefill.is_finished():
                    prefill_future = step_pool.submit(prefill.step)
                    made_progress = True

            if decode_future is None:
                while decode_ready:
                    _attach_end_to_end_metric(decode, decode_ready.popleft())
                    made_progress = True
                if not decode.is_finished():
                    decode_future = step_pool.submit(decode.step)
                    made_progress = True

            if len(completed_ids) == num_requests:
                break

            if not made_progress:
                active_futures = {
                    future
                    for future in (prefill_future, decode_future)
                    if future is not None
                }
                if active_futures:
                    wait(
                        active_futures,
                        timeout=0.01,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    if requests_sent >= num_requests:
                        raise RuntimeError(
                            "P/D benchmark stalled after admitting all "
                            f"{num_requests} requests"
                        )
                    next_arrival = arrival_times[requests_sent]
                    sleep_s = max(
                        0.0,
                        next_arrival
                        - (time.perf_counter() - benchmark_start),
                    )
                    time.sleep(min(sleep_s, 0.01))

    if requests_sent != num_requests:
        raise RuntimeError(
            f"Only sent {requests_sent} of {num_requests} scheduled requests"
        )
    if prefill_pending or decode_ready or source_kv_held or source_release_ready:
        raise RuntimeError("P/D benchmark completed with undrained request state")
    return time.perf_counter() - benchmark_start, sequences


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    arrival_rng = np.random.default_rng(args.seed)
    arrival_times = generate_poisson_arrival_times(
        args.duration,
        args.request_rate,
        arrival_rng,
    )
    if not arrival_times:
        raise RuntimeError(
            "The sampled Poisson window contains no requests; increase "
            "--duration or --request-rate"
        )

    # Advance once before timing so the shared loader reads and filters CSV
    # metadata up front. Keep token IDs lazy: materializing minutes of long
    # prompts can consume hundreds of GB on the driver.
    args.num_requests = len(arrival_times)
    request_generator = serving.get_dataset_generator(args)
    try:
        first_request = next(request_generator)
    except StopIteration as exc:
        raise RuntimeError("The serving dataset produced no requests") from exc
    requests = chain((first_request,), request_generator)

    print(
        "P/D serving workload:",
        f"arrival_window={args.duration:.2f}s",
        f"target_rate={args.request_rate:.3f}req/s",
        f"sampled_requests={len(arrival_times)}",
        f"sampled_rate={len(arrival_times) / args.duration:.3f}req/s",
        f"dataset={args.dataset}",
        f"decode_loop_count={DECODE_LOOP_COUNT}",
        flush=True,
    )

    rdma_env = pd_example.configure_driver_environment()
    print(f"RDMA environment: {rdma_env}", flush=True)
    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": rdma_env},
    )
    pd_example.validate_ray_cluster(
        args.prefill_master_address,
        args.decode_master_address,
    )

    decode: LLM | None = None
    prefill: LLM | None = None
    try:
        decode, prefill = pd_parallel.build_engines_parallel(args)
        if decode.config.loop_count != DECODE_LOOP_COUNT:
            raise RuntimeError(
                "Decode engine violated serving invariant: "
                f"loop_count={decode.config.loop_count}"
            )
        pd_example.connect_kv_transfer(prefill, decode)

        if args.warmup_requests:
            warmup_start = time.perf_counter()
            run_pd_benchmark(
                prefill,
                decode,
                _warmup_requests(args.warmup_requests),
                [0.0] * args.warmup_requests,
                description="P/D warmup",
                show_progress=not args.no_tqdm,
            )
            print(
                f"P/D warmup completed in "
                f"{time.perf_counter() - warmup_start:.2f}s",
                flush=True,
            )

        total_time, sequence_map = run_pd_benchmark(
            prefill,
            decode,
            requests,
            arrival_times,
            show_progress=not args.no_tqdm,
        )
        drain_time = max(0.0, total_time - args.duration)
        print(
            "P/D request injection and drain completed:",
            f"total_time={total_time:.2f}s",
            f"drain_time={drain_time:.2f}s",
            flush=True,
        )
        serving.calculate_and_print_metrics(
            total_time,
            sequence_map,
            len(arrival_times),
            itl_log_path=args.itl_log_path or None,
        )
    finally:
        pd_example.close_engine(prefill, "prefill")
        pd_example.close_engine(decode, "decode")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

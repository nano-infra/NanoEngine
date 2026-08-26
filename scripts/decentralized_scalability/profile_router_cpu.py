#!/usr/bin/env python3
"""Profile the production SP8 RequestRouter without network or scheduler work."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AdmissionReservation,
    AbortResult,
    IngressAck,
    OwnerState,
)
from nanodeploy.router.request_router import RequestRouter
from scripts.decentralized_scalability.common import (
    base_metadata,
    fixed_sp_planner_config,
    parse_positive_ints,
    parse_scaling_modes,
    requests_for_case,
    sp_load_snapshot,
    summarize,
    write_results,
)


@dataclass(frozen=True, slots=True)
class _ImmediateFlight:
    acks: tuple[IngressAck, ...]


class _ImmediateTransport:
    """Production-shaped transport whose positive receipt is immediately ready."""

    def __init__(self, engine_id: int) -> None:
        self.engine_id = engine_id
        self.ingress_version = 0
        self.request_ids: list[int] = []
        self.batch_sizes: list[int] = []
        self.poll_calls = 0

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> _ImmediateFlight:
        if len(commands) != len(reservations):
            raise AssertionError("command/reservation count mismatch")
        acks = []
        for command, reservation in zip(commands, reservations, strict=True):
            if (
                reservation.request_id != command.request_id
                or reservation.engine_id != self.engine_id
            ):
                raise AssertionError("invalid RequestRouter reservation")
            self.ingress_version += 1
            self.request_ids.append(command.request_id)
            acks.append(
                IngressAck(
                    request_id=command.request_id,
                    engine_id=self.engine_id,
                    enqueued=True,
                    ingress_version=self.ingress_version,
                )
            )
        self.batch_sizes.append(len(commands))
        return _ImmediateFlight(tuple(acks))

    def poll_admission_batch(
        self, flight: _ImmediateFlight
    ) -> tuple[bool, tuple[IngressAck, ...]]:
        self.poll_calls += 1
        return True, flight.acks

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        del allow_future_ingress
        return AbortResult(request_id=request_id, status="abort_pending")

    def clear_ingress_abort(self, request_id: int) -> None:
        del request_id


def run_trial(
    *,
    engines: int,
    requests: int,
    batch_size: int,
    prompt_tokens: int,
    attention_sp: int,
) -> dict[str, Any]:
    if min(engines, requests, batch_size, prompt_tokens, attention_sp) <= 0:
        raise ValueError("all benchmark dimensions must be positive")
    transports = {
        engine_id: _ImmediateTransport(engine_id)
        for engine_id in range(engines)
    }
    router = RequestRouter(
        transports,
        router_policy="least_batch",
        admission_batch_size=batch_size,
        admission_planner_config=fixed_sp_planner_config(
            attention_sp=attention_sp,
            capacity_requests=requests,
            prompt_tokens=prompt_tokens,
        ),
    )
    router.record_loads(
        tuple(
            sp_load_snapshot(
                engine_id,
                attention_sp=attention_sp,
                capacity_requests=requests,
            )
            for engine_id in transports
        )
    )

    payload = b"router-cpu-profiler-does-not-transfer-sequence-bytes"
    observed: list[IngressAck] = []
    started_at = time.perf_counter()
    for request_id in range(requests):
        router.submit_async(
            request_id=request_id,
            prompt_len=prompt_tokens,
            num_tokens=prompt_tokens,
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            sequence_payload=payload,
        )
    while len(observed) < requests:
        ready = router.poll_ingress_acks()
        if not ready and router.pending_ingress_count == 0:
            raise RuntimeError("router stopped with missing ingress receipts")
        observed.extend(ready)
    elapsed_s = time.perf_counter() - started_at

    sent_ids = [
        request_id
        for transport in transports.values()
        for request_id in transport.request_ids
    ]
    counts = Counter(sent_ids)
    per_engine_requests = tuple(
        len(transport.request_ids) for transport in transports.values()
    )
    batch_messages = sum(
        len(transport.batch_sizes) for transport in transports.values()
    )
    correctness = {
        "all_receipts_positive": (
            len(observed) == requests and all(ack.enqueued for ack in observed)
        ),
        "each_request_transferred_once": (
            len(counts) == requests and all(count == 1 for count in counts.values())
        ),
        "all_requests_pending_scheduler_commit": (
            router.pending_add_count == requests
            and all(
                router.owner(request_id) is not None
                and router.owner(request_id).state is OwnerState.PENDING_ADD
                for request_id in range(requests)
            )
        ),
        "router_has_no_ingress_flight": router.pending_ingress_count == 0,
        "balanced_equal_load_assignment": (
            max(per_engine_requests) - min(per_engine_requests) <= batch_size
        ),
    }
    if not all(correctness.values()):
        raise AssertionError(f"router profiler invariant failed: {correctness}")
    return {
        "elapsed_ms": elapsed_s * 1000.0,
        "requests_per_second": requests / elapsed_s,
        "microseconds_per_request": elapsed_s * 1_000_000.0 / requests,
        "batch_messages": batch_messages,
        "requests_per_message": requests / batch_messages,
        "per_engine_requests": per_engine_requests,
        "transport_poll_calls": sum(
            transport.poll_calls for transport in transports.values()
        ),
        "correctness": correctness,
    }


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    total_cases = (
        len(args.scaling_modes)
        * len(args.engine_counts)
        * len(args.batch_sizes)
        * len(args.prompt_lengths)
    )
    case_index = 0
    for scaling_mode in args.scaling_modes:
        for batch_size in args.batch_sizes:
            for prompt_tokens in args.prompt_lengths:
                for engines in args.engine_counts:
                    case_index += 1
                    requests = requests_for_case(
                        scaling_mode=scaling_mode,
                        engines=engines,
                        strong_total_requests=args.strong_total_requests,
                        requests_per_engine=args.requests_per_engine,
                    )
                    print(
                        f"[{case_index}/{total_cases}] mode={scaling_mode} "
                        f"engines={engines} requests={requests} "
                        f"batch={batch_size} prompt={prompt_tokens}",
                        flush=True,
                    )
                    trials = [
                        run_trial(
                            engines=engines,
                            requests=requests,
                            batch_size=batch_size,
                            prompt_tokens=prompt_tokens,
                            attention_sp=args.attention_sp,
                        )
                        for _ in range(args.repeats)
                    ]
                    elapsed = summarize(trial["elapsed_ms"] for trial in trials)
                    throughput = summarize(
                        trial["requests_per_second"] for trial in trials
                    )
                    per_request = summarize(
                        trial["microseconds_per_request"] for trial in trials
                    )
                    records.append(
                        {
                            "component": "request_router",
                            "scaling_mode": scaling_mode,
                            "engines": engines,
                            "attention_sp": args.attention_sp,
                            "logical_workers": engines * args.attention_sp,
                            "requests": requests,
                            "batch_size": batch_size,
                            "prompt_tokens": prompt_tokens,
                            "repeats": args.repeats,
                            "elapsed_ms": elapsed,
                            "requests_per_second": throughput,
                            "microseconds_per_request": per_request,
                            "median_requests_per_message": summarize(
                                trial["requests_per_message"] for trial in trials
                            )["p50"],
                            "throughput_retention_vs_one_engine": None,
                            "trial_correctness": [
                                trial["correctness"] for trial in trials
                            ],
                        }
                    )
                    print(
                        f"  qps median={throughput['p50']:.1f} "
                        f"p99-trial={throughput['p99']:.1f}",
                        flush=True,
                    )

    baselines = {
        (
            record["scaling_mode"],
            record["batch_size"],
            record["prompt_tokens"],
        ): record["requests_per_second"]["p50"]
        for record in records
        if record["engines"] == 1
    }
    for record in records:
        baseline = baselines.get(
            (
                record["scaling_mode"],
                record["batch_size"],
                record["prompt_tokens"],
            )
        )
        if baseline is not None:
            record["throughput_retention_vs_one_engine"] = (
                record["requests_per_second"]["p50"] / baseline
            )

    metadata = base_metadata("nanodeploy-decentralized-router-cpu-scalability")
    metadata.update(
        {
            "timed_scope": (
                "RequestRouter submit, SP8 least-batch planning, batching, and "
                "immediately-ready staged receipt processing"
            ),
            "excluded_scope": (
                "Sequence construction/serialization, ZMQ, Ray, LocalScheduler, "
                "workers, RDMA, and GPU execution"
            ),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    return metadata, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-counts",
        type=parse_positive_ints,
        default=parse_positive_ints("1,2,3"),
    )
    parser.add_argument("--attention-sp", type=int, default=8)
    parser.add_argument(
        "--scaling-modes",
        type=parse_scaling_modes,
        default=parse_scaling_modes("strong,weak"),
    )
    parser.add_argument("--strong-total-requests", type=int, default=4096)
    parser.add_argument("--requests-per-engine", type=int, default=4096)
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=parse_positive_ints("32,64,128"),
    )
    parser.add_argument(
        "--prompt-lengths",
        type=parse_positive_ints,
        default=parse_positive_ints("32,8000"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.attention_sp <= 0:
        parser.error("--attention-sp must be positive")
    for name in ("strong_total_requests", "requests_per_engine", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    metadata, records = run_benchmark(args)
    json_path, csv_path = write_results(
        args.output_dir,
        stem="router_cpu_scalability",
        metadata=metadata,
        records=records,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

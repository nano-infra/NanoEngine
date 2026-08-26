#!/usr/bin/env python3
"""Profile production LocalScheduler instances on CPU-only Ray actors.

One actor represents one production DP engine and owns an SP8 LocalScheduler.
Actors are hard-pinned one per selected Ray node. Sequence creation, frontend
planning, actor startup, fake worker-result construction, Ray RPC, consensus,
and GPU execution are excluded from the per-phase scheduler measurements.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Sequence

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    HIERARCHICAL_LOOP_COUNT,
    LocalDecodeBatch,
    WorkerDecodeResult,
)
from nanodeploy.engine.local_scheduler import LocalScheduler
from nanodeploy.engine.sequence import Sequence as NanoDeploySequence
from nanodeploy.router.admission_planner import (
    AdmissionPlanner,
    AdmissionPlannerConfig,
)
from nanodeploy.sampling_params import SamplingParams
from scripts.decentralized_scalability.common import (
    base_metadata,
    clear_proxy_environment,
    node_metadata,
    parse_positive_ints,
    parse_scaling_modes,
    select_cluster_nodes,
    summarize,
    write_results,
)


DEFAULT_MODEL = Path(
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V3/snapshots/"
    "e815299b0bcbac849fa540c768ef21845365c9eb"
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def scheduler_kv_blocks(
    *, batch_size: int, prompt_tokens: int, completion_tokens: int, attention_sp: int
) -> int:
    """Conservative per-SP-rank CPU block-manager capacity for fixed SP."""
    block_size = 64
    prompt_tokens_per_rank = _ceil_div(prompt_tokens, attention_sp)
    prompt_blocks = batch_size * _ceil_div(prompt_tokens_per_rank, block_size)
    mastered_requests = _ceil_div(batch_size, attention_sp)
    completion_blocks = mastered_requests * _ceil_div(
        completion_tokens + 1, block_size
    )
    return prompt_blocks + completion_blocks + batch_size + 128


def build_worker_results(
    batch: LocalDecodeBatch, *, token_base: int
) -> list[WorkerDecodeResult]:
    return [
        WorkerDecodeResult(
            wave_id=batch.wave_id,
            quantum_id=batch.quantum_id,
            global_rank=global_rank,
            forward_count=HIERARCHICAL_LOOP_COUNT,
            mastered_request_ids=batch.expected_request_ids(global_rank),
            sampled_token_ids=tuple(
                tuple(
                    token_base + offset
                    for offset in range(HIERARCHICAL_LOOP_COUNT)
                )
                for _ in batch.expected_request_ids(global_rank)
            ),
        )
        for global_rank in batch.per_rank_sequences
    ]


class LocalSchedulerCpuWorkload:
    """Reusable non-Ray workload so smoke tests exercise the same profiler."""

    def __init__(
        self,
        *,
        model: str,
        attention_dp: int,
        engine_id: int,
        batch_size: int,
        prompt_tokens: int,
        warmup_iterations: int,
        measured_iterations: int,
    ) -> None:
        if attention_dp not in {1, 2, 4}:
            raise ValueError("production SP8 attention_dp must be 1, 2, or 4")
        if not 0 <= engine_id < attention_dp:
            raise ValueError("engine_id is outside attention_dp")
        if min(batch_size, prompt_tokens, measured_iterations) <= 0:
            raise ValueError("batch, prompt, and measured iterations must be positive")
        if warmup_iterations < 0:
            raise ValueError("warmup iterations must be non-negative")

        self.batch_size = batch_size
        self.prompt_tokens = prompt_tokens
        self.warmup_iterations = warmup_iterations
        self.measured_iterations = measured_iterations
        self.total_profile_quantums = warmup_iterations + measured_iterations
        self.max_tokens = (
            self.total_profile_quantums + 2
        ) * HIERARCHICAL_LOOP_COUNT
        max_model_len = prompt_tokens + 1 + self.max_tokens
        num_kvcache_blocks = scheduler_kv_blocks(
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            completion_tokens=self.max_tokens,
            attention_sp=8,
        )

        self.config = Config(
            model=model,
            scheduler_arch="hierarchical",
            mode="decode",
            dummy_prefill=True,
            attention_dp=attention_dp,
            attention_sp=8,
            attention_tp=1,
            ffn_dp=1,
            ffn_ep=attention_dp * 8,
            ffn_tp=1,
            kvcache_block_size=64,
            num_kvcache_blocks=num_kvcache_blocks,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(
                max_model_len, batch_size * prompt_tokens + 1
            ),
            max_num_seqs=batch_size + 8,
            max_num_recv_seqs=batch_size + 8,
            hierarchical_queue_capacity=batch_size + 8,
            fixed_sp_size=8,
            segment_size=64,
            reserved_blocks_per_req=1.0,
            engine_id="local-scheduler-cpu-profile",
        )
        self.scheduler = LocalScheduler(
            self.config, self.config.hierarchical_topology.engine(engine_id)
        )
        self.quantum_id = 0

        token_ids = [token_id % 32_000 for token_id in range(prompt_tokens)]
        sequence_begin = time.perf_counter()
        command_sequences = []
        request_base = engine_id * 1_000_000_000
        for offset in range(batch_size):
            request_id = request_base + offset
            sequence = NanoDeploySequence(
                token_ids,
                sampling_params=SamplingParams(
                    temperature=0.1,
                    max_tokens=self.max_tokens,
                    ignore_eos=True,
                ),
            )
            sequence.seq_id = request_id
            command_sequences.append(
                (
                    AddCommand(
                        request_id=request_id,
                        prompt_len=prompt_tokens,
                        num_tokens=prompt_tokens,
                        max_tokens=self.max_tokens,
                        temperature=0.1,
                        ignore_eos=True,
                        wave_id=1,
                        sequence_payload=b"excluded-from-scheduler-profile",
                    ),
                    sequence,
                )
            )
        self.sequence_build_ms = (time.perf_counter() - sequence_begin) * 1000.0
        commands = tuple(item[0] for item in command_sequences)
        sequences = tuple(item[1] for item in command_sequences)

        planner_begin = time.perf_counter()
        planner = AdmissionPlanner(AdmissionPlannerConfig.from_config(self.config))
        shadow = planner.shadow_from_snapshot(
            self.scheduler.load_snapshot(wave_id=1, quantum_id=0)
        )
        if shadow is None:
            raise RuntimeError("empty LocalScheduler did not yield a planning shadow")
        reservations = tuple(planner.plan(shadow, command) for command in commands)
        if any(reservation is None for reservation in reservations):
            raise RuntimeError("frontend planner rejected scheduler profile workload")
        self.frontend_planner_ms = (time.perf_counter() - planner_begin) * 1000.0

        commit_begin = time.perf_counter()
        results = self.scheduler.commit_planned_batch(
            commands,
            tuple(
                reservation
                for reservation in reservations
                if reservation is not None
            ),
            sequences,
        )
        self.planned_commit_ms = (time.perf_counter() - commit_begin) * 1000.0
        if not all(result.accepted for result in results):
            raise RuntimeError(f"LocalScheduler rejected profile workload: {results}")
        self.request_ids = tuple(command.request_id for command in commands)

    def _quantum(self) -> dict[str, float]:
        begin = time.perf_counter()
        admitted = self.scheduler.admit()
        admit_ms = (time.perf_counter() - begin) * 1000.0
        if admitted:
            raise RuntimeError("steady-state scheduler unexpectedly admitted work")

        begin = time.perf_counter()
        self.scheduler.load_snapshot(wave_id=1, quantum_id=self.quantum_id)
        pre_load_ms = (time.perf_counter() - begin) * 1000.0

        begin = time.perf_counter()
        batch = self.scheduler.plan_decode(
            wave_id=1, quantum_id=self.quantum_id
        )
        plan_decode_ms = (time.perf_counter() - begin) * 1000.0
        if not batch.engine_has_real:
            raise RuntimeError("scheduler profile produced an all-dummy batch")

        begin = time.perf_counter()
        self.scheduler.mark_first_forward_started(batch)
        mark_first_schedule_ms = (time.perf_counter() - begin) * 1000.0

        # Fake output creation intentionally sits outside all scheduler timers.
        worker_results = build_worker_results(
            batch,
            token_base=100 + self.quantum_id * HIERARCHICAL_LOOP_COUNT,
        )

        begin = time.perf_counter()
        events = self.scheduler.postprocess(batch, worker_results)
        postprocess_ms = (time.perf_counter() - begin) * 1000.0
        if events:
            raise RuntimeError("scheduler profile workload finished too early")

        begin = time.perf_counter()
        self.scheduler.load_snapshot(wave_id=1, quantum_id=self.quantum_id + 1)
        post_load_ms = (time.perf_counter() - begin) * 1000.0
        self.quantum_id += 1
        scheduler_cpu_ms = (
            admit_ms
            + pre_load_ms
            + plan_decode_ms
            + mark_first_schedule_ms
            + postprocess_ms
            + post_load_ms
        )
        return {
            "admit_ms": admit_ms,
            "pre_load_ms": pre_load_ms,
            "plan_decode_ms": plan_decode_ms,
            "mark_first_schedule_ms": mark_first_schedule_ms,
            "postprocess_ms": postprocess_ms,
            "post_load_ms": post_load_ms,
            "scheduler_cpu_ms": scheduler_cpu_ms,
        }

    def profile(self) -> dict[str, Any]:
        for _ in range(self.warmup_iterations):
            self._quantum()
        samples = [self._quantum() for _ in range(self.measured_iterations)]
        final_load = self.scheduler.load_snapshot(
            wave_id=1, quantum_id=self.quantum_id
        )
        phase_names = tuple(samples[0])
        phase_stats = {
            phase_name: summarize(sample[phase_name] for sample in samples)
            for phase_name in phase_names
        }
        expected_completed = self.total_profile_quantums * HIERARCHICAL_LOOP_COUNT
        completed = tuple(
            self.scheduler._records[request_id].sequence.num_completed_tokens
            for request_id in self.request_ids
        )
        correctness = {
            "all_requests_remain_live": (
                not self.scheduler.is_finished()
                and len(self.scheduler._records) == self.batch_size
            ),
            "all_quantums_advanced": self.quantum_id == self.total_profile_quantums,
            "all_requests_advanced_equally": (
                completed == (expected_completed,) * self.batch_size
            ),
            "sp8_load_snapshot": len(final_load.rank_loads) == 8,
            "no_preemption": final_load.preemption_count == 0,
        }
        if not all(correctness.values()):
            raise AssertionError(
                f"LocalScheduler profile invariant failed: {correctness}"
            )
        scheduler_cpu_ms_total = sum(
            sample["scheduler_cpu_ms"] for sample in samples
        )
        return {
            "batch_size": self.batch_size,
            "prompt_tokens": self.prompt_tokens,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "sequence_build_ms_excluded": self.sequence_build_ms,
            "frontend_planner_ms_excluded": self.frontend_planner_ms,
            "planned_commit_ms": self.planned_commit_ms,
            "phase_stats": phase_stats,
            "scheduler_cpu_ms_total": scheduler_cpu_ms_total,
            "scheduler_quantums_per_second": (
                self.measured_iterations * 1000.0 / scheduler_cpu_ms_total
            ),
            "correctness": correctness,
        }


def _accelerator_assignment() -> dict[str, list[str]]:
    return {
        key: list(values)
        for key, values in ray.get_runtime_context().get_accelerator_ids().items()
    }


@ray.remote(num_cpus=1, num_gpus=0)
class _LocalSchedulerActor:
    def __init__(self, workload_args: dict[str, Any]) -> None:
        self.node_id = str(ray.get_runtime_context().get_node_id())
        self.node_ip = ray.util.get_node_ip_address()
        self.engine_id = int(workload_args["engine_id"])
        self.workload = LocalSchedulerCpuWorkload(**workload_args)

    def descriptor(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "requested_gpus": 0,
            "assigned_accelerators": _accelerator_assignment(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "num_kvcache_blocks": self.workload.config.num_kvcache_blocks,
        }

    def profile(self) -> dict[str, Any]:
        result = self.workload.profile()
        result["engine_id"] = self.engine_id
        return result


def _batch_counts(
    *, scaling_mode: str, nodes: int, total_or_per_engine_batch: int
) -> tuple[int, ...]:
    if scaling_mode == "weak":
        return (total_or_per_engine_batch,) * nodes
    base, extra = divmod(total_or_per_engine_batch, nodes)
    if base == 0:
        raise ValueError("strong-scaling total batch must be at least node count")
    return tuple(base + (index < extra) for index in range(nodes))


def configured_attention_dp(actor_count: int) -> int:
    """Map three physical actors onto the first three engines of DP4SP8.

    NanoDeploy has no deployable DP3SP8 topology. LocalScheduler itself owns
    only one DP engine, so a three-node CPU microbenchmark can still measure
    three independent local schedulers by using engine 0..2 from the nearest
    supported DP4SP8 configuration. Results are explicitly labelled partial.
    """
    if actor_count in {1, 2, 4}:
        return actor_count
    if actor_count == 3:
        return 4
    raise ValueError("LocalScheduler actor count must be 1, 2, 3, or 4")


def _validate_placements(
    descriptors: Sequence[dict[str, Any]], nodes: Sequence[dict[str, Any]]
) -> None:
    for descriptor, node in zip(descriptors, nodes, strict=True):
        if descriptor["node_id"] != str(node["NodeID"]):
            raise RuntimeError(
                f"engine {descriptor['engine_id']} placement mismatch"
            )
        if (
            descriptor["requested_gpus"] != 0
            or descriptor["assigned_accelerators"].get("GPU", [])
        ):
            raise RuntimeError(
                f"CPU-only scheduler actor received GPU resources: {descriptor}"
            )


def _profile_case(
    *,
    nodes: Sequence[dict[str, Any]],
    model: str,
    scaling_mode: str,
    batch_size: int,
    prompt_tokens: int,
    warmup_iterations: int,
    iterations: int,
    startup_timeout_s: float,
) -> dict[str, Any]:
    batch_counts = _batch_counts(
        scaling_mode=scaling_mode,
        nodes=len(nodes),
        total_or_per_engine_batch=batch_size,
    )
    config_attention_dp = configured_attention_dp(len(nodes))
    actors = []
    try:
        for engine_id, (node, local_batch) in enumerate(
            zip(nodes, batch_counts, strict=True)
        ):
            workload_args = {
                "model": model,
                "attention_dp": config_attention_dp,
                "engine_id": engine_id,
                "batch_size": local_batch,
                "prompt_tokens": prompt_tokens,
                "warmup_iterations": warmup_iterations,
                "measured_iterations": iterations,
            }
            actor = _LocalSchedulerActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    str(node["NodeID"]), soft=False
                )
            ).remote(workload_args)
            actors.append(actor)
        descriptors = tuple(
            ray.get(
                [actor.descriptor.remote() for actor in actors],
                timeout=startup_timeout_s,
            )
        )
        _validate_placements(descriptors, nodes)
        begin = time.perf_counter()
        profiles = tuple(
            ray.get(
                [actor.profile.remote() for actor in actors],
                timeout=startup_timeout_s,
            )
        )
        concurrent_wall_ms = (time.perf_counter() - begin) * 1000.0
        if not all(all(profile["correctness"].values()) for profile in profiles):
            raise AssertionError("one or more LocalScheduler actors failed invariants")

        scheduler_critical_ms = max(
            profile["scheduler_cpu_ms_total"] for profile in profiles
        )
        aggregate_quantums = len(nodes) * iterations
        return {
            "component": "local_scheduler",
            "scaling_mode": scaling_mode,
            "nodes": len(nodes),
            "engines": len(nodes),
            "configured_attention_dp": config_attention_dp,
            "topology_scope": (
                "complete_production_topology"
                if config_attention_dp == len(nodes)
                else "partial_dp4_cpu_scaling_only"
            ),
            "attention_sp": 8,
            "logical_workers": len(nodes) * 8,
            "batch_size_argument": batch_size,
            "local_batch_sizes": batch_counts,
            "total_active_requests": sum(batch_counts),
            "prompt_tokens": prompt_tokens,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": iterations,
            "planned_commit_ms": summarize(
                profile["planned_commit_ms"] for profile in profiles
            ),
            "admit_ms": summarize(
                profile["phase_stats"]["admit_ms"]["mean"]
                for profile in profiles
            ),
            "load_snapshot_ms": summarize(
                (
                    profile["phase_stats"]["pre_load_ms"]["mean"]
                    + profile["phase_stats"]["post_load_ms"]["mean"]
                )
                for profile in profiles
            ),
            "plan_decode_ms": summarize(
                profile["phase_stats"]["plan_decode_ms"]["mean"]
                for profile in profiles
            ),
            "mark_first_schedule_ms": summarize(
                profile["phase_stats"]["mark_first_schedule_ms"]["mean"]
                for profile in profiles
            ),
            "postprocess_ms": summarize(
                profile["phase_stats"]["postprocess_ms"]["mean"]
                for profile in profiles
            ),
            "scheduler_cpu_ms": summarize(
                profile["phase_stats"]["scheduler_cpu_ms"]["mean"]
                for profile in profiles
            ),
            "aggregate_scheduler_quantums_per_second": (
                aggregate_quantums * 1000.0 / scheduler_critical_ms
            ),
            "concurrent_harness_wall_ms": concurrent_wall_ms,
            "aggregate_harness_quantums_per_second": (
                aggregate_quantums * 1000.0 / concurrent_wall_ms
            ),
            "weak_scaling_efficiency_vs_one_node": None,
            "placement": descriptors,
            "per_engine_profiles": profiles,
        }
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--node-counts",
        type=parse_positive_ints,
        default=parse_positive_ints("1,2,3"),
    )
    parser.add_argument(
        "--node-ip",
        action="append",
        default=[],
        help="Target Ray NodeManagerAddress; repeat in desired node order.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--scaling-modes",
        type=parse_scaling_modes,
        default=parse_scaling_modes("strong,weak"),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=parse_positive_ints("32,64,128"),
        help=(
            "Strong mode: total batch across engines; weak mode: batch per engine."
        ),
    )
    parser.add_argument(
        "--prompt-lengths",
        type=parse_positive_ints,
        default=parse_positive_ints("32,8000"),
    )
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if any(node_count not in {1, 2, 3, 4} for node_count in args.node_counts):
        parser.error("--node-counts must contain only 1,2,3,4")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations must be non-negative")
    if args.iterations <= 0 or args.startup_timeout_s <= 0:
        parser.error("iterations and timeout must be positive")
    if not (Path(args.model) / "config.json").is_file():
        parser.error(f"model config not found below --model {args.model!r}")
    if "strong" in args.scaling_modes and min(args.batch_sizes) < max(
        args.node_counts
    ):
        parser.error("strong-scaling batch size must be at least max node count")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if ray.is_initialized():
        raise RuntimeError("profiler requires a fresh Ray driver")
    removed_proxies = clear_proxy_environment()
    ray_context = ray.init(
        address=args.ray_address,
        ignore_reinit_error=False,
        log_to_driver=False,
    )
    records: list[dict[str, Any]] = []
    try:
        max_nodes = max(args.node_counts)
        selected_nodes = select_cluster_nodes(
            ray.nodes(),
            count=max_nodes,
            requested_node_ips=args.node_ip,
        )
        total_cases = (
            len(args.scaling_modes)
            * len(args.batch_sizes)
            * len(args.prompt_lengths)
            * len(args.node_counts)
        )
        case_index = 0
        for scaling_mode in args.scaling_modes:
            for batch_size in args.batch_sizes:
                for prompt_tokens in args.prompt_lengths:
                    for node_count in args.node_counts:
                        case_index += 1
                        print(
                            f"[{case_index}/{total_cases}] mode={scaling_mode} "
                            f"nodes={node_count} batch={batch_size} "
                            f"prompt={prompt_tokens}",
                            flush=True,
                        )
                        record = _profile_case(
                            nodes=selected_nodes[:node_count],
                            model=args.model,
                            scaling_mode=scaling_mode,
                            batch_size=batch_size,
                            prompt_tokens=prompt_tokens,
                            warmup_iterations=args.warmup_iterations,
                            iterations=args.iterations,
                            startup_timeout_s=args.startup_timeout_s,
                        )
                        records.append(record)
                        print(
                            "  scheduler qps="
                            f"{record['aggregate_scheduler_quantums_per_second']:.1f} "
                            "cpu mean="
                            f"{record['scheduler_cpu_ms']['mean']:.3f} ms",
                            flush=True,
                        )

        weak_baselines = {
            (record["batch_size_argument"], record["prompt_tokens"]): record[
                "aggregate_scheduler_quantums_per_second"
            ]
            for record in records
            if record["scaling_mode"] == "weak" and record["nodes"] == 1
        }
        for record in records:
            if record["scaling_mode"] != "weak":
                continue
            baseline = weak_baselines.get(
                (record["batch_size_argument"], record["prompt_tokens"])
            )
            if baseline is not None:
                record["weak_scaling_efficiency_vs_one_node"] = (
                    record["aggregate_scheduler_quantums_per_second"]
                    / (record["nodes"] * baseline)
                )

        metadata = base_metadata(
            "nanodeploy-local-scheduler-ray-cpu-scalability"
        )
        metadata.update(
            {
                "ray_version": ray.__version__,
                "ray_address": ray_context.address_info.get(
                    "gcs_address", args.ray_address
                ),
                "proxy_variables_removed": removed_proxies,
                "selected_nodes": node_metadata(selected_nodes),
                "timed_scope": (
                    "production LocalScheduler steady-state admit, SP8 load "
                    "snapshot, plan_decode, first-schedule marking, and postprocess"
                ),
                "excluded_scope": (
                    "Config/Scheduler construction, Sequence construction, frontend "
                    "planning, Ray actor RPC, consensus, fake result construction, "
                    "ModelRunner, RDMA, CUDA, and GPU execution"
                ),
                "model_config_path": str(Path(args.model) / "config.json"),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
        )
        json_path, csv_path = write_results(
            args.output_dir,
            stem="local_scheduler_cpu_scalability",
            metadata=metadata,
            records=records,
        )
        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

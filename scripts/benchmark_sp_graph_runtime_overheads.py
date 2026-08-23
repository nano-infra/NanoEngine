#!/usr/bin/env python3
"""Microbenchmarks for SP CUDA Graph routing runtime overhead."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import nanodeploy
from nanodeploy.worker.sp_graph_policy import (
    copy_decode_context_to_graph_vars,
    copy_graph_actual_attn_bs,
    copy_graph_q_dst_rows,
    materialize_sp_graph_padding,
    select_decode_graph_bucket,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        required=True,
        choices=["bucket", "metadata", "replay-submit"],
    )
    parser.add_argument(
        "--shape-json",
        type=Path,
        help="ModelRunner runtime-overhead summary containing observed shapes.",
    )
    parser.add_argument(
        "--timing-json",
        type=Path,
        nargs="+",
        help="Non-profiled ModelRunner summaries for replay-submit reporting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=_positive_int, default=None)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument(
        "--cases",
        default="no_padding,issue1_padded",
        help="Comma-separated metadata cases.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help="Pin the CPU benchmark to this logical CPU.",
    )
    args = parser.parse_args()

    if args.component in {"bucket", "metadata"} and args.shape_json is None:
        parser.error(f"--shape-json is required for {args.component}")
    if args.component == "replay-submit" and not args.timing_json:
        parser.error("--timing-json is required for replay-submit")
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest(component: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "component": component,
        "git_sha": _git_sha(),
        "nanodeploy_file": str(Path(nanodeploy.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "pid": os.getpid(),
    }
    if component == "metadata" and torch.cuda.is_available():
        device = torch.cuda.current_device()
        manifest.update(
            {
                "cuda_device": device,
                "cuda_device_name": torch.cuda.get_device_name(device),
            }
        )
    return manifest


def _summarize(values: list[float], suffix: str) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        f"min_{suffix}": float(samples.min()),
        f"p50_{suffix}": float(np.percentile(samples, 50)),
        f"p95_{suffix}": float(np.percentile(samples, 95)),
        f"max_{suffix}": float(samples.max()),
        f"mean_{suffix}": float(samples.mean()),
    }


def _most_common_shape(summary: dict[str, Any]) -> dict[str, int]:
    shapes = summary.get("shape_counts")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("shape JSON has no observed shape_counts")
    shape = shapes[0]
    required = {
        "actual_master_bs",
        "graph_master_bs",
        "actual_attn_bs",
        "graph_attn_bs",
        "block_table_rows",
        "block_table_width",
    }
    missing = required.difference(shape)
    if missing:
        raise ValueError(f"observed shape is missing: {sorted(missing)}")
    return {key: int(shape[key]) for key in required}


def _graph_candidates(
    summary: dict[str, Any],
) -> tuple[list[int], dict[int, list[int]]]:
    master_buckets = [int(value) for value in summary["graph_master_rank_bs"]]
    sp_graph_map = {
        int(master_bs): [int(value) for value in attn_buckets]
        for master_bs, attn_buckets in summary["sp_graph_map"].items()
    }
    if not master_buckets or not sp_graph_map:
        raise ValueError("shape JSON has no captured Graph candidates")
    return master_buckets, sp_graph_map


def _run_bucket_case(
    *,
    actual_master_bs: int,
    actual_attn_bs: int,
    master_buckets: list[int],
    sp_graph_map: dict[int, list[int]],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    graphs = {
        (master_bs, attn_bs): object()
        for master_bs, attn_buckets in sp_graph_map.items()
        for attn_bs in attn_buckets
    }

    def run_loop(count: int) -> object:
        graph = None
        for _ in range(count):
            master_bs, graph_attn_bs = select_decode_graph_bucket(
                actual_master_bs,
                actual_attn_bs,
                graph_master_rank_bs=master_buckets,
                sp_graph_map=sp_graph_map,
                use_sp_a2a=True,
                sp_backend="hao_basic",
                fixed_sp_size=0,
                sp_comm_bs=None,
            )
            graph = graphs[(master_bs, graph_attn_bs)]
        assert graph is not None
        return graph

    if warmup:
        run_loop(warmup)

    per_call_ns = []
    for _ in range(repeats):
        begin_ns = time.perf_counter_ns()
        run_loop(iterations)
        elapsed_ns = time.perf_counter_ns() - begin_ns
        per_call_ns.append(elapsed_ns / iterations)

    selected_master_bs, selected_attn_bs = select_decode_graph_bucket(
        actual_master_bs,
        actual_attn_bs,
        graph_master_rank_bs=master_buckets,
        sp_graph_map=sp_graph_map,
        use_sp_a2a=True,
        sp_backend="hao_basic",
        fixed_sp_size=0,
        sp_comm_bs=None,
    )
    return {
        "actual_master_bs": actual_master_bs,
        "actual_attn_bs": actual_attn_bs,
        "graph_master_bs": selected_master_bs,
        "graph_attn_bs": selected_attn_bs,
        "repeat_ns_per_call": per_call_ns,
        "summary": _summarize(per_call_ns, "ns"),
    }


def _benchmark_bucket(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    source = _load_json(args.shape_json)
    shape = _most_common_shape(source)
    master_buckets, sp_graph_map = _graph_candidates(source)
    warmup = 10_000 if args.warmup is None else args.warmup
    iterations = 1_000_000 if args.iterations is None else args.iterations

    typical = _run_bucket_case(
        actual_master_bs=shape["actual_master_bs"],
        actual_attn_bs=shape["actual_attn_bs"],
        master_buckets=master_buckets,
        sp_graph_map=sp_graph_map,
        warmup=warmup,
        iterations=iterations,
        repeats=args.repeats,
    )
    boundary_master_bs = master_buckets[-1]
    boundary_attn_bs = sp_graph_map[boundary_master_bs][-1]
    boundary = _run_bucket_case(
        actual_master_bs=boundary_master_bs,
        actual_attn_bs=boundary_attn_bs,
        master_buckets=master_buckets,
        sp_graph_map=sp_graph_map,
        warmup=warmup,
        iterations=iterations,
        repeats=args.repeats,
    )
    return {
        "schema_version": 1,
        "manifest": _manifest("bucket"),
        "source_shape_json": str(args.shape_json.resolve()),
        "warmup": warmup,
        "iterations": iterations,
        "repeats": args.repeats,
        "cases": {"typical": typical, "boundary": boundary},
    }


def _allocate_metadata_case(
    source: dict[str, Any], shape: dict[str, int], case_name: str
) -> tuple[dict[str, torch.Tensor | None], SimpleNamespace, torch.Tensor, torch.Tensor, dict[str, int]]:
    max_num_seqs = int(source["max_num_seqs"])
    max_num_recv_seqs = int(source["max_num_recv_seqs"])
    attention_sp = int(source["attention_sp"])
    max_model_len = int(source["max_model_len"])
    block_size = int(source["kvcache_block_size"])
    max_master_bs = min(max_num_seqs, 512)
    max_attn_bs = max_master_bs + max_num_recv_seqs
    max_num_blocks = (max_model_len + block_size - 1) // block_size

    if case_name == "no_padding":
        actual_master_bs = shape["graph_master_bs"]
        actual_attn_bs = shape["graph_attn_bs"]
    elif case_name == "issue1_padded":
        actual_master_bs = shape["actual_master_bs"]
        actual_attn_bs = shape["actual_attn_bs"]
        if (
            actual_master_bs == shape["graph_master_bs"]
            and actual_attn_bs == shape["graph_attn_bs"]
        ):
            raise ValueError(
                "issue1_padded requires an observed shape with a padded tail"
            )
    else:
        raise ValueError(f"Unsupported metadata case: {case_name}")
    graph_master_bs = shape["graph_master_bs"]
    graph_attn_bs = shape["graph_attn_bs"]
    if actual_master_bs > graph_master_bs or actual_attn_bs > graph_attn_bs:
        raise ValueError(f"Invalid observed Graph shape for {case_name}: {shape}")
    if graph_master_bs > max_master_bs or graph_attn_bs > max_attn_bs:
        raise ValueError(
            f"Observed Graph shape exceeds configured capacities: {shape}"
        )

    device = torch.device("cuda")
    graph_vars: dict[str, torch.Tensor | None] = {
        "input_ids": torch.zeros(max_master_bs, dtype=torch.int64, device=device),
        "positions": torch.zeros(max_master_bs, dtype=torch.int64, device=device),
        "slot_mapping": torch.zeros(max_master_bs, dtype=torch.int32, device=device),
        "context_lens": torch.zeros(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        "global_context_lens": torch.zeros(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        "q_mask": torch.zeros(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        "q_dst_row_indices": torch.full((attention_sp, max_num_seqs), -1, dtype=torch.int32, device=device),
        "actual_attn_bs": torch.zeros((), dtype=torch.int32, device=device),
        "res_lse_mask": torch.zeros(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        "block_tables": torch.zeros(max_attn_bs, max_num_blocks, dtype=torch.int32, device=device),
        "context_lens_for_attn": torch.zeros(max_attn_bs, dtype=torch.int32, device=device),
        "q_slice_get": torch.full((max_master_bs,), -1, dtype=torch.int32, device=device),
        "q_slice_fill": torch.full((max_master_bs,), -1, dtype=torch.int32, device=device),
        "q_copy_mask": torch.zeros(max_master_bs, dtype=torch.int32, device=device),
        "res_slice_get_to_buffer_output": torch.full((max_master_bs,), -1, dtype=torch.int32, device=device),
        "res_slice_fill_to_buffer_output": torch.full((max_master_bs,), -1, dtype=torch.int32, device=device),
        "res_to_buffer_output_mask": torch.zeros(max_master_bs, dtype=torch.int32, device=device),
        "res_slice_get_to_buffer_input": torch.full((max_num_recv_seqs,), -1, dtype=torch.int32, device=device),
        "res_slice_fill_to_buffer_input": torch.full((max_num_recv_seqs,), -1, dtype=torch.int32, device=device),
        "res_to_buffer_input_mask": torch.zeros(max_num_recv_seqs, dtype=torch.int32, device=device),
        "q_offsets": torch.zeros(attention_sp + 1, dtype=torch.int32, device=device),
    }

    block_table_rows = actual_attn_bs
    block_table_width = min(shape["block_table_width"], max_num_blocks)
    if block_table_rows <= 0 or block_table_width <= 0:
        raise ValueError("Observed block-table shape must be non-empty")
    actual_block_tables = torch.arange(
        block_table_width, dtype=torch.int32, device=device
    )[None, :].expand(block_table_rows, -1).clone()
    remote_rows = min(
        max_num_recv_seqs, max(actual_attn_bs - actual_master_bs, 0)
    )
    context = SimpleNamespace(
        slot_mapping=torch.zeros(actual_master_bs, dtype=torch.int32, device=device),
        q_dst_row_indices=torch.zeros(attention_sp, max_num_seqs, dtype=torch.int32, device=device),
        context_lens=torch.ones(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        global_context_lens=torch.ones(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        q_mask=torch.ones(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        res_lse_mask=torch.ones(attention_sp, max_master_bs, dtype=torch.int32, device=device),
        block_tables=actual_block_tables,
        context_lens_for_attn=torch.ones(actual_attn_bs, dtype=torch.int32, device=device),
        q_slice_get=torch.arange(actual_master_bs, dtype=torch.int32, device=device),
        q_slice_fill=torch.arange(actual_master_bs, dtype=torch.int32, device=device),
        q_copy_mask=torch.ones(actual_master_bs, dtype=torch.int32, device=device),
        res_slice_get_to_buffer_output=torch.arange(actual_master_bs, dtype=torch.int32, device=device),
        res_slice_fill_to_buffer_output=torch.arange(actual_master_bs, dtype=torch.int32, device=device),
        res_to_buffer_output_mask=torch.ones(actual_master_bs, dtype=torch.int32, device=device),
        res_slice_get_to_buffer_input=torch.arange(remote_rows, dtype=torch.int32, device=device),
        res_slice_fill_to_buffer_input=torch.arange(remote_rows, dtype=torch.int32, device=device),
        res_to_buffer_input_mask=torch.ones(remote_rows, dtype=torch.int32, device=device),
        q_offsets=torch.zeros(attention_sp + 1, dtype=torch.int32, device=device),
        use_sp_a2a=True,
        attention_compute_bs=actual_attn_bs,
    )
    input_ids = torch.zeros(actual_master_bs, dtype=torch.int64, device=device)
    positions = torch.zeros(actual_master_bs, dtype=torch.int64, device=device)
    case_shape = {
        "actual_master_bs": actual_master_bs,
        "graph_master_bs": graph_master_bs,
        "actual_attn_bs": actual_attn_bs,
        "graph_attn_bs": graph_attn_bs,
        "block_table_rows": block_table_rows,
        "block_table_width": block_table_width,
        "sp_rank": int(source["rank"]) % attention_sp,
        "max_num_seqs": max_num_seqs,
    }
    return graph_vars, context, input_ids, positions, case_shape


def _routing_metadata_once(
    graph_vars: dict[str, torch.Tensor | None],
    context: SimpleNamespace,
    shape: dict[str, int],
) -> None:
    copy_graph_q_dst_rows(
        graph_vars["q_dst_row_indices"], context.q_dst_row_indices
    )
    copy_graph_actual_attn_bs(
        graph_vars["actual_attn_bs"], shape["actual_attn_bs"]
    )
    materialize_sp_graph_padding(
        graph_vars,
        actual_block_tables=context.block_tables,
        actual_attn_bs=shape["actual_attn_bs"],
        graph_attn_bs=shape["graph_attn_bs"],
        actual_master_bs=shape["actual_master_bs"],
        graph_master_bs=shape["graph_master_bs"],
        local_result_rows=context.res_slice_get_to_buffer_output.numel(),
        sp_rank=shape["sp_rank"],
        max_num_seqs=shape["max_num_seqs"],
    )


def _time_cuda_operation(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    host_us_per_call = []
    for _ in range(repeats):
        host_elapsed_ns = 0
        for _ in range(iterations):
            # Start every host sample from an idle stream.  Otherwise a long
            # enqueue batch eventually back-pressures on device execution and
            # the CPU timer stops measuring submission overhead.
            torch.cuda.synchronize()
            host_begin_ns = time.perf_counter_ns()
            operation()
            host_elapsed_ns += time.perf_counter_ns() - host_begin_ns
        # Keep completion outside the timed interval while ensuring that the
        # final operation cannot leak into the next repeat.
        torch.cuda.synchronize()
        host_us_per_call.append(host_elapsed_ns / iterations / 1_000)

    device_us_per_call = []
    for _ in range(repeats):
        begin_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        begin_event.record()
        for _ in range(iterations):
            operation()
        end_event.record()
        end_event.synchronize()
        device_us_per_call.append(
            begin_event.elapsed_time(end_event) * 1_000 / iterations
        )
    return {
        "host_us_per_call": host_us_per_call,
        "device_span_us_per_call": device_us_per_call,
        "host_summary": _summarize(host_us_per_call, "us"),
        "device_span_summary": _summarize(device_us_per_call, "us"),
    }


def _routing_bytes(shape: dict[str, int], attention_sp: int) -> int:
    q_dst_bytes = 2 * attention_sp * shape["max_num_seqs"] * 4
    scalar_bytes = 4
    attn_tail = shape["graph_attn_bs"] - shape["actual_attn_bs"]
    master_tail = shape["graph_master_bs"] - shape["actual_master_bs"]
    attn_padding_bytes = attn_tail * (1 + shape["block_table_width"]) * 4
    master_padding_bytes = master_tail * 5 * 4
    return q_dst_bytes + scalar_bytes + attn_padding_bytes + master_padding_bytes


def _all_metadata_bytes(
    graph_vars: dict[str, torch.Tensor | None],
    context: SimpleNamespace,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    shape: dict[str, int],
    attention_sp: int,
) -> int:
    def destination_bytes(name: str) -> int:
        tensor = graph_vars[name]
        assert tensor is not None
        return tensor.numel() * tensor.element_size()

    def source_bytes(tensor: torch.Tensor, destination_name: str) -> int:
        destination = graph_vars[destination_name]
        assert destination is not None
        return tensor.numel() * destination.element_size()

    writes = 0
    for name, source_tensor in (
        ("input_ids", input_ids),
        ("positions", positions),
        ("slot_mapping", context.slot_mapping),
        ("q_dst_row_indices", context.q_dst_row_indices),
        ("context_lens", context.context_lens),
        ("global_context_lens", context.global_context_lens),
        ("q_mask", context.q_mask),
        ("res_lse_mask", context.res_lse_mask),
        ("block_tables", context.block_tables),
        ("context_lens_for_attn", context.context_lens_for_attn),
        ("q_slice_get", context.q_slice_get),
        ("q_slice_fill", context.q_slice_fill),
        ("q_copy_mask", context.q_copy_mask),
        (
            "res_slice_get_to_buffer_output",
            context.res_slice_get_to_buffer_output,
        ),
        (
            "res_slice_fill_to_buffer_output",
            context.res_slice_fill_to_buffer_output,
        ),
        ("res_to_buffer_output_mask", context.res_to_buffer_output_mask),
        (
            "res_slice_get_to_buffer_input",
            context.res_slice_get_to_buffer_input,
        ),
        (
            "res_slice_fill_to_buffer_input",
            context.res_slice_fill_to_buffer_input,
        ),
        ("res_to_buffer_input_mask", context.res_to_buffer_input_mask),
        ("q_offsets", context.q_offsets),
    ):
        writes += destination_bytes(name)
        writes += source_bytes(source_tensor, name)
    writes += destination_bytes("actual_attn_bs")

    routing_writes = _routing_bytes(shape, attention_sp)
    q_dst_and_scalar = (
        2 * attention_sp * shape["max_num_seqs"] * 4 + 4
    )
    writes += routing_writes - q_dst_and_scalar
    return writes


def _benchmark_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("metadata benchmark requires CUDA")
    source = _load_json(args.shape_json)
    observed_shape = _most_common_shape(source)
    warmup = 200 if args.warmup is None else args.warmup
    iterations = 2_000 if args.iterations is None else args.iterations
    case_names = [name.strip() for name in args.cases.split(",") if name.strip()]
    if not case_names:
        raise ValueError("--cases must not be empty")

    results = {}
    for case_name in case_names:
        graph_vars, context, input_ids, positions, shape = _allocate_metadata_case(
            source, observed_shape, case_name
        )
        routing_result = _time_cuda_operation(
            lambda: _routing_metadata_once(graph_vars, context, shape),
            warmup=warmup,
            iterations=iterations,
            repeats=args.repeats,
        )
        all_result = _time_cuda_operation(
            lambda: copy_decode_context_to_graph_vars(
                graph_vars,
                input_ids,
                positions,
                shape["actual_master_bs"],
                shape["graph_master_bs"],
                shape["graph_attn_bs"],
                context,
                sp_rank=shape["sp_rank"],
                max_num_seqs=shape["max_num_seqs"],
            ),
            warmup=warmup,
            iterations=iterations,
            repeats=args.repeats,
        )
        results[case_name] = {
            "shape": shape,
            "routing_bytes_touched_estimate": _routing_bytes(
                shape, int(source["attention_sp"])
            ),
            "all_metadata_bytes_touched_estimate": _all_metadata_bytes(
                graph_vars,
                context,
                input_ids,
                positions,
                shape,
                int(source["attention_sp"]),
            ),
            "routing_metadata": routing_result,
            "all_graph_metadata": all_result,
        }
        del graph_vars, context, input_ids, positions
        torch.cuda.empty_cache()

    return {
        "schema_version": 1,
        "manifest": _manifest("metadata"),
        "source_shape_json": str(args.shape_json.resolve()),
        "warmup": warmup,
        "iterations": iterations,
        "repeats": args.repeats,
        "cases": results,
    }


def _report_replay_submit(args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    key = "nanodeploy.graph.replay_submit"
    for path in args.timing_json:
        source = _load_json(path)
        summary = source.get("sample_summaries", {}).get(key)
        if summary is None:
            raise ValueError(f"{path} has no {key!r} samples")
        runs.append(
            {
                "source": str(path.resolve()),
                "rank": source.get("rank"),
                "summary": summary,
                "most_common_shape": _most_common_shape(source),
            }
        )
    return {
        "schema_version": 1,
        "manifest": _manifest("replay-submit"),
        "runs": runs,
        "note": (
            "Host graph.replay() submission only; this is not device Graph "
            "execution time."
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.component == "bucket":
        result = _benchmark_bucket(args)
    elif args.component == "metadata":
        result = _benchmark_metadata(args)
    else:
        result = _report_replay_submit(args)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

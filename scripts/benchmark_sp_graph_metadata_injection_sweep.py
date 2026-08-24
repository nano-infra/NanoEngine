#!/usr/bin/env python3
"""Sweep full SP Graph metadata injection latency for no-padding (m, n)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.benchmark_sp_graph_runtime_overheads import (
    _all_metadata_bytes,
    _allocate_metadata_case,
    _load_json,
    _manifest,
    _time_cuda_operation,
    _write_json,
)
from scripts._sp_graph_metadata_injection_fused import (
    copy_decode_context_to_graph_vars_fused_no_padding,
)
from nanodeploy.worker.sp_graph_policy import copy_decode_context_to_graph_vars


@dataclass(frozen=True)
class MnPair:
    """No-padding Graph shape: m=master batch, n=attention batch."""

    m: int
    n: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _mn_pair(value: str) -> MnPair:
    match = re.fullmatch(r"\s*(\d+)\s*[,x:]\s*(\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid (m,n) pair {value!r}; expected M,N (also accepts MxN or M:N)"
        )
    pair = MnPair(m=int(match.group(1)), n=int(match.group(2)))
    if pair.m <= 0 or pair.n <= 0:
        raise argparse.ArgumentTypeError("m and n must both be positive")
    return pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape-json",
        type=Path,
        required=True,
        help=(
            "ModelRunner runtime-overhead summary supplying the same persistent "
            "buffer capacities used by the production metadata benchmark."
        ),
    )
    parser.add_argument(
        "--mn-pairs",
        type=_mn_pair,
        nargs="+",
        metavar="M,N",
        help=(
            "No-padding (master batch, attention batch) pairs. If omitted, "
            "sweep every captured pair in shape-json's sp_graph_map."
        ),
    )
    parser.add_argument(
        "--block-table-width",
        type=_positive_int,
        help=(
            "Active block-table width for every pair. Defaults to the most "
            "frequent observed width in shape-json."
        ),
    )
    parser.add_argument(
        "--capacity-mode",
        choices=("fixed", "per-pair"),
        default="fixed",
        help=(
            "Persistent metadata buffer sizing. 'fixed' uses shape-json's "
            "global capacities for every pair, matching the current production "
            "allocation. 'per-pair' sets max_num_seqs=m and "
            "max_num_recv_seqs=n-m for each pair."
        ),
    )
    parser.add_argument(
        "--implementation",
        choices=("production", "fused"),
        default="production",
        help=(
            "Metadata injection implementation. 'production' calls the current "
            "serving path; 'fused' uses a benchmark-only two-launch Triton "
            "prototype and verifies it against production before timing."
        ),
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=_positive_int, default=2_000)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _captured_mn_pairs(source: dict[str, Any]) -> list[MnPair]:
    master_values = source.get("graph_master_rank_bs")
    graph_map = source.get("sp_graph_map")
    if not isinstance(master_values, list) or not isinstance(graph_map, dict):
        raise ValueError(
            "shape JSON must contain graph_master_rank_bs and sp_graph_map "
            "when --mn-pairs is omitted"
        )

    pairs = []
    for raw_m in master_values:
        m = int(raw_m)
        raw_n_values = graph_map.get(str(m), graph_map.get(m))
        if not isinstance(raw_n_values, list) or not raw_n_values:
            raise ValueError(f"shape JSON has no attention buckets for m={m}")
        pairs.extend(MnPair(m=m, n=int(raw_n)) for raw_n in raw_n_values)
    return pairs


def _block_table_width(source: dict[str, Any], override: int | None) -> int:
    if override is not None:
        width = override
    else:
        shape_counts = source.get("shape_counts")
        if not isinstance(shape_counts, list) or not shape_counts:
            raise ValueError(
                "shape JSON has no observed shape_counts; pass --block-table-width"
            )
        width = int(shape_counts[0]["block_table_width"])

    max_model_len = int(source["max_model_len"])
    block_size = int(source["kvcache_block_size"])
    max_num_blocks = (max_model_len + block_size - 1) // block_size
    if not 0 < width <= max_num_blocks:
        raise ValueError(
            f"block-table width {width} exceeds valid range [1, {max_num_blocks}]"
        )
    return width


def _validate_mn_pairs(
    pairs: list[MnPair], source: dict[str, Any]
) -> list[MnPair]:
    if not pairs:
        raise ValueError("at least one (m,n) pair is required")
    if len(set(pairs)) != len(pairs):
        raise ValueError("(m,n) pairs must not contain duplicates")

    max_num_seqs = int(source["max_num_seqs"])
    max_num_recv_seqs = int(source["max_num_recv_seqs"])
    max_master_bs = min(max_num_seqs, 512)
    max_attn_bs = max_master_bs + max_num_recv_seqs
    for pair in pairs:
        if pair.m > max_master_bs:
            raise ValueError(
                f"m={pair.m} exceeds master capacity {max_master_bs}"
            )
        if pair.n < pair.m:
            raise ValueError(
                f"invalid pair ({pair.m},{pair.n}): n must be at least m"
            )
        if pair.n > max_attn_bs:
            raise ValueError(
                f"n={pair.n} exceeds attention capacity {max_attn_bs}"
            )
        if pair.n - pair.m > max_num_recv_seqs:
            raise ValueError(
                f"invalid pair ({pair.m},{pair.n}): n-m exceeds remote-row "
                f"capacity {max_num_recv_seqs}"
            )
    return pairs


def _shape_hint(pair: MnPair, block_table_width: int) -> dict[str, int]:
    # _allocate_metadata_case uses graph_* as both actual and Graph sizes for
    # no_padding. Keeping both equal is the defining invariant of this sweep.
    return {
        "actual_master_bs": pair.m,
        "graph_master_bs": pair.m,
        "actual_attn_bs": pair.n,
        "graph_attn_bs": pair.n,
        "block_table_rows": pair.n,
        "block_table_width": block_table_width,
    }


def _source_for_pair(
    source: dict[str, Any], pair: MnPair, capacity_mode: str
) -> dict[str, Any]:
    if capacity_mode == "fixed":
        return source
    if capacity_mode != "per-pair":
        raise ValueError(f"Unsupported capacity mode: {capacity_mode}")

    pair_source = dict(source)
    pair_source["max_num_seqs"] = pair.m
    pair_source["max_num_recv_seqs"] = pair.n - pair.m
    return pair_source


def _capacities(source: dict[str, Any]) -> dict[str, int]:
    max_num_seqs = int(source["max_num_seqs"])
    max_num_recv_seqs = int(source["max_num_recv_seqs"])
    max_model_len = int(source["max_model_len"])
    block_size = int(source["kvcache_block_size"])
    max_master_bs = min(max_num_seqs, 512)
    return {
        "attention_sp": int(source["attention_sp"]),
        "max_num_seqs": max_num_seqs,
        "max_num_recv_seqs": max_num_recv_seqs,
        "max_master_bs": max_master_bs,
        "max_attn_bs": max_master_bs + max_num_recv_seqs,
        "max_num_blocks": (max_model_len + block_size - 1) // block_size,
        "max_model_len": max_model_len,
        "kvcache_block_size": block_size,
    }


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _destination_manifest(
    graph_vars: dict[str, torch.Tensor | None],
) -> dict[str, dict[str, Any] | None]:
    return {
        name: None if tensor is None else _tensor_description(tensor)
        for name, tensor in graph_vars.items()
    }


def _source_manifest(
    context: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    tensors = {
        "input_ids": input_ids,
        "positions": positions,
        "slot_mapping": context.slot_mapping,
        "context_lens": context.context_lens,
        "global_context_lens": context.global_context_lens,
        "q_mask": context.q_mask,
        "q_dst_row_indices": context.q_dst_row_indices,
        "res_lse_mask": context.res_lse_mask,
        "block_tables": context.block_tables,
        "context_lens_for_attn": context.context_lens_for_attn,
        "q_slice_get": context.q_slice_get,
        "q_slice_fill": context.q_slice_fill,
        "q_copy_mask": context.q_copy_mask,
        "res_slice_get_to_buffer_output": (
            context.res_slice_get_to_buffer_output
        ),
        "res_slice_fill_to_buffer_output": (
            context.res_slice_fill_to_buffer_output
        ),
        "res_to_buffer_output_mask": context.res_to_buffer_output_mask,
        "res_slice_get_to_buffer_input": context.res_slice_get_to_buffer_input,
        "res_slice_fill_to_buffer_input": (
            context.res_slice_fill_to_buffer_input
        ),
        "res_to_buffer_input_mask": context.res_to_buffer_input_mask,
        "q_offsets": context.q_offsets,
    }
    return {name: _tensor_description(tensor) for name, tensor in tensors.items()}


def _clone_graph_vars(
    graph_vars: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor | None]:
    return {
        name: None if tensor is None else tensor.clone()
        for name, tensor in graph_vars.items()
    }


def _verify_fused_result(
    reference: dict[str, torch.Tensor | None],
    fused: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    tensors = {}
    all_passed = True
    for name, reference_tensor in reference.items():
        fused_tensor = fused[name]
        if reference_tensor is None or fused_tensor is None:
            passed = reference_tensor is None and fused_tensor is None
            mismatch_count = 0 if passed else 1
        else:
            mismatch_count = int(
                torch.count_nonzero(reference_tensor != fused_tensor).item()
            )
            passed = mismatch_count == 0
        tensors[name] = {
            "passed": passed,
            "mismatch_count": mismatch_count,
        }
        all_passed = all_passed and passed
    return {"passed": all_passed, "tensors": tensors}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("full metadata injection sweep requires CUDA")

    source = _load_json(args.shape_json)
    pairs = args.mn_pairs
    if pairs is None:
        pairs = _captured_mn_pairs(source)
    pairs = _validate_mn_pairs(pairs, source)
    block_table_width = _block_table_width(source, args.block_table_width)

    results = []
    persistent_destinations = None
    for pair in pairs:
        case_source = _source_for_pair(source, pair, args.capacity_mode)
        graph_vars, context, input_ids, positions, shape = _allocate_metadata_case(
            case_source,
            _shape_hint(pair, block_table_width),
            "no_padding",
        )
        case_destinations = _destination_manifest(graph_vars)
        if args.capacity_mode == "fixed":
            if persistent_destinations is None:
                persistent_destinations = case_destinations
            elif case_destinations != persistent_destinations:
                raise RuntimeError(
                    "persistent destination shapes changed across fixed-capacity cases"
                )
        source_tensors = _source_manifest(context, input_ids, positions)

        def inject_production_metadata(
            destination_vars: dict[str, torch.Tensor | None] = graph_vars,
        ) -> None:
            copy_decode_context_to_graph_vars(
                destination_vars,
                input_ids,
                positions,
                shape["actual_master_bs"],
                shape["graph_master_bs"],
                shape["graph_attn_bs"],
                context,
                sp_rank=shape["sp_rank"],
                max_num_seqs=shape["max_num_seqs"],
            )

        correctness = None
        if args.implementation == "fused":
            reference_graph_vars = _clone_graph_vars(graph_vars)
            inject_production_metadata(reference_graph_vars)
            copy_decode_context_to_graph_vars_fused_no_padding(
                graph_vars,
                input_ids,
                positions,
                shape["actual_master_bs"],
                shape["graph_master_bs"],
                shape["graph_attn_bs"],
                context,
            )
            torch.cuda.synchronize()
            correctness = _verify_fused_result(reference_graph_vars, graph_vars)
            del reference_graph_vars
            if not correctness["passed"]:
                failed = [
                    name
                    for name, result in correctness["tensors"].items()
                    if not result["passed"]
                ]
                raise RuntimeError(
                    f"fused metadata mismatch for ({pair.m},{pair.n}): {failed}"
                )

            def inject_all_metadata() -> None:
                copy_decode_context_to_graph_vars_fused_no_padding(
                    graph_vars,
                    input_ids,
                    positions,
                    shape["actual_master_bs"],
                    shape["graph_master_bs"],
                    shape["graph_attn_bs"],
                    context,
                )

        else:
            inject_all_metadata = inject_production_metadata

        timing = _time_cuda_operation(
            inject_all_metadata,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        results.append(
            {
                "m": pair.m,
                "n": pair.n,
                "shape": shape,
                "implementation": args.implementation,
                "correctness_against_production": correctness,
                "capacities": _capacities(case_source),
                "destination_tensors": case_destinations,
                "source_tensors": source_tensors,
                "all_metadata_bytes_touched_estimate": _all_metadata_bytes(
                    graph_vars,
                    context,
                    input_ids,
                    positions,
                    shape,
                    int(source["attention_sp"]),
                ),
                "all_graph_metadata": timing,
            }
        )
        del graph_vars, context, input_ids, positions
        torch.cuda.empty_cache()

    manifest = _manifest("metadata")
    manifest["component"] = "full-metadata-injection-sweep"
    return {
        "schema_version": 1,
        "manifest": manifest,
        "source_shape_json": str(args.shape_json.resolve()),
        "definition": {
            "m": "actual_master_bs == graph_master_bs",
            "n": "actual_attn_bs == graph_attn_bs",
            "padding": False,
            "capacity_mode": args.capacity_mode,
            "implementation": args.implementation,
            "scope": "copy_decode_context_to_graph_vars (all Graph metadata)",
        },
        "capacity_mode": args.capacity_mode,
        "implementation": args.implementation,
        "capacities": _capacities(source),
        "persistent_destination_tensors": persistent_destinations,
        "block_table_width": block_table_width,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cases": results,
    }


def main() -> None:
    args = _parse_args()
    result = _run(args)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

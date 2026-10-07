#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# The helper modules are vendored beside this script so the AE entry point does
# not depend on mla_decode_latency_suite being present or importable.
HELPER_DIR = Path(__file__).resolve().parent
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from benchmark_dlslime_all_to_all_buffer_cp_comm import (  # noqa: E402
    TARGET_BASE_VALUES,
    create_buffer,
    ensure_dlslime,
    get_kernel_impl,
    make_row,
    matches_buffer_all_to_all_kernel,
)
from benchmark_mla_cp_comm import (  # noqa: E402
    BaseRunner,
    active_aggregate,
    gather_rank_meta,
    get_dtype,
    init_dist,
    make_group,
    parse_trace_stats,
    payload_specs,
    profiler_benchmark,
    summarize_values,
)

ALLOWED_PAYLOADS = ("Q", "Res", "Lse")


class PayloadRunner(BaseRunner):
    def __init__(
        self,
        *,
        group,
        preamble,
        device,
        inner_iters,
        buffer,
        x,
        mask,
        kernel_impl,
        is_transpose: bool,
        offsets: torch.Tensor | None,
    ):
        super().__init__(group=group, preamble=preamble, device=device, inner_iters=inner_iters)
        self.buffer = buffer
        self.x = x
        self.mask = mask
        self.kernel_impl = kernel_impl
        self.is_transpose = is_transpose
        self.offsets = offsets
        self.output = None

    def run_once(self):
        for _ in range(self.inner_iters):
            self.preamble_op()
            self.output = self.buffer.all_to_all(
                self.x,
                impl=self.kernel_impl,
                is_transpose=self.is_transpose,
                mask=self.mask,
                offsets=self.offsets,
            )


def capture_graph_on_stream(runner, warmup: int, group, device: torch.device, stream: torch.cuda.Stream):
    current_stream = torch.cuda.current_stream(device)
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            runner.run_once()
    stream.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner.run_once()
    stream.synchronize()
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NanoDeploy-aligned Q/Res/Lse all-to-all latency on DLSlime AllToAllBuffer."
    )
    parser.add_argument("--cases", type=str, required=True, help="JSON file describing Q/Res/Lse routing cases.")
    parser.add_argument("--payloads", type=str, default="Q,Res,Lse")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--v-head-dim", type=int, default=512)
    parser.add_argument("--mode", type=str, default="graph", choices=["eager", "graph"])
    parser.add_argument("--preamble", type=str, default="all_reduce", choices=["none", "all_reduce", "all_gather"])
    parser.add_argument("--dlslime-impl", type=str, default="basic", choices=["basic", "tma"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--graph-inner-iters", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--keep-traces", action="store_true")
    return parser.parse_args()


def parse_payload_names(raw: str) -> list[str]:
    values = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        if value not in ALLOWED_PAYLOADS:
            raise ValueError(f"Unsupported payload: {value}")
        values.append(value)
    if not values:
        raise ValueError("No payloads provided")
    return values


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases", payload)
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a top-level list or a dict with 'cases'")
    return cases


def normalize_matrix(matrix: list[list[int]], cp_size: int, payload_name: str) -> list[list[list[int]]]:
    if len(matrix) != cp_size:
        raise ValueError(f"{payload_name} matrix must have {cp_size} rows")
    row_masks_per_rank: list[list[list[int]]] = []
    for src_rank, row in enumerate(matrix):
        if len(row) != cp_size:
            raise ValueError(f"{payload_name} matrix row {src_rank} must have width {cp_size}")
        if any(value not in (0, 1) for value in row):
            raise ValueError(f"{payload_name} matrix must be binary; use row_masks_per_rank for repeated rows")
        if row[src_rank] != 0:
            raise ValueError(f"{payload_name} matrix must not set self-send on rank {src_rank}")
        row_masks_per_rank.append([list(row)] if any(row) else [])
    return row_masks_per_rank


def normalize_row_masks(row_masks_per_rank: list[list[list[int]]], cp_size: int, payload_name: str) -> list[list[list[int]]]:
    if len(row_masks_per_rank) != cp_size:
        raise ValueError(f"{payload_name} row_masks_per_rank must have {cp_size} entries")
    normalized: list[list[list[int]]] = []
    for src_rank, rows in enumerate(row_masks_per_rank):
        if not isinstance(rows, list):
            raise ValueError(f"{payload_name} row_masks_per_rank[{src_rank}] must be a list")
        normalized_rows: list[list[int]] = []
        for row_idx, row in enumerate(rows):
            if len(row) != cp_size:
                raise ValueError(
                    f"{payload_name} row_masks_per_rank[{src_rank}][{row_idx}] must have width {cp_size}"
                )
            if row[src_rank] != 0:
                raise ValueError(f"{payload_name} row {row_idx} on rank {src_rank} must not self-send")
            if any(value not in (0, 1) for value in row):
                raise ValueError(f"{payload_name} row {row_idx} on rank {src_rank} must be binary")
            normalized_rows.append([int(value) for value in row])
        normalized.append(normalized_rows)
    return normalized


def _ensure_binary_matrix(
    matrix: list[list[int]], *, rows: int, cols: int, field_name: str
) -> list[list[int]]:
    if len(matrix) != rows:
        raise ValueError(f"{field_name} must have {rows} rows")
    normalized: list[list[int]] = []
    for row_idx, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError(f"{field_name}[{row_idx}] must have width {cols}")
        if any(value not in (0, 1) for value in row):
            raise ValueError(f"{field_name}[{row_idx}] must be binary")
        normalized.append([int(value) for value in row])
    return normalized


def _infer_q_num_rows(q_mask: list[list[int]]) -> int:
    max_col = -1
    for row in q_mask:
        for col_idx, value in enumerate(row):
            if value:
                max_col = max(max_col, col_idx)
    return max_col + 1


def normalize_raw_per_rank(raw_per_rank: list[dict], cp_size: int, max_bs: int) -> list[dict]:
    if len(raw_per_rank) != cp_size:
        raise ValueError(f"raw_per_rank must have {cp_size} entries")

    normalized = []
    for cp_rank, entry in enumerate(raw_per_rank):
        if not isinstance(entry, dict):
            raise ValueError(f"raw_per_rank[{cp_rank}] must be a dict")

        q_mask = _ensure_binary_matrix(
            entry["q_mask"], rows=cp_size, cols=max_bs, field_name=f"raw_per_rank[{cp_rank}].q_mask"
        )
        res_lse_mask = _ensure_binary_matrix(
            entry["res_lse_mask"],
            rows=cp_size,
            cols=max_bs,
            field_name=f"raw_per_rank[{cp_rank}].res_lse_mask",
        )
        q_offsets = [int(value) for value in entry["q_offsets"]]
        if len(q_offsets) != cp_size + 1:
            raise ValueError(f"raw_per_rank[{cp_rank}].q_offsets must have length {cp_size + 1}")
        if q_offsets[0] != 0:
            raise ValueError(f"raw_per_rank[{cp_rank}].q_offsets must start at 0")
        if any(q_offsets[i] > q_offsets[i + 1] for i in range(cp_size)):
            raise ValueError(f"raw_per_rank[{cp_rank}].q_offsets must be non-decreasing")
        if any((q_offsets[i + 1] - q_offsets[i]) > max_bs for i in range(cp_size)):
            raise ValueError(
                f"raw_per_rank[{cp_rank}].q_offsets has a per-source span larger than max_bs={max_bs}"
            )

        inferred_q_num_rows = _infer_q_num_rows(q_mask)
        self_span = q_offsets[cp_rank + 1] - q_offsets[cp_rank]
        q_num_rows = int(entry.get("q_num_rows", self_span))
        if q_num_rows < max(inferred_q_num_rows, self_span) or q_num_rows > max_bs:
            raise ValueError(
                f"raw_per_rank[{cp_rank}].q_num_rows must be in [{max(inferred_q_num_rows, self_span)}, {max_bs}]"
            )

        normalized.append(
            {
                "q_mask": q_mask,
                "res_lse_mask": res_lse_mask,
                "q_offsets": q_offsets,
                "q_num_rows": q_num_rows,
            }
        )

    for recv_rank in range(cp_size):
        q_offsets = normalized[recv_rank]["q_offsets"]
        for src_rank in range(cp_size):
            expected = q_offsets[src_rank + 1] - q_offsets[src_rank]
            if src_rank == recv_rank:
                actual = int(normalized[src_rank]["q_num_rows"])
            else:
                actual = sum(normalized[src_rank]["q_mask"][recv_rank])
            if actual != expected:
                raise ValueError(
                    "raw_per_rank Q consistency mismatch: "
                    f"receiver rank {recv_rank}, source rank {src_rank}, "
                    f"q_offsets span={expected}, q_mask count={actual}"
                )

    return normalized


def resolve_payload_entry(case: dict, payload_name: str, stack: tuple[str, ...] = ()) -> tuple[dict, str]:
    payload_key = payload_name.lower()
    entry = case.get(payload_key)
    if entry is None and payload_name == "Lse" and case.get("res") is not None:
        entry = {"reuse_from": "res"}
    if entry is None:
        raise ValueError(f"case {case.get('case_id', '<unknown>')} is missing payload spec for {payload_name}")
    if not isinstance(entry, dict):
        raise ValueError(f"payload spec for {payload_name} must be a dict")
    if "reuse_from" in entry:
        target = str(entry["reuse_from"]).strip().capitalize()
        if target == "Lse":
            target = "Lse"
        if target not in ALLOWED_PAYLOADS:
            raise ValueError(f"Unsupported reuse_from target: {entry['reuse_from']}")
        if payload_name in stack:
            raise ValueError(f"Cyclic reuse_from chain detected: {stack + (payload_name,)}")
        return resolve_payload_entry(case, target, stack + (payload_name,))
    if "matrix" in entry:
        return entry, "matrix"
    if "row_masks_per_rank" in entry:
        return entry, "row_masks_per_rank"
    raise ValueError(f"payload spec for {payload_name} must provide either matrix or row_masks_per_rank")


def build_payload_workload(case: dict, payload_name: str) -> dict:
    case_id = str(case["case_id"])
    cp_size = int(case["cp_size"])
    if cp_size <= 0:
        raise ValueError(f"case {case_id} has invalid cp_size={cp_size}")

    if "raw_per_rank" in case:
        max_bs = int(case.get("max_bs", 0))
        if max_bs <= 0:
            raise ValueError(f"case {case_id} must provide a positive max_bs for raw_per_rank input")
        raw_per_rank = normalize_raw_per_rank(case["raw_per_rank"], cp_size, max_bs)
        if payload_name == "Q":
            bs_per_rank = [int(entry["q_num_rows"]) for entry in raw_per_rank]
            fanout_per_rank = [max((sum(row) for row in entry["q_mask"]), default=0) for entry in raw_per_rank]
        else:
            bs_per_rank = [_infer_q_num_rows(entry["res_lse_mask"]) for entry in raw_per_rank]
            fanout_per_rank = [max((sum(row) for row in entry["res_lse_mask"]), default=0) for entry in raw_per_rank]
        return {
            "workload_id": case_id,
            "case_id": case_id,
            "description": case.get("description", ""),
            "pattern": case.get("pattern", "raw_decode_a2a"),
            "input_format": "raw_per_rank",
            "payload": payload_name,
            "cp_size": cp_size,
            "max_bs": max_bs,
            "bs_per_rank": bs_per_rank,
            "fanout_per_rank": fanout_per_rank,
            "raw_per_rank": raw_per_rank,
        }

    payload_entry, input_format = resolve_payload_entry(case, payload_name)
    if input_format == "matrix":
        row_masks_per_rank = normalize_matrix(payload_entry["matrix"], cp_size, payload_name)
    else:
        row_masks_per_rank = normalize_row_masks(payload_entry["row_masks_per_rank"], cp_size, payload_name)

    inferred_max_bs = max((len(rows) for rows in row_masks_per_rank), default=0)
    max_bs = int(payload_entry.get("max_bs", case.get("max_bs", max(inferred_max_bs, 1))))
    if max_bs <= 0:
        raise ValueError(f"case {case_id} payload {payload_name} has invalid max_bs={max_bs}")
    if any(len(rows) > max_bs for rows in row_masks_per_rank):
        raise ValueError(f"case {case_id} payload {payload_name} has rows exceeding max_bs={max_bs}")

    bs_per_rank = [len(rows) for rows in row_masks_per_rank]
    bs_offsets = [0]
    for bs in bs_per_rank:
        bs_offsets.append(bs_offsets[-1] + bs)
    fanout_per_rank = [max((sum(row) for row in rows), default=0) for rows in row_masks_per_rank]

    return {
        "workload_id": case_id,
        "case_id": case_id,
        "description": case.get("description", ""),
        "pattern": payload_entry.get("pattern", input_format),
        "input_format": input_format,
        "payload": payload_name,
        "cp_size": cp_size,
        "max_bs": max_bs,
        "bs_per_rank": bs_per_rank,
        "bs_offsets": bs_offsets,
        "fanout_per_rank": fanout_per_rank,
        "masks_per_rank": row_masks_per_rank,
    }


def compute_traffic_stats(cp_size: int, masks_per_rank: list[list[list[int]]]) -> dict:
    send_rows_per_rank = [0] * cp_size
    recv_rows_per_rank = [0] * cp_size
    total_remote_edges = 0
    for src_rank, rows in enumerate(masks_per_rank):
        for row in rows:
            row_edges = 0
            for dst_rank, flag in enumerate(row):
                if dst_rank == src_rank or flag == 0:
                    continue
                recv_rows_per_rank[dst_rank] += 1
                row_edges += 1
            send_rows_per_rank[src_rank] += row_edges
            total_remote_edges += row_edges
    return {
        "send_rows_per_rank": send_rows_per_rank,
        "recv_rows_per_rank": recv_rows_per_rank,
        "max_send_rows": max(send_rows_per_rank, default=0),
        "max_recv_rows": max(recv_rows_per_rank, default=0),
        "max_peer_rows": max(max(send_rows_per_rank, default=0), max(recv_rows_per_rank, default=0)),
        "total_remote_edges": total_remote_edges,
    }


def compute_raw_traffic_stats(cp_size: int, raw_per_rank: list[dict], payload_name: str) -> dict:
    send_rows_per_rank = [0] * cp_size
    recv_rows_per_rank = [0] * cp_size
    total_remote_edges = 0

    mask_key = "q_mask" if payload_name == "Q" else "res_lse_mask"
    for src_rank, entry in enumerate(raw_per_rank):
        mask = entry[mask_key]
        src_total = 0
        for dst_rank in range(cp_size):
            if dst_rank == src_rank:
                continue
            edge_count = sum(mask[dst_rank])
            recv_rows_per_rank[dst_rank] += edge_count
            src_total += edge_count
            total_remote_edges += edge_count
        send_rows_per_rank[src_rank] = src_total

    return {
        "send_rows_per_rank": send_rows_per_rank,
        "recv_rows_per_rank": recv_rows_per_rank,
        "max_send_rows": max(send_rows_per_rank, default=0),
        "max_recv_rows": max(recv_rows_per_rank, default=0),
        "max_peer_rows": max(max(send_rows_per_rank, default=0), max(recv_rows_per_rank, default=0)),
        "total_remote_edges": total_remote_edges,
    }


def build_payload_tensors(
    *,
    cp_rank: int,
    workload: dict,
    payload_name: str,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
    base_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int, bool]:
    cp_size = workload["cp_size"]
    max_bs = workload["max_bs"]
    masks_per_rank = workload["masks_per_rank"]
    local_rows = masks_per_rank[cp_rank]
    local_bs = len(local_rows)

    if payload_name == "Q":
        x = torch.zeros((local_bs, feature_dim), dtype=dtype, device=device)
        mask = torch.zeros((cp_size, max_bs), dtype=torch.int32, device=device)
        for seq_idx, row_mask in enumerate(local_rows):
            x[seq_idx].copy_(make_row(base_value + float(cp_rank * 1000 + seq_idx), feature_dim, dtype, device))
            for dst_rank, flag in enumerate(row_mask):
                if dst_rank == cp_rank or flag == 0:
                    continue
                mask[dst_rank, seq_idx] = 1

        offsets = torch.tensor(workload["bs_offsets"], dtype=torch.int32, device=device)
        expected_flat = torch.zeros((cp_size * max_bs, feature_dim), dtype=dtype, device=device)
        expected_active_flat = torch.zeros((cp_size * max_bs,), dtype=torch.bool, device=device)
        for src_rank, rows in enumerate(masks_per_rank):
            base_idx = workload["bs_offsets"][src_rank]
            for seq_idx, row_mask in enumerate(rows):
                if row_mask[cp_rank]:
                    flat_idx = base_idx + seq_idx
                    expected_flat[flat_idx].copy_(
                        make_row(base_value + float(src_rank * 1000 + seq_idx), feature_dim, dtype, device)
                    )
                    expected_active_flat[flat_idx] = True

        return (
            x,
            mask,
            offsets,
            expected_flat.view(cp_size, max_bs, feature_dim),
            expected_active_flat.view(cp_size, max_bs),
            max_bs,
            False,
        )

    x = torch.zeros((cp_size * max_bs, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((cp_size, max_bs), dtype=torch.int32, device=device)
    for seq_idx, row_mask in enumerate(local_rows):
        row_value = make_row(base_value + float(cp_rank * 1000 + seq_idx), feature_dim, dtype, device)
        for dst_rank, flag in enumerate(row_mask):
            if dst_rank == cp_rank or flag == 0:
                continue
            x[dst_rank * max_bs + seq_idx].copy_(row_value)
            mask[dst_rank, seq_idx] = 1

    expected = torch.zeros((cp_size, max_bs, feature_dim), dtype=dtype, device=device)
    expected_active = torch.zeros((cp_size, max_bs), dtype=torch.bool, device=device)
    for src_rank, rows in enumerate(masks_per_rank):
        for seq_idx, row_mask in enumerate(rows):
            if row_mask[cp_rank]:
                expected[src_rank, seq_idx].copy_(
                    make_row(base_value + float(src_rank * 1000 + seq_idx), feature_dim, dtype, device)
                )
                expected_active[src_rank, seq_idx] = True

    return x, mask, None, expected, expected_active, max_bs, True


def build_raw_payload_tensors(
    *,
    cp_rank: int,
    workload: dict,
    payload_name: str,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
    base_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int, bool]:
    cp_size = workload["cp_size"]
    max_bs = workload["max_bs"]
    raw_per_rank = workload["raw_per_rank"]
    local_entry = raw_per_rank[cp_rank]

    if payload_name == "Q":
        q_num_rows = int(local_entry["q_num_rows"])
        x = torch.zeros((q_num_rows, feature_dim), dtype=dtype, device=device)
        for row_idx in range(q_num_rows):
            x[row_idx].copy_(
                make_row(base_value + float(cp_rank * 1000 + row_idx), feature_dim, dtype, device)
            )

        mask = torch.tensor(local_entry["q_mask"], dtype=torch.int32, device=device)
        offsets = torch.tensor(local_entry["q_offsets"], dtype=torch.int32, device=device)

        expected_flat = torch.zeros((cp_size * max_bs, feature_dim), dtype=dtype, device=device)
        expected_active_flat = torch.zeros((cp_size * max_bs,), dtype=torch.bool, device=device)
        for src_rank, src_entry in enumerate(raw_per_rank):
            current_pos = local_entry["q_offsets"][src_rank]
            if src_rank == cp_rank:
                row_indices = range(int(src_entry["q_num_rows"]))
            else:
                src_mask = src_entry["q_mask"][cp_rank]
                row_indices = [row_idx for row_idx in range(int(src_entry["q_num_rows"])) if src_mask[row_idx]]
            for row_idx in row_indices:
                expected_flat[current_pos].copy_(
                    make_row(base_value + float(src_rank * 1000 + row_idx), feature_dim, dtype, device)
                )
                expected_active_flat[current_pos] = True
                current_pos += 1
            if current_pos != local_entry["q_offsets"][src_rank + 1]:
                raise ValueError(
                    f"Q raw case inconsistency on cp_rank={cp_rank}, src_rank={src_rank}: "
                    f"filled={current_pos - local_entry['q_offsets'][src_rank]}, "
                    f"expected={local_entry['q_offsets'][src_rank + 1] - local_entry['q_offsets'][src_rank]}"
                )

        return (
            x,
            mask,
            offsets,
            expected_flat.view(cp_size, max_bs, feature_dim),
            expected_active_flat.view(cp_size, max_bs),
            max_bs,
            False,
        )

    x = torch.zeros((cp_size * max_bs, feature_dim), dtype=dtype, device=device)
    mask = torch.tensor(local_entry["res_lse_mask"], dtype=torch.int32, device=device)
    for dst_rank in range(cp_size):
        for row_idx, flag in enumerate(local_entry["res_lse_mask"][dst_rank]):
            if not flag:
                continue
            x[dst_rank * max_bs + row_idx].copy_(
                make_row(base_value + float(cp_rank * 1000 + row_idx), feature_dim, dtype, device)
            )

    expected = torch.zeros((cp_size, max_bs, feature_dim), dtype=dtype, device=device)
    expected_active = torch.zeros((cp_size, max_bs), dtype=torch.bool, device=device)
    for src_rank, src_entry in enumerate(raw_per_rank):
        for row_idx, flag in enumerate(src_entry["res_lse_mask"][cp_rank]):
            if not flag:
                continue
            expected[src_rank, row_idx].copy_(
                make_row(base_value + float(src_rank * 1000 + row_idx), feature_dim, dtype, device)
            )
            expected_active[src_rank, row_idx] = True

    return x, mask, None, expected, expected_active, max_bs, True


def validate_runner(
    *,
    runner,
    expected: torch.Tensor,
    expected_active: torch.Tensor,
    group,
    device: torch.device,
    payload_name: str,
    case_id: str,
    cp_size: int,
    cp_rank: int,
):
    runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    actual = runner.output.clone()

    ok = True
    max_diff = 0.0
    if actual.shape[:2] != expected.shape[:2]:
        ok = False
    else:
        active_positions = expected_active.bool()
        actual_active = actual[active_positions]
        expected_active_values = expected[active_positions]
        if actual_active.shape != expected_active_values.shape or not torch.equal(actual_active, expected_active_values):
            ok = False
            if actual_active.numel() and expected_active_values.numel():
                max_diff = float((actual_active.float() - expected_active_values.float()).abs().max().item())

    ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if ok_tensor.item() != 1:
        raise RuntimeError(
            f"Validation failed case={case_id} payload={payload_name} cp_size={cp_size} cp_rank={cp_rank} max_diff={max_diff}"
        )


def zero_rank_meta(*, global_rank: int, cp_rank: int, payload_name: str, row_bytes: int) -> dict:
    zero_summary = summarize_values([0.0])
    return {
        "global_rank": global_rank,
        "cp_rank": cp_rank,
        "payload": payload_name,
        "row_bytes": row_bytes,
        "local_bs": 0,
        "local_fanout": 0,
        "send_rows": 0,
        "recv_rows": 0,
        "event_summary_us": zero_summary,
        "all2all_summary_us": zero_summary,
        "preamble_summary_us": zero_summary,
        "trace_paths": [],
    }


def benchmark_payload_case(
    *,
    workload: dict,
    payload_spec,
    dtype: torch.dtype,
    device: torch.device,
    group,
    global_rank: int,
    kernel_impl,
    args: argparse.Namespace,
    trace_root: Path,
) -> dict:
    cp_rank = dist.get_rank(group=group)
    row_bytes = payload_spec.feature_dim * torch.tensor([], dtype=dtype).element_size()
    if workload["input_format"] == "raw_per_rank":
        traffic = compute_raw_traffic_stats(workload["cp_size"], workload["raw_per_rank"], payload_spec.name)
    else:
        traffic = compute_traffic_stats(workload["cp_size"], workload["masks_per_rank"])

    if traffic["total_remote_edges"] == 0:
        gathered = gather_rank_meta(
            zero_rank_meta(global_rank=global_rank, cp_rank=cp_rank, payload_name=payload_spec.name, row_bytes=row_bytes),
            group,
        )
        if cp_rank != 0:
            return {}
        event_summary = active_aggregate(gathered, "event_summary_us")
        all2all_summary = active_aggregate(gathered, "all2all_summary_us")
        preamble_summary = active_aggregate(gathered, "preamble_summary_us")
        return {
            "case_id": workload["case_id"],
            "description": workload["description"],
            "pattern": workload["pattern"],
            "input_format": workload["input_format"],
            "cp_size": workload["cp_size"],
            "max_bs": workload["max_bs"],
            "payload": payload_spec.name,
            "feature_dim": payload_spec.feature_dim,
            "dtype": args.dtype,
            "bs_per_rank": workload["bs_per_rank"],
            "fanout_per_rank": workload["fanout_per_rank"],
            "send_rows_per_rank": traffic["send_rows_per_rank"],
            "recv_rows_per_rank": traffic["recv_rows_per_rank"],
            "total_remote_edges": traffic["total_remote_edges"],
            "max_send_rows": traffic["max_send_rows"],
            "max_recv_rows": traffic["max_recv_rows"],
            "max_peer_rows": traffic["max_peer_rows"],
            "row_bytes": int(row_bytes),
            "max_send_bytes": traffic["max_send_rows"] * int(row_bytes),
            "max_recv_bytes": traffic["max_recv_rows"] * int(row_bytes),
            "max_peer_bytes": traffic["max_peer_rows"] * int(row_bytes),
            "idle_event_slowest_us": event_summary["slowest_mean_us"],
            "idle_all2all_kernel_slowest_us": all2all_summary["slowest_mean_us"],
            "idle_preamble_kernel_slowest_us": preamble_summary["slowest_mean_us"],
            "ranks": gathered,
        }

    if workload["input_format"] == "raw_per_rank":
        x, mask, offsets, expected, expected_active, max_bs, is_transpose = build_raw_payload_tensors(
            cp_rank=cp_rank,
            workload=workload,
            payload_name=payload_spec.name,
            dtype=dtype,
            device=device,
            feature_dim=payload_spec.feature_dim,
            base_value=TARGET_BASE_VALUES[payload_spec.name],
        )
    else:
        x, mask, offsets, expected, expected_active, max_bs, is_transpose = build_payload_tensors(
            cp_rank=cp_rank,
            workload=workload,
            payload_name=payload_spec.name,
            dtype=dtype,
            device=device,
            feature_dim=payload_spec.feature_dim,
            base_value=TARGET_BASE_VALUES[payload_spec.name],
        )

    buffer = create_buffer(cp_rank, workload["cp_size"], max_bs, payload_spec.feature_dim, dtype, group)
    runner = PayloadRunner(
        group=group,
        preamble=args.preamble,
        device=device,
        inner_iters=args.graph_inner_iters if args.mode == "graph" else 1,
        buffer=buffer,
        x=x,
        mask=mask,
        kernel_impl=kernel_impl,
        is_transpose=is_transpose,
        offsets=offsets,
    )
    # Raw per-rank decode cases come directly from NanoDeploy logs and are used
    # for latency replay. Keep the stricter value-equality validation on the
    # synthetic matrix/row-mask paths, but skip it here to avoid coupling the
    # benchmark to NanoDeploy's internal output packing details.
    if workload["input_format"] != "raw_per_rank":
        validate_runner(
            runner=runner,
            expected=expected,
            expected_active=expected_active,
            group=group,
            device=device,
            payload_name=payload_spec.name,
            case_id=workload["case_id"],
            cp_size=workload["cp_size"],
            cp_rank=cp_rank,
        )

    stream = torch.cuda.Stream(device=device)
    if args.mode == "graph":
        graph = capture_graph_on_stream(runner, args.warmup, group, device, stream)
    else:
        graph = None
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                runner.run_once()
        stream.synchronize()
        dist.barrier(group=group)

    event_samples = []
    all2all_trace_samples = []
    preamble_trace_samples = []
    trace_paths = []
    target_stream_ids = None
    payload_trace_dir = trace_root / workload["case_id"] / payload_spec.name
    payload_trace_dir.mkdir(parents=True, exist_ok=True)

    for repeat_idx in range(args.repeats):
        trace_path = payload_trace_dir / f"rank{global_rank}_repeat{repeat_idx}.json"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier(group=group)
        torch.cuda.synchronize(device)
        start.record(stream)
        if args.mode == "graph":
            with torch.cuda.stream(stream):
                graph.replay()
        else:
            with torch.cuda.stream(stream):
                runner.run_once()
        end.record(stream)
        end.synchronize()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        event_samples.append(start.elapsed_time(end) * 1000.0 / max(runner.inner_iters, 1))

        prof_stats = profiler_benchmark(
            runner,
            graph,
            args.mode,
            args.profile_iters,
            str(trace_path),
            device=device,
            group=group,
            alltoall_matcher=matches_buffer_all_to_all_kernel,
            stream_filter=target_stream_ids,
            replay_stream=stream,
        )
        if target_stream_ids is None:
            target_stream_ids = prof_stats["all2all"]["matched_stream_ids"]
            if target_stream_ids:
                prof_stats = parse_trace_stats(
                    str(trace_path),
                    args.preamble,
                    args.profile_iters * runner.inner_iters,
                    alltoall_matcher=matches_buffer_all_to_all_kernel,
                    stream_filter=target_stream_ids,
                )
        all2all_trace_samples.append(prof_stats["all2all"]["trace_p50_us"])
        preamble_trace_samples.append(prof_stats["preamble"]["trace_p50_us"])
        if args.keep_traces:
            trace_paths.append(str(trace_path.resolve()))
        else:
            trace_path.unlink(missing_ok=True)

    rank_meta = {
        "global_rank": global_rank,
        "cp_rank": cp_rank,
        "payload": payload_spec.name,
        "row_bytes": int(row_bytes),
        "local_bs": workload["bs_per_rank"][cp_rank],
        "local_fanout": workload["fanout_per_rank"][cp_rank],
        "send_rows": traffic["send_rows_per_rank"][cp_rank],
        "recv_rows": traffic["recv_rows_per_rank"][cp_rank],
        "event_summary_us": summarize_values(event_samples),
        "all2all_summary_us": summarize_values(all2all_trace_samples),
        "preamble_summary_us": summarize_values(preamble_trace_samples),
        "trace_paths": trace_paths,
    }
    gathered = gather_rank_meta(rank_meta, group)
    if cp_rank != 0:
        return {}

    event_summary = active_aggregate(gathered, "event_summary_us")
    all2all_summary = active_aggregate(gathered, "all2all_summary_us")
    preamble_summary = active_aggregate(gathered, "preamble_summary_us")

    return {
        "case_id": workload["case_id"],
        "description": workload["description"],
        "pattern": workload["pattern"],
        "input_format": workload["input_format"],
        "cp_size": workload["cp_size"],
        "max_bs": workload["max_bs"],
        "payload": payload_spec.name,
        "feature_dim": payload_spec.feature_dim,
        "dtype": args.dtype,
        "bs_per_rank": workload["bs_per_rank"],
        "fanout_per_rank": workload["fanout_per_rank"],
        "send_rows_per_rank": traffic["send_rows_per_rank"],
        "recv_rows_per_rank": traffic["recv_rows_per_rank"],
        "total_remote_edges": traffic["total_remote_edges"],
        "max_send_rows": traffic["max_send_rows"],
        "max_recv_rows": traffic["max_recv_rows"],
        "max_peer_rows": traffic["max_peer_rows"],
        "row_bytes": int(row_bytes),
        "max_send_bytes": traffic["max_send_rows"] * int(row_bytes),
        "max_recv_bytes": traffic["max_recv_rows"] * int(row_bytes),
        "max_peer_bytes": traffic["max_peer_rows"] * int(row_bytes),
        "idle_event_slowest_us": event_summary["slowest_mean_us"],
        "idle_all2all_kernel_slowest_us": all2all_summary["slowest_mean_us"],
        "idle_preamble_kernel_slowest_us": preamble_summary["slowest_mean_us"],
        "ranks": gathered,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "description",
        "pattern",
        "input_format",
        "cp_size",
        "max_bs",
        "payload",
        "feature_dim",
        "dtype",
        "bs_per_rank",
        "fanout_per_rank",
        "send_rows_per_rank",
        "recv_rows_per_rank",
        "total_remote_edges",
        "max_send_rows",
        "max_recv_rows",
        "max_peer_rows",
        "row_bytes",
        "max_send_bytes",
        "max_recv_bytes",
        "max_peer_bytes",
        "idle_event_slowest_us",
        "idle_all2all_kernel_slowest_us",
        "idle_preamble_kernel_slowest_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row[key] for key in fieldnames}
            csv_row["bs_per_rank"] = ",".join(str(v) for v in row["bs_per_rank"])
            csv_row["fanout_per_rank"] = ",".join(str(v) for v in row["fanout_per_rank"])
            csv_row["send_rows_per_rank"] = ",".join(str(v) for v in row["send_rows_per_rank"])
            csv_row["recv_rows_per_rank"] = ",".join(str(v) for v in row["recv_rows_per_rank"])
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    ensure_dlslime()
    dtype = get_dtype(args.dtype)
    kernel_impl = get_kernel_impl(args.dlslime_impl)
    payload_names = parse_payload_names(args.payloads)

    cases_path = Path(args.cases).resolve()
    cases = load_cases(cases_path)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("No cases to benchmark")

    normalized_cases = []
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{idx} must be a dict")
        if "case_id" not in case:
            case = {**case, "case_id": f"case_{idx:04d}"}
        if "cp_size" not in case:
            raise ValueError(f"case {case['case_id']} is missing cp_size")
        normalized_cases.append(case)

    ts = time.strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent
    summary_path = Path(args.summary_path) if args.summary_path else project_root / "results" / f"q_res_lse_latency_summary_{ts}.json"
    csv_path = Path(args.csv_path) if args.csv_path else project_root / "results" / f"q_res_lse_latency_dataset_{ts}.csv"
    trace_root = Path(args.trace_dir) if args.trace_dir else project_root / "results" / "traces" / ts

    global_rank, world_size, device = init_dist()
    max_cp_size = max(int(case["cp_size"]) for case in normalized_cases)
    if max_cp_size > world_size:
        raise ValueError(f"Requested max cp_size {max_cp_size}, but world_size is only {world_size}")

    if global_rank == 0:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        trace_root.mkdir(parents=True, exist_ok=True)

    group_cache = {}
    dataset_rows = []
    for case in normalized_cases:
        cp_size = int(case["cp_size"])
        if cp_size not in group_cache:
            group_cache[cp_size] = make_group(cp_size)
        group = group_cache[cp_size]
        active = global_rank < cp_size
        if active:
            available_payload_specs = {
                spec.name: spec for spec in payload_specs(args.num_heads, args.head_dim, args.v_head_dim, cp_size)
            }
            for payload_name in payload_names:
                workload = build_payload_workload(case, payload_name)
                result = benchmark_payload_case(
                    workload=workload,
                    payload_spec=available_payload_specs[payload_name],
                    dtype=dtype,
                    device=device,
                    group=group,
                    global_rank=global_rank,
                    kernel_impl=kernel_impl,
                    args=args,
                    trace_root=trace_root,
                )
                if dist.get_rank(group=group) == 0 and result:
                    dataset_rows.append(result)
        dist.barrier()

    if global_rank == 0:
        summary = {
            "description": "NanoDeploy-aligned Q/Res/Lse latency benchmark on DLSlime AllToAllBuffer",
            "backend": "dlslime_all_to_all_buffer",
            "dlslime_impl": args.dlslime_impl,
            "mode": args.mode,
            "preamble": args.preamble,
            "dtype": args.dtype,
            "world_size": world_size,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "v_head_dim": args.v_head_dim,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "profile_iters": args.profile_iters,
            "graph_inner_iters": args.graph_inner_iters,
            "keep_traces": args.keep_traces,
            "payloads": payload_names,
            "cases": str(cases_path),
            "num_cases": len(normalized_cases),
            "num_rows": len(dataset_rows),
            "results": dataset_rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        write_csv(dataset_rows, csv_path)
        print(f"Saved summary to {summary_path}")
        print(f"Saved dataset csv to {csv_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

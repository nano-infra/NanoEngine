from __future__ import annotations

import argparse
import csv
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from nanodeploy.worker import distributed as distributed_mod
from nanodeploy.worker.sp_backend import SPBackend
from nanodeploy.worker.sp_context import get_sp_context, set_sp_context


MASTER_RANK = 0
BACKEND_CHOICES: tuple[SPBackend, ...] = (
    "hao_basic",
    "nccl",
    "nccl_compact",
)
PAYLOAD_CHOICES = ("Q", "Res", "Lse")
PATTERN_CHOICES = ("fan_out", "uniform", "fan_in")

DEFAULT_NUM_HEADS = 128
DEFAULT_KV_LORA_RANK = 512
DEFAULT_QK_ROPE_HEAD_DIM = 64
DEFAULT_HEAD_DIM = DEFAULT_KV_LORA_RANK + DEFAULT_QK_ROPE_HEAD_DIM
DEFAULT_V_HEAD_DIM = DEFAULT_KV_LORA_RANK


@dataclass
class _BenchDistContext:
    group: dist.ProcessGroup

    @property
    def attn_sp_group(self) -> dist.ProcessGroup:
        return self.group

    @property
    def attn_sp_rank(self) -> int:
        return dist.get_rank(group=self.group)

    @property
    def attn_sp_world_size(self) -> int:
        return dist.get_world_size(group=self.group)


@dataclass(frozen=True)
class PayloadCase:
    name: str
    buffer_name: str
    feature_dim: int
    is_transpose: bool
    builder: Callable[
        ["BaseTrafficPattern", object, torch.dtype, torch.device, int],
        "BuiltPayload",
    ]


@dataclass
class BuiltPayload:
    x: torch.Tensor
    mask: torch.Tensor
    offsets: torch.Tensor | None
    expected: torch.Tensor
    local_owned_seqs: int
    local_compute_slots: int
    local_send_tokens: int
    local_send_targets: int


@dataclass
class PreambleState:
    preamble: str
    sync_tensor: torch.Tensor | None = None
    gather_out: torch.Tensor | None = None


class BaseRunner:
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        preamble: str,
        device: torch.device,
        inner_iters: int,
    ) -> None:
        self.group = group
        self.preamble = preamble
        self.device = device
        self.inner_iters = inner_iters
        self.preamble_state = create_preamble_state(preamble, group, device)

    def preamble_op(self) -> None:
        run_preamble(self.preamble_state, self.group)


class BufferRunner(BaseRunner):
    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        preamble: str,
        device: torch.device,
        inner_iters: int,
        buffer,
        x: torch.Tensor,
        mask: torch.Tensor,
        offsets: torch.Tensor | None,
        is_transpose: bool,
    ) -> None:
        super().__init__(
            group=group,
            preamble=preamble,
            device=device,
            inner_iters=inner_iters,
        )
        self.buffer = buffer
        self.x = x
        self.mask = mask
        self.offsets = offsets
        self.is_transpose = is_transpose
        self.output: torch.Tensor | None = None

    def run_once(self) -> None:
        for _ in range(self.inner_iters):
            self.preamble_op()
            self.output = self.buffer.all_to_all_ll(
                self.x,
                is_transpose=self.is_transpose,
                mask=self.mask,
                offsets=self.offsets,
            )


class BaseTrafficPattern(ABC):
    def __init__(
        self,
        *,
        sp_size: int,
        batch_size: int,
        max_num_seqs: int,
        rank: int,
        master_rank: int,
    ) -> None:
        if sp_size <= 0:
            raise ValueError(f"sp_size must be positive, got {sp_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_num_seqs < batch_size:
            raise ValueError(
                f"max_num_seqs ({max_num_seqs}) must be >= batch_size ({batch_size})"
            )
        if not 0 <= master_rank < sp_size:
            raise ValueError(
                f"master_rank must be in [0, {sp_size}), got {master_rank}"
            )
        if not 0 <= rank < sp_size:
            raise ValueError(f"rank must be in [0, {sp_size}), got {rank}")

        self.sp_size = sp_size
        self.batch_size = batch_size
        self.max_num_seqs = max_num_seqs
        self.rank = rank
        self.master_rank = master_rank

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def owner_ranks(self) -> list[int]: ...

    @abstractmethod
    def owned_seq_count(self, owner_rank: int) -> int: ...

    @abstractmethod
    def participants_for(self, owner_rank: int, seq_idx: int) -> list[int]: ...

    def active_ranks(self) -> list[int]:
        active: set[int] = set()
        for owner_rank in self.owner_ranks():
            owned_seqs = self.owned_seq_count(owner_rank)
            if owned_seqs <= 0:
                continue
            active.add(owner_rank)
            for seq_idx in range(owned_seqs):
                active.update(self.participants_for(owner_rank, seq_idx))
        if not active:
            active.add(MASTER_RANK)
        return sorted(active)

    def total_owned_seqs(self) -> int:
        return sum(self.owned_seq_count(owner_rank) for owner_rank in self.owner_ranks())

    def describe(self) -> str:
        return (
            f"{self.name}(sp_size={self.sp_size}, batch_size={self.batch_size}, "
            f"max_num_seqs={self.max_num_seqs}, master_rank={self.master_rank})"
        )


class FanOutPattern(BaseTrafficPattern):
    @property
    def name(self) -> str:
        return "fan_out"

    def owner_ranks(self) -> list[int]:
        return [self.master_rank]

    def owned_seq_count(self, owner_rank: int) -> int:
        return self.batch_size if owner_rank == self.master_rank else 0

    def participants_for(self, owner_rank: int, seq_idx: int) -> list[int]:
        if owner_rank != self.master_rank or not 0 <= seq_idx < self.batch_size:
            return []
        return list(range(self.sp_size))


class UniformPattern(BaseTrafficPattern):
    @property
    def name(self) -> str:
        return "uniform"

    def owner_ranks(self) -> list[int]:
        return list(range(self.sp_size))

    def owned_seq_count(self, owner_rank: int) -> int:
        return self.batch_size if 0 <= owner_rank < self.sp_size else 0

    def participants_for(self, owner_rank: int, seq_idx: int) -> list[int]:
        if not 0 <= owner_rank < self.sp_size or not 0 <= seq_idx < self.batch_size:
            return []
        return list(range(self.sp_size))


class FanInPattern(BaseTrafficPattern):
    @property
    def name(self) -> str:
        return "fan_in"

    def owner_ranks(self) -> list[int]:
        return [rank for rank in range(self.sp_size) if rank != self.master_rank]

    def owned_seq_count(self, owner_rank: int) -> int:
        return self.batch_size if owner_rank != self.master_rank else 0

    def participants_for(self, owner_rank: int, seq_idx: int) -> list[int]:
        if owner_rank == self.master_rank or not 0 <= seq_idx < self.batch_size:
            return []
        return [self.master_rank]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NanoDeploy MLA SP all-to-all backends under traffic "
            "patterns that approximate DeepSeek V3 decode routing."
        )
    )
    parser.add_argument(
        "--cp-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated SP/CP world sizes to benchmark.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="hao_basic,nccl",
        help="Comma-separated backends to benchmark.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="fan_out,uniform,fan_in",
        help="Comma-separated traffic patterns to benchmark.",
    )
    parser.add_argument(
        "--payloads",
        type=str,
        default="Q,Res,Lse",
        help="Comma-separated payloads to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated per-owner batch sizes used to form curves.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Deprecated alias for a single batch size. If set, overrides --batch-sizes.",
    )
    parser.add_argument("--master-rank", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float16", "bfloat16"),
    )
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--kv-lora-rank", type=int, default=DEFAULT_KV_LORA_RANK)
    parser.add_argument(
        "--qk-rope-head-dim", type=int, default=DEFAULT_QK_ROPE_HEAD_DIM
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Q feature head dim. Default: kv_lora_rank + qk_rope_head_dim.",
    )
    parser.add_argument(
        "--v-head-dim",
        type=int,
        default=None,
        help="Res feature head dim. Default: kv_lora_rank.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="SP buffer max_num_seqs. 0 = auto = batch_size for each curve point.",
    )
    parser.add_argument("--mode", type=str, default="graph", choices=("eager", "graph"))
    parser.add_argument(
        "--preamble",
        type=str,
        default="all_reduce",
        choices=("none", "all_reduce", "all_gather"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--graph-inner-iters", type=int, default=20)
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="profiler_traces/mla_sp_backend_bench/traces",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="profiler_traces/mla_sp_backend_bench/summary.json",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="profiler_traces/mla_sp_backend_bench/comparison.csv",
    )
    return parser.parse_args()


def parse_csv_choices(raw: str) -> list[str]:
    values = []
    for part in raw.split(","):
        value = part.strip()
        if value:
            values.append(value)
    return values


def parse_int_list(raw: str, *, label: str) -> list[int]:
    values = []
    for part in parse_csv_choices(raw):
        value = int(part)
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"No {label}s provided")
    return sorted(dict.fromkeys(values))


def parse_choice_list(raw: str, *, label: str, allowed: tuple[str, ...]) -> list[str]:
    values = []
    for part in parse_csv_choices(raw):
        if part not in allowed:
            raise ValueError(f"Unsupported {label}: {part}")
        values.append(part)
    if not values:
        raise ValueError(f"No {label}s provided")
    return values


def parse_cp_sizes(raw: str) -> list[int]:
    return parse_int_list(raw, label="cp_size")


def parse_batch_sizes(args) -> list[int]:
    if args.num_requests is not None:
        if args.num_requests <= 0:
            raise ValueError("--num-requests must be positive")
        return [args.num_requests]
    return parse_int_list(args.batch_sizes, label="batch_size")


def parse_backends(raw: str) -> list[SPBackend]:
    values = parse_choice_list(raw, label="backend", allowed=BACKEND_CHOICES)
    return [value for value in values]  # type: ignore[return-value]


def parse_patterns(raw: str) -> list[str]:
    return parse_choice_list(raw, label="pattern", allowed=PATTERN_CHOICES)


def parse_payloads(raw: str) -> list[str]:
    return parse_choice_list(raw, label="payload", allowed=PAYLOAD_CHOICES)


def resolve_mla_dims(args) -> tuple[int, int]:
    head_dim = args.head_dim
    if head_dim is None:
        head_dim = args.kv_lora_rank + args.qk_rope_head_dim
    v_head_dim = args.v_head_dim
    if v_head_dim is None:
        v_head_dim = args.kv_lora_rank
    if head_dim <= 0 or v_head_dim <= 0:
        raise ValueError("Resolved head_dim/v_head_dim must be positive")
    return head_dim, v_head_dim


def resolve_max_num_seqs(configured_max_num_seqs: int, batch_size: int) -> int:
    if configured_max_num_seqs == 0:
        return batch_size
    if configured_max_num_seqs < batch_size:
        raise ValueError(
            f"--max-num-seqs ({configured_max_num_seqs}) must be >= batch_size ({batch_size})"
        )
    return configured_max_num_seqs


def create_traffic_pattern(
    *,
    pattern_name: str,
    sp_size: int,
    batch_size: int,
    max_num_seqs: int,
    rank: int,
    master_rank: int,
) -> BaseTrafficPattern:
    common_kwargs = {
        "sp_size": sp_size,
        "batch_size": batch_size,
        "max_num_seqs": max_num_seqs,
        "rank": rank,
        "master_rank": master_rank,
    }
    if pattern_name == "fan_out":
        return FanOutPattern(**common_kwargs)
    if pattern_name == "uniform":
        return UniformPattern(**common_kwargs)
    if pattern_name == "fan_in":
        return FanInPattern(**common_kwargs)
    raise ValueError(f"Unsupported pattern: {pattern_name}")


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q / 100.0).item())


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_us": 0.0,
            "p50_us": 0.0,
            "p90_us": 0.0,
            "p99_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "count": len(values),
        "mean_us": float(sum(values) / len(values)),
        "p50_us": percentile(values, 50.0),
        "p90_us": percentile(values, 90.0),
        "p99_us": percentile(values, 99.0),
        "min_us": float(min(values)),
        "max_us": float(max(values)),
    }


def summarize_rank_numbers(rank_items: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in rank_items]
    return {
        "sum": float(sum(values)),
        "mean": float(sum(values) / len(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
    }


def log_once(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == MASTER_RANK:
        print(message, flush=True)


def init_dist() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def install_bench_dist_context(group: dist.ProcessGroup) -> None:
    distributed_mod._DIST_CONTEXT = _BenchDistContext(group=group)


def create_preamble_state(
    preamble: str, group: dist.ProcessGroup, device: torch.device
) -> PreambleState:
    if preamble == "none":
        return PreambleState(preamble=preamble)
    sync_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
    if preamble == "all_reduce":
        return PreambleState(preamble=preamble, sync_tensor=sync_tensor)
    if preamble == "all_gather":
        gather_out = torch.empty(
            dist.get_world_size(group=group),
            dtype=sync_tensor.dtype,
            device=device,
        )
        return PreambleState(
            preamble=preamble,
            sync_tensor=sync_tensor,
            gather_out=gather_out,
        )
    raise ValueError(f"Unsupported preamble: {preamble}")


def run_preamble(state: PreambleState, group: dist.ProcessGroup) -> None:
    if state.preamble == "none":
        return
    if state.preamble == "all_reduce":
        assert state.sync_tensor is not None
        dist.all_reduce(state.sync_tensor, op=dist.ReduceOp.SUM, group=group)
        state.sync_tensor.fill_(1.0)
        return
    if state.preamble == "all_gather":
        assert state.sync_tensor is not None
        assert state.gather_out is not None
        dist.all_gather_into_tensor(state.gather_out, state.sync_tensor, group=group)
        return
    raise ValueError(f"Unsupported preamble: {state.preamble}")


def q_seq_value(owner_rank: int, seq_idx: int) -> float:
    return 1.0 + float(owner_rank * 1024 + seq_idx)


def res_seq_value(src_rank: int, owner_rank: int, seq_idx: int) -> float:
    return 16.0 + float(src_rank * 2048 + owner_rank * 64 + seq_idx)


def lse_seq_value(src_rank: int, owner_rank: int, seq_idx: int) -> float:
    return 64.0 + float(src_rank * 2048 + owner_rank * 64 + seq_idx)


def local_buffer_view(
    buffer,
    *,
    world_size: int,
    max_num_seqs: int,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = world_size * max_num_seqs * feature_dim
    return buffer.local_buffer.view(dtype)[:numel].view(world_size, max_num_seqs, feature_dim)


def build_q_case_strided(
    pattern: BaseTrafficPattern,
    buffer,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
) -> BuiltPayload:
    rank = pattern.rank
    x = torch.zeros((pattern.max_num_seqs, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros(
        (pattern.sp_size, pattern.max_num_seqs), dtype=torch.int32, device=device
    )
    expected = torch.zeros(
        (pattern.sp_size, pattern.max_num_seqs, feature_dim),
        dtype=dtype,
        device=device,
    )

    buffer.local_buffer.zero_()
    local = local_buffer_view(
        buffer,
        world_size=pattern.sp_size,
        max_num_seqs=pattern.max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    )

    local_owned_seqs = pattern.owned_seq_count(rank)
    local_compute_slots = 0
    for owner_rank in pattern.owner_ranks():
        owned_seqs = pattern.owned_seq_count(owner_rank)
        for seq_idx in range(owned_seqs):
            participants = pattern.participants_for(owner_rank, seq_idx)
            if rank in participants:
                local_compute_slots += 1

    for seq_idx in range(local_owned_seqs):
        value = q_seq_value(rank, seq_idx)
        participants = pattern.participants_for(rank, seq_idx)
        x[seq_idx].fill_(value)
        if rank in participants:
            local[rank, seq_idx].fill_(value)
        for target_rank in participants:
            if target_rank != rank:
                mask[target_rank, seq_idx] = 1

    for owner_rank in pattern.owner_ranks():
        owned_seqs = pattern.owned_seq_count(owner_rank)
        for seq_idx in range(owned_seqs):
            if rank in pattern.participants_for(owner_rank, seq_idx):
                expected[owner_rank, seq_idx].fill_(q_seq_value(owner_rank, seq_idx))

    local_send_tokens = int(mask.sum().item())
    local_send_targets = sum(1 for count in mask.sum(dim=1).tolist() if count > 0)
    return BuiltPayload(
        x=x,
        mask=mask,
        offsets=None,
        expected=expected,
        local_owned_seqs=local_owned_seqs,
        local_compute_slots=local_compute_slots,
        local_send_tokens=local_send_tokens,
        local_send_targets=local_send_targets,
    )


def build_q_case(
    pattern: BaseTrafficPattern,
    buffer,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
) -> BuiltPayload:
    if pattern.name == "fan_in":
        return build_q_case_strided(pattern, buffer, dtype, device, feature_dim)

    rank = pattern.rank
    x = torch.zeros((pattern.max_num_seqs, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros(
        (pattern.sp_size, pattern.max_num_seqs), dtype=torch.int32, device=device
    )
    expected = torch.zeros(
        (pattern.sp_size * pattern.max_num_seqs, feature_dim),
        dtype=dtype,
        device=device,
    )
    offsets = torch.zeros(pattern.sp_size + 1, dtype=torch.int32, device=device)

    buffer.local_buffer.zero_()
    local_flat = local_buffer_view(
        buffer,
        world_size=pattern.sp_size,
        max_num_seqs=pattern.max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    ).view(pattern.sp_size * pattern.max_num_seqs, feature_dim)

    local_owned_seqs = pattern.owned_seq_count(rank)
    local_compute_slots = 0
    receive_counts = [0 for _ in range(pattern.sp_size)]
    for owner_rank in pattern.owner_ranks():
        owned_seqs = pattern.owned_seq_count(owner_rank)
        receive_count = 0
        for seq_idx in range(owned_seqs):
            participants = pattern.participants_for(owner_rank, seq_idx)
            if rank in participants:
                local_compute_slots += 1
                receive_count += 1
        if 0 <= owner_rank < pattern.sp_size:
            receive_counts[owner_rank] = receive_count

    running = 0
    for src_rank in range(pattern.sp_size):
        offsets[src_rank] = running
        running += receive_counts[src_rank]
    offsets[pattern.sp_size] = running

    for seq_idx in range(local_owned_seqs):
        value = q_seq_value(rank, seq_idx)
        participants = pattern.participants_for(rank, seq_idx)
        x[seq_idx].fill_(value)
        if rank in participants:
            local_idx = sum(
                1
                for prev_seq in range(seq_idx)
                if rank in pattern.participants_for(rank, prev_seq)
            )
            local_flat[int(offsets[rank].item()) + local_idx].fill_(value)
        for target_rank in participants:
            if target_rank != rank:
                mask[target_rank, seq_idx] = 1

    for src_rank in range(pattern.sp_size):
        owned_seqs = pattern.owned_seq_count(src_rank)
        local_idx = 0
        for seq_idx in range(owned_seqs):
            if rank in pattern.participants_for(src_rank, seq_idx):
                packed_idx = int(offsets[src_rank].item()) + local_idx
                expected[packed_idx].fill_(q_seq_value(src_rank, seq_idx))
                local_idx += 1

    local_send_tokens = int(mask.sum().item())
    local_send_targets = sum(1 for count in mask.sum(dim=1).tolist() if count > 0)
    return BuiltPayload(
        x=x,
        mask=mask,
        offsets=offsets,
        expected=expected,
        local_owned_seqs=local_owned_seqs,
        local_compute_slots=local_compute_slots,
        local_send_tokens=local_send_tokens,
        local_send_targets=local_send_targets,
    )


def build_reduce_case(
    pattern: BaseTrafficPattern,
    buffer,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
    value_fn: Callable[[int, int, int], float],
) -> BuiltPayload:
    rank = pattern.rank
    x = torch.zeros(
        (pattern.sp_size * pattern.max_num_seqs, feature_dim), dtype=dtype, device=device
    )
    mask = torch.zeros(
        (pattern.sp_size, pattern.max_num_seqs), dtype=torch.int32, device=device
    )
    expected = torch.zeros(
        (pattern.sp_size, pattern.max_num_seqs, feature_dim),
        dtype=dtype,
        device=device,
    )

    buffer.local_buffer.zero_()
    local = local_buffer_view(
        buffer,
        world_size=pattern.sp_size,
        max_num_seqs=pattern.max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    )

    local_owned_seqs = pattern.owned_seq_count(rank)
    local_compute_slots = 0
    for owner_rank in pattern.owner_ranks():
        owned_seqs = pattern.owned_seq_count(owner_rank)
        for seq_idx in range(owned_seqs):
            participants = pattern.participants_for(owner_rank, seq_idx)
            if rank not in participants:
                continue
            local_compute_slots += 1
            value = value_fn(rank, owner_rank, seq_idx)
            if owner_rank == rank:
                local[rank, seq_idx].fill_(value)
            else:
                x[owner_rank * pattern.max_num_seqs + seq_idx].fill_(value)
                mask[owner_rank, seq_idx] = 1

    for seq_idx in range(local_owned_seqs):
        for src_rank in pattern.participants_for(rank, seq_idx):
            expected[src_rank, seq_idx].fill_(value_fn(src_rank, rank, seq_idx))

    local_send_tokens = int(mask.sum().item())
    local_send_targets = sum(1 for count in mask.sum(dim=1).tolist() if count > 0)
    return BuiltPayload(
        x=x,
        mask=mask,
        offsets=None,
        expected=expected,
        local_owned_seqs=local_owned_seqs,
        local_compute_slots=local_compute_slots,
        local_send_tokens=local_send_tokens,
        local_send_targets=local_send_targets,
    )


def build_res_case(
    pattern: BaseTrafficPattern,
    buffer,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
) -> BuiltPayload:
    return build_reduce_case(
        pattern,
        buffer,
        dtype,
        device,
        feature_dim,
        res_seq_value,
    )


def build_lse_case(
    pattern: BaseTrafficPattern,
    buffer,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
) -> BuiltPayload:
    return build_reduce_case(
        pattern,
        buffer,
        dtype,
        device,
        feature_dim,
        lse_seq_value,
    )


def build_payload_cases(args) -> dict[str, PayloadCase]:
    return {
        "Q": PayloadCase(
            name="Q",
            buffer_name="q_buffer",
            feature_dim=args.num_heads * args.head_dim,
            is_transpose=False,
            builder=build_q_case,
        ),
        "Res": PayloadCase(
            name="Res",
            buffer_name="res_buffer",
            feature_dim=args.num_heads * args.v_head_dim,
            is_transpose=True,
            builder=build_res_case,
        ),
        "Lse": PayloadCase(
            name="Lse",
            buffer_name="lse_buffer",
            feature_dim=args.num_heads,
            is_transpose=True,
            builder=build_lse_case,
        ),
    }


def canonical_compare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor.reshape(-1)


def validate_output(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    actual_cmp = canonical_compare_tensor(actual)
    expected_cmp = canonical_compare_tensor(expected)
    local_ok = actual_cmp.shape == expected_cmp.shape and bool(
        torch.equal(actual_cmp, expected_cmp)
    )
    ok_tensor = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if ok_tensor.item() == 1:
        return
    max_diff = 0.0
    if actual_cmp.shape == expected_cmp.shape:
        max_diff = float((actual_cmp.float() - expected_cmp.float()).abs().max().item())
    raise RuntimeError(
        f"Validation failed for {name} on rank={dist.get_rank(group=group)} "
        f"actual_shape={tuple(actual.shape)} expected_shape={tuple(expected.shape)} "
        f"max_diff={max_diff}"
    )


def capture_graph(
    runner: BufferRunner,
    warmup: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run_once()
    return graph


def walltime_benchmark(
    runner: BufferRunner,
    graph: torch.cuda.CUDAGraph | None,
    mode: str,
    iters: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> float:
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        if mode == "graph":
            assert graph is not None
            graph.replay()
        else:
            runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    total_ops = max(iters * runner.inner_iters, 1)
    return (time.perf_counter() - start) * 1_000_000.0 / total_ops


def kernel_stream_id(evt: dict) -> int | str | None:
    args = evt.get("args", {})
    stream = args.get("stream", evt.get("tid"))
    if stream is None:
        return None
    try:
        return int(stream)
    except (TypeError, ValueError):
        return stream


def matches_preamble_kernel(low_name: str, preamble: str) -> bool:
    if preamble == "all_reduce":
        return "allreduce" in low_name
    if preamble == "all_gather":
        return "allgather" in low_name
    return False


def matches_all_to_all_kernel(low_name: str) -> bool:
    excluded_collectives = (
        "allreduce",
        "all_reduce",
        "allgather",
        "all_gather",
        "broadcast",
        "reducescatter",
        "reduce_scatter",
    )
    if any(token in low_name for token in excluded_collectives):
        return False

    return (
        "all_to_all_intra_ll_kernel" in low_name
        or "all_to_all_intra_ll" in low_name
        or "alltoall" in low_name
        or "all_to_all" in low_name
        or (
            "nccl" in low_name
            and (
                "sendrecv" in low_name
                or "send" in low_name
                or "recv" in low_name
            )
        )
    )


def parse_trace_stats(
    trace_path: Path,
    preamble: str,
    total_all2all_iters: int,
    *,
    split_mask_all2all: bool = False,
) -> dict[str, object]:
    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    alltoall_us = 0.0
    preamble_us = 0.0
    alltoall_count = 0
    preamble_count = 0
    matched_alltoall_names: set[str] = set()
    matched_preamble_names: set[str] = set()
    matched_stream_ids: set[int | str] = set()
    alltoall_durations: list[float] = []
    preamble_durations: list[float] = []
    observed_cuda_kernel_names: list[str] = []

    for evt in trace.get("traceEvents", []):
        if evt.get("ph") != "X" or evt.get("cat") != "kernel":
            continue
        name = evt.get("name", "")
        if name and name not in observed_cuda_kernel_names:
            observed_cuda_kernel_names.append(name)
        low_name = name.lower()
        dur = float(evt.get("dur", 0.0))
        stream_id = kernel_stream_id(evt)

        if matches_all_to_all_kernel(low_name):
            alltoall_us += dur
            alltoall_count += 1
            matched_alltoall_names.add(name)
            alltoall_durations.append(dur)
            if stream_id is not None:
                matched_stream_ids.add(stream_id)

        if matches_preamble_kernel(low_name, preamble):
            preamble_us += dur
            preamble_count += 1
            matched_preamble_names.add(name)
            preamble_durations.append(dur)
            if stream_id is not None:
                matched_stream_ids.add(stream_id)

    if preamble != "none":
        comm_region_durations = [
            preamble_durations[i] + alltoall_durations[i]
            for i in range(min(len(preamble_durations), len(alltoall_durations)))
        ]
    else:
        comm_region_durations = list(alltoall_durations)

    if split_mask_all2all:
        payload_alltoall_durations = alltoall_durations[0::2]
        mask_alltoall_durations = alltoall_durations[1::2]
    else:
        payload_alltoall_durations = list(alltoall_durations)
        mask_alltoall_durations = []

    payload_alltoall_us = float(sum(payload_alltoall_durations))
    mask_alltoall_us = float(sum(mask_alltoall_durations))

    return {
        "all2all": {
            "trace_avg_us": alltoall_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(alltoall_durations)["p50_us"],
            "kernel_count": alltoall_count,
            "matched_kernel_names": sorted(matched_alltoall_names),
        },
        "payload_all2all": {
            "trace_avg_us": payload_alltoall_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(payload_alltoall_durations)["p50_us"],
            "kernel_count": len(payload_alltoall_durations),
            "split_from_total": split_mask_all2all,
        },
        "mask_all2all": {
            "trace_avg_us": mask_alltoall_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(mask_alltoall_durations)["p50_us"],
            "kernel_count": len(mask_alltoall_durations),
            "split_from_total": split_mask_all2all,
        },
        "preamble": {
            "trace_avg_us": preamble_us / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(preamble_durations)["p50_us"],
            "kernel_count": preamble_count,
            "matched_kernel_names": sorted(matched_preamble_names),
        },
        "comm_region": {
            "trace_avg_us": (alltoall_us + preamble_us) / max(total_all2all_iters, 1),
            "trace_p50_us": summarize_values(comm_region_durations)["p50_us"],
            "kernel_count": alltoall_count + preamble_count,
        },
        "matched_stream_ids": sorted(matched_stream_ids, key=str),
        "observed_cuda_kernel_names": observed_cuda_kernel_names[:32],
    }


def profiler_benchmark(
    *,
    runner: BufferRunner,
    graph: torch.cuda.CUDAGraph | None,
    mode: str,
    profile_iters: int,
    trace_path: Path,
    device: torch.device,
    group: dist.ProcessGroup,
    split_mask_all2all: bool,
) -> dict[str, object]:
    if profile_iters <= 0:
        raise ValueError("profile_iters must be positive")
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    schedule = torch.profiler.schedule(wait=0, warmup=3, active=profile_iters, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        schedule=schedule,
    ) as prof:
        for _ in range(profile_iters + 3):
            if mode == "graph":
                assert graph is not None
                graph.replay()
            else:
                runner.run_once()
            torch.cuda.synchronize(device)
            prof.step()
    dist.barrier(group=group)
    prof.export_chrome_trace(str(trace_path))
    stats = parse_trace_stats(
        trace_path,
        runner.preamble,
        total_all2all_iters=profile_iters * runner.inner_iters,
        split_mask_all2all=split_mask_all2all,
    )
    if stats["all2all"]["kernel_count"] == 0:  # type: ignore[index]
        raise RuntimeError(
            "Failed to find target all_to_all kernels in trace "
            f"{trace_path}. Observed kernels: {stats['observed_cuda_kernel_names']}"
        )
    return stats


def gather_rank_meta(meta: dict[str, object], group: dist.ProcessGroup) -> list[dict[str, object]]:
    gathered = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered, meta, group=group)
    return gathered


def active_aggregate(rank_items: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(item[key]["mean_us"]) for item in rank_items]  # type: ignore[index]
    return {
        "mean_of_means_us": float(sum(values) / len(values)) if values else 0.0,
        "slowest_mean_us": float(max(values)) if values else 0.0,
        "fastest_mean_us": float(min(values)) if values else 0.0,
    }


def zero_payload_summary(
    *,
    payload: PayloadCase,
    cp_size: int,
    pattern: BaseTrafficPattern,
    itemsize: int,
) -> dict[str, object]:
    zero_summary = summarize_values([0.0])
    zero_rank = {
        "global_rank": 0,
        "cp_rank": 0,
        "local_owned_seqs": float(pattern.owned_seq_count(0)),
        "local_compute_slots": float(pattern.owned_seq_count(0)),
        "local_send_tokens": 0.0,
        "local_send_targets": 0.0,
        "local_send_bytes": 0.0,
        "wall_summary_us": zero_summary,
        "all2all_summary_us": zero_summary,
        "payload_all2all_summary_us": zero_summary,
        "mask_all2all_summary_us": zero_summary,
        "comm_region_summary_us": zero_summary,
        "trace_paths": [],
    }
    rank_items = [zero_rank]
    return {
        "payload": payload.name,
        "pattern": pattern.name,
        "pattern_desc": pattern.describe(),
        "cp_size": cp_size,
        "batch_size": pattern.batch_size,
        "max_num_seqs": pattern.max_num_seqs,
        "feature_dim": payload.feature_dim,
        "bytes_per_row": payload.feature_dim * itemsize,
        "buffer_max_dispatch_per_msg": cp_size if payload.name == "Q" else 1,
        "active_ranks": [0],
        "ranks": rank_items,
        "wall_active_summary": active_aggregate(rank_items, "wall_summary_us"),
        "all2all_active_summary": active_aggregate(rank_items, "all2all_summary_us"),
        "payload_all2all_active_summary": active_aggregate(
            rank_items,
            "payload_all2all_summary_us",
        ),
        "mask_all2all_active_summary": active_aggregate(
            rank_items,
            "mask_all2all_summary_us",
        ),
        "comm_region_active_summary": active_aggregate(
            rank_items,
            "comm_region_summary_us",
        ),
        "local_owned_seqs_summary": summarize_rank_numbers(rank_items, "local_owned_seqs"),
        "local_compute_slots_summary": summarize_rank_numbers(rank_items, "local_compute_slots"),
        "local_send_tokens_summary": summarize_rank_numbers(rank_items, "local_send_tokens"),
        "local_send_targets_summary": summarize_rank_numbers(rank_items, "local_send_targets"),
        "local_send_bytes_summary": summarize_rank_numbers(rank_items, "local_send_bytes"),
    }


def benchmark_payload(
    *,
    args,
    backend: SPBackend,
    payload: PayloadCase,
    group: dist.ProcessGroup,
    pattern: BaseTrafficPattern,
    dtype: torch.dtype,
    device: torch.device,
    trace_dir: Path,
    itemsize: int,
) -> dict[str, object]:
    if pattern.sp_size == 1:
        return zero_payload_summary(
            payload=payload,
            cp_size=pattern.sp_size,
            pattern=pattern,
            itemsize=itemsize,
        )

    ctx = get_sp_context()
    buffer = getattr(ctx, payload.buffer_name)
    built = payload.builder(pattern, buffer, dtype, device, payload.feature_dim)

    runner = BufferRunner(
        group=group,
        preamble=args.preamble,
        device=device,
        inner_iters=args.graph_inner_iters if args.mode == "graph" else 1,
        buffer=buffer,
        x=built.x,
        mask=built.mask,
        offsets=built.offsets,
        is_transpose=payload.is_transpose,
    )

    runner.run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    assert runner.output is not None
    # Synthetic fan-in Q has no direct decode equivalent: only the master rank
    # consumes the output, and native backends may choose different unused-slot
    # layouts. Keep it as a timing-only case.
    if not (payload.name == "Q" and pattern.name == "fan_in"):
        validate_output(
            name=(
                f"{backend}/{pattern.name}/bs{pattern.batch_size}/"
                f"{payload.name}/cp{pattern.sp_size}"
            ),
            actual=runner.output.clone(),
            expected=built.expected,
            group=group,
            device=device,
        )

    graph = None
    if args.mode == "graph":
        graph = capture_graph(runner, args.warmup, group, device)

    wall_samples: list[float] = []
    all2all_trace_samples: list[float] = []
    payload_all2all_trace_samples: list[float] = []
    mask_all2all_trace_samples: list[float] = []
    comm_region_trace_samples: list[float] = []
    trace_paths: list[str] = []
    split_mask_all2all = backend == "nccl" and payload.name == "Q" and built.offsets is not None

    for repeat_idx in range(args.repeats):
        wall_samples.append(
            walltime_benchmark(runner, graph, args.mode, args.iters, device, group)
        )
        trace_path = (
            trace_dir
            / (
                f"{backend}_cp{pattern.sp_size}_{pattern.name}_bs{pattern.batch_size}_"
                f"{payload.name}_rank{pattern.rank}_repeat{repeat_idx}.json"
            )
        )
        trace_stats = profiler_benchmark(
            runner=runner,
            graph=graph,
            mode=args.mode,
            profile_iters=args.profile_iters,
            trace_path=trace_path,
            device=device,
            group=group,
            split_mask_all2all=split_mask_all2all,
        )
        trace_paths.append(str(trace_path))
        all2all_trace_samples.append(float(trace_stats["all2all"]["trace_avg_us"]))  # type: ignore[index]
        payload_all2all_trace_samples.append(float(trace_stats["payload_all2all"]["trace_avg_us"]))  # type: ignore[index]
        mask_all2all_trace_samples.append(float(trace_stats["mask_all2all"]["trace_avg_us"]))  # type: ignore[index]
        comm_region_trace_samples.append(float(trace_stats["comm_region"]["trace_avg_us"]))  # type: ignore[index]

    rank_meta = {
        "global_rank": dist.get_rank(),
        "cp_rank": dist.get_rank(group=group),
        "local_owned_seqs": float(built.local_owned_seqs),
        "local_compute_slots": float(built.local_compute_slots),
        "local_send_tokens": float(built.local_send_tokens),
        "local_send_targets": float(built.local_send_targets),
        "local_send_bytes": float(built.local_send_tokens * payload.feature_dim * itemsize),
        "wall_summary_us": summarize_values(wall_samples),
        "all2all_summary_us": summarize_values(all2all_trace_samples),
        "payload_all2all_summary_us": summarize_values(payload_all2all_trace_samples),
        "mask_all2all_summary_us": summarize_values(mask_all2all_trace_samples),
        "comm_region_summary_us": summarize_values(comm_region_trace_samples),
        "trace_paths": trace_paths,
    }
    rank_items = gather_rank_meta(rank_meta, group)
    active_rank_set = set(pattern.active_ranks())
    active_rank_items = [
        item for item in rank_items if int(item["cp_rank"]) in active_rank_set
    ]
    return {
        "payload": payload.name,
        "pattern": pattern.name,
        "pattern_desc": pattern.describe(),
        "cp_size": pattern.sp_size,
        "batch_size": pattern.batch_size,
        "max_num_seqs": pattern.max_num_seqs,
        "feature_dim": payload.feature_dim,
        "bytes_per_row": payload.feature_dim * itemsize,
        "buffer_max_dispatch_per_msg": pattern.sp_size if payload.name == "Q" else 1,
        "active_ranks": sorted(active_rank_set),
        "ranks": rank_items,
        "wall_active_summary": active_aggregate(active_rank_items, "wall_summary_us"),
        "all2all_active_summary": active_aggregate(active_rank_items, "all2all_summary_us"),
        "payload_all2all_active_summary": active_aggregate(
            active_rank_items,
            "payload_all2all_summary_us",
        ),
        "mask_all2all_active_summary": active_aggregate(
            active_rank_items,
            "mask_all2all_summary_us",
        ),
        "comm_region_active_summary": active_aggregate(
            active_rank_items,
            "comm_region_summary_us",
        ),
        "local_owned_seqs_summary": summarize_rank_numbers(
            active_rank_items,
            "local_owned_seqs",
        ),
        "local_compute_slots_summary": summarize_rank_numbers(
            active_rank_items,
            "local_compute_slots",
        ),
        "local_send_tokens_summary": summarize_rank_numbers(
            active_rank_items,
            "local_send_tokens",
        ),
        "local_send_targets_summary": summarize_rank_numbers(
            active_rank_items,
            "local_send_targets",
        ),
        "local_send_bytes_summary": summarize_rank_numbers(
            active_rank_items,
            "local_send_bytes",
        ),
    }


def run_backend_cp_bench(
    *,
    args,
    backend: SPBackend,
    cp_size: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
    payload_names: list[str],
    pattern_names: list[str],
    batch_sizes: list[int],
    trace_root: Path,
    itemsize: int,
) -> dict[str, dict[str, dict[str, object]]]:
    if cp_size > world_size:
        raise ValueError(f"cp_size={cp_size} exceeds world_size={world_size}")
    group = dist.new_group(ranks=list(range(cp_size)), backend="nccl")
    dist.barrier()
    if rank >= cp_size:
        dist.barrier()
        return {}

    install_bench_dist_context(group)
    payload_cases = build_payload_cases(args)
    results: dict[str, dict[str, dict[str, object]]] = {}
    group_rank = dist.get_rank(group=group)

    for pattern_name in pattern_names:
        results[pattern_name] = {}
        for batch_size in batch_sizes:
            max_num_seqs = resolve_max_num_seqs(args.max_num_seqs, batch_size)
            pattern = create_traffic_pattern(
                pattern_name=pattern_name,
                sp_size=cp_size,
                batch_size=batch_size,
                max_num_seqs=max_num_seqs,
                rank=group_rank,
                master_rank=args.master_rank,
            )

            dist.barrier(group=group)
            set_sp_context(
                max_num_seqs=max_num_seqs,
                head_size=args.head_dim,
                num_attention_heads=args.num_heads,
                dtype=dtype,
                rank=group_rank,
                sp_size=cp_size,
                backend=backend,
            )
            dist.barrier(group=group)

            case_trace_dir = trace_root / backend / f"cp{cp_size}" / pattern_name / f"bs{batch_size}"
            case_trace_dir.mkdir(parents=True, exist_ok=True)

            payload_results: dict[str, object] = {}
            for payload_name in payload_names:
                result = benchmark_payload(
                    args=args,
                    backend=backend,
                    payload=payload_cases[payload_name],
                    group=group,
                    pattern=pattern,
                    dtype=dtype,
                    device=device,
                    trace_dir=case_trace_dir,
                    itemsize=itemsize,
                )
                payload_results[payload_name] = result
                log_once(
                    f"[{backend}] cp_size={cp_size} pattern={pattern_name} "
                    f"batch_size={batch_size} payload={payload_name} "
                    f"all2all_mean_us={result['all2all_active_summary']['mean_of_means_us']:.2f} "
                    f"payload_all2all_mean_us={result['payload_all2all_active_summary']['mean_of_means_us']:.2f} "
                    f"mask_all2all_mean_us={result['mask_all2all_active_summary']['mean_of_means_us']:.2f} "
                    f"comm_region_mean_us={result['comm_region_active_summary']['mean_of_means_us']:.2f}"
                )

            results[pattern_name][str(batch_size)] = {
                "pattern": pattern_name,
                "pattern_desc": pattern.describe(),
                "batch_size": batch_size,
                "max_num_seqs": max_num_seqs,
                "payloads": payload_results,
            }

    dist.barrier(group=group)
    dist.barrier()
    return results


def flatten_summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = summary["config"]  # type: ignore[index]
    for backend, backend_results in summary["results"].items():  # type: ignore[index]
        for cp_size, cp_results in backend_results.items():
            for pattern_name, pattern_cases in cp_results.items():
                for batch_size, batch_case in pattern_cases.items():
                    for payload_name, payload_result in batch_case["payloads"].items():
                        rows.append(
                            {
                                "backend": backend,
                                "cp_size": int(cp_size),
                                "pattern": pattern_name,
                                "pattern_desc": batch_case["pattern_desc"],
                                "batch_size": int(batch_size),
                                "max_num_seqs": batch_case["max_num_seqs"],
                                "payload": payload_name,
                                "feature_dim": payload_result["feature_dim"],
                                "bytes_per_row": payload_result["bytes_per_row"],
                                "buffer_max_dispatch_per_msg": payload_result[
                                    "buffer_max_dispatch_per_msg"
                                ],
                                "active_rank_count": len(payload_result["active_ranks"]),
                                "mode": config["mode"],
                                "preamble": config["preamble"],
                                "num_heads": config["num_heads"],
                                "kv_lora_rank": config["kv_lora_rank"],
                                "qk_rope_head_dim": config["qk_rope_head_dim"],
                                "head_dim": config["head_dim"],
                                "v_head_dim": config["v_head_dim"],
                                "dtype": config["dtype"],
                                "wall_mean_us": payload_result["wall_active_summary"]["mean_of_means_us"],
                                "wall_slowest_mean_us": payload_result["wall_active_summary"]["slowest_mean_us"],
                                "wall_fastest_mean_us": payload_result["wall_active_summary"]["fastest_mean_us"],
                                "all2all_mean_us": payload_result["all2all_active_summary"]["mean_of_means_us"],
                                "all2all_slowest_mean_us": payload_result["all2all_active_summary"]["slowest_mean_us"],
                                "all2all_fastest_mean_us": payload_result["all2all_active_summary"]["fastest_mean_us"],
                                "payload_all2all_mean_us": payload_result["payload_all2all_active_summary"]["mean_of_means_us"],
                                "payload_all2all_slowest_mean_us": payload_result["payload_all2all_active_summary"]["slowest_mean_us"],
                                "payload_all2all_fastest_mean_us": payload_result["payload_all2all_active_summary"]["fastest_mean_us"],
                                "mask_all2all_mean_us": payload_result["mask_all2all_active_summary"]["mean_of_means_us"],
                                "mask_all2all_slowest_mean_us": payload_result["mask_all2all_active_summary"]["slowest_mean_us"],
                                "mask_all2all_fastest_mean_us": payload_result["mask_all2all_active_summary"]["fastest_mean_us"],
                                "comm_region_mean_us": payload_result["comm_region_active_summary"]["mean_of_means_us"],
                                "comm_region_slowest_mean_us": payload_result["comm_region_active_summary"]["slowest_mean_us"],
                                "comm_region_fastest_mean_us": payload_result["comm_region_active_summary"]["fastest_mean_us"],
                                "local_owned_seqs_sum": payload_result["local_owned_seqs_summary"]["sum"],
                                "local_owned_seqs_mean": payload_result["local_owned_seqs_summary"]["mean"],
                                "local_owned_seqs_max": payload_result["local_owned_seqs_summary"]["max"],
                                "local_compute_slots_sum": payload_result["local_compute_slots_summary"]["sum"],
                                "local_compute_slots_mean": payload_result["local_compute_slots_summary"]["mean"],
                                "local_compute_slots_max": payload_result["local_compute_slots_summary"]["max"],
                                "local_send_tokens_sum": payload_result["local_send_tokens_summary"]["sum"],
                                "local_send_tokens_mean": payload_result["local_send_tokens_summary"]["mean"],
                                "local_send_tokens_max": payload_result["local_send_tokens_summary"]["max"],
                                "local_send_targets_sum": payload_result["local_send_targets_summary"]["sum"],
                                "local_send_targets_mean": payload_result["local_send_targets_summary"]["mean"],
                                "local_send_targets_max": payload_result["local_send_targets_summary"]["max"],
                                "local_send_bytes_sum": payload_result["local_send_bytes_summary"]["sum"],
                                "local_send_bytes_mean": payload_result["local_send_bytes_summary"]["mean"],
                                "local_send_bytes_max": payload_result["local_send_bytes_summary"]["max"],
                            }
                        )
    return rows


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No benchmark rows to write")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.head_dim, args.v_head_dim = resolve_mla_dims(args)
    if args.master_rank < 0:
        raise ValueError("--master-rank must be non-negative")
    if args.iters <= 0 or args.repeats <= 0 or args.profile_iters <= 0:
        raise ValueError("--iters/--repeats/--profile-iters must be positive")

    cp_sizes = parse_cp_sizes(args.cp_sizes)
    batch_sizes = parse_batch_sizes(args)
    backends = parse_backends(args.backends)
    if "nccl_compact" in backends and args.mode != "eager":
        raise ValueError(
            "nccl_compact uses variable split-size NCCL collectives and only "
            "supports --mode eager in this microbenchmark."
        )
    pattern_names = parse_patterns(args.patterns)
    payload_names = parse_payloads(args.payloads)

    rank = -1
    try:
        rank, world_size, device = init_dist()
        dtype = get_dtype(args.dtype)
        itemsize = torch.tensor([], dtype=dtype).element_size()
        trace_root = Path(args.trace_dir)
        if rank == MASTER_RANK:
            trace_root.mkdir(parents=True, exist_ok=True)

        log_once(
            "Running MLA SP backend benchmark "
            f"(world_size={world_size}, cp_sizes={cp_sizes}, patterns={pattern_names}, "
            f"batch_sizes={batch_sizes}, backends={backends}, payloads={payload_names}, "
            f"mode={args.mode}, dtype={args.dtype}, head_dim={args.head_dim}, "
            f"v_head_dim={args.v_head_dim})"
        )

        results: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
        for backend in backends:
            results[backend] = {}
            for cp_size in cp_sizes:
                results[backend][str(cp_size)] = run_backend_cp_bench(
                    args=args,
                    backend=backend,
                    cp_size=cp_size,
                    rank=rank,
                    world_size=world_size,
                    dtype=dtype,
                    device=device,
                    payload_names=payload_names,
                    pattern_names=pattern_names,
                    batch_sizes=batch_sizes,
                    trace_root=trace_root,
                    itemsize=itemsize,
                )

        if rank == MASTER_RANK:
            summary = {
                "config": {
                    "cp_sizes": cp_sizes,
                    "batch_sizes": batch_sizes,
                    "backends": backends,
                    "patterns": pattern_names,
                    "payloads": payload_names,
                    "dtype": args.dtype,
                    "num_heads": args.num_heads,
                    "kv_lora_rank": args.kv_lora_rank,
                    "qk_rope_head_dim": args.qk_rope_head_dim,
                    "head_dim": args.head_dim,
                    "v_head_dim": args.v_head_dim,
                    "max_num_seqs": args.max_num_seqs,
                    "mode": args.mode,
                    "preamble": args.preamble,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "profile_iters": args.profile_iters,
                    "graph_inner_iters": args.graph_inner_iters,
                    "master_rank": args.master_rank,
                    "world_size": world_size,
                },
                "results": results,
            }
            rows = flatten_summary_rows(summary)
            summary["rows"] = rows

            summary_path = Path(args.summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            write_csv(rows, Path(args.csv_path))
            log_once(f"Summary written to {summary_path}")
            log_once(f"CSV written to {args.csv_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank in (-1, MASTER_RANK) and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
Compare NanoDeploy's MLA SP all-to-all backends under the real backend switch path.

This script is intentionally narrower than the exploratory SP attention benchmarks:
it only validates the three MLA all-to-all payloads used by NanoDeploy's decode path.

Covered cases:
- `Q`: masked non-transpose + offsets dispatch from master rank to remote SP ranks
- `Res`: masked transpose gather back to master rank
- `Lse`: masked transpose gather back to master rank

The script initializes `SPContext` twice, once per backend, so the exercised path
matches NanoDeploy's startup-time backend selection:

`set_sp_context(..., backend="legacy_ll" | "hao_basic" | "nccl" | "nccl_compact")`

Usage:
`torchrun --nproc_per_node=8 tests/test_mla_sp_backend_correctness.py --mode both`
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.distributed as dist

from nanodeploy.worker import distributed as distributed_mod
from nanodeploy.worker.sp_backend import SPBackend
from nanodeploy.worker.sp_context import get_sp_context, set_sp_context


MASTER_RANK = 0
BACKEND_CHOICES: tuple[SPBackend, ...] = (
    "legacy_ll",
    "hao_basic",
    "nccl",
    "nccl_compact",
)
MODE_CHOICES = ("eager", "graph", "both")


@dataclass(frozen=True)
class PayloadCase:
    name: str
    buffer_name: str
    feature_dim: int
    is_transpose: bool
    builder: Callable[
        [torch.Tensor, int, int, int, int, torch.dtype, torch.device],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor],
    ]


@dataclass
class _TestDistContext:
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


@dataclass
class PreambleState:
    preamble: str
    sync_tensor: torch.Tensor | None = None
    gather_out: torch.Tensor | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate that NanoDeploy's MLA SP all-to-all backends produce "
            "identical outputs."
        )
    )
    parser.add_argument(
        "--reference-backend",
        type=str,
        default="hao_basic",
        choices=BACKEND_CHOICES,
        help="Known-good backend to compare against.",
    )
    parser.add_argument(
        "--candidate-backend",
        type=str,
        default="nccl",
        choices=BACKEND_CHOICES,
        help="Backend under validation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float16", "bfloat16"),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=128,
        help="MLA query head count.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=576,
        help="MLA Q/K head dim.",
    )
    parser.add_argument(
        "--v-head-dim",
        type=int,
        default=512,
        help="MLA output V dim.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="SP buffer max_num_seqs used to build SPContext.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="Active sequence count inside the max_num_seqs window.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=MODE_CHOICES,
        help="Run eager only, graph only, or both.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before CUDAGraph capture.",
    )
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=3,
        help="How many graph replays to run before validating graph output.",
    )
    parser.add_argument(
        "--preamble",
        type=str,
        default="all_reduce",
        choices=("none", "all_reduce", "all_gather"),
        help="Optional sync collective before each all_to_all launch.",
    )
    return parser.parse_args()


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def log_once(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == MASTER_RANK:
        print(message, flush=True)


def init_dist() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test script.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    return dist.get_rank(), dist.get_world_size(), device


def install_test_dist_context(group: dist.ProcessGroup) -> None:
    distributed_mod._DIST_CONTEXT = _TestDistContext(group=group)


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


def q_seq_value(seq_idx: int) -> float:
    return 1.0 + float(seq_idx)


def res_seq_value(rank: int, seq_idx: int) -> float:
    return 16.0 + float(rank * 8 + seq_idx)


def lse_seq_value(rank: int, seq_idx: int) -> float:
    return 64.0 + float(rank * 4 + seq_idx)


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


def build_q_case(
    buffer,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    num_requests: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = [
        num_requests // world_size + (1 if src_rank < (num_requests % world_size) else 0)
        for src_rank in range(world_size)
    ]
    local_q_count = counts[rank]
    x = torch.zeros((local_q_count, feature_dim), dtype=dtype, device=device)
    mask = torch.zeros((world_size, max_num_seqs), dtype=torch.int32, device=device)
    offsets = torch.zeros(world_size + 1, dtype=torch.int32, device=device)
    expected = torch.zeros(
        (world_size, max_num_seqs, feature_dim), dtype=dtype, device=device
    )
    expected_flat = expected.view(world_size * max_num_seqs, feature_dim)

    running = 0
    for src_rank, count in enumerate(counts):
        offsets[src_rank] = running
        running += count
    offsets[world_size] = running

    buffer.local_buffer.zero_()
    local_flat = local_buffer_view(
        buffer,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    ).view(world_size * max_num_seqs, feature_dim)

    global_seq_base = sum(counts[:rank])
    for local_idx in range(local_q_count):
        global_seq_idx = global_seq_base + local_idx
        value = q_seq_value(global_seq_idx)
        x[local_idx].fill_(value)
        local_flat[int(offsets[rank].item()) + local_idx].fill_(value)

    if world_size > 1 and local_q_count > 0:
        for target_rank in range(world_size):
            if target_rank != rank:
                mask[target_rank, :local_q_count] = 1

    for src_rank, count in enumerate(counts):
        seq_base = sum(counts[:src_rank])
        for local_idx in range(count):
            packed_idx = int(offsets[src_rank].item()) + local_idx
            expected_flat[packed_idx].fill_(q_seq_value(seq_base + local_idx))

    return x, mask, offsets, expected


def build_reduce_case(
    buffer,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    num_requests: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    feature_dim: int,
    value_fn: Callable[[int, int], float],
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    x = torch.zeros(
        (world_size * max_num_seqs, feature_dim), dtype=dtype, device=device
    )
    mask = torch.zeros((world_size, max_num_seqs), dtype=torch.int32, device=device)
    expected = torch.zeros(
        (world_size, max_num_seqs, feature_dim), dtype=dtype, device=device
    )

    buffer.local_buffer.zero_()
    local = local_buffer_view(
        buffer,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    )

    for seq_idx in range(num_requests):
        value = value_fn(rank, seq_idx)
        local[rank, seq_idx].fill_(value)
        expected[rank, seq_idx].fill_(value)

        if rank != MASTER_RANK:
            x[MASTER_RANK * max_num_seqs + seq_idx].fill_(value)
            mask[MASTER_RANK, seq_idx] = 1

    if rank == MASTER_RANK:
        for src_rank in range(world_size):
            for seq_idx in range(num_requests):
                expected[src_rank, seq_idx].fill_(value_fn(src_rank, seq_idx))

    return x, mask, None, expected


def build_res_case(
    buffer,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    num_requests: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    return build_reduce_case(
        buffer,
        rank,
        world_size,
        max_num_seqs,
        num_requests,
        dtype,
        device,
        feature_dim=feature_dim,
        value_fn=res_seq_value,
    )


def build_lse_case(
    buffer,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    num_requests: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    return build_reduce_case(
        buffer,
        rank,
        world_size,
        max_num_seqs,
        num_requests,
        dtype,
        device,
        feature_dim=feature_dim,
        value_fn=lse_seq_value,
    )


def canonical_compare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor.reshape(-1)


def mismatch_summary(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_cmp = canonical_compare_tensor(actual)
    expected_cmp = canonical_compare_tensor(expected)
    summary = {
        "name": name,
        "rank": dist.get_rank(),
        "actual_shape": tuple(actual.shape),
        "expected_shape": tuple(expected.shape),
        "actual_compare_shape": tuple(actual_cmp.shape),
        "expected_compare_shape": tuple(expected_cmp.shape),
    }

    if actual_cmp.shape != expected_cmp.shape:
        return summary

    matches = torch.eq(actual_cmp, expected_cmp)
    if bool(matches.all()):
        summary["max_diff"] = 0.0
        return summary

    mismatch_index = (~matches).nonzero(as_tuple=False)[0].tolist()
    actual_value = float(actual_cmp[tuple(mismatch_index)].float().item())
    expected_value = float(expected_cmp[tuple(mismatch_index)].float().item())
    summary["max_diff"] = float(
        (actual_cmp.float() - expected_cmp.float()).abs().max().item()
    )
    summary["first_mismatch_index"] = mismatch_index
    summary["actual_value"] = actual_value
    summary["expected_value"] = expected_value
    return summary


def assert_tensors_equal(
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

    local_summary = None if local_ok else mismatch_summary(name, actual, expected)
    gathered = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered, local_summary, group=group)
    first_failure = next(item for item in gathered if item is not None)
    raise AssertionError(f"{name} mismatch: {first_failure}")


def run_eager(
    *,
    buffer,
    x: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor | None,
    is_transpose: bool,
    preamble_state: PreambleState,
    group: dist.ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    dist.barrier(group=group)
    run_preamble(preamble_state, group)
    output = buffer.all_to_all_ll(
        x,
        is_transpose=is_transpose,
        mask=mask,
        offsets=offsets,
    )
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return output.clone()


def run_cudagraph(
    *,
    buffer,
    x: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor | None,
    is_transpose: bool,
    warmup: int,
    replays: int,
    preamble_state: PreambleState,
    group: dist.ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    holder: dict[str, torch.Tensor | None] = {"output": None}

    def run_once() -> None:
        run_preamble(preamble_state, group)
        holder["output"] = buffer.all_to_all_ll(
            x,
            is_transpose=is_transpose,
            mask=mask,
            offsets=offsets,
        )

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_once()

    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    output = holder["output"]
    if output is None:
        raise RuntimeError("CUDAGraph run did not produce an output tensor.")
    return output.clone()


def run_payload_case(
    *,
    case: PayloadCase,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    num_requests: int,
    dtype: torch.dtype,
    device: torch.device,
    preamble: str,
    mode: Literal["eager", "graph", "both"],
    warmup: int,
    graph_replays: int,
    group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    ctx = get_sp_context()
    buffer = getattr(ctx, case.buffer_name)
    outputs: dict[str, torch.Tensor] = {}
    preamble_state = create_preamble_state(preamble, group, device)

    if mode in ("eager", "both"):
        x, mask, offsets, expected = case.builder(
            buffer,
            rank,
            world_size,
            max_num_seqs,
            num_requests,
            dtype,
            device,
        )
        eager_output = run_eager(
            buffer=buffer,
            x=x,
            mask=mask,
            offsets=offsets,
            is_transpose=case.is_transpose,
            preamble_state=preamble_state,
            group=group,
            device=device,
        )
        assert_tensors_equal(
            name=f"{case.name}/eager_vs_expected",
            actual=eager_output,
            expected=expected,
            group=group,
            device=device,
        )
        outputs["eager"] = eager_output

    if mode in ("graph", "both"):
        x, mask, offsets, expected = case.builder(
            buffer,
            rank,
            world_size,
            max_num_seqs,
            num_requests,
            dtype,
            device,
        )
        graph_output = run_cudagraph(
            buffer=buffer,
            x=x,
            mask=mask,
            offsets=offsets,
            is_transpose=case.is_transpose,
            warmup=warmup,
            replays=graph_replays,
            preamble_state=preamble_state,
            group=group,
            device=device,
        )
        assert_tensors_equal(
            name=f"{case.name}/graph_vs_expected",
            actual=graph_output,
            expected=expected,
            group=group,
            device=device,
        )
        outputs["graph"] = graph_output

    if mode == "both":
        assert_tensors_equal(
            name=f"{case.name}/eager_vs_graph",
            actual=outputs["eager"],
            expected=outputs["graph"],
            group=group,
            device=device,
        )

    return outputs


def run_backend_suite(
    *,
    backend: SPBackend,
    rank: int,
    world_size: int,
    args,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> dict[str, dict[str, torch.Tensor]]:
    dist.barrier(group=group)
    set_sp_context(
        max_num_seqs=args.max_num_seqs,
        head_size=args.head_dim,
        num_attention_heads=args.num_heads,
        dtype=dtype,
        rank=rank,
        sp_size=world_size,
        backend=backend,
    )
    dist.barrier(group=group)

    payload_cases = [
        PayloadCase(
            name="Q",
            buffer_name="q_buffer",
            feature_dim=args.num_heads * args.head_dim,
            is_transpose=False,
            builder=lambda buffer, rank, world_size, max_num_seqs, num_requests, dtype, device: build_q_case(
                buffer,
                rank,
                world_size,
                max_num_seqs,
                num_requests,
                dtype,
                device,
                feature_dim=args.num_heads * args.head_dim,
            ),
        ),
        PayloadCase(
            name="Res",
            buffer_name="res_buffer",
            feature_dim=args.num_heads * args.v_head_dim,
            is_transpose=True,
            builder=lambda buffer, rank, world_size, max_num_seqs, num_requests, dtype, device: build_res_case(
                buffer,
                rank,
                world_size,
                max_num_seqs,
                num_requests,
                dtype,
                device,
                feature_dim=args.num_heads * args.v_head_dim,
            ),
        ),
        PayloadCase(
            name="Lse",
            buffer_name="lse_buffer",
            feature_dim=args.num_heads,
            is_transpose=True,
            builder=lambda buffer, rank, world_size, max_num_seqs, num_requests, dtype, device: build_lse_case(
                buffer,
                rank,
                world_size,
                max_num_seqs,
                num_requests,
                dtype,
                device,
                feature_dim=args.num_heads,
            ),
        ),
    ]

    results: dict[str, dict[str, torch.Tensor]] = {}
    for case in payload_cases:
        results[case.name] = run_payload_case(
            case=case,
            rank=rank,
            world_size=world_size,
            max_num_seqs=args.max_num_seqs,
            num_requests=args.num_requests,
            dtype=dtype,
            device=device,
            preamble=args.preamble,
            mode=args.mode,
            warmup=args.warmup,
            graph_replays=args.graph_replays,
            group=group,
        )
        log_once(f"[{backend}] {case.name} passed in mode={args.mode}")

    return results


def compare_backend_results(
    *,
    reference_backend: SPBackend,
    candidate_backend: SPBackend,
    reference_results: dict[str, dict[str, torch.Tensor]],
    candidate_results: dict[str, dict[str, torch.Tensor]],
    mode: Literal["eager", "graph", "both"],
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    modes = ("eager", "graph") if mode == "both" else (mode,)
    for payload_name in ("Q", "Res", "Lse"):
        for mode_name in modes:
            assert_tensors_equal(
                name=f"{payload_name}/{mode_name}/{reference_backend}_vs_{candidate_backend}",
                actual=reference_results[payload_name][mode_name],
                expected=candidate_results[payload_name][mode_name],
                group=group,
                device=device,
            )
            log_once(
                f"[compare] {payload_name} {mode_name}: "
                f"{reference_backend} == {candidate_backend}"
            )


def main() -> None:
    args = parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive.")
    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive.")
    if args.num_requests > args.max_num_seqs:
        raise ValueError("--num-requests must be <= --max-num-seqs.")
    if args.graph_replays <= 0:
        raise ValueError("--graph-replays must be positive.")
    if (
        "nccl_compact" in {args.reference_backend, args.candidate_backend}
        and args.mode != "eager"
    ):
        raise ValueError(
            "nccl_compact uses variable split-size NCCL collectives and only "
            "supports --mode eager in this buffer-level correctness test. "
            "End-to-end piecewise CUDA Graph keeps these collectives outside graph capture."
        )

    rank = -1
    try:
        rank, world_size, device = init_dist()
        group = dist.group.WORLD
        install_test_dist_context(group)
        dtype = get_dtype(args.dtype)

        if args.num_requests < world_size:
            raise ValueError(
                f"--num-requests must be >= world_size for this Q-offsets test; "
                f"got num_requests={args.num_requests}, world_size={world_size}."
            )

        log_once(
            "Running MLA SP backend correctness check "
            f"(world_size={world_size}, dtype={args.dtype}, mode={args.mode}, "
            f"max_num_seqs={args.max_num_seqs}, num_requests={args.num_requests}, "
            f"preamble={args.preamble})"
        )

        reference_results = run_backend_suite(
            backend=args.reference_backend,
            rank=rank,
            world_size=world_size,
            args=args,
            dtype=dtype,
            device=device,
            group=group,
        )
        candidate_results = run_backend_suite(
            backend=args.candidate_backend,
            rank=rank,
            world_size=world_size,
            args=args,
            dtype=dtype,
            device=device,
            group=group,
        )

        compare_backend_results(
            reference_backend=args.reference_backend,
            candidate_backend=args.candidate_backend,
            reference_results=reference_results,
            candidate_results=candidate_results,
            mode=args.mode,
            group=group,
            device=device,
        )

        log_once("All MLA SP all-to-all backend checks passed.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank in (-1, MASTER_RANK):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

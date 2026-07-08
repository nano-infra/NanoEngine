"""LoongServe no-migration decode metadata SP communication CUDA graph smoke.

Run with torchrun, for example:

    torchrun --nproc_per_node=2 tests/test_loongserve_decode_sp_comm_graph.py --mode graph

This script does not instantiate a full model. It verifies that the metadata
emitted by the C++ LoongServe decode scheduler/prepare_decode_cpp path can drive
the same Q/Res/LSE all-to-all operators used by decode attention under eager and
CUDA graph replay.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

import torch
import torch.distributed as dist

sys.meta_path = [
    finder
    for finder in sys.meta_path
    if type(finder).__module__ != "_editable_skbc_nanodeploy"
]

from nanodeploy._cpp import (
    BlockContextSlot,
    Scheduler,
    Sequence,
    SequenceStatus,
    prepare_decode_cpp,
)
from nanodeploy.worker.sp_backend import SPBackend
from nanodeploy.worker.sp_context import get_sp_context, set_sp_context

from test_mla_sp_backend_correctness import (
    assert_tensors_equal,
    create_preamble_state,
    init_dist,
    install_test_dist_context,
    local_buffer_view,
    log_once,
    run_cudagraph,
    run_eager,
)


MODE_CHOICES = ("eager", "graph", "both")
BACKEND_CHOICES = ("legacy_ll", "hao_basic", "nccl")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate LoongServe no-migration decode metadata against SP "
            "Q/Res/LSE communication operators under CUDA graph replay."
        )
    )
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="nccl")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="both")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--min-batch", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--v-head-dim", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--graph-replays", type=int, default=2)
    parser.add_argument(
        "--preamble",
        choices=("none", "all_reduce", "all_gather"),
        default="all_reduce",
    )
    return parser.parse_args()


def make_scheduler(
    *,
    attention_sp: int,
    max_num_seqs: int,
    block_size: int,
    min_batch: int,
) -> Scheduler:
    return Scheduler(
        "",
        1,
        max_num_seqs,
        1024,
        max_num_seqs,
        -1,
        1,
        attention_sp,
        256,
        block_size,
        "decode",
        0.0,
        block_size,
        False,
        False,
        "legacy",
        100000,
        0,
        False,
        "",
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1,
        1,
        1,
        False,
        "RoundRobin",
        False,
        0,
        "centralized",
        True,
        False,
        "block",
        min_batch,
    )


def make_running_seq(seq_idx: int, attention_sp: int, block_size: int) -> Sequence:
    token_ids = [seq_idx * 100 + token_idx for token_idx in range(block_size)]
    seq = Sequence(token_ids, 1.0, 32, False)
    seq.active("", attention_sp, 1)
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    ctx.dp_idx = 0
    ctx.master_sp_idx = 0
    ctx.num_dispatched_tokens = [
        len(token_ids) if sp_idx == 0 else 0 for sp_idx in range(attention_sp)
    ]
    seq.status = SequenceStatus.RUNNING
    return seq


def token_ids_for(result):
    return [
        [[1000 + sp_idx * 100 + seq_idx] for seq_idx, _seq in enumerate(sp_seqs)]
        for sp_idx, sp_seqs in enumerate(result.filtered_dp_sp_seqs)
    ]


def build_loongserve_decode_batch(
    *,
    world_size: int,
    max_num_seqs: int,
    block_size: int,
    min_batch: int,
) -> list[Sequence]:
    scheduler = make_scheduler(
        attention_sp=world_size,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        min_batch=min_batch,
    )
    worker = scheduler.worker_state[0]
    num_seqs = min(max_num_seqs, world_size * min_batch)
    seqs = [make_running_seq(idx, world_size, block_size) for idx in range(num_seqs)]
    for seq in seqs:
        worker.allocate(seq)
        worker.running.append(seq)

    first = scheduler.schedule()
    append_targets = {
        seq.block_ctx(BlockContextSlot.ACTIVE).append_sp_idx
        if seq.block_ctx(BlockContextSlot.ACTIVE).append_sp_idx >= 0
        else seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        for seq in seqs
    }
    if len(append_targets) < min(world_size, 2):
        raise AssertionError(f"LoongServe scale-up did not add append targets: {append_targets}")

    scheduler.postprocess(first.filtered_dp_sp_seqs, token_ids_for(first), False, 0.0, 1)
    second = scheduler.schedule()
    masters = {
        seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        for seq in second.dp_seqs[0]
        if seq in seqs
    }
    if len(masters) < min(world_size, 2):
        raise AssertionError(f"LoongServe postprocess did not move masters: {masters}")
    return second.dp_seqs[0]


def grouped_by_master(dp_seqs: list[Sequence], world_size: int) -> list[list[Sequence]]:
    groups = [[] for _ in range(world_size)]
    for seq in dp_seqs:
        master = seq.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        if 0 <= master < world_size:
            groups[master].append(seq)
    return groups


def context_len(seq: Sequence, sp_rank: int) -> int:
    tokens = seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens
    if sp_rank >= len(tokens):
        return 0
    return int(tokens[sp_rank])


def q_value(seq: Sequence) -> float:
    return 1.0 + float(seq.last_token)


def partial_value(
    *,
    owner_rank: int,
    master_rank: int,
    seq_id: int,
    base: float,
) -> float:
    return base + float(owner_rank * 100 + master_rank * 10 + seq_id)


def metadata_tensors(meta, world_size: int, max_num_seqs: int, device: torch.device):
    context_lens = torch.tensor(
        meta.context_lens_flat, dtype=torch.int32, device=device
    ).view(world_size, max_num_seqs)
    global_context_lens = torch.tensor(
        meta.global_context_lens_flat, dtype=torch.int32, device=device
    ).view(world_size, max_num_seqs)
    q_mask = global_context_lens.clone()
    res_lse_mask = context_lens.clone()
    return context_lens, global_context_lens, q_mask, res_lse_mask


def build_q_payload(
    *,
    buffer,
    meta,
    dp_seqs: list[Sequence],
    rank: int,
    world_size: int,
    max_num_seqs: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    _context_lens, _global_context_lens, q_mask, _res_lse_mask = metadata_tensors(
        meta, world_size, max_num_seqs, device
    )
    q_mask[rank].fill_(0)
    q_mask[q_mask != 0] = 1
    q_offsets = torch.tensor(meta.q_offsets, dtype=torch.int32, device=device)

    x = torch.zeros((len(meta.input_ids), feature_dim), dtype=dtype, device=device)
    input_to_value = {int(token_id): 1.0 + float(token_id) for token_id in meta.input_ids}
    for row_idx, token_id in enumerate(meta.input_ids):
        x[row_idx].fill_(input_to_value[int(token_id)])

    buffer.local_buffer.zero_()
    local_flat = local_buffer_view(
        buffer,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    ).view(world_size * max_num_seqs, feature_dim)
    for get_idx, fill_idx, mask in zip(meta.q_slice_get, meta.q_slice_fill, meta.q_copy_mask):
        if mask:
            local_flat[int(fill_idx)].copy_(x[int(get_idx)])

    expected = torch.zeros((world_size, max_num_seqs, feature_dim), dtype=dtype, device=device)
    expected_flat = expected.view(world_size * max_num_seqs, feature_dim)
    groups = grouped_by_master(dp_seqs, world_size)
    for master_rank in range(world_size):
        write_pos = int(meta.q_offsets[master_rank])
        for seq in groups[master_rank]:
            if context_len(seq, rank) <= 0:
                continue
            expected_flat[write_pos].fill_(q_value(seq))
            write_pos += 1

    return x, q_mask, q_offsets, expected


def fill_local_partials(
    *,
    buffer,
    meta,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    feature_dim: int,
    dtype: torch.dtype,
    base: float,
) -> None:
    buffer.local_buffer.zero_()
    local = local_buffer_view(
        buffer,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
    )
    for fill_idx, mask in zip(
        meta.res_slice_fill_to_buffer_output,
        meta.res_to_buffer_output_mask,
    ):
        if not mask:
            continue
        target_rank = int(fill_idx) // max_num_seqs
        seq_id = int(fill_idx) % max_num_seqs
        local[target_rank, seq_id].fill_(
            partial_value(
                owner_rank=rank,
                master_rank=target_rank,
                seq_id=seq_id,
                base=base,
            )
        )


def build_reduce_payload(
    *,
    buffer,
    meta,
    dp_seqs: list[Sequence],
    rank: int,
    world_size: int,
    max_num_seqs: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    base: float,
):
    context_lens, _global_context_lens, _q_mask, res_lse_mask = metadata_tensors(
        meta, world_size, max_num_seqs, device
    )
    res_lse_mask = context_lens.clone()
    res_lse_mask[rank].fill_(0)
    res_lse_mask[res_lse_mask != 0] = 1

    fill_local_partials(
        buffer=buffer,
        meta=meta,
        rank=rank,
        world_size=world_size,
        max_num_seqs=max_num_seqs,
        feature_dim=feature_dim,
        dtype=dtype,
        base=base,
    )

    x = torch.zeros(
        (world_size * max_num_seqs, feature_dim), dtype=dtype, device=device
    )
    for fill_idx, mask in zip(
        meta.res_slice_fill_to_buffer_input,
        meta.res_to_buffer_input_mask,
    ):
        if not mask:
            continue
        target_rank = int(fill_idx) // max_num_seqs
        seq_id = int(fill_idx) % max_num_seqs
        x[int(fill_idx)].fill_(
            partial_value(
                owner_rank=rank,
                master_rank=target_rank,
                seq_id=seq_id,
                base=base,
            )
        )

    expected = torch.zeros((world_size, max_num_seqs, feature_dim), dtype=dtype, device=device)
    groups = grouped_by_master(dp_seqs, world_size)
    for seq_id, seq in enumerate(groups[rank]):
        for owner_rank in range(world_size):
            if context_len(seq, owner_rank) <= 0:
                continue
            expected[owner_rank, seq_id].fill_(
                partial_value(
                    owner_rank=owner_rank,
                    master_rank=rank,
                    seq_id=seq_id,
                    base=base,
                )
            )

    return x, res_lse_mask, None, expected


def run_payload(
    *,
    name: str,
    buffer,
    x: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor | None,
    expected: torch.Tensor,
    is_transpose: bool,
    mode: Literal["eager", "graph", "both"],
    warmup: int,
    graph_replays: int,
    preamble: str,
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    preamble_state = create_preamble_state(preamble, group, device)
    outputs: dict[str, torch.Tensor] = {}

    if mode in {"eager", "both"}:
        eager = run_eager(
            buffer=buffer,
            x=x,
            mask=mask,
            offsets=offsets,
            is_transpose=is_transpose,
            preamble_state=preamble_state,
            group=group,
            device=device,
        ).view_as(expected)
        assert_tensors_equal(
            name=f"{name}/eager_vs_expected",
            actual=eager,
            expected=expected,
            group=group,
            device=device,
        )
        outputs["eager"] = eager

    if mode in {"graph", "both"}:
        graph = run_cudagraph(
            buffer=buffer,
            x=x,
            mask=mask,
            offsets=offsets,
            is_transpose=is_transpose,
            warmup=warmup,
            replays=graph_replays,
            preamble_state=preamble_state,
            group=group,
            device=device,
        ).view_as(expected)
        assert_tensors_equal(
            name=f"{name}/graph_vs_expected",
            actual=graph,
            expected=expected,
            group=group,
            device=device,
        )
        outputs["graph"] = graph

    if mode == "both":
        assert_tensors_equal(
            name=f"{name}/eager_vs_graph",
            actual=outputs["eager"],
            expected=outputs["graph"],
            group=group,
            device=device,
        )


def main() -> None:
    args = parse_args()
    rank = -1
    try:
        rank, world_size, device = init_dist()
        if world_size < 2:
            raise RuntimeError("This smoke requires at least 2 SP ranks.")
        if args.max_num_seqs < world_size:
            raise ValueError("--max-num-seqs must be >= world_size.")

        group = dist.group.WORLD
        install_test_dist_context(group)
        dtype = torch.bfloat16
        set_sp_context(
            max_num_seqs=args.max_num_seqs,
            head_size=args.head_dim,
            num_attention_heads=args.num_heads,
            dtype=dtype,
            rank=rank,
            sp_size=world_size,
            backend=args.backend,  # type: ignore[arg-type]
        )

        dp_seqs = build_loongserve_decode_batch(
            world_size=world_size,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            min_batch=args.min_batch,
        )
        meta = prepare_decode_cpp(
            dp_seqs,
            rank,
            world_size,
            args.block_size,
            args.max_num_seqs,
        )
        if not meta.use_sp_a2a:
            raise AssertionError("LoongServe scale-up metadata did not enable SP A2A.")

        ctx = get_sp_context()
        q_dim = args.num_heads * args.head_dim
        res_dim = args.num_heads * args.v_head_dim
        lse_dim = args.num_heads

        q_payload = build_q_payload(
            buffer=ctx.q_buffer,
            meta=meta,
            dp_seqs=dp_seqs,
            rank=rank,
            world_size=world_size,
            max_num_seqs=args.max_num_seqs,
            feature_dim=q_dim,
            dtype=dtype,
            device=device,
        )
        run_payload(
            name="LoongServe/Q",
            buffer=ctx.q_buffer,
            x=q_payload[0],
            mask=q_payload[1],
            offsets=q_payload[2],
            expected=q_payload[3],
            is_transpose=False,
            mode=args.mode,  # type: ignore[arg-type]
            warmup=args.warmup,
            graph_replays=args.graph_replays,
            preamble=args.preamble,
            group=group,
            device=device,
        )
        log_once(f"[{args.backend}] LoongServe Q metadata passed in mode={args.mode}")

        res_payload = build_reduce_payload(
            buffer=ctx.res_buffer,
            meta=meta,
            dp_seqs=dp_seqs,
            rank=rank,
            world_size=world_size,
            max_num_seqs=args.max_num_seqs,
            feature_dim=res_dim,
            dtype=dtype,
            device=device,
            base=16.0,
        )
        run_payload(
            name="LoongServe/Res",
            buffer=ctx.res_buffer,
            x=res_payload[0],
            mask=res_payload[1],
            offsets=res_payload[2],
            expected=res_payload[3],
            is_transpose=True,
            mode=args.mode,  # type: ignore[arg-type]
            warmup=args.warmup,
            graph_replays=args.graph_replays,
            preamble=args.preamble,
            group=group,
            device=device,
        )
        log_once(f"[{args.backend}] LoongServe Res metadata passed in mode={args.mode}")

        lse_payload = build_reduce_payload(
            buffer=ctx.lse_buffer,
            meta=meta,
            dp_seqs=dp_seqs,
            rank=rank,
            world_size=world_size,
            max_num_seqs=args.max_num_seqs,
            feature_dim=lse_dim,
            dtype=dtype,
            device=device,
            base=64.0,
        )
        run_payload(
            name="LoongServe/Lse",
            buffer=ctx.lse_buffer,
            x=lse_payload[0],
            mask=lse_payload[1],
            offsets=lse_payload[2],
            expected=lse_payload[3],
            is_transpose=True,
            mode=args.mode,  # type: ignore[arg-type]
            warmup=args.warmup,
            graph_replays=args.graph_replays,
            preamble=args.preamble,
            group=group,
            device=device,
        )
        log_once(f"[{args.backend}] LoongServe Lse metadata passed in mode={args.mode}")
        log_once("LoongServe decode SP communication graph smoke passed.")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

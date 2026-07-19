"""Manual NCCL preflight for the complete LS KV scale-down transaction.

Examples:
    CUDA_VISIBLE_DEVICES=0,1 python tests/ls_kv_scale_down_nccl_preflight.py --sp-size 2
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python tests/ls_kv_scale_down_nccl_preflight.py --sp-size 8
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python tests/ls_kv_scale_down_nccl_preflight.py --sp-size 8 --transport-only
"""

from __future__ import annotations

import argparse
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence
from nanodeploy.worker.kv_p2p import KVCacheP2PMove, KVCacheP2PTransport


_BLOCK_SIZE = 4
_NUM_BLOCKS = 16


def _make_scheduler(sp_size: int) -> Scheduler:
    return Scheduler(
        "nccl-scale-down-preflight",
        1,
        16,
        4096,
        16,
        -1,
        1,
        sp_size,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        "decode",
        0.0,
        _BLOCK_SIZE,
        False,
        False,
        "legacy",
        100_000,
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
        True,
        sp_size,
        64,
        True,
        "centralized",
    )


def _worker(
    rank: int,
    sp_size: int,
    source_rank: int,
    init_method: str,
    moves: list[KVCacheP2PMove],
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=sp_size,
        init_method=init_method,
        device_id=device,
    )

    shape = (2, 2, _NUM_BLOCKS, _BLOCK_SIZE, 1, 2)
    values_per_rank = int(torch.tensor(shape).prod().item())
    cache = torch.arange(values_per_rank, dtype=torch.float32, device=device).reshape(
        shape
    )
    cache += rank * values_per_rank
    source_before = cache.clone() if rank == source_rank else None

    result = KVCacheP2PTransport(cache, group=dist.group.WORLD, chunk_tokens=2).execute(
        moves, current_dp_idx=0
    )
    expected_role = (
        "source" if rank == source_rank else "destination" if rank == 0 else "idle"
    )
    assert result.role == expected_role

    if rank == 0:
        expected_source = torch.arange(
            values_per_rank, dtype=torch.float32, device=device
        ).reshape(shape)
        expected_source += source_rank * values_per_rank
        for move in moves:
            actual = cache[
                :,
                :,
                move.dst_block_id,
                move.dst_token_offset : move.dst_token_offset + move.num_tokens,
                :,
                :,
            ]
            expected = expected_source[
                :,
                :,
                move.src_block_id,
                move.src_token_offset : move.src_token_offset + move.num_tokens,
                :,
                :,
            ]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif rank == source_rank:
        torch.testing.assert_close(cache, source_before, rtol=0, atol=0)

    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def transport_only(sp_size: int) -> None:
    source_rank = sp_size - 1
    moves = [
        KVCacheP2PMove(
            dp_idx=0,
            src_sp_rank=source_rank,
            dst_sp_rank=0,
            src_block_id=0,
            src_token_offset=0,
            dst_block_id=1,
            dst_token_offset=0,
            num_tokens=_BLOCK_SIZE,
        )
    ]
    mp.spawn(
        _worker,
        args=(sp_size, source_rank, f"tcp://127.0.0.1:{_free_port()}", moves),
        nprocs=sp_size,
        join=True,
    )


def main(sp_size: int) -> None:
    source_rank = sp_size - 1
    scheduler = _make_scheduler(sp_size)
    sequence = Sequence(list(range(sp_size)), 1.0, 32, True)
    scheduler.add(sequence)
    admission = scheduler.schedule()

    state = scheduler.worker_state[0]
    assert state.may_append(sequence, 1)
    sequence.append_token(99, BlockContextSlot.ACTIVE)
    sequence.mark_last_token_pending(BlockContextSlot.ACTIVE)
    state.add_running_tokens(sequence.block_ctx().master_sp_idx, 1)

    group_ids = {
        record.group_id_after_commit
        for record in admission.ls_admission_records
        if not record.bootstrap_finished
    }
    if len(group_ids) != 1 or None in group_ids:
        raise RuntimeError("preflight admission did not publish one typed survivor group")
    group_id = group_ids.pop()
    assert scheduler.get_ls_group_allocated_ranks(group_id) == list(range(sp_size))
    plan = scheduler.plan_ls_kv_scale_down(group_id, source_rank)
    assert plan.success, plan.failure_reason
    moves = [
        KVCacheP2PMove(
            move.dp_idx,
            move.src_sp_rank,
            move.dst_sp_rank,
            move.src_block_id,
            move.src_token_offset,
            move.dst_block_id,
            move.dst_token_offset,
            move.num_tokens,
        )
        for move in plan.moves
    ]

    assert scheduler.mark_ls_kv_scale_down_dispatched(plan)
    mp.spawn(
        _worker,
        args=(
            sp_size,
            source_rank,
            f"tcp://127.0.0.1:{_free_port()}",
            moves,
        ),
        nprocs=sp_size,
        join=True,
    )

    assert scheduler.commit_ls_kv_scale_down(plan)
    retained = list(range(source_rank))
    assert scheduler.get_ls_group_allocated_ranks(group_id) == retained
    expected_committed = [2, *([1] * (sp_size - 2)), 0]
    assert [
        sequence.committed_context_len(BlockContextSlot.ACTIVE, rank)
        for rank in range(sp_size)
    ] == expected_committed
    decode = scheduler.schedule()
    assert decode.ls_group_rank_allocations == [retained]
    assert decode.ls_kv_dops == [sp_size - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-size", type=int, choices=(2, 8), default=2)
    parser.add_argument("--transport-only", action="store_true")
    args = parser.parse_args()
    if args.transport_only:
        transport_only(args.sp_size)
    else:
        main(args.sp_size)

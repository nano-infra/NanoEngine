"""Validate DLSlime's exact fixed-SP8, eight-request decode traffic.

This is a DLSlime-only regression test.  It does not use NanoDeploy's NCCL SP
backend as a reference.  Instead, each payload carries an analytic rank/request
tag and the received tensor is checked against the expected routing result.

The traffic mirrors ``pd_disagg_deepseek_v3_parallel.py`` with:

- decode attention DP1/SP8/TP1
- ``fixed_sp_size=8``
- ``max_num_seqs=8``
- eight requests assigned RoundRobin, one master request per SP rank
- eager execution

The process group uses Gloo only for startup/barriers and DLSlime handle
exchange.  Q, Res, and LSE payloads all go through ``hao_basic``.

Run manually after configuring the required SLIME environment:

``torchrun --nproc_per_node=8 tests/test_dlslime_fixed_sp_batch_correctness.py``
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from nanodeploy.worker import distributed as distributed_mod
from nanodeploy.worker.sp_context import get_sp_context, set_sp_context


_SP_SIZE = 8
_MAX_NUM_SEQS = 8
_NUM_HEADS = 128
_Q_HEAD_DIM = 576
_V_HEAD_DIM = 512
_DTYPE = torch.bfloat16


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


def _init_dist() -> tuple[int, torch.device, dist.ProcessGroup]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DLSlime SP regression test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        # Gloo is control-plane-only here.  The payload transport under test is
        # exclusively DLSlime's hao_basic AllToAllBuffer.
        dist.init_process_group(backend="gloo")
    group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    if world_size != _SP_SIZE:
        raise RuntimeError(
            f"This regression mirrors fixed SP8 and requires 8 ranks, got {world_size}"
        )

    distributed_mod._DIST_CONTEXT = _TestDistContext(group=group)
    return dist.get_rank(group=group), device, group


def _local_buffer_view(
    buffer,
    *,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = _SP_SIZE * _MAX_NUM_SEQS * feature_dim
    return buffer.local_buffer.view(dtype)[:numel].view(
        _SP_SIZE,
        _MAX_NUM_SEQS,
        feature_dim,
    )


def _assert_payload_equal(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
) -> None:
    local_ok = actual.shape == expected.shape and bool(torch.equal(actual, expected))
    status = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    if status.item() == 1:
        return

    summary = None
    if not local_ok:
        summary = {
            "payload": name,
            "rank": rank,
            "actual_shape": tuple(actual.shape),
            "expected_shape": tuple(expected.shape),
        }
        if actual.shape == expected.shape:
            mismatch = torch.ne(actual, expected).nonzero(as_tuple=False)[0]
            index = tuple(int(item) for item in mismatch.tolist())
            summary.update(
                {
                    "first_mismatch_index": index,
                    "actual": float(actual[index].float().item()),
                    "expected": float(expected[index].float().item()),
                    "max_diff": float(
                        (actual.float() - expected.float()).abs().max().item()
                    ),
                }
            )

    summaries = [None for _ in range(_SP_SIZE)]
    dist.all_gather_object(summaries, summary, group=group)
    first_failure = next(item for item in summaries if item is not None)
    raise AssertionError(f"DLSlime {name} routing mismatch: {first_failure}")


def _run_q_case(
    *,
    rank: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """Broadcast one master-local Q row from every rank to every participant."""

    feature_dim = _NUM_HEADS * _Q_HEAD_DIM
    buffer = get_sp_context().q_buffer
    local = _local_buffer_view(
        buffer,
        feature_dim=feature_dim,
        dtype=_DTYPE,
    ).view(_SP_SIZE * _MAX_NUM_SEQS, feature_dim)
    local.zero_()

    # With one request per master, prepare_decode_cpp produces q_offsets=0..8
    # and q_slice_fill=[rank].  The local buffer supplies the self contribution.
    local[rank].fill_(float(rank + 1))
    x = torch.full(
        (1, feature_dim),
        float(rank + 1),
        dtype=_DTYPE,
        device=device,
    )
    mask = torch.zeros(
        (_SP_SIZE, _MAX_NUM_SEQS), dtype=torch.int32, device=device
    )
    mask[:, 0] = 1
    mask[rank, 0] = 0
    offsets = torch.arange(_SP_SIZE + 1, dtype=torch.int32, device=device)

    dist.barrier(group=group)
    output = buffer.all_to_all_ll(x, mask=mask, offsets=offsets)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # Production slices the first attention_compute_bs=8 packed rows.
    actual = output.view(-1, feature_dim)[:_SP_SIZE]
    expected = torch.stack(
        [
            torch.full(
                (feature_dim,),
                float(master_rank + 1),
                dtype=_DTYPE,
                device=device,
            )
            for master_rank in range(_SP_SIZE)
        ]
    )
    _assert_payload_equal(
        name="Q/full_all_to_all",
        actual=actual,
        expected=expected,
        rank=rank,
        group=group,
    )


def _run_transpose_case(
    *,
    name: str,
    buffer,
    feature_dim: int,
    dtype: torch.dtype,
    value_base: int,
    rank: int,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """Return every participant's partial result to all eight masters."""

    local = _local_buffer_view(
        buffer,
        feature_dim=feature_dim,
        dtype=dtype,
    )
    local.zero_()
    x = torch.zeros(
        (_SP_SIZE * _MAX_NUM_SEQS, feature_dim),
        dtype=dtype,
        device=device,
    )
    mask = torch.zeros(
        (_SP_SIZE, _MAX_NUM_SEQS), dtype=torch.int32, device=device
    )

    def value(participant_rank: int, master_rank: int) -> float:
        # Values remain exactly representable for both BF16 Res and FP32 LSE.
        return float(value_base + participant_rank * _SP_SIZE + master_rank)

    for master_rank in range(_SP_SIZE):
        tagged_value = value(rank, master_rank)
        if master_rank == rank:
            # Local patch is already source-participant-major at the receiver.
            local[rank, 0].fill_(tagged_value)
        else:
            # The transpose input is destination-master-major.
            x[master_rank * _MAX_NUM_SEQS].fill_(tagged_value)
            mask[master_rank, 0] = 1

    dist.barrier(group=group)
    output = buffer.all_to_all_ll(x, mask=mask, is_transpose=True)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # On rank R, output[source_rank, slot=0] must contain that source's
    # contribution for the request mastered by R.
    actual = output.view(
        _SP_SIZE,
        _MAX_NUM_SEQS,
        feature_dim,
    )[:, 0]
    expected = torch.stack(
        [
            torch.full(
                (feature_dim,),
                value(participant_rank, rank),
                dtype=dtype,
                device=device,
            )
            for participant_rank in range(_SP_SIZE)
        ]
    )
    _assert_payload_equal(
        name=name,
        actual=actual,
        expected=expected,
        rank=rank,
        group=group,
    )


def main() -> None:
    rank = -1
    try:
        rank, device, group = _init_dist()
        set_sp_context(
            max_num_seqs=_MAX_NUM_SEQS,
            head_size=_Q_HEAD_DIM,
            num_attention_heads=_NUM_HEADS,
            dtype=_DTYPE,
            rank=rank,
            sp_size=_SP_SIZE,
            backend="hao_basic",
        )

        _run_q_case(rank=rank, device=device, group=group)
        _run_transpose_case(
            name="Res/full_all_to_all",
            buffer=get_sp_context().res_buffer,
            feature_dim=_NUM_HEADS * _V_HEAD_DIM,
            dtype=_DTYPE,
            value_base=16,
            rank=rank,
            device=device,
            group=group,
        )
        _run_transpose_case(
            name="LSE/full_all_to_all",
            buffer=get_sp_context().lse_buffer,
            feature_dim=_NUM_HEADS,
            dtype=torch.float32,
            value_base=128,
            rank=rank,
            device=device,
            group=group,
        )

        if rank == 0:
            print(
                "DLSlime fixed-SP8 eight-request Q/Res/LSE routing passed.",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

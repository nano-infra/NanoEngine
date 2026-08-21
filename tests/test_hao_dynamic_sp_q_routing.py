"""Exercise dynamic-SP Q routing and replay-varying Graph padding.

Run with:

``torchrun --nproc_per_node=8 -m pytest -q tests/test_hao_dynamic_sp_q_routing.py -s``
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from nanodeploy._cpp import BlockContextSlot, Scheduler, Sequence, prepare_decode_cpp
from nanodeploy.kernels.copy import (
    copy_batch_indexed_triton,
    zero_padded_rows_triton,
)
from nanodeploy.worker.sp_backend import HaoAllToAllBufferAdapter


_SP_SIZE = 8
_MAX_NUM_SEQS = 8
_BLOCK_SIZE = 64
_FEATURE_DIM = 128
_DTYPE = torch.float16
_SENTINEL = -999.0
_LONG_PROMPT_BUCKET_POLICY = (
    "1:1-512;5:513-768;6:769-1024;7:1025-1280;8:1281-4096"
)
_PROMPT_LENGTHS = (2048, 1792, 1536, 1280, 1024, 768, 512, 500)


def _make_scheduler() -> Scheduler:
    return Scheduler(
        "hao-dynamic-sp-q-routing",
        1,
        _MAX_NUM_SEQS,
        8192,
        16,
        -1,
        1,
        _SP_SIZE,
        4096,
        _BLOCK_SIZE,
        "decode",
        1.0,
        65536,
        "bucket",
        True,
        _LONG_PROMPT_BUCKET_POLICY,
        True,
        "LeastBatch",
        0,
    )


def _schedule_mixed_bucket_batch() -> tuple[Sequence, ...]:
    scheduler = _make_scheduler()
    for request_index, prompt_length in enumerate(_PROMPT_LENGTHS):
        handoff_length = prompt_length + 1
        token_base = 10_000 * (request_index + 1)
        sequence = Sequence(
            [token_base + offset for offset in range(handoff_length)],
            1e-5,
            128,
            True,
        )
        sequence.seq_id = 100 + request_index
        scheduler.add(sequence)

    migration = scheduler.schedule()
    assert migration.is_prefill
    decode = scheduler.schedule()
    assert not decode.is_prefill
    return tuple(decode.dp_seqs[0])


def _tag(source: int, local_idx: int) -> float:
    return float(1 + source * _MAX_NUM_SEQS + local_idx)


def _expected_receiver_rows(
    scheduled: tuple[Sequence, ...], receiver: int, device: torch.device
) -> torch.Tensor:
    sequences_by_master: list[list[Sequence]] = [[] for _ in range(_SP_SIZE)]
    for sequence in scheduled:
        master = sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
        sequences_by_master[master].append(sequence)

    values = []
    for source, source_sequences in enumerate(sequences_by_master):
        for local_idx, sequence in enumerate(source_sequences):
            dispatched = sequence.block_ctx(
                BlockContextSlot.ACTIVE
            ).num_dispatched_tokens
            if dispatched[receiver] > 0:
                values.append(_tag(source, local_idx))

    return torch.tensor(values, dtype=_DTYPE, device=device).unsqueeze(1).expand(
        -1, _FEATURE_DIM
    )


def _prepare_local_buffer(
    buffer: HaoAllToAllBufferAdapter,
    x: torch.Tensor,
    q_slice_get: torch.Tensor,
    q_slice_fill: torch.Tensor,
    q_copy_mask: torch.Tensor,
    *,
    reset: bool,
) -> torch.Tensor:
    local = buffer.local_buffer.view(_DTYPE)[:
        _SP_SIZE * _MAX_NUM_SEQS * _FEATURE_DIM
    ].view(_SP_SIZE * _MAX_NUM_SEQS, 1, _FEATURE_DIM)
    if reset:
        local.fill_(_SENTINEL)
    copy_batch_indexed_triton(
        x.view(x.size(0), 1, _FEATURE_DIM),
        local,
        q_slice_get,
        q_slice_fill,
        q_copy_mask,
    )
    return local


def _assert_output(
    output: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    *,
    graph_rows: int | None = None,
) -> None:
    flat = output.view(_SP_SIZE * _MAX_NUM_SEQS, _FEATURE_DIM)
    assert torch.equal(flat[: expected.size(0)], expected), (
        f"rank {rank} packed Q rows differ from the receiver oracle"
    )
    if graph_rows is None:
        assert torch.all(flat[expected.size(0) :] == _SENTINEL), (
            f"rank {rank} wrote outside the packed Q prefix"
        )
        return

    assert torch.all(flat[expected.size(0) : graph_rows] == 0), (
        f"rank {rank} did not initialize the Graph-only Q tail"
    )
    assert torch.all(flat[graph_rows:] == _SENTINEL), (
        f"rank {rank} wrote outside the captured attention rows"
    )


def test_dynamic_sp_destination_rows_eager_and_cudagraph():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the dynamic-SP Q routing test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != _SP_SIZE:
        raise RuntimeError(f"This test requires SP8, got {world_size} ranks")

    try:
        scheduled = _schedule_mixed_bucket_batch()
        metadata = prepare_decode_cpp(
            list(scheduled), rank, _SP_SIZE, _BLOCK_SIZE, _MAX_NUM_SEQS
        )
        local_sequences = [
            sequence
            for sequence in scheduled
            if sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx == rank
        ]
        x = torch.stack(
            [
                torch.full(
                    (_FEATURE_DIM,),
                    _tag(rank, local_idx),
                    dtype=_DTYPE,
                    device=device,
                )
                for local_idx in range(len(local_sequences))
            ]
        )
        dst_row_indices = torch.tensor(
            metadata.q_dst_row_indices_flat,
            dtype=torch.int32,
            device=device,
        ).view(_SP_SIZE, _MAX_NUM_SEQS)
        q_slice_get = torch.tensor(
            metadata.q_slice_get, dtype=torch.int32, device=device
        )
        q_slice_fill = torch.tensor(
            metadata.q_slice_fill, dtype=torch.int32, device=device
        )
        q_copy_mask = torch.tensor(
            metadata.q_copy_mask, dtype=torch.int32, device=device
        )
        expected = _expected_receiver_rows(scheduled, rank, device)
        expected_counts = [
            _expected_receiver_rows(scheduled, receiver, device).size(0)
            for receiver in range(_SP_SIZE)
        ]
        graph_rows = max(expected_counts)
        actual_rows = torch.tensor(
            expected.size(0), dtype=torch.int32, device=device
        )
        original_dst_row_indices = dst_row_indices.clone()

        buffer_size_bytes = (
            _SP_SIZE * _MAX_NUM_SEQS * _FEATURE_DIM * _DTYPE.itemsize
        )
        buffer = HaoAllToAllBufferAdapter(
            max_dispatch_per_msg=_SP_SIZE,
            max_bs=_MAX_NUM_SEQS,
            rank=rank,
            world_size=_SP_SIZE,
            buffer_size_bytes=buffer_size_bytes,
        )
        buffer.connect_full_mesh(dist.group.WORLD)

        _prepare_local_buffer(
            buffer,
            x,
            q_slice_get,
            q_slice_fill,
            q_copy_mask,
            reset=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        eager_output = buffer.all_to_all_ll(
            x,
            dst_row_indices=dst_row_indices,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        _assert_output(eager_output, expected, rank)

        graph = torch.cuda.CUDAGraph()
        for _ in range(3):
            _prepare_local_buffer(
                buffer,
                x,
                q_slice_get,
                q_slice_fill,
                q_copy_mask,
                reset=True,
            )
            torch.cuda.synchronize(device)
            dist.barrier()
            warmup_output = buffer.all_to_all_ll(
                x, dst_row_indices=dst_row_indices
            ).view(_SP_SIZE * _MAX_NUM_SEQS, _FEATURE_DIM)
            zero_padded_rows_triton(warmup_output, actual_rows, graph_rows)
            torch.cuda.synchronize(device)
            dist.barrier()

        _prepare_local_buffer(
            buffer,
            x,
            q_slice_get,
            q_slice_fill,
            q_copy_mask,
            reset=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        with torch.cuda.graph(graph):
            _prepare_local_buffer(
                buffer,
                x,
                q_slice_get,
                q_slice_fill,
                q_copy_mask,
                reset=False,
            )
            graph_output = buffer.all_to_all_ll(
                x,
                dst_row_indices=dst_row_indices,
            ).view(_SP_SIZE * _MAX_NUM_SEQS, _FEATURE_DIM)
            zero_padded_rows_triton(graph_output, actual_rows, graph_rows)
        torch.cuda.synchronize(device)
        dist.barrier()

        _prepare_local_buffer(
            buffer,
            x,
            q_slice_get,
            q_slice_fill,
            q_copy_mask,
            reset=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        _assert_output(graph_output, expected, rank, graph_rows=graph_rows)

        second_counts = [max(1, count - 1) for count in expected_counts]
        second_dst_row_indices = original_dst_row_indices.clone()
        for receiver, receiver_count in enumerate(second_counts):
            receiver_rows = second_dst_row_indices[receiver]
            receiver_rows[receiver_rows >= receiver_count] = -1
        dst_row_indices.copy_(second_dst_row_indices)
        actual_rows.fill_(second_counts[rank])

        _prepare_local_buffer(
            buffer,
            x,
            q_slice_get,
            q_slice_fill,
            q_copy_mask,
            reset=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        _assert_output(
            graph_output,
            expected[: second_counts[rank]],
            rank,
            graph_rows=graph_rows,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

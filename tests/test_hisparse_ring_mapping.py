import pytest
import torch
from dlengine.context.batch import set_batch_context
from dlengine.context.cache.hisparse import (
    initialize_hisparse_context,
    reset_hisparse_context,
)
from dlengine.kernel.jit.sgl.hisparse import build_ring_slot_mapping
from dlengine.layers.generic.attention import _hisparse_prefill_fresh_slot_mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HiSparse ring mapping requires CUDA"
)


def test_hisparse_device_buffer_size_is_per_sequence():
    reset_hisparse_context()
    context = initialize_hisparse_context(
        max_num_seqs=16, device="cuda", device_buffer_size=4096
    )
    assert context.tokens_per_seq == 4096
    assert context.tokens_per_seq * context.max_num_seqs == 65536


def _inputs():
    slots = torch.tensor([0, 2, 4, 1], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 9, 3, 17], dtype=torch.int64, device="cuda")
    output = torch.empty(4, dtype=torch.int32, device="cuda")
    num_real_reqs = torch.tensor([3], dtype=torch.int32, device="cuda")
    return slots, positions, output, num_real_reqs


def test_hisparse_ring_mapping_invalid_and_padded_rows():
    slots, positions, output, num_real_reqs = _inputs()
    build_ring_slot_mapping(
        slots, positions, output, num_real_reqs, max_num_seqs=4, tokens_per_seq=8
    )
    assert output.cpu().tolist() == [0, 17, -1, -1]


def test_hisparse_ring_mapping_cuda_graph_replay():
    slots, positions, output, num_real_reqs = _inputs()
    for _ in range(3):
        build_ring_slot_mapping(
            slots, positions, output, num_real_reqs, max_num_seqs=4, tokens_per_seq=8
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build_ring_slot_mapping(
            slots, positions, output, num_real_reqs, max_num_seqs=4, tokens_per_seq=8
        )

    slots.copy_(torch.tensor([3, 2, 1, 0], dtype=torch.int64, device="cuda"))
    positions.copy_(torch.tensor([8, 9, 10, 11], dtype=torch.int64, device="cuda"))
    num_real_reqs.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [24, 17, -1, -1]


def test_hisparse_prefill_mapping_resets_position_per_sequence():
    reset_hisparse_context()
    initialize_hisparse_context(max_num_seqs=4, device="cuda", device_buffer_size=32)
    set_batch_context(
        is_prefill=True,
        max_bs=4,
        hisparse_slots=torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
    )
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    mapping = _hisparse_prefill_fresh_slot_mapping(cu_seqlens, cu_seqlens)
    assert mapping.cpu().tolist() == [0, 1, 32, 33, 34]

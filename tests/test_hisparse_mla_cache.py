import pytest
import torch

from dlengine.runtime.context.cache.hisparse import (
    initialize_hisparse_context,
    initialize_mla_hisparse_cache,
    map_mla_output_slots,
    reset_hisparse_context,
    stage_mla_sparse_indices,
    update_mla_hisparse_slot_owners,
    writeback_mla_output_pages,
)
from dlengine.runtime.context.cache.mla import allocate_mla_kvcache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned cache requires CUDA")
def test_cpu_mla_cache_is_cuda_pinned_and_preserves_padding_stride():
    class Context:
        device = "cpu"
        is_fp8_kvcache = True
        num_hidden_layers = 1
        num_local_kvcache_blocks = 2
        block_size = 4
        num_local_kv_heads = 1
        _fp8_head_dim = 8
        dtype = torch.bfloat16

    context = Context()
    allocate_mla_kvcache(context)
    assert context.kv_cache.is_pinned()
    assert context.kv_cache.shape == (1, 1, 2, 4, 1, 8)
    assert context.kv_cache.stride(2) == 5 * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_stage_uses_fixed_per_request_slots():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(2, "cuda", 8)
    ctx.num_real_reqs.fill_(2)
    cold = (
        torch.arange(1 * 1 * 6 * 4, dtype=torch.float32)
        .reshape(1, 1, 6, 4, 1, 1)
        .pin_memory()
    )
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=2, device_buffer_size=8)

    indices = torch.tensor(
        [[4, 5, 12, -1], [0, 9, 10, -1]], dtype=torch.int32, device="cuda"
    )
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    output_slots = torch.tensor([6, 11], dtype=torch.int32, device="cuda")
    logical_indices = torch.tensor(
        [[0, 1, 2, -1], [0, 1, 2, -1]], dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    remapped, hot_outputs = stage_mla_sparse_indices(
        0, logical_indices, indices, slots, output_slots, seq_lens
    )

    torch.cuda.synchronize()
    assert remapped.cpu().tolist() == [[0, 1, 2, -1], [12, 13, 14, -1]]
    assert hot_outputs.cpu().tolist() == [3, 15]
    assert hot[0, 0, 0, 0].item() == cold[0, 0, 1, 0].item()
    assert hot[0, 0, 0, 1].item() == cold[0, 0, 1, 1].item()
    assert hot[0, 0, 0, 2].item() == cold[0, 0, 3, 0].item()
    assert hot[0, 0, 3, 0].item() == cold[0, 0, 0, 0].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_writeback_persists_generated_token():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(1, "cuda", 4)
    ctx.num_real_reqs.fill_(1)
    cold = torch.zeros(1, 1, 2, 4, 1, 1).pin_memory()
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=4)
    hot[0, 0, 1, 0] = 7

    writeback_mla_output_pages(
        0,
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([4], dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert cold[0, 0, 0, 2].item() == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_slot_reassignment_clears_residency():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(2, "cuda", 4)
    cold = torch.zeros(1, 2, 2, 4, 1, 1).pin_memory()
    initialize_mla_hisparse_cache(cold, max_num_seqs=2, device_buffer_size=4)
    ctx.resident_tokens[:, 0].fill_(1)

    update_mla_hisparse_slot_owners([0], [101])
    torch.cuda.synchronize()
    assert not ctx.resident_tokens[:, 0].any()

    ctx.resident_tokens[:, 0].fill_(1)
    update_mla_hisparse_slot_owners([0], [101])
    torch.cuda.synchronize()
    assert ctx.resident_tokens[:, 0].all()

    update_mla_hisparse_slot_owners([0], [202])
    torch.cuda.synchronize()
    assert not ctx.resident_tokens[:, 0].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_short_context_only_loads_newly_selected_tokens():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(1, "cuda", 4)
    ctx.num_real_reqs.fill_(1)
    cold = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4, 1, 1).pin_memory()
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=4)
    slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    output_slots = torch.tensor([2], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([3], dtype=torch.int32, device="cuda")

    stage_mla_sparse_indices(
        0,
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        slots,
        output_slots,
        seq_lens,
    )
    torch.cuda.synchronize()
    assert hot[0, 0, 0, 1].item() == 1

    cold[0, 0, 0, 1] = 99
    stage_mla_sparse_indices(
        0,
        torch.tensor([[1, 0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1, 0]], dtype=torch.int32, device="cuda"),
        slots,
        output_slots,
        seq_lens,
    )
    torch.cuda.synchronize()
    assert hot[0, 0, 0, 1].item() == 1
    assert hot[0, 0, 0, 0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_long_context_maps_current_token_to_output_page():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(1, "cuda", 4)
    ctx.num_real_reqs.fill_(1)
    cold = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4, 1, 1).pin_memory()
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=4)
    hot[0, 0, 1, 0] = 77

    remapped, hot_outputs = stage_mla_sparse_indices(
        0,
        torch.tensor([[0, 4]], dtype=torch.int32, device="cuda"),
        torch.tensor([[0, 4]], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int64, device="cuda"),
        torch.tensor([4], dtype=torch.int32, device="cuda"),
        torch.tensor([5], dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert remapped.cpu().tolist() == [[0, 4]]
    assert hot_outputs.cpu().tolist() == [4]
    assert hot[0, 0, 1, 0].item() == 77


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_slot_loader_is_cuda_graph_capturable():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(1, "cuda", 4)
    ctx.num_real_reqs.fill_(1)
    cold = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4, 1, 1)
    cold = cold.pin_memory()
    initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=4)
    indices = torch.tensor([[4, 9]], dtype=torch.int32, device="cuda")
    slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    output_slots = torch.tensor([6], dtype=torch.int32, device="cuda")
    logical_indices = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([3], dtype=torch.int32, device="cuda")

    stage_mla_sparse_indices(0, logical_indices, indices, slots, output_slots, seq_lens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        remapped, hot_outputs = stage_mla_sparse_indices(
            0, logical_indices, indices, slots, output_slots, seq_lens
        )

    indices.copy_(torch.tensor([[0, 13]], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert remapped.cpu().tolist() == [[0, 1]]
    assert hot_outputs.cpu().tolist() == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_dummy_slots_produce_all_invalid_indices():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(2, "cuda", 4)
    ctx.num_real_reqs.fill_(2)
    cold = torch.zeros(1, 1, 2, 4, 1, 1).pin_memory()
    initialize_mla_hisparse_cache(cold, max_num_seqs=2, device_buffer_size=4)
    indices = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    dummy_slots = torch.full((2,), 2, dtype=torch.int64, device="cuda")
    output_slots = torch.zeros(2, dtype=torch.int32, device="cuda")
    logical_indices = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    seq_lens = torch.ones(2, dtype=torch.int32, device="cuda")

    remapped, hot_outputs = stage_mla_sparse_indices(
        0, logical_indices, indices, dummy_slots, output_slots, seq_lens
    )
    torch.cuda.synchronize()
    assert remapped.cpu().tolist() == [[-1] * 4, [-1] * 4]
    assert hot_outputs.cpu().tolist() == [-1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA HiSparse requires CUDA")
def test_mla_speculative_rows_share_union_and_keep_distinct_output_slots():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(1, "cuda", 6)
    ctx.num_real_reqs.fill_(1)
    cold = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4, 1, 1).pin_memory()
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=6)

    remapped, hot_outputs = stage_mla_sparse_indices(
        0,
        torch.tensor([[0, 1], [1, 6], [0, 8]], dtype=torch.int32, device="cuda"),
        torch.tensor([[0, 1], [1, 6], [0, 8]], dtype=torch.int32, device="cuda"),
        torch.zeros(3, dtype=torch.int64, device="cuda"),
        torch.tensor([6, 7, 8], dtype=torch.int32, device="cuda"),
        torch.full((3,), 9, dtype=torch.int32, device="cuda"),
        num_tokens_per_seq=3,
        phase_id=0,
    )

    torch.cuda.synchronize()
    mapped = remapped.cpu()
    assert mapped[0, 0] == mapped[2, 0]
    assert mapped[0, 1] == mapped[1, 0]
    assert 0 <= mapped[0, 0] < 6
    assert 0 <= mapped[0, 1] < 6
    assert mapped[1, 1] == 6
    assert mapped[2, 1] == 8
    assert hot_outputs.cpu().tolist() == [6, 7, 8]

    def hot_value(slot):
        slot = int(slot)
        return hot[0, 0, slot // 4, slot % 4, 0, 0].item()

    assert hot_value(mapped[0, 0]) == 0
    assert hot_value(mapped[0, 1]) == 1

    hot[0, 0, 1, 2, 0, 0] = 60
    hot[0, 0, 1, 3, 0, 0] = 70
    hot[0, 0, 2, 0, 0, 0] = 80
    writeback_mla_output_pages(
        0,
        torch.tensor([6, 7, 8], dtype=torch.int32, device="cuda"),
        hot_outputs,
        num_tokens_per_seq=3,
    )
    torch.cuda.synchronize()

    cold_flat = cold[0, 0].reshape(-1)
    assert cold_flat[6].item() == 60
    assert cold_flat[7].item() == 70
    assert cold_flat[8].item() == 80


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph")
def test_mla_hisparse_speculative_union_is_graph_safe():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(
        max_num_seqs=1,
        device=torch.device("cuda"),
        device_buffer_size=6,
    )
    ctx.num_real_reqs.fill_(1)
    cold = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4, 1, 1).pin_memory()
    hot = initialize_mla_hisparse_cache(cold, max_num_seqs=1, device_buffer_size=6)
    logical = torch.tensor([[0, 1], [1, 6], [0, 8]], dtype=torch.int32, device="cuda")
    physical = logical.clone()
    request_slots = torch.zeros(3, dtype=torch.int64, device="cuda")
    output_slots = torch.tensor([6, 7, 8], dtype=torch.int32, device="cuda")
    seq_lens = torch.full((3,), 9, dtype=torch.int32, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        remapped, hot_outputs = stage_mla_sparse_indices(
            0,
            logical,
            physical,
            request_slots,
            output_slots,
            seq_lens,
            num_tokens_per_seq=3,
            phase_id=0,
        )
        writeback_mla_output_pages(
            0,
            output_slots,
            hot_outputs,
            num_tokens_per_seq=3,
        )

    graph.replay()
    torch.cuda.synchronize()
    mapped = remapped.cpu()
    assert mapped[0, 0] == mapped[2, 0]
    assert mapped[0, 1] == mapped[1, 0]
    assert hot_outputs.cpu().tolist() == [6, 7, 8]

    ctx.num_real_reqs.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert remapped.cpu().tolist() == [[-1, -1], [-1, -1], [-1, -1]]
    assert hot_outputs.cpu().tolist() == [-1, -1, -1]
    assert hot.numel() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph")
def test_mla_cached_topk_output_mapping_is_graph_safe():
    reset_hisparse_context()
    ctx = initialize_hisparse_context(2, "cuda", 6)
    ctx.num_real_reqs.fill_(2)
    cold = torch.zeros(1, 1, 4, 4, 1, 1).pin_memory()
    initialize_mla_hisparse_cache(cold, max_num_seqs=2, device_buffer_size=6)
    slots = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([9, 9, 9, 3, 3, 3], dtype=torch.int32, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped = map_mla_output_slots(slots, seq_lens, num_tokens_per_seq=3)

    graph.replay()
    torch.cuda.synchronize()
    stride = ctx.tokens_per_seq
    assert mapped.cpu().tolist() == [6, 7, 8, stride, stride + 1, stride + 2]

    ctx.num_real_reqs.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert mapped.cpu().tolist() == [6, 7, 8, -1, -1, -1]

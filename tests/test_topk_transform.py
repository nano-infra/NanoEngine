import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the fused TopK transform requires a Hopper GPU",
)


@pytest.mark.parametrize("top_k", [512, 2048])
def test_topk_transform_matches_torch(top_k):
    pytest.importorskip("tvm_ffi")

    from dlengine.runtime.kernel.jit.sgl.deepseek_v4 import topk_transform

    torch.manual_seed(0)
    page_size = 64
    max_context_len = 16384
    seq_lens = torch.tensor(
        [min(64, top_k), max_context_len],
        dtype=torch.int32,
        device="cuda",
    )
    scores = torch.randn(
        (seq_lens.numel(), max_context_len), dtype=torch.float32, device="cuda"
    )
    page_tables = torch.arange(
        max_context_len // page_size, dtype=torch.int32, device="cuda"
    ).repeat(seq_lens.numel(), 1)
    output = torch.empty((seq_lens.numel(), top_k), dtype=torch.int32, device="cuda")
    raw_output = torch.empty_like(output)

    topk_transform(
        scores,
        seq_lens,
        page_tables,
        output,
        page_size,
        top_k,
        raw_output,
    )

    for row, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor)
        valid_k = min(seq_len, top_k)
        expected_raw = torch.topk(scores[row, :seq_len], valid_k).indices
        expected = (
            page_tables[row, expected_raw // page_size] * page_size
            + expected_raw % page_size
        ).to(torch.int32)

        torch.testing.assert_close(
            output[row, :valid_k].sort().values,
            expected.sort().values,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            raw_output[row, :valid_k].sort().values,
            expected_raw.to(torch.int32).sort().values,
            rtol=0,
            atol=0,
        )
        assert (output[row, valid_k:] == -1).all()
        assert (raw_output[row, valid_k:] == -1).all()


@torch.inference_mode()
def test_topk_transform_ragged_handles_long_multiseq_ranges():
    pytest.importorskip("tvm_ffi")

    from dlengine.runtime.kernel.jit.sgl.deepseek_v4 import topk_transform_ragged

    torch.manual_seed(7)
    width = 70_000
    lengths = torch.tensor([17, 4096, 65_536], dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([9, 123, 1000], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 17, 4113], dtype=torch.int32, device="cuda")
    scores = torch.randn((3, width), dtype=torch.float32, device="cuda")
    output = torch.empty((3, 2048), dtype=torch.int32, device="cuda")

    topk_transform_ragged(scores, lengths, row_starts, offsets, output, 2048)
    torch.cuda.synchronize()

    assert torch.equal(output[0, :17].sort().values, torch.arange(17, device="cuda"))
    assert (output[0, 17:] == -1).all()
    for row in (1, 2):
        length = int(lengths[row])
        start = int(row_starts[row])
        offset = int(offsets[row])
        expected = torch.topk(scores[row, start : start + length], 2048).indices
        actual = output[row] - offset
        # The bounded threshold candidate buffer permits up to five non-tie
        # differences; random inputs are normally exact.
        missing = set(expected.cpu().tolist()) - set(actual.cpu().tolist())
        extra = set(actual.cpu().tolist()) - set(expected.cpu().tolist())
        assert len(missing) == len(extra) <= 5


@torch.inference_mode()
def test_topk_transform_ragged_ties_stay_unique_and_in_range():
    pytest.importorskip("tvm_ffi")

    from dlengine.runtime.kernel.jit.sgl.deepseek_v4 import topk_transform_ragged

    length = 65_536
    row_start = 321
    offset = 777
    scores = torch.zeros((1, 70_000), dtype=torch.float32, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([row_start], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([offset], dtype=torch.int32, device="cuda")
    output = torch.empty((1, 2048), dtype=torch.int32, device="cuda")

    topk_transform_ragged(scores, lengths, row_starts, offsets, output, 2048)
    torch.cuda.synchronize()

    selected = output[0]
    assert int(selected.min()) >= offset
    assert int(selected.max()) < offset + length
    assert torch.unique(selected).numel() == 2048

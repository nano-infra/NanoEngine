import numpy as np
from bench.bench_decode_metadata import build_payload, HEADER, run_case


def test_decode_metadata_payload_layout_and_reference():
    payload = build_payload(
        batch_size=3,
        context_len=129,
        block_size=64,
        max_num_seqs=8,
        seed=7,
    )
    header = HEADER.unpack_from(payload.data)

    assert header[0] == 0x444D444C
    assert header[1] == 1
    assert header[3] == len(payload.data)
    assert header[4:8] == (3, 8, payload.max_num_blocks, 64)
    assert all(offset % 16 == 0 for offset in header[8:15])
    assert payload.expected["input_ids"].shape == (8,)
    assert payload.expected["block_tables"].shape == (
        8,
        payload.max_num_blocks,
    )
    assert np.all(payload.expected["context_lens"][:3] > 0)
    assert np.all(payload.expected["slot_mapping"][:3] >= 0)
    assert np.all(payload.expected["slot_mapping"][3:] == -1)


def test_decode_metadata_payload_is_reproducible():
    left = build_payload(8, 4096, 64, 16, seed=4)
    right = build_payload(8, 4096, 64, 16, seed=4)

    assert left.data == right.data
    for name in left.expected:
        np.testing.assert_array_equal(left.expected[name], right.expected[name])


def test_decode_metadata_cuda_paths_match():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    result = run_case(
        batch_size=3,
        context_len=129,
        block_size=64,
        max_num_seqs=8,
        warmup=1,
        iterations=2,
        variants=2,
    )

    assert set(result["paths"]) == {
        "mapped_uva",
        "aggregated_memcpy",
        "per_field_torch",
    }

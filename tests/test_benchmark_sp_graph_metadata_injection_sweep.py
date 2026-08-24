import argparse

import pytest
import torch

from scripts.benchmark_sp_graph_metadata_injection_sweep import (
    MnPair,
    _block_table_width,
    _captured_mn_pairs,
    _mn_pair,
    _shape_hint,
    _source_for_pair,
    _validate_mn_pairs,
    _verify_fused_result,
)


def _source() -> dict:
    return {
        "attention_sp": 8,
        "max_num_seqs": 192,
        "max_num_recv_seqs": 32,
        "max_model_len": 1_000_000,
        "kvcache_block_size": 64,
        "graph_master_rank_bs": [1, 8],
        "sp_graph_map": {"1": [1, 17, 33], "8": [8, 24, 40]},
        "shape_counts": [{"block_table_width": 1384}],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("64,80", MnPair(64, 80)),
        ("64x80", MnPair(64, 80)),
        ("64:80", MnPair(64, 80)),
    ],
)
def test_parse_mn_pair(value: str, expected: MnPair) -> None:
    assert _mn_pair(value) == expected


def test_parse_mn_pair_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _mn_pair("64")


def test_captured_pairs_preserve_graph_map_order() -> None:
    assert _captured_mn_pairs(_source()) == [
        MnPair(1, 1),
        MnPair(1, 17),
        MnPair(1, 33),
        MnPair(8, 8),
        MnPair(8, 24),
        MnPair(8, 40),
    ]


def test_no_padding_shape_uses_m_and_n_for_actual_and_graph_sizes() -> None:
    assert _shape_hint(MnPair(64, 80), 1384) == {
        "actual_master_bs": 64,
        "graph_master_bs": 64,
        "actual_attn_bs": 80,
        "graph_attn_bs": 80,
        "block_table_rows": 80,
        "block_table_width": 1384,
    }


def test_per_pair_capacity_right_sizes_master_and_remote_rows() -> None:
    source = _source()

    pair_source = _source_for_pair(source, MnPair(64, 80), "per-pair")

    assert pair_source["max_num_seqs"] == 64
    assert pair_source["max_num_recv_seqs"] == 16
    assert source["max_num_seqs"] == 192
    assert source["max_num_recv_seqs"] == 32


def test_fixed_capacity_reuses_configured_limits() -> None:
    source = _source()

    assert _source_for_pair(source, MnPair(8, 24), "fixed") is source


def test_verify_fused_result_reports_tensor_mismatches() -> None:
    reference = {
        "matching": torch.tensor([1, 2]),
        "different": torch.tensor([3, 4]),
        "optional": None,
    }
    fused = {
        "matching": torch.tensor([1, 2]),
        "different": torch.tensor([3, 5]),
        "optional": None,
    }

    result = _verify_fused_result(reference, fused)

    assert not result["passed"]
    assert result["tensors"]["matching"] == {
        "passed": True,
        "mismatch_count": 0,
    }
    assert result["tensors"]["different"] == {
        "passed": False,
        "mismatch_count": 1,
    }


def test_validate_pairs_enforces_remote_row_capacity() -> None:
    with pytest.raises(ValueError, match="n-m exceeds"):
        _validate_mn_pairs([MnPair(8, 41)], _source())


def test_default_block_table_width_matches_observed_shape() -> None:
    assert _block_table_width(_source(), None) == 1384

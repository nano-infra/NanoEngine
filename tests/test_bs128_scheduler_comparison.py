from __future__ import annotations

import pytest

from scripts.scheduler_overhead.profile_bs128_centralized_decentralized import (
    BS_PER_GPU,
    DEFAULT_SCENARIOS,
    SUPPORTED_LOGICAL_NODES,
    build_case_matrix,
    expected_result_cell_count,
    result_cells_for_comparison,
    validate_matrix,
)


def test_bs128_matrix_has_12_scheduler_cases_and_48_result_cells():
    cases = build_case_matrix()

    assert len(cases) == 12
    assert cases[0] == (1, DEFAULT_SCENARIOS[0])
    assert cases[-1] == (4, DEFAULT_SCENARIOS[-1])
    assert expected_result_cell_count(
        SUPPORTED_LOGICAL_NODES, DEFAULT_SCENARIOS
    ) == 48


def test_bs128_matrix_request_counts_follow_gpu_count():
    cases = build_case_matrix(logical_nodes=(1, 2, 4), scenarios=("no_sp",))

    assert [nodes * 8 * BS_PER_GPU for nodes, _ in cases] == [1024, 2048, 4096]


def test_matrix_rejects_partial_topology_and_non_bs128():
    with pytest.raises(ValueError, match="only complete production topologies"):
        validate_matrix((3,), ("no_sp",), BS_PER_GPU)
    validate_matrix(
        (32,),
        ("no_sp",),
        BS_PER_GPU,
        allow_modelled_topologies=True,
    )
    with pytest.raises(ValueError, match="fixed at BS/GPU=128"):
        validate_matrix((1,), ("no_sp",), 64)


def test_comparison_expands_to_admission_and_decode_cells():
    comparison = {
        "logical_nodes": 1,
        "logical_gpus": 8,
        "batch_size_per_gpu": 128,
        "total_requests": 1024,
        "scenario": "no_sp",
        "loop_count": 16,
        "centralized_admission_mean_ms": 1.0,
        "centralized_admission_p99_ms": 1.2,
        "decentralized_admission_mean_ms": 2.0,
        "decentralized_admission_p99_ms": 2.2,
        "centralized_decode_mean_ms": 3.0,
        "centralized_decode_p99_ms": 3.5,
        "centralized_decode_mean_ms_per_step": 3.0 / 16,
        "decentralized_decode_mean_ms": 4.0,
        "decentralized_decode_p99_ms": 4.5,
        "decentralized_decode_mean_ms_per_step": 4.0 / 16,
        "admission_result_passed": True,
        "decode_result_passed": True,
        "centralized_admission_samples": 10,
        "decentralized_admission_samples": 10,
        "centralized_decode_samples": 100,
        "decentralized_decode_samples": 100,
    }

    cells = result_cells_for_comparison(comparison)

    assert {(cell["architecture"], cell["phase"]) for cell in cells} == {
        ("centralized", "admission"),
        ("decentralized", "admission"),
        ("centralized", "decode"),
        ("decentralized", "decode"),
    }
    assert cells[-1]["mean_ms_per_step"] == pytest.approx(0.25)

from __future__ import annotations

import pytest
from nanodeploy._cpp import Scheduler

from scripts.scheduler_overhead.profile_bs128_mixed_admission_decode import (
    BS_PER_GPU,
    MIXED_LOGICAL_NODES,
    MIXED_SCENARIOS,
    FIG9_ADMISSION_HORIZON_S,
    FIG9_LOGICAL_NODES,
    FIG9_PRESET_NAME,
    FIG9_PRESET_POINTS,
    MixedProfileCase,
    build_case_matrix,
    build_fig9_case_matrix,
    run_case,
)
from scripts.scheduler_overhead.profile_hierarchical_scheduler_scalability import (
    DEFAULT_MODEL,
)


def test_mixed_matrix_covers_requested_nodes_and_dynamic_policies() -> None:
    cases = build_case_matrix()

    assert len(cases) == 8
    assert tuple(case.logical_nodes for case in cases[::2]) == (
        1,
        2,
        4,
        32,
    )
    assert {case.scenario_name for case in cases} == set(MIXED_SCENARIOS)
    assert set(MIXED_LOGICAL_NODES) == {1, 2, 4, 32}
    assert all(case.batch_size_per_gpu == BS_PER_GPU for case in cases)


@pytest.mark.parametrize(
    ("scenario", "expected_per_local", "target"),
    [
        ("dynamic_sp8_1pct", 31, 3 / 100),
        ("dynamic_sp8_5pct", 10, 1 / 100),
    ],
)
def test_admission_cohort_is_rounded_per_local_scheduler(
    scenario: str,
    expected_per_local: int,
    target: float,
) -> None:
    case = MixedProfileCase(scenario, logical_nodes=1)

    assert case.decode_requests == 1_024
    assert case.admission_requests_per_local_scheduler == expected_per_local
    assert case.admission_requests == expected_per_local
    assert case.admission_to_decode_ratio == pytest.approx(target, abs=0.001)
    assert case.total_requests == case.decode_requests + case.admission_requests


def test_mixed_admission_is_defined_relative_to_decode_baseline() -> None:
    one = MixedProfileCase("dynamic_sp8_1pct", logical_nodes=1)
    thirty_two = MixedProfileCase("dynamic_sp8_1pct", logical_nodes=32)

    assert one.admission_requests == 31
    assert thirty_two.admission_requests == 32 * 31
    assert thirty_two.admission_to_decode_ratio == pytest.approx(
        one.admission_to_decode_ratio
    )


def test_fig9_preset_defaults_to_4_and_32_nodes() -> None:
    cases = build_fig9_case_matrix()

    assert len(cases) == 10
    assert {case.logical_nodes for case in cases} == set(FIG9_LOGICAL_NODES)
    assert {case.preset_name for case in cases} == {FIG9_PRESET_NAME}
    assert {case.preset_point_name for case in cases} == {
        point.name for point in FIG9_PRESET_POINTS
    }


@pytest.mark.parametrize(
    ("point_name", "expected_bs", "expected_per_local"),
    [
        ("fig9_issue1_r80", 135, 32),
        ("fig9_issue1_r90", 173, 36),
        ("fig9_issue5_r40", 56, 16),
        ("fig9_issue5_r45", 58, 18),
        ("fig9_issue5_r50", 59, 20),
    ],
)
def test_fig9_admission_cohort_uses_16_loop_100ms_horizon(
    point_name: str,
    expected_bs: int,
    expected_per_local: int,
) -> None:
    case = next(
        case
        for case in build_fig9_case_matrix(logical_nodes=(4,))
        if case.preset_point_name == point_name
    )

    assert FIG9_ADMISSION_HORIZON_S == pytest.approx(1.6)
    assert case.batch_size_per_gpu == expected_bs
    assert case.admission_requests_per_local_scheduler == expected_per_local
    assert case.admission_requests_per_gpu == pytest.approx(
        expected_per_local / 8
    )
    assert case.admission_requests == expected_per_local * 4
    assert case.decode_requests == case.logical_gpus * expected_bs


def test_fig9_rate_scales_per_gpu_between_4_and_32_nodes() -> None:
    four = next(
        case
        for case in build_fig9_case_matrix(logical_nodes=(4,))
        if case.preset_point_name == "fig9_issue1_r90"
    )
    thirty_two = next(
        case
        for case in build_fig9_case_matrix(logical_nodes=(32,))
        if case.preset_point_name == "fig9_issue1_r90"
    )

    assert four.admission_requests == 144
    assert thirty_two.admission_requests == 1_152
    assert thirty_two.admission_requests == four.admission_requests * 8
    assert four.effective_global_admission_rate_rps == pytest.approx(90.0)
    assert thirty_two.effective_global_admission_rate_rps == pytest.approx(720.0)


@pytest.mark.skipif(
    not (DEFAULT_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found below {DEFAULT_MODEL}",
)
@pytest.mark.skipif(
    not hasattr(Scheduler, "commit_planned_sequences"),
    reason="requires rebuilt native bulk admission extension",
)
def test_small_mixed_case_transitions_baseline_then_injected_admission() -> None:
    # A tiny case exercises both native scheduler boundaries without starting
    # Ray/CUDA.  The production CLI remains fixed at BS/GPU=128.
    record = run_case(
        MixedProfileCase(
            "dynamic_sp8_1pct",
            logical_nodes=1,
            batch_size_per_gpu=1,
        ),
        model=str(DEFAULT_MODEL),
        warmup_iterations=0,
        measured_iterations=1,
        admission_iterations=1,
        admission_batch_size=4,
    )

    assert record["decode_requests"] == 8
    assert record["admission_requests"] == 1
    assert record["centralized"]["case_passed"]
    assert record["decentralized"]["case_passed"]
    assert record["centralized"]["admission_scheduler_ms"]["count"] == 1
    assert record["decentralized"]["admission"]["scheduler_critical_ms"][
        "count"
    ] == 1
    assert record["centralized"]["decode_scheduler_ms"]["count"] == 1
    assert record["decentralized"]["decode_scheduler_ms"]["count"] == 1

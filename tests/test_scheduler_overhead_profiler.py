from __future__ import annotations

import pytest

from scripts.scheduler_overhead.profile_scheduler_scalability import (
    SCENARIOS,
    ProfileCase,
    _long_request_indices,
    _percentile,
    run_case,
)


def _case(scenario: str) -> ProfileCase:
    return ProfileCase(
        scenario=SCENARIOS[scenario],
        logical_nodes=1,
        batch_size_per_gpu=2,
        short_context_len=64,
        long_context_len=512,
        block_size=64,
        loop_count=1,
        seed=0,
    )


def test_no_sp_and_sp8_topologies_preserve_total_workload() -> None:
    no_sp = _case("no_sp")
    fixed_sp8 = _case("fixed_sp8")

    assert (no_sp.attention_dp, no_sp.attention_sp) == (8, 1)
    assert (fixed_sp8.attention_dp, fixed_sp8.attention_sp) == (1, 8)
    assert no_sp.total_requests == fixed_sp8.total_requests == 16


@pytest.mark.parametrize(
    ("scenario", "expected_sp8"),
    [
        ("no_sp", 0),
        ("fixed_sp8", 16),
        ("dynamic_sp8_1pct", 0),
        ("dynamic_sp8_5pct", 1),
    ],
)
def test_expected_sp8_request_count_uses_requested_ratio(
    scenario: str,
    expected_sp8: int,
) -> None:
    assert _case(scenario).expected_sp8_requests == expected_sp8


def test_long_request_selection_is_exact_and_deterministic() -> None:
    first = _long_request_indices(100, 5, seed=7)
    second = _long_request_indices(100, 5, seed=7)

    assert first == second
    assert len(first) == 5


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 100.0) == 4.0


@pytest.mark.parametrize(
    "scenario",
    ["no_sp", "fixed_sp8", "dynamic_sp8_5pct"],
)
def test_run_case_profiles_steady_decode_and_validates_sp_mix(
    scenario: str,
) -> None:
    case = _case(scenario)
    record = run_case(
        case,
        warmup_iterations=1,
        measured_iterations=3,
    )

    assert record["scenario"] == scenario
    assert record["total_requests"] == 16
    assert record["measured_iterations"] == 3
    assert record["actual_sp8_requests"] == case.expected_sp8_requests
    assert record["actual_sp1_requests"] == (
        case.total_requests - case.expected_sp8_requests
    )
    assert 0.0 < record["min_ms"] <= record["p50_ms"] <= record["max_ms"]


def test_run_case_advances_decode_state_across_block_boundaries() -> None:
    record = run_case(
        _case("no_sp"),
        warmup_iterations=1,
        measured_iterations=70,
    )

    assert record["measured_iterations"] == 70
    assert record["actual_sp1_requests"] == 16

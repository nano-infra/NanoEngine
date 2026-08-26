from __future__ import annotations

import pytest

from scripts.scheduler_overhead.profile_hierarchical_scheduler_scalability import (
    DEFAULT_MODEL,
    HierarchicalProfileCase,
    _annotate_weak_scaling,
    run_case,
    scheduler_kv_blocks,
)
from scripts.scheduler_overhead.profile_scheduler_scalability import SCENARIOS


def _case(
    scenario: str,
    *,
    logical_nodes: int = 2,
    batch_size_per_gpu: int = 3,
) -> HierarchicalProfileCase:
    return HierarchicalProfileCase(
        scenario=SCENARIOS[scenario],
        logical_nodes=logical_nodes,
        batch_size_per_gpu=batch_size_per_gpu,
        short_context_len=1_024,
        long_context_len=428_033,
        block_size=64,
        seed=7,
    )


def test_hierarchical_workload_matches_legacy_total_request_count() -> None:
    no_sp = _case("no_sp")
    fixed_sp8 = _case("fixed_sp8")

    assert no_sp.logical_gpus == fixed_sp8.logical_gpus == 16
    assert no_sp.total_requests == fixed_sp8.total_requests == 48
    assert no_sp.local_scheduler_count == 16
    assert no_sp.requests_per_local_scheduler == 3
    assert fixed_sp8.local_scheduler_count == 2
    assert fixed_sp8.requests_per_local_scheduler == 24


def test_dynamic_long_requests_are_exactly_distributed_across_locals() -> None:
    case = _case(
        "dynamic_sp8_5pct",
        logical_nodes=4,
        batch_size_per_gpu=8,
    )

    counts = case.long_requests_per_local_scheduler()

    assert len(counts) == case.local_scheduler_count == 4
    assert sum(counts) == case.expected_sp8_requests == 13
    assert max(counts) - min(counts) <= 3


def test_hierarchical_capacity_covers_prompt_and_completion_growth() -> None:
    fixed = _case("fixed_sp8", logical_nodes=1, batch_size_per_gpu=2)
    dynamic = _case(
        "dynamic_sp8_5pct",
        logical_nodes=1,
        batch_size_per_gpu=3,
    )

    baseline = scheduler_kv_blocks(fixed, max_tokens=64)

    assert scheduler_kv_blocks(fixed, max_tokens=1_024) > baseline
    assert scheduler_kv_blocks(dynamic, max_tokens=64) > baseline


def test_logical_replica_scope_is_explicit_beyond_supported_topology() -> None:
    supported = _case("fixed_sp8", logical_nodes=4)
    modelled = _case("fixed_sp8", logical_nodes=8)

    assert supported.deployment_topology_supported
    assert supported.topology_scope == "complete_production_topology_cpu_model"
    assert not modelled.deployment_topology_supported
    assert modelled.topology_scope == (
        "logical_independent_local_scheduler_replica_model"
    )


def test_weak_scaling_annotation_uses_smallest_matching_node_count() -> None:
    records = [
        {
            "scenario": "fixed_sp8",
            "batch_size_per_gpu": 32,
            "logical_nodes": 2,
            "local_scheduler_count": 2,
            "modelled_aggregate_local_quantums_per_second": 180.0,
        },
        {
            "scenario": "fixed_sp8",
            "batch_size_per_gpu": 32,
            "logical_nodes": 1,
            "local_scheduler_count": 1,
            "modelled_aggregate_local_quantums_per_second": 100.0,
        },
    ]

    _annotate_weak_scaling(records)

    assert records[1]["weak_scaling_efficiency_vs_smallest_node"] == 1.0
    assert records[0]["weak_scaling_efficiency_vs_smallest_node"] == 0.9


@pytest.mark.skipif(
    not (DEFAULT_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found below {DEFAULT_MODEL}",
)
@pytest.mark.parametrize(
    ("scenario", "batch_size_per_gpu", "expected_sp8"),
    [
        ("no_sp", 1, 0),
        ("fixed_sp8", 1, 8),
        ("dynamic_sp8_5pct", 3, 1),
    ],
)
def test_run_case_profiles_production_hierarchical_contract(
    scenario: str,
    batch_size_per_gpu: int,
    expected_sp8: int,
) -> None:
    case = _case(
        scenario,
        logical_nodes=1,
        batch_size_per_gpu=batch_size_per_gpu,
    )

    record = run_case(
        case,
        model=str(DEFAULT_MODEL),
        warmup_iterations=0,
        measured_iterations=1,
        admission_iterations=1,
        admission_batch_size=4,
    )

    assert record["total_requests"] == case.total_requests
    assert record["actual_sp8_requests"] == expected_sp8
    assert record["actual_sp1_requests"] == case.total_requests - expected_sp8
    assert record["admission_ms"]["router_plan_receipt_ms"]["mean"] > 0.0
    assert record["admission_ms"]["local_commit_critical_ms"]["mean"] > 0.0
    assert record["modelled_parallel_quantum_ms"]["count"] == 1
    assert record["modelled_global_quantums_per_second"] > 0.0
    assert all(
        all(profile["correctness"].values())
        for profile in record["per_engine_profiles"]
    )

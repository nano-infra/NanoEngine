from __future__ import annotations

import pytest

from nanodeploy.engine.hierarchical_contract import AddCommand
from nanodeploy.router.admission_planner import AdmissionPlanner
from scripts.decentralized_scalability.common import (
    fixed_sp_planner_config,
    requests_for_case,
    sp_load_snapshot,
)
from scripts.decentralized_scalability.profile_frontend_ray_cpu import (
    _annotate_scaling,
    _event_counts,
    build_frontend_event_batch,
)
from scripts.decentralized_scalability.profile_local_scheduler_ray_cpu import (
    DEFAULT_MODEL,
    LocalSchedulerCpuWorkload,
    _batch_counts,
    configured_attention_dp,
    scheduler_kv_blocks,
)
from scripts.decentralized_scalability.profile_router_cpu import run_trial


def test_sp8_profiler_load_and_planner_use_all_production_ranks():
    snapshot = sp_load_snapshot(
        2,
        attention_sp=8,
        capacity_requests=64,
    )
    assert tuple(rank.global_rank for rank in snapshot.rank_loads) == tuple(
        range(16, 24)
    )
    assert tuple(rank.sp_idx for rank in snapshot.rank_loads) == tuple(range(8))

    planner = AdmissionPlanner(
        fixed_sp_planner_config(
            attention_sp=8,
            capacity_requests=64,
            prompt_tokens=64,
        )
    )
    shadow = planner.shadow_from_snapshot(snapshot)
    assert shadow is not None
    reservation = planner.plan(
        shadow,
        AddCommand(
            request_id=7,
            prompt_len=64,
            num_tokens=64,
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
            sequence_payload=b"unused",
        ),
    )
    assert reservation is not None
    assert reservation.engine_id == 2
    assert len(reservation.dispatched_tokens) == 8
    assert sum(reservation.dispatched_tokens) == 64
    assert all(token_count == 8 for token_count in reservation.dispatched_tokens)


def test_router_cpu_profiler_preserves_ownership_and_balances_sp8_engines():
    result = run_trial(
        engines=2,
        requests=17,
        batch_size=4,
        prompt_tokens=64,
        attention_sp=8,
    )

    assert all(result["correctness"].values())
    assert sum(result["per_engine_requests"]) == 17
    assert max(result["per_engine_requests"]) - min(
        result["per_engine_requests"]
    ) <= 4
    assert result["requests_per_second"] > 0


def test_scaling_case_request_counts_distinguish_strong_and_weak_scaling():
    assert requests_for_case(
        scaling_mode="strong",
        engines=4,
        strong_total_requests=100,
        requests_per_engine=30,
    ) == 100
    assert requests_for_case(
        scaling_mode="weak",
        engines=4,
        strong_total_requests=100,
        requests_per_engine=30,
    ) == 120


def test_frontend_event_payload_matches_sp8_production_contract():
    batch = build_frontend_event_batch(
        engine_id=3,
        attention_sp=8,
        event_count=5,
        quantum_id=11,
    )

    assert batch.engine_id == 3
    assert tuple(rank.global_rank for rank in batch.load.rank_loads) == tuple(
        range(24, 32)
    )
    assert len(batch.add_results) == 5
    assert all(event.engine_id == 3 for event in batch.add_results)
    assert [event.admission_version for event in batch.add_results] == [
        1,
        2,
        3,
        4,
        5,
    ]

    with pytest.raises(ValueError, match="event_count"):
        build_frontend_event_batch(
            engine_id=0,
            attention_sp=8,
            event_count=-1,
            quantum_id=0,
        )


def test_frontend_event_counts_define_strong_and_weak_scaling():
    assert _event_counts(
        scaling_mode="weak", nodes=2, max_nodes=4, batch_size=32
    ) == (32, 32)
    assert _event_counts(
        scaling_mode="strong", nodes=2, max_nodes=4, batch_size=32
    ) == (64, 64)


def test_frontend_scaling_annotation_uses_matching_one_node_case():
    records = [
        {
            "direction": "ingress",
            "scaling_mode": "weak",
            "batch_size": 32,
            "prompt_tokens": 64,
            "nodes": 1,
            "items_per_second": {"p50": 100.0},
        },
        {
            "direction": "ingress",
            "scaling_mode": "weak",
            "batch_size": 32,
            "prompt_tokens": 64,
            "nodes": 2,
            "items_per_second": {"p50": 180.0},
        },
    ]

    _annotate_scaling(records)

    assert records[1]["throughput_ratio_vs_one_node"] == 1.8
    assert records[1]["weak_scaling_efficiency_vs_one_node"] == 0.9


def test_local_scheduler_batch_counts_define_strong_and_weak_scaling():
    assert _batch_counts(
        scaling_mode="strong", nodes=4, total_or_per_engine_batch=32
    ) == (8, 8, 8, 8)
    assert _batch_counts(
        scaling_mode="weak", nodes=4, total_or_per_engine_batch=32
    ) == (32, 32, 32, 32)
    assert configured_attention_dp(1) == 1
    assert configured_attention_dp(2) == 2
    assert configured_attention_dp(3) == 4
    assert configured_attention_dp(4) == 4
    with pytest.raises(ValueError, match="actor count"):
        configured_attention_dp(5)


def test_local_scheduler_capacity_grows_with_batch_prompt_and_completion():
    baseline = scheduler_kv_blocks(
        batch_size=2,
        prompt_tokens=64,
        completion_tokens=64,
        attention_sp=8,
    )
    assert scheduler_kv_blocks(
        batch_size=4,
        prompt_tokens=64,
        completion_tokens=64,
        attention_sp=8,
    ) > baseline
    assert scheduler_kv_blocks(
        batch_size=2,
        prompt_tokens=8000,
        completion_tokens=64,
        attention_sp=8,
    ) > baseline
    assert scheduler_kv_blocks(
        batch_size=2,
        prompt_tokens=64,
        completion_tokens=8000,
        attention_sp=8,
    ) > baseline


@pytest.mark.skipif(
    not (DEFAULT_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found below {DEFAULT_MODEL}",
)
def test_local_scheduler_cpu_workload_runs_production_quantum_contract():
    workload = LocalSchedulerCpuWorkload(
        model=str(DEFAULT_MODEL),
        attention_dp=1,
        engine_id=0,
        batch_size=2,
        prompt_tokens=64,
        warmup_iterations=1,
        measured_iterations=2,
    )

    result = workload.profile()

    assert all(result["correctness"].values())
    assert result["phase_stats"]["plan_decode_ms"]["count"] == 2
    assert result["phase_stats"]["postprocess_ms"]["p99"] >= 0
    assert result["scheduler_quantums_per_second"] > 0

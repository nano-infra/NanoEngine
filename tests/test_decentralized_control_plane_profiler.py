from __future__ import annotations

import json

import pytest

from scripts.decentralized_control_plane.profile_decentralized_control_plane import (
    EVENT_MIXES,
    _build_event_batch,
    _event_counts,
    _event_quantum_stats,
    _new_router,
    _parse_nonnegative_floats,
    _parse_positive_ints,
    _percentile,
    _process_router_batch,
    _process_router_batches,
    _select_nodes,
    _stats,
    _write_results,
)


def test_parsers_and_statistics_are_deterministic() -> None:
    assert _parse_positive_ints("1,2,4") == (1, 2, 4)
    assert _parse_nonnegative_floats("0,0.5,2") == (0.0, 0.5, 2.0)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    stats = _stats([1.0, 2.0, 3.0, 4.0])
    assert stats.count == 4
    assert stats.mean == 2.5
    assert stats.p99 == pytest.approx(3.97)

    with pytest.raises(Exception):
        _parse_positive_ints("1,0")
    with pytest.raises(Exception):
        _parse_nonnegative_floats("0,-1")


def test_event_counts_define_strong_and_weak_scaling() -> None:
    assert _event_counts(
        scaling_mode="weak",
        engines=2,
        max_engines=4,
        batch_size=32,
    ) == (32, 32)
    assert _event_counts(
        scaling_mode="strong",
        engines=2,
        max_engines=4,
        batch_size=32,
    ) == (64, 64)
    assert _event_counts(
        scaling_mode="strong",
        engines=4,
        max_engines=4,
        batch_size=32,
    ) == (32, 32, 32, 32)


@pytest.mark.parametrize("event_mix", EVENT_MIXES)
def test_event_batch_matches_production_contract(event_mix: str) -> None:
    batch = _build_event_batch(
        engine_id=2,
        quantum_id=7,
        event_count=3,
        event_mix=event_mix,
    )

    assert batch.engine_id == 2
    assert batch.load.engine_id == 2
    assert batch.load.quantum_id == 7
    assert tuple(rank.global_rank for rank in batch.load.rank_loads) == tuple(
        range(16, 24)
    )
    assert len(batch.load.rank_loads) == 8
    expected = 0 if event_mix == "load" else 3
    assert batch.load.running == expected
    if event_mix == "mixed":
        assert len(batch.add_results) == 3
        assert len(batch.first_schedule_events) == 3
        assert len(batch.first_token_events) == 3
        assert len(batch.finish_events) == 3


@pytest.mark.parametrize("event_mix", EVENT_MIXES)
def test_router_receipt_processing_releases_harness_state(
    event_mix: str,
) -> None:
    batch = _build_event_batch(
        engine_id=0,
        quantum_id=0,
        event_count=4,
        event_mix=event_mix,
    )
    router = _new_router(1)

    result = _process_router_batch(router, batch)

    assert result["router_load_ms"] >= 0
    assert result["router_total_ms"] >= 0
    assert result["add_events"] == len(batch.add_results)
    assert result["first_schedule_events"] == len(
        batch.first_schedule_events
    )
    assert result["first_token_events"] == len(batch.first_token_events)
    assert result["finish_events"] == len(batch.finish_events)
    assert router.active_count == 0
    assert not router._terminal


def test_router_receipt_processing_batches_ready_engines_together() -> None:
    batches = tuple(
        _build_event_batch(
            engine_id=engine_id,
            quantum_id=0,
            event_count=3,
            event_mix="mixed",
        )
        for engine_id in range(2)
    )
    router = _new_router(2)

    result = _process_router_batches(router, batches)

    assert result["add_events"] == 6
    assert result["first_schedule_events"] == 6
    assert result["first_token_events"] == 6
    assert result["finish_events"] == 6
    assert router.active_count == 0
    assert not router._terminal


def test_event_quantum_stats_take_network_max_and_router_sum() -> None:
    samples = [
        {
            "engine_id": engine_id,
            "quantum_id": quantum_id,
            "flight_age_ms": float(engine_id + quantum_id + 1),
            "router_total_ms": 0.1 * (engine_id + 1),
            "payload_bytes": 100.0,
            "event_items": 4.0,
        }
        for quantum_id in range(2)
        for engine_id in range(2)
    ]

    flight, router, payload, events = _event_quantum_stats(
        samples,
        engines=2,
        iterations=2,
    )

    assert flight == [2.0, 3.0]
    assert router == pytest.approx([0.3, 0.3])
    assert payload == [200.0, 200.0]
    assert events == [8.0, 8.0]


def test_node_selection_honors_explicit_order() -> None:
    nodes = [
        {
            "Alive": True,
            "NodeID": "node-a",
            "NodeManagerAddress": "10.0.0.1",
            "Resources": {"node:__internal_head__": 1.0},
        },
        {
            "Alive": True,
            "NodeID": "node-b",
            "NodeManagerAddress": "10.0.0.2",
            "Resources": {},
        },
    ]

    selected = _select_nodes(
        nodes,
        count=2,
        requested_ips=("10.0.0.2", "10.0.0.1"),
    )

    assert tuple(node["NodeID"] for node in selected) == (
        "node-b",
        "node-a",
    )


def test_result_writer_emits_dedicated_json_and_csv(tmp_path) -> None:
    _write_results(
        tmp_path,
        {"benchmark": "test"},
        [
            {
                "component": "consensus",
                "nodes": 2,
                "nested": {"valid": True},
            },
            {
                "component": "events",
                "nodes": 4,
                "extra": 1.0,
            },
        ],
    )

    json_path = tmp_path / "decentralized_control_plane.json"
    csv_path = tmp_path / "decentralized_control_plane.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["benchmark"] == "test"
    assert [record["component"] for record in payload["records"]] == [
        "consensus",
        "events",
    ]

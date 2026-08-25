from __future__ import annotations

import pytest

from scripts.benchmark_least_batch_cpu import (
    run_router_pipeline_trial,
    run_zmq_staged_receipt_trial,
)
from scripts.benchmark_least_batch_ray_cpu import _select_cluster_nodes


def test_cpu_router_pipeline_dispatches_all_requests_before_commit():
    result = run_router_pipeline_trial(
        requests=128,
        engines=2,
        batch_size=16,
        prompt_tokens=8,
        receipt_delay_ms=0.0,
    )

    assert result["positive_receipts"] == 128
    assert result["scheduler_commits"] == 0
    assert result["pending_add"] == 128
    assert result["batch_messages"] == 8
    assert result["requests_per_message"] == 16.0
    assert all(result["correctness"].values())


def test_cpu_zmq_receipt_stages_payload_without_scheduler_commit():
    result = run_zmq_staged_receipt_trial(
        requests=32,
        batch_size=8,
        prompt_tokens=8,
    )

    assert result["batch_messages"] == 4
    assert result["scheduler_commit_calls"] == 0
    assert result["admission_version"] == 0
    assert result["reserved_slots"] == 32
    assert result["staged_ingress_depth_max"] == 32
    assert result["sequence_payload_bytes"] > 0
    assert all(result["correctness"].values())


@pytest.mark.parametrize(
    "overrides",
    (
        {"requests": 0},
        {"engines": 0},
        {"batch_size": 0},
        {"prompt_tokens": 0},
        {"receipt_delay_ms": -1.0},
    ),
)
def test_cpu_router_benchmark_rejects_invalid_parameters(overrides):
    parameters = {
        "requests": 8,
        "engines": 1,
        "batch_size": 4,
        "prompt_tokens": 4,
        "receipt_delay_ms": 0.0,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError):
        run_router_pipeline_trial(**parameters)


def test_ray_cpu_benchmark_selects_two_alive_cpu_nodes():
    nodes = [
        {
            "NodeID": "node-b",
            "NodeManagerAddress": "10.0.0.2",
            "Alive": True,
            "Resources": {"CPU": 8.0, "GPU": 8.0},
        },
        {
            "NodeID": "dead",
            "NodeManagerAddress": "10.0.0.3",
            "Alive": False,
            "Resources": {"CPU": 8.0},
        },
        {
            "NodeID": "node-a",
            "NodeManagerAddress": "10.0.0.1",
            "Alive": True,
            "Resources": {"CPU": 8.0},
        },
    ]

    selected = _select_cluster_nodes(nodes, num_nodes=2)

    assert [node["NodeID"] for node in selected] == ["node-a", "node-b"]


def test_ray_cpu_benchmark_rejects_missing_requested_node():
    nodes = [
        {
            "NodeID": "node-a",
            "NodeManagerAddress": "10.0.0.1",
            "Alive": True,
            "Resources": {"CPU": 8.0},
        }
    ]

    with pytest.raises(RuntimeError, match="not alive"):
        _select_cluster_nodes(
            nodes,
            num_nodes=1,
            requested_node_ips=("10.0.0.2",),
        )

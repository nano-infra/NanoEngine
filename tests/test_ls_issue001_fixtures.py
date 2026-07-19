from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanodeploy._cpp import LSAddError, Scheduler, Sequence, SequenceStatus


_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "ls_issue001_windows.json"
_SOURCE_SHA256 = "6e319608aaaca20656e838812b1188f966e77524c7f56da5feb20661bfdc216b"

# These tuples intentionally duplicate the checked-in JSON. The JSON is the
# reusable fixture, while this literal is the reviewable oracle that prevents a
# self-consistent edit of both request rows and expected pool snapshots from
# silently changing the six source windows.
_AUTHORITATIVE_WINDOWS = {
    "arrival_1482_1488": (
        (
            (1482, 1482, 276, 510, "short"),
            (1483, 1483, 199, 640, "short"),
            (1484, 1484, 923230, 628, "long"),
            (1485, 1485, 229, 512, "short"),
            (1486, 1486, 197, 639, "short"),
            (1487, 1487, 213, 494, "short"),
            (1488, 1488, 227, 633, "short"),
        ),
        ((1484, 1488), (1485,), (1482, 1486), (1483, 1487)),
    ),
    "arrival_5234_5236": (
        (
            (5234, 5234, 170, 490, "short"),
            (5235, 5235, 960909, 571, "long"),
            (5236, 5236, 199, 669, "short"),
        ),
        ((5236,), (), (5234,), (5235,)),
    ),
    "arrival_5372_5375": (
        (
            (5372, 5372, 971548, 107, "long"),
            (5373, 5373, 198, 707, "short"),
            (5374, 5374, 215, 593, "short"),
            (5375, 5375, 277, 399, "short"),
        ),
        ((5372,), (5373,), (5374,), (5375,)),
    ),
    "arrival_5722_5725": (
        (
            (5722, 5722, 228, 848, "short"),
            (5723, 5723, 921913, 285, "long"),
            (5724, 5724, 217, 496, "short"),
            (5725, 5725, 286, 558, "short"),
        ),
        ((5724,), (5725,), (5722,), (5723,)),
    ),
    "arrival_6407_6410": (
        (
            (6407, 6407, 941787, 588, "long"),
            (6408, 6408, 199, 488, "short"),
            (6409, 6409, 198, 671, "short"),
            (6410, 6410, 214, 733, "short"),
        ),
        ((6408,), (6409,), (6410,), (6407,)),
    ),
    "arrival_6915_6920": (
        (
            (6915, 6915, 205, 508, "short"),
            (6916, 6916, 130, 430, "short"),
            (6917, 6917, 214, 638, "short"),
            (6918, 6918, 198, 579, "short"),
            (6919, 6919, 840341, 627, "long"),
            (6920, 6920, 201, 745, "short"),
        ),
        ((6916, 6920), (6917,), (6918,), (6915, 6919)),
    ),
}


@pytest.fixture(scope="module")
def issue001_document() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _window_by_name(document: dict, name: str) -> dict:
    matches = [window for window in document["windows"] if window["name"] == name]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("window_name", tuple(_AUTHORITATIVE_WINDOWS))
def test_issue001_window_matches_authoritative_rows_and_dp4_fifo(
    issue001_document: dict,
    window_name: str,
):
    assert issue001_document["schema_version"] == 1
    assert issue001_document["source"]["csv_name"] == (
        "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
    )
    assert issue001_document["source"]["csv_sha256"] == _SOURCE_SHA256
    assert issue001_document["assignment"] == {
        "attention_dp": 4,
        "initial_dp": 0,
        "policy": "assigned_dp = arrival_index % attention_dp",
    }

    window = _window_by_name(issue001_document, window_name)
    expected_requests, expected_pools = _AUTHORITATIVE_WINDOWS[window_name]
    actual_requests = tuple(
        (
            request["arrival_index"],
            request["request_id"],
            request["prompt_len"],
            request["output_len"],
            request["type"],
        )
        for request in window["requests"]
    )
    assert actual_requests == expected_requests

    snapshots = window["expected_dp4_pool_snapshots"]
    assert [snapshot["dp_idx"] for snapshot in snapshots] == list(range(4))
    actual_pools = tuple(tuple(snapshot["request_ids"]) for snapshot in snapshots)
    assert actual_pools == expected_pools

    request_by_id = {
        request["request_id"]: request
        for request in window["requests"]
    }
    assert len(request_by_id) == len(window["requests"])
    for snapshot in snapshots:
        assert snapshot["prompt_lengths"] == [
            request_by_id[request_id]["prompt_len"]
            for request_id in snapshot["request_ids"]
        ]

    # Reconstruct the pool queues from global arrival order. This is the key
    # contract: the listed length window is never treated as one global pool.
    derived_pools = [[] for _ in range(4)]
    for request in window["requests"]:
        assert request["request_id"] == request["arrival_index"]
        derived_pools[request["arrival_index"] % 4].append(request["request_id"])
    assert tuple(tuple(pool) for pool in derived_pools) == expected_pools

    arrivals = [request["arrival_index"] for request in window["requests"]]
    assert arrivals == list(range(arrivals[0], arrivals[-1] + 1))
    assert sum(request["type"] == "long" for request in window["requests"]) == 1
    assert all(request["output_len"] > 0 for request in window["requests"])


def test_issue001_fixture_ids_are_a_complete_unique_union(issue001_document: dict):
    assert [window["name"] for window in issue001_document["windows"]] == list(
        _AUTHORITATIVE_WINDOWS
    )
    all_request_ids: list[int] = []
    all_arrival_indices: list[int] = []

    for window in issue001_document["windows"]:
        request_ids = [request["request_id"] for request in window["requests"]]
        all_request_ids.extend(request_ids)
        all_arrival_indices.extend(
            request["arrival_index"]
            for request in window["requests"]
        )

        snapshot_ids = [
            request_id
            for snapshot in window["expected_dp4_pool_snapshots"]
            for request_id in snapshot["request_ids"]
        ]
        assert len(snapshot_ids) == len(set(snapshot_ids))
        assert set(snapshot_ids) == set(request_ids)

    assert len(all_request_ids) == len(set(all_request_ids))
    assert len(all_arrival_indices) == len(set(all_arrival_indices))
    assert set(all_request_ids) == set(all_arrival_indices)


def _make_assignment_scheduler() -> Scheduler:
    # Assignment and pool-local waiting FIFO do not depend on SP width. Use SP1
    # here so a 971K singleton has real empty-system capacity without allocating
    # eight copies of that capacity in a CPU routing-only test. The JSON/pure
    # contract above remains the authoritative DP4×SP8 pool snapshot. This test
    # deliberately makes no claim about blocker admission or elasticity.
    return Scheduler(
        "",
        1,
        32,
        4096,
        32,
        -1,
        4,
        1,
        16000,
        64,
        "decode",
        0.0,
        64,
        False,
        False,
        "legacy",
        100_000,
        0,
        False,
        "",
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1,
        1,
        1,
        False,
        "RoundRobin",
        False,
        0,
        True,
        1,
        128,
        True,
        "centralized",
        "off",
        0.50,
        0.80,
        32,
        64,
        8,
        0,
        True,
        10,
        1000,
        1_024_000,
        128,
    )


@pytest.mark.parametrize("window_name", tuple(_AUTHORITATIVE_WINDOWS))
def test_real_scheduler_preserves_arrival_rr_and_pool_fifo_without_reroute(
    issue001_document: dict,
    window_name: str,
):
    scheduler = _make_assignment_scheduler()
    window = _window_by_name(issue001_document, window_name)
    expected_requests, expected_pools = _AUTHORITATIVE_WINDOWS[window_name]

    # Reproduce the original global RR residue without replaying thousands of
    # preceding requests. Structurally new typed rejections consume RR but do
    # not enter any queue or arrival-order index.
    target_start_dp = expected_requests[0][0] % 4
    for expected_dp in range(target_start_dp):
        rotation = Sequence([0], 1.0, 0, True)
        rotation.seq_id = 1_000_000_000 + expected_requests[0][0] + expected_dp
        rejected = scheduler.add(rotation)
        assert rejected.accepted is False
        assert rejected.error == LSAddError.INVALID_MAX_TOKENS
        assert rejected.assigned_dp == expected_dp

    sequences_by_request_id: dict[int, Sequence] = {}
    for arrival_order, request in enumerate(window["requests"]):
        sequence = Sequence(
            [0] * request["prompt_len"],
            1.0,
            request["output_len"],
            True,
        )
        sequence.seq_id = request["request_id"]
        added = scheduler.add(sequence)
        expected_dp = request["arrival_index"] % 4

        assert added.accepted is True
        assert added.error == LSAddError.NONE
        assert added.assigned_dp == expected_dp
        assert sequence.assigned_dp == expected_dp
        assert sequence.status == SequenceStatus.WAITING
        assert scheduler.get_ls_arrival_order(sequence.seq_id) == arrival_order
        sequences_by_request_id[request["request_id"]] = sequence

    actual_pools = scheduler.get_ls_waiting_sequence_ids_by_dp()
    assert tuple(tuple(pool) for pool in actual_pools) == expected_pools

    flattened = [request_id for pool in actual_pools for request_id in pool]
    expected_ids = [request[1] for request in expected_requests]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(expected_ids)
    assert set(sequences_by_request_id) == set(expected_ids)
    for dp_idx, request_ids in enumerate(actual_pools):
        assert all(
            sequences_by_request_id[request_id].assigned_dp == dp_idx
            for request_id in request_ids
        )

    # No schedule/admission has run, so every fixture request must remain a
    # request-level queue member without a group or historical batch identity.
    assert scheduler.get_ls_group_ids() == []
    assert scheduler.get_ls_group_sequence_ids() == []
    assert scheduler.get_ls_pending_batch_ids() == []
    assert scheduler.get_ls_pending_batch_sequence_ids() == []
    assert scheduler.get_ls_active_batch_owners() == []

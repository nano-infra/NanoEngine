from __future__ import annotations

import json
import pickle

import pytest

from scripts.ray_rpc_overhead import profile_ray_vs_dlslime as profiler


def test_sequence_batch_preserves_length_through_pickle() -> None:
    batch = profiler._sample_sequence_batch(3, 2, 8_000)

    assert len(batch) == 2
    assert all(sequence.num_tokens == 8_000 for sequence in batch)
    restored = pickle.loads(
        pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert all(sequence.num_tokens == 8_000 for sequence in restored)
    assert all(len(sequence.token_ids) == 8_000 for sequence in restored)


def test_parse_transports_rejects_invalid_or_duplicate_values() -> None:
    assert profiler._parse_transports("ray,dlslime") == ("ray", "dlslime")
    with pytest.raises(Exception):
        profiler._parse_transports("ray,ray")
    with pytest.raises(Exception):
        profiler._parse_transports("other")


def test_encode_transport_imm_supports_legacy_and_slotted_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        profiler.rpc_endpoint_module,
        "_encode_imm",
        raising=False,
    )
    assert profiler._encode_transport_imm(64, 0) == 64
    with pytest.raises(RuntimeError, match="legacy RPC endpoint"):
        profiler._encode_transport_imm(64, 1)

    monkeypatch.setattr(
        profiler.rpc_endpoint_module,
        "_encode_imm",
        lambda payload_bytes, slot: payload_bytes * 10 + slot,
        raising=False,
    )
    assert profiler._encode_transport_imm(64, 1) == 641


def test_plan_actor_node_ids_places_eight_workers_per_node() -> None:
    nodes = [
        {
            "Alive": True,
            "NodeID": "remote",
            "NodeManagerAddress": "10.0.0.2",
            "NodeManagerHostname": "remote-host",
            "Resources": {},
        },
        {
            "Alive": True,
            "NodeID": "head",
            "NodeManagerAddress": "10.0.0.1",
            "NodeManagerHostname": "head-host",
            "Resources": {"node:__internal_head__": 1.0},
        },
    ]

    assignments, selected = profiler._plan_actor_node_ids(
        nodes,
        logical_workers=16,
        workers_per_node=8,
    )

    assert assignments == ("head",) * 8 + ("remote",) * 8
    assert [node["hostname"] for node in selected] == [
        "head-host",
        "remote-host",
    ]


class _FakeRemoteMethod:
    def __init__(self, calls, result_ref) -> None:
        self.calls = calls
        self.result_ref = result_ref

    def remote(self, *args):
        self.calls.append(args)
        return self.result_ref


class _FakeActor:
    def __init__(self, calls, result_ref) -> None:
        self.run = _FakeRemoteMethod(calls, result_ref)


def test_invoke_once_uses_expected_ray_input_path(monkeypatch) -> None:
    calls = []
    results = [([[1]], 1.0), ([[2]], 1.0)]
    actors = [_FakeActor(calls, 0), _FakeActor(calls, 1)]
    batches = [[object()], [object()]]
    monkeypatch.setattr(profiler.ray, "get", lambda refs: results)

    observed = profiler._invoke_once(
        actors,
        batches,
        transport="ray",
        dlslime_endpoint=None,
    )

    assert [call[0] for call in calls] == batches
    assert all(call[2] is False for call in calls)
    assert observed[1] == 0.0
    assert observed[4] >= 0.0
    assert observed[5]["total_bytes"] == 0
    assert observed[6] == results


def test_invoke_once_uses_expected_dlslime_input_path(monkeypatch) -> None:
    calls = []
    results = [([[1]], 1.0), ([[2]], 1.0)]
    actors = [_FakeActor(calls, 0), _FakeActor(calls, 1)]
    batches = [[object()], [object()]]
    endpoint = object()
    monkeypatch.setattr(profiler.ray, "get", lambda refs: results)
    observed_profile = {
        "total_ms": 10.0,
        "serialize_ms": 1.0,
        "write_with_imm_ms": 2.0,
        "future_wait_ms": 6.0,
        "unattributed_ms": 1.0,
        "future_wait_ms_by_rank": [4.0, 2.0],
        "total_bytes": 128,
    }
    monkeypatch.setattr(
        profiler,
        "_profiled_send_seqs",
        lambda observed_endpoint, sequences, *, is_prefill: observed_profile,
    )

    observed = profiler._invoke_once(
        actors,
        batches,
        transport="dlslime",
        dlslime_endpoint=endpoint,
    )

    assert [call[0] for call in calls] == [[], []]
    assert all(call[2] is True for call in calls)
    assert observed[1] >= 0.0
    assert observed[4] >= 0.0
    assert observed[5] == observed_profile
    assert observed[6] == results


def test_invoke_once_rejects_dlslime_without_endpoint() -> None:
    actors = [_FakeActor([], 0)]
    with pytest.raises(RuntimeError, match="connected endpoint"):
        profiler._invoke_once(
            actors,
            [[object()]],
            transport="dlslime",
            dlslime_endpoint=None,
        )


def test_result_writer_emits_json_and_csv(tmp_path) -> None:
    _ = profiler._write_results(
        tmp_path,
        {"benchmark": "test"},
        [
            {"transport": "ray", "roundtrip_mean_ms": 1.0},
            {"transport": "dlslime", "roundtrip_mean_ms": 2.0},
        ],
    )

    json_path = tmp_path / "ray_vs_dlslime.json"
    csv_path = tmp_path / "ray_vs_dlslime.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["benchmark"] == "test"
    assert [record["transport"] for record in payload["records"]] == [
        "ray",
        "dlslime",
    ]


def test_validate_args_requires_worker_partition() -> None:
    parser = profiler._build_parser()
    args = parser.parse_args(
        [
            "--ray-address",
            "10.0.0.1:6380",
            "--logical-workers",
            "10",
            "--workers-per-node",
            "8",
            "--transports",
            "ray",
            "--output-dir",
            "/tmp/test-ray-vs-dlslime",
        ]
    )
    with pytest.raises(SystemExit):
        profiler._validate_args(args, parser)

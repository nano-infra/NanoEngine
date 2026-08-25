from __future__ import annotations

import pickle

import pytest

from scripts.ray_rpc_overhead import profile_ray_vs_dlslime as profiler


def test_sample_sequence_batch_preserves_length_through_pickle() -> None:
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


class _FakeEndpoint:
    def __init__(self) -> None:
        self.calls = []

    def send_seqs(self, sequences, is_prefill):
        self.calls.append((sequences, is_prefill))


@pytest.mark.parametrize("transport", ["ray", "dlslime"])
def test_invoke_once_uses_expected_input_path(monkeypatch, transport) -> None:
    calls = []
    results = [([[1]], 1.0), ([[2]], 1.0)]
    actors = [
        _FakeActor(calls, 0),
        _FakeActor(calls, 1),
    ]
    batches = [[object()], [object()]]
    endpoint = _FakeEndpoint()
    monkeypatch.setattr(profiler.ray, "get", lambda refs: results)

    *_, observed_results = profiler._invoke_once(
        actors,
        batches,
        transport=transport,
        dlslime_endpoint=endpoint,
    )

    assert observed_results == results
    if transport == "ray":
        assert [call[0] for call in calls] == batches
        assert all(call[2] is False for call in calls)
        assert endpoint.calls == []
    else:
        assert [call[0] for call in calls] == [[], []]
        assert all(call[2] is True for call in calls)
        assert endpoint.calls == [(batches, False)]


def test_invoke_once_rejects_dlslime_without_endpoint() -> None:
    calls = []
    actors = [_FakeActor(calls, 0)]

    with pytest.raises(RuntimeError, match="connected endpoint"):
        profiler._invoke_once(
            actors,
            [[object()]],
            transport="dlslime",
            dlslime_endpoint=None,
        )

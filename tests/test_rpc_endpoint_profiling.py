from __future__ import annotations

from nanodeploy.endpoint import rpc_endpoint
from scripts.ray_rpc_overhead import profile_ray_vs_dlslime as profiler


class _FakeBuffer:
    def __init__(self, pointer: int, size: int = 4096) -> None:
        self.pointer = pointer
        self.size = size

    def data_ptr(self) -> int:
        return self.pointer

    def storage_offset(self) -> int:
        return 0

    def numel(self) -> int:
        return self.size


class _FakeFuture:
    def __init__(self, rank: int, events: list[tuple]) -> None:
        self.rank = rank
        self.events = events

    def wait(self) -> None:
        self.events.append(("wait", self.rank))


class _FakeRDMAEndpoint:
    def __init__(self, rank: int, events: list[tuple]) -> None:
        self.rank = rank
        self.events = events

    def write_with_imm(self, segments, immediate_data):
        self.events.append(
            ("write_with_imm", self.rank, segments, immediate_data)
        )
        return _FakeFuture(self.rank, self.events)


def _server_endpoint(events: list[tuple]) -> rpc_endpoint.RPCServerEndpoint:
    endpoint = rpc_endpoint.RPCServerEndpoint.__new__(
        rpc_endpoint.RPCServerEndpoint
    )
    endpoint.world_size = 2
    endpoint.buffer_size = 4096
    endpoint.num_slots = 1
    endpoint.slot_size = 4096
    endpoint.attention_sp = 2
    endpoint.attention_tp = 1
    endpoint.optimize_decode_block_table = True
    endpoint.server_bindings = [
        rpc_endpoint.EndpointBinding(
            _FakeRDMAEndpoint(rank, events),
            _FakeBuffer(1000 + rank),
            2000 + rank,
        )
        for rank in range(2)
    ]
    return endpoint


def test_send_seqs_profile_preserves_production_operation_order(
    monkeypatch,
) -> None:
    events: list[tuple] = []
    endpoint = _server_endpoint(events)

    def fake_serialize(
        buffer_ptr,
        buffer_size,
        sequences,
        is_prefill,
        sp_rank,
        sp_size,
    ):
        events.append(
            (
                "serialize",
                buffer_ptr,
                buffer_size,
                sequences,
                is_prefill,
                sp_rank,
                sp_size,
            )
        )
        return 100 + len(sequences)

    monkeypatch.setattr(profiler, "serialize", fake_serialize)

    profile = profiler._profiled_send_seqs(
        endpoint,
        [["rank-0"], ["rank-1-a", "rank-1-b"]],
        is_prefill=False,
    )

    assert profile["total_bytes"] == 203
    assert len(profile["future_wait_ms_by_rank"]) == 2
    assert profile["total_ms"] >= (
        profile["serialize_ms"]
        + profile["write_with_imm_ms"]
        + profile["future_wait_ms"]
    )
    assert [event[0] for event in events] == [
        "serialize",
        "write_with_imm",
        "serialize",
        "write_with_imm",
        "wait",
        "wait",
    ]
    assert events[0][-2:] == (0, 2)
    assert events[2][-2:] == (1, 2)


def test_profiled_send_seqs_supports_legacy_single_slot_endpoint(
    monkeypatch,
) -> None:
    events: list[tuple] = []
    endpoint = _server_endpoint(events)
    del endpoint.num_slots
    del endpoint.slot_size
    monkeypatch.setattr(
        profiler,
        "serialize",
        lambda *_args: 64,
    )

    profile = profiler._profiled_send_seqs(
        endpoint,
        [[], []],
        is_prefill=False,
    )

    assert profile["total_bytes"] == 128
    assert [event[0] for event in events] == [
        "write_with_imm",
        "write_with_imm",
        "wait",
        "wait",
    ]

import pytest
import torch

import nanodeploy.worker.kv_p2p as kv_p2p
from nanodeploy.worker.kv_p2p import KVCacheP2PMove, KVCacheP2PTransport


class _CompletedWork:
    def __init__(self, calls: dict[str, int]):
        self.calls = calls

    def wait(self):
        self.calls["wait"] += 1
        return True


def _install_fake_dist(monkeypatch, rank_ref, payloads, calls):
    monkeypatch.setattr(kv_p2p.dist, "get_rank", lambda group: rank_ref["value"])
    monkeypatch.setattr(kv_p2p.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        kv_p2p.dist,
        "get_global_rank",
        lambda group, group_rank: group_rank + 10,
    )

    def fake_isend(tensor, dst, group):
        calls["send_peers"].append(dst)
        payloads.append(tensor.clone())
        return _CompletedWork(calls)

    def fake_irecv(tensor, src, group):
        calls["recv_peers"].append(src)
        tensor.copy_(payloads.pop(0))
        return _CompletedWork(calls)

    monkeypatch.setattr(kv_p2p.dist, "isend", fake_isend)
    monkeypatch.setattr(kv_p2p.dist, "irecv", fake_irecv)


def test_isend_irecv_copies_chunked_token_ranges(monkeypatch):
    source_cache = torch.arange(1 * 2 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
        1, 2, 3, 4, 1, 2
    )
    destination_cache = torch.full_like(source_cache, -1)
    source_before = source_cache.clone()

    # Reverse planner order deliberately; both workers normalize it identically.
    moves = [
        KVCacheP2PMove(
            dp_idx=0,
            src_sp_rank=0,
            dst_sp_rank=1,
            src_block_id=2,
            src_token_offset=0,
            dst_block_id=0,
            dst_token_offset=2,
            num_tokens=2,
        ),
        KVCacheP2PMove(
            dp_idx=0,
            src_sp_rank=0,
            dst_sp_rank=1,
            src_block_id=0,
            src_token_offset=0,
            dst_block_id=1,
            dst_token_offset=0,
            num_tokens=4,
        ),
    ]

    rank_ref = {"value": 0}
    payloads = []
    calls = {"send_peers": [], "recv_peers": [], "wait": 0}
    _install_fake_dist(monkeypatch, rank_ref, payloads, calls)

    source_transport = KVCacheP2PTransport(
        source_cache, group="fake-group", chunk_tokens=3
    )
    source_result = source_transport.execute(moves, current_dp_idx=0)

    assert source_result.role == "source"
    assert source_result.num_chunks == 2
    assert source_result.sent_bytes == 6 * source_transport.token_bytes
    assert calls["send_peers"] == [11, 11]
    assert [payload.size(0) for payload in payloads] == [3, 3]
    assert torch.equal(source_cache, source_before)

    rank_ref["value"] = 1
    destination_transport = KVCacheP2PTransport(
        destination_cache, group="fake-group", chunk_tokens=3
    )
    destination_result = destination_transport.execute(moves, current_dp_idx=0)

    assert destination_result.role == "destination"
    assert destination_result.num_chunks == 2
    assert destination_result.received_bytes == 6 * source_transport.token_bytes
    assert calls["recv_peers"] == [10, 10]
    assert calls["wait"] == 4
    assert payloads == []
    assert torch.equal(
        destination_cache[:, :, 1, :, :, :],
        source_cache[:, :, 0, :, :, :],
    )
    assert torch.equal(
        destination_cache[:, :, 0, 2:4, :, :],
        source_cache[:, :, 2, 0:2, :, :],
    )
    assert torch.all(destination_cache[:, :, 0, 0:2, :, :] == -1)
    assert torch.all(destination_cache[:, :, 2, :, :, :] == -1)


def test_workers_outside_selected_dp_do_not_issue_p2p(monkeypatch):
    cache = torch.zeros(1, 1, 2, 4, 1, 1)
    moves = [
        KVCacheP2PMove(
            dp_idx=1,
            src_sp_rank=0,
            dst_sp_rank=1,
            src_block_id=0,
            src_token_offset=0,
            dst_block_id=1,
            dst_token_offset=0,
            num_tokens=1,
        )
    ]
    monkeypatch.setattr(kv_p2p.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(kv_p2p.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        kv_p2p.dist,
        "isend",
        lambda *args, **kwargs: pytest.fail("idle DP issued isend"),
    )
    monkeypatch.setattr(
        kv_p2p.dist,
        "irecv",
        lambda *args, **kwargs: pytest.fail("idle DP issued irecv"),
    )

    result = KVCacheP2PTransport(cache, group="fake-group", chunk_tokens=2).execute(
        moves, current_dp_idx=0
    )

    assert result.role == "idle"
    assert result.num_chunks == 0


def test_source_sends_to_multiple_destinations_in_rank_order(monkeypatch):
    cache = torch.arange(1 * 1 * 2 * 4 * 1 * 1, dtype=torch.float32).reshape(
        1, 1, 2, 4, 1, 1
    )
    moves = [
        KVCacheP2PMove(0, 0, 2, 1, 0, 0, 0, 1),
        KVCacheP2PMove(0, 0, 1, 0, 0, 1, 0, 1),
    ]
    peers = []
    calls = {"wait": 0}
    monkeypatch.setattr(kv_p2p.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(kv_p2p.dist, "get_world_size", lambda group: 3)
    monkeypatch.setattr(
        kv_p2p.dist,
        "get_global_rank",
        lambda group, group_rank: group_rank + 10,
    )

    def fake_isend(tensor, dst, group):
        peers.append(dst)
        return _CompletedWork(calls)

    monkeypatch.setattr(kv_p2p.dist, "isend", fake_isend)

    result = KVCacheP2PTransport(cache, group="fake-group", chunk_tokens=2).execute(
        moves, current_dp_idx=0
    )

    assert result.role == "source"
    assert result.num_chunks == 2
    assert peers == [11, 12]
    assert calls["wait"] == 2


@pytest.mark.parametrize(
    ("moves", "message"),
    [
        (
            [
                KVCacheP2PMove(0, 0, 0, 0, 0, 1, 0, 1),
            ],
            "must differ",
        ),
        (
            [
                KVCacheP2PMove(0, 0, 1, 0, 0, 1, 0, 2),
                KVCacheP2PMove(0, 0, 1, 0, 1, 2, 0, 2),
            ],
            "overlapping source",
        ),
        (
            [
                KVCacheP2PMove(0, 0, 1, 0, 0, 1, 0, 1),
                KVCacheP2PMove(0, 1, 0, 0, 1, 1, 1, 1),
            ],
            "exactly one source",
        ),
        (
            [
                KVCacheP2PMove(0, 0, 1, 0, 3, 1, 0, 2),
            ],
            "source token range",
        ),
    ],
)
def test_invalid_plans_are_rejected(monkeypatch, moves, message):
    cache = torch.zeros(1, 1, 3, 4, 1, 1)
    monkeypatch.setattr(kv_p2p.dist, "get_world_size", lambda group: 2)
    transport = KVCacheP2PTransport(cache, group="fake-group", chunk_tokens=2)

    with pytest.raises(ValueError, match=message):
        transport.execute(moves, current_dp_idx=0)


def test_scratch_is_token_major_and_chunk_views_are_contiguous():
    cache = torch.zeros(2, 3, 4, 5, 2, 7)
    transport = KVCacheP2PTransport(cache, group="fake-group", chunk_tokens=4)

    assert transport.scratch.shape == (4, 2, 3, 2, 7)
    assert transport.scratch[:3].is_contiguous()

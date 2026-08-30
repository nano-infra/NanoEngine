from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from dlengine.runtime.context.cache._allocator import allocate_device_tensor
from dlengine.runtime.context.peer import normalize_peer_placements, PeerContext
from dlengine.runtime.disagg.p2p.cache_transfer import _validate_named_region_ops
from dlengine.runtime.layers.indexer import IndexerCache


def _peer_context(agent):
    return PeerContext(
        agent=agent,
        alias="engine:0",
        server_url="http://ctrl",
        device=None,
    )


def test_fabric_capability_requires_one_ready_gpu_and_imex_channel():
    agent = Mock()
    agent.get_resource.return_value = {
        "runtime_capabilities": {"cuda": {"imex": {"channel_ids": [0]}}},
        "accelerators": [
            {
                "uuid": "GPU-a",
                "mnnvl": {
                    "membership_ready": True,
                    "cluster_uuid": "CLUSTER-A",
                    "clique_id": 7,
                },
            }
        ],
        "topology_epoch": 3,
    }

    context = _peer_context(agent)
    assert context.supports_cuda_fabric()
    assert context.local_placement() == {
        "rank": 0,
        "peer_agent_id": "engine:0",
        "gpu_uuid": "GPU-a",
        "cluster_uuid": "cluster-a",
        "clique_id": 7,
        "fabric_domain_id": "cluster-a:7",
        "topology_epoch": 3,
        "membership_ready": True,
        "imex_channel_ids": [0],
    }

    agent.get_resource.return_value["runtime_capabilities"]["cuda"]["imex"][
        "channel_ids"
    ] = []
    assert not _peer_context(agent).supports_cuda_fabric()


def test_peer_connections_request_automatic_transport():
    connection = Mock()
    connection.wait.return_value = connection
    agent = Mock()
    agent.connect_to.return_value = connection
    context = _peer_context(agent)

    context.ensure_connected("remote:0")

    agent.connect_to.assert_called_once_with(
        "remote:0", transport="auto", ib_port=1, qp_num=1
    )
    assert context.is_connected("remote:0")


def test_peer_connections_fall_back_for_older_dlslime():
    connection = Mock()
    connection.wait.return_value = connection
    agent = Mock()
    agent.connect_to.side_effect = [
        ValueError("unsupported transport 'auto'"),
        connection,
    ]
    context = _peer_context(agent)

    context.ensure_connected("remote:0")

    assert agent.connect_to.call_args_list[0].kwargs["transport"] == "auto"
    assert "transport" not in agent.connect_to.call_args_list[1].kwargs


def test_cache_allocator_delegates_to_peer_owned_fabric_storage():
    expected = torch.empty(2, 3)
    peer = Mock()
    peer.allocate_tensor.return_value = expected
    context = SimpleNamespace(
        peer_fabric_enabled=True, peer_context=peer, device="cuda"
    )

    actual = allocate_device_tensor(context, "kv_cache", (2, 3), torch.float32)

    assert actual is expected
    peer.allocate_tensor.assert_called_once_with(
        "kv_cache", (2, 3), torch.float32, zero=False
    )


def test_indexer_cache_accepts_injected_contiguous_buffer():
    buffer = torch.zeros((2, 3, 64 * 132), dtype=torch.uint8)

    cache = IndexerCache(2, 3, 64, 128, device="cpu", buffer=buffer)

    assert cache.buffer is buffer
    with pytest.raises(ValueError, match="must be uint8 with shape"):
        IndexerCache(2, 3, 64, 128, device="cpu", buffer=torch.zeros(1))


def test_named_region_bounds_are_checked_before_submission():
    agent = Mock()
    agent.get_mr_info.return_value = {"length": 128}

    _validate_named_region_ops(
        agent, "remote:0", [("kv_cache", "kv_cache", 0, 64, 64)], {"kv_cache": 64}
    )

    with pytest.raises(RuntimeError, match="exceeds region bounds"):
        _validate_named_region_ops(
            agent,
            "remote:0",
            [("kv_cache", "kv_cache", 1, 64, 64)],
            {"kv_cache": 64},
        )


def _placement(rank: int, domain: str = "fabric-a") -> dict:
    return {"rank": rank, "fabric_domain_id": domain}


def test_engine_placement_validation_preserves_all_rdma_workers():
    assert normalize_peer_placements([None, None], 2) == []


def test_engine_placement_validation_sorts_one_fabric_domain():
    placements = [_placement(1), _placement(0)]

    assert normalize_peer_placements(placements, 2) == [
        _placement(0),
        _placement(1),
    ]


def test_engine_placement_validation_rejects_partial_or_mixed_domains():
    with pytest.raises(RuntimeError, match="partial Fabric placement"):
        normalize_peer_placements([_placement(0), None], 2)
    with pytest.raises(RuntimeError, match="incompatible Fabric domains"):
        normalize_peer_placements([_placement(0), _placement(1, "fabric-b")], 2)

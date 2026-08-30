from types import SimpleNamespace

import pytest
import torch
from dlengine.runtime.context.cache.hca import DSV4_BYTES_PER_TOKEN
from dlengine.runtime.disagg.p2p.cache_layout import CacheTensorLayout
from dlengine.runtime.disagg.p2p.cache_transfer import (
    _coalesce_rdma_ops,
    P2PCacheTransfer,
)


def _layout(**overrides):
    values = dict(
        num_blocks=10,
        block_size=64,
        num_local_kv_heads=1,
        head_dim=128,
        dtype_itemsize=2,
        num_hidden_layers=3,
        mode="gqa",
    )
    values.update(overrides)
    return CacheTensorLayout(**values)


def test_gqa_remote_stage_uses_stage_local_layer_count_for_v_plane():
    layout = _layout(num_hidden_layers=2)
    block_bytes = 64 * 1 * 128 * 2

    assert layout.kv_stride(1, 1, 3) == (2 * 10 + 1 * 10 + 3) * block_bytes


def test_dsv4_hca_stride_includes_dummy_page_between_layers():
    layout = _layout(
        mode="dsv4",
        head_dim=512,
        dtype_itemsize=2,
        num_hidden_layers=2,
    )
    page_bytes = 64 * DSV4_BYTES_PER_TOKEN

    assert layout.kv_stride(0, 1, 3) == (11 + 3) * page_bytes
    assert layout.block_stride(1) == page_bytes
    with pytest.raises(ValueError, match="single cache plane"):
        layout.kv_stride(1, 0, 0)


def test_recurrent_mtp_handoff_is_registered_as_own_peer_memory_region():
    class FakeAgent:
        def __init__(self):
            self.registrations = {}

        def register_memory_region(self, name, address, offset, size):
            self.registrations[name] = (address, offset, size)
            return f"handle:{name}"

    class Transfer(P2PCacheTransfer):
        def __init__(self, cache_context):
            self._cache_context = cache_context
            super().__init__()

        @property
        def cache_context(self):
            return self._cache_context

    cache_context = SimpleNamespace(
        kv_cache=torch.zeros((1, 2, 4), dtype=torch.float16),
        mtp_handoff=torch.full((4, 6), -1, dtype=torch.int64),
        mtp_num_drafts=5,
        mode="mla",
        host_kv_cache=None,
        gdn_conv_states=None,
        gdn_recurrent_states=None,
        indexer_cache=None,
    )
    agent = FakeAgent()
    transfer = Transfer(cache_context)
    transfer.set_peer_agent_context(
        SimpleNamespace(alias="local", server_url="peer", agent=agent)
    )

    transfer.register_peer_agent_memory_regions(mode="decode")

    assert "kv_cache" in agent.registrations
    assert agent.registrations["mtp_handoff"][2] == 4 * 6 * 8
    assert transfer._local_mtp_handoff_mr_handler == "handle:mtp_handoff"


def test_recurrent_mtp_handoff_rdma_uses_remote_and_local_state_slots(monkeypatch):
    class Completion:
        def wait(self):
            return None

    class Endpoint:
        def __init__(self):
            self.ops = None

        def read(self, ops, _):
            self.ops = ops
            return Completion()

    class Agent:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def query_connection(self, _):
            return SimpleNamespace(endpoint=self.endpoint)

        def get_mr_info(self, peer_alias, name):
            return (
                {"length": handoff.numel() * handoff.element_size()}
                if name == "mtp_handoff"
                else None
            )

        def read(self, peer_alias, ops, _stream):
            self.endpoint.ops = ops
            return Completion()

    class PeerContext:
        alias = "local"
        server_url = "peer"

        def __init__(self, agent):
            self.agent = agent

        def is_connected(self, _):
            return True

    class Transfer(P2PCacheTransfer):
        def __init__(self, cache_context):
            self._cache_context = cache_context
            super().__init__()

        @property
        def cache_context(self):
            return self._cache_context

    handoff = torch.full((4, 6), -1, dtype=torch.int64)
    endpoint = Endpoint()
    transfer = Transfer(SimpleNamespace(mtp_handoff=handoff, mtp_num_drafts=5))
    transfer.set_peer_agent_context(PeerContext(Agent(endpoint)))
    transfer._local_mtp_handoff_mr_handler = "local:mtp_handoff"
    transfer._local_mr_sizes["mtp_handoff"] = handoff.numel() * handoff.element_size()
    transfer.remote_max_num_seqs["prefill"] = 4
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    transfer._execute_rdma_reads(
        {},
        {},
        mtp_handoff_assigns={"prefill": {"peer": [(3, 1)]}},
    )

    assert endpoint.ops == [("mtp_handoff", "mtp_handoff", 1 * 48, 3 * 48, 48)]


def test_rdma_ops_coalesce_only_when_both_regions_are_contiguous():
    ops = [
        ("local", "remote", 128, 1128, 64),
        ("local", "remote", 0, 1000, 64),
        ("other-local", "other-remote", 0, 0, 32),
        ("local", "remote", 64, 1064, 64),
        ("local", "remote", 192, 2000, 64),
    ]

    assert _coalesce_rdma_ops(ops, max_bytes=128) == [
        ("local", "remote", 0, 1000, 128),
        ("local", "remote", 128, 1128, 64),
        ("local", "remote", 192, 2000, 64),
        ("other-local", "other-remote", 0, 0, 32),
    ]


def test_rdma_ops_coalesce_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        _coalesce_rdma_ops([], max_bytes=0)


def test_rdma_reads_submit_all_before_wait_and_drain_after_failure(monkeypatch):
    events = []
    fail_alias = "peer-a"

    class Completion:
        def __init__(self, alias):
            self.alias = alias

        def wait(self):
            events.append(("wait", self.alias))
            if self.alias == fail_alias:
                raise RuntimeError(f"wait failed for {self.alias}")

    class Endpoint:
        def __init__(self, alias):
            self.alias = alias

        def read(self, _ops, _):
            events.append(("read", self.alias))
            return Completion(self.alias)

    endpoints = {alias: Endpoint(alias) for alias in ("peer-a", "peer-b")}

    class Agent:
        def query_connection(self, alias):
            return SimpleNamespace(endpoint=endpoints[alias])

        def get_mr_info(self, _peer_alias, name):
            return (
                {"length": handoff.numel() * handoff.element_size()}
                if name == "mtp_handoff"
                else None
            )

        def read(self, peer_alias, ops, stream):
            return endpoints[peer_alias].read(ops, stream)

    class PeerContext:
        alias = "local"
        server_url = "peer"
        agent = Agent()

        def is_connected(self, alias):
            return alias in endpoints

    class Transfer(P2PCacheTransfer):
        def __init__(self, cache_context):
            self._cache_context = cache_context
            super().__init__()

        @property
        def cache_context(self):
            return self._cache_context

    handoff = torch.full((4, 6), -1, dtype=torch.int64)
    transfer = Transfer(SimpleNamespace(mtp_handoff=handoff, mtp_num_drafts=5))
    transfer.set_peer_agent_context(PeerContext())
    transfer._local_mtp_handoff_mr_handler = "local:mtp_handoff"
    transfer._local_mr_sizes["mtp_handoff"] = handoff.numel() * handoff.element_size()
    transfer.remote_max_num_seqs["prefill"] = 4
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: events.append(("sync", None))
    )

    with pytest.raises(RuntimeError, match="wait failed for peer-a"):
        transfer._execute_rdma_reads(
            {},
            {},
            mtp_handoff_assigns={
                "prefill": {
                    "peer-a": [(0, 0)],
                    "peer-b": [(1, 1)],
                }
            },
        )

    kinds = [kind for kind, _ in events]
    assert kinds == ["read", "read", "wait", "wait", "sync"]
    assert {alias for kind, alias in events if kind == "read"} == {
        "peer-a",
        "peer-b",
    }
    assert {alias for kind, alias in events if kind == "wait"} == {
        "peer-a",
        "peer-b",
    }

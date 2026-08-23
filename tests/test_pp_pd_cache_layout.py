from types import SimpleNamespace

import pytest
import torch

from dlengine.runtime.context.cache.hca import DSV4_BYTES_PER_TOKEN
from dlengine.runtime.disagg.p2p.cache_layout import CacheTensorLayout
from dlengine.runtime.disagg.p2p.cache_transfer import P2PCacheTransfer


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
            return object() if name == "mtp_handoff" else None

        def get_handle(self, name, peer_alias=None):
            return f"remote:{name}"

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
    transfer.remote_max_num_seqs["prefill"] = 4
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    transfer._execute_rdma_reads(
        {},
        {},
        mtp_handoff_assigns={"prefill": {"peer": [(3, 1)]}},
    )

    assert endpoint.ops == [
        ("local:mtp_handoff", "remote:mtp_handoff", 3 * 48, 1 * 48, 48)
    ]

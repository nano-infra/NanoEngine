import torch

from nanodeploy.endpoint import rpc_endpoint


class _WriteFuture:
    @staticmethod
    def wait():
        return None


class _ServerEndpoint:
    def __init__(self):
        self.writes = []

    def write_with_imm(self, writes, immediate):
        self.writes.append((writes, immediate))
        return _WriteFuture()


class _RecvFuture:
    @staticmethod
    def wait():
        return None

    @staticmethod
    def imm_data():
        return rpc_endpoint._encode_imm(3, 1)


class _ClientEndpoint:
    @staticmethod
    def imm_recv():
        return _RecvFuture()


def test_server_writes_each_payload_to_its_transport_slot(monkeypatch):
    endpoint = object.__new__(rpc_endpoint.RPCServerEndpoint)
    endpoint.buffer_size = 20
    endpoint.num_slots = 2
    endpoint.slot_size = 10
    endpoint.world_size = 1
    endpoint.attention_sp = 1
    endpoint.attention_tp = 1
    endpoint.optimize_decode_block_table = True
    buffer = torch.empty(20, dtype=torch.int8)
    rdma = _ServerEndpoint()
    endpoint.server_bindings = [
        rpc_endpoint.EndpointBinding(rdma, buffer, 1_000)
    ]
    serialized = []

    def fake_serialize(ptr, capacity, *_args):
        serialized.append((ptr, capacity))
        return 3

    monkeypatch.setattr(rpc_endpoint, "serialize", fake_serialize)

    endpoint.send_seqs(
        [[]], is_prefill=False, transport_slot=1
    )

    base = buffer.data_ptr() + buffer.storage_offset()
    assert serialized == [(base + 10, 10)]
    assert rdma.writes == [
        (
            [(base + 10, 1_010, 0, 0, 3)],
            rpc_endpoint._encode_imm(3, 1),
        )
    ]


def test_client_deserializes_from_selected_transport_slot(monkeypatch):
    endpoint = object.__new__(rpc_endpoint.RPCClientEndpoint)
    endpoint.buffer_size = 20
    endpoint.num_slots = 2
    endpoint.slot_size = 10
    buffer = torch.empty(20, dtype=torch.int8)
    endpoint.client_binding = rpc_endpoint.EndpointBinding(
        _ClientEndpoint(), buffer, 0
    )
    deserialized = []

    def fake_deserialize(ptr, size):
        deserialized.append((ptr, size))
        return ["sequence"]

    monkeypatch.setattr(rpc_endpoint, "deserialize", fake_deserialize)

    result = endpoint.recv_seqs(transport_slot=1)

    base = buffer.data_ptr() + buffer.storage_offset()
    assert result == ["sequence"]
    assert deserialized == [(base + 10, 3)]

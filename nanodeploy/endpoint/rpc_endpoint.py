import dataclasses
import os

import torch
from dlslime import _slime_c
from nanodeploy._cpp import deserialize, Sequence, serialize
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")

_IMM_SLOT_BITS = 1
_IMM_SLOT_MASK = (1 << _IMM_SLOT_BITS) - 1
_IMM_MAX_PAYLOAD_BYTES = (1 << (31 - _IMM_SLOT_BITS)) - 1


def _encode_imm(payload_size: int, transport_slot: int) -> int:
    if not 0 <= payload_size <= _IMM_MAX_PAYLOAD_BYTES:
        raise ValueError(f"RPC payload size cannot fit immediate data: {payload_size}")
    if not 0 <= transport_slot <= _IMM_SLOT_MASK:
        raise ValueError(f"RPC slot cannot fit immediate data: {transport_slot}")
    return (payload_size << _IMM_SLOT_BITS) | transport_slot


def _decode_imm(immediate: int) -> tuple[int, int]:
    if immediate < 0:
        raise RuntimeError(f"invalid RPC immediate data {immediate}")
    return immediate >> _IMM_SLOT_BITS, immediate & _IMM_SLOT_MASK


def _get_slime_qp_num() -> int:
    raw = os.environ.get("SLIME_QP_NUM", "1")
    try:
        num_qp = int(raw)
    except ValueError:
        logger.warning("Invalid SLIME_QP_NUM=%r; falling back to 1", raw)
        return 1
    if num_qp < 1:
        logger.warning("Invalid SLIME_QP_NUM=%r; falling back to 1", raw)
        return 1
    return num_qp


@dataclasses.dataclass
class EndpointBinding:
    endpoint: _slime_c.RDMAEndpoint
    buffer: torch.Tensor
    remote_buffer_ptr: int


class RPCServerEndpoint:
    def __init__(self, buffer_size: int, world_size: int, attention_sp: int = 1, attention_tp: int = 1, optimize_decode_block_table: bool = True, num_slots: int = 1):
        if num_slots <= 0 or buffer_size < num_slots:
            raise ValueError("RPC endpoint slots must fit in the buffer")
        self.buffer_size = buffer_size
        self.num_slots = num_slots
        self.slot_size = buffer_size // num_slots
        self.world_size = world_size
        self.attention_sp = attention_sp
        self.attention_tp = attention_tp
        self.optimize_decode_block_table = optimize_decode_block_table

        self.devices = _slime_c.available_nic()
        self.num_qp = _get_slime_qp_num()
        self.server_bindings: list[EndpointBinding] = []

    def init_server_endpoint(self):
        self.server_bindings.clear()
        endpoint_info = []
        for i in range(self.world_size):
            endpoint = _slime_c.RDMAEndpoint(
                self.devices[i % len(self.devices)], num_qp=self.num_qp
            )
            buffer = torch.empty([self.buffer_size], dtype=torch.int8)
            endpoint.register_memory_region(
                buffer.data_ptr(), buffer.data_ptr(), buffer.numel()
            )
            self.server_bindings.append(EndpointBinding(endpoint, buffer, 0))
            endpoint_info.append(
                (endpoint.endpoint_info(), buffer.data_ptr() + buffer.storage_offset())
            )
        return endpoint_info

    def connect(self, client_info):
        for i, info in enumerate(client_info):
            self.server_bindings[i].endpoint.connect(info[0])
            self.server_bindings[i].remote_buffer_ptr = info[1]

    def send_seqs(
        self,
        dp_seqs: list[list[Sequence]],
        is_prefill: bool,
        *,
        transport_slot: int = 0,
    ):
        import time
        start = time.perf_counter()
        assert len(dp_seqs) == self.world_size
        if not 0 <= transport_slot < self.num_slots:
            raise ValueError(f"invalid RPC transport slot {transport_slot}")
        futures: list[_slime_c.SlimeReadWriteFuture] = []
        total_bytes = 0
        for i in range(self.world_size):
            binding = self.server_bindings[i]
            buffer = binding.buffer
            slot_offset = transport_slot * self.slot_size
            buffer_ptr = (
                buffer.data_ptr() + buffer.storage_offset() + slot_offset
            )
            
            # Decode optimize path still sends the full sequence skeleton.
            # The serializer only trims heavy per-target fields inside each
            # BlockContext (for example non-target block tables).
            if not is_prefill and self.optimize_decode_block_table:
                # rank = dp_rank * (attention_sp * attention_tp) + sp_rank * attention_tp + tp_rank
                # 所以 sp_rank = (rank // attention_tp) % attention_sp
                sp_rank = (i // self.attention_tp) % self.attention_sp
                sp_size = self.attention_sp
            else:
                # Prefill阶段或未启用优化：传输完整 BlockContext
                sp_rank = -1
                sp_size = -1
            
            off = serialize(
                buffer_ptr,
                self.slot_size,
                dp_seqs[i],
                is_prefill,
                sp_rank,
                sp_size,
            )
            total_bytes += off
            future = binding.endpoint.write_with_imm(
                [
                    (
                        buffer_ptr,
                        binding.remote_buffer_ptr + slot_offset,
                        0,
                        0,
                        off,
                    )
                ],
                _encode_imm(off, transport_slot),
            )
            futures.append(future)
        [future.wait() for future in futures]
        end = time.perf_counter()
        logger.info(
            f"[METRIC] dlslime_send_seqs_overhead_ms: {(end - start) * 1000:.4f}, "
            f"dlslime_send_seqs_BYTES: {total_bytes}"
        )

    def recv_tokens(self):
        pass


class RPCClientEndpoint:
    def __init__(self, buffer_size: int, rank, num_slots: int = 1):
        if num_slots <= 0 or buffer_size < num_slots:
            raise ValueError("RPC endpoint slots must fit in the buffer")
        self.buffer_size = buffer_size
        self.num_slots = num_slots
        self.slot_size = buffer_size // num_slots
        self.rank = rank

        self.devices = _slime_c.available_nic()
        self.num_qp = _get_slime_qp_num()
        self.client_binding: EndpointBinding

    def init_client_endpoint(self):
        endpoint_info = []
        endpoint = _slime_c.RDMAEndpoint(self.devices[0], num_qp=self.num_qp)
        buffer = torch.empty(
            [self.buffer_size], dtype=torch.int8, device="cpu", pin_memory=True
        )
        endpoint.register_memory_region(
            buffer.data_ptr(), buffer.data_ptr(), buffer.numel()
        )
        self.client_binding = EndpointBinding(endpoint, buffer, 0)

        return (endpoint.endpoint_info(), buffer.data_ptr() + buffer.storage_offset())

    def connect(self, server_info):
        self.client_binding.endpoint.connect(server_info[self.rank][0])
        self.client_binding.remote_buffer_ptr = server_info[self.rank][1]

    def recv_seqs(self, transport_slot: int = 0):
        if not 0 <= transport_slot < self.num_slots:
            raise ValueError(f"invalid RPC transport slot {transport_slot}")
        binding = self.client_binding
        future = binding.endpoint.imm_recv()
        future.wait()
        payload_size, received_slot = _decode_imm(future.imm_data())
        if received_slot != transport_slot:
            raise RuntimeError(
                "RPC sequence payload arrived for the wrong transport slot: "
                f"expected={transport_slot}, got={received_slot}"
            )
        if payload_size > self.slot_size:
            raise RuntimeError(
                "RPC sequence payload exceeds its transport slot: "
                f"payload={payload_size}, slot_size={self.slot_size}"
            )
        buffer = binding.buffer
        buffer_ptr = (
            buffer.data_ptr()
            + buffer.storage_offset()
            + transport_slot * self.slot_size
        )
        return deserialize(buffer_ptr, payload_size)

    def send_tokens(self):
        pass

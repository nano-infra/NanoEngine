import dataclasses

import torch
from dlslime import _slime_c
from nanodeploy._cpp import BlockContextSlot, deserialize, Sequence, serialize
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")


@dataclasses.dataclass
class EndpointBinding:
    endpoint: _slime_c.RDMAEndpoint
    buffer: torch.Tensor
    remote_buffer_ptr: int


class RPCServerEndpoint:
    def __init__(self, buffer_size: int, world_size: int):
        self.buffer_size = buffer_size
        self.world_size = world_size

        self.devices = _slime_c.available_nic()
        self.server_bindings: list[EndpointBinding] = []

    def init_server_endpoint(self):
        self.server_bindings.clear()
        endpoint_info = []
        for i in range(self.world_size):
            endpoint = _slime_c.RDMAEndpoint(self.devices[i % len(self.devices)])
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

    def send_seqs(self, dp_seqs: list[list[Sequence]], is_prefill: bool):
        assert len(dp_seqs) == self.world_size
        futures: list[_slime_c.SlimeReadWriteFuture] = []
        for i in range(self.world_size):
            binding = self.server_bindings[i]
            buffer = binding.buffer
            buffer_ptr = buffer.data_ptr() + buffer.storage_offset()
            off = serialize(buffer_ptr, buffer.numel(), dp_seqs[i], is_prefill)

            num_seqs = len(dp_seqs[i])
            total_tokens = sum(s.num_tokens for s in dp_seqs[i])
            total_blocks = 0
            for s in dp_seqs[i]:
                try:
                    ctx = s.block_ctx(BlockContextSlot.ACTIVE)
                    sp_size = ctx.attention_sp
                    for sp_idx in range(sp_size):
                        total_blocks += s.num_blocks(BlockContextSlot.ACTIVE, sp_idx)
                except Exception:
                    pass

            logger.info(
                f"Send sequences size: {off} bytes, "
                f"Total Sequences: {num_seqs}, "
                f"Total Tokens: {total_tokens}, "
                f"Total Blocks: {total_blocks}"
            )
            future = binding.endpoint.write_with_imm(
                [(buffer_ptr, binding.remote_buffer_ptr, 0, 0, off)], off
            )
            futures.append(future)
        [future.wait() for future in futures]

    def recv_tokens(self):
        pass


class RPCClientEndpoint:
    def __init__(self, buffer_size: int, rank):
        self.buffer_size = buffer_size
        self.rank = rank

        self.devices = _slime_c.available_nic()
        self.client_binding: EndpointBinding

    def init_client_endpoint(self):
        endpoint_info = []
        endpoint = _slime_c.RDMAEndpoint(self.devices[0])
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

    def recv_seqs(self):
        binding = self.client_binding
        future = binding.endpoint.imm_recv()
        future.wait()
        buffer = binding.buffer
        buffer_ptr = buffer.data_ptr() + buffer.storage_offset()
        logger.info(f"Received sequences size: {future.imm_data()} bytes")
        return deserialize(buffer_ptr, future.imm_data())

    def send_tokens(self):
        pass

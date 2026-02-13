import dataclasses

import torch
from dlslime import _slime_c
from nanodeploy._cpp import deserialize, Sequence, serialize
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")


@dataclasses.dataclass
class EndpointBinding:
    endpoint: _slime_c.RDMAEndpoint
    buffer: torch.Tensor
    remote_buffer_ptr: int


class RPCServerEndpoint:
    def __init__(self, buffer_size: int, world_size: int, attention_sp: int = 1, attention_tp: int = 1, optimize_decode_block_table: bool = True):
        self.buffer_size = buffer_size
        self.world_size = world_size
        self.attention_sp = attention_sp
        self.attention_tp = attention_tp
        self.optimize_decode_block_table = optimize_decode_block_table

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
        import time
        start = time.perf_counter()
        assert len(dp_seqs) == self.world_size
        futures: list[_slime_c.SlimeReadWriteFuture] = []
        total_bytes = 0
        for i in range(self.world_size):
            binding = self.server_bindings[i]
            buffer = binding.buffer
            buffer_ptr = buffer.data_ptr() + buffer.storage_offset()
            
            # 计算目标rank的sp_rank和sp_size
            # 如果启用优化且在Decode阶段，才进行过滤；否则传输全量BlockTable
            if not is_prefill and self.optimize_decode_block_table:
                # rank = dp_rank * (attention_sp * attention_tp) + sp_rank * attention_tp + tp_rank
                # 所以 sp_rank = (rank // attention_tp) % attention_sp
                sp_rank = (i // self.attention_tp) % self.attention_sp
                sp_size = self.attention_sp
            else:
                # Prefill阶段或未启用优化：传输全量BlockTable
                sp_rank = -1
                sp_size = -1
            
            off = serialize(buffer_ptr, buffer.numel(), dp_seqs[i], is_prefill, sp_rank, sp_size)
            total_bytes += off
            future = binding.endpoint.write_with_imm(
                [(buffer_ptr, binding.remote_buffer_ptr, 0, 0, off)], off
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
        return deserialize(buffer_ptr, future.imm_data())

    def send_tokens(self):
        pass

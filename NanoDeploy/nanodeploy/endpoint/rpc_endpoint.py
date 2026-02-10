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
    local_mr_handler: int = None  # Handler for local buffer MR
    remote_mr_handler: int = None  # Handler for remote buffer MR


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
            # Register local buffer MR with a name (as in control plane API)
            local_handler = endpoint.register_memory_region(
                "rpc_buffer",  # MR name
                buffer.data_ptr() + buffer.storage_offset(),
                buffer.numel(),
            )
            self.server_bindings.append(
                EndpointBinding(endpoint, buffer, 0, local_mr_handler=local_handler)
            )
            endpoint_info.append(
                (endpoint.endpoint_info(), buffer.data_ptr() + buffer.storage_offset())
            )
        return endpoint_info

    def connect(self, client_info):
        for i, info in enumerate(client_info):
            remote_endpoint_info = info[0]  # JSON endpoint info
            self.server_bindings[i].endpoint.connect(remote_endpoint_info)
            self.server_bindings[i].remote_buffer_ptr = info[1]

            # Register remote buffer MR from endpoint_info (as in control plane API)
            # remote_endpoint_info contains "mr_info" with MR information
            if (
                "mr_info" in remote_endpoint_info
                and "rpc_buffer" in remote_endpoint_info["mr_info"]
            ):
                remote_mr_info = remote_endpoint_info["mr_info"]["rpc_buffer"]
                remote_handler = self.server_bindings[
                    i
                ].endpoint.register_remote_memory_region(
                    "rpc_buffer",
                    remote_mr_info,
                )
                self.server_bindings[i].remote_mr_handler = remote_handler
                logger.debug(
                    f"Registered remote MR for server binding {i}: handler={remote_handler}"
                )
            else:
                logger.warning(
                    f"Remote MR info not found in endpoint_info for server binding {i}"
                )

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

            logger.debug(
                f"Send sequences size: {off} bytes, "
                f"Total Sequences: {num_seqs}, "
                f"Total Tokens: {total_tokens}, "
                f"Total Blocks: {total_blocks}"
            )
            # Use handler-based API if available, otherwise fall back to pointer-based API
            if (
                binding.local_mr_handler is not None
                and binding.remote_mr_handler is not None
            ):
                # New API: use handlers (as in control plane API)
                # Format: (local_handler, remote_handler, local_off, remote_off, length)
                future = binding.endpoint.write_with_imm(
                    [(binding.local_mr_handler, binding.remote_mr_handler, 0, 0, off)],
                    off,
                )
            else:
                # Fallback to old API: use pointers
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
        # Register local buffer MR with a name (as in control plane API)
        local_handler = endpoint.register_memory_region(
            "rpc_buffer",  # MR name
            buffer.data_ptr() + buffer.storage_offset(),
            buffer.numel(),
        )
        self.client_binding = EndpointBinding(
            endpoint, buffer, 0, local_mr_handler=local_handler
        )

        return (endpoint.endpoint_info(), buffer.data_ptr() + buffer.storage_offset())

    def connect(self, server_info):
        remote_endpoint_info = server_info[self.rank][0]  # JSON endpoint info
        self.client_binding.endpoint.connect(remote_endpoint_info)
        self.client_binding.remote_buffer_ptr = server_info[self.rank][1]

        # Register remote buffer MR from endpoint_info (as in control plane API)
        # remote_endpoint_info contains "mr_info" with MR information
        if (
            "mr_info" in remote_endpoint_info
            and "rpc_buffer" in remote_endpoint_info["mr_info"]
        ):
            remote_mr_info = remote_endpoint_info["mr_info"]["rpc_buffer"]
            remote_handler = self.client_binding.endpoint.register_remote_memory_region(
                "rpc_buffer",
                remote_mr_info,
            )
            self.client_binding.remote_mr_handler = remote_handler
            logger.debug(
                f"Registered remote MR for client binding: handler={remote_handler}"
            )
        else:
            logger.warning(
                f"Remote MR info not found in endpoint_info for client binding"
            )

    def recv_seqs(self):
        binding = self.client_binding
        future = binding.endpoint.imm_recv()
        future.wait()
        buffer = binding.buffer
        buffer_ptr = buffer.data_ptr() + buffer.storage_offset()
        size = future.imm_data()
        logger.debug(f"Received sequences size: {size} bytes")

        if size < 0 or size > buffer.numel():
            raise RuntimeError(
                f"Invalid FlatBuffer size: {size} (buffer capacity: {buffer.numel()})"
            )

        return deserialize(buffer_ptr, size)

    def send_tokens(self):
        pass

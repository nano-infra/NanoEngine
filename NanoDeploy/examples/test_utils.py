import os
import time

import dlslime
import numpy as np
import ray
import torch

# Shared constants
BLOCK_SIZE = 16 * 1024  # 16KB per block for testing
NUM_LAYERS = 2
NUM_KVS = 2  # K and V
BUFFER_ID = "kv_cache_buffer"


def cleanup_ports(start_port, num_ranks):
    ports = ",".join([str(start_port + i) for i in range(num_ranks)])
    print(f"Cleaning up ports: {ports}")

    # Use lsof to find and kill processes
    cmd = f"lsof -ti :{ports} | xargs -r kill -9"
    os.system(cmd)

    time.sleep(3)  # Wait longer for cleanup


@ray.remote(num_gpus=1)
class TestWorker:
    def __init__(self, rank, host, base_port):
        self.rank = rank
        self.host = host
        self.port = base_port + rank
        self.addr = f"{host}:{self.port}"

        # Initialize KV Cache: [Layers, KVs, NumBlocks, BlockSize]
        # Using CPU tensor for simplicity unless GPU is required for RDMA
        self.num_blocks = 128
        self.kv_cache = torch.zeros(
            (NUM_LAYERS, NUM_KVS, self.num_blocks, BLOCK_SIZE),
            dtype=torch.uint8,
            device="cpu",
        )

        # Fill with unique pattern: rank * 1000 + block_idx
        for l in range(NUM_LAYERS):
            for k in range(NUM_KVS):
                for b in range(self.num_blocks):
                    val = (self.rank + 1) * 100 + b
                    self.kv_cache[l, k, b, :].fill_(val % 256)

        # Start PeerAgent
        self.agent = dlslime.start_peer_agent(self.addr)

        # Register buffer
        ptr = self.kv_cache.data_ptr()
        size = self.kv_cache.numel() * self.kv_cache.element_size()
        self.agent.register_buffer(BUFFER_ID, ptr, size)

        print(f"Worker {rank} started at {self.addr}, registered buffer of size {size}")

    def get_addr(self):
        return self.addr

    def get_local_mr(self, remote_addr):
        return self.agent.get_local_mr_key(remote_addr, BUFFER_ID)

    def get_remote_mr(self, remote_addr):
        return self.agent.get_remote_mr_key(remote_addr, BUFFER_ID)

    def rdma_read(self, remote_addr, assignments):
        """
        assignments: list of (local_off, remote_off, length)
        """
        # Connect lazily if not connected
        # Python-level retry to handle ZMQ startup race conditions
        max_retries = 5
        for i in range(max_retries):
            try:
                self.agent.connect(remote_addr, timeout_ms=5000)
                break
            except Exception as e:
                if i == max_retries - 1:
                    print(
                        f"Rank {self.rank} failed to connect to {remote_addr} after {max_retries} attempts: {e}"
                    )
                    raise
                time.sleep(1)

        local_mr = self.get_local_mr(remote_addr)
        remote_mr = self.get_remote_mr(remote_addr)

        print(
            f"Rank {self.rank} -> {remote_addr}: local_mr={local_mr}, remote_mr={remote_mr}"
        )

        rdma_assigns = []
        for l_off, r_off, length in assignments:
            rdma_assigns.append((local_mr, remote_mr, r_off, l_off, length))

        future = self.agent.read(remote_addr, rdma_assigns)
        future.wait()
        return True

    def verify_data(self, assignments, expected_rank):
        """
        Verify that the data at local offsets matches expected_rank's patterns
        """
        for l_off, r_off, length in assignments:
            # Calculate block index from offset
            # offset = layer * (2 * num_blocks * block_size) + kv * (num_blocks * block_size) + b_idx * block_size
            # Simplified for this test: we assume offsets are block-aligned
            b_idx = (l_off % (self.num_blocks * BLOCK_SIZE)) // BLOCK_SIZE

            # Extract data from tensor
            # Since it's a 4D tensor, we need to be careful with offsets
            # For simplicity, we just use a flattened view
            flat_cache = self.kv_cache.view(-1)
            actual_data = flat_cache[l_off : l_off + 16]  # check first 16 bytes

            expected_val = (expected_rank + 1) * 100 + (
                r_off % (self.num_blocks * BLOCK_SIZE)
            ) // BLOCK_SIZE
            expected_val %= 256

            if not torch.all(actual_data == expected_val):
                print(
                    f"Rank {self.rank} Verification FAILED for remote {expected_rank} bloxk {r_off}! Expected {expected_val}, got {actual_data[0].item()}"
                )
                return False
        return True

    def clear_cache(self):
        self.kv_cache.zero_()
        return True

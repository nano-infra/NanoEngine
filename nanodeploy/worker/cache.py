import dataclasses
from collections import defaultdict
from typing import Literal

import dlslime

import torch
import torch.distributed as dist
from dlslime.assignment import Assignment

from nanodeploy.engine.sequence import Sequence


@dataclasses.dataclass
class CacheContext:
    num_kv_heads: int
    head_dim: int

    block_size: int

    num_hidden_layers: int

    attention_tp: int

    gpu_memory_utilization: float

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    mode: Literal["gqa", "mla"] = "gqa"

    num_local_kvcache_blocks = -1
    num_remote_kvcache_blocks: dict[str, int] = None

    kv_cache: torch.Tensor = None

    selected_nic: str | None = None
    endpoints: dict[str, dict[int, dlslime.RDMAEndpoint]] = None

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def __post_init__(self):

        assert self.mode == "gqa"
        assert self.attention_tp <= self.num_kv_heads

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        block_bytes = (
            2
            * self.num_hidden_layers
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

        self.num_local_kvcache_blocks = (
            int(total * self.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        assert self.num_local_kvcache_blocks > 0

        available_nics = dlslime.available_nic()
        self.selected_nic = available_nics[dist.get_rank() % len(available_nics)]
        assert self.selected_nic

        self.endpoints = {}
        self.num_remote_kvcache_blocks = {}

    def block_stride(self, block_idx: int):
        return (
            block_idx
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    def local_layer_stride(self, layer_idx: int, block_idx: int):
        return (
            self.num_local_kvcache_blocks * self.block_stride(1)
        ) * layer_idx + self.block_stride(block_idx)

    def remote_layer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return (
            self.num_remote_kvcache_blocks[remote_engine_id] * self.block_stride(1)
        ) * layer_idx + self.block_stride(block_idx)

    def local_kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int):
        return self.num_hidden_layers * self.local_layer_stride(
            1, 0
        ) * kv_idx + self.local_layer_stride(layer_idx, block_idx)

    def remote_kv_stride(
        self, kv_idx: int, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return self.num_hidden_layers * self.remote_layer_stride(
            1, 0, remote_engine_id
        ) * kv_idx + self.remote_layer_stride(layer_idx, block_idx, remote_engine_id)

    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks
        self.kv_cache = torch.empty(
            2,
            self.num_hidden_layers,
            self.num_local_kvcache_blocks,
            self.block_size,
            self.num_local_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ) -> dict[int, dict]:
        # init endpoint
        # register memory region
        endpoints = self.endpoints[remote_engine_name] = {}
        endpoints_info = {}
        self.num_remote_kvcache_blocks[remote_engine_name] = num_kv_blocks
        for i in range(remote_world_size):
            endpoint = dlslime.RDMAEndpoint(self.selected_nic, qp_num=2)
            endpoint.register_memory_region(
                mr_key="kv",
                addr=self.kv_cache.data_ptr(),
                offset=self.kv_cache.storage_offset(),
                length=self.kv_cache.numel() * self.kv_cache.itemsize,
            )
            endpoint_info = endpoint.endpoint_info
            endpoints[i] = endpoint
            endpoints_info[i] = endpoint_info
        return endpoints_info

    def p2p_connect(
        self, remote_engine_id: str, endpoints_info_list: list[dict[int, dict]]
    ):
        for i, endpoints_info in enumerate(endpoints_info_list):
            endpoint_info = endpoints_info[dist.get_rank()]
            self.endpoints[remote_engine_id][i].connect(endpoint_info)

    def migrate(self, seqs: list[Sequence]):
        assigns: dict[dict[int, list[Assignment]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for seq in seqs:
            for remote_block_idx, source_block_idx in zip(
                seq.block_table(seq.backup_engine_id),
                seq.block_table(seq.active_engine_id),
            ):
                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        assignment = Assignment(
                            mr_key="kv",
                            target_offset=self.remote_kv_stride(
                                kv_idx,
                                layer_idx,
                                remote_block_idx,
                                seq.backup_engine_id,
                            ),
                            source_offset=self.local_kv_stride(
                                kv_idx, layer_idx, source_block_idx
                            ),
                            length=self.block_stride(1),
                        )
                        assigns[seq.backup_engine_id][
                            seq.selected_replica(seq.backup_engine_id)
                        ].append(assignment)

            futures = []
            for endpoint_key, endpoint_assign_batch in assigns.items():
                for replica_key, assign_batch in endpoint_assign_batch.items():
                    futures.append(
                        self.endpoints[endpoint_key][replica_key].read_batch(
                            assign_batch, async_op=True
                        )
                    )

            [future.wait() for future in futures]


_CACHE_CONTEXT = None


def get_cache_context():
    return _CACHE_CONTEXT


def set_cache_context(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_hidden_layers: int,
    attention_tp: int,
    gpu_memory_utilization: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mode: Literal["gqa", "mla"] = "gqa",
):
    global _CACHE_CONTEXT
    _CACHE_CONTEXT = CacheContext(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_hidden_layers=num_hidden_layers,
        attention_tp=attention_tp,
        gpu_memory_utilization=gpu_memory_utilization,
        device=device,
        dtype=dtype,
        mode=mode,
    )
    return _CACHE_CONTEXT

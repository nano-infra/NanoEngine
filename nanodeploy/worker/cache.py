import dataclasses
from typing import Literal

import dlslime

import torch
import torch.distributed as dist

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

    num_kvcache_blocks = -1
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

        self.num_kvcache_blocks = (
            int(total * self.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        assert self.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2,
            self.num_hidden_layers,
            self.num_kvcache_blocks,
            self.block_size,
            self.num_local_kv_heads,
            self.head_dim,
        )

        available_nics = dlslime.available_nic()
        self.selected_nic = available_nics[dist.get_rank() % len(available_nics)]
        assert self.selected_nic

        self.endpoints = {}

    def p2p_init(
        self, remote_engine_name: str, remote_world_size: int
    ) -> dict[int, dict]:
        # init endpoint
        # register memory region
        endpoints = self.endpoints[remote_engine_name] = {}
        endpoints_info = {}
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
        for i in endpoints_info_list:
            endpoint_info = endpoints_info_list[i][dist.get_rank()]
            self.endpoints[remote_engine_id][i].connect(endpoint_info)

    def migrate(self, seqs: list[Sequence]):
        print("dummy migration")


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

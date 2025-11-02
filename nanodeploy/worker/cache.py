import dataclasses
from typing import Literal

import torch


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

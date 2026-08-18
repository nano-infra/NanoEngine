from dataclasses import dataclass

import torch

from dlengine.context import BaseContext
from dlengine.context.cache._backend import CacheBackend, CacheKind


@dataclass
class GQAContext(BaseContext):
    kv_cache: torch.Tensor | None = None

    @classmethod
    def get_context_type(cls) -> str:
        return "gqa"

    @classmethod
    def get_context_name(cls) -> str:
        return "GQAContext"

    def clear_context(self) -> None:
        self.kv_cache = None

    def reset_context(self) -> None:
        self.clear_context()


_GQA_CONTEXT = GQAContext()


def get_gqa_context() -> GQAContext:
    return _GQA_CONTEXT


def reset_gqa_context() -> None:
    global _GQA_CONTEXT
    _GQA_CONTEXT = GQAContext()


def configure_gqa_cache(context) -> None:
    assert context.attention_tp <= context.num_kv_heads
    context._fp8_head_dim = 0


def get_gqa_block_bytes(context) -> int:
    return (
        2
        * context.num_hidden_layers
        * context.block_size
        * context.num_local_kv_heads
        * context.head_dim
        * context.dtype.itemsize
    )


def allocate_gqa_kvcache(context) -> None:
    context.kv_cache = torch.empty(
        2,
        context.num_hidden_layers,
        context.num_local_kvcache_blocks,
        context.block_size,
        context.num_local_kv_heads,
        context.head_dim,
        dtype=context.dtype,
        device=context.device,
    )


GQA_CACHE_BACKEND = CacheBackend(
    kind=CacheKind.GQA,
    configure=configure_gqa_cache,
    get_block_bytes=get_gqa_block_bytes,
    allocate=allocate_gqa_kvcache,
)


__all__ = [
    "GQA_CACHE_BACKEND",
    "GQAContext",
    "allocate_gqa_kvcache",
    "configure_gqa_cache",
    "get_gqa_block_bytes",
    "get_gqa_context",
    "reset_gqa_context",
]

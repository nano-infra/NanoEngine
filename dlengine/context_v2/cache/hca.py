from dataclasses import dataclass
from typing import Any

import torch

from dlengine.context_v2 import BaseContext
from dlengine.context_v2.cache._registry import register_cache_backend
from dlengine.logging import get_logger

logger = get_logger("dlengine")

DSV4_BYTES_PER_TOKEN = 584


def configure_dsv4_cache(context) -> None:
    assert context.attention_tp == 1
    assert context.block_size % 64 == 0, "DSv4 block_size must be multiple of 64"
    context.num_kv_heads = 1
    context.head_dim = 512
    context._fp8_head_dim = 0


def get_dsv4_block_bytes(context) -> int:
    return context.num_hidden_layers * context.block_size * DSV4_BYTES_PER_TOKEN


def allocate_dsv4_kvcache(context) -> None:
    # DSv4 HCA paged cache: [num_layers, num_pages+1, page_size, 1, 584].
    # Extra +1 page is a dummy absorbing invalid writes (graph-safe).
    context.kv_cache = torch.zeros(
        context.num_hidden_layers,
        context.num_local_kvcache_blocks + 1,
        context.block_size,
        1,
        DSV4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=context.device,
    )
    logger.info(
        f"DSv4 HCA cache: {context.kv_cache.shape} (incl dummy page), "
        f"{context.kv_cache.nelement() / 1e9:.2f} GB"
    )


DSV4_CACHE_BACKEND = register_cache_backend(
    "dsv4",
    configure=configure_dsv4_cache,
    get_block_bytes=get_dsv4_block_bytes,
    allocate=allocate_dsv4_kvcache,
)


@dataclass
class HCAContext(BaseContext):
    tile_scheduler_metadata: Any = None
    kv_cache: torch.Tensor | None = None

    @classmethod
    def get_context_type(cls) -> str:
        return "hca"

    @classmethod
    def get_context_name(cls) -> str:
        return "HCAContext"

    def clear_context(self) -> None:
        self.tile_scheduler_metadata = None
        self.kv_cache = None

    def reset_context(self) -> None:
        self.clear_context()


_HCA_CONTEXT = HCAContext()


def get_hca_context() -> HCAContext:
    return _HCA_CONTEXT


def set_hca_context(
    tile_scheduler_metadata: Any = None,
    kv_cache: torch.Tensor | None = None,
) -> HCAContext:
    global _HCA_CONTEXT
    _HCA_CONTEXT = HCAContext(
        tile_scheduler_metadata=tile_scheduler_metadata,
        kv_cache=kv_cache,
    )
    return _HCA_CONTEXT


def reset_hca_context() -> None:
    global _HCA_CONTEXT
    _HCA_CONTEXT = HCAContext()


__all__ = [
    "DSV4_CACHE_BACKEND",
    "DSV4_BYTES_PER_TOKEN",
    "HCAContext",
    "allocate_dsv4_kvcache",
    "configure_dsv4_cache",
    "get_dsv4_block_bytes",
    "get_hca_context",
    "reset_hca_context",
    "set_hca_context",
]

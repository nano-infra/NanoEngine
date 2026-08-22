from dataclasses import dataclass
from typing import Any

import torch

from dlengine.runtime.context import BaseContext
from dlengine.runtime.context.cache._backend import CacheBackend, CacheKind

FP8_QUANT_TILE_SIZE = 128


@dataclass
class MLAContext(BaseContext):
    kv_cache: torch.Tensor | None = None
    sparse_tile_scheduler_metadata: Any = None

    @classmethod
    def get_context_type(cls) -> str:
        return "mla"

    @classmethod
    def get_context_name(cls) -> str:
        return "MLAContext"

    def clear_context(self) -> None:
        self.kv_cache = None
        self.sparse_tile_scheduler_metadata = None

    def reset_context(self) -> None:
        self.clear_context()


_MLA_CONTEXT = MLAContext()


def get_mla_context() -> MLAContext:
    return _MLA_CONTEXT


def reset_mla_context() -> None:
    global _MLA_CONTEXT
    _MLA_CONTEXT = MLAContext()


def configure_mla_cache(context) -> None:
    assert context.block_size == 64, "MLA mode only support block_size=64"
    context.num_kv_heads = 1
    context.head_dim = context.kv_lora_rank + context.qk_rope_head_dim


def get_mla_block_bytes(context) -> int:
    if context.is_fp8_kvcache:
        # FP8 MLA layout per token:
        #   NoPE:  kv_lora_rank bytes (float8_e4m3fn)
        #   Scale: (kv_lora_rank // tile_size) * 4 bytes (float32 per tile)
        #   RoPE:  qk_rope_head_dim * 2 bytes (bfloat16)
        nope_bytes = context.kv_lora_rank
        scale_bytes = (context.kv_lora_rank // FP8_QUANT_TILE_SIZE) * 4
        rope_bytes = context.qk_rope_head_dim * 2
        context._fp8_head_dim = nope_bytes + scale_bytes + rope_bytes
        return (
            context.num_hidden_layers
            * (context.block_size + 1)
            * 1  # num_kv_heads
            * context._fp8_head_dim
            * 1  # fp8 element size
        )

    context._fp8_head_dim = 0
    return (
        context.num_hidden_layers
        * context.block_size
        * context.num_local_kv_heads
        * context.head_dim
        * context.dtype.itemsize
    )


def allocate_mla_kvcache(context) -> None:
    device = torch.device(context.device)
    cpu_pinned = device.type == "cpu" and torch.cuda.is_available()
    if context.is_fp8_kvcache:
        # FP8 MLA: allocate (block_size+1) rows per block for stride padding,
        # then slice back to block_size. This ensures the FlashMLA kernel
        # never reads out-of-bounds on the last row.
        kv_cache_padded = torch.empty(
            1,
            context.num_hidden_layers,
            context.num_local_kvcache_blocks,
            context.block_size + 1,
            1,
            context._fp8_head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
            pin_memory=cpu_pinned,
        )
        context.kv_cache = kv_cache_padded[:, :, :, : context.block_size, :, :]
        return

    context.kv_cache = torch.empty(
        1,
        context.num_hidden_layers,
        context.num_local_kvcache_blocks,
        context.block_size,
        context.num_local_kv_heads,
        context.head_dim,
        dtype=context.dtype,
        device=device,
        pin_memory=cpu_pinned,
    )


MLA_CACHE_BACKEND = CacheBackend(
    kind=CacheKind.MLA,
    configure=configure_mla_cache,
    get_block_bytes=get_mla_block_bytes,
    allocate=allocate_mla_kvcache,
)


__all__ = [
    "FP8_QUANT_TILE_SIZE",
    "MLA_CACHE_BACKEND",
    "MLAContext",
    "allocate_mla_kvcache",
    "configure_mla_cache",
    "get_mla_block_bytes",
    "get_mla_context",
    "reset_mla_context",
]

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


def resolve_mla_cache_format(
    config, cache_plan, hardware_backend: str, *, force_reference: bool = False
) -> tuple[bool, bool]:
    """Return (FP8 enabled, raw FP8 layout) without allocating any cache.

    Dense Blackwell MLA supports raw E4M3 pages independently of the indexer.
    Hopper FP8 decode requires the packed sparse layout. Automatic selection
    retains the existing indexer-driven policy.
    """
    requested = getattr(config, "kv_cache_dtype", "auto")
    if requested not in ("auto", "bfloat16", "fp8_e4m3"):
        raise ValueError(f"Unsupported kv_cache_dtype={requested!r}")
    if requested != "auto" and not cache_plan.has_mla():
        raise ValueError("Explicit kv_cache_dtype is currently supported only for MLA")
    hf = config.hf_config
    sparse = (
        cache_plan.has_indexer()
        and getattr(hf, "index_head_dim", 0) > 0
        and not getattr(config, "disable_nsa", False)
    )
    if requested == "bfloat16" and sparse:
        raise ValueError(
            "kv_cache_dtype=bfloat16 requires disable_nsa=True for sparse MLA; "
            "the sparse decode path requires FP8 KV cache"
        )
    fp8 = requested == "fp8_e4m3" or (requested == "auto" and sparse)
    rank = getattr(hf, "kv_lora_rank", 0)
    rope = getattr(hf, "qk_rope_head_dim", 0)
    reference_decode = force_reference or (
        getattr(config, "enable_mla_reference_fallback", False)
        and rank + rope not in (512, 576)
    )
    if reference_decode:
        if requested == "fp8_e4m3":
            raise ValueError(
                "kv_cache_dtype=fp8_e4m3 is incompatible with MLA reference decode"
            )
        fp8 = False
    if requested == "fp8_e4m3":
        if (rank, rope) != (512, 64):
            raise ValueError(
                "Explicit FP8 MLA cache requires kv_lora_rank=512 and "
                f"qk_rope_head_dim=64, got {rank} and {rope}"
            )
        if hardware_backend != "blackwell" and not (
            hardware_backend == "hopper" and sparse
        ):
            raise ValueError(
                "FP8 MLA cache requires Blackwell dense/sparse MLA or Hopper "
                "sparse MLA; dense Hopper FP8 decode is not supported"
            )
    if cache_plan.has_hisparse() and cache_plan.has_mla() and not fp8:
        raise ValueError("HiSparse requires FP8 MLA KV cache")
    return bool(fp8), bool(fp8 and hardware_backend == "blackwell")


def configure_mla_cache(context) -> None:
    assert context.block_size == 64, "MLA mode only support block_size=64"
    context.num_kv_heads = 1
    context.head_dim = context.kv_lora_rank + context.qk_rope_head_dim


def get_mla_block_bytes(context) -> int:
    if context.is_fp8_kvcache:
        if context.raw_fp8_mla_layout:
            context._fp8_head_dim = context.head_dim
            return (
                context.num_hidden_layers
                * context.block_size
                * context.num_local_kv_heads
                * context.head_dim
            )
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
        if context.raw_fp8_mla_layout:
            from dlengine.runtime.context.cache._allocator import allocate_device_tensor

            context.kv_cache = allocate_device_tensor(
                context,
                "kv_cache",
                (
                    1,
                    context.num_hidden_layers,
                    context.num_local_kvcache_blocks,
                    context.block_size,
                    1,
                    context.head_dim,
                ),
                torch.float8_e4m3fn,
            )
            return
        # FP8 MLA: allocate (block_size+1) rows per block for stride padding,
        # then slice back to block_size. This ensures the FlashMLA kernel
        # never reads out-of-bounds on the last row.
        from dlengine.runtime.context.cache._allocator import allocate_device_tensor

        kv_cache_padded = allocate_device_tensor(
            context,
            "kv_cache",
            (
                1,
                context.num_hidden_layers,
                context.num_local_kvcache_blocks,
                context.block_size + 1,
                1,
                context._fp8_head_dim,
            ),
            torch.float8_e4m3fn,
        )
        context.kv_cache = kv_cache_padded[:, :, :, : context.block_size, :, :]
        return

    from dlengine.runtime.context.cache._allocator import allocate_device_tensor

    context.kv_cache = allocate_device_tensor(
        context,
        "kv_cache",
        (
            1,
            context.num_hidden_layers,
            context.num_local_kvcache_blocks,
            context.block_size,
            context.num_local_kv_heads,
            context.head_dim,
        ),
        context.dtype,
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
    "resolve_mla_cache_format",
]

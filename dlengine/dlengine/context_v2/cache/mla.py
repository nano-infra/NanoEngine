import torch

FP8_QUANT_TILE_SIZE = 128


def configure_mla_cache(context) -> None:
    assert context.attention_tp == 1
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
            * context.block_size
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
            device=context.device,
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
        device=context.device,
    )


__all__ = [
    "FP8_QUANT_TILE_SIZE",
    "allocate_mla_kvcache",
    "configure_mla_cache",
    "get_mla_block_bytes",
]

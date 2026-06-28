import torch


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


__all__ = [
    "allocate_gqa_kvcache",
    "configure_gqa_cache",
    "get_gqa_block_bytes",
]

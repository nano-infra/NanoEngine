INDEXER_QUANT_BLOCK_SIZE = 128


def get_indexer_block_bytes(context) -> int:
    if context.index_head_dim <= 0:
        return 0
    indexer_bytes_per_token = (
        context.index_head_dim + context.index_head_dim // INDEXER_QUANT_BLOCK_SIZE * 4
    )
    return context.num_hidden_layers * context.block_size * indexer_bytes_per_token


def allocate_indexer_cache(context, hf_config) -> None:
    """Allocate NSA indexer FP8 cache for DeepSeek V3.2."""
    from dlengine.layers.indexer import IndexerCache

    index_head_dim = getattr(hf_config, "index_head_dim", 0)
    if index_head_dim == 0:
        return

    context.indexer_cache = IndexerCache(
        num_layers=context.num_hidden_layers,
        num_pages=context.num_local_kvcache_blocks,
        page_size=context.block_size,
        head_dim=index_head_dim,
        device=context.device,
    )


__all__ = [
    "INDEXER_QUANT_BLOCK_SIZE",
    "allocate_indexer_cache",
    "get_indexer_block_bytes",
]

"""NSA Lightning-Indexer for the DSA sparse-attention family.

The indexer scores keys and returns per-query top-k logical (and optionally
physical) indices that the sparse MLA kernel then attends. It is consumed both
by ``DsaAttention`` and directly by MTP / graph-capture runners for state
seeding and reuse, so its public API is re-exported here (and, during
migration, from the compatibility shim ``dlengine.runtime.layers.indexer``).
"""

from dlengine.runtime.layers.backends.dsa.indexer.lightning import (
    INDEXER_QUANT_BLOCK_SIZE,
    Indexer,
    IndexerCache,
    _expand_decode_context_lens,
    _hadamard_rotate,
    _interleaved_to_half,
    _per_token_cast_to_fp8_ue8m0,
    _prefill_mqa_chunk_rows,
    _uses_linear_mtp_indexer_path,
    _weighted_relu_mqa_scores,
    append_pool_tail,
    pool_indexer_states,
    pool_indexer_topk,
)

__all__ = [
    "INDEXER_QUANT_BLOCK_SIZE",
    "Indexer",
    "IndexerCache",
    "_expand_decode_context_lens",
    "_hadamard_rotate",
    "_interleaved_to_half",
    "_per_token_cast_to_fp8_ue8m0",
    "_prefill_mqa_chunk_rows",
    "_uses_linear_mtp_indexer_path",
    "_weighted_relu_mqa_scores",
    "append_pool_tail",
    "pool_indexer_states",
    "pool_indexer_topk",
]

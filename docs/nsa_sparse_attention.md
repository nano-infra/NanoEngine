# NSA / DSA Sparse Attention (DeepSeek V3.2, GLM-5.1-FP8)

This document covers the Native Sparse Attention (NSA / DSA) path used by
DeepSeek-V3.2-style models (e.g. `GLM-5.1-FP8`), the bugs that were fixed, and
the remaining work for fully scalable long-context serving.

## Overview

These models are *trained* with a lightning **indexer** that, for every query,
selects only the top-`index_topk` (≈2048 for GLM-5.1-FP8) most relevant cached
keys. Attention then runs only over that sparse selection. Dense full attention
is a pattern the model never saw during training, so once a sequence exceeds
`index_topk` the dense path diverges and corrupts long-context output.

Key components:

- `Indexer` (`dlengine/layers/indexer.py`): scores cached KV blocks via the FP8
  lightning indexer and selects the per-query top-k.
- `IndexerCache.store_key_fp8`: quantizes indexer keys to FP8 + per-token scale
  and writes them into the paged buffer.
- `deep_gemm.fp8_paged_mqa_logits`: the decode-time kernel that computes the
  indexer logits from the paged FP8 cache.
- `flash_mla_sparse_fwd` / `flash_mla_with_kvcache`: FlashMLA kernels that run
  the actual sparse attention for prefill / decode.

## Fixed: indexer cache byte layout (root cause of long-context garbling)

**Symptom.** Coherent output for prompts up to ~`index_topk` (~2k–5k) tokens,
then progressively garbled (repeated digits, U+FFFD replacement chars) for
longer prompts — in both eager and CUDA-graph modes, and regardless of whether
prefill ran sparse or dense.

**Root cause.** `store_key_fp8` wrote each token's data **interleaved**:

```
[tok0_fp8(128) | tok0_scale(4)][tok1_fp8(128) | tok1_scale(4)] ...
```

But `deep_gemm.fp8_paged_mqa_logits` (and SGLang's `index_buf_accessor`
kernels) expect each page laid out **block-contiguous**: all tokens' FP8 first,
then all tokens' scales, i.e. `SCALE_OFFSET = page_size * head_dim`:

```
[tok0_fp8(128) .. tok63_fp8(128)] [tok0_scale(4) .. tok63_scale(4)]
```

So the kernel read "scale" bytes out of the middle of the FP8 data, producing
garbage key scales and therefore garbage logits (`|logit| ~ 1e29`, many `+inf`;
~90% of finite logits `> 1e6` when real logits are O(100s)).

**Why the ~2k boundary.** For `ctx <= index_topk`, decode top-k selects *all*
available tokens, so the garbage logit values don't change the selection and
output stays coherent. Beyond `index_topk`, real ranking kicks in, garbage
scores dominate the top-k, the wrong keys are attended, and output degrades.

**Fix.** `IndexerCache.store_key_fp8` now writes the block-contiguous layout,
matching SGLang's authoritative store/load kernels
(`index_buf_accessor`: `S_OFFSET_NBYTES_IN_PAGE = page_size * index_head_dim`):

```python
fp8_byte_offset   = offset_in_page * head_dim
scale_byte_offset = page_size * head_dim + offset_in_page * 4
```

**Validation.** Needle-in-haystack at 4k / 8k / 13k tokens: both needles
retrieved, zero replacement chars, fully coherent (previously garbled past ~5k).

### Secondary safety nets (kept)

- **Logit sanitization** before top-k: non-finite / saturated logits are mapped
  to `-inf` so they can never be selected. With the layout fixed this should
  rarely trigger, but it guards against residual FP8 outliers.
- **Attention sink**: the first `DLENGINE_DSV4_DEBUG_NSA_NUM_SINKS` (default 1)
  leading tokens are forced into the selection (token 0 is an essential sink in
  NSA models).

## Future work

### 1. Sparse chunked prefill (prompts beyond a single chunk)

Sparse prefill currently only fires for a **fresh, single-chunk** prompt
(`total_cached == 0`), gated in `deepseek_v2.py`. When a prompt exceeds the
single-chunk budget (`max_num_batched_tokens`), the engine splits it into
chunks; every chunk after the first has a cached prefix (`total_cached > 0`) and
falls back to **dense** attention.

Consequences for long prompts (observed at ~18k–28k tokens):

- **Divergence** — dense attention over `> index_topk` keys is off-distribution.
- **OOM** — the dense fallback (`hopper/attention.py::_interleave_cached_fresh`)
  materializes the full cached+fresh KV and can exhaust GPU memory on top of the
  large `max_model_len` KV reservation.

**Plan.** Implement paged sparse chunked-prefill: for each fresh query position
in a chunk, run the lightning indexer over `[cached prefix keys ++ fresh keys]`
(reading the cached indexer keys from the paged buffer via the now-correct
block-contiguous layout), select top-`index_topk`, and run `flash_mla_sparse_fwd`
against the paged KV. This removes the dense fallback entirely and bounds memory
to the sparse selection rather than the full prefix.

### 2. Sparse prefix-cache prefill

Same mechanism as (1), for the prefix-cache hit case (warm `total_cached > 0`
with no chunking): reuse cached indexer keys from the paged buffer and run the
indexer + sparse attention over the combined prefix, instead of the dense
fallback. Depends on the same paged sparse-prefill primitive as chunked prefill.

### Acceptance criteria for (1)/(2)

- No dense-attention fallback on the NSA path for any prompt length.
- Coherent needle retrieval at ≥ 64k tokens (multi-chunk) with no OOM at the
  configured `max_model_len`.
- Decode parity preserved (single-chunk results unchanged).

## Relevant debug env vars

All must be prefixed `DLENGINE_DSV4_DEBUG_` to be forwarded to Ray workers
(see `engine/ray_executor.py`).

- `DLENGINE_DSV4_DEBUG_NSA_SPARSE_PREFILL` — enable/disable sparse prefill
  (default on). Leftover `=0` from a debug session silently disables it.
- `DLENGINE_DSV4_DEBUG_NSA_NUM_SINKS` — number of forced attention-sink tokens
  (default 1).
- `DLENGINE_DSV4_DEBUG_NSA_DUMP` / `..._NSA_DUMP_SELECT` — one-shot dump of the
  learned selection plus logit-magnitude stats (used to diagnose the layout bug).
- `DLENGINE_DSV4_DEBUG_NSA_SELECT` — force a selection strategy
  (`trailing`, `sink_recent`) for decode, for A/B isolation.

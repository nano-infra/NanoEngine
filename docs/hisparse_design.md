# HiSparse Design for DLEngine

## 1. Goal

This document sketches a staged HiSparse implementation for DLEngine.

The first target is **DeepSeek V3.2 / NSA decode-only**. This is intentionally
smaller than a full SGLang-style HiSparse port:

- Keep the NSA `IndexerCache` resident on GPU.
- Offload the main MLA KV cache to a host/cold tier.
- Keep a smaller GPU hot KV buffer for the sparse tokens selected at decode.
- Translate the indexer's top-k logical token positions into hot-buffer indices
  before calling `flash_mla_with_kvcache(indices=...)`.

The initial design optimizes for correctness and debuggability. Performance
optimizations such as async prefetch, page reuse, and CUDA graph capture can be
layered on after the basic mapping is stable.

## 2. Why Start With DSV3.2 Decode

DSV3.2 decode is the cleanest entry point because the existing sparse path is
already explicit:

- `Indexer` scores cached keys and returns top-`index_topk` logical token
  positions.
- `topk_indices_to_physical` currently maps those logical positions through the
  normal block table into full GPU KV cache slots.
- `FlashMLAImpl` already accepts sparse physical indices and dispatches
  `flash_mla_with_kvcache`.

HiSparse changes only the middle step:

```text
current:
  topk logical token -> normal physical slot -> full GPU KV cache

hisparse:
  topk logical token -> cold/source slot -> swap into hot GPU buffer
                    -> hot physical slot -> hot GPU KV cache
```

This avoids the DSv4 complications of SWA ring cache plus CSA compressed pages
and lets us validate the core hot/cold machinery first.

## 3. Non-Goals For MVP

The MVP does not attempt to solve every cache tiering case:

- No prefill offload path.
- No sparse chunked-prefill changes.
- No DSv4 C4/CSA support.
- No PD disaggregation integration.
- No L3/3FS integration.
- No multi-node host-cache sharing.
- No attempt to offload the NSA indexer cache.
- No aggressive async scheduling in the first correctness pass.

These constraints keep the first implementation small enough to reason about.

## 4. Existing Pieces

The current code already provides most of the sparse attention surface.

### 4.1 NSA Indexer

`dlengine/runtime/layers/indexer.py` owns the decode-time selector:

- Computes query/key features for the lightning indexer.
- Stores indexer keys in `IndexerCache`.
- Calls `deep_gemm.fp8_paged_mqa_logits`.
- Sanitizes logits.
- Returns top-k logical token positions.

The MVP should leave `IndexerCache` unchanged and GPU-resident. The indexer must
be able to score the full context even when the main MLA KV is offloaded.

### 4.2 Sparse MLA Decode

`dlengine/runtime/layers/hopper/attention.py::FlashMLAImpl` supports sparse FP8 decode:

- Receives `sparse_indices`.
- Reshapes them to `[bs, ntps, topk]`.
- Calls `flash_mla_with_kvcache(..., indices=indices_3d)`.

HiSparse should preserve this contract. The only difference is that
`sparse_indices` will point into the HiSparse hot KV buffer rather than the
normal full KV cache.

### 4.3 Normal KV Allocation

`CacheContext` currently allocates full GPU MLA KV and optional indexer cache.
For HiSparse, the normal MLA KV allocation becomes too large for the target use
case. We need a new cache mode or configuration flag that allocates:

- A host/cold MLA KV store.
- A smaller GPU hot MLA KV store.
- Mapping tensors connecting logical/full slots to hot slots.

## 5. Proposed MVP Architecture

### 5.1 Components

Add a small set of HiSparse-specific components:

```text
HiSparseConfig
  enable_hisparse
  hisparse_hot_tokens
  hisparse_page_size
  hisparse_pin_host_memory
  hisparse_debug_checks

HiSparseMLACache
  host_k_cache
  hot_k_cache
  full_to_hot_mapping
  hot_to_full_mapping
  hot_page_ref/state metadata

HiSparseCoordinator
  prepare_decode_indices(...)
  swap_in_selected_tokens_or_pages(...)
  release_request(...)
  clear()
```

The naming can stay simple at first. The important boundary is that model code
should not directly manage host pages or hot-buffer allocation. It should ask the
coordinator for hot sparse indices.

### 5.2 Data Flow

Decode flow for one DSV3.2 attention layer:

```text
1. Store current token's MLA K into the cold/source cache.

2. Run the NSA indexer against the GPU-resident IndexerCache.
   Output: topk_indices, logical token positions.

3. Convert topk logical positions to source slots using the normal block table.
   Output: source_indices.

4. HiSparseCoordinator ensures selected source slots are present in hot_k_cache.
   Output: hot_indices.

5. FlashMLA reads hot_k_cache using hot_indices.

6. Request/layer lifetime code releases hot entries when they are no longer
   pinned by active decode batches.
```

Step 1 is the awkward part: in the current implementation the attention backend
stores K directly into `k_cache`. Under HiSparse, `k_cache` passed to FlashMLA
should be the hot cache, but the newly produced K must also become available in
the cold/source store. The cleanest shape is to make `FlashMLAImpl` delegate KV
store to a cache object rather than writing only to the tensor it later reads.

### 5.3 Token Versus Page Granularity

The first implementation should swap at **page granularity**, not individual
token granularity.

Reasons:

- Normal block tables already describe pages.
- H2D copies are more efficient at page granularity.
- It reduces duplicate copies when top-k contains many tokens from the same
  page.
- It matches future allocator accounting better.

The top-k result is token-level, so the coordinator still returns token-level
hot indices:

```text
logical token -> source page + offset
source page   -> hot page
hot index     -> hot page * page_size + offset
```

For MLA V3.2 the page size is expected to match the current MLA cache block size
(normally 64).

## 6. Cache Layout

### 6.1 Cold Store

The cold store should use the same per-token layout as the current FP8 MLA KV
cache so copies can be byte-preserving.

Open question for implementation:

- CPU torch tensor with pinned memory is simplest.
- A CUDA-visible host allocation or custom extension may be needed later for
  higher bandwidth and fewer Python-level copies.

For the first pass:

```python
host_k_cache: torch.Tensor  # CPU pinned, same logical page shape as MLA k_cache
```

### 6.2 Hot Store

The hot store is a normal GPU tensor accepted by FlashMLA:

```python
hot_k_cache: torch.Tensor  # CUDA, [hot_pages, page_size, 1, fp8_head_dim]
```

It should include one dummy page or dummy slot if that is required to keep
invalid indices graph-safe. Invalid top-k entries should still become `-1` for
FlashMLA unless the kernel contract requires a dummy slot.

### 6.3 Mapping Tensors

Minimum mapping state:

```text
full_page_to_hot_page: int32[num_full_pages]  # -1 if not resident
hot_page_to_full_page: int32[num_hot_pages]   # -1 if free
hot_page_refcount:     int32[num_hot_pages]
```

For graph compatibility, keep tensor shapes fixed and update contents in-place.
Host-side mirrors may be useful for allocator decisions, but the source of truth
for decode kernels should be fixed-size tensors.

## 7. Scheduler And Lifetime

The scheduler must eventually know that HiSparse has a second capacity limit:
hot GPU pages.

For MVP, keep admission conservative:

- Support `bs=1` first, or require `bs * index_topk` to fit the hot buffer.
- Reserve enough hot pages for one decode batch's selected pages.
- Clear or recycle the hot buffer between decode steps if necessary.

This is not optimal, but it avoids subtle lifetime bugs. Once correctness is
stable, improve to a real residency policy:

- Reuse hot pages across steps.
- Pin pages selected by the current batch.
- Evict unpinned pages with LRU.
- On request finish, release pages owned only by that request.

The scheduler-facing API should be explicit:

```text
can_admit_hisparse(batch_shape, index_topk) -> bool
reserve_hisparse_decode_budget(...)
release_hisparse_request(seq_id)
```

The C++ scheduler does not need to know implementation details at first, but it
must have enough information to avoid launching a batch that cannot fit in the
hot buffer.

## 8. CUDA Graph Considerations

The correctness MVP can run outside CUDA graph first. A graph-compatible version
needs:

- Fixed hot buffer shape.
- Fixed `indices` shape `[max_bs, ntps, index_topk]`.
- In-place updates to mapping tensors.
- No Python allocation in the captured region.
- No data-dependent tensor shape changes.

Recommended sequence:

1. Eager only, full debug assertions.
2. Eager with fixed preallocated buffers.
3. CUDA graph warmup with all-invalid indices.
4. CUDA graph replay with in-place updated indices and mapping.

## 9. Correctness Strategy

### 9.1 Unit Tests

Start with synthetic tensors:

- Fill cold KV with deterministic page/token values.
- Select top-k logical positions with duplicates and invalid `-1`.
- Swap pages into the hot buffer.
- Verify hot indices read exactly the source values.
- Verify page reuse and eviction do not corrupt still-pinned pages.

### 9.2 Model-Level Tests

Use a two-backend comparison:

```text
baseline: full GPU MLA KV + normal sparse indices
hisparse: cold KV + hot KV + translated sparse indices
```

For the same prompt and deterministic sampling:

- `ctx <= index_topk`: output should match very closely because all visible
  tokens are selected.
- `ctx > index_topk`: compare selected source positions, translated hot
  positions, logits, and next-token ids.

### 9.3 Long-Context Tests

Use DSV3.2/GLM-5.1-FP8 style models:

- Needle retrieval at 4k, 8k, 16k, 64k.
- Repeated decode for hundreds of steps to catch stale hot-page bugs.
- Batch sizes 1, 2, 4 after the single-request path is stable.

Gemma-style SWA models are useful for validating generic cache mapping and
window behavior, but they do not validate NSA top-k selection or sparse MLA
HiSparse correctness.

## 10. Performance Strategy

The naive implementation may be slower than full GPU KV because every layer may
copy up to `index_topk` tokens/pages from host to GPU. Performance work should
focus on reducing copies:

- Deduplicate selected pages before H2D.
- Keep hot pages resident across decode steps.
- Batch H2D copies by contiguous source/destination page ranges.
- Use a dedicated CUDA stream for H2D copies.
- Start swap-in as soon as top-k is available for a layer.
- Track hot hit rate, copied bytes, and copy latency per layer.

Important metrics:

```text
hisparse_hot_pages
hisparse_hot_hit_rate
hisparse_h2d_pages_per_step
hisparse_h2d_bytes_per_step
hisparse_swap_in_latency_ms
hisparse_evictions_per_step
```

## 11. Staged Plan

### Stage 0: Documentation And Flags

- Add config flags, default disabled.
- Add docs and debug metric names.
- Do not change runtime behavior.

### Stage 1: Tensor-Level Hot/Cold Prototype

- Implement `HiSparseMLACache` with cold/hot tensors.
- Implement page swap and logical-to-hot translation.
- Add synthetic unit tests.

### Stage 2: DSV3.2 Decode Eager Path

- Keep indexer cache GPU-resident.
- Replace `topk_indices_to_physical` with HiSparse translation when enabled.
- Call FlashMLA with `hot_k_cache`.
- Compare against full GPU sparse decode.

### Stage 3: Lifetime And Scheduler Integration

- Add conservative hot-buffer admission.
- Add request-finish cleanup.
- Add debug assertions for stale mappings.
- Add metrics.

### Stage 4: CUDA Graph Compatibility

- Preallocate buffers.
- Move all per-step outputs into fixed tensors.
- Capture warmup with dummy indices.
- Replay with in-place updates.

### Stage 5: DSv4 Extension

After DSV3.2 decode is stable:

- Add DSv4 C4/CSA hot/cold split.
- Keep SWA ring on GPU.
- Translate compressed selected pages into hot CSA/C4 pages.
- Reuse the same coordinator concepts with DSv4-specific page layout.

## 12. Main Risks

### Silent Wrong Output

The highest risk is an index translation bug that still produces valid tensor
shapes. FlashMLA will run, but it will attend to the wrong keys.

Mitigation:

- Debug mode verifies copied hot values against cold source values.
- Log selected logical indices, source slots, and hot slots for one layer.
- Run deterministic replay against the full GPU baseline.

### Bandwidth Collapse

If selected pages churn every decode step, H2D copies can dominate runtime.

Mitigation:

- Page-level deduplication.
- Hot residency across steps.
- LRU and per-request pinning.
- Metrics before clever policies.

### Lifetime Bugs

Hot pages can be reused while still referenced by an active FlashMLA call.

Mitigation:

- Start with synchronous swap and no overlap.
- Add stream/event ownership only after correctness tests pass.
- Keep refcounts conservative.

### Indexer Cache Memory

The MVP leaves `IndexerCache` on GPU. This costs memory, but it keeps top-k
scoring fast and avoids a much harder problem: sparse selection over an offloaded
indexer cache.

This is an explicit tradeoff. If indexer cache memory later becomes the blocker,
that should be treated as a separate design.

## 13. Open Questions

- Should cold KV be stored in CPU pinned torch tensors, a custom allocator, or
  reused through the existing L3/host storage abstraction?
- Should hot residency be per-layer or shared across layers?
- Is page size always 64 for the DSV3.2 MLA path we care about?
- How much hot buffer is needed for useful hit rate at realistic batch sizes?
- Should scheduler admission reserve pages pessimistically by `index_topk`, or
  optimistically by observed unique selected pages?
- How should HiSparse interact with prefix cache and session cache ownership?

## 14. Summary

The recommended first implementation is DSV3.2 decode-only HiSparse:

- Keep indexer cache on GPU.
- Store main MLA KV in a cold host tier.
- Maintain a small hot GPU KV cache.
- Translate indexer-selected logical positions into hot sparse indices.
- Feed the existing FlashMLA sparse decode path.

This is the smallest path that validates the core HiSparse idea in DLEngine
without entangling DSv4 SWA/CSA, prefill, PD disaggregation, or L3 storage.

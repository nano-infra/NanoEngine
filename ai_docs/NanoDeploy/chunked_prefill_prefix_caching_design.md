# Plan: Chunked Prefill + Prefix Caching in NanoInfra

## Context

Two related features:

**Prefix Caching**: The `BlockManager` already does hash-based block deduplication (shared KV blocks when token content matches). However, `num_cached_tokens` is never set during allocation, so prefill still recomputes KV for all tokens even when their blocks are shared from another sequence. The `prepare_prefill_from_bytes` C++ code already handles `num_cached_tokens` correctly — we just need to wire it from the allocation side.

**Chunked Prefill**: Sequences longer than `max_num_batched_tokens` cannot currently be scheduled (stuck in `waiting` forever). The `allocate()` signature accepts `token_idx_from/to` params but they are `(void)`-cast and ignored. We need sequential chunking: each engine step is still pure-prefill or pure-decode, but a long prompt is split into multiple consecutive prefill steps.

## Critical Files

| File                                                          | Role                                                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `NanoDeploy/nanodeploy/csrc/scheduler/block_manager.h`        | Add `count_active_prefix_hits()`                                                      |
| `NanoDeploy/nanodeploy/csrc/scheduler/block_manager.cpp`      | Core prefix caching wiring                                                            |
| `NanoDeploy/nanodeploy/csrc/scheduler/sp_state_manager.h/cpp` | Updated `can_allocate()`, `allocate()` counters                                       |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler.h`            | Add `prefilling` deque                                                                |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler.cpp`          | `_schedule_prefill()` chunking logic                                                  |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler_utils.cpp`    | Postprocess non-final chunk detection                                                 |
| `NanoSequence/proto/sequence.fbs`                             | Add `PREFILLING = 4` status                                                           |
| `NanoSequence/nanosequence/csrc/sequence/sequence.h`          | Already has `num_cached_tokens`, `num_prompt_tokens` — no change needed               |
| `NanoDeploy/nanodeploy/csrc/engine/serialization.cpp`         | Already uses `seq->num_tokens()` / `num_cached_tokens()` correctly — no change needed |
| `NanoDeploy/nanodeploy/csrc/worker/model_runner_utils.cpp`    | Already handles `num_cached_tokens` correctly — no change needed                      |
| `NanoDeploy/nanodeploy/backends/hopper/layers/attention.py`   | Fix the dead `block_tables` branch; add paged KV gather for chunked/cached prefill    |

## Part 1: Prefix Caching — Computation Skip

### 1a. `block_manager.h` — new method

Add:

```cpp
int count_active_prefix_hits(Sequence& seq) const;
```

This non-mutating scan counts consecutive leading blocks (from block 0) where both:

- The hash matches an entry in `hash_to_block_id_`
- The block is in `used_block_ids_` (actively shared by another sequence)
- Token content matches exactly

"Active" means ref_count > 0 already — these genuinely don't need a free block (just ref_count++).

### 1b. `block_manager.cpp` — wire `num_cached_tokens` in `allocate()`

In the existing `allocate()` loop, track consecutive leading cache hits before the first miss:

```cpp
int num_prefix_cached = 0;  // consecutive leading cache-hit FULL blocks
// ... existing loop ...
// Before first cache_miss:
if (!cache_miss && view.second == block_size_ && /* hash hit */) {
    num_prefix_cached++;
} else {
    cache_miss = true;
}
// After loop:
seq.set_num_cached_tokens(num_prefix_cached * block_size_);
```

Since `SPStateManager::allocate()` calls non-master ranks first then master last, the master rank's call to `set_num_cached_tokens` wins. For SP=1 this is the only call.

### 1c. `block_manager.cpp` — update `can_allocate()`

Subtract actively-shared prefix blocks from the required free block count:

```cpp
bool BlockManager::can_allocate(Sequence& seq) const {
    int n_cached = count_active_prefix_hits(seq);
    int blocks_needed = seq.num_blocks(BlockContextSlot::ACTIVE, sp_idx_) - n_cached;
    return static_cast<int>(free_block_ids_.size()) >= blocks_needed;
}
```

Note: The `needed_blocks` in `SPStateManager::can_allocate()` (line 128) stays pessimistic; correctness is guaranteed by the final `block_manager->can_allocate()` physical check.

### 1d. No Python changes needed

`prepare_prefill_from_bytes` already uses `si->num_cached_tokens()` to:

- Compute `seqlen_q = num_tokens - num_cached` (only new tokens forwarded)
- Compute `positions[t] = num_cached + t` (correct absolute positions)
- Build `slot_mapping` only for non-cached blocks
- Set `use_block_tables = true` when `seqlen_k > seqlen_q` (prefix KV read via block table)

## Part 2: Chunked Prefill — Sequential Chunks

### 2a. `NanoSequence/proto/sequence.fbs` — add PREFILLING status

```flatbuffers
enum SequenceStatus : byte {
  WAITING = 0,
  RUNNING = 1,
  FINISHED = 2,
  TO_BE_MIGRATED = 3,
  PREFILLING = 4,   // NEW: mid-prompt, between chunks
}
```

After editing, regenerate `sequence_generated.h` (run flatc or the project's codegen script).

### 2b. `scheduler.h` — add `prefilling` deque

```cpp
std::deque<std::shared_ptr<Sequence>> prefilling;  // mid-prefill sequences (dp-agnostic)
```

### 2c. `scheduler.cpp` — `_schedule_prefill()` changes

**Process `prefilling` queue FIRST** (higher priority, they hold allocated blocks):

```cpp
// --- Step 1: Schedule PREFILLING (in-progress) sequences ---
std::deque<std::shared_ptr<Sequence>> not_scheduled_prefilling;
while (!prefilling.empty()) {
    auto seq = prefilling.front(); prefilling.pop_front();
    int dp_idx = seq->block_ctx(BlockContextSlot::ACTIVE).dp_idx;
    int master_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;

    int prev_tokens = seq->num_tokens();
    int budget_remaining = max_num_batched_tokens_ - num_batched_tokens[dp_idx][master_sp];
    int new_tokens = std::min(budget_remaining,
                              seq->num_prompt_tokens() - prev_tokens);
    if (new_tokens <= 0) { not_scheduled_prefilling.push_back(seq); continue; }

    // *** DEADLOCK PREVENTION + SHRINK BEFORE PREEMPT ***
    // Only preempt if truly saturated (zero free blocks). Otherwise shrink the
    // chunk to what the cache can actually absorb — avoids thrashing when decode
    // sequences hold most blocks but some space remains.
    {
        int free_blocks = worker_state[dp_idx]->block_manager[master_sp]->num_free_blocks();
        int max_appendable_tokens = free_blocks * kvcache_block_size_;
        if (max_appendable_tokens <= 0) {
            // Truly saturated — preempt to avoid starvation deadlock
            preempt(dp_idx, seq);
            continue;
        }
        new_tokens = std::min(new_tokens, max_appendable_tokens);
    }

    seq->set_num_tokens(prev_tokens + new_tokens);
    worker_state[dp_idx]->block_manager[master_sp]->may_append(*seq, new_tokens);
    // Update num_dispatched_tokens so future may_append() computes correctly
    seq->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens[master_sp] = seq->num_tokens();

    num_seqs[dp_idx][master_sp] += 1;
    num_batched_tokens[dp_idx][master_sp] += new_tokens;
    scheduled_seqs[dp_idx].push_back(seq);
}
// Put back unscheduled (budget exhausted) at front of prefilling
for (auto it = not_scheduled_prefilling.rbegin(); it != not_scheduled_prefilling.rend(); ++it)
    prefilling.push_front(*it);
```

**Modify WAITING queue processing** to chunk long sequences:

```cpp
// --- Step 2: Schedule fresh WAITING sequences with chunking ---
while (!waiting_queue.empty()) {
    auto seq = waiting_queue.front();

    // Compute chunk size for this sequence
    int budget_for_this_seq = ...; // max_num_batched_tokens_ - total_already_scheduled
    int chunk_end = seq->num_cached_tokens()
                  + std::min(budget_for_this_seq,
                             seq->num_prompt_tokens() - seq->num_cached_tokens());
    // Temporarily set num_tokens to chunk_end for can_allocate/allocate
    seq->set_num_tokens(chunk_end);

    bool can_alloc = worker_state[dp_idx]->can_allocate(*seq, num_seqs[dp_idx], num_batched_tokens[dp_idx]);
    if (!can_alloc) {
        seq->set_num_tokens(seq->num_prompt_tokens()); // restore
        break;
    }
    worker_state[dp_idx]->allocate(*seq);  // sets num_cached_tokens via prefix hits

    num_seqs[dp_idx][master_sp] += 1;
    num_batched_tokens[dp_idx][master_sp] += (seq->num_tokens() - seq->num_cached_tokens());
    seq->set_status(SequenceStatus::RUNNING);
    waiting_queue.pop_front();
    worker_state[dp_idx]->running.push_back(seq);
    scheduled_seqs[dp_idx].push_back(seq);
    // ...
}
```

### 2d. `scheduler_utils.cpp` — postprocess non-final chunk detection

In `worker_func`, before `seq->append_token(token_id, ...)`:

```cpp
// Detect non-final prefill chunk
bool is_nonfinal_chunk = is_prefill
    && (seq->num_tokens() < seq->num_prompt_tokens());

if (is_nonfinal_chunk) {
    // KV is computed for this chunk; mark all chunk tokens as cached
    seq->set_num_cached_tokens(seq->num_tokens());
    seq->set_status(SequenceStatus::PREFILLING);
    result_ctx->chunk_continuations.push_back(seq);
    break;  // don't process further token_ids for this seq
}
// ... existing append_token + migration/finish logic unchanged ...
```

Also extend the `running` queue cleanup to remove PREFILLING sequences:

```cpp
running.erase(std::remove_if(running.begin(), running.end(),
    [](const std::shared_ptr<Sequence>& s) {
        return s->status() == SequenceStatus::FINISHED
            || s->status() == SequenceStatus::TO_BE_MIGRATED
            || s->status() == SequenceStatus::PREFILLING; // NEW
    }), running.end());
```

Add `chunk_continuations` to `WorkerContext`:

```cpp
struct WorkerContext {
    std::vector<Task> tasks;
    MigrationList migration_candidates;
    std::vector<std::shared_ptr<Sequence>> chunk_continuations;  // NEW
    std::exception_ptr eptr = nullptr;
    int dp_idx;
};
```

Update `postprocess_sequences()` return type or signature to also return `chunk_continuations`. Simplest: return a struct `PostprocessResult { MigrationList migrations; std::vector<shared_ptr<Sequence>> continuations; }`.

### 2e. `scheduler.cpp` — `Scheduler::postprocess()` — move continuations to prefilling

```cpp
void Scheduler::postprocess(...) {
    auto result = postprocess_sequences(...);
    for (auto& [seq, dp_idx] : result.migrations)
        to_be_migrated[seq->seq_id()] = {seq, dp_idx};
    for (auto& seq : result.continuations)
        prefilling.push_back(seq);  // will be picked up in next _schedule_prefill()
}
```

### 2f. `preempt()` adjustment

When preempting a PREFILLING sequence (mid-chunk), restore `num_tokens = num_prompt_tokens` so it starts over cleanly after re-queuing:

```cpp
void Scheduler::preempt(int dp_idx, std::shared_ptr<Sequence> seq) {
    seq->set_status(SequenceStatus::WAITING);
    worker_state[dp_idx]->deallocate(*seq);
    seq->set_num_tokens(seq->num_prompt_tokens());  // Reset to full length
    seq->set_num_cached_tokens(0);  // Reset (deallocate already calls this)
    seq->set_num_checkpointed_tokens(static_cast<int>(seq->token_ids().size()));
    waiting.push_front(seq);
}
```

## Part 3: Skip `lm_head` + Sampler for Non-Final Chunks

### The problem

The GPU still computes `lm_head(hidden_states)` and runs the sampler for **every** chunk, including non-final ones. For non-final chunks the sampled token is discarded in `postprocess`. This wastes significant GPU compute (lm_head over the full vocabulary is large).

### Fix: per-sequence `sampling_token_indices`

A global `is_final_chunk = false` would skip the sampler for **all** sequences in the batch. But a single batch can contain both non-final chunks (need no sample) and final-chunk sequences (need a sample). We must only run `lm_head` on the hidden states for sequences that complete their prefill.

**`model_runner_utils.h`** — add to `PrefillMetadata`:

```cpp
struct PrefillMetadata {
    // ... existing fields ...
    // Indices into the Q (hidden_states) tensor for sequences at their final chunk.
    // Empty means all sequences are non-final — skip lm_head entirely.
    std::vector<int> sampling_token_indices;
};
```

Each entry is the index of the **last** Q token for a sequence with `num_tokens == num_prompt_tokens`. These are the positions the sampler must read from.

**`model_runner_utils.cpp` — `prepare_prefill_from_bytes`**: populate after building `cu_seqlens_q`:

```cpp
// After the per-sequence loop that builds cu_seqlens_q:
meta.sampling_token_indices.clear();
for (size_t i = 0; i < si_vec->size(); ++i) {
    auto* si = si_vec->Get(i);
    if (si->num_tokens() == si->num_prompt_tokens()) {
        // last Q token index for this sequence = cu_seqlens_q[i+1] - 1
        meta.sampling_token_indices.push_back(meta.cu_seqlens_q[i + 1] - 1);
    }
}
```

This requires `num_prompt_tokens` in `SequenceInput`. Add it to `interface.fbs`:

```flatbuffers
table SequenceInput {
  // ... existing ...
  num_prompt_tokens: int;   // NEW: original prompt length, for chunk detection
}
```

And serialize it in `serialization.cpp`: `si_builder.add_num_prompt_tokens(seq->num_prompt_tokens())`.

**`model_runner.py`**: use `sampling_token_indices` to only run `lm_head` on the finishing tokens:

```python
meta = prepare_prefill_bytes(...)
if meta.sampling_token_indices:
    # Extract only the hidden states that need sampling
    sample_hs = hidden_states[meta.sampling_token_indices]  # [n_final, hidden_dim]
    logits = self.lm_head(sample_hs)
    sampled = self.sampler(logits)
    # Expand back to full batch size: non-final positions get a placeholder token
    # (e.g. -1 or EOS); worker_func discards them for PREFILLING sequences anyway
    output_tokens = [-1] * num_seqs
    for idx, seq_idx in enumerate(meta.sampling_seq_indices):
        output_tokens[seq_idx] = sampled[idx]
else:
    output_tokens = []   # entirely non-final batch; postprocess detects PREFILLING
```

**Note**: `meta.sampling_seq_indices` (parallel list of which sequence number each index corresponds to) can be built in C++ alongside `sampling_token_indices` so Python can reconstruct the sparse output without another Python loop.

The C++ `worker_func` detects `is_nonfinal_chunk` via `num_tokens < num_prompt_tokens` and correctly skips `append_token` for those sequences regardless of what token value is in the output list.

## Part 4: Attention — Paged KV Gather for Cached Prefill

### Flash-Attn API Audit

Neither public ragged-prefill function supports paged (block-table) KV in the base H100 environment:

| Function                  | Ragged Q (`cu_seqlens_q`)   | Paged KV (`page_table`) | Usable here?                                                                    |
| ------------------------- | --------------------------- | ----------------------- | ------------------------------------------------------------------------------- |
| `flash_attn_varlen_func`  | ✓                           | ✗                       | Q only                                                                          |
| `flash_attn_with_kvcache` | `cu_seqlens_q` param exists | ✓                       | Shape mismatch — expects `q: [batch, seqlen_q, ...]` (padded), not truly ragged |

`flash_attn_with_kvcache` with `cu_seqlens_q` interprets `q` as `[batch_size, seqlen_q, nheads, headdim]`. If we pass `q` with shape `[total_q_tokens, nheads, headdim]` (the ragged layout), the kernel treats `total_q_tokens` as `batch_size` and `nheads` as `seqlen_q` — wrong.

### Solution: Gather + `flash_attn_varlen_func`

`store_kvcache` writes new tokens' K/V to their paged slots **before** the attention call, so `k_cache`/`v_cache` already contains all relevant tokens. Gather them into contiguous tensors, then use the ragged kernel:

```python
def _gather_kv_paged(
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
    cu_seqlens_k: torch.Tensor,   # [num_seqs + 1], cumulative K lengths
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged K/V into ragged contiguous tensors."""
    total_k = int(cu_seqlens_k[-1].item())
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    k_out = k_cache.new_empty(total_k, num_kv_heads, head_dim)
    v_out = v_cache.new_empty(total_k, num_kv_heads, head_dim)
    num_seqs = block_table.shape[0]
    for i in range(num_seqs):
        start = int(cu_seqlens_k[i].item())
        end   = int(cu_seqlens_k[i + 1].item())
        seq_len = end - start
        for tok in range(seq_len):
            blk = int(block_table[i, tok // block_size].item())
            off = tok % block_size
            k_out[start + tok] = k_cache[blk, off]
            v_out[start + tok] = v_cache[blk, off]
    return k_out, v_out
```

In practice, replace the Python loop with a fused CUDA kernel or `torch.index_select` + reshape for performance. The Python loop above is the reference implementation; optimize after correctness is confirmed.

### Fix: replace dead branch in `attention.py`

```python
if context.is_prefill:
    if context.block_tables is not None:   # prefix cache hit or chunk 2+
        # Gather paged K/V into contiguous ragged tensors
        bt = context.block_tables[sp_rank]        # [num_seqs, max_blocks]
        k_gathered, v_gathered = _gather_kv_paged(
            k_cache, v_cache, bt,
            context.cu_seqlens_k, self.block_size
        )
        o = flash_attn_varlen_func(
            q, k_gathered, v_gathered,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )
    else:                                          # first chunk, no cached prefix
        o = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )
```

**Notes**:

- `cu_seqlens_k[i]` = total context length for sequence i (cached tokens + new tokens), already computed in `prepare_prefill_from_bytes` from `num_tokens` (= chunk endpoint).
- `block_tables[sp_rank]`: shape `[num_seqs, max_blocks_per_seq]` — the dense table row for this SP rank.
- `store_kvcache` already runs first and populates slots `[num_cached_blocks .. num_blocks)`, so `k_cache`/`v_cache` holds the complete context for each sequence.
- No changes to `store_kvcache` or the slot-mapping logic needed.

### KV storage correctness (no change needed)

`slot_mapping` in `prepare_prefill_from_bytes` iterates `b = num_cached_blocks .. num_blocks` (only new blocks):

```cpp
int num_cached_blocks = num_cached / block_size;
for (int b = num_cached_blocks; b < num_blocks; ++b) { ... }
```

Since `num_cached_tokens` is always a multiple of `block_size` (only full-block hits counted), this correctly maps each new token to its physical cache slot. `store_kvcache` writes only those new tokens — no offset bug.

## Implementation Order

01. `sequence.fbs` + `interface.fbs` — add `PREFILLING = 4`; add `num_prompt_tokens` to `SequenceInput`; regenerate headers
02. `block_manager.h/cpp` — prefix caching: `count_active_prefix_hits`, `allocate()` update, `can_allocate()` update
03. `model_runner_utils.h/cpp` — add `sampling_token_indices` + `sampling_seq_indices` to `PrefillMetadata`; populate from sequences where `num_tokens == num_prompt_tokens`
04. `serialization.cpp` — serialize `num_prompt_tokens` from sequence
05. `scheduler_utils.h/cpp` — add `chunk_continuations` to `WorkerContext`; detect non-final chunks via `num_tokens < num_prompt_tokens`; update `postprocess_sequences` return
06. `scheduler.h` — add `prefilling` deque
07. `scheduler.cpp` — update `_schedule_prefill()` (PREFILLING queue first, shrink-then-preempt); update `postprocess()`; update `preempt()`
08. `model_runner.py` — selective `lm_head` only on `sampling_token_indices`
09. `attention.py` — replace dead `block_tables` branch with gather + `flash_attn_varlen_func`
10. Build and test

## Verification

**Prefix caching test**:

1. Send two requests with the same long prefix (e.g., system prompt)
2. First request: all tokens prefilled normally
3. Second request: `num_cached_tokens > 0` in logs, `seqlen_q < num_prompt_tokens`, TTFP is faster
4. Check that KV outputs match a non-cached run (correctness)

**Attention test**:

1. Run a 2-chunk sequence; confirm the second chunk takes the gather path (log/assert that `block_tables is not None`)
2. Confirm numerical output of the 2-chunk run matches a single-chunk reference (with `max_num_batched_tokens` large enough to fit the whole prompt)

**Chunked prefill test**:

1. Set `max_num_batched_tokens = 512`, send a 2000-token prompt
2. Verify the request is scheduled (not stuck in waiting)
3. Verify it processes in ~4 chunks, each triggering a prefill step
4. Verify the final output matches a run with `max_num_batched_tokens = 4096`
5. Mix with concurrent decode requests — verify decode still runs between chunk prefill steps

**Integration**: Run the existing test suite to confirm no regressions.

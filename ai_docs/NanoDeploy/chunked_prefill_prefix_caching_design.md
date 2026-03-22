# Chunked Prefill + Prefix Caching in NanoInfra

## Context

Two related features:

**Prefix Caching**: The `BlockManager` already does hash-based block deduplication (shared KV blocks when token content matches). However, `num_cached_tokens` is never set during allocation, so prefill still recomputes KV for all tokens even when their blocks are shared from another sequence. The `prepare_prefill_from_bytes` C++ code already handles `num_cached_tokens` correctly — we just need to wire it from the allocation side.

**Chunked Prefill**: Sequences longer than `max_num_batched_tokens` are split into consecutive prefill steps. Each chunk processes up to `max_num_batched_tokens` tokens, with intermediate KV results cached for subsequent chunks.

## End-to-End Walkthrough

This section traces a 1209-token prompt through the system with `kvcache_block_size=64` and `max_num_batched_tokens=128`.

### Phase 1: Scheduling the First Chunk

1. **`Scheduler::_schedule_prefill()`** processes the WAITING queue.

   - `seq->num_cached_tokens()` = 0, budget = 128.
   - `chunk_end = 0 + min(128, 1209 - 0) = 128`.
   - `seq->set_num_tokens(128)` — temporarily set to chunk endpoint.
   - `can_allocate()` / `allocate()` succeeds → allocates 2 blocks (128 / 64).
   - `seq->set_status(RUNNING)` → pushed to `running` queue and `scheduled_seqs`.

2. **Serialization** (`serialization.cpp`): serializes `num_tokens=128`, `num_cached_tokens=0`, `num_prompt_tokens=1209`, token_ids\[0..127\], and block_tables.

3. **`prepare_prefill_from_bytes()`** (`model_runner_utils.cpp`):

   - `seqlen_q = 128 - 0 = 128`, `seqlen_k = 128`.
   - `cu_seqlens_q = [0, 128]`, `cu_seqlens_k = [0, 128]`.
   - `seqlen_k == seqlen_q` → `use_block_tables = false`, no gather needed.
   - `sampling_token_indices`: `num_tokens(128) < num_prompt_tokens(1209)` → **not final**, no entry added.

4. **GPU execution**: model runs prefill on 128 tokens with standard `flash_attn_varlen_func`.

   - `store_kvcache` writes K/V to cache slots 0..127.
   - `ParallelLMHead`: `sampling_token_indices` is set (empty → all non-final), so `lm_head` is skipped entirely.

5. **`postprocess_sequences()`** (`scheduler_utils.cpp`):

   - `seq->num_tokens()(128) < seq->num_prompt_tokens()(1209)` → **non-final chunk detected**.
   - `seq->set_num_cached_tokens(128)`.
   - `seq->set_status(PREFILLING)`.
   - Pushes to `result.continuations`.
   - Running queue cleanup removes `PREFILLING` sequences.

6. **`Scheduler::postprocess()`**: moves continuations to `prefilling` deque.

### Phase 2: Scheduling Subsequent Chunks (2..9)

1. **`Scheduler::_schedule_prefill()`** now processes `prefilling` queue FIRST (higher priority):

   - `prev_tokens = seq->num_tokens()` (e.g. 128 after chunk 1).
   - `budget_remaining = 128`, `new_tokens = min(128, 1209 - 128) = 128`.
   - Shrink-before-preempt: checks `free_blocks > 0` → shrinks if needed.
   - `seq->set_num_tokens(256)`, `may_append()` allocates 2 more blocks.
   - Pushed to `running` and `scheduled_seqs`.

2. **`prepare_prefill_from_bytes()`**:

   - `seqlen_q = 256 - 128 = 128` (only new tokens).
   - `seqlen_k = 256` (total context including cached).
   - `seqlen_k > seqlen_q` → `use_block_tables = true`, dense block_tables built.
   - `slot_mapping`: iterates blocks `[num_cached_blocks..num_blocks)`, maps only new tokens.
   - `sampling_token_indices`: still non-final (256 \< 1209) → empty.

3. **GPU execution**:

   - `store_kvcache` writes fresh K/V to slots 128..255.
   - **GQA** (`FlashAttentionImpl`): `context.block_tables is not None` → `_gather_kv_paged(k_cache, v_cache, bt, cu_seqlens_k, block_size)` gathers ALL 256 K/V tokens from paged cache into contiguous tensors → `flash_attn_varlen_func(q[128 tokens], k[256 tokens], v[256 tokens], ...)`.
   - **MLA** (`DeepseekV2Attention`): `context.block_tables is not None` → `_gather_cache_paged(k_cache, bt, cu_seqlens_k, block_size)` gathers compressed keys (576d) → split into `compressed_kv` (512d) + `k_pe` (64d) → expand K/V via `kc`/`vc` weight matrices → `flash_attn_varlen_func`.
   - **GDN** (`GenericGatedDeltaNet`): `context.block_tables is not None` → load conv state from `gdn_conv_states[layer, slot, :, 1:]` and recurrent state from `gdn_recurrent_states[layer, slot]` → `chunk_gated_delta_rule` with loaded initial state.

4. **Postprocess**: same as Phase 1 — sets `PREFILLING`, pushes to continuations.

Repeat until chunk 10 (the final chunk).

### Phase 3: Final Chunk (chunk 10)

1. **Scheduling**: `new_tokens = min(128, 1209 - 1152) = 57`. `seq->set_num_tokens(1209)`.

2. **`prepare_prefill_from_bytes()`**:

   - `seqlen_q = 57`, `seqlen_k = 1209`.
   - `sampling_token_indices`: `num_tokens(1209) == num_prompt_tokens(1209)` → **final chunk**, `q_end - 1 = 56` added.

3. **GPU execution**: attention gathers all 1209 K/V tokens. `ParallelLMHead` extracts hidden state at index 56, computes logits, samples first output token.

4. **`postprocess_sequences()`**:

   - `seq->num_tokens()(1209) >= seq->num_prompt_tokens()(1209)` → **final chunk**.
   - `seq->status() == PREFILLING` → `seq->set_status(RUNNING)` (transition to decode).
   - Normal `append_token()` runs, sequence enters decode phase.

### Phase 4: Decode

Standard decode loop. Sequence has status `RUNNING`, stays in `running` queue. Each step generates one token until `max_tokens` or EOS.

## Critical Files

| File                                                                   | Role                                                                                  |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `NanoDeploy/nanodeploy/csrc/scheduler/block_manager.h`                 | Add `count_active_prefix_hits()`                                                      |
| `NanoDeploy/nanodeploy/csrc/scheduler/block_manager.cpp`               | Core prefix caching wiring                                                            |
| `NanoDeploy/nanodeploy/csrc/scheduler/sp_state_manager.h/cpp`          | Updated `can_allocate()`, `allocate()` counters                                       |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler.h`                     | `prefilling` deque                                                                    |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler.cpp`                   | `_schedule_prefill()` chunking logic; `postprocess()` routing; `preempt()` reset      |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler_utils.h`               | `PostprocessResult` struct with `continuations`                                       |
| `NanoDeploy/nanodeploy/csrc/scheduler/scheduler_utils.cpp`             | Non-final chunk detection; PREFILLING→RUNNING transition                              |
| `NanoSequence/proto/sequence.fbs`                                      | `PREFILLING = 4` status                                                               |
| `NanoDeploy/nanodeploy/csrc/engine/serialization.cpp`                  | Serializes `num_prompt_tokens` for chunk detection in workers                         |
| `NanoDeploy/nanodeploy/csrc/worker/model_runner_utils.cpp`             | `sampling_token_indices/seq_indices`; `slot_mapping` with `cached_offset_in_block`    |
| `NanoDeploy/nanodeploy/backends/hopper/layers/attention.py`            | `_build_paged_gather_indices`, `_gather_kv_paged`, `_gather_cache_paged`; GQA prefill |
| `NanoDeploy/nanodeploy/models/deepseek_v2/deepseek_v2.py`              | MLA chunked prefill: gather compressed K + expand K/V                                 |
| `NanoDeploy/nanodeploy/backends/gpu_generic/layers/gated_delta_net.py` | GDN chunked prefill: conv/recurrent state continuity                                  |
| `NanoDeploy/nanodeploy/layers/embed_head.py`                           | Sparse `lm_head`: only final-chunk hidden states                                      |
| `NanoDeploy/nanodeploy/worker/model_runner.py`                         | Wire `sampling_token_indices` from C++ metadata to context                            |

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

### 2a. `NanoSequence/proto/sequence.fbs` — PREFILLING status

```flatbuffers
enum SequenceStatus : byte {
  WAITING = 0,
  RUNNING = 1,
  FINISHED = 2,
  TO_BE_MIGRATED = 3,
  PREFILLING = 4,   // mid-prompt, between chunks
}
```

### 2b. `scheduler.h` — prefilling deque

```cpp
std::deque<std::shared_ptr<Sequence>> prefilling;  // mid-prefill sequences
```

### 2c. `scheduler.cpp` — `_schedule_prefill()` two-step scheduling

**Step 1: Schedule PREFILLING sequences first** (higher priority — they already hold allocated blocks):

```cpp
std::deque<std::shared_ptr<Sequence>> not_scheduled_prefilling;
while (!prefilling.empty()) {
    auto seq = prefilling.front(); prefilling.pop_front();
    int dp_idx    = block_ctx.dp_idx;
    int master_sp = block_ctx.master_sp_idx;

    int prev_tokens     = seq->num_tokens();
    int budget_remaining = max_num_batched_tokens_ - num_batched_tokens[dp_idx][master_sp];
    int new_tokens      = std::min(budget_remaining,
                                   seq->num_prompt_tokens() - prev_tokens);
    if (new_tokens <= 0) { not_scheduled_prefilling.push_back(seq); continue; }

    // Shrink-before-preempt: only preempt if truly saturated.
    int free_blocks = worker_state[dp_idx]->block_manager[master_sp]->num_free_blocks();
    int max_appendable = free_blocks * kvcache_block_size_;
    if (max_appendable <= 0) { preempt(dp_idx, seq); continue; }
    new_tokens = std::min(new_tokens, max_appendable);

    seq->set_num_tokens(prev_tokens + new_tokens);
    worker_state[dp_idx]->block_manager[master_sp]->may_append(*seq, new_tokens);
    block_ctx.num_dispatched_tokens[master_sp] = seq->num_tokens();

    num_seqs[dp_idx][master_sp] += 1;
    num_batched_tokens[dp_idx][master_sp] += new_tokens;
    worker_state[dp_idx]->running.push_back(seq);
    scheduled_seqs[dp_idx].push_back(seq);
}
// Put back unscheduled at front (preserve order)
for (auto it = not_scheduled_prefilling.rbegin(); ...) prefilling.push_front(*it);
```

**Step 2: Schedule fresh WAITING sequences** with chunking:

```cpp
while (!waiting_queue.empty()) {
    auto seq = waiting_queue.front();
    int budget = ...;  // min across all SP ranks
    int num_cached = seq->num_cached_tokens();
    int chunk_end = num_cached + std::min(budget, seq->num_prompt_tokens() - num_cached);
    seq->set_num_tokens(chunk_end);  // temporarily set

    bool can_alloc = worker_state[dp_idx]->can_allocate(...);
    if (!can_alloc) { seq->set_num_tokens(seq->num_prompt_tokens()); break; }

    worker_state[dp_idx]->allocate(*seq);  // sets num_cached_tokens via prefix hits
    seq->set_status(SequenceStatus::RUNNING);
    waiting_queue.pop_front();
    worker_state[dp_idx]->running.push_back(seq);
    scheduled_seqs[dp_idx].push_back(seq);
}
```

### 2d. `scheduler_utils.cpp` — non-final chunk detection and status transitions

In `worker_func`, before `append_token`:

```cpp
// Non-final chunk detection — must NOT be guarded by is_prefill.
// In combined (non-disaggregated) mode, mode_ is never "prefill",
// but chunked sequences still need to continue.
if (seq->num_tokens() < seq->num_prompt_tokens()) {
    seq->set_num_cached_tokens(seq->num_tokens());
    seq->set_status(SequenceStatus::PREFILLING);
    result_ctx->chunk_continuations.push_back(seq);
    continue;
}

// Final prefill chunk: transition PREFILLING → RUNNING for decode.
if (seq->status() == SequenceStatus::PREFILLING) {
    seq->set_status(SequenceStatus::RUNNING);
}

// ... existing append_token + finish logic ...
```

Running queue cleanup removes `PREFILLING` sequences:

```cpp
running.erase(std::remove_if(running.begin(), running.end(),
    [](const auto& s) {
        return s->status() == SequenceStatus::FINISHED
            || s->status() == SequenceStatus::TO_BE_MIGRATED
            || s->status() == SequenceStatus::PREFILLING;
    }), running.end());
```

### 2e. `scheduler_utils.h` — PostprocessResult

```cpp
struct PostprocessResult {
    MigrationList                          migrations;
    std::vector<std::shared_ptr<Sequence>> continuations;  // non-final prefill chunks
};
```

### 2f. `scheduler.cpp` — `postprocess()` routes continuations

```cpp
void Scheduler::postprocess(...) {
    auto result = postprocess_sequences(worker_state, dp_sp_seqs, dp_sp_token_ids,
                                        eos_, mode_ == "prefill", update_metrics, thread_pool_.get());
    for (const auto& [seq, dp_idx] : result.migrations)
        to_be_migrated[seq->seq_id()] = {seq, dp_idx};
    for (auto& seq : result.continuations)
        prefilling.push_back(seq);
}
```

### 2g. `preempt()` — reset mid-chunk sequences

```cpp
void Scheduler::preempt(int dp_idx, std::shared_ptr<Sequence> seq) {
    seq->set_status(SequenceStatus::WAITING);
    worker_state[dp_idx]->deallocate(*seq);
    seq->set_num_tokens(seq->num_prompt_tokens());  // reset to full length
    seq->set_num_checkpointed_tokens(static_cast<int>(seq->token_ids().size()));
    waiting.push_front(seq);
}
```

## Part 3: Selective lm_head for Non-Final Chunks

### Problem

Computing `lm_head(hidden_states)` and sampling for non-final chunks wastes GPU compute — the sampled token is discarded in `postprocess`. lm_head over the full vocabulary is expensive.

### Implementation

**`model_runner_utils.cpp`** — populates `sampling_token_indices` and `sampling_seq_indices`:

```cpp
int seq_pos = 0, q_start = 0;
for (size_t i = 0; i < si_vec->size(); ++i) {
    auto* si = si_vec->Get(i);
    if (si->master_sp_idx() != sp_rank) continue;

    int seqlen_q = si->num_tokens() - si->num_cached_tokens();
    int q_end = q_start + seqlen_q;

    int num_prompt = si->num_prompt_tokens();
    bool is_final = (num_prompt == 0) || (si->num_tokens() >= num_prompt);
    if (is_final) {
        meta.sampling_token_indices.push_back(q_end - 1);
        meta.sampling_seq_indices.push_back(seq_pos);
    }
    q_start = q_end;
    seq_pos++;
}
```

**`model_runner.py`** — wires indices to context only when some sequences are non-final:

```python
if len(meta.sampling_token_indices) < num_sp_seqs:
    sampling_token_indices = torch.tensor(meta.sampling_token_indices, ...).cuda()
    sampling_seq_indices = torch.tensor(meta.sampling_seq_indices, ...).cuda()
```

**`embed_head.py`** (`ParallelLMHead.forward`) — sparse extraction:

```python
if context.is_prefill:
    if context.sampling_token_indices is not None:
        x = x[context.sampling_token_indices].contiguous()
    else:
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()
logits = F.linear(x, self.weight)
```

## Part 4: Attention — Paged KV Gather for Chunked/Cached Prefill

### Flash-Attn API Constraint

`flash_attn_varlen_func` supports ragged Q (`cu_seqlens_q`) but NOT paged KV (`page_table`). For chunks 2+ where `seqlen_k > seqlen_q`, we need K/V for the full context but only have them in paged cache.

### Solution: Gather + `flash_attn_varlen_func`

`store_kvcache` writes new tokens' K/V to their paged slots BEFORE the attention call, so `k_cache`/`v_cache` already contains all relevant tokens. Gather them into contiguous tensors, then use the ragged kernel.

### 4a. Gather Utilities (`attention.py`)

Three functions, factored for reuse across attention types:

**`_build_paged_gather_indices`** — builds flat linear indices from block tables:

```python
def _build_paged_gather_indices(block_table, cu_seqlens_k, block_size) -> torch.Tensor:
    """Returns linear_indices: [total_k] — index into cache.reshape(-1, ...)."""
    for i in range(num_seqs):
        t = torch.arange(seqlen, device=device)
        block_ids = block_table[i, t // block_size]
        linear_indices[start:end] = block_ids * block_size + t % block_size
    return linear_indices
```

**`_gather_kv_paged`** — gathers K and V caches (for GQA with separate K/V):

```python
def _gather_kv_paged(k_cache, v_cache, block_table, cu_seqlens_k, block_size):
    indices = _build_paged_gather_indices(block_table, cu_seqlens_k, block_size)
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)
    return k_flat[indices], v_flat[indices]
```

**`_gather_cache_paged`** — gathers a single cache tensor (for MLA with only k_cache):

```python
def _gather_cache_paged(cache, block_table, cu_seqlens_k, block_size):
    indices = _build_paged_gather_indices(block_table, cu_seqlens_k, block_size)
    flat = cache.reshape(-1, *cache.shape[2:])
    return flat[indices]
```

### 4b. GQA Attention (`FlashAttentionImpl.forward`)

Used by Qwen3, Qwen3-MoE, Qwen3.5-MoE (full-attention layers).

```python
if context.is_prefill:
    if context.block_tables is not None:
        num_seqs = context.cu_seqlens_k.shape[0] - 1
        bt = context.block_tables[sp_rank, :num_seqs, :]
        k, v = _gather_kv_paged(k_cache, v_cache, bt, context.cu_seqlens_k, block_size)
    o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., cu_seqlens_k=..., causal=True)
```

### 4c. MLA Attention (`DeepseekV2Attention.forward`)

MLA stores compressed key states (576d = 512 latent + 64 RoPE'd k_pe) in `k_cache`. There is no separate `v_cache` — V is reconstructed from the latent.

For chunks 2+:

```python
if context.block_tables is not None:
    # Gather compressed keys from paged k_cache: [total_k, 1, 576] -> [total_k, 576]
    k_gathered = _gather_cache_paged(k_cache, bt, context.cu_seqlens_k, block_size).squeeze(1)

    # Split into latent and RoPE components
    compressed_kv_all = k_gathered[:, :kv_lora_rank]      # [total_k, 512]
    k_pe_all          = k_gathered[:, kv_lora_rank:]       # [total_k, 64]

    # Expand K: latent → k_nope via W_UK, concat with k_pe
    k_nope    = (compressed_kv_all @ kc_t).view(-1, H, 128)
    k_expanded = torch.cat([k_nope, k_pe_all.expand(-1, H, -1)], dim=-1)  # [total_k, H, 192]

    # Expand V: latent → v via W_UV
    v_expanded = (compressed_kv_all @ vc_reshaped).view(-1, H, 128)  # [total_k, H, 128]
```

Then `flash_attn_varlen_func(q_full, k_expanded, v_expanded, ...)`.

For the first chunk (no block_tables): K/V expanded directly from fresh `compressed_kv` projections (no gather needed).

### 4d. GDN Linear Attention (`GenericGatedDeltaNet`)

Used by Qwen3.5-MoE linear-attention layers. GDN does not use paged KV cache; it maintains recurrent state and conv1d state. For chunked prefill, state continuity across chunks is critical.

**Conv1d state** — `_conv1d_prefill_fast`:

```python
has_prev_state = context.block_tables is not None and gdn_conv_states is not None

if has_prev_state:
    # Chunks 2+: per-sequence processing with initial_states from previous chunk
    for i in range(num_seqs):
        init_state = gdn_conv_states[layer_idx, slot:slot+1, :, 1:].contiguous()
        seq_out = causal_conv1d_fn(
            x=qkv[start:end].T.unsqueeze(0),
            weight=conv_weight,
            initial_states=init_state,  # [1, conv_dim, kernel_size-1]
            activation="silu",
        )
else:
    # First chunk: batched with seq_idx (efficient single kernel)
    qkv_out = causal_conv1d_fn(x=..., weight=..., seq_idx=..., activation="silu")

# Store conv states for next chunk/decode (same for both paths)
states = causal_conv1d_varlen_states(qkv, cu_seqlens, kernel_size - 1)
gdn_conv_states[layer_idx, slots, :, 1:] = states
```

Naive fallback (`_conv1d_prefill_naive`): for chunks 2+, manually prepends the stored conv state and runs `F.conv1d` without built-in padding.

**Recurrent state** — `_gdn_prefill`:

```python
if gdn_recurrent_states is not None:
    if context.block_tables is not None:
        # Chunks 2+: load state from previous chunk
        initial_state = gdn_recurrent_states[layer_idx, slots[:num_seqs]]
    else:
        # First chunk: zero initial state
        initial_state = zeros(num_seqs, num_v_heads, head_k_dim, head_v_dim)

o, final_state = chunk_gated_delta_rule(
    q, k, v, g=g, beta=beta, initial_state=initial_state,
    output_final_state=True, cu_seqlens=cu_seqlens,
)
gdn_recurrent_states[layer_idx, slots[:num_seqs]] = final_state
```

Naive fallback (`_naive_gdn_prefill`): accepts `initial_state` parameter, clones `initial_state[i]` per sequence instead of zeros.

### KV storage correctness (`slot_mapping`)

`slot_mapping` in `prepare_prefill_from_bytes` accounts for partial-block offsets:

```cpp
int num_cached_blocks      = num_cached / block_size;
int cached_offset_in_block = num_cached % block_size;

for (int b = num_cached_blocks; b < num_blocks; ++b) {
    int64_t start = block_id * block_size;
    int64_t end   = (b != num_blocks - 1) ? start + block_size : start + last_block_tokens;
    if (b == num_cached_blocks)
        start += cached_offset_in_block;  // skip already-cached slots in first block
    for (int64_t k = start; k < end; ++k)
        meta.slot_mapping.push_back(k);
}
```

This handles the case where `num_cached_tokens` is not block-aligned (e.g. prefix cache hit of 192 tokens with block_size=64 → 3 full blocks cached, `cached_offset_in_block=0`; but if prefix cache hit of 200 tokens with block_size=64 → 3 full blocks + 8 tokens in block 3, `cached_offset_in_block=8`).

## Implementation Status

| Component                                    | Status     | Notes                                                           |
| -------------------------------------------- | ---------- | --------------------------------------------------------------- |
| `PREFILLING` status in `sequence.fbs`        | ✅ Done    |                                                                 |
| `prefilling` deque in `scheduler.h`          | ✅ Done    |                                                                 |
| `_schedule_prefill()` two-step scheduling    | ✅ Done    | PREFILLING queue first, then WAITING with chunking              |
| `postprocess_sequences()` chunk detection    | ✅ Done    | Removed `is_prefill` guard; added PREFILLING→RUNNING transition |
| `PostprocessResult` with `continuations`     | ✅ Done    |                                                                 |
| `preempt()` reset                            | ✅ Done    |                                                                 |
| `num_prompt_tokens` in `interface.fbs`       | ✅ Done    |                                                                 |
| `sampling_token_indices/seq_indices`         | ✅ Done    | C++ populates, Python wires to context                          |
| `ParallelLMHead` sparse extraction           | ✅ Done    |                                                                 |
| `slot_mapping` with `cached_offset_in_block` | ✅ Done    |                                                                 |
| `_build_paged_gather_indices`                | ✅ Done    |                                                                 |
| `_gather_kv_paged` (GQA)                     | ✅ Done    |                                                                 |
| `_gather_cache_paged` (MLA)                  | ✅ Done    |                                                                 |
| MLA chunked prefill in `DeepseekV2Attention` | ✅ Done    |                                                                 |
| GDN conv/recurrent state continuity          | ✅ Done    |                                                                 |
| Prefix caching (`block_manager` wiring)      | 🔲 Pending | `count_active_prefix_hits`, `allocate()`, `can_allocate()`      |

## Verification

**Chunked prefill (verified)**:

1. Model: Qwen3-30B-A3B-FP8, `kvcache_block_size=64`, `max_num_batched_tokens=128`
2. Prompt: 1209 tokens → 10 chunks (9×128 + 57)
3. Results:
   - Output coherent: `<think>\nOkay, the user sent a lot of repetition...`
   - `max_tokens=128` respected: Output Length = 128
   - TTFT = 2519.95ms, Decode = 441 tok/s
   - Sequence completed successfully (Completed: 1)

**Attention correctness test**:

1. Run a 2-chunk sequence; confirm the second chunk takes the gather path (log/assert that `block_tables is not None`)
2. Confirm numerical output of the 2-chunk run matches a single-chunk reference (with `max_num_batched_tokens` large enough to fit the whole prompt)

**Prefix caching test** (pending):

1. Send two requests with the same long prefix
2. Second request: `num_cached_tokens > 0`, TTFP faster
3. Check KV outputs match a non-cached run

**Integration**: Run the existing test suite to confirm no regressions.

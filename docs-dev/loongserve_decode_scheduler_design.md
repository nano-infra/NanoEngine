# LoongServe-style Elastic Decode Scheduler Design

## Scope

This design integrates LoongServe-style elastic decode scheduling into NanoDeploy, but only for the decode attention/KV path.

The intended baseline is:

- attention parallelism: DP + SP
- expert parallelism: NanoDeploy's existing EP backend
- model priority: DeepSeek-V3 MLA decode
- no prefill SP replication goal
- no TP/PP/other parallel strategy replication
- no claim that EP is a LoongServe feature

The correct description should be:

> We implement a LoongServe-style elastic decode scheduler for the attention/KV path, while using our own MoE expert-parallel backend for DeepSeek-V3. The LoongServe-style part includes dynamic per-batch SP groups, multi-master decode, per-SP KV accounting, and decode-time KV migration for scale-up/down.

## Existing NanoDeploy Baseline

NanoDeploy already has several pieces that should be reused instead of replaced.

1. Request-level per-SP KV ownership exists.

   `BlockContext` already stores:

   - `master_sp_idx_`
   - `sp_block_table`
   - `block_location`
   - `num_dispatched_tokens`

   `num_dispatched_tokens[sp]` can serve as the NanoDeploy equivalent of LoongServe's `cur_kv_len_list`.

2. Master-peer decode attention is mostly present.

   In MLA decode, the master rank sends Q through SP all-to-all, peer ranks compute partial attention on local KV, and res/lse are gathered back for merge.

3. Multi-master has a partial scheduler/execution shell.

   `filtered_dp_sp_seqs` is already grouped by `master_sp_idx_`, so multiple master ranks can be represented by setting different requests' `master_sp_idx_` in the same decode step.

4. DeepSeek-V3 EP is already NanoDeploy-owned.

   `DeepseekV2MoE` uses DeepEP-style expert parallel backend through NanoDeploy's FFN mesh. This should remain outside the LoongServe-style claim.

The main missing layer is batch-level elastic SP state: NanoDeploy currently places requests and then appends decode KV to the current master rank. It does not yet model a decode batch with dynamic `occupied_instances`, nor does it schedule scale-up/down and KV migration as first-class decode policies.

## Core State Model

Add a C++ scheduler-side state object:

```cpp
struct DecodeBatchState {
    uint64_t batch_id;
    int dp_idx;
    std::vector<std::shared_ptr<Sequence>> seqs;
    std::vector<int> occupied_instances;
    std::vector<int> batch_used_tokens_per_sp;
    std::vector<int> master_sp_for_step;
    std::vector<std::pair<int, int>> mini_batch_ranges;
};
```

Mapping to LoongServe:

- `DecodeBatchState::occupied_instances` maps to LoongServe `Batch.occupied_instances`.
- `Sequence.block_ctx().num_dispatched_tokens` maps to LoongServe `Req.cur_kv_len_list`.
- `batch_used_tokens_per_sp` is recomputed as the sum of request-level per-SP KV lengths.
- `master_sp_for_step` and `mini_batch_ranges` represent multi-master decode planning for the current step.

The scheduler should treat `occupied_instances` as dynamic runtime state, not as a launch-time fixed SP size.

## Decode Step Flow

Each decode step should execute the following flow.

### 1. Recompute KV Accounting

For every DP worker:

- aggregate global SP usage across decode batches:
  - `total_used_tokens_per_sp[sp]`
- aggregate per-batch usage:
  - `batch.batch_used_tokens_per_sp[sp]`
- derive current occupied set:
  - all `sp` where `batch_used_tokens_per_sp[sp] > 0`
  - plus ranks reserved as decode masters for this step

This must be based on `num_dispatched_tokens`, not a single global KV counter.

### 2. Memory-aware Decode Admission

Before a decode step, a batch with `N` active requests needs `N` new KV token slots.

Check whether:

```text
sum(free_tokens(sp) for sp in batch.occupied_instances) >= N
```

If false:

1. Try decode-time scale-up by adding idle SP ranks.
2. If still tight, try KV migration to compact other batches or rebalance this batch.
3. If still impossible, fall back to preemption/offload. The first implementation can fail closed or reuse existing preemption.

This is the core LoongServe-style policy and should live in scheduler C++, not in the worker attention layer.

### 3. Multi-master Mini-batch Planning

For each decode batch:

1. Sort `occupied_instances` by available token capacity and load.
2. Select one or more ranks as masters for the current step.
3. Split `batch.seqs` into mini-batch ranges.
4. For requests in each range, set:

```cpp
seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ = chosen_master;
```

This lets the existing `filtered_dp_sp_seqs` mechanism naturally produce one master-local request list per SP rank.

The policy should mirror LoongServe's decode plan:

- if memory-bound, add ranks so the batch can append one KV per active request;
- if compute-bound, add masters so token compute is split across SP ranks;
- non-master occupied ranks still serve as attention peers.

### 4. Master-peer Attention Execution

The existing MLA path should remain the primary execution path.

The current metadata generator already rebuilds decode metadata from:

- `master_sp_idx_`
- `num_dispatched_tokens`
- `sp_block_table`

It emits:

- `context_lens_flat`
- `global_context_lens_flat`
- `block_tables_flat`
- `context_lens_for_attn`
- `q_slice_get`
- `q_slice_fill`
- `res_*`
- `q_offsets`

The attention layer then performs:

- Q all-to-all from masters to KV owners;
- local partial attention on each occupied peer;
- res/lse all-to-all back to masters;
- inter-rank attention output merge.

This means the first implementation should not rewrite attention kernels. It should make the scheduler produce correct dynamic per-step `master_sp_idx_` and per-SP KV ownership.

### 5. Postprocess and KV Ownership Update

Current postprocess requires `task.sp_idx == master_sp_idx` and appends generated KV to that rank. This is acceptable if the scheduler updates `master_sp_idx_` dynamically for each step.

After sampling:

- append generated token to the current step's master rank;
- increment `num_dispatched_tokens[master_sp]`;
- allocate a new block if the append crosses a block boundary;
- update batch and global per-SP token accounting.

## Scale-up

Decode-time scale-up adds one or more SP ranks to a running decode batch.

Policy:

1. Select idle ranks first:

   ```text
   total_used_tokens_per_sp[sp] == 0
   ```

2. If no idle rank exists, select rank with maximum free KV capacity, only if policy allows sharing.

3. Add selected ranks to:

   ```cpp
   batch.occupied_instances
   ```

4. The new rank may initially hold no historical KV, but it can become a master for some requests and own the new decode KV tokens from this step onward.

This is sufficient to demonstrate compute-bound scale-up without immediate KV migration.

## Scale-down

Decode-time scale-down removes ranks from a batch.

Policy:

1. If `batch_used_tokens_per_sp[sp] == 0`, remove `sp` from `occupied_instances`.
2. If the rank still has KV but scheduler wants to release it, first migrate its KV to another rank.
3. After migration, update request-level per-SP ownership and free source blocks.

Scale-down must be reflected in scheduler state before the next `prepare_decode_cpp()` call.

## KV Migration

NanoDeploy uses paged KV cache, so the first version should use block-level migration rather than copying LoongServe's token-level migration directly.

### Migration Plan

Add a scheduler plan:

```cpp
struct DecodeKVMigrationItem {
    uint64_t seq_id;
    int src_sp;
    int dst_sp;
    std::vector<int> src_block_ids;
    std::vector<int> dst_block_ids;
    int migrated_tokens;
};

struct DecodeKVMigrationPlan {
    int dp_idx;
    std::vector<DecodeKVMigrationItem> items;
};
```

### Migration Execution

For each migration item:

1. Scheduler allocates destination blocks from `BlockManager[dst_sp]`.
2. Worker copies KV cache block data from `src_sp` to `dst_sp`.
3. Scheduler updates:

   - `num_dispatched_tokens[src_sp] -= migrated_tokens`
   - `num_dispatched_tokens[dst_sp] += migrated_tokens`
   - `sp_block_table[src_sp]`
   - `sp_block_table[dst_sp]`
   - `block_location`

4. Source blocks are freed.

### First-version Limitation

Use full-block migration only:

- DeepSeek-V3 MLA block size is 64 in NanoDeploy.
- This is enough to demonstrate decode-time KV migration.
- Partial-block migration can be added later if scale-down needs exact token compaction.

This is different from LoongServe's token-level migration kernel, but it is a better fit for NanoDeploy's paged block manager.

## File-level Development Plan

### `csrc/nanodeploy/scheduler/scheduler.{h,cpp}`

Add:

- `DecodeBatchState`
- a LoongServe-style decode scheduler branch
- dynamic `occupied_instances`
- multi-master mini-batch planning
- decode scale-up/down calls
- migration plan production

The existing `_schedule_decode()` can remain as the legacy fixed-placement path.

### `csrc/nanodeploy/scheduler/sp_state_manager.{h,cpp}`

Add:

- per-batch KV accounting helpers
- free-token checks per SP rank
- scale-up rank selector
- scale-down candidate selector
- block-level migration planner

Keep request-level allocation logic reusable where possible.

### `csrc/nanodeploy/scheduler/block_manager.{h,cpp}`

Expose safe primitives for migration:

- allocate a specific number of fresh blocks;
- attach allocated blocks to a sequence's `sp_block_table`;
- release source blocks after successful GPU copy;
- query block capacity and block size.

### `csrc/nanodeploy/worker/model_runner_utils.cpp`

Avoid heavy rewrites.

Required checks:

- metadata remains correct when `master_sp_idx_` changes every step;
- block tables remain packed only for ranks with local KV;
- `context_lens_for_attn` and `q_offsets` match dynamic occupied state.

### `nanodeploy/worker/cache.py`

Add intra-DP/SP KV block copy support.

Current `CacheContext.migrate()` targets cross-engine RDMA using `ACTIVE/MIGRATE` contexts. The new migration path should be same-engine, SP-to-SP, block-oriented.

### `nanodeploy/worker/model_runner.py`

Add a worker RPC method such as:

```python
def migrate_decode_kv_blocks(self, migration_plan):
    ...
```

This method should execute GPU copy for all local source/destination assignments relevant to the worker's SP rank.

### `nanodeploy/engine/ray_executor.py`

Add a collective migration RPC wrapper:

```python
def migrate_decode_kv(self, plans):
    ...
```

### `nanodeploy/config.py`

Add feature flags:

```python
loongserve_decode_scheduler: bool = False
loongserve_enable_kv_migration: bool = False
loongserve_migration_granularity: Literal["block"] = "block"
loongserve_min_comp_bound_batch_size: int = 16
```

Keep this independent from existing `enable_dynamic_sp_size` until behavior is stable.

## Phased Delivery

### Phase 1: Dynamic Batch State + Multi-master Decode

Goal:

- prove dynamic per-batch `occupied_instances`;
- prove multi-master decode;
- no KV migration yet.

Implementation:

- build `DecodeBatchState`;
- allow scale-up into empty ranks;
- schedule multiple masters per step;
- append new KV to per-step master rank;
- remove empty occupied ranks.

This already avoids being merely "SP attention helper".

### Phase 2: Block-level KV Migration

Goal:

- move KV blocks between SP ranks during decode;
- update per-SP ownership;
- support scale-down after migration.

Implementation:

- scheduler creates migration plan;
- workers copy KV blocks;
- block manager metadata is committed after copy succeeds.

### Phase 3: Full Memory-aware Policy

Goal:

- match LoongServe-style elastic decode policy more closely.

Implementation:

- batch merge under memory pressure;
- migration cost model;
- compute-bound scale-up based on batch size;
- preempt/offload fallback;
- richer metrics and logging;
- CUDA graph compatibility review.

## Validation Plan

### Unit Tests

Add C++/Python tests for:

- `occupied_instances` updates;
- multi-master mini-batch splitting;
- per-SP `num_dispatched_tokens` after decode;
- scale-up adds idle ranks;
- scale-down removes ranks with zero KV;
- migration updates `sp_block_table`, `block_location`, and `num_dispatched_tokens`.

### Metadata Tests

Build synthetic `Sequence` sets and compare `prepare_decode_cpp()` outputs before/after:

- dynamic master reassignment;
- scale-up;
- scale-down;
- block-level migration.

Must validate:

- `context_lens_flat`
- `global_context_lens_flat`
- `block_tables_flat`
- `context_lens_for_attn`
- `q_slice_get`
- `q_slice_fill`
- `res_slice_*`
- `q_offsets`

### Distributed Correctness

Run SP decode correctness with:

- SP=1 baseline;
- SP>1 fixed placement;
- SP>1 dynamic LoongServe-style scheduler;
- deterministic sampling or dummy logits.

Compare output token sequence and no-hang behavior.

### Observability

Per step, log:

- batch id
- `occupied_instances`
- `batch_used_tokens_per_sp`
- `master_sp_for_step`
- `mini_batch_ranges`
- `sp_q_matrix`
- `sp_res_matrix`
- migration bytes/blocks/tokens
- scale-up/down decisions

## Minimal Defensible Version

For a tight rebuttal timeline, implement these in order:

1. Dynamic `occupied_instances`.
2. Per-request per-SP KV length tracking through `num_dispatched_tokens`.
3. Master-peer attention partial + merge using existing MLA SP backend.
4. Multi-master decode mini-batch splitting.
5. Decode-time scale-up/down plus block-level KV migration.

Without item 4, the system is only an SP attention helper.

Without item 5, it is fixed-SP decode.

Without item 2, the scheduler cannot claim memory-aware elastic policy.

## Main Risk

The main risk is not the attention kernel. The existing attention path already has the right master-peer structure.

The main risk is metadata consistency after dynamic changes:

- `master_sp_idx_` changes per step;
- `num_dispatched_tokens` changes through append and migration;
- block tables must remain aligned with `context_lens_for_attn`;
- RPC optimization must not filter away sequence skeletons needed by decode metadata.

The implementation should therefore first stabilize scheduler state and metadata tests before optimizing migration performance.

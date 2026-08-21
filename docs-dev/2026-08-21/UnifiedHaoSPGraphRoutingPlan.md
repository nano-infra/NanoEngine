# Unified Hao Q Routing and SP CUDA Graph Padding

## Status

- Design and review plan only. No runtime code is changed by this document.
- Recorded at: `2026-08-21`.
- NanoDeploy baseline: `7f62a8fb972d8bc2cdf1c4d4695d2fb277d86286`.
- DLSlime baseline: `8078d0123fedca3b99e2b941c5e32230d31f2dd6`.
- The pre-existing NanoDeploy `.gitignore` modification is user-owned and is
  outside this work.
- The destination-aware DLSlime primitive is already implemented. This plan is
  expected to require NanoDeploy changes only.

This document follows
[`DestinationAwareQRouting.md`](./DestinationAwareQRouting.md). The first
change fixed non-nested dynamic-SP Q placement. This follow-up removes the
remaining fixed-full-SP Q-routing exception and restores one shared SP data
path without losing CUDA Graph correctness after EOS.

## Decision summary

Fixed SP is a scheduling policy, not a separate communication protocol.

The target architecture is:

```text
fixed or dynamic participant policy
              |
              v
common receiver-local packed metadata
              |
              v
Hao Q always uses destination-aware rows
              |
              v
common CUDA Graph padding for unused graph rows
```

The implementation should:

1. use `dst_row_indices` for every Hao decode-Q all-to-all, including
   `fixed_sp_size == attention_sp`;
2. keep real attention rows receiver-local and packed;
3. make Graph-only rows locally safe without sending padding Q over the
   network;
4. remove the fixed-full-SP dense transport layout and its packed-to-dense
   remapping;
5. retain `fixed_sp_size` only as scheduler/admission policy;
6. leave NCCL and Res/LSE transpose communication unchanged in the first
   patch;
7. retain `q_mask` for NCCL, diagnostics, compatibility, and regression
   comparison even when Hao Q no longer consumes it.

No new `SPGraphPaddedLayout` class is required. The preferred implementation
is to extend the existing common Graph metadata materialization path.

## Historical behavior

Before commits `4bb17b3` and `ebb69bf`, fixed and dynamic SP shared the same
Graph metadata copy path:

1. `prepare_decode_cpp` produced receiver-local packed metadata;
2. `_copy_decode_context_to_graph_vars` cleared persistent Graph tensors;
3. actual packed metadata was copied into the leading rows;
4. the selected CUDA Graph replayed an `attn_bs` bucket greater than or equal
   to the actual attention row count.

There was no `FixedSPGraphLayout`, no `packed_attention_rows_to_dense`, and no
fixed-specific metadata copier. Full fixed SP already selected the maximum
`sp_world_size * master_bs` attention shape, but it still used the common copy
mechanism.

That mechanism was unsafe after EOS. For SP8 with one master request per rank,
removing the rank-0 request and inserting its rank-local control dummy changes
the actual receiver rows from:

```text
before EOS: rank 0..7 each receive 8 packed rows
after EOS:  rank 0 receives 8 rows; ranks 1..7 receive 7 rows
```

The old offset protocol then wrote some Q values to the wrong receiver rows,
while the final captured row could contain zero/invalid attention metadata or
stale Q data from the persistent DLSlime buffer.

The repository addressed this incrementally:

- `4bb17b3`: fall back to eager for non-uniform fixed-full-SP batches;
- `ebb69bf`: introduce a fixed-only dense `[source_master, local_slot]` layout
  so full CUDA Graph could remain enabled;
- `ef8bb24`: introduce destination-aware rows for dynamic/partial SP while
  excluding the already-special-cased fixed-full-SP path.

The fixed dense layout was therefore a compatibility repair for the old
protocol, not a fundamental requirement.

## Required invariants

The shared implementation must enforce all of the following.

### Real Q rows

- Each receiver's real attention rows occupy `[0, actual_attn_bs)`.
- For every active `(source, local_slot, destination)` edge,
  `dst_row_indices[destination, local_slot]` names exactly the row paired with
  that receiver's context length and block table.
- Every real row is written exactly once: no collision and no missing row.
- `-1` continues to mean no transfer. Fixed SP must not turn padding into
  network traffic.

### CUDA Graph padding rows

- The selected graph satisfies `graph_attn_bs >= actual_attn_bs`.
- Rows `[actual_attn_bs, graph_attn_bs)` contain a deterministic finite dummy
  Q before FlashMLA reads them. They must never expose a previous replay's
  data.
- Padding rows use a legal read-only block table and a supported nonzero
  context length.
- Padding rows produce no Res/LSE transfer and do not participate in the real
  request reduction.
- Graph-only local master slots have one finite local dummy partial so the
  combine kernel never reduces an all-empty row. Those outputs remain outside
  the actual sampling slice.
- `slot_mapping == -1` prevents Graph-only inputs from writing KV cache.

### Collective and Graph shape safety

- Q/Res/LSE collective launch contracts must remain compatible on every SP
  rank even when ranks have different `actual_attn_bs` values.
- Runtime row count must not change a captured tensor shape or pointer.
- Padding initialization must not race with remote writes. Do not clear a
  persistent receive buffer concurrently with peers writing into it.
- Actual remote rows and local padding rows must be disjoint by construction.

## Proposed implementation

### 1. Establish a reproducible pre-change baseline

Before changing runtime code, record the exact commit hashes and save results
for:

- the existing destination-row DLSlime microbenchmark;
- fixed SP8 Hao full CUDA Graph with global BS8 and BS32;
- fixed SP8 EOS correctness with mixed completion lengths;
- dynamic bucket SP8 full CUDA Graph with mixed long/short requests;
- `max_num_seqs=256`, with actual per-rank traffic reported separately from
  configured capacity.

For latency comparisons, use `ignore_eos=True` and a fixed decode length so
the old and new runs execute the same request topology and token count. Run a
separate EOS-enabled test for correctness.

The benchmark must report at least:

- active remote Q rows per rank;
- Q payload bytes per rank;
- Q all-to-all kernel latency;
- total Q communication-region latency;
- decode inter-token latency and effective output throughput;
- Graph capture count, capture duration, and reserved GPU memory.

### 2. Enable destination rows for all Hao Q calls

Remove the `fixed_sp_size != sp_size` / `fixed_sp_size != sp_world_size`
conditions that currently suppress destination rows for fixed-full SP.

For eager and ordinary packed Graph execution, consume the existing
`DecodeMetadata::q_dst_row_indices_flat` generated by `prepare_decode_cpp`.
The attention layer continues to make one call:

```python
q_buffer.all_to_all_ll(q, dst_row_indices=q_dst_row_indices)
```

Hao Q must not simultaneously pass `mask` or `offsets`. NCCL keeps its current
mask/packing behavior.

### 3. Generalize the existing Graph metadata copier

Extend the common Graph copy operation so it knows both:

- `actual_attn_bs`: number of receiver-local real packed rows;
- `graph_attn_bs`: attention batch shape captured by the selected Graph.

The common copier should materialize:

- actual packed context lengths and block tables in the leading rows;
- destination-aware Q row indices for real rows;
- safe local dummy metadata for the padding tail;
- sparse Res/LSE mappings only for actual rows;
- legal local combine metadata for Graph-only master slots.

The current fixed-only `_copy_fixed_sp_context_to_graph_vars` behavior should
be decomposed into generic padding operations rather than selected by
`fixed_sp_size`.

### 4. Initialize padding Q locally and race-free

Do not send dummy Q to every remote rank, and do not asynchronously clear the
whole persistent receive buffer.

Preferred design:

1. destination-aware A2A writes only real rows;
2. a captured local operation writes deterministic finite values only to the
   known padding-row indices on the receiver;
3. padding-row indices are disjoint from every valid destination row;
4. this local initialization completes before FlashMLA consumes the Q buffer.

The local operation may copy one valid local Q row or write zero values. The
chosen dummy Q value does not affect real output because its attention result
is masked, but it must remain finite and deterministic.

The implementation must include a multi-rank test that intentionally skews
rank progress to catch reset/write races; the earlier test-only buffer reset
race must not be reintroduced.

### 5. Remove fixed-only dense transport state

After the common path passes correctness tests, remove the fixed-only state
whose sole purpose is dense transport:

- `FixedSPGraphLayout`;
- `build_fixed_sp_graph_layout`;
- `packed_attention_rows_to_dense`;
- `_fixed_sp_graph_layout_cache`;
- `_get_fixed_sp_graph_device_layout`;
- `_copy_fixed_sp_context_to_graph_vars`;
- `sp_graph_master_bs` and `sp_graph_packed_row_to_dense`, if they no longer
  serve any non-transport purpose.

Do not delete scheduler/admission handling for `fixed_sp_size` in this change.
That option still controls how many ranks receive KV for a request.

### 6. Keep Graph bucket selection separate from correctness

Fixed and dynamic modes must share the same row-routing and padding logic, but
they do not have to capture identical sets of Graph shapes.

Initially preserve the current fixed-full maximum attention bucket if that
avoids extra Graph capture time and memory. Treat it as a bucket-selection
optimization, not a distinct metadata or communication path. Once the common
path is stable, benchmark whether smaller post-EOS attention buckets improve
steady-state latency enough to justify additional captures.

Any change to bucket candidates must be accepted on measured startup time,
GPU memory, and decode latency rather than assumed to be free.

### 7. Preserve compatibility paths

- Keep the legacy DLSlime all-to-all entry point. Res/LSE transpose still uses
  it, and out-of-tree callers may depend on it.
- Keep `q_offsets` until NCCL and diagnostics no longer require it.
- Keep `q_mask`; Hao destination-row Q may ignore it, but NCCL, logging, CPU
  semantic oracles, and compatibility tests still consume it.
- Do not modify the scheduler's participant decisions as part of the data-path
  refactor.

## Correctness validation

Run the smallest tests first, reinstalling NanoDeploy after any C++ change.

### CPU and metadata tests

- Existing scheduler and packed-row semantic tests.
- A common fixed/dynamic destination-row oracle asserting no collision and no
  missing real row.
- EOS transition tests covering receiver row counts such as `8/7/7/...`.
- Padding tests asserting legal context lengths/block tables, finite dummy Q,
  zero Res/LSE contribution, and unchanged real-row ordering.
- Tests showing fixed-full and an equivalent constant-participant bucket policy
  produce the same communication metadata.

### Single-node SP8 GPU tests

- Hao eager: fixed SP8 and dynamic bucket.
- Hao full CUDA Graph: fixed SP8 and dynamic bucket.
- Hao piecewise CUDA Graph if that mode remains supported.
- Global BS8 where each rank owns one request.
- Multiple local master rows per rank.
- Mixed EOS order: one early EOS, several later EOS, and at least one request
  reaching `max_tokens`.
- Verify generated token IDs against eager or a recorded trusted baseline, not
  only output lengths.
- Re-run NCCL regressions to ensure retained masks and offsets remain valid.

### P/D acceptance

Run the two-node P/D scenario with Hao, SP8 decode, full CUDA Graph, BS8, and
mixed long/short prompts. Verify that early EOS requests finish normally and
survivors continue producing coherent tokens. Preserve the complete command,
topology, generated sequences, per-step latency, and SP participant histogram.

## Performance validation and acceptance gates

Compare pre-change and post-change results on the same node allocation and
driver environment.

Mandatory matched tests:

1. fixed SP8 full Graph, global BS8, fixed decode length;
2. fixed SP8 full Graph, global BS32 or higher;
3. dynamic bucket with 20-30 active remote Q rows per rank;
4. `max_num_seqs=256` with actual traffic substantially below capacity;
5. EOS-enabled tail where the number of live requests decreases over time.

Acceptance criteria:

- identical valid Q payload bytes for identical participant matrices;
- no network transmission for Graph-only padding rows;
- no correctness regression in eager, full Graph, piecewise Graph, or NCCL;
- fixed-SP matched-topology decode ITL does not regress beyond normal run noise;
- no material increase in Graph capture time or GPU memory without an explicit
  reviewed tradeoff;
- dynamic sparse performance remains within noise of the current
  destination-row baseline;
- if a regression exceeds 1%, repeat enough runs to establish confidence and
  profile before merging.

The already-recorded uniform SP8 Q microbenchmark shows destination rows within
approximately 0.6% of the legacy fixed route across the tested batch sizes,
and slightly faster for the global-BS8-equivalent case. That result supports
the design but does not replace the production-level before/after test above.

## Patch and commit sequence

Keep changes reviewable and independently testable:

1. `test: cover common SP graph padding semantics`
   - add CPU row/padding oracles and any benchmark instrumentation needed for
     configured capacity versus active rows;
2. `refactor: unify Hao Q routing across SP policies`
   - enable destination rows for fixed-full SP and use the common metadata
     path without deleting the old fixed helper yet;
3. `fix: make SP graph padding locally safe`
   - add race-free local Q padding and legal attention metadata;
4. `refactor: remove fixed SP dense transport layout`
   - delete fixed-only transport state after all correctness tests pass;
5. optional performance patch only if profiling identifies a measured issue.

After every patch, record exact commands and results. Do not combine a
performance optimization with the first correctness refactor unless the
benchmark demonstrates it is necessary.

## Rollback and review checkpoints

- Keep commits narrow so the unified Hao Q-routing commit can be reverted
  without reverting the DLSlime destination-row primitive.
- Stop and review if safe local Graph padding requires new network traffic,
  whole-buffer clears, or a new cross-rank barrier.
- Stop and review if fixed and dynamic ranks would launch incompatible
  collective shapes.
- Stop and review if the implementation requires DLSlime changes beyond the
  already-committed destination-row API.
- Do not remove fixed-only helpers until both EOS correctness and matched
  performance have passed.

## Expected end state

After completion:

- scheduling policy determines participants;
- one receiver-local packed metadata protocol represents both fixed and
  dynamic SP;
- every Hao Q uses destination-aware rows;
- one generic CUDA Graph materializer handles real rows and local padding;
- fixed SP has no special Q transport or dense row-remapping path;
- Graph padding is local, deterministic, race-free, and excluded from real
  reductions;
- sparse communication volume is determined only by real participant edges.

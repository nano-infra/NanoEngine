# Fixed-SP CUDA Graph Padding Across EOS

## Symptom

`DP1/SP8` eager decode is correct after the KV block-size fix.  Full CUDA
Graph decode corrupts one request after another request reaches EOS.  Before
EOS, eight requests produce eight packed attention rows on every participant.
After request 0 exits, the scheduler inserts a rank-0-only control dummy:

- rank 0 still has eight valid local KV rows;
- ranks 1 through 7 have seven packed local KV rows;
- the fixed graph still replays `attention_compute_bs == 8` on every rank.

The old graph copy path zero-filled the missing context length and block table.
DLSlime's persistent Q receive buffer also retained whatever occupied that
unfilled row during the preceding replay.  FlashMLA therefore consumed an
invalid zero-length row and a stale query.  Falling the whole batch back to
eager avoided corruption but discarded the graph speedup.

## Reference behavior

vLLM separates the number of actual tokens from the padded CUDA Graph batch
shape.  Its attention backend declares the graph shapes it supports (FlashMLA
uses uniform batches), while graph-padded tokens receive stable metadata and
invalid KV writes use a padding slot.  NanoDeploy pure DP already has the same
basic property: model inputs are padded to a captured bucket and
`slot_mapping == -1` prevents graph-only rows from writing KV.  Fixed SP needs
to extend that separation through the Q and Res/LSE all-to-all metadata.

## Dense fixed-SP layout

For captured per-master bucket `master_bs`, every participant replays the
same dense attention layout:

```text
dense_row = source_master * master_bs + master_local_slot
attention_bs = sp_world_size * master_bs
```

Runtime C++ metadata remains packed.  Python records a packed-to-dense row map
when preparing the decode quantum, then materializes the dense graph metadata
before each replay.

### Query transport

- Every local graph input slot, including graph padding, gets a deterministic
  `q_slice_get` and dense `q_slice_fill` index.
- `q_offsets` is `[0, master_bs, 2 * master_bs, ...]` on every rank.
- Every dense query is explicitly sent to every remote participant.  A missing
  KV shard therefore cannot expose stale data from DLSlime's persistent receive
  buffer.

### Attention rows

- Real packed context lengths and block tables are scattered into their dense
  master/slot rows.
- A row with no local KV borrows the first valid block table and uses
  `context_len == 1` so FlashMLA receives legal metadata.
- The borrowed page is read-only.  Graph-only model inputs retain
  `slot_mapping == -1`, so the attention padding path never writes KV.

### Result transport and reduction

- Original zero values in `context_lens` and `global_context_lens` are
  preserved for real logical requests.  A participant without KV for that
  request cannot contribute its throw-away attention result.
- Res/LSE masks are derived from real local context presence, not from the
  dense Q transport mask.
- A graph-only local master slot enables only its local borrowed partial.  This
  keeps the combine kernel finite while the row remains outside the actual
  model output and sampling slices.

This distinction is essential: making every dense row look globally valid
would mix borrowed KV into real requests.

## Scope and validation

The change is Python-only.  It replaces the eager fallback for fixed-full-SP
graphs and applies to full and piecewise graph metadata.  Eager, local graphs,
pure DP, partial fixed SP, and non-fixed SP retain their existing packed path.

CPU regression coverage verifies the post-EOS `8/7/7/...` packed-row split,
the common eight-row dense graph layout, stable master/slot identities, legal
attention lengths, explicit dense Q transport, and sparse result masks.

The end-to-end acceptance test is the existing P/D smoke with:

- decode topology `sp8` and backend `hao_basic`;
- full CUDA Graph;
- eight requests and 128 maximum tokens;
- EOS enabled.

It must show no eager-fallback warning, no repeated-token corruption, and a
completion-length vector in which an EOS request can finish early while the
remaining requests continue correctly.

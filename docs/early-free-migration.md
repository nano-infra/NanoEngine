# Early Free Migration

## Overview

Early free migration is an optimization for NanoDeploy's prefill-decode disaggregated architecture that significantly reduces KV cache memory pressure on the prefill engine by releasing resources immediately after migration completes, rather than waiting for the entire generation to finish.

## Problem Statement

In the original implementation, the decode engine only sent free requests to the prefill engine **after full generation completes** (`seq.is_finished == True`). This caused the prefill engine to hold KV cache blocks unnecessarily during the entire decode phase:

- A typical 100-token generation at 5ms/token takes ~500ms
- During this time, the prefill engine's KV cache blocks remain allocated
- This limits the prefill engine's ability to serve new requests
- Resources are locked despite the migration being complete

## Solution

Send P2P free requests **immediately after KV cache migration completes** on the decode engine, allowing the prefill engine to reclaim blocks hundreds of milliseconds earlier and serve new requests faster.

## Architecture

### Original Flow

1. Prefill engine completes prefill → marks sequence `TO_BE_MIGRATED`
2. Decode engine receives migration message → calls `executor.migrate()` (RDMA transfer)
3. Decode engine runs generation loop (many steps, ~500ms for 100 tokens)
4. **Only when** `seq.is_finished == True` → decode engine sends P2P free request
5. Prefill engine receives free → calls `scheduler.free_to_be_migrated()` → releases blocks

### Optimized Flow

1. Prefill engine completes prefill → marks sequence `TO_BE_MIGRATED`
2. Decode engine receives migration message → calls `executor.migrate()` (RDMA transfer)
3. **Immediately after migration** → decode engine sends P2P free request ⚡ **EARLY FREE**
4. Prefill engine receives free → releases blocks (hundreds of ms earlier)
5. Decode engine continues generation → when finished, cleanup (duplicate free prevented)

## Implementation Details

### Components Modified

All changes are in `/NanoDeploy/nanodeploy/server/engine_server.py` (~30 lines of code).

### Key Mechanisms

#### 1. Sequence Tracking

Two instance variables track sequences across engine steps:

```python
self._previous_running_seqs: set[int] = set()  # Sequences running in previous step
self._freed_sequences: set[int] = set()        # Sequences already freed (prevents duplicates)
```

**Memory bounds**: Both sets are bounded by `max_num_seqs` (typically ~100s of sequences).

#### 2. Migration Detection

After each `engine.step()`, the decode engine:

1. Builds a set of currently running sequence IDs
2. Compares with previous step to find **newly appeared sequences**
3. For each newly appeared sequence:
   - Checks if it has a `MIGRATE` BlockContext with `engine_id`
   - If yes → this sequence was just migrated → send early free

```python
# Build current running set
current_running_seqs = set()
for seqs in dp_seqs:
    for seq in seqs:
        current_running_seqs.add(seq.seq_id)

# Detect newly migrated sequences (decode engine only)
if self.engine.config.mode == "decode":
    newly_appeared = current_running_seqs - self._previous_running_seqs

    for seqs in dp_seqs:
        for seq in seqs:
            if seq.seq_id in newly_appeared:
                migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
                if migrate_ctx and migrate_ctx.engine_id:
                    logger.info(f"Early free: seq {seq.seq_id} migrated from {migrate_ctx.engine_id}")
                    self._send_p2p_free_if_migrated(seq)

# Update tracking for next step
self._previous_running_seqs = current_running_seqs
```

#### 3. Duplicate Prevention

The `_send_p2p_free_if_migrated()` method checks if a sequence was already freed:

```python
def _send_p2p_free_if_migrated(self, seq):
    """Send P2P free instruction to source engine if sequence was migrated."""
    # Skip if already freed (prevents duplicate free requests)
    if seq.seq_id in self._freed_sequences:
        return

    migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
    if migrate_ctx and migrate_ctx.engine_id:
        self.engine.send_free_sequences(migrate_ctx.engine_id, [seq.seq_id])
        # Mark as freed to prevent duplicates
        self._freed_sequences.add(seq.seq_id)
```

This ensures:

- Each sequence sends at most one free request
- The free-on-finish fallback is skipped if early free succeeded
- No duplicate P2P messages

#### 4. Memory Management

When sequences finish, cleanup tracking data:

```python
if seq.is_finished:
    self._send_stepout(seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED)
    self._send_p2p_free_if_migrated(seq)  # Will skip if already freed
    # Clean up tracking to prevent memory leak
    self._freed_sequences.discard(seq.seq_id)
```

### Design Rationale

**Q: Why track sequences across steps?**
Migration completes when the decode engine receives the sequence and it first appears in `dp_seqs`. The appearance in the running set signals successful migration.

**Q: Why keep free-on-finish?**
Safety fallback in case early free somehow fails. Since the operation is idempotent (prefill engine's `free_to_be_migrated()` is a map lookup), duplicate free requests are harmless.

**Q: Why decode engine only?**
The prefill engine doesn't receive migrated sequences, so the mode check (`config.mode == "decode"`) prevents unnecessary work.

**Q: Won't the tracking sets grow unbounded?**
No - both sets are bounded by `max_num_seqs` (~100s). The `_freed_sequences` set is explicitly cleaned on sequence finish to prevent memory leaks.

## Performance Impact

### Benefits

- **Latency improvement**: 100-token generation at 5ms/token = **~500ms saved** per sequence on prefill engine
- **Throughput improvement**: Prefill engine can serve new requests immediately (more free blocks available)
- **Better resource utilization**: KV cache blocks freed as soon as they're no longer needed
- **Scales with generation length**: Longer generations = more time saved

### Overhead

- **Minimal computational cost**: Set operations are O(1), ~10-20 extra lines in hot path
- **No additional network messages**: Same P2P free message, just sent earlier
- **Negligible memory overhead**: Two small sets bounded by max_num_seqs

## Testing and Verification

### Test Setup

1. **Start prefill engine**:

   ```bash
   python -m nanodeploy.server.engine_server --mode prefill --config configs/prefill.yaml
   ```

2. **Start decode engine**:

   ```bash
   python -m nanodeploy.server.engine_server --mode decode --config configs/decode.yaml
   ```

3. **Send test request** with long output (e.g., 100 tokens) to trigger migration

### Expected Log Output

**Decode Engine**:

```
Early free: seq 12345 migrated from engine_prefill_0
Sequence 12345 sending P2P free to source engine engine_prefill_0
```

**Prefill Engine**:

```
Received P2P free request from engine_decode_0 for 1 sequences: [12345]
Freed sequence 12345
```

### Verification Checklist

- [ ] Free request appears immediately after migration (1-2 steps), not after generation completes
- [ ] Each sequence shows only ONE "sending P2P free" log entry
- [ ] Prefill engine logs show "Freed sequence" message hundreds of ms earlier than before
- [ ] No errors or warnings related to P2P free operations
- [ ] Generation completes successfully with correct output

### Timing Analysis

Before optimization:

```
Migration complete: T=0ms
Generation continues: T=0-500ms (100 tokens @ 5ms/token)
Generation finishes: T=500ms
Free request sent: T=500ms ❌
Prefill blocks freed: T=500ms
```

After optimization:

```
Migration complete: T=0ms
Free request sent: T=1-2ms ✅ (immediately after migration)
Prefill blocks freed: T=1-2ms
Generation continues: T=0-500ms
Generation finishes: T=500ms
```

**Time saved per sequence**: ~498ms

### Edge Cases Covered

| Edge Case              | Behavior                                               |
| ---------------------- | ------------------------------------------------------ |
| Migration failure      | Sequence won't appear in `dp_seqs`, no early free sent |
| P2P free failure       | Logged but non-fatal (existing error handling)         |
| Duplicate prevention   | `_freed_sequences` set ensures free sent only once     |
| Non-migrated sequences | Only sequences with MIGRATE BlockContext trigger free  |
| Memory leaks           | Tracking sets cleaned on sequence finish               |
| Prefill engine mode    | Mode check prevents unnecessary tracking work          |

## Configuration

The feature is enabled by default with no configuration required.

### Optional: Disable Early Free (if needed)

If issues arise, add a config flag to disable:

```python
# In engine_server.py
if self.engine.config.enable_early_free:  # Default True
    # Early free logic
```

Add to config:

```yaml
enable_early_free: false  # Disable optimization
```

## Monitoring and Metrics

### Key Metrics to Track

1. **Prefill engine KV cache utilization**: Should decrease due to faster block recycling
2. **Prefill engine request queue length**: Should decrease as capacity increases
3. **Time between migration and free**: Should drop from ~500ms to ~1-2ms
4. **Decode engine memory usage**: Should remain stable (no memory leaks from tracking)

### Log Analysis

Search for early free events:

```bash
# Count early free occurrences
grep "Early free:" decode_engine.log | wc -l

# Verify no duplicates (should match early free count)
grep "sending P2P free" decode_engine.log | wc -l

# Check prefill side
grep "Freed sequence" prefill_engine.log | wc -l
```

## Troubleshooting

### Symptom: No early free logs on decode engine

**Possible causes**:

- Decode engine not receiving migrated sequences
- Sequences don't have MIGRATE BlockContext
- `config.mode` is not set to "decode"

**Debug**:

```bash
# Check if sequences are being migrated
grep "migrate" decode_engine.log

# Verify engine mode
grep "Mode:" decode_engine.log
```

### Symptom: Duplicate free requests

**Possible causes**:

- `_freed_sequences` set not being checked properly
- Race condition in concurrent execution

**Debug**:

```bash
# Count free requests per sequence
grep "sending P2P free" decode_engine.log | awk '{print $3}' | sort | uniq -c
```

### Symptom: Memory leak in decode engine

**Possible causes**:

- `_freed_sequences` not being cleaned up on finish
- Sequences not reaching finish state

**Debug**:
Check tracking set sizes (add temporary logging):

```python
logger.debug(f"Tracking sets: running={len(self._previous_running_seqs)}, freed={len(self._freed_sequences)}")
```

## Related Components

### Modified Files

- **`/NanoDeploy/nanodeploy/server/engine_server.py`** (PRIMARY)
  - Lines 36-38: Tracking sets initialization
  - Lines 135-160: `_send_p2p_free_if_migrated()` with duplicate prevention
  - Lines 186-207: Early free detection logic
  - Line 222: Cleanup on sequence finish

### Referenced (No Changes)

- **`/NanoDeploy/nanodeploy/llm_component.py`**

  - Lines 181-267: `send_free_sequences()` - existing P2P infrastructure

- **`/NanoDeploy/nanodeploy/engine/llm_engine.py`**

  - Line 202: `executor.migrate()` - where migration happens

## Future Enhancements

1. **Metrics Integration**: Add Prometheus metrics for early free timing and success rate
2. **Adaptive Thresholds**: Configure early free based on generation length estimates
3. **Batch Free Requests**: Combine multiple free requests into a single P2P message
4. **Cross-Engine Coordination**: Coordinate early free with prefill scheduling decisions

## References

- [NanoDeploy Architecture](./architecture.md)
- [P2P Communication Protocol](./p2p-protocol.md)
- [KV Cache Management](./kv-cache.md)
- [Prefill-Decode Disaggregation](./disaggregation.md)

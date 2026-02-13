# Bug Finding Plan: Pickle Corruption in Sequence Serialization

## Problem Statement

After removing RPCEndpoint and switching to Ray's native pickle serialization, we get SIGSEGV when trying to serialize Sequence objects:

```
*** SIGSEGV received at time=1770805466 on cpu 95 ***
PC: @     0x7f075da234e8  (unknown)  (unknown)
Stack: ray_executor.py:146 → pickle.dumps(seq)
```

## What We Know

1. **RDMA approach worked**: Used custom `serialize(buffer_ptr, ...)` function successfully
2. **Python pickle fails**: Both Ray's delayed pickle AND immediate `pickle.dumps()` crash
3. **Custom pickle exists**: `sequence_binding.cpp:291-348` implements `__getstate__`/`__setstate__`
4. **Crash location**: Inside FlatBuffer's `Pack()` function when called from pickle

## What We Don't Know

- **Is the object corrupted before pickle?** Or is pickle itself broken?
- **What specific pointer/field causes the crash?** Which member of `seq.data_` is invalid?
- **When does corruption happen?** During schedule()? During postprocess()? Or never?
- **Why did RDMA work?** What does `serialize()` do differently than pickle's `Pack()`?

## Hypotheses to Test

### Hypothesis 1: Objects Are Valid, Pickle Implementation Is Broken

**Theory**: The Sequence objects are actually fine, but the pickle implementation has a bug (buffer size, threading, etc.)

**Evidence For**:

- Sequences work fine during scheduling
- Custom pickle might have implementation issues

**Evidence Against**:

- Immediate `pickle.dumps()` also crashes (not just Ray's delayed pickle)
- Other pickle implementations in the same file (BlockContext) work fine

**Test**:

```python
# In ray_executor.py, before pickle.dumps():
for seq in seqs:
    print(f"seq_id: {seq.seq_id}")  # Access simple field
    print(f"num_tokens: {seq.num_tokens}")  # Access another field
    print(f"status: {seq.status}")  # Check status
    # If these work, object is valid

    # Try to access block context
    try:
        ctx = seq.block_ctx()
        print(f"engine_id: {ctx.engine_id}")
        print(f"attention_sp: {ctx.attention_sp}")
    except Exception as e:
        print(f"ERROR accessing block_ctx: {e}")
```

### Hypothesis 2: Pointers Corrupted During Scheduler Postprocess

**Theory**: The scheduler's `postprocess()` method invalidates internal pointers (sp_block_table, etc.)

**Evidence For**:

- Crash happens when `Pack()` walks the object graph
- Sequences contain unique_ptr that could be moved

**Evidence Against**:

- Why would postprocess corrupt objects that need to be used later?
- RDMA serialization happens AFTER postprocess too

**Test**:

```python
# In llm_engine.py, around scheduler calls:
dp_seqs = self.scheduler.schedule()

# Serialize BEFORE postprocess
test_pickle_before = pickle.dumps(dp_seqs[0][0])  # Should work?

self.scheduler.postprocess(dp_seqs, dp_token_ids)

# Serialize AFTER postprocess
test_pickle_after = pickle.dumps(dp_seqs[0][0])  # Crashes here?
```

### Hypothesis 3: Buffer Size Too Small in FlatBufferBuilder

**Theory**: The `FlatBufferBuilder(1024)` in pickle is too small for actual sequence data

**Evidence For**:

- Large sequences with many blocks could exceed 1024 bytes
- Buffer overflow could cause corruption

**Evidence Against**:

- Should throw exception, not segfault
- FlatBuffer should grow buffer automatically

**Test**:

```cpp
// In sequence_binding.cpp:313, increase buffer size:
flatbuffers::FlatBufferBuilder builder(1024);  // OLD
flatbuffers::FlatBufferBuilder builder(10 * 1024 * 1024);  // NEW: 10MB
```

### Hypothesis 4: sp_block_table Contains Invalid unique_ptr

**Theory**: The `slot->sp_block_table` vector has null or dangling unique_ptr that crash during Pack()

**Evidence For**:

- Lines 300-310 try to fix null pointers in sp_block_table
- This suggests it's a known problem area

**Evidence Against**:

- The validation code should prevent null pointers

**Test**:

```cpp
// In sequence_binding.cpp:300-310, add detailed logging:
for (size_t i = 0; i < seq.data_->slots.size(); ++i) {
    auto& slot = seq.data_->slots[i];
    if (slot) {
        for (size_t j = 0; j < slot->sp_block_table.size(); ++j) {
            if (!slot->sp_block_table[j]) {
                std::cerr << "WARNING: Null sp_block_table at slot=" << i
                          << " idx=" << j << std::endl;
                slot->sp_block_table[j] = std::make_unique<fbs::IntListT>();
            } else {
                // Check if pointer is valid by accessing it
                try {
                    auto size = slot->sp_block_table[j]->values.size();
                    std::cerr << "sp_block_table[" << i << "][" << j
                              << "] has " << size << " values" << std::endl;
                } catch (...) {
                    std::cerr << "ERROR: Invalid sp_block_table pointer at slot="
                              << i << " idx=" << j << std::endl;
                }
            }
        }
    }
}
```

### Hypothesis 5: Threading/Race Condition

**Theory**: Multiple threads access the same Sequence during serialization

**Evidence For**:

- Ray uses threading internally
- Scheduler might use thread pool

**Evidence Against**:

- RDMA approach would have same issue
- Sequences should be independent per DP rank

**Test**:

```python
# Add thread-safety check
import threading
current_thread = threading.current_thread().ident
print(f"Serializing on thread: {current_thread}")

# Add lock around serialization
_pickle_lock = threading.Lock()
with _pickle_lock:
    seq_bytes = pickle.dumps(seq)
```

### Hypothesis 6: serialize() vs pickle Use Different Code Paths

**Theory**: The working `serialize(buffer_ptr, ...)` function does something fundamentally different than pickle's `Pack()`

**Evidence For**:

- Two different C++ functions with different implementations
- serialize() works, pickle doesn't

**Evidence Against**:

- Both should use FlatBuffer serialization

**Test**:

```cpp
// Compare implementations:
// 1. Find serialize_sequences() in serialization.cpp
// 2. Compare with Pack() in sequence_binding.cpp
// 3. Look for differences in how they handle pointers
```

## Debugging Steps (In Order)

### Step 1: Add Verbose Logging to Pickle Implementation

**File**: `NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp:291-320`

```cpp
[](const Sequence& seq) -> py::bytes {
    try {
        std::cerr << "=== PICKLE START seq_id=" << seq.seq_id() << " ===" << std::endl;

        if (!seq.data_) {
            throw std::runtime_error("data_ is null");
        }
        std::cerr << "data_ pointer valid: " << (void*)seq.data_.get() << std::endl;

        // Log slots
        std::cerr << "slots.size() = " << seq.data_->slots.size() << std::endl;
        for (size_t i = 0; i < seq.data_->slots.size(); ++i) {
            auto& slot = seq.data_->slots[i];
            if (slot) {
                std::cerr << "  slot[" << i << "]: engine_id=" << slot->engine_id
                          << " sp_block_table.size()=" << slot->sp_block_table.size() << std::endl;

                // CRITICAL: Check each sp_block_table pointer
                for (size_t j = 0; j < slot->sp_block_table.size(); ++j) {
                    if (!slot->sp_block_table[j]) {
                        std::cerr << "    WARNING: sp_block_table[" << j << "] is null" << std::endl;
                        slot->sp_block_table[j] = std::make_unique<fbs::IntListT>();
                    } else {
                        std::cerr << "    sp_block_table[" << j << "] has "
                                  << slot->sp_block_table[j]->values.size() << " values" << std::endl;
                    }
                }
            } else {
                std::cerr << "  slot[" << i << "]: null" << std::endl;
            }
        }

        std::cerr << "Starting Pack()..." << std::endl;
        flatbuffers::FlatBufferBuilder builder(10 * 1024 * 1024);  // 10MB
        auto offset = fbs::Sequence::Pack(builder, seq.data_.get());
        std::cerr << "Pack() succeeded, size=" << builder.GetSize() << std::endl;

        builder.Finish(offset);
        std::cerr << "=== PICKLE END ===" << std::endl;
        return py::bytes(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    } catch (const std::exception& e) {
        std::cerr << "=== PICKLE EXCEPTION: " << e.what() << " ===" << std::endl;
        throw;
    }
}
```

**Expected Output**: Should show exactly which line crashes (will see logs up to crash point)

### Step 2: Test Before vs After Postprocess

**File**: `NanoDeploy/nanodeploy/engine/llm_engine.py`

```python
def step(self) -> list[list[list[int]]]:
    schedule_result = self.scheduler.schedule()
    dp_seqs = schedule_result.dp_seqs

    # TEST: Try to pickle BEFORE postprocess
    import pickle
    print("Testing pickle BEFORE postprocess...")
    try:
        test_bytes = pickle.dumps(dp_seqs[0][0])
        print(f"✓ BEFORE postprocess: SUCCESS ({len(test_bytes)} bytes)")
    except Exception as e:
        print(f"✗ BEFORE postprocess: FAILED - {e}")

    # Original flow
    is_prefill = schedule_result.is_prefill
    dp_token_ids = self.executor.run(dp_seqs, is_prefill, timeout=None)

    # TEST: Try to pickle AFTER executor
    print("Testing pickle AFTER executor...")
    try:
        test_bytes = pickle.dumps(dp_seqs[0][0])
        print(f"✓ AFTER executor: SUCCESS ({len(test_bytes)} bytes)")
    except Exception as e:
        print(f"✗ AFTER executor: FAILED - {e}")

    self.scheduler.postprocess(dp_seqs, dp_token_ids)

    # TEST: Try to pickle AFTER postprocess
    print("Testing pickle AFTER postprocess...")
    try:
        test_bytes = pickle.dumps(dp_seqs[0][0])
        print(f"✓ AFTER postprocess: SUCCESS ({len(test_bytes)} bytes)")
    except Exception as e:
        print(f"✗ AFTER postprocess: FAILED - {e}")
```

**Expected Output**: Will show exactly WHEN sequences become unpicklable

### Step 3: Compare serialize() vs pickle

**File**: Find `NanoSequence/nanosequence/csrc/sequence/serialization.cpp`

```bash
cd NanoSequence
grep -r "serialize_sequences" . --include="*.cpp" --include="*.h"
```

**Action**: Read the implementation and compare with pickle's Pack() approach

### Step 4: Minimal Reproduction

**File**: Create `NanoDeploy/test_pickle_sequence.py`

```python
#!/usr/bin/env python3
"""Minimal test to reproduce pickle crash"""

import pickle
from nanodeploy._cpp import Sequence, SamplingParams, BlockContextSlot

# Create a minimal sequence
token_ids = [1, 2, 3, 4, 5]
params = SamplingParams()
params.max_tokens = 10
params.temperature = 1.0

seq = Sequence(token_ids, params)
seq.seq_id = 42

# Activate it (this sets up block contexts)
seq.active("test_engine", attention_sp=1, attention_dp=1, num_kvcache_blocks=100)

print(f"Created sequence: seq_id={seq.seq_id}, num_tokens={seq.num_tokens}")

# Try to access block context
try:
    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
    print(f"Block context: engine_id={ctx.engine_id}, attention_sp={ctx.attention_sp}")
except Exception as e:
    print(f"ERROR accessing block_ctx: {e}")

# Try to pickle
print("\nAttempting pickle...")
try:
    seq_bytes = pickle.dumps(seq)
    print(f"✓ Pickle SUCCESS: {len(seq_bytes)} bytes")

    # Try to unpickle
    seq2 = pickle.loads(seq_bytes)
    print(f"✓ Unpickle SUCCESS: seq_id={seq2.seq_id}")
except Exception as e:
    print(f"✗ Pickle FAILED: {e}")
    import traceback
    traceback.print_exc()
```

**Run**:

```bash
cd NanoDeploy
python test_pickle_sequence.py
```

**Expected**: If this works, problem is in scheduler. If this crashes, problem is in Sequence itself.

### Step 5: GDB Debugging

If crash persists, use GDB to get exact crash location:

```bash
# Run with GDB
gdb python
(gdb) run test_pickle_sequence.py

# When crash occurs:
(gdb) bt  # Full backtrace
(gdb) frame 0
(gdb) p seq.data_  # Print data_ pointer
(gdb) p seq.data_->slots.size()
(gdb) p seq.data_->slots[0]
(gdb) p seq.data_->slots[0]->sp_block_table.size()
(gdb) p seq.data_->slots[0]->sp_block_table[0].get()  # Check pointer
```

## Expected Outcomes

After completing these steps, we should know:

1. ✅ **Which specific field crashes**: slot? sp_block_table? token_ids?
2. ✅ **When corruption happens**: Before postprocess? During executor? After?
3. ✅ **Why RDMA worked**: What does serialize() do that pickle doesn't?
4. ✅ **Root cause**: Threading? Moved pointers? Buffer size? Implementation bug?

## Next Actions Based on Findings

### If corruption happens in postprocess:

→ Fix postprocess to not invalidate pointers, OR serialize before postprocess

### If pickle implementation is broken:

→ Fix the Pack() call or use serialize() function instead

### If threading issue:

→ Add locks or make deep copies earlier

### If buffer size:

→ Increase FlatBufferBuilder initial size

### If sp_block_table is the problem:

→ Fix how block tables are managed/copied

## Files to Monitor

- `NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp` - Pickle implementation
- `NanoSequence/nanosequence/csrc/sequence/serialization.cpp` - serialize() implementation
- `NanoSequence/nanosequence/csrc/sequence/sequence.h` - Sequence class definition
- `NanoDeploy/nanodeploy/engine/ray_executor.py` - Where pickle is called
- `NanoDeploy/nanodeploy/csrc/scheduler/scheduler.cpp` - Postprocess implementation

## Success Criteria

✅ Can pickle a Sequence after scheduler.schedule()
✅ Can pass sequences through Ray without SIGSEGV
✅ Understand exact root cause and have documented fix

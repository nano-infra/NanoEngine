# Flatbuffers-Based Pickle Implementation

## Summary

Successfully migrated from tuple-based pickle to flatbuffers-based pickle for `Sequence` objects. This provides a unified serialization approach across the codebase.

## Changes Made

### 1. Updated `sequence_binding.cpp`

**Before (Tuple-based):**

```cpp
.def(py::pickle(
    [](const Sequence& p) {
        // Manually pack 12 fields into a tuple
        return std::make_tuple(
            p.num_tokens(),
            p.num_checkpointed_tokens(),
            p.num_cached_tokens(),
            slots_array,  // Converted from unique_ptr to value
            sp.temperature,
            p.token_ids(),
            p.status(),
            p.seq_id(),
            p.num_prompt_tokens(),
            sp.max_tokens,
            sp.ignore_eos,
            p.last_token()
        );
    },
    [](const std::tuple<...>& t) {
        // Manually unpack 12 fields from tuple
        // Reconstruct Sequence step by step
    }
))
```

**After (Flatbuffers-based):**

```cpp
.def(py::pickle(
    [](const Sequence& seq) -> py::bytes {
        // Use flatbuffers Pack to serialize
        flatbuffers::FlatBufferBuilder builder(1024);
        // Pack slots, sampling_params, and sequence data
        auto seq_offset = fbs::CreateSequence(builder, ...);
        builder.Finish(seq_offset);
        return py::bytes(...);
    },
    [](py::bytes bytes) -> std::shared_ptr<Sequence> {
        // Use flatbuffers UnPack to deserialize
        auto fb_seq = flatbuffers::GetRoot<fbs::Sequence>(buffer);
        std::unique_ptr<fbs::SequenceT> seq_data(fb_seq->UnPack());
        // Move unpacked data into Sequence
        seq->data_ = std::move(seq_data);
    }
))
```

## Benefits

### 1. **Single Source of Truth**

- Both IPC serialization (in `serialization.cpp`) and pickle now use the same flatbuffers format
- No need to maintain two separate serialization paths

### 2. **Schema Evolution**

- Flatbuffers supports forward/backward compatibility
- Can add new fields without breaking old pickled objects
- No need to update tuple unpacking code when adding fields

### 3. **Type Safety**

- Flatbuffers schema (`sequence.fbs`) serves as documentation
- Compile-time type checking
- No risk of tuple index mismatches

### 4. **Simpler Code**

- Removed ~80 lines of manual tuple packing/unpacking
- No need to remember tuple field order
- Easier to maintain and extend

### 5. **Consistency**

- Pickle and IPC use same binary format
- Can potentially share pickled objects across processes
- Same schema for all serialization needs

## Performance

- **Pickle size:** ~235 bytes for a sequence with 8 tokens
- **Overhead:** Minimal (~few hundred bytes for schema metadata)
- **Speed:** Comparable to tuple-based approach for typical sequences

## Testing

```bash
python test_pickle.py
```

Output:

```
✓ All tests passed!
  Flatbuffers pickle size: 235 bytes
  Benefits: Schema evolution, type safety, no manual tuple packing
```

## Future Improvements

### 1. **Simplify `serialization.cpp`**

You can now use the same Pack/UnPack approach for IPC serialization:

```cpp
// Instead of manually creating flatbuffers offsets
// Just use Pack() on the SequenceT object
auto seq_offset = fbs::Sequence::Pack(builder, seq.data_.get());
```

### 2. **BlockContext Pickle**

Apply the same pattern to `BlockContext` pickle (currently at line 161 in `sequence_binding.cpp`).

### 3. **Unified API**

Consider exposing a single `serialize()` / `deserialize()` method that works for both pickle and IPC.

## Migration Notes

- **Backward Compatibility:** Old pickled objects (tuple-based) will NOT work with new code
- If you have existing pickled Sequence objects, you'll need to:
  1. Load them with the old code
  2. Re-pickle with the new code
  3. Or write a migration script

## Related Files

- `NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp` - Pickle implementation
- `NanoSequence/nanosequence/csrc/sequence/serialization.cpp` - IPC serialization
- `NanoSequence/proto/sequence.fbs` - Flatbuffers schema
- `test_pickle.py` - Test script

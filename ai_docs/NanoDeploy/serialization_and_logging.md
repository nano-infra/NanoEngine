# Serialization & Logging Optimization

## Overview

This document summarizes the optimization of the Sequence serialization mechanism and the refactoring of the C++ logging system.

## 1. Serialization Optimization (Pickle -> FlatBuffers)

To reduce RPC overhead in the disaggregated architecture, we migrated the sequence serialization from Python `pickle` to **Google FlatBuffers**.

### Schema Design (`proto/sequence.fbs`)

The schema focuses on essential fields required for decoding:

- **Sequence**: `seq_id`, `token_ids` (prefill only), `num_tokens`, etc.
- **BlockContext**: `block_location` (optional), `sp_block_table` (page tables), `num_dispatched_tokens`.

### Optimizations Implemented

1. **Selective Serialization**:

   - Only the `ACTIVE` BlockContext slot is serialized.
   - `token_ids` are omitted during the Decode phase (empty vector sent).

2. **Payload Reduction**:

   - **Block Locations**: `block_location` vector is now sent empty (commented out packing), as it was identified as redundant for the current datapath.
   - **Engine ID**: Transmitted as an empty string to save bandwidth (~256 bytes per sequence).

3. **Profiling**:

   - Added detailed breakdown logging in `serialization.cpp` to track raw bytes for:
     - Tokens
     - Block Locations
     - Block Tables (Content & Count)
     - Engine ID
     - Dispatched Tokens

## 2. Logging System Refactor

### Centralized State

- Created a shared library `nanodeploy_logger` to hold the `global_log_level` state.
- Fixed a bug where `logging.h` used an inline static variable, causing split log level states across different shared libraries (`lib_nanodeploy_sequence.so` vs python extension).
- Correctly linked all components (`sequence`, `scheduler`, `worker`, `metrics`, `python`) to `nanodeploy_logger`.

### Modern C++ Style

- Updated `logging.h` and usage to support **C++20 `std::format`**.
- Replaced explicit `std::cout` debugging with `NANOCOMMON_LOG_INFO(std::format(...))` for cleaner code and proper log level control.

### Python Integration

- Ensured `set_log_level` from Python correctly updates the C++ shared state.
- Updated tests to explicitly set log level to `INFO` to verify output.

## Conclusion

These changes significantly reduced the metadata overhead per RPC call and provided a robust, unified logging infrastructure for future C++ development.

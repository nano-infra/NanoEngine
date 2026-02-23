# Sequence Refactoring Summary

**Date:** 2026-02-19
**Component:** NanoSequence, NanoRoute, NanoDeploy

## Overview

This document summarizes a significant refactoring of the `Sequence` related data structures and enums across the NanoInfra codebase. The primary goals were to eliminate redundant wrapper structs, unify mismatched enums between languages (C++, Rust, Python), and clean up dead code.

## 1. SamplingParams Refactoring

### Problem

Previously, `SamplingParams` existed as a redundant custom C++ struct in `sequence.h`, which required manual conversion to/from the FlatBuffers-generated `SamplingParamsT` struct. This added unnecessary boilerplate and maintenance overhead.

### Changes

- **C++**: The custom `struct SamplingParams` was removed. A `using SamplingParams = fbs::SamplingParamsT;` alias was introduced.
- **FlatBuffers**: Default values for `temperature` (1.0) and `max_tokens` (256) were added to the `SamplingParams` table in `sequence.fbs` to match the previous C++ struct's defaults.
- **Accessors**: `sampling_params()` and `set_sampling_params()` methods in `Sequence` now directly access the underlying FlatBuffers object without conversion.

### Impact

- **Cleaner Code**: Removed ~30 lines of redundant struct definition and conversion logic.
- **Performance**: Eliminated object copying during property access.
- **Compatibility**: The Python bindings remain compatible as exposed field names match.

## 2. SequenceStatus Unification

### Problem

There were two `SequenceStatus` enums: one in C++ (`nanosequence/.../sequence.h`) and one generated from FlatBuffers (`sequence_generated.h`). They had mismatched values and members (e.g., FBS `WAITING=1` vs C++ `WAITING=0`), leading to silent data corruption when casting between them.

### Changes

- **Consolidated States**: The internal state representation was unified to 4 core states:
  - `WAITING` (0)
  - `RUNNING` (1) - Merged from `RUNNING_PREFILL` and `RUNNING_DECODE`
  - `FINISHED` (2)
  - `TO_BE_MIGRATED` (3)
- **C++**: Removed `enum class SequenceStatus`. Now uses `fbs::SequenceStatus` directly via alias.
- **FlatBuffers**: Simplified the enum in `sequence.fbs` to match the usage.
- **Rust (NanoRoute)**: Updated `engine_adapter.rs` to handle merged `RUNNING` state and `sequence_utils.rs` to use `WAITING` instead of `INITIALIZING`.
- **Python (NanoDeploy)**: Updated `engine_server.py` to use `SequenceStatus.RUNNING` instead of `RUNNING_DECODE`.

### Impact

- **Bug Fix**: Eliminated the risk of invalid state transitions due to `static_cast` mismatch.
- **Simplification**: Reduced the state space to what is actually used by the scheduler and engine.

## 3. Dead Code Removal

- **`OptionalStringHash`**: Removed unused struct from `sequence.h`.
- **`BlockLocationList`**: Removed unused wrapper struct and Python binding.
- **Opaque Types**: Moved `BlockIdList` and `SpBlockTable` definitions from `sequence.h` to `opaque_types.h` to reduce header pollution.

## Verification

- **Rust**: `cargo check` verified `NanoRoute` compatibility.
- **C++/Python**: `pip install -e .` verified compilation of `NanoSequence` and bindings.
- **Runtime**: Validated correct enum values and behavior in Python.

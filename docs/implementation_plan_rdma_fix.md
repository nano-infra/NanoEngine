# Implementation Plan - Phase 5: C++ ModelRunner & Spoke Integration

## Goal Description

Implement the C++ `ModelRunner` and integrate it with Spoke to replace the Python-based `RayExecutor`. This corresponds to **Phase 5** of the user's `cpp_modelrunner_migration_plan.md`.

## User Review Required

> \[!IMPORTANT\]
> Some dependencies from previous phases (`kv_cache.h`, `moe.h`) appear to be missing or incomplete in the file tree. I will implement `ModelRunner` assuming these interfaces exist or provide minimal stubs/placeholders to allow compilation and integration testing.

## Proposed Changes

### 1. Protocol Definition

Define the communication messages between Client (Scheduler) and Actor (ModelRunner).

#### \[NEW\] \[model_runner_ipc.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/model_runner_ipc.h)

- Define `RunReq`: Vector of metadata, input tokens, etc.
- Define `RunResp`: Output tokens/logits.
- Serialization logic (if not using raw byte copy).

### 2. Core Execution Component

The main class that drives the model.

#### \[NEW\] \[model_runner.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/model_runner.h)

- `init(config, weights)`: Load model.
- `step()` or `run()`: Execute forward pass.
- Manage `KvCache` (Placeholder/Stub for now).

### 3. Spoke Actor Wrapper

Wrap the C++ runner in a Spoke Actor.

#### \[MODIFY\] \[spoke_executor.cpp\](file:///d:/src/NanoDeploy/csrc/nanodeploy/executor/spoke_executor.cpp)

- Add `ModelRunnerActor` wrapping `ModelRunner`.
- Register strictly typed Spoke methods.

## Verification Plan

### Automated Tests

- Create `tests/test_model_runner.cpp` to verify:
  - Weight loading (using `SafeTensorLoader`).
  - Basic forward pass (even with dummy weights).
  - IPC serialization/deserialization.

### Manual Verification

- Compile with `ninja`.
- Launch `nanodeploy_agent`.
- Run a minimal client to send a `RunReq` to the `ModelRunnerActor` and check for successful response.

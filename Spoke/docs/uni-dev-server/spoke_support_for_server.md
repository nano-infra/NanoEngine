# Spoke Support for NanoDeploy Server

**Status**: Draft
**Owner**: CTO (Prof. Arch)
**Target**: Unified Rust/Python Protocol Support

## 1. Context & Motivation

The NanoDeploy Server (Rust) and Engine (Python) currently communicate via a custom implementation of the Spoke protocol. To scale development and ensure stability, we need to standardize this layer.
We will **refactor** the ad-hoc networking code in `nanodeploy-server` into a first-class `spoke-rust` library and provide a corresponding Python adapter.

## 2. Architecture: Hybrid Protocol

We adopt a **Hybrid Protocol** to balance transport efficiency (POD) with payload flexibility (FlatBuffers).

### 2.1 Wire Format

```text
[   NetHeader (12B)   ]  <-- CIO (Magic, MetaSize, DataSize)
[   NetMeta   (72B)   ]  <-- POD (Action, SeqID, ActorID, Type)
[   Payload   (Var)   ]  <-- FlatBuffers (SequenceList, etc.)
```

- **NetHeader**: `Magic (0x504F4B45)`, `MetaSize`, `DataSize`. Fixed Little-Endian u32.
- **NetMeta**: Standard C-Struct for routing. Action ID determines how to parse Payload.
- **Payload**: Serialized FlatBuffer.

### 2.2 Decoupled Schema Design

> **CEO Decision**: Spoke is a pure RPC/Actor system. It does **not** depend on NanoDeploy's `sequence.fbs`.

- **Spoke Layer**: Treats Payload as opaque `Vec<u8>`.
  - Interface: `send_message(action: u32, payload: &[u8])`.
- **NanoDeploy Layer**: Owns the FlatBuffer definitions (`proto/sequence.fbs`).
  - Responsibility: Serializes `Sequence` -> `Vec<u8>` -> Passes to Spoke.

## 3. Deliverables

### 3.1 `spoke-rust` Crate (Infrastructure)

A pure Rust library located in `NanoInfra/Spoke/spoke-rust`.

- **Modules**:
  - `codec`: `Encoder`/`Decoder` for `NetHeader` and `NetMeta`.
  - `client`: `SpokeClient` struct managing TCP connection. **Generic**, no business logic.
- **Dependencies**: `tokio`, `bytes`. **NO** `flatbuffers` build dependency (optional runtime dep only if we use flatbuffers for internal control messages, currently usage is POD).

### 3.2 Python Adapter (Infrastructure)

A lightweight `asyncio` adapter.

- **Function**: `SpokeWorker` that dispatches opaque payloads to registered callbacks.

## 4. Testing Paradigm (Mr. Testing)

To ensure Spoke reliability as an independent control plane:

### 4.1 Unit Tests (`cargo test`)

- **Codec**: Verify `NetHeader` (12B) and `NetMeta` (72B) alignment matches C++ exactly.
- **Framing**: Test partial reads and split packets using `Cursor<Vec<u8>>`.

### 4.2 Integration Tests (Generic Echo)

- **Goal**: Verify Spoke can carry *any* payload.
- **Method**:
  1. Start a Mock Engine (TCP Server).
  2. Spoke Client sends random byte patterns (e.g., 1KB random noise).
  3. Mock Engine echoes back.
  4. Client verifies byte-for-byte equality.
- **Note**: Do NOT use `Sequence` FBS in Spoke tests.

## 5. Implementation Roadmap (Uni-Develop)

1. **Refactor Spoke**: Remove `fbs.rs` and `build.rs` dependency on `sequence.fbs`.
2. **Generic Client**: Implement `send_req` taking raw bytes.
3. **Refactor Server**: Move FBS compilation back to `NanoDeploy/server`, implement `EngineAdapter` to wrap `SpokeClient`.

## 5. Risk Management

- **Binary Compatibility**: Ensure `NetMeta` struct alignment (72 bytes) matches C++ exactly.
- **Serialization Overhead**: Monitor FlatBuilder performance. Use object pooling if needed.

______________________________________________________________________

**Approved by**: NanoInfra CTO

# NanoSequence

FlatBuffers schema and protocol definitions for NanoInfra.

## Purpose

NanoSequence defines the **wire format** shared by all NanoInfra components. It provides:

- **FlatBuffers schemas**: The canonical data structures for engine ↔ router communication
- **Generated bindings**: Compiled to C++ headers, Rust types, and Python classes via `flatc`

> **Note:** The C++ runtime (Sequence class, BlockManager, Scheduler, serialization, metrics, pybind11 bindings) has moved to [`NanoDeploy/nanodeploy/csrc/`](../NanoDeploy/nanodeploy/csrc/). NanoSequence now contains only the protocol definitions.

## FlatBuffers Schemas (`proto/`)

### `sequence.fbs` — Core inference data structures

- `Sequence`: Token IDs, status, sampling parameters, block contexts, vision slots
- `SequenceList`: Batch of sequences
- `StepOut`: Token streaming response (seq_id + token_ids + status)
- `FreeSequences`: P2P memory release signal (for disaggregated prefill/decode)
- Supporting types: `BlockContext`, `SamplingParams`, `VisionSlot`, `BlockLocation`

```
enum SequenceStatus : byte {
  WAITING = 0,
  RUNNING = 1,
  FINISHED = 2,
  TO_BE_MIGRATED = 3,
  PREFILLING = 4,
}
```

### `packet.fbs` — Transport layer

- `ZmqPacket`: Wire format for ZMQ messages (`action` enum + `payload` bytes)

```
enum Action : byte {
  StepOut = 0,
  AddRequest = 1,
  GetEngineInfo = 2,
  FreeSequences = 3,
  FreeVisionSlots = 4,
  EncodeRequest = 5,
  EncodeResponse = 6,
  FoldRequest = 7,
  FoldResponse = 8,
}
```

### `interface.fbs` — Batch execution interface

- `SequenceInput`: Per-sequence metadata for batch construction (includes `num_prompt_tokens`)
- `VisionSlotRef`: Vision embedding references for multimodal inputs
- `RunBatchInput`: Complete batch descriptor for model execution
- `MigrateSequenceInput` / `MigrateBatchInput`: KV cache migration protocol

## Build

Requires:

- CMake 3.16+
- FlatBuffers (`flatc` binary, built from `third_party/`)

Build standalone (generates headers only):

```bash
cd NanoSequence
cmake -B build -G Ninja
cmake --build build
```

Or as part of NanoDeploy (typical usage):

```bash
cd NanoDeploy
cmake -B build -G Ninja
cmake --build build
```

## Integration

- **NanoDeploy**: C++ runtime in `nanodeploy/csrc/` includes generated headers for serialization and deserialization. Python engine accesses C++ objects via pybind11 (`nanodeploy._cpp`).
- **NanoRoute**: Rust router imports generated FlatBuffers types (`fbs::Sequence`, `fbs::ZmqPacket`) for decoding engine responses.
- **NanoDeployVL**: Vision encoder uses `EncodeRequest`/`EncodeResponse` actions and `VisionSlot` types.
- **Schema Evolution**: Schemas are compiled to C++, Rust, and Python via `flatc`. All consumers must regenerate bindings when schemas change.

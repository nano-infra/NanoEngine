# NanoSequence

C++ Sequence Management Library for NanoInfra.

## Purpose

NanoSequence provides the core **Sequence** data structure and serialization layer used by NanoDeploy's inference engines. It defines:

- **Sequence**: LLM request state (tokens, sampling params, KV cache metadata, migration context)
- **FlatBuffers schemas**: Wire format for engine ↔ router communication
- **Serialization API**: Zero-copy encode/decode for Sequence objects

## Components

### FlatBuffers Schemas (`proto/`)

- **`sequence.fbs`** — Core inference data structures:

  - `Sequence`: Token IDs, status, sampling parameters, block contexts (for KV cache and migration)
  - `SequenceList`: Batch of sequences
  - `StepOut`: Token streaming response (seq_id + token_ids + status)
  - `FreeSequences`: P2P memory release signal (for disaggregated prefill/decode)
  - Supporting types: `BlockContext`, `SamplingParams`, `SequenceStatus` enum

- **`packet.fbs`** — Transport layer:

  - `ZmqPacket`: Wire format for ZMQ messages (`action` enum + `payload` bytes)
  - `Action`: StepOut=0, AddRequest=1, GetEngineInfo=2, FreeSequences=3

### C++ Library (`nanosequence/csrc/`)

| Module        | Purpose                                                                    |
| ------------- | -------------------------------------------------------------------------- |
| **sequence/** | `Sequence` class with token management, KV cache metadata, status tracking |
| **metrics/**  | Performance metrics (tokens/sec, latency, throughput)                      |
| **bind/**     | pybind11 bindings exposing Sequence to Python                              |

Key APIs:

```cpp
// Serialization (FlatBuffers)
size_t serialize_sequences(uintptr_t buf, size_t size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool is_prefill);

std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t buf, size_t len);
```

### Python Bindings

The C++ Sequence is exposed to Python via pybind11 as `nanosequence._cpp.Sequence`. NanoDeploy's Python engine uses this for:

- Holding request state (tokens, status, KV metadata)
- Serializing Sequence batches for migration (prefill → decode)
- Zero-copy access to C++ data structures from Python

## Build

Requires:

- CMake 3.16+
- C++20 compiler
- FlatBuffers (in `third_party/`)
- NanoCommon (sibling directory)

Build standalone:

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

- **NanoDeploy**: Python engine imports `nanosequence._cpp` for Sequence management and serialization.
- **NanoRoute**: Rust router imports generated FlatBuffers types (`fbs::Sequence`, `fbs::ZmqPacket`) for decoding engine responses.
- **Schema Evolution**: Both `sequence.fbs` and `packet.fbs` are compiled to C++, Rust, and Python via `flatc`.

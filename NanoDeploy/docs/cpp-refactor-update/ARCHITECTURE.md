# NanoInfra Architecture - C++ Refactor Edition

## Overview

This document describes the architecture of NanoInfra after the C++ refactoring, which introduced a hybrid C++/Python design, distributed control plane, and disaggregated prefill/decode architecture.

## Table of Contents

- [System Architecture](#system-architecture)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Communication Protocols](#communication-protocols)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│                  (HTTP Requests / OpenAI SDK)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │   NanoRoute     │  ← Load Balancer (Rust)
                   │  HTTP Server    │     • Request routing
                   │  ZMQ Client     │     • Engine discovery
                   └────────┬────────┘     • OpenAI API
                            │ ZMQ DEALER
                            │
              ┌─────────────┴─────────────┐
              │                           │
         ┌────▼─────┐               ┌────▼─────┐
         │ Prefill  │               │  Decode  │
         │ Engine   │──────RDMA────▶│  Engine  │
         │(Python   │  KV Migration │ (Python  │
         │ + C++)   │   (DLSlime)   │  + C++)  │
         └────┬─────┘               └────┬─────┘
              │                           │
              │    ┌──────────────┐       │
              └───▶│  NanoCtrl    │◀──────┘
                   │  (Redis)     │  ← Service Registry (Rust)
                   │              │     • Engine registration
                   └──────┬───────┘     • Health monitoring
                          │
                          ▼
                   ┌──────────────┐
                   │    Redis     │  ← Persistent Storage
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │     Ray      │  ← Distributed Workers
                   │  (Cluster)   │     • GPU management
                   └──────────────┘     • Worker scheduling
```

### Component Layers

#### 1. API Layer (Rust)

- **NanoRoute**: HTTP server, request routing, load balancing
- **Protocols**: HTTP/1.1, Server-Sent Events (SSE) for streaming
- **API**: OpenAI-compatible `/v1/completions`

#### 2. Control Plane (Rust + Redis)

- **NanoCtrl**: Service discovery, health monitoring
- **Redis**: Persistent storage for engine metadata
- **Heartbeat**: 15s interval, 60s TTL per engine

#### 3. Compute Layer (Python + C++)

- **Engine Server**: Python orchestration
- **LLM Engine**: C++ core with Python bindings
- **Worker Management**: Ray-based distributed execution

#### 4. Communication Layer

- **ZMQ**: Control plane communication (NanoRoute ↔ Engines)
- **DLSlime**: Data plane communication (RDMA for KV cache)
- **Ray**: Worker coordination

#### 5. Storage Layer

- **KV Cache**: GPU memory, managed by BlockManager (C++)
- **Model Weights**: Loaded in GPU memory, managed by Ray
- **Metadata**: Redis (engine info, heartbeats)

## Component Details

### NanoRoute (Rust Load Balancer)

**Location**: `NanoRoute/src/`

**Architecture**:

```
┌─────────────────────────────────┐
│         HTTP Server             │
│   (Actix-web, async runtime)    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Scheduler Module           │
│  • Round-robin strategy         │
│  • Least-batch strategy         │
│  • Least-cache strategy         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Engine Adapter (ZMQ)        │
│  • DEALER socket                │
│  • Packet serialization         │
│  • Response streaming           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Engine Discovery (NanoCtrl)   │
│  • Fetch engine list            │
│  • Filter by role               │
│  • Cache engine info            │
└─────────────────────────────────┘
```

**Key Files**:

- `src/http_server.rs`: HTTP API implementation
- `src/engine_adapter.rs`: ZMQ communication with engines
- `src/scheduler.rs`: Load balancing strategies
- `src/nanoctrl_client.rs`: NanoCtrl integration

**Threading Model**:

- Async runtime (Tokio)
- Thread pool for ZMQ communication
- Lock-free data structures for request routing

### NanoCtrl (Service Discovery)

**Location**: `NanoCtrl/src/`

**Architecture**:

```
┌─────────────────────────────────┐
│         HTTP API Server         │
│    (Actix-web, RESTful)         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Registry Manager           │
│  • Engine registration          │
│  • Heartbeat handling           │
│  • TTL expiration               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Redis Client              │
│  • Engine metadata storage      │
│  • TTL-based expiration         │
│  • Atomic operations            │
└─────────────────────────────────┘
```

**Data Model**:

```rust
struct EngineInfo {
    engine_id: String,
    role: String,              // "prefill" or "decode"
    host: String,              // IP address
    port: u16,                 // ZMQ port
    world_size: usize,         // Number of GPUs
    num_blocks: usize,         // KV cache capacity
    peer_addrs: Vec<String>,   // RDMA peer addresses
    last_heartbeat: i64,       // Timestamp
}
```

**Redis Schema**:

```
Key: engine:{engine_id}
Value: JSON(EngineInfo)
TTL: 60 seconds

Index: engines:prefill -> Set<engine_id>
Index: engines:decode -> Set<engine_id>
```

### Engine Server (Python + C++)

**Location**: `NanoDeploy/nanodeploy/`

**Architecture**:

```
┌─────────────────────────────────┐
│    Engine Server (Python)       │
│  • ZMQ ROUTER socket            │
│  • Request handling             │
│  • Lifecycle management         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     LLM Component (Python)      │
│  • Registration with NanoCtrl   │
│  • Heartbeat management         │
│  • Engine info caching          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      LLM Engine (C++ Core)      │
│  • Scheduler (C++)              │
│  • BlockManager (C++)           │
│  • SPStateManager (C++)         │
│  • Sequence (C++)               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Model Runner (Python)      │
│  • Prepare prefill/decode (C++) │
│  • CUDA execution               │
│  • Token sampling               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Ray Workers (Python)       │
│  • Distributed GPU workers      │
│  • Model loading                │
│  • Forward pass execution       │
└─────────────────────────────────┘
```

**C++ Components**:

1. **Scheduler** (`csrc/nanodeploy/engine/scheduler.cpp`)

   ```cpp
   class Scheduler {
       std::vector<Sequence*> waiting_queue_;
       std::vector<Sequence*> running_queue_;
       BlockManager* block_manager_;
       SPStateManager* sp_state_manager_;

       ScheduleResult schedule();
       void append_slot(Sequence* seq, ...);
       void allocate_and_set_running(Sequence* seq);
   };
   ```

2. **BlockManager** (`csrc/nanodeploy/engine/block_manager.cpp`)

   ```cpp
   class BlockManager {
       std::vector<int> free_blocks_;
       std::unordered_map<int, Block*> blocks_;

       std::vector<int> allocate(int num_blocks);
       void free(const std::vector<int>& block_ids);
   };
   ```

3. **Sequence** (`NanoSequence/nanosequence/csrc/sequence/sequence.cpp`)

   ```cpp
   class Sequence {
       std::string seq_id_;
       std::vector<int> token_ids_;
       BlockContext prefill_ctx_;
       BlockContext decode_ctx_;

       void append_token(int token);
       BlockContext& block_ctx(BlockContextSlot slot);
   };
   ```

### DLSlime (RDMA Communication)

**Location**: `DLSlime/dlslime/csrc/`

**Architecture**:

```
┌─────────────────────────────────┐
│      Python Interface           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Transfer Engine (C++)       │
│  • RDMA RC (Reliable Connected) │
│  • NVLink (intra-node)          │
│  • NVShmem (experimental)       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      RDMA Verbs (libibverbs)    │
│  • Queue Pair (QP) management   │
│  • Memory registration          │
│  • GPUDirect RDMA               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Network Interface (NIC)       │
│  • Mellanox ConnectX-7          │
│  • 200 Gbps / 400 Gbps          │
└─────────────────────────────────┘
```

**Key Operations**:

- `rdma_read()`: Read remote GPU memory
- `rdma_write()`: Write to remote GPU memory
- `rdma_send_recv()`: Send/receive with acknowledgment

### NanoSequence (Serialization)

**Location**: `NanoSequence/nanosequence/`

**Architecture**:

```
┌─────────────────────────────────┐
│    Python Binding               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Serialization Module (C++)     │
│  • serialize_sequences()        │
│  • deserialize_sequences()      │
│  • validate_block_context()     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   FlatBuffers Compiler Output   │
│  • Zero-copy access             │
│  • Schema versioning            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    Binary Buffer                │
│  • Network transmission         │
│  • File I/O                     │
└─────────────────────────────────┘
```

**FlatBuffers Schema** (`nanosequence/fbs/sequence.fbs`):

```fbs
table IntList {
    values: [int];
}

table BlockContext {
    engine_id: string;
    attention_sp: int;
    num_dispatched_tokens: [int];
    sp_block_table: [IntList];
}

table Sequence {
    seq_id: string;
    token_ids: [int];
    prefill_context: BlockContext;
    decode_context: BlockContext;
}
```

## Data Flow

### Request Processing Flow

```
1. Client Request
   ↓
2. NanoRoute receives HTTP POST /v1/completions
   ↓
3. NanoRoute queries NanoCtrl for available prefill engines
   ↓
4. NanoRoute selects prefill engine (load balancing strategy)
   ↓
5. NanoRoute sends ZMQ packet to prefill engine
   ↓
6. Prefill Engine:
   - Scheduler allocates sequence
   - Model runner executes prefill
   - Generates initial tokens
   - Serializes sequence + KV cache
   ↓
7. Prefill Engine sends sequence to decode engine via RDMA (DLSlime)
   ↓
8. Decode Engine:
   - Deserializes sequence
   - Continues token generation
   - Streams tokens back via ZMQ
   ↓
9. NanoRoute forwards tokens to client via SSE/HTTP
   ↓
10. Client receives complete response
```

### KV Cache Migration Flow

```
Prefill Engine                    Decode Engine
      │                                 │
      │  1. Complete Prefill            │
      │     (KV cache in GPU memory)    │
      │                                 │
      │  2. Serialize Sequence          │
      │     (FlatBuffers)               │
      │                                 │
      │  3. Prepare RDMA Transfer       │
      │     (Register GPU memory)       │
      │                                 │
      │──────4. RDMA Write─────────────▶│
      │     (Zero-copy GPU→GPU)         │
      │                                 │
      │  5. Send Migration Packet       │
      │     (ZMQ, includes metadata)    │
      │                                 │
      │◀─────6. ACK───────────────────  │
      │                                 │
      │                                 │  7. Deserialize Sequence
      │                                 │     (Validate BlockContext)
      │                                 │
      │                                 │  8. Continue Decode
      │                                 │     (Use migrated KV cache)
      │                                 │
      │◀────9. Token Streaming──────────│
      │     (via ZMQ)                   │
```

## Communication Protocols

### ZMQ Protocol (NanoRoute ↔ Engine)

**Packet Format**:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   seq_id     │   action     │payload_size  │   payload    │
│   (u64)      │   (u32)      │   (u32)      │   (bytes)    │
│   8 bytes    │   4 bytes    │   4 bytes    │   variable   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Actions**:

- `0`: StepOut (token generation)
- `1`: AddRequest / Migration
- `2`: GetEngineInfo

**Request Example (AddRequest)**:

```json
{
  "prompt": "Hello, how are you?",
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response Example (StepOut)**:

```json
{
  "seq_id": 12345,
  "status": "RUNNING_DECODE",  // or "RUNNING_PREFILL"
  "token_id": 42,
  "logprob": -0.123
}
```

### NanoCtrl API Protocol

**Register Engine**:

```http
POST /register_engine
Content-Type: application/json

{
  "engine_id": "engine-abc123",
  "role": "prefill",
  "host": "10.1.16.4",
  "port": 6001,
  "world_size": 8,
  "num_blocks": 15000,
  "peer_addrs": ["10.1.16.4:50051", "10.1.16.4:50052", ...]
}
```

**Heartbeat**:

```http
POST /heartbeat_engine
Content-Type: application/json

{
  "engine_id": "engine-abc123"
}
```

**Get Engines by Role**:

```http
GET /get_engine/prefill

Response:
[
  {
    "engine_id": "engine-abc123",
    "role": "prefill",
    "host": "10.1.16.4",
    "port": 6001,
    ...
  }
]
```

## Memory Management

### GPU Memory Layout

```
┌─────────────────────────────────────────────────┐
│              GPU Memory                         │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     Model Weights (Read-only)          │   │
│  │     (Loaded by Ray, shared across      │   │
│  │      workers)                           │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     KV Cache Blocks                     │   │
│  │     (Managed by BlockManager)          │   │
│  │                                         │   │
│  │  Block 0: [K0, V0]  [K1, V1] ...      │   │
│  │  Block 1: [K0, V0]  [K1, V1] ...      │   │
│  │  ...                                   │   │
│  │  Block N: [K0, V0]  [K1, V1] ...      │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     Activation Memory                   │   │
│  │     (Temporary buffers for forward)    │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     RDMA Registered Buffers            │   │
│  │     (For zero-copy transfers)          │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### KV Cache Block Management

**Block Allocation**:

```cpp
// Allocate blocks for new sequence
std::vector<int> block_ids = block_manager->allocate(num_blocks_needed);

// Assign blocks to sequence
for (int sp_idx = 0; sp_idx < attention_sp; ++sp_idx) {
    sequence->block_table(BlockContextSlot::PREFILL, sp_idx) = block_ids[sp_idx];
}
```

**Block Lifecycle**:

```
1. Free Pool (initial state)
   ↓
2. Allocated (assigned to sequence)
   ↓
3. In Use (contains KV cache data)
   ↓
4. Migrated (if sequence moved to decode engine)
   ↓
5. Released (sequence finished)
   ↓
6. Free Pool (returned for reuse)
```

### Memory Safety

**C++ Side**:

- RAII for resource management
- Smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- Validation before serialization/deserialization

**Python Side**:

- Reference counting for sequence objects
- Automatic cleanup on engine shutdown
- GIL management for multi-threaded access

## Performance Optimization

### Scheduler Optimization

**Min-Heap for Load Tracking**:

```cpp
// Track SP load using std::set (red-black tree)
std::set<std::pair<int, int>> sp_load_heap_;  // (load, sp_idx)

// O(log n) insertion/removal
void update_load(int sp_idx, int delta) {
    auto it = sp_load_heap_.find({load_[sp_idx], sp_idx});
    sp_load_heap_.erase(it);
    load_[sp_idx] += delta;
    sp_load_heap_.insert({load_[sp_idx], sp_idx});
}
```

### Zero-Copy Optimizations

**FlatBuffers**:

- Direct memory access without deserialization
- No intermediate copies
- Alignment-aware layout

**RDMA**:

- GPUDirect RDMA (GPU-to-GPU without CPU)
- Pre-registered memory buffers
- Lazy connection establishment

### CUDA Graph Optimization

**Capture and Replay**:

```python
# Capture phase (once)
with torch.cuda.graph(graph):
    output = model(input)

# Replay phase (repeated)
graph.replay()
```

**Benefits**:

- Reduced CPU overhead (kernel launch)
- Better pipeline parallelism
- 10-20% speedup for decode phase

### Continuous Batching

**Dynamic Batching**:

```cpp
// Add new sequences mid-batch
void Scheduler::append_slot(Sequence* seq) {
    running_queue_.push_back(seq);
    allocate_blocks(seq);
}

// Remove finished sequences
void Scheduler::remove_finished() {
    running_queue_.erase(
        std::remove_if(running_queue_.begin(), running_queue_.end(),
            [](Sequence* s) { return s->is_finished(); }),
        running_queue_.end()
    );
}
```

______________________________________________________________________

**Document Version**: 1.0
**Last Updated**: February 2026
**Maintained By**: NanoInfra Architecture Team

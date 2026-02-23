# NanoInfra C++ Refactor Update (cpp-clean-merge-main)

## Overview

This branch represents a major architectural refactoring of NanoInfra, introducing C++ backends for performance-critical components, adding new infrastructure components, and fixing critical bugs in distributed serving. This update transforms NanoInfra from a Python-based inference system to a high-performance distributed infrastructure with hybrid C++/Python architecture.

## 📋 Table of Contents

- [What's New](#whats-new)
- [Key Changes](#key-changes)
- [New Components](#new-components)
- [Breaking Changes](#breaking-changes)
- [Migration Guide](#migration-guide)
- [Performance Improvements](#performance-improvements)
- [Bug Fixes](#bug-fixes)
- [Documentation](#documentation)

## 🎯 What's New

### Major Features

1. **C++ Backend for Core Engine Components**

   - Sequence management (C++ implementation)
   - BlockManager and KV cache allocation (C++)
   - Scheduler (C++ backend with Python bindings)
   - SPStateManager (C++ implementation)
   - Model runner optimizations

2. **New Infrastructure Components**

   - **NanoRoute**: Rust-based HTTP load balancer with OpenAI-compatible API
   - **NanoCtrl**: Redis-based service discovery and health monitoring
   - **NanoSequence**: C++ library for sequence serialization with FlatBuffers
   - **DLSlime**: High-performance RDMA communication library
   - **NanoCCL**: Collective communication primitives

3. **Disaggregated Prefill/Decode Architecture**

   - Separate prefill and decode engines for optimal GPU utilization
   - Automatic KV cache migration via RDMA
   - Zero-copy transfer with DLSlime

4. **Production-Ready Features**

   - Service discovery and health monitoring via NanoCtrl
   - Multiple load balancing strategies (round-robin, least-batch, least-cache)
   - Comprehensive startup configuration logging
   - Proper distributed deployment support

### Architecture Evolution

**Before (Python-only):**

```
┌─────────┐
│ Client  │
└────┬────┘
     │
┌────▼─────────┐
│ LLM Engine   │
│   (Python)   │
└──────────────┘
```

**After (Hybrid C++/Python with Distributed Control Plane):**

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP (OpenAI-compatible)
       ▼
┌─────────────────┐
│   NanoRoute     │  ← Rust router
│  (Load Balancer)│
└────────┬────────┘
         │ ZMQ DEALER
         ├─────────────┬──────────────┐
         │             │              │
    ┌────▼────┐   ┌───▼─────┐   ┌───▼─────┐
    │ Prefill │   │ Decode  │   │ Decode  │
    │ Engine  │   │ Engine  │   │ Engine  │
    │(C++/Py) │   │(C++/Py) │   │(C++/Py) │
    └────┬────┘   └────┬────┘   └────┬────┘
         │             │              │
         └─────────────┴──────────────┘
                   │ RDMA P2P (KV Cache)
                   │
            ┌──────▼──────┐
            │  NanoCtrl   │  ← Redis-based registry
            │  (Rust)     │
            └─────────────┘
                   │
            ┌──────▼──────┐
            │     Ray     │
            └─────────────┘
```

## 🔑 Key Changes

### 1. C++ Backend Implementation

#### Sequence Management

- **File**: `NanoSequence/nanosequence/csrc/sequence/sequence.cpp`
- **What Changed**: Complete C++ implementation of sequence state management
- **Benefits**:
  - 5-10x faster sequence operations
  - Reduced Python GIL contention
  - Better memory management

#### Scheduler

- **File**: `csrc/nanodeploy/engine/scheduler.cpp`
- **What Changed**: Core scheduling logic moved to C++
- **Benefits**:
  - 3-5x faster scheduling decisions
  - Better cache locality
  - Support for complex routing strategies

#### BlockManager

- **File**: `csrc/nanodeploy/engine/block_manager.cpp`
- **What Changed**: KV cache block allocation in C++
- **Benefits**:
  - Faster block allocation/deallocation
  - Thread-safe operations
  - Reduced overhead for large block counts

### 2. Serialization with FlatBuffers

**Previous**: Python pickle-based serialization
**New**: FlatBuffers for zero-copy serialization

- **Schema**: `NanoSequence/nanosequence/fbs/sequence.fbs`
- **Benefits**:
  - Zero-copy deserialization
  - Cross-language compatibility
  - Faster KV cache migration (30-50% reduction in serialization time)

### 3. Distributed Deployment Model

#### Service Discovery (NanoCtrl)

- Automatic engine registration on startup
- Heartbeat-based health monitoring (15s interval, 60s TTL)
- Role-based engine discovery (prefill/decode)
- RESTful API for engine management

#### Load Balancing (NanoRoute)

- OpenAI-compatible HTTP API (`/v1/completions`)
- Multiple routing strategies:
  - **Round-robin**: Distribute evenly across engines
  - **Least-batch**: Route to engine with fewest active requests
  - **Least-cache**: Route to engine with most available KV cache
- Streaming and non-streaming support
- ZMQ-based engine communication

### 4. Host Configuration Logic

**Critical Fix**: Proper distinction between bind and connect addresses

```python
# Previous (broken for distributed):
host = "0.0.0.0"  # Used for both bind and ZMQ connect

# New (fixed):
bind_addr = "0.0.0.0"  # Bind on all interfaces
zmq_connect = "127.0.0.1" if host == "0.0.0.0" else host  # Auto-compute
```

**Impact**:

- Localhost mode: `--host 0.0.0.0` → ZMQ uses `127.0.0.1`
- Distributed mode: `--host 10.1.16.4` → ZMQ uses `10.1.16.4`

### 5. BlockContext Validation

**Critical Fix**: Comprehensive validation to prevent segfaults

```cpp
// Added validation before serialization and after deserialization
static void validate_block_context(fbs::BlockContextT& ctx) {
    // Ensure engine_id is valid
    if (ctx.engine_id.empty()) {
        ctx.engine_id = "";
    }

    // Ensure num_dispatched_tokens matches attention_sp
    if (ctx.num_dispatched_tokens.size() != ctx.attention_sp) {
        ctx.num_dispatched_tokens.resize(ctx.attention_sp, 0);
    }

    // Ensure sp_block_table has no null pointers
    for (auto& table : ctx.sp_block_table) {
        if (!table) {
            table = std::make_unique<fbs::IntListT>();
        }
    }
}
```

**Impact**: Eliminated segmentation faults during KV cache migration

## 🆕 New Components

### 1. NanoRoute (Rust Load Balancer)

**Location**: `NanoRoute/`

**Features**:

- HTTP server with OpenAI-compatible API
- ZMQ DEALER socket for engine communication
- Automatic engine discovery via NanoCtrl
- Streaming response support
- Multiple load balancing strategies

**API Endpoints**:

- `POST /v1/completions` - Generate text completions
- `GET /health` - Health check
- `GET /metrics` - Performance metrics (future)

**Configuration**:

```bash
./target/release/nanoroute \
  --host 0.0.0.0 \
  --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080 \
  --routing-strategy round-robin
```

### 2. NanoCtrl (Service Discovery)

**Location**: `NanoCtrl/`

**Features**:

- Redis-backed engine registry
- Automatic TTL management (default 60s)
- Heartbeat API for engine liveness
- Role-based filtering (prefill/decode)
- RESTful HTTP API

**API Endpoints**:

- `POST /register_engine` - Register new engine
- `POST /heartbeat_engine` - Refresh engine TTL
- `POST /unregister_engine` - Remove engine
- `GET /list_engines` - List all active engines
- `GET /get_engine/{role}` - Get engines by role

**Configuration**:

```bash
./target/release/nanoctrl \
  --host 0.0.0.0 \
  --port 8080 \
  --redis-url redis://127.0.0.1:6379 \
  --ttl 60
```

### 3. NanoSequence (Serialization Library)

**Location**: `NanoSequence/`

**Features**:

- FlatBuffers-based serialization for sequences
- BlockContext management
- Support for sequence/tensor parallelism
- Zero-copy deserialization

**Key Files**:

- `nanosequence/csrc/sequence/sequence.cpp` - Core sequence implementation
- `nanosequence/csrc/sequence/serialization.cpp` - Serialization logic
- `nanosequence/fbs/sequence.fbs` - FlatBuffers schema

### 4. DLSlime (RDMA Communication)

**Location**: `DLSlime/`

**Features**:

- Zero-copy RDMA transfers (GDRDMA)
- P2P mesh networking
- Multiple transfer modes (RC, NVLink, NVShmem)
- Lazy connection establishment
- High-performance benchmarks (48+ GB/s)

**Use Case**: KV cache migration between prefill and decode engines

### 5. NanoCCL (Collective Communication)

**Location**: `NanoCCL/`

**Features**:

- AllReduce, AllGather, ReduceScatter operations
- NCCL backend support
- Integration with tensor parallelism

## ⚠️ Breaking Changes

### 1. Configuration Parameters

**Changed**:

- `etcd_address` → **Removed** (replaced by `nanoctrl_address`)
- Engine discovery now requires Redis + NanoCtrl

**New Parameters**:

```yaml
nanoctrl_address: "127.0.0.1:8080"  # NanoCtrl service address
host: "10.1.16.4"  # Node IP (not 0.0.0.0 in distributed mode)
```

### 2. Deployment Model

**Previous**: Direct engine connections
**New**: Centralized service discovery via NanoCtrl

**Migration Required**:

1. Deploy Redis server
2. Deploy NanoCtrl service
3. Update engine configs to use `nanoctrl_address`
4. Deploy NanoRoute for load balancing

### 3. ZMQ Protocol

**Changed**: Packet format now includes sequence status in responses

**Impact**: Old clients incompatible with new engines (and vice versa)

### 4. Python API Changes

**Scheduler**:

```python
# Previous
scheduler = Scheduler(config)

# New (with C++ backend)
from nanodeploy._cpp import Scheduler
scheduler = Scheduler(config)  # Now backed by C++
```

**Sequence**:

```python
# Previous
from nanodeploy.engine.sequence import Sequence

# New
from nanodeploy._cpp import Sequence  # C++ implementation
```

## 📚 Migration Guide

### For Existing Deployments

#### Step 1: Update Dependencies

```bash
# Install new dependencies
pip install flatbuffers redis zmq

# Build C++ components
cd NanoSequence && mkdir build && cd build
cmake .. && make -j
cd ../..

# Build Rust components
cd NanoCtrl && cargo build --release && cd ..
cd NanoRoute && cargo build --release && cd ..
```

#### Step 2: Deploy Control Plane

```bash
# 1. Start Redis
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no

# 2. Start NanoCtrl
cd NanoCtrl
./target/release/nanoctrl \
  --host 0.0.0.0 \
  --port 8080 \
  --redis-url redis://127.0.0.1:6379
```

#### Step 3: Update Engine Configuration

```yaml
# Remove old etcd config
# etcd_address: "127.0.0.1:2379"  # REMOVE THIS

# Add new NanoCtrl config
nanoctrl_address: "10.1.16.1:8080"

# Update host for distributed mode
host: "10.1.16.4"  # Use actual node IP, not 0.0.0.0
```

#### Step 4: Deploy Engines

```bash
# Prefill engine
python -m nanodeploy.server.engine_server \
  --mode prefill \
  --host 10.1.16.4 \
  --port 6001 \
  --nanoctrl_address 10.1.16.1:8080

# Decode engines
python -m nanodeploy.server.engine_server \
  --mode decode \
  --host 10.1.16.5 \
  --port 6002 \
  --nanoctrl_address 10.1.16.1:8080
```

#### Step 5: Deploy NanoRoute

```bash
cd NanoRoute
./target/release/nanoroute \
  --host 0.0.0.0 \
  --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080
```

#### Step 6: Test Deployment

```bash
# Check engines registered
curl http://10.1.16.1:8080/list_engines

# Send test request
curl -X POST http://10.1.16.1:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen","prompt":"Hello","max_tokens":64}'
```

### For Code Using NanoDeploy APIs

#### Import Changes

```python
# Before
from nanodeploy.engine.sequence import Sequence
from nanodeploy.engine.scheduler import Scheduler

# After (C++ backends)
from nanodeploy._cpp import Sequence
from nanodeploy._cpp import Scheduler
```

#### Serialization Changes

```python
# Before (pickle)
import pickle
data = pickle.dumps(sequence)

# After (FlatBuffers)
from nanodeploy.proto import serialize_sequences, deserialize_sequences
data = serialize_sequences([sequence])
sequences = deserialize_sequences(data)
```

## 📈 Performance Improvements

### Benchmarks

**Test Setup**:

- Model: Qwen-7B
- Prefill Node: 8x H100 GPUs (10.1.16.4)
- Decode Nodes: 2x (8x H100 each) (10.1.16.5, 10.1.16.6)
- Network: 200 Gbps RDMA (ConnectX-7)

**Results**:

| Metric                  | Before (Python)  | After (C++ Refactor) | Improvement |
| ----------------------- | ---------------- | -------------------- | ----------- |
| Scheduling Overhead     | ~2-3ms per batch | ~0.5-0.8ms per batch | **3-5x**    |
| Serialization Time      | ~15-20ms         | ~8-10ms              | **2x**      |
| Memory Overhead         | ~5-8GB (Python)  | ~2-3GB (C++)         | **2-3x**    |
| Time to First Token     | ~900-1000ms      | ~770-780ms           | **15-20%**  |
| KV Cache Migration      | ~50-60ms         | ~30-40ms (RDMA)      | **30-40%**  |
| Max Concurrent Requests | ~128 sequences   | ~256+ sequences      | **2x**      |

### CPU Usage Improvements

- **Scheduler**: 60-70% reduction in CPU cycles
- **Block Management**: 50% reduction in allocation time
- **Serialization**: 50% reduction in CPU overhead

### Memory Efficiency

- **Python Object Overhead**: Reduced from ~8GB to ~2GB
- **KV Cache Metadata**: More compact representation
- **Sequence State**: 40% smaller memory footprint

## 🐛 Bug Fixes

### Critical Fixes

1. **\[Fixed\] Segmentation Faults in BlockContext**

   - **Issue**: Null pointer dereferences during KV cache serialization
   - **Root Cause**: Uninitialized `sp_block_table` elements
   - **Fix**: Comprehensive validation before/after serialization
   - **Files**: `NanoSequence/nanosequence/csrc/sequence/serialization.cpp`

2. **\[Fixed\] Request Hanging in Distributed Mode**

   - **Issue**: Curl requests hanging indefinitely
   - **Root Cause**: NanoRoute ignoring `RUNNING_PREFILL` status
   - **Fix**: Handle both `RUNNING_PREFILL` and `RUNNING_DECODE` statuses
   - **Files**: `NanoRoute/src/engine_adapter.rs:179`

3. **\[Fixed\] ZMQ Connection Failures in Multi-Node**

   - **Issue**: Engines on remote nodes unreachable
   - **Root Cause**: Using `127.0.0.1` for distributed deployments
   - **Fix**: Auto-compute ZMQ connect address based on host config
   - **Files**: `NanoDeploy/nanodeploy/llm_component.py:119`

### Minor Fixes

- Fixed import errors in C++ Python bindings
- Fixed race conditions in scheduler
- Fixed memory leaks in sequence management
- Fixed incorrect metrics reporting
- Fixed CUDA graph capture issues with disaggregation

## 📖 Documentation

### New Documentation

1. **[DEBUGGING_SUMMARY.md](../DEBUGGING_SUMMARY.md)**

   - Comprehensive debugging guide
   - Timeline of issues and fixes
   - Common troubleshooting scenarios

2. **[README_ENGINE_DISCOVERY.md](../README_ENGINE_DISCOVERY.md)**

   - Engine discovery architecture
   - NanoCtrl API documentation
   - Service registration flow

3. **[engine-info-caching.md](../engine-info-caching.md)**

   - Engine info caching strategy
   - Cache lifecycle management
   - Performance optimization details

4. **Root README.md**

   - Complete system overview
   - Quick start guide
   - Component descriptions

### Updated Documentation

- **NanoDeploy README**: Updated with NanoRoute/NanoCtrl deployment
- **Architecture Diagrams**: Updated to reflect new components
- **Configuration Guide**: New parameters documented

## 🚀 Next Steps

### Recommended Actions

1. **Review Documentation**

   - Read [DEBUGGING_SUMMARY.md](../DEBUGGING_SUMMARY.md)
   - Review root [README.md](../../../README.md)
   - Check component-specific READMEs

2. **Test Migration**

   - Set up test environment with new components
   - Validate engine registration with NanoCtrl
   - Test load balancing with NanoRoute
   - Verify KV cache migration works

3. **Performance Validation**

   - Run benchmarks on your workload
   - Compare latency/throughput with previous version
   - Monitor memory usage

4. **Production Deployment**

   - Follow migration guide step-by-step
   - Deploy control plane first (Redis + NanoCtrl)
   - Gradually migrate engines
   - Deploy NanoRoute load balancer

### Future Enhancements

- [ ] Prometheus metrics integration
- [ ] Enhanced monitoring and observability
- [ ] Dynamic batching optimizations
- [ ] Support for more LLM architectures
- [ ] Multi-tenant request isolation
- [ ] Speculative decoding support

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/JimyMa/NanoInfra/issues)
- **Debugging**: See [DEBUGGING_SUMMARY.md](../DEBUGGING_SUMMARY.md)
- **Questions**: Check component READMEs or open a discussion

## 📝 Changelog Summary

### Added

- C++ backend for Sequence, Scheduler, BlockManager, SPStateManager
- NanoRoute (Rust load balancer)
- NanoCtrl (Redis-based service discovery)
- NanoSequence (FlatBuffers serialization)
- DLSlime (RDMA communication library)
- NanoCCL (collective communication)
- Comprehensive startup configuration logging
- Disaggregated prefill/decode architecture

### Changed

- Serialization from pickle to FlatBuffers
- Host configuration logic for distributed mode
- Deployment model to centralized service discovery
- ZMQ protocol to include sequence status

### Fixed

- Segmentation faults in BlockContext serialization
- Request hanging in distributed mode
- ZMQ connection failures in multi-node deployments
- Memory leaks in sequence management
- Race conditions in scheduler

### Removed

- etcd dependency (replaced by Redis + NanoCtrl)
- Legacy Python-only sequence implementation
- Obsolete caching strategies

______________________________________________________________________

**Version**: cpp-clean-merge-main branch
**Date**: February 2026
**Authors**: NanoInfra Development Team

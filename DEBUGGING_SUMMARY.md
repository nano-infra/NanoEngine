# Debugging Summary: NanoDeploy Distributed Serving Fixes

## Overview

This document summarizes the debugging and fixes made to enable distributed LLM inference with prefill/decode disaggregation using NanoDeploy and NanoRoute.

## Timeline of Issues and Fixes

### 1. Initial Problem: Hanging Curl Requests

**Symptom:**

- Curl requests to NanoRoute server hanging indefinitely
- Engine servers receiving requests but no responses reaching client

**Investigation:**

- Added debug logging to NanoRoute's engine_adapter.rs
- Discovered ZMQ packets being received but tokens not forwarded to client

**Root Cause:**

- NanoRoute only handled `RUNNING_DECODE` status
- Prefill tokens with `RUNNING_PREFILL` status were ignored

**Fix:**

```rust
// NanoRoute/src/engine_adapter.rs:179
} else if (status == SequenceStatus::RUNNING_PREFILL || status == SequenceStatus::RUNNING_DECODE) && token_id > 0 {
    if let Some(state) = map.get_mut(&seq_id) {
        state.accumulated_tokens.push(token_id);
        let _ = state.sender.send(StreamEvent::Token(token_id));
    }
}
```

### 2. Segmentation Faults in BlockContext

**Symptom:**

- `SIGSEGV` crashes in `CreateBlockContext()` during serialization
- Crashes occurring during prefill→decode migration

**Investigation:**

- Stack traces pointed to `block_table()` function accessing null pointers
- Issue in `sp_block_table` vector containing uninitialized elements

**Root Cause:**

- When resizing `sp_block_table`, new elements were default-initialized to nullptr
- Subsequent access to these null pointers caused segfaults

**Initial Fix:**

```cpp
// NanoSequence/nanosequence/csrc/sequence/sequence.cpp:137-152
std::vector<int>& Sequence::block_table(BlockContextSlot slot, int sp_idx)
{
    auto& ctx = block_ctx(slot);
    if (sp_idx >= static_cast<int>(ctx.sp_block_table.size())) {
        size_t old_size = ctx.sp_block_table.size();
        ctx.sp_block_table.resize(sp_idx + 1);
        // Initialize all new elements to avoid null pointers
        for (size_t i = old_size; i < ctx.sp_block_table.size(); ++i) {
            ctx.sp_block_table[i] = std::make_unique<fbs::IntListT>();
        }
    }
    if (!ctx.sp_block_table[sp_idx]) {
        ctx.sp_block_table[sp_idx] = std::make_unique<fbs::IntListT>();
    }
    return ctx.sp_block_table[sp_idx]->values;
}
```

### 3. Persistent Segfaults in Serialization

**Symptom:**

- Segfaults continued even after initial fix
- Null pointers appearing from other code paths

**Root Cause:**

- Multiple code paths could create invalid BlockContext state
- Need comprehensive validation before serialization

**Comprehensive Fix:**

```cpp
// NanoSequence/nanosequence/csrc/sequence/serialization.cpp:42-68
static void validate_block_context(fbs::BlockContextT& ctx)
{
    // Ensure engine_id is valid
    if (ctx.engine_id.empty()) {
        ctx.engine_id = "";
    }

    // Ensure num_dispatched_tokens matches attention_sp
    if (ctx.num_dispatched_tokens.size() != static_cast<size_t>(ctx.attention_sp)) {
        ctx.num_dispatched_tokens.resize(ctx.attention_sp, 0);
    }

    // Ensure sp_block_table matches attention_sp and has no null pointers
    if (ctx.sp_block_table.size() != static_cast<size_t>(ctx.attention_sp)) {
        ctx.sp_block_table.resize(ctx.attention_sp);
    }
    for (size_t i = 0; i < ctx.sp_block_table.size(); ++i) {
        if (!ctx.sp_block_table[i]) {
            ctx.sp_block_table[i] = std::make_unique<fbs::IntListT>();
        }
    }
}
```

Applied validation:

- Before serialization in `serialize_sequences()`
- After deserialization in `deserialize_sequences()`
- Null slot initialization in serialization loop

### 4. Distributed ZMQ Connection Issues

**Symptom:**

- Engines running on remote nodes not receiving connections
- NanoRoute couldn't connect to engines on other nodes

**Root Cause:**

- Engines registering with `host="0.0.0.0"` (bind on all interfaces)
- Registration used `127.0.0.1` for ZMQ connect address
- NanoRoute tried connecting to `127.0.0.1` which doesn't work for remote nodes

**Fix:**

```python
# NanoDeploy/nanodeploy/server/llm_component.py:119
# For ZMQ connection: use 127.0.0.1 if host is 0.0.0.0 (localhost mode),
# otherwise use the specified host IP (distributed mode)
zmq_host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host

payload = {
    "engine_id": self.engine_id,
    "role": self.config.mode,
    "world_size": self.config.attn_world_size,
    "num_blocks": self.config.num_kvcache_blocks,
    "host": zmq_host,  # Use computed zmq_host
    "port": self.config.port,
    "peer_addrs": peer_addrs,
}
```

Same logic applied to `get_engine_info()` method.

### 5. Operational Visibility Enhancement

**Enhancement:**

- Added startup configuration summary logging

**Implementation:**

```python
# NanoDeploy/nanodeploy/server/engine_server.py:152-170
zmq_host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host

logger.info("=" * 80)
logger.info("Engine Server Started - Configuration Summary")
logger.info("=" * 80)
logger.info(f"Engine ID:       {self.engine.engine_id}")
logger.info(f"Mode:            {self.config.mode}")
logger.info(f"Model:           {self.config.model}")
logger.info(f"Bind Address:    {listen_addr} (listening on all interfaces)")
logger.info(f"ZMQ Connect:     tcp://{zmq_host}:{self.config.port}")
logger.info(f"World Size:      {self.config.attn_world_size}")
logger.info(f"Attention:       DP={self.config.attention_dp}, SP={self.config.attention_sp}, TP={self.config.attention_tp}")
logger.info(f"FFN:             DP={self.config.ffn_dp}, EP={self.config.ffn_ep}, TP={self.config.ffn_tp}")
logger.info(f"KV Cache:        {self.config.num_kvcache_blocks} blocks x {self.config.kvcache_block_size} tokens")
logger.info(f"Max Tokens:      {self.config.max_num_batched_tokens} batched, {self.config.max_model_len} model length")
logger.info(f"NanoCtrl:        {self.config.nanoctrl_address or 'Not configured'}")
logger.info(f"Ray Address:     {self.config.ray_address}")
logger.info("=" * 80)
```

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│  curl/HTTP → NanoRoute (Rust, OpenAI-compatible API)       │
└────────────────────┬────────────────────────────────────────┘
                     │ ZMQ DEALER
                     ├──────────────┬─────────────────────────┐
                     │              │                         │
         ┌───────────▼──────┐  ┌───▼────────────┐  ┌────────▼────────┐
         │  Prefill Engine  │  │  Decode Engine │  │  Decode Engine  │
         │   (Node 1)       │  │   (Node 2)     │  │   (Node 3)      │
         │   Python         │  │   Python       │  │   Python        │
         └──────────────────┘  └────────────────┘  └─────────────────┘
                │                       │                    │
                └───────────────────────┴────────────────────┘
                          RDMA P2P Links (KV Cache Migration)
                                      │
                            ┌─────────▼─────────┐
                            │   NanoCtrl        │
                            │   (Redis-based)   │
                            │   Service Disc.   │
                            └───────────────────┘
```

## Key Concepts

### Sequence Status Flow

```
WAITING → RUNNING_PREFILL → RUNNING_DECODE → FINISHED
```

### BlockContext

- Stores KV cache block allocation information
- Contains `sp_block_table`: vector of block tables for sequence parallelism
- Must be properly initialized to avoid null pointer dereferences
- Validated before serialization and after deserialization

### ZMQ Communication

- Protocol: DEALER sockets
- Packet format: `[seq_id: u64][action: u32][payload_size: u32][payload: bytes]`
- Actions:
  - 0: StepOut (token generation)
  - 1: AddRequest / Migration
  - 2: GetEngineInfo

### Host Configuration

- `--host 0.0.0.0`: Bind on all interfaces, use `127.0.0.1` for ZMQ connect (localhost mode)
- `--host <IP>`: Bind on all interfaces, use specified IP for ZMQ connect (distributed mode)

## Files Modified

### NanoRoute (Rust)

- `NanoRoute/src/engine_adapter.rs`
  - Line 179: Handle both RUNNING_PREFILL and RUNNING_DECODE status
  - Added debug logging for packet tracking
  - Added warning for unhandled statuses

### NanoSequence (C++)

- `NanoSequence/nanosequence/csrc/sequence/sequence.cpp`

  - Lines 137-152: Initialize sp_block_table elements after resize

- `NanoSequence/nanosequence/csrc/sequence/serialization.cpp`

  - Lines 42-68: Added validate_block_context() helper
  - Lines 86-102: Validate/initialize slots before serialization
  - Lines 148-153: Validate BlockContext after deserialization

### NanoDeploy (Python)

- `NanoDeploy/nanodeploy/server/llm_component.py`

  - Lines 45, 119: Use `127.0.0.1` if host is `0.0.0.0`, else use specified IP
  - Applied to both `_register_with_nanoctrl()` and `get_engine_info()`

- `NanoDeploy/nanodeploy/server/engine_server.py`

  - Lines 152-170: Added comprehensive startup configuration summary
  - Lines 210-214: Added banner to main() function
  - Added extensive debug logging for packet handling

- `NanoDeploy/nanodeploy/config.py`

  - No changes needed (already had host configuration)

## Testing Results

### Successful End-to-End Test

```bash
curl -X POST http://10.1.16.4:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen","prompt":"hello","max_tokens":64}'
```

**Response:**

```json
{
  "id": "req-1738978993",
  "object": "text_completion",
  "created": 1738978993,
  "model": "Qwen",
  "choices": [{
    "index": 0,
    "text": "! How can I assist you today?",
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 8,
    "total_tokens": 9
  }
}
```

**Performance:**

- Time to First Token (TTFT): ~770-780ms
- Prefill→Decode migration: Working via RDMA links
- Token generation: Stable

## Operational Notes

### Starting Engine Servers

**Localhost Mode:**

```bash
python -m nanodeploy.server.engine_server \
  --mode prefill \
  --host 0.0.0.0 \
  --port 6001
# Will use 127.0.0.1 for ZMQ connections
```

**Distributed Mode:**

```bash
python -m nanodeploy.server.engine_server \
  --mode decode \
  --host 10.1.16.4 \
  --port 6002
# Will use 10.1.16.4 for ZMQ connections
```

### Startup Log Example

```
================================================================================
Engine Server Started - Configuration Summary
================================================================================
Engine ID:       engine-abc123
Mode:            decode
Model:           /path/to/Qwen-7B
Bind Address:    tcp://*:6002 (listening on all interfaces)
ZMQ Connect:     tcp://10.1.16.4:6002
World Size:      8
Attention:       DP=2, SP=2, TP=2
FFN:             DP=2, EP=2, TP=2
KV Cache:        15000 blocks x 256 tokens
Max Tokens:      16384 batched, 16384 model length
NanoCtrl:        10.1.16.1:8080
Ray Address:     10.1.16.1:6379
================================================================================
```

## Lessons Learned

1. **Status Handling Completeness**: Always handle all possible status values in state machines, especially when status changes are critical to functionality.

2. **Memory Safety in C++ Vectors**: When resizing vectors of smart pointers, always explicitly initialize new elements to avoid null pointer dereferences.

3. **Serialization Validation**: Add comprehensive validation before serialization and after deserialization to catch data corruption early.

4. **Network Configuration Clarity**: In distributed systems, clearly distinguish between bind addresses (0.0.0.0) and connection addresses (actual IPs or localhost).

5. **Operational Visibility**: Startup summary logs significantly improve debugging and operational confidence in distributed systems.

## Future Improvements

1. **Error Handling**: Add more robust error handling for ZMQ communication failures
2. **Retry Logic**: Implement retry mechanism for transient network failures
3. **Health Checks**: Add periodic health checks for engine connectivity
4. **Metrics**: Add Prometheus metrics for request latency, token throughput
5. **Configuration Validation**: Add pre-flight checks to validate distributed configuration before startup

## References

- FlatBuffers Documentation: https://google.github.io/flatbuffers/
- ZeroMQ Guide: https://zguide.zeromq.org/
- NanoDeploy Architecture: `docs/architecture.md`
- Sequence Parallelism: `docs/sequence_parallelism.md`

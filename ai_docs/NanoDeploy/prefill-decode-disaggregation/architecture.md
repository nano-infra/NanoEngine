# Prefill-Decode Disaggregation with Automatic Peer Discovery

## Overview

This document describes the automatic peer discovery system for prefill-decode disaggregation in NanoDeploy. The system enables separate prefill and decode engines to discover each other automatically via NanoCtrl, coordinate RDMA-based KV cache migration, and generate tokens efficiently.

## Architecture

### Components

1. **NanoCtrl**: Control plane service that manages engine registration, heartbeat, and discovery

   - Provides `/register_engine` endpoint for engine registration
   - Provides `/list_engines` endpoint for peer discovery
   - Maintains engine metadata in Redis with TTL-based expiration

2. **LLMEngine**: Inference engine for prefill or decode

   - Auto-registers with NanoCtrl on initialization
   - Queries NanoCtrl for peer endpoints during migration
   - Manages RDMA connections via PeerAgent

3. **PeerAgent**: RDMA P2P connection manager (DLSlime)

   - Uses TopologyReconciler for declarative topology management
   - Reads topology specs from Redis
   - Establishes RDMA links for KV cache migration

4. **RPCEndpoint**: Sequence serialization and transfer

   - Uses FlatBuffers for efficient binary serialization
   - Transfers sequences via RDMA write_with_imm

### Data Flow

```
┌──────────────┐      register_engine      ┌──────────────┐
│ Prefill Eng  │─────────────────────────>│   NanoCtrl   │
└──────────────┘                           └──────────────┘
                                                   │
┌──────────────┐      register_engine              │ Redis
│ Decode Eng   │─────────────────────────>│   (TTL keys)  │
└──────────────┘                           └──────────────┘
       │                                           │
       │          list_engines                     │
       └───────────────────────────────────────────┘
                      (auto-discovery)

Migration Flow:
1. Prefill → Decode: RPC send sequences (FlatBuffers)
2. PeerAgent: RDMA read KV cache blocks
3. Decode: Generate tokens using migrated cache
```

## Configuration

### NanoCtrl Setup

NanoCtrl must be running and accessible:

```bash
# Start NanoCtrl with Redis
export REDIS_URL=redis://localhost:6379
./nanoctrl --server-address 0.0.0.0:3000
```

### Engine Configuration

Both prefill and decode engines need `nanoctrl_address` configured:

```python
from nanodeploy import LLMEngine, EngineConfig

config = EngineConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",  # NanoCtrl address
    enable_disaggregated_prefill=True,  # or False for decode
    # ... other config
)

engine = LLMEngine(config)
```

The engine will automatically:

1. Register with NanoCtrl on initialization
2. Send heartbeat every 30 seconds
3. Query peer endpoints when migration is needed

### Redis Key Schema

The system uses unprefixed Redis keys for simplicity:

```
:engine:{engine_id}              # Engine metadata (TTL: 60s)
:spec:topology:{peer_alias}      # PeerAgent topology spec
:status:topology:{peer_alias}    # PeerAgent connection status
```

Note: Previous versions attempted Redis key scoping with `nanoctrl_address` prefix, but this was removed for simplicity and to avoid prefix mismatch issues.

## Usage Example

### Basic Prefill-Decode Disaggregation

```python
import ray
from nanodeploy.server.llm_component import LLMComponent

# Start prefill engine
prefill = LLMComponent.options(
    num_gpus=1,
    name="prefill_actor"
).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",
    enable_disaggregated_prefill=True,
)

# Start decode engine
decode = LLMComponent.options(
    num_gpus=1,
    name="decode_actor"
).remote(
    model="meta-llama/Llama-3.1-8B-Instruct",
    nanoctrl_address="localhost:3000",
    enable_disaggregated_prefill=False,
)

# Engines auto-register and discover each other
# No manual set_peer_info() needed!

# Generate with automatic migration
prompt = "Write an essay about AI."
request_id = ray.get(prefill.generate.remote(prompt, seq_id=1))
result = ray.get(decode.generate.remote(request_id=request_id))

print(result.text)
```

### Manual Peer Discovery (Optional)

If you need to manually set peer info (e.g., for testing):

```python
prefill_info = ray.get(prefill.get_engine_info.remote())
decode_info = ray.get(decode.get_engine_info.remote())

ray.get(prefill.set_peer_info.remote(decode_info))
ray.get(decode.set_peer_info.remote(prefill_info))
```

## Implementation Details

### Automatic Peer Discovery

The `LLMEngine._fetch_peer_endpoints_from_nanoctrl()` method implements automatic discovery:

```python
def _fetch_peer_endpoints_from_nanoctrl(self) -> Dict[str, List[str]]:
    """Query peer_endpoints from NanoCtrl /list_engines with caching."""
    # Check cache (TTL: 10 seconds)
    if self._peer_endpoints_cache is not None:
        cached_at, cached = self._peer_endpoints_cache
        if time.time() - cached_at < 10.0:
            return cached

    # Query NanoCtrl
    url = f"http://{self.config.nanoctrl_address}/list_engines"
    response = httpx.post(url, json={}, timeout=5.0)
    data = response.json()

    # Build peer_endpoints map
    peer_endpoints = {}
    for eng in data.get("engines", []):
        engine_id = eng.get("id")
        peer_addrs = eng.get("peer_addrs", [])
        if engine_id:
            peer_endpoints[engine_id] = peer_addrs

    # Cache result
    self._peer_endpoints_cache = (time.time(), peer_endpoints)
    return peer_endpoints
```

Key features:

- **Caching**: Results cached for 10 seconds to reduce NanoCtrl load
- **Error handling**: Returns empty dict on failure, doesn't crash
- **Invocation**: Called during `step()` when sequences need migration

### Engine Registration

Engines register automatically in `LLMEngine.__init__()`:

```python
def _register_with_nanoctrl(self):
    """Register this engine with NanoCtrl."""
    if not self.config.nanoctrl_address:
        return

    url = f"http://{self.config.nanoctrl_address}/register_engine"
    payload = {
        "id": self.config.engine_id,
        "peer_addrs": [f"0.0.0.0:{port}" for port in self.peer_ports],
    }

    response = httpx.post(url, json=payload, timeout=5.0)
    response.raise_for_status()
```

A background heartbeat thread refreshes the registration every 30 seconds.

### BlockLocation Binding

FlatBuffers `BlockLocation` structs are exposed to Python via pybind11:

```cpp
py::class_<fbs::BlockLocation>(m, "BlockLocation")
    .def(py::init<>())
    .def(py::init<int, int>())
    .def_property_readonly("first", [](const fbs::BlockLocation& bl) {
        return bl.first();
    })
    .def_property_readonly("second", [](const fbs::BlockLocation& bl) {
        return bl.second();
    })
    .def("__repr__", [](const fbs::BlockLocation& bl) {
        return "BlockLocation(first=" + std::to_string(bl.first()) +
               ", second=" + std::to_string(bl.second()) + ")";
    });
```

Access in Python:

```python
# OLD (incorrect): block_idx[0], block_idx[1]
# NEW (correct):   block_idx.first, block_idx.second

sp_idx = block_location.first
page_id = block_location.second
```

### Sequence Migration

The migration flow uses RDMA for efficient KV cache transfer:

1. **Prepare**: Prefill calls `seq.migrate()` to move ACTIVE → MIGRATE slot
2. **Serialize**: FlatBuffers serializes sequences including BlockLocation references
3. **RPC Transfer**: RDMA write_with_imm sends serialized sequences
4. **Deserialize**: Decode reconstructs sequences from FlatBuffers
5. **RDMA Read**: PeerAgent reads KV cache blocks using BlockLocation mapping
6. **Generate**: Decode generates tokens using migrated cache

## Troubleshooting

### Common Issues

#### 1. Timeout waiting for RDMA peers

**Symptoms:**

```
Timeout waiting for peers... Connected: set()
```

**Causes:**

- Redis key prefix mismatch between NanoCtrl and PeerAgent
- Network connectivity issues
- PeerAgent not reading topology specs

**Fix:**

- Ensure both use same prefix (currently: empty string)
- Check Redis keys: `redis-cli KEYS "*topology*"`
- Verify peer_addrs in NanoCtrl: `curl http://localhost:3000/list_engines`

#### 2. SIGSEGV in CreateBlockContext

**Symptoms:**

```
PC: @ nanodeploy::fbs::CreateBlockContext()
Stack trace: llm_engine.py:542
```

**Cause:**

- After `migrate()`, ACTIVE BlockContext was nullptr
- FlatBuffers Pack() crashed dereferencing null pointer

**Fix:**

- Changed `sequence.h` to create empty BlockContext instead of nullptr:

```cpp
// Before: get_slot(BlockContextSlot::ACTIVE) = nullptr;
// After:  get_slot(BlockContextSlot::ACTIVE) = std::make_unique<BlockContext>();
```

#### 3. TypeError: BlockLocation not subscriptable

**Symptoms:**

```
TypeError: 'nanodeploy._nanodeploy_cpp.BlockLocation' object is not subscriptable
```

**Cause:**

- Code used `block_idx[0]` but BlockLocation is a struct, not tuple

**Fix:**

- Updated all code to use `.first` and `.second` properties
- Added pybind11 binding for BlockLocation

#### 4. Garbage tokens / Empty peer_endpoints

**Symptoms:**

- Generated tokens are incorrect
- Logs show `peer_endpoints: {}`

**Cause:**

- NanoCtrl `/list_engines` returning empty or filtered engines
- Prefix mismatch in Redis key lookup

**Fix:**

- Fixed `list_engines` to use correct prefix: `{prefix}:engine:`
- Changed filtering to include engines with empty peer_addrs

#### 5. Engine key deleted (TTL expiration)

**Symptoms:**

- Engine registered but disappeared after 60 seconds

**Cause:**

- Redis TTL expired without heartbeat refresh

**Fix:**

- Heartbeat thread refreshes registration every 30 seconds
- TTL is 60 seconds, providing safety margin

## Performance

Typical performance on single GPU:

```
Prefill:  ~1900 tok/s
Decode:   ~300 tok/s
Latency:  ~10ms per token (decode)
```

RDMA migration overhead: ~5-10ms for typical sequences (\< 1000 tokens)

## Future Improvements

1. **Multi-region support**: Namespace isolation for multiple NanoCtrl instances
2. **Load balancing**: Smart routing to least-loaded engine
3. **Failure recovery**: Automatic reconnection on peer failure
4. **Metrics**: Prometheus metrics for migration latency
5. **Security**: mTLS for engine-to-engine communication

## References

- [NanoCtrl REST API](../NanoCtrl/README.md)
- [DLSlime RDMA Layer](../DLSlime/README.md)
- [FlatBuffers Serialization](../NanoSequence/proto/sequence.fbs)
- [Example: pd_disagg.py](examples/pd_disagg.py)

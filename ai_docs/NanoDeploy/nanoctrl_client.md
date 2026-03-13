# NanoCtrlClient — Engine Lifecycle Shared Module

**File:** `NanoDeploy/nanodeploy/server/nanoctrl_client.py`

______________________________________________________________________

## Why It Exists

Before this module, `LLMComponent` and `EncoderEngine` each had independent,
copy-pasted implementations of the same four NanoCtrl operations:

| Operation      | `llm_component.py`            | `encoder_engine.py`                                                |
| -------------- | ----------------------------- | ------------------------------------------------------------------ |
| register URL   | `f"{addr}/register_engine"`   | `f"http://{addr}/register_engine"`                                 |
| heartbeat URL  | `f"{addr}/heartbeat_engine"`  | `f"http://{addr}/heartbeat_engine"`                                |
| unregister URL | `f"{addr}/unregister_engine"` | `f"http://{addr}/unregister_engine"` ← **BUG** (missing `http://`) |

The inconsistency in the unregister URL caused a real bug: encoder shutdown
silently failed to call `/unregister_engine`, so NanoRoute never received the
REMOVE event and continued routing requests to a dead encoder.

`NanoCtrlClient` is a single source of truth for all NanoCtrl HTTP transport
and heartbeat threading. Each engine provides its own registration payload; the
client owns all transport concerns.

______________________________________________________________________

## API

```python
class NanoCtrlClient:
    def __init__(self, address: str, scope: str | None = None)
```

`address` accepts both `"host:port"` and `"http://host:port"` — the `http://`
scheme is added when absent, **once at construction**.

### Methods

| Method                                          | Description                                                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `register(engine_id, extra: dict) -> bool`      | POST `/register_engine`. Injects `engine_id` and `scope` automatically.                                 |
| `unregister() -> bool`                          | POST `/unregister_engine` for the registered engine.                                                    |
| `heartbeat() -> str`                            | POST `/heartbeat_engine`. Returns `"ok"`, `"not_found"`, or `"error"`.                                  |
| `get_engine_info(engine_id) -> dict \| None`    | POST `/get_engine_info`. Used for on-demand peer discovery.                                             |
| `start_heartbeat(interval, on_not_found, name)` | Starts a daemon thread sending heartbeats every `interval` seconds. No-op if thread is already running. |
| `stop_heartbeat(timeout)`                       | Signals heartbeat thread to stop and joins it.                                                          |
| `stop(timeout)`                                 | `stop_heartbeat()` + `unregister()`. Call from engine shutdown.                                         |

### `on_not_found` callback

When NanoCtrl responds `status=not_found` (e.g. after a NanoCtrl restart), the
heartbeat thread calls `on_not_found()` if provided. This lets the engine
re-register itself without restarting the heartbeat thread.

`LLMComponent` uses this to re-register on NanoCtrl restart:

```python
self._nanoctrl.start_heartbeat(
    on_not_found=self._register_with_nanoctrl,  # re-registers, then start_heartbeat no-ops
    name=f"heartbeat-{self.engine_id}",
)
```

`EncoderEngine` does not use `on_not_found` (simpler lifecycle).

______________________________________________________________________

## Usage Pattern

Every engine follows this pattern:

```python
# __init__
self._nanoctrl: NanoCtrlClient | None = None
if config.nanoctrl_address:
    self._register_with_nanoctrl()

# _register_with_nanoctrl
def _register_with_nanoctrl(self):
    if self._nanoctrl is None:
        self._nanoctrl = NanoCtrlClient(config.nanoctrl_address, config.nanoctrl_scope)
    ok = self._nanoctrl.register(self.engine_id, {...engine-specific fields...})
    if ok:
        self._nanoctrl.start_heartbeat(name=f"hb-{self.engine_id}")

# shutdown
if self._nanoctrl:
    self._nanoctrl.stop()   # stop heartbeat + unregister
```

______________________________________________________________________

## Engine-Specific Registration Payloads

Each engine provides its own `extra` dict to `register()`.
`engine_id` and `scope` are injected automatically by the client.

### LLMComponent

```python
extra = {
    "role": config.mode,           # "prefill" | "decode" | "hybrid"
    "world_size": config.attn_world_size,
    "num_blocks": config.num_kvcache_blocks,
    "host": zmq_host,
    "port": config.port,
    "peer_addrs": peer_addrs,      # RDMA peer agent addresses
    "p2p_host": zmq_host,
    "p2p_port": self.p2p_port,
    "max_num_seqs": config.max_num_seqs,
}
```

### EncoderEngine

```python
extra = {
    "role": "encoder",
    "world_size": 1,
    "num_blocks": 0,
    "host": info["host"],
    "port": self._zmq_port,        # ZMQ encode service port
    "peer_addrs": [peer_agent_alias],
    "p2p_host": info["host"],
    "p2p_port": self._p2p_port,    # P2P free-slot listener port
}
```

______________________________________________________________________

## Adding a New Engine Type

1. Import `NanoCtrlClient` from `nanodeploy.server.nanoctrl_client`
2. Add `self._nanoctrl: NanoCtrlClient | None = None` to `__init__`
3. Write `_register_with_nanoctrl()` following the pattern above
4. Call `self._nanoctrl.stop()` in shutdown

No URL building, no heartbeat threading, no scope injection needed in the engine.

______________________________________________________________________

## NanoCtrl TTL & Lease

| Parameter                       | Value                                            |
| ------------------------------- | ------------------------------------------------ |
| Redis key TTL                   | 60 s (set by `NanoCtrl/lua/register_engine.lua`) |
| Heartbeat interval              | 15 s                                             |
| Missed heartbeats before expiry | 4                                                |

When an engine crashes without calling `/unregister_engine`, its Redis key
expires after 60 s. **Redis key expiry does NOT publish a Pub/Sub REMOVE
event.** NanoRoute detects these stale engines via a separate periodic resync
task (every 90 s) that diffs its local pool against NanoCtrl's `/list_engines`.
See `NanoRoute/src/engine_manager.rs: handle_periodic_sync()`.

______________________________________________________________________

## Key Files

| File                                                  | Role                                     |
| ----------------------------------------------------- | ---------------------------------------- |
| `NanoDeploy/nanodeploy/server/nanoctrl_client.py`     | The shared client (this module)          |
| `NanoDeploy/nanodeploy/llm_component.py`              | LLM engine — uses `NanoCtrlClient`       |
| `NanoDeployVL/nanodeployvl/encoder/encoder_engine.py` | Vision encoder — uses `NanoCtrlClient`   |
| `NanoCtrl/src/handlers/engine.rs`                     | HTTP endpoint handlers                   |
| `NanoCtrl/lua/register_engine.lua`                    | Atomic register + publish ADD event      |
| `NanoCtrl/lua/heartbeat_engine.lua`                   | Refresh TTL, returns `ok` or `not_found` |
| `NanoCtrl/lua/unregister_engine.lua`                  | Atomic unregister + publish REMOVE event |
| `NanoRoute/src/engine_manager.rs`                     | Dynamic discovery + periodic sync        |

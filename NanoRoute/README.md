# NanoRoute

High-performance Inference Router written in Rust.

## Architecture

NanoRoute acts as the request routing layer for distributed inference. It exposes an OpenAI-compatible HTTP API and connects to Python Inference Engines via **ZMQ**, with dynamic service discovery powered by **NanoCtrl + Redis**.

### Key Components

- **HTTP Layer**: Axum-based REST API (`/health`, `/v1/chat/completions`). Supports both streaming (SSE) and non-streaming responses.
- **Engine Adapter**: ZMQ DEALER socket for low-latency communication with inference engines. Wire format is a single FlatBuffers `ZmqPacket` (schema: `NanoSequence/proto/packet.fbs`) containing `action` enum + `payload` bytes. Inner payloads (SequenceList, StepOut, etc.) are defined in `NanoSequence/proto/sequence.fbs`.
- **Engine Manager**: Manages ZMQ connections to engines discovered via NanoCtrl API and Redis. Supports prefill/decode disaggregation.
- **Engine Watcher**: Redis pub/sub listener for real-time engine add/remove/update events, with gap detection and automatic full-sync recovery.
- **Tokenizer**: HuggingFace `tokenizers` + `minijinja` for ChatML template rendering.

### Service Discovery Flow

1. On startup, NanoRoute queries **NanoCtrl** for the Redis address (`/get_redis_address`).
2. Loads an initial engine snapshot from both **Redis** and the **NanoCtrl API** (`/list_engines`).
3. Subscribes to Redis pub/sub channel `{scope}:nano_events:engine_update` for incremental updates.
4. On gap detection (missed revision), triggers a full re-sync.

## Getting Started

### Prerequisites

- Rust Toolchain (latest stable)
- FlatBuffers Compiler (`flatc`)
- A running **NanoCtrl** instance (provides Redis URL and engine registry)
- A running **Redis** instance (used for engine discovery pub/sub)

### Configuration

`config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 3001
model_name = "qwen3-235B"

[tokenizer]
path = "/path/to/tokenizer.json"

[engine]
mode = "Disaggregated"
nanoctrl_address = "http://10.0.0.1:3000"

[scheduler]
queue_size = 1000
timeout_ms = 30000
```

Key fields:

| Field                     | Description                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `engine.mode`             | `"Unified"` (all engines serve both prefill & decode) or `"Disaggregated"` (separate prefill/decode pools) |
| `engine.nanoctrl_address` | **Required.** NanoCtrl HTTP endpoint for Redis URL retrieval and engine listing                            |

Optional environment variable:

- `NANOCTRL_SCOPE` — Redis key prefix for multi-tenant isolation (default: empty).

### Running

```bash
cargo run -- --config config.toml
```

Override tokenizer path via CLI:

```bash
cargo run -- --config config.toml --tokenizer-path /path/to/tokenizer.json
```

### API

#### Health Check

```bash
curl http://127.0.0.1:3001/health
```

#### Chat Completions (Non-streaming)

```bash
curl -X POST http://127.0.0.1:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-235B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

#### Chat Completions (Streaming)

```bash
curl -X POST http://127.0.0.1:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-235B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": true
  }'
```

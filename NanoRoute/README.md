# NanoDeploy Server

High-performance Inference Control Plane written in Rust.

## Architecture

NanoDeploy Server acts as the central coordinator for distributed inference. It exposes an OpenAI-compatible HTTP API and manages Python Inference Engines via **Spoke**.

### Key Components

- **HTTP Layer**: Axum-based REST API (`/v1/*`).
- **Spoke Client**: Decoupled RPC layer for low-latency control messages.
- **Engine Manager**: Manages lifecycle of Python Engine processes.
- **Hybrid Protocol**: Uses POD headers for efficiency + FlatBuffers for schema-defined payloads.

## Getting Started

### Prerequisites

- Rust Toolchain (latest stable)
- FlatBuffers Compiler (`flatc`)

### Configuration

`config.toml` (Default):

```toml
[server]
host = "127.0.0.1"
port = 8080
model_name = "opt-1.3b"

[tokenizer]
path = "/path/to/tokenizer.json"

[engine]
mode = "Unified"
count = 1
config_path = "../examples/pd_non_disagg.py"
tensor_parallel = 1

[scheduler]
queue_size = 128
timeout_ms = 1000
```

### Running the Server

```bash
cargo run -- --config config.toml
```

## Smoke Testing (E2E)

This verification involves the **Server** connecting to a **Dummy Engine** via Spoke and performing a Request/Reply cycle.

### 1. Start the Dummy Engine

Simulates a Python inference engine listening on port 5000. It replies to Requests with a `StepOut` message.

```bash
# In a separate terminal
# Assuming you are in NanoDeploy/server
python3 ../examples/dummy_engine.py
```

### 2. Start the Server

The server will attempt to connect to the Engine on `127.0.0.1:5000`.

```bash
# In NanoDeploy/server
cargo run -- --config config.toml
```

*Wait for "Connected to Engine!" log.*

### 3. Send Request

```bash
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-1.3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10
  }'
```

### 4. Verify Output

- **Server Log**: `Sent AddRequest(123) to Engine`
- **Dummy Engine Log**: `Received Action`, `Generating StepOut`, `StepOut sent`

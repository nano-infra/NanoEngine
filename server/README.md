# NanoDeploy Rust Server

This is a Rust-based server for NanoDeploy that manages the `EngineActor` and exposes a streaming HTTP API.

## Prerequisites

1. **Build C++ Components**: Ensure `nanodeploy` is built.
   ```bash
   cd ../build
   ninja
   ```
2. **Models**: Ensure `/models/qwen3-0.6b-local` exists (or update `src/main.rs`).

## Setup & Run

### 1. Start Spoke Hub (and Agent)

In a separate terminal, start the Hub:

```bash
# In nano-deploy root
./build/tools/spoke_hub 8888
```

In another terminal, start the Agent (if running distributed, or if Hub doesn't auto-spawn local):

```bash
# In nano-deploy root
./build/bin/nanodeploy_agent 127.0.0.1 8888 --gpus 8 --log-dir ./logs/
```

### 2. Run Rust Server

```bash
cd server
cargo run
```

The server will:

1. Connect to Spoke Hub.
2. Allocate resources.
3. Launch `EngineActor`.
4. Initialize the Engine.
5. Start listening on port `3000`.

### 3. Usage (Streaming Chat)

Use `curl` to send a prompt. The response will be Server-Sent Events (SSE).

```bash
curl -N -X POST http://localhost:3000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "prompt_ids": [151644, 872, 198, 9707, 11, 151645],
        "max_tokens": 20
    }'
```

**Note:** `prompt_ids` are the token IDs for "Hello," (example). You need a tokenizer to convert text to IDs first, or use the raw IDs provided in Python logs.

Output format:

```
data: [151644]

data: [198]

...
```

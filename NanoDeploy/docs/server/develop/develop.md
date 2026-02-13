# NanoDeploy Server Development & Deployment Guide

## Develop

This section covers how to set up your environment for contributing to the NanoDeploy Rust Server.

### prerequisites

- **Rust**: Version 1.83.0 or later (`rustup update`).
- **Python**: Version 3.10+ (for running Engines).
- **Protoc**: Protocol Buffers Compiler (`apt install protobuf-compiler`).
- **FlatBuffers**: `flatc` compiler (automatically handled by build script, or install manually).

### Build Steps

1. **Clone Request**:

   ```bash
   git clone https://github.com/YourOrg/NanoDeploy.git
   cd NanoDeploy/server
   ```

2. **Build**:

   ```bash
   cargo build
   ```

3. **Run Tests**:

   ```bash
   cargo test
   ```

______________________________________________________________________

## Deploy (README Tutorial)

This tutorial guides you through deploying the NanoDeploy Server.

### 1. Environment Setup

Ensure you have the model weights and tokenizer files ready locally.

- Example Path: `/home/majinming/models/qwen3-0.6b-local`

### 2. Python Engine Setup

You need to install the `nanodeploy` python package.

```bash
cd NanoDeploy
pip install -e .
```

### 3. Deployment Modes

#### Mode A: Unified (Single Node / Single Process)

Suitable for testing or simple deployments.

1. **Start Engine**:
   ```bash
   python3 -m nanodeploy.server.engine_server --mode unified --port 5000 --model /path/to/model
   ```
2. **Config**: Point `config.toml` to `127.0.0.1:5000`.
3. **Run Server**: `cargo run --release ...`

#### Mode B: Disaggregated (Multi-Node / Cross-Node)

Suitable for production with separate Prefill and Decode clusters.

**Step 1: Start Prefill Engine (Host A)**

```bash
# On Host A (e.g., 192.168.1.10)
python3 -m nanodeploy.server.engine_server \
    --mode prefill \
    --port 6000 \
    --model /path/to/model
```

**Step 2: Start Decode Engine (Host B)**

```bash
# On Host B (e.g., 192.168.1.11)
python3 -m nanodeploy.server.engine_server \
    --mode decode \
    --port 7000 \
    --model /path/to/model
```

**Step 3: Configure Server (Gateway Host)**
Update `config.toml`:

```toml
[server]
port = 3000

[engine]
mode = "Disaggregated"

[[engine.prefill]]
host = "192.168.1.10"
port = 6000

[[engine.decode]]
host = "192.168.1.11"
port = 7000
```

**Step 4: Start Rust Server**

```bash
cargo run --release -- --config config.toml
```

> **Note on RDMA Bootstrap**: Once the Server connects to the manually started Engines, it will automatically orchestrate the P2P Handshake (exchanging the `public-ip` or RDMA handles you provided) to establish the high-speed Data Plane.

### 4. Verification

**Using curl:**

```bash
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-1.3b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 128
  }'
```

**Expected Output:**
You should see a stream of JSON Server-Sent Events:

```text
data: {"choices":[{"delta":{"content":" I"},"finish_reason":null,"index":0}]}
data: {"choices":[{"delta":{"content":" am"},"finish_reason":null,"index":0}]}
...
data: [DONE]
```

### 5. Troubleshooting

- **"Tokenizer not loaded"**: Check your `config.toml` path.
- **"Connection refused"**: Ensure the Python environment where `nanodeploy` is installed is active before running `cargo run`.
- **Template Error**: The server uses a simplified ChatML template by default to ensure stability. Customizing `tokenizer_config.json` is currently experimental.

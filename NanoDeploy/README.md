# NanoDeploy

A lightweight, high-performance LLM inference deployment system featuring automated node discovery and disaggregated P2P mesh coordination.

## 🚀 Quick Start: Etcd-Based Deployment

This guide explains how to deploy NanoDeploy in a **Disaggregated** configuration (Prefill and Decode separation) using **Etcd** for automated node discovery.

### 1. Prerequisites

- **Etcd**: Version 3.4+.
- **Rust**: For the API Server (`cargo` installed).
- **protoc**: For building (etcd-client). Install: `apt install protobuf-compiler` or `bash scripts/setup-protoc.sh`
- **Python 3.9+**: For the Engine nodes.
- **FlatBuffers**: Used for high-efficiency internal communication.

### 2. Configuration (`config.toml`)

The server and engines use a shared configuration file. Key sections for Etcd:

```toml
[etcd]
address = "127.0.0.1:2379"  # Your Etcd endpoint
cluster_id = "default"       # Namespace for node discovery

[engine]
mode = "Disaggregated"       # Enables Prefill/Decode separation
# Manual 'prefill' and 'decode' lists are optional when using Etcd
```

### 3. Step-by-Step Deployment

#### A. Start Etcd

If running locally, you can start a simple Etcd instance:

```bash
etcd
```

#### B. Start the NanoDeploy API Server (Rust)

The server acts as the entry point, discovering engine nodes via Etcd and routing requests.

```bash
cd NanoDeploy/server
cargo run --release -- --config config.toml
```

#### C. Start Engine Nodes (Python)

Start at least one Prefill and one Decode node. They will automatically find each other and establish a DLSlime P2P mesh.

**Start Decode Node:**

```bash
python -m nanodeploy.server.engine_server \
    --config config.toml \
    --mode decode \
    --port 6002
```

**Start Prefill Node:**

```bash
python -m nanodeploy.server.engine_server \
    --config config.toml \
    --mode prefill \
    --port 6001
```

### 4. How it Works (Automated Mesh)

1. **Registration**: Each Engine node registers itself in Etcd under `/nanodeploy/mesh/{cluster_id}/nodes/{uuid}` with a TTL lease.
2. **Readiness**: Nodes initially register as `initializing`. They transition to `ready` once they have scanned existing peers.
3. **Discovery**: The API Server watches Etcd. When a node becomes `ready`, it is added to the routing pool.
4. **P2P Handshake**: Engines watch Etcd for peers. Prefill nodes automatically initiate DLSlime handshakes with Decode nodes to enable KV Cache migration.

### 5. Verifying the Deployment

Send a test request to the API Server (port 3001):

```bash
curl -X POST http://127.0.0.1:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 🧠 Features

- **Disaggregated Prefill/Decode**: Maximize throughput by separating computation phases.
- **Automated Discovery**: No need to hardcode IP addresses in configuration.
- **Resiliency**: If an engine node crashes, Etcd leases expire and the node is automatically removed from the server's pool.
- **Efficient Migration**: Disaggregated KV Cache transfer via DLSlime P2P.

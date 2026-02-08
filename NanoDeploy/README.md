# NanoDeploy

A lightweight, high-performance LLM inference deployment system with disaggregated prefill/decode architecture, featuring NanoCtrl (Redis-based service discovery) and NanoRoute (Rust-based load balancer).

## 🧠 Key Features

- **Disaggregated Prefill/Decode**: Maximize throughput by separating computation phases
- **Automated Service Discovery**: Redis-based control plane for dynamic engine registration
- **Load Balancing**: Intelligent routing with multiple strategies (round-robin, least-batch, least-cache)
- **Efficient Migration**: Automatic KV cache migration via RDMA for zero-copy transfer
- **Distributed Execution**: Ray-based worker management across multiple nodes
- **OpenAI-Compatible API**: Standard HTTP endpoints for easy integration

## 🚀 Deployment Guide

This guide covers deploying NanoDeploy using **NanoCtrl** (Redis-based control plane) and **NanoRoute** (Rust-based load balancer) with ZMQ communication.

### Architecture Overview

```
┌─────────────┐
│   Client    │
│  (curl/SDK) │
└──────┬──────┘
       │ HTTP (OpenAI-compatible)
       ▼
┌─────────────────┐
│   NanoRoute     │  ← Rust router with load balancing
│  (Load Balancer)│
└────────┬────────┘
         │ ZMQ DEALER
         ├─────────────┬──────────────┐
         │             │              │
    ┌────▼────┐   ┌───▼─────┐   ┌───▼─────┐
    │ Prefill │   │ Decode  │   │ Decode  │
    │ Engine  │   │ Engine  │   │ Engine  │
    │(Python) │   │(Python) │   │(Python) │
    └────┬────┘   └────┬────┘   └────┬────┘
         │             │              │
         └─────────────┴──────────────┘
                   │ RDMA P2P (KV Cache)
                   │
            ┌──────▼──────┐
            │  NanoCtrl   │  ← Redis-based service registry
            │  (Redis)    │     with heartbeat & TTL
            └─────────────┘
                   │
            ┌──────▼──────┐
            │     Ray     │  ← Distributed worker management
            └─────────────┘
```

### Prerequisites

#### Hardware

- NVIDIA GPUs with CUDA support
- RDMA-capable NICs (for multi-node deployment)
- Sufficient GPU memory for model and KV cache

#### Software

- Python 3.8+
- CUDA 11.8+ or 12.1+
- Ray (distributed workers)
- Redis (for NanoCtrl)
- Rust 1.70+ (for NanoRoute)
- ZMQ (inter-process communication)

#### Python Dependencies

```bash
pip install torch transformers accelerate
pip install ray redis zmq flatbuffers httpx pydantic jsonargparse
```

### Step-by-Step Deployment

#### 1. Start Redis (for NanoCtrl)

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis (bind to all interfaces for multi-node)
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no
```

#### 2. Start NanoCtrl

```bash
cd NanoCtrl
cargo build --release

# Start NanoCtrl service
./target/release/nanoctrl \
  --host 0.0.0.0 \
  --port 8080 \
  --redis-url redis://127.0.0.1:6379 \
  --ttl 60
```

**NanoCtrl Configuration:**

- `--host`: Bind address (default: `0.0.0.0`)
- `--port`: HTTP API port (default: `8080`)
- `--redis-url`: Redis connection URL
- `--ttl`: Engine registration TTL in seconds (default: 60)

**NanoCtrl API Endpoints:**

- `POST /register_engine` - Register new engine
- `POST /heartbeat_engine` - Refresh engine TTL
- `POST /unregister_engine` - Remove engine
- `GET /list_engines` - List all active engines
- `GET /get_engine/{role}` - Get engines by role (prefill/decode)

#### 3. Start Ray Cluster

**Head Node:**

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

**Worker Nodes:**

```bash
ray start --address=<head-node-ip>:6379
```

**Verify:**

```bash
ray status
```

#### 4. Start Engine Servers

Create a configuration file (e.g., `engine_config.yaml`):

```yaml
# Model configuration
model: "/path/to/model"  # or HuggingFace model ID
max_model_len: 16384
max_num_batched_tokens: 16384

# Scheduler
max_num_seqs: 256
loop_count: 16

# KV Cache
kvcache_block_size: 256
num_kvcache_blocks: 15000
gpu_memory_utilization: 0.9

# Parallelism (example: 8 GPUs)
attention_dp: 2
attention_sp: 2
attention_tp: 2
ffn_dp: 2
ffn_ep: 2
ffn_tp: 2

# Deployment
mode: "prefill"  # or "decode"
host: "0.0.0.0"  # always bind on all interfaces
port: 6001

# Distributed
ray_address: "127.0.0.1:6379"
nanoctrl_address: "127.0.0.1:8080"

# Logging
log_level: "INFO"
```

**Start Prefill Engine (Localhost Mode):**

```bash
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode prefill \
  --host 0.0.0.0 \
  --port 6001 \
  --nanoctrl_address 127.0.0.1:8080 \
  --ray_address 127.0.0.1:6379
```

**Start Prefill Engine (Distributed Mode on Node 10.1.16.4):**

```bash
# Join Ray cluster first
ray start --address=10.1.16.1:6379

# Start engine with node's actual IP
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode prefill \
  --host 10.1.16.4 \
  --port 6001 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

**Start Decode Engines (on different nodes):**

```bash
# Decode Engine 1 (Node 10.1.16.5)
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode decode \
  --host 10.1.16.5 \
  --port 6002 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379

# Decode Engine 2 (Node 10.1.16.6)
python -m nanodeploy.server.engine_server \
  --config engine_config.yaml \
  --mode decode \
  --host 10.1.16.6 \
  --port 6003 \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

**Important: Host Configuration**

- Always use `--host 0.0.0.0` to bind on all interfaces
- ZMQ connection address is auto-computed:
  - If `host=0.0.0.0`: Use `127.0.0.1` (localhost mode)
  - Otherwise: Use specified IP (distributed mode)
- Engine auto-registers with NanoCtrl on startup
- Heartbeat sent every 15 seconds to maintain registration

**Verify Engine Startup:**

Look for the startup summary in logs:

```
================================================================================
Engine Server Started - Configuration Summary
================================================================================
Engine ID:       engine-abc123
Mode:            prefill
Model:           /path/to/Qwen-7B
Bind Address:    tcp://*:6001 (listening on all interfaces)
ZMQ Connect:     tcp://10.1.16.4:6001
World Size:      8
Attention:       DP=2, SP=2, TP=2
FFN:             DP=2, EP=2, TP=2
KV Cache:        15000 blocks x 256 tokens
Max Tokens:      16384 batched, 16384 model length
NanoCtrl:        10.1.16.1:8080
Ray Address:     10.1.16.1:6379
================================================================================
```

#### 5. Start NanoRoute

```bash
cd NanoRoute
cargo build --release

# Start NanoRoute load balancer
./target/release/nanoroute \
  --host 0.0.0.0 \
  --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080 \
  --routing-strategy round-robin
```

**NanoRoute Configuration:**

- `--host`: Bind address (default: `0.0.0.0`)
- `--port`: HTTP API port (default: `38080`)
- `--nanoctrl-url`: NanoCtrl URL for engine discovery
- `--routing-strategy`: Load balancing strategy
  - `round-robin`: Distribute evenly (default)
  - `least-batch`: Route to engine with fewest requests
  - `least-cache`: Route to engine with most available KV cache

### Testing the Deployment

#### Health Check

```bash
# Check NanoCtrl
curl http://localhost:8080/list_engines

# Check NanoRoute
curl http://localhost:38080/health
```

#### Send Test Request

**Non-Streaming Completions:**

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "prompt": "Hello, how are you?",
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

**Expected Response:**

```json
{
  "id": "req-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen",
  "choices": [{
    "index": 0,
    "text": " I'm doing well, thank you! How can I assist you?",
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 11,
    "total_tokens": 17
  }
}
```

**Streaming Completions:**

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "prompt": "Write a haiku about AI",
    "max_tokens": 64,
    "stream": true
  }'
```

### Configuration Parameters

#### Engine Server Parameters

| Parameter                | Type  | Default            | Description                                |
| ------------------------ | ----- | ------------------ | ------------------------------------------ |
| `model`                  | str   | Required           | Model path or HuggingFace ID               |
| `mode`                   | str   | `"hybrid"`         | Engine mode: `prefill`, `decode`, `hybrid` |
| `host`                   | str   | `"0.0.0.0"`        | Bind address (ZMQ connect auto-computed)   |
| `port`                   | int   | `5000`             | ZMQ port                                   |
| `engine_id`              | str   | Auto               | Unique engine identifier                   |
| `max_model_len`          | int   | `16384`            | Maximum sequence length                    |
| `max_num_batched_tokens` | int   | `16384`            | Max tokens per batch                       |
| `max_num_seqs`           | int   | `256`              | Max concurrent sequences                   |
| `kvcache_block_size`     | int   | `256`              | KV cache block size (tokens)               |
| `num_kvcache_blocks`     | int   | `15000`            | Number of KV cache blocks                  |
| `gpu_memory_utilization` | float | `0.9`              | GPU memory usage fraction                  |
| `attention_dp`           | int   | `1`                | Attention data parallelism                 |
| `attention_sp`           | int   | `1`                | Attention sequence parallelism             |
| `attention_tp`           | int   | `1`                | Attention tensor parallelism               |
| `ffn_dp`                 | int   | `1`                | FFN data parallelism                       |
| `ffn_ep`                 | int   | `1`                | FFN expert parallelism (for MoE)           |
| `ffn_tp`                 | int   | `1`                | FFN tensor parallelism                     |
| `ray_address`            | str   | `"127.0.0.1:6379"` | Ray cluster address                        |
| `nanoctrl_address`       | str   | `None`             | NanoCtrl address (optional)                |
| `log_level`              | str   | `"CRITICAL"`       | Logging level                              |

#### Parallelism Configuration

**World Size:** Must satisfy `attention_dp × attention_sp × attention_tp == ffn_dp × ffn_ep × ffn_tp`

**Example Configurations:**

```yaml
# Small (1 GPU)
attention_dp: 1
attention_sp: 1
attention_tp: 1
ffn_dp: 1
ffn_ep: 1
ffn_tp: 1

# Medium (8 GPUs)
attention_dp: 2
attention_sp: 2
attention_tp: 2
ffn_dp: 2
ffn_ep: 2
ffn_tp: 2

# Large (64 GPUs)
attention_dp: 4
attention_sp: 4
attention_tp: 4
ffn_dp: 4
ffn_ep: 4
ffn_tp: 4
```

### Deployment Patterns

#### Single Node (Development)

```bash
# Terminal 1: Redis
redis-server --port 6379

# Terminal 2: NanoCtrl
cd NanoCtrl && ./target/release/nanoctrl

# Terminal 3: Ray
ray start --head

# Terminal 4: Prefill Engine
python -m nanodeploy.server.engine_server \
  --mode prefill --host 0.0.0.0 --port 6001 \
  --nanoctrl_address 127.0.0.1:8080

# Terminal 5: Decode Engine
python -m nanodeploy.server.engine_server \
  --mode decode --host 0.0.0.0 --port 6002 \
  --nanoctrl_address 127.0.0.1:8080

# Terminal 6: NanoRoute
cd NanoRoute && ./target/release/nanoroute --port 38080
```

#### Multi-Node (Production)

**Control Node (10.1.16.1):**

```bash
# Redis
redis-server --bind 0.0.0.0 --port 6379 --protected-mode no

# NanoCtrl
cd NanoCtrl && ./target/release/nanoctrl \
  --host 0.0.0.0 --port 8080

# Ray Head
ray start --head --port=6379 --dashboard-host=0.0.0.0

# NanoRoute
cd NanoRoute && ./target/release/nanoroute \
  --host 0.0.0.0 --port 38080 \
  --nanoctrl-url http://10.1.16.1:8080
```

**Compute Nodes:**

```bash
# Join Ray cluster
ray start --address=10.1.16.1:6379

# Start engine with node's actual IP
python -m nanodeploy.server.engine_server \
  --mode <prefill|decode> \
  --host <node-ip> \
  --port <port> \
  --nanoctrl_address 10.1.16.1:8080 \
  --ray_address 10.1.16.1:6379
```

### Monitoring

#### Check Engine Status

```bash
# List all engines
curl http://localhost:8080/list_engines

# Get prefill engines
curl http://localhost:8080/get_engine/prefill

# Get decode engines
curl http://localhost:8080/get_engine/decode
```

#### Engine Logs

Engines log important events:

- Startup configuration summary
- Request reception and processing
- Token generation (RUNNING_PREFILL, RUNNING_DECODE)
- Migration events (prefill → decode)
- Heartbeat status

### Troubleshooting

#### Engine Not Registering

**Problem:** Engine starts but doesn't appear in `/list_engines`

**Solutions:**

1. Check NanoCtrl is accessible:
   ```bash
   curl http://<nanoctrl-host>:8080/list_engines
   ```
2. Verify `nanoctrl_address` in engine config
3. Check network connectivity
4. Review engine logs for registration errors

#### ZMQ Connection Failures

**Problem:** "Connection refused" in logs, packets not reaching engines

**Solutions:**

1. **Host Configuration:**

   - Localhost mode: Use `--host 0.0.0.0` (auto-uses 127.0.0.1 for ZMQ)
   - Distributed mode: Use `--host <actual-ip>` (uses that IP for ZMQ)

2. **Firewall Rules:**

   ```bash
   sudo ufw allow 6001:6010/tcp
   ```

3. **Verify Ports:**

   ```bash
   netstat -tlnp | grep <port>
   ```

#### Request Hanging

**Problem:** Curl hangs, no tokens generated

**Solutions:**

1. Check engine status in NanoCtrl
2. Verify at least one prefill and one decode engine running
3. Check NanoRoute logs for connection errors
4. Ensure engines handle both RUNNING_PREFILL and RUNNING_DECODE status

#### Segmentation Faults

**Problem:** Engine crashes with SIGSEGV during migration

**Solutions:**

1. Update to latest version (includes BlockContext validation fixes)
2. Check GPU memory: `nvidia-smi`
3. Reduce `gpu_memory_utilization` or `num_kvcache_blocks`

### Performance Tuning

#### Latency Optimization

- Reduce `max_num_batched_tokens` for lower per-request latency
- Use fewer decode engines to reduce routing overhead
- Enable RDMA for faster KV cache migration

#### Throughput Optimization

- Increase `max_num_batched_tokens` for higher throughput
- Increase `max_num_seqs` for more concurrent requests
- Add more decode engines for horizontal scaling
- Tune parallelism (higher DP for throughput)

#### Memory Optimization

- Reduce `num_kvcache_blocks` to save GPU memory
- Increase `attention_sp` to split long sequences across GPUs
- Use `gpu_memory_limit_gb` to cap memory usage

### Advanced Features

**KV Cache Migration:**

- Automatic prefill→decode migration via RDMA
- Zero-copy transfer for minimal latency
- Requires RDMA-capable NICs for multi-node

**Sequence Parallelism:**

- Split long sequences across GPUs (`attention_sp > 1`)
- Reduces memory per GPU for long contexts

**Expert Parallelism (MoE Models):**

- Distribute MoE experts across GPUs (`ffn_ep > 1`)
- Supported: DeepSeek-V3, Mixtral

**Continuous Batching:**

- Dynamic request batching
- Configurable via `max_num_seqs` and `max_num_batched_tokens`

### References

- [Debugging Summary](../DEBUGGING_SUMMARY.md) - Common issues and fixes
- [Architecture Documentation](../docs/architecture.md)
- [API Reference](../docs/api.md)
- [Performance Tuning Guide](../docs/performance.md)

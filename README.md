# NanoInfra

A high-performance infrastructure for distributed LLM inference with disaggregated prefill/decode architecture, featuring RDMA-based KV cache migration and efficient resource management.

## 🌟 Overview

NanoInfra is a complete distributed system for serving large language models at scale. It separates prefill and decode workloads across different GPU nodes, automatically migrating KV cache between them via RDMA for optimal throughput and latency.

### Key Features

- **Disaggregated Architecture**: Separate prefill and decode engines for maximum GPU utilization
- **Zero-Copy KV Cache Migration**: RDMA-based transfer via DLSlime for minimal latency
- **Service Discovery**: Redis-based control plane (NanoCtrl) for dynamic engine registration
- **Intelligent Load Balancing**: Multiple routing strategies via NanoRoute (round-robin, least-batch, least-cache)
- **Distributed Execution**: Ray-based worker management across multiple nodes
- **Advanced Parallelism**: Support for data, tensor, sequence, and expert parallelism
- **OpenAI-Compatible API**: Standard HTTP endpoints for easy integration

## 📦 Components

### Core Infrastructure

#### [NanoDeploy](./NanoDeploy)

Python-based LLM inference engine with distributed execution support.

**Features:**

- Prefill, decode, and hybrid engine modes
- Automatic KV cache block management
- Continuous batching for high throughput
- Ray-based distributed worker coordination
- Support for sequence parallelism and expert parallelism (MoE models)

**Key Files:**

- `nanodeploy/server/engine_server.py` - Main engine server
- `nanodeploy/server/llm_component.py` - Engine lifecycle and registration
- `nanodeploy/config.py` - Configuration parameters

#### [NanoRoute](./NanoRoute)

Rust-based HTTP load balancer and request router with OpenAI-compatible API.

**Features:**

- OpenAI-compatible `/v1/completions` endpoint
- Multiple load balancing strategies
- Automatic engine discovery via NanoCtrl
- Streaming and non-streaming responses
- ZMQ-based communication with engines

**Key Files:**

- `src/http_server.rs` - HTTP API server
- `src/engine_adapter.rs` - ZMQ engine communication
- `src/scheduler.rs` - Request routing logic

#### [NanoCtrl](./NanoCtrl)

Rust-based control plane for service discovery and health monitoring.

**Features:**

- Redis-backed engine registry
- Automatic TTL and heartbeat management
- Role-based engine discovery (prefill/decode)
- RESTful API for engine management

**API Endpoints:**

- `POST /register_engine` - Register new engine
- `POST /heartbeat_engine` - Refresh engine TTL
- `POST /unregister_engine` - Remove engine
- `GET /list_engines` - List all active engines
- `GET /get_engine/{role}` - Get engines by role

#### [NanoSequence](./NanoSequence)

C++ library for sequence and KV cache management with FlatBuffers serialization.

**Features:**

- Efficient sequence state management
- BlockContext for KV cache allocation tracking
- FlatBuffers-based serialization for fast migration
- Support for sequence and tensor parallelism

**Key Files:**

- `nanosequence/csrc/sequence/sequence.cpp` - Sequence implementation
- `nanosequence/csrc/sequence/serialization.cpp` - Serialization logic
- `nanosequence/fbs/sequence.fbs` - FlatBuffers schema

### Communication & Networking

#### [DLSlime](./DLSlime)

High-performance RDMA communication library for KV cache migration.

**Features:**

- Zero-copy RDMA transfers
- P2P mesh networking between engines
- Lazy connection establishment
- Support for NVIDIA GPUDirect RDMA

#### [NanoCCL](./NanoCCL)

Collective communication primitives for distributed GPU operations.

**Features:**

- AllReduce, AllGather, ReduceScatter operations
- Support for NCCL and custom backends
- Integration with tensor parallelism

#### [NanoCommon](./NanoCommon)

Shared utilities and logging infrastructure.

**Features:**

- Unified logging macros
- Common data structures
- Error handling utilities

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│                  (HTTP Requests / OpenAI SDK)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │   NanoRoute     │  ← Load Balancer
                   │  (Rust/HTTP)    │     • Request routing
                   └────────┬────────┘     • Engine discovery
                            │ ZMQ           • OpenAI API
                            │
              ┌─────────────┴─────────────┐
              │                           │
         ┌────▼─────┐               ┌────▼─────┐
         │ Prefill  │               │  Decode  │
         │ Engine   │──────RDMA────▶│  Engine  │
         │ (Python) │  KV Migration │ (Python) │
         └────┬─────┘               └────┬─────┘
              │                           │
              │    ┌──────────────┐       │
              └───▶│  NanoCtrl    │◀──────┘
                   │  (Redis)     │  ← Service Registry
                   └──────────────┘     • Engine registration
                                        • Health monitoring
                   ┌──────────────┐
                   │     Ray      │  ← Distributed Workers
                   │  (Cluster)   │     • GPU management
                   └──────────────┘     • Worker scheduling
```

### Request Flow

1. **Client** sends HTTP request to NanoRoute
2. **NanoRoute** queries NanoCtrl for available prefill engine
3. **Prefill Engine** processes prompt, generates KV cache
4. **RDMA Transfer** migrates KV cache to decode engine via DLSlime
5. **Decode Engine** generates tokens incrementally
6. **NanoRoute** streams tokens back to client

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPUs with CUDA support, RDMA-capable NICs (for multi-node)
- **Software**: Python 3.8+, Rust 1.70+, CUDA 11.8+, Ray, Redis
- **Dependencies**: See individual component READMEs

### Single Node Deployment

```bash
# 1. Start Redis
redis-server --port 6379

# 2. Start NanoCtrl
cd NanoCtrl
cargo run --release -- --host 0.0.0.0 --port 8080

# 3. Start Ray
ray start --head

# 4. Start Prefill Engine
cd NanoDeploy
python -m nanodeploy.server.engine_server \
  --mode prefill \
  --host 0.0.0.0 \
  --port 6001 \
  --nanoctrl_address 127.0.0.1:8080

# 5. Start Decode Engine
python -m nanodeploy.server.engine_server \
  --mode decode \
  --host 0.0.0.0 \
  --port 6002 \
  --nanoctrl_address 127.0.0.1:8080

# 6. Start NanoRoute
cd NanoRoute
cargo run --release -- --host 0.0.0.0 --port 38080
```

### Test Request

```bash
curl -X POST http://localhost:38080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "prompt": "Hello, how are you?",
    "max_tokens": 64
  }'
```

### Multi-Node Deployment

See [NanoDeploy README](./NanoDeploy/README.md) for detailed multi-node deployment instructions.

## 📊 Performance

### Benchmarks (Example Configuration)

- **Model**: Qwen-7B
- **Setup**: 1 Prefill Node (8 GPUs) + 2 Decode Nodes (8 GPUs each)
- **Results**:
  - Time to First Token (TTFT): ~770-780ms
  - Token Throughput: High (varies by batch size)
  - KV Cache Migration: Zero-copy RDMA transfer

### Optimization Tips

**For Latency:**

- Reduce `max_num_batched_tokens`
- Use fewer decode engines
- Enable RDMA for KV cache migration
- Use sequence parallelism for long contexts

**For Throughput:**

- Increase `max_num_batched_tokens`
- Increase `max_num_seqs`
- Add more decode engines
- Tune data parallelism settings

**For Memory:**

- Reduce `num_kvcache_blocks`
- Increase sequence parallelism
- Use expert parallelism for MoE models

## 🛠️ Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/JimyMa/NanoInfra.git
cd NanoInfra

# Build Rust components
cd NanoCtrl && cargo build --release && cd ..
cd NanoRoute && cargo build --release && cd ..

# Build C++ components
cd NanoSequence
mkdir build && cd build
cmake ..
make -j
cd ../..

# Install Python dependencies
cd NanoDeploy
pip install -e .
cd ..
```

### Running Tests

```bash
# Python tests
cd NanoDeploy
pytest tests/

# Rust tests
cd NanoRoute
cargo test

cd ../NanoCtrl
cargo test
```

## 📖 Documentation

- [NanoDeploy Deployment Guide](./NanoDeploy/README.md) - Comprehensive deployment instructions
- [Debugging Summary](./DEBUGGING_SUMMARY.md) - Common issues and fixes
- [Architecture Details](./docs/architecture.md) - System architecture deep dive
- [API Reference](./docs/api.md) - API documentation
- [Performance Tuning](./docs/performance.md) - Optimization guide

## 🔧 Configuration

### Engine Configuration Example

```yaml
# Model
model: "/path/to/model"
max_model_len: 16384
max_num_batched_tokens: 16384

# Parallelism
attention_dp: 2
attention_sp: 2
attention_tp: 2
ffn_dp: 2
ffn_ep: 2
ffn_tp: 2

# KV Cache
kvcache_block_size: 256
num_kvcache_blocks: 15000
gpu_memory_utilization: 0.9

# Deployment
mode: "prefill"  # or "decode"
host: "0.0.0.0"
port: 6001
nanoctrl_address: "127.0.0.1:8080"
ray_address: "127.0.0.1:6379"
```

### Key Parameters

| Parameter                | Description                    | Typical Values |
| ------------------------ | ------------------------------ | -------------- |
| `attention_dp`           | Attention data parallelism     | 1-8            |
| `attention_sp`           | Attention sequence parallelism | 1-8            |
| `attention_tp`           | Attention tensor parallelism   | 1-8            |
| `ffn_ep`                 | FFN expert parallelism (MoE)   | 1-8            |
| `num_kvcache_blocks`     | KV cache capacity              | 10000-30000    |
| `max_num_batched_tokens` | Max tokens per batch           | 8192-32768     |

## 🐛 Troubleshooting

### Common Issues

**Engine not registering with NanoCtrl:**

- Check NanoCtrl is running: `curl http://localhost:8080/list_engines`
- Verify `nanoctrl_address` configuration
- Check network connectivity

**ZMQ connection failures:**

- For localhost: Use `--host 0.0.0.0` (auto-uses 127.0.0.1 for ZMQ)
- For distributed: Use `--host <node-ip>` (uses that IP for ZMQ)
- Check firewall rules: `sudo ufw allow 6001:6010/tcp`

**Segmentation faults:**

- Update to latest version (includes BlockContext fixes)
- Check GPU memory: `nvidia-smi`
- Reduce `gpu_memory_utilization` or `num_kvcache_blocks`

**High latency:**

- Enable RDMA for multi-node setups
- Reduce batch size
- Check network bandwidth

See [DEBUGGING_SUMMARY.md](./DEBUGGING_SUMMARY.md) for detailed troubleshooting.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Code Style

- **Python**: Follow PEP 8, use `black` formatter
- **Rust**: Follow Rust conventions, use `rustfmt`
- **C++**: Follow Google C++ style guide, use `clang-format`

## 📄 License

See individual component licenses.

## 🙏 Acknowledgments

- Ray Project for distributed computing framework
- NVIDIA for CUDA and GPUDirect RDMA
- Redis for high-performance key-value store
- FlatBuffers for efficient serialization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JimyMa/NanoInfra/issues)
- **Documentation**: Check component READMEs and debugging guide
- **Questions**: Open a GitHub discussion

## 🗺️ Roadmap

- [ ] Support for more LLM architectures (Llama 3, GPT, etc.)
- [ ] Enhanced monitoring and observability (Prometheus metrics)
- [ ] Multi-tenant request isolation
- [ ] Dynamic batching optimizations
- [ ] Support for speculative decoding
- [ ] Integration with more serving frameworks

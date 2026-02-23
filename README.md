# NanoInfra

A high-performance infrastructure for distributed LLM inference with disaggregated prefill/decode architecture, featuring RDMA-based KV cache migration and efficient resource management.

## 📦 Components

| Component                      | Language | Description                    | Key Features                                                                                    |
| ------------------------------ | -------- | ------------------------------ | ----------------------------------------------------------------------------------------------- |
| [DLSlime](./DLSlime)           | C++      | RDMA communication             | Zero-copy KV cache migration, P2P mesh networking, GPUDirect RDMA                               |
| [NanoBench](./NanoBench)       | Python   | Benchmarking tools             | Performance testing and profiling                                                               |
| [NanoCCL](./NanoCCL)           | C++      | Collective communication       | AllReduce, AllGather, ReduceScatter, tensor parallelism support                                 |
| [NanoCommon](./NanoCommon)     | C++      | Shared utilities               | Logging, common data structures, error handling                                                 |
| [NanoCtrl](./NanoCtrl)         | Rust     | Control plane                  | Redis-backed service registry, health monitoring, engine discovery                              |
| [NanoDeploy](./NanoDeploy)     | Python   | LLM inference engine           | Prefill/decode engines, KV cache management, continuous batching, Ray-based distributed workers |
| [NanoOps](./NanoOps)           | -        | Operations tools               | Deployment and management utilities                                                             |
| [NanoRoute](./NanoRoute)       | Rust     | HTTP load balancer             | OpenAI-compatible API, multiple routing strategies, engine discovery                            |
| [NanoSequence](./NanoSequence) | C++      | Sequence & KV cache management | Sequence state management, FlatBuffers serialization, block allocation                          |

## 🏗️ Architecture

```mermaid
graph TB
    Client[Client Layer<br/>HTTP Requests / OpenAI SDK]
    Route[NanoRoute<br/>Rust/HTTP<br/>Load Balancer]
    Prefill[Prefill Engine<br/>Python]
    Decode[Decode Engine<br/>Python]
    Ctrl[NanoCtrl<br/>Redis<br/>Service Registry]
    Ray[Ray Cluster<br/>Distributed Workers]
    Ops[NanoOps<br/>DevOps/Orchestration<br/>Session Management]

    Client -->|HTTP| Route
    Route -->|ZMQ| Prefill
    Route -->|ZMQ| Decode
    Prefill -->|RDMA<br/>KV Migration| Decode
    Prefill -->|Register/Heartbeat| Ctrl
    Decode -->|Register/Heartbeat| Ctrl
    Route -->|Engine Discovery| Ctrl
    Prefill -->|Worker Management| Ray
    Decode -->|Worker Management| Ray
    Ops -->|Deploy & Manage| Route
    Ops -->|Deploy & Manage| Prefill
    Ops -->|Deploy & Manage| Decode
    Ops -->|Start & Monitor| Ctrl
    Ops -->|Job Submission| Ray
    Ops -->|Session Config| Ctrl
```

## 📖 Documentation

- [Deployment Guide](./docs/deployment.md) — Step-by-step instructions for deploying with NanoCtrl, NanoRoute, and NanoDeploy

## 📄 License

See individual component [license](./LICENSE).

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JimyMa/NanoInfra/issues)
- **Documentation**: Check component READMEs

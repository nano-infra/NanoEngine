# Rust Server Development Process

## Phase I: Foundation (Completed)

- [x] Basic Axum Server.
- [x] Config Loading.
- [x] Spoke IPC Skeleton.

## Phase II: Single-Node Inference (Completed)

- [x] Tokenizer Service.
- [x] FlatBuffers Protocol Integration.
- [x] End-to-End Chat Completion (Unified Mode).
- [x] Streaming (SSE) Support.
- [x] Chat Template Support.

## Phase III: PD Disaggregation (Current)

This phase transforms the server into a cluster coordinator.

### Milestone 3.1: Multi-Engine Architecture

- [ ] Refactor `EngineManager` to support a list/map of Engines instead of a single instance.
- [ ] Implement `EngineRole` (Prefill/Decode) configuration parsing.
- [ ] Implement **Wait-for-All** logic (Server waits for all configured engines to connect).

### Milestone 3.2: P2P Handshake

- [ ] Implement `P2P_INIT` and `P2P_CONNECT` command logic.
- [ ] Implement the handshake state machine in `main.rs` or `Router`.
- [ ] Verify that Python Engines successfully establish their internal P2P links.

### Milestone 3.3: Scheduler Version 2.0

- [ ] Separate `RequestQueue` into `prefill_queue` and `decode_queue`.
- [ ] Implement the "Prefill Done" event listener.
- [ ] Implement the migration trigger logic.

### Milestone 3.4: Integration & Optimization

- [ ] Test the full PD flow with `pd_disagg.py` logic ported to Rust.
- [ ] Benchmark and optimize latency.

## Phase IV: Production Hardening (Future)

- [ ] Error Recovery (Engine Restart).
- [ ] Metrics Exporter (Prometheus).
- [ ] Distributed Tracing.

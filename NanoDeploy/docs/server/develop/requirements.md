# Rust Server Requirements

## 1. Core Objectives

Provide a high-throughput, low-latency API Gateway for LLM Serving that supports **Disaggregated Architecture**.

## 2. Functional Requirements

### 2.1 API Compatibility

- \[Req-API-01\] Support `v1/chat/completions` (OpenAI format).
- \[Req-API-02\] Support Server-Sent Events (SSE) for streaming.
- \[Req-API-03\] Support custom headers for tracing (e.g., `x-request-id`).

### 2.2 Disaggregation (PD Separation)

- \[Req-PD-01\] **Role Management**: The system MUST support configuring Engines as strictly `Prefill` or `Decode`.
- \[Req-PD-02\] **P2P Mesh**: The system MUST orchestrate the establishment of P2P connections between all Prefill and Decode nodes at startup.
- \[Req-PD-03\] **KV Migration**: The system MUST support seamless migration of context (KV Cache) from Prefill to Decode nodes.
  - *Constraint*: Migration latency SHOULD be minimized (target \< 100ms for typical loads) to prevent "first token latency" visible to user.
- \[Req-PD-04\] **Resource Cleanup**: The system MUST guarantee that KV Cache on Prefill nodes is freed after successful migration to prevent memory leaks.

### 2.3 Scheduler

- \[Req-Sched-01\] **Dual Queues**: The Scheduler MUST maintain logical separation between "New Requests" (for Prefill) and "Migrated Requests" (for Decode).
- \[Req-Sched-02\] **Load Balancing**:
  - Prefill: Least-Outstanding-Requests or Round-Robin.
  - Decode: Must consider **KV Block Availability**. Do not schedule to a Decode node if it cannot fit the new request's KV Cache.

### 2.4 Tokenizer

- \[Req-Tok-01\] Support HuggingFace `tokenizer.json`.
- \[Req-Tok-02\] Support Chat Templates (`tokenizer_config.json` or fallback).
- \[Req-Tok-03\] Async/Non-blocking execution (offload heavy encoding/decoding to worker threads).

## 3. Configuration Requirements (`config.toml`)

The configuration file MUST support defining the cluster topology:

```toml
[server]
port = 3000

[engine.prefill_pool]
counts = 2
...

[engine.decode_pool]
counts = 4
...
```

## 4. Observability Requirements

- \[Req-Obs-01\] Metrics: `prefill_latency` (P99), `decode_token_latency` (P99), `migration_latency`.
- \[Req-Obs-02\] Logs: Explicit log entries for "Request X migrated from Engine A to Engine B".

# Rust Server Testing Strategy

## 1. Unit Testing

- **Tokenizer**: Verify Chat Template rendering for key models (Qwen, Llama).
- **Scheduler**: Test the priority queue and state transitions (e.g., ensure a request moves `Pending -> Prefill -> Migrating -> Decode`).

## 2. Integration Testing (Mock Engines)

To test PD Disaggregation without heavy Python processes:

- **Mock Network**: Create a `MockSpokeServer` in Rust that simulates *two* engines listening on different ports.
  - **Port A (Prefill)**: Accepts `AddRequest`, sleeps, sends `PrefillDone`.
  - **Port B (Decode)**: Accepts `AddRequest` (migrated), streams fake tokens.
- **Validation**: Assert that the Router correctly sends the initial request to Port A, receives the completion, and then sends the *same* ID to Port B.

## 3. End-to-End Testing (Real Engines)

- **Setup**: Launch Rust Server with `config_pd.toml` pointing to 2 real Python Engines.
- **Workload**: Send a long prompt (force prefill load) and verify output quality.
- **Chaos Testing**: Kill the Decode Engine mid-stream and verify Server behavior (should 500 or retry).

## 4. Performance Testing

- **Metrics**:
  - **TTFT (Time to First Token)**: Measures Prefill + Migration Latency.
  - **TPOT (Time Per Output Token)**: Measures Decode efficiency.
- **Tools**: `wrk` or custom Python benchmark script pounding the `/v1/chat/completions` endpoint.

# Development progress

## 2026-07-30 — decentralized admission latency decomposition

### Checkpoint

- Previous decentralized admission, observability, tests, and development
  notes were committed as `dd1054b` (`Implement decentralized admission and
  observability`).
- Generated `bench_logs/` were intentionally excluded.
- Pre-checkpoint CPU validation: 92 tests passed.

### Goal

Split the high-load latency path without comparing absolute clocks across Ray
nodes:

- T0-T1: scheduled arrival to actual benchmark dispatch;
- T1-T3: dispatch to authoritative LocalEngine admission ACK;
- T1-T3 internals: RequestRouter pending, admission RPC, LocalEngine command
  queue, and local admission work.

### Implementation

`IngressAck` now carries durations computed within the owning process:

- `router_pending_ms`;
- `admission_rpc_ms`;
- `local_command_queue_ms`;
- `local_admission_ms`.

The benchmark persists those fields and derives:

- `admission_rpc_residual_ms`, covering actor mailbox/transport/polling plus
  any earlier fallback attempts;
- `frontend_ack_overhead_ms`, closing the remaining T1-T3 frontend boundary.

Router and RPC totals accumulate across fallback/global retry attempts. The
existing `global_capacity_queue_ms` remains the authoritative subset for true
DP/SP/KV capacity blocking.

### Interpretation for the next rate-50 run

- high `router_pending_ms`: frontend global queue/drain bottleneck;
- high `local_command_queue_ms`: LocalEngine cannot pick up commands between
  decode quantums quickly enough;
- high `local_admission_ms`: SP/KV planner work is expensive;
- high `admission_rpc_residual_ms`: Ray actor mailbox/concurrency or transport
  dominates before the command reaches the LocalEngine loop;
- high `frontend_ack_overhead_ms`: benchmark/router polling is lagging.

The two-node matrix uses
`scripts/sp_ablation/bench_serving_overhead.py`, so the next
`decentralized_dp2sp8` run will emit the new fields. No GPU run has been
started yet.

### Verification

- 92 explicit CPU control-plane, contract, routing, decode-RPC, SP-backend,
  sequence-proxy, and benchmark tests passed.
- `python -m compileall` and `git diff --check` passed.
- A rate-50, 18,000-request `decentralized_dp2sp8` matrix dry-run resolved to
  `scripts/sp_ablation/start_bench.sh` and its updated benchmark driver.

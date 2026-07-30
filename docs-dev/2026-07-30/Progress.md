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

## 2026-07-30 — rate-50 LocalEngine concurrency experiment

### Change and run

- Increased `LocalEngineCore` Ray `max_concurrency` from 32 to 64.
- Re-ran `decentralized_dp2sp8` at rate 50 for 360 seconds against Ray
  `10.102.206.14:8776`: 18,000/18,000 requests completed with no failures.
- Result:
  `bench_logs/rate50_decentralized_dp2sp8_mc64_20260730_0655/`.
- Before the GPU run, 37 focused tests, `compileall`, and `git diff --check`
  passed.

### Result

The higher actor concurrency improved low-percentile dispatch/admission but did
not prevent a late-run queue collapse. Compared with the July 29 rate-50
historical run:

- ingress ACK P50 improved from 12.23 s to 1.20 s, while P90/P99 worsened from
  43.44/45.51 s to 55.84/120.68 s;
- model TTFT P50 improved from 11.80 s to 2.66 s, while P90/P99 worsened from
  43.95/49.69 s to 54.43/119.55 s;
- TPOT-with-queue P50/P90/P99 changed from 81.61/84.79/87.27 ms to
  93.25/191.25/200.10 ms;
- hierarchical decode ITL P50/P90/P99 changed from 76.93/80.41/81.96 ms to
  86.09/182.01/187.20 ms.

The new decomposition attributes the ACK tail primarily to
`admission_rpc_residual_ms` (P50/P90/P99 36.87 ms/52.20 s/92.73 s), not local
admission work (37.86/84.79/133.06 ms). Local command queueing reached
1.12/2.95/3.09 s. Around 294 seconds, pending RPCs rose from roughly 50 to 604;
the run ended with 2,490 fallbacks and 802 global retries.

### Conclusion and next change

`max_concurrency=64` adds early headroom but is not a complete fix. Blocking
`admit_add()` calls retain actor concurrency slots until the single-writer loop
handles them, while the router performs one zero-timeout `ray.wait()` per
pending request. Once pending work grows, fallback and global retry attempts
amplify both RPC count and decode interference.

Next, make admission completion non-blocking and batch-oriented: batch-poll
object refs with one `ray.wait`, return an admission receipt immediately, emit
the authoritative decision from the LocalEngine loop, and add bounded
retry/backoff to avoid retry storms. Re-run current-code rate 50 at 32 and 64
only if an isolated concurrency A/B is still needed; the historical comparison
also includes intervening admission-path changes.

## 2026-07-30 — bounded admission flights and merged events

Implemented the global-queue dispatch redesign:

- LeastBatch now issues one bounded admission batch per available DP and keeps
  at most one batch ObjectRef per DP.
- The frontend polls all active admission refs with one nonblocking
  `ray.wait`, followed by one `ray.get` for ready batches.
- `capacity_epoch` blocks a deferred DP until a lifecycle reservation is
  actually released; this replaces retry-on-any-load-change behavior.
- LocalEngine processes a complete admission batch as one actor mailbox
  command while preserving single-writer scheduler admission.
- Health, cached load, add results, first-schedule, first-token, and terminal
  events now return in one consolidated RPC per DP/frontend poll cycle.
- Capacity queue time excludes ordinary frontend poll cadence. Router pending,
  batch RPC, and true capacity-blocked time remain separate.

Focused CPU validation passed: 78 tests covering control plane, contracts,
serving ingress, and routing configuration. The rate-50 Ray/GPU validation is
still pending.

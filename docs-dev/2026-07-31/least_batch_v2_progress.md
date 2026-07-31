# least_batch_v2 Progress

## Baseline

- Recorded at: 2026-07-31 UTC
- Branch: `decentralized-july`
- Commit: `e72d894d44a0320f7874de06f9beb841507f8f1b`
- Subject: `fix: remove decentralized admission history scans`

The worktree was not clean at the baseline. Preserve these pre-existing user
artifacts and do not include them in the implementation commit unless they are
explicitly required:

- Modified: `scripts/sp_ablation/start_bench.sh`
  (`DEFAULT_BATCH_SIZE` changed from 192 to 256)
- Untracked: `bench_logs/`

## Objective

Implement an opt-in `least_batch_v2` routing path with:

1. a frontend global FIFO;
2. a scalar KV-credit admission gate instead of mirrored SP placement;
3. batched, fast LocalEngine ingress receipts that do not wait for a decode
   quantum boundary;
4. one authoritative SP/KV admission decision in `LocalScheduler.admit()`;
5. bounded per-engine ingress RPC cardinality and explicit queueing metrics;
6. the existing `least_batch` path retained for A/B and rollback.

## Design decisions

- Global FIFO order is preserved at dispatch time. Cross-engine first-schedule
  order is not guaranteed.
- Dynamic KV credit estimates the immediate prompt admission footprint plus a
  small decode reserve. It does not reserve `max_tokens` worth of live KV.
- Static request validation remains authoritative locally so an impossible
  request cannot wait forever.
- P0 keeps at most one *fast* ingress batch flight per engine. The existing
  performance bug is the quantum-bound completion semantics, not the bounded
  cardinality itself.

## Implemented

- Added opt-in `router_policy=least_batch_v2`; the original `least_batch`
  implementation and default remain unchanged for A/B and rollback.
- Kept a Router-owned global FIFO. Each item has a stable `queue_seq`, and
  retries are merged with the whole pending deque by that sequence, including
  when DP flights complete in separate frontend polls.
- Replaced frontend SP-placement mirroring with an aggregate KV-credit gate:
  prompt blocks, worst-case SP block rounding, and the configured one-request
  decode reserve. Queue slots remain a bounded-ingress safety check.
- Added a per-request KV charge ledger. Charges survive cached and new quantum
  snapshots until `FirstScheduleEvent` proves the prompt allocation is visible
  locally, preventing repeated credit grants for requests still in ingress or
  LocalScheduler waiting.
- `LocalEngineCore.enqueue_add_batch()` now takes the ingress lock once, inserts
  the whole accepted prefix into its bounded lifecycle queue, and sends one
  wakeup. It returns a transport receipt without touching or waiting for
  `LocalScheduler`.
- A `queue_full` receipt blocks only the engine that returned it for that load
  generation. Healthy engines can immediately take the restored global FIFO
  head.
- Closed the abort-before-enqueue race caused by Ray actor concurrency with a
  future-ingress cancellation tombstone. Negative receipts clean the tombstone
  and produce an `ABORTED` Router terminal instead of being retried elsewhere.
- Added `dispatch_tpot_ms` to the SP ablation benchmark. It is calculated as
  `(dispatch-to-observed-terminal - unused final-quantum slots) / output tokens`
  and includes Router, RPC, LocalEngine ingress, LocalScheduler waiting,
  execution, and frontend observation. Use it as the primary cross-policy TPOT;
  retain `tpot_with_queue_ms` only as the legacy execution-boundary breakdown.
  Compare TTFT policies with `model_ttft_ms`, not bootstrap ACK timing.

## Review outcome

Independent reviews initially identified three blockers: cross-DP retry FIFO
reordering, cross-quantum KV re-granting, and abort/enqueue actor-call races.
The implementation and regressions above close all three. The final review
verdict is GO for GPU smoke and controlled A/B, with these documented limits:

- FIFO means dispatch-time head/no-bypass, not cross-DP first-schedule or
  completion ordering.
- Transport/protocol exceptions remain fail-stop; there is no batch-ID-based
  recovery or idempotent retry protocol yet.
- Outstanding queue slots are conservatively double-counted until first
  schedule, which can reduce utilization only with unusually small queue caps.

## CPU validation

```text
python -m pytest \
  tests/test_hierarchical_control_plane.py \
  tests/test_hierarchical_contract.py \
  tests/test_hierarchical_serving_ingress.py \
  tests/test_routing_config.py -q
# 105 passed

python tests/test_sequence_proxy.py
# [ok] C++ proxy containers behave as mutable views

python -m compileall -q nanodeploy examples scripts/issue003 scripts/sp_ablation
git diff --check
```

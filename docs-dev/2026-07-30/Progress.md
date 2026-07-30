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

## 2026-07-30 — batched-admission rate-50 validation

### Runtime fix

The first initialized validation exposed a merged-event lifecycle edge case:
the consolidated ingress poll could buffer the final `FinishEvent` and remove
the last router owner before the serving loop called `step()`. Consequently,
`is_finished()` returned true while a completion was still buffered. The
hierarchical completion predicate now also requires the frontend finish buffer
to be empty. A focused regression was added; 79 control-plane, contract,
serving-ingress, and routing tests pass. The fix is commit `1c5deac`.

### Successful run

- Ray: `10.102.206.14:8776`, nodes `10.102.206.14` and
  `10.102.252.174`;
- topology/workload: DP2 × SP8, rate 50, 360-second injection,
  18,000 requests;
- result:
  `bench_logs/rate50_decentralized_dp2sp8_admission_batch_fixed_20260730_0811/`;
- 18,000/18,000 requests completed, no ingress/scheduler rejection or
  benchmark failure;
- achieved dispatch rate: 49.70 req/s; total drain time: 469.96 s;
- diagnostic maxima: 2 admission ObjectRefs, 312 requests represented by
  those refs, 284 requests in the global admission FIFO, and 3,045 client
  requests outstanding.

Capacity saturation produced 2,978 request-level `admission_deferred`
retries, but zero same-generation fallback and zero `queue_full`. These are
global-queue retries gated by `capacity_epoch`; they did not increase the Ray
wait set beyond DP=2. The final per-engine commit counters include the
256-request warmup, while benchmark acceptance is exactly 18,000.

### P50 / P90 / P99

- bootstrap/ingress ACK: 2.240 / 5.002 / 9.523 s;
- model TTFT: 3.601 / 6.427 / 10.981 s;
- true global capacity queue: 0.747 / 1.661 / 4.141 s;
- admission RPC: 1.541 / 3.286 / 5.371 s;
- admission RPC residual: 0.025 / 1.665 / 3.588 s;
- LocalEngine command queue: 1.425 / 1.560 / 1.665 s;
- local batch admission: 0.070 / 0.152 / 0.290 s;
- TPOT with queue: 93.999 / 101.852 / 109.617 ms;
- weighted hierarchical decode ITL: 84.771 / 89.580 / 90.701 ms;
- E2E: 55.632 / 76.692 / 101.142 s.

Against the same-day `max_concurrency=64` per-request-RPC run, ACK P90/P99
fell by 91.0%/92.1%, admission-RPC-residual P90/P99 by 96.8%/96.1%,
TPOT-with-queue P90/P99 by 46.7%/45.2%, and decode ITL P90/P99 by
50.8%/51.5%. Total drain time fell 14.2%; TPOT-under-100-ms goodput increased
from 58.28% to 80.16%. ACK P50 increased from 1.20 s to 2.24 s because one
bounded batch commonly waits for the current decode quantum before the
single-writer loop picks it up.

### Remaining bottleneck

The request-scaled Ray mailbox collapse is fixed, but median admission still
waits roughly one decode quantum in the LocalEngine command queue. The next
control-plane optimization should target that pickup boundary or add a
capacity-change long poll; increasing actor concurrency is no longer useful.
As a rough execution-side reference, the new decode ITL is close to the
July 29 centralized TPOT (84.77/89.58/90.70 ms versus
83.62/87.83/90.61 ms); these are not identical metric boundaries. The
decentralized TPOT-with-queue remains higher because it now includes real
capacity waiting.

## 2026-07-30 — real-token final-quantum TPOT accounting

The comparison metric previously used the complete first-forward-to-terminal
wall interval. A request that finished after only part of its final 16-loop
quantum was therefore charged for all 16 decode slots. The legacy C++
`SequenceMetric::record_step_tokens()` ITL path instead divides each step by
the loop count and records samples only for tokens actually generated.

The request benchmark now applies that real-token method consistently to both
architectures:

- centralized runs subtract unused final slots using the last recorded
  step-ITL sample;
- hierarchical runs carry the final `executor.run` duration in `FinishEvent`
  and subtract `(16 - final_real_tokens) * execute_ms / 16`;
- true GPU-capacity queue time is retained in full, while raw
  `first_forward_to_terminal_ms` remains available for diagnostics.

New per-request fields are
`first_forward_to_terminal_real_token_ms`,
`final_quantum_real_tokens`, and `final_quantum_unused_decode_ms`.
`tpot_with_queue_ms` and goodput use the corrected real-token interval. A
17-token regression case verifies that a 160 ms final quantum contributes
10 ms and excludes the remaining 150 ms. Focused control-plane, scheduler,
serving-benchmark, and routing validation passed 82 tests; `py_compile` and
`git diff --check` also passed.

## 2026-07-30 — corrected rate-50 decentralized validation

The corrected `decentralized_dp2sp8` benchmark completed successfully against
Ray at `10.102.206.14:8776`: all 18,000 requests completed, with no failures,
ingress/scheduler rejections, fallbacks, queue-full events, preemptions,
tracebacks, CUDA errors, or NCCL errors. Runtime was 464.64 s. Across 464
diagnostic samples, admission stayed bounded at two concurrent Ray
RPCs (one per DP engine); a single RPC represented as many as 340 batched
requests. Capacity saturation caused 1,733 epoch-gated deferred retries but no
busy retry collapse.

All 18,000 request records use schema version 2 and contain the real-token
fields. The raw-minus-corrected terminal interval exactly matches
`final_quantum_unused_decode_ms` within 1 microsecond, and the recomputed TPOT
matches the recorded value within 0.000001 ms/token.

Corrected TPOT-with-capacity-queue P50/P90/P99 is
94.05/96.67/101.13 ms/token, with 98.53% of requests below 100 ms. Reapplying
the previous whole-final-quantum boundary to the same records gives
94.85/98.13/105.37 ms/token, so the correction removes
0.80/1.46/4.24 ms/token at those percentiles. The unused final-quantum time
itself is 596.16/1,137.93/1,319.79 ms per request.

Decode ITL is 85.40/88.51/89.65 ms and GPU-capacity queue time is
655.84/1,318.80/5,086.20 ms per request. Real execution-boundary time without
capacity queue is 92.55/94.86/95.45 ms/token. Against the July 29 centralized
TPOT of 83.62/87.83/90.61 ms/token, corrected decentralized TPOT remains
10.42/8.84/10.53 ms/token higher. The centralized baseline predates the new
final-quantum correction, so this is conservative rather than a perfectly
matched metric comparison. The close decode-ITL values, together with the
higher per-request execution boundary, show that most of the remaining gap is
real inter-quantum scheduling/coordination and capacity waiting, not the
removed final-token accounting artifact.

Artifacts are under
`bench_logs/rate50_decentralized_dp2sp8_real_token_tpot_20260730_0934/`.

## 2026-07-30 — rate-50 qdiag-off A/B

The same 18,000-request `decentralized_dp2sp8` command was rerun with
`HIERARCHICAL_QUANTUM_DIAGNOSTICS=0`; the normalized benchmark commands differ
only by the removed hierarchical quantum log option. The run completed all
requests without failures or rejections, and emitted zero quantum diagnostic
records.

Disabling qdiag did not improve performance. Runtime increased from 464.64 s
to 479.44 s. TPOT P50/P90/P99 changed from
94.05/96.67/101.13 to 96.82/111.28/125.86 ms/token, and sub-100-ms goodput
fell from 98.53% to 61.87%. Capacity-queue P50/P90/P99 increased from
655.84/1,318.80/5,086.20 to 939.26/4,861.86/9,036.11 ms/request.

Using actual output-token weighting, corrected TPOT increased by
6.408 ms/token (88.731 to 95.139). About 1.280 ms/token came from the
benchmark-window executor ITL increase, 3.381 ms/token from additional
first-forward-to-terminal non-executor gaps, and 1.746 ms/token from capacity
queue. The reported hierarchical ITL contains 65,280 pre-benchmark token
slots in both runs; removing the qdiag-on pre-window accumulator gives
81.376 versus an estimated 82.656 ms/token for the two benchmark windows.

The dominant change was admission retry amplification. Epoch-gated deferred
retries rose from 1,733 to 11,578, exactly accounting for the 9,845 extra
LocalEngine commands. Command-queue-delay total increased 64% and Gloo
coordination-wait total 58%. The retry RPC residual, which is not currently
included in `global_capacity_queue_ms`, rose from 175.78 to
1,275.47 ms/request on average.

Least-batch also exposed timing-sensitive KV imbalance: qdiag-off assigned
50.12M versus 39.52M prompt-plus-output tokens to the two engines despite
balancing request counts at 8,988 versus 9,012. The qdiag-on split was
47.41M versus 42.23M. This explains the later capacity pressure and much
higher deferral count, especially on engine 0.

The result rules out qdiag construction as the primary TPOT bottleneck but
does not establish that qdiag intrinsically improves performance. Removing
its inter-quantum delay perturbs least-batch placement and the capacity-event
pickup race, exposing an unstable admission feedback loop. The next
instrumentation should time the post-quantum-to-next-command pickup window;
the scheduler should balance predicted KV demand and avoid retrying a
256-request batch after only a small capacity release.

Artifacts are under
`bench_logs/rate50_decentralized_dp2sp8_real_token_tpot_no_qdiag_20260730_1141/`.

## 2026-07-30 — deterministic LB-side decentralized admission

The qdiag-off run showed that the old admission path still speculated: the
frontend balanced request counts, sent candidates to each LocalEngine, and
learned actual SP/KV infeasibility from `admission_deferred`. A deferred
request was removed from the local queue, returned through Ray, reinserted in
the global queue, and retried after a capacity epoch. That feedback loop
produced 11,578 retry attempts and made `admission_rpc_residual_ms` include
whole earlier attempts rather than only transport overhead.

The normal path now makes the capacity decision in `RequestRouter`:

- each DP LocalEngine attaches its aggregate per-SP master/receiver counts,
  dispatched-token load, free/total blocks, and control-dummy blocks to the
  existing consolidated frontend event RPC; SP ranks do not contact the
  frontend and no new Ray RPC is added;
- the LB reconstructs a versioned capacity shadow for each DP, examines only
  the global FIFO head, and evaluates DPs in projected-batch order using the
  same SP placement, receiver, KV reservation, lifetime, and queue limits as
  the LocalEngine planner;
- every accepted plan immediately deducts its placement from the LB shadow,
  so the rest of that Ray batch is planned against tentative capacity;
- the admission batch carries the selected master and per-SP token placement.
  LocalEngine validates current capacity and commits that placement verbatim;
  it does not run a second placement decision that prefix-cache reuse could
  steer to different ranks;
- if the FIFO head fits no DP, dispatch stops in place. No admission RPC is
  sent for that request and later requests cannot bypass it;
- only prevalidated batches are sent, still with at most one flight per DP
  and one `ray.wait` over all DP flights. `admission_state_mismatch`,
  legacy `admission_deferred`, and `queue_full` remain as resynchronization
  recovery and are counted separately as `state_mismatches`, rather than
  serving as the normal capacity probe.

Native rank ordering now breaks equal-free-block ties by SP index so the LB
mirror and LocalEngine choose the same deterministic placement. A load
snapshot that contains a local preemption queue is temporarily excluded from
new global admission until the LocalEngine clears its older local FIFO.

Verification after rebuilding the editable C++ extension:

- 87 focused control-plane, hierarchical contract, serving-ingress, and
  routing-config tests passed;
- regressions cover an infeasible FIFO head producing zero RPCs, no
  small-request bypass, tentative batch capacity deduction, and DP selection
  from aggregated receiver limits. A compiled C++ integration case confirms
  that the placements selected by the frontend are committed unchanged even
  when same-prefix KV blocks are reusable;
- a synthetic SP8 planner pass over 18,000 requests took 0.205 seconds
  (11.4 microseconds/request) on the frontend CPU;
- `py_compile` and `git diff --check` passed.

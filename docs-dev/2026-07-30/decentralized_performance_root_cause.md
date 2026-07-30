# Current decentralized performance root cause

Date: 2026-07-30 UTC

## Scope and constraints

This investigation is read-only with respect to runtime code. No C++ source was
edited, and no GPU, Ray-cluster, or multi-node experiment was launched.

The newest completed runs are:

- BS192:
  `bench_logs/rate50_decentralized_dp2sp8_lb_planned_admission_no_qdiag_1136a72_20260730_1343/`
- BS256:
  `bench_logs/rate50_decentralized_dp2sp8_bs256_lb_planned_no_qdiag_20260730_2243/`

The BS192 directory records runtime commit `1136a72`. The BS256 run was also
launched after `1136a72` and before any later runtime-code commit, but the
artifact itself does not record a SHA. There is no centralized run from exactly
the same runtime commit, so comparisons to the 2026-07-29 centralized run are
diagnostically useful but not a final controlled A/B.

Neither newest artifact records `SLIME_QP_NUM`. The launch script did not set
it, the code default is 1, and the repository's QP=4 experiment rule was added
after these runs. Therefore QP=1 is a strong inference, not an artifact-proven
fact.

## Executive conclusion

The current large regression is not caused primarily by Ray result RPC or
DLSlime payload transfer. It is a coupled control-plane and load-feedback
problem:

1. Planned admission is approximately
   `O(admission batch size * lifetime request count)`. Finished requests remain
   forever in `LocalScheduler._records`, and every new request scans all of
   them at least twice.
2. The LocalEngine drains the complete normal-command queue before the next
   decode quantum. Planned admission therefore stalls already-running requests.
   This time is inside their first-forward-to-terminal TPOT boundary but outside
   the existing admission/schedule/consensus/execute/postprocess phase timers.
3. Every 16-token quantum performs two serial CPU Gloo all-reduces. Their cost
   includes both communication and waiting for the slower LocalEngine, so
   admission and execution jitter are amplified into a DP-wide stall.
4. Longer quantum boundaries keep more requests alive. The larger active batch
   makes GPU/model execution nonlinear and increases DP rendezvous skew. That
   further lengthens request lifetime and adds global capacity queueing.
5. Ray/DLSlime and repeated Python batch construction add measurable secondary
   overhead, but their observed deltas are much too small to explain the total
   regression.

Thus, the larger batch seen in decentralized logs is mainly a consequence and
amplifier of slower service, not the original single cause.

## Result summary

| Run | Mean TPOT | P99 TPOT | Runtime | Mean E2E | Goodput at 100 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical central, rate 50, BS192 | 80.464 ms | 90.605 ms | 454.80 s | 48.19 s | effectively 100% |
| Current decentralized, rate 50, BS192 | 101.253 ms | 142.563 ms | 475.74 s | 62.38 s | 32.772% |
| Current decentralized, rate 50, BS256 | 108.627 ms | 139.373 ms | 487.80 s | 67.38 s | 26.011% |

For current BS192, request-level weighted accounting gives approximately:

- execution boundary: 94.610 ms/token;
- global capacity queue: 6.643 ms/token;
- total: 101.253 ms/token.

Relative to the historical centralized value, the approximate 20.789 ms/token
gap can be split as:

- executor ITL: +5.565 ms/token;
- non-executor execution-boundary residual: +8.583 ms/token;
- capacity queue: +6.640 ms/token.

This split crosses versions and metric schemas, so it is a prioritization
estimate rather than final causal attribution. It does show that no single
20-ms/token RPC tax exists.

## Root cause 1: lifetime-record scans make admission quadratic

`LocalScheduler._records` keeps terminal requests. Completion changes a
record's state to `FINISHED`, but does not remove it.

The current planned-admission path then repeatedly scans the lifetime table:

- `LocalScheduler._active_request_count()` scans all records:
  `nanodeploy/engine/local_scheduler.py:121-124`.
- `LocalScheduler.add()` calls it once for every new request:
  `nanodeploy/engine/local_scheduler.py:257-267`.
- `_planned_admission_fits()` recomputes active master and receiver counts by
  scanning all records for every request:
  `nanodeploy/engine/local_scheduler.py:384-426`.
- `commit_planned_batch()` invokes both operations per request:
  `nanodeploy/engine/local_scheduler.py:482-570`.
- `load_snapshot()` scans the lifetime table three times:
  `nanodeploy/engine/local_scheduler.py:964-1030`.

For a batch of `B` new requests after `N` historical requests, the dominant
admission work is `O(B*N)`. Across an 18,000-request run, cumulative work
approaches `O(total_requests^2)`.

### CPU reproduction on the current code

A no-GPU DP2/SP8 reproduction inserted terminal records into a real
`LocalScheduler`, then committed valid 32-request planned batches:

| Terminal records | Batch size | Commit time |
| ---: | ---: | ---: |
| 0 | 32 | 1.904 ms |
| 1,000 | 32 | 15.575 ms |
| 3,000 | 32 | 42.607 ms |
| 6,000 | 32 | 84.223 ms |
| 9,000 | 32 | 125.574 ms |

At a fixed 9,000 terminal records:

| Batch size | Commit time |
| ---: | ---: |
| 8 | 31.448 ms |
| 16 | 63.274 ms |
| 32 | 129.102 ms |
| 64 | 257.007 ms |

Both dimensions are linear, directly confirming `O(B*N)`. The 129-ms
reproduction also agrees with the production run's late-stage admission
latency.

A second no-GPU probe measured `load_snapshot()` with terminal-only history.
It grew from 0.029 ms at zero records to 4.115 ms at 9,000 records per call.
The active event loop refreshes the cached load up to three times per quantum,
so this is another growing inter-quantum tax.

### Production-log confirmation

Grouping requests that share an identical `local_admission_ms` reconstructs
the LocalEngine admission batches:

- earlier qdiag-on path: about 17.31 s total across both LocalEngines;
- current planned BS192: about 54.94 s total;
- current planned BS256: about 55.97 s total.

For current BS192, the mean reconstructed batch processing time grows as the
lifetime table grows:

| Dispatch interval | Mean time per reconstructed admission batch |
| --- | ---: |
| 0-60 s | 32.6 ms |
| 60-120 s | 85.4 ms |
| 120-180 s | 126.8 ms |
| 180-240 s | 140.7 ms |
| 240-300 s | 148.5 ms |
| 300-360 s | 161.5 ms |

The request-weighted `local_admission_ms` means are 148.529 ms for BS192 and
265.502 ms for BS256.

The current event-loop phase totals leave about 20.5 s per LocalEngine outside
the measured execute/consensus/schedule/postprocess/admit phases, or about
47.4 ms per quantum. Earlier qdiag-on data leave only about 1-2.5 s. The
increase is the same order as the reconstructed planned-admission CPU increase.

## Root cause 2: unbudgeted admission blocks decode and is counted in TPOT

The LocalEngine loop executes:

1. drain abort commands;
2. drain ingress;
3. drain all normal commands;
4. refresh cached load;
5. start the measured decode quantum.

The relevant loop is `nanodeploy/engine/local_engine.py:1026-1048`.
`_drain_queue()` has no request-count or time budget, and planned admission is
synchronously executed by
`nanodeploy/engine/local_engine.py:583-728`.

This placement matters for metric interpretation:

- a newly arriving request's pre-first-forward admission is excluded from its
  execution TPOT;
- the same CPU work pauses every already-running request between two token
  quanta, so it is included in those requests' first-forward-to-terminal TPOT;
- it is outside `quantum_begin`, so current phase diagnostics do not attribute
  it.

This explains why the latest run has a large "boundary residual" even though
worker execution and RPC return deltas are modest.

## Root cause 3: two Gloo collectives per 16-token quantum

`LocalEngineCore._consensus()` creates a CPU tensor and performs:

1. `all_reduce(MIN)`;
2. `all_reduce(MAX)`.

See `nanodeploy/engine/local_engine.py:956-976`.

The synchronization is necessary because the EP16/model collectives require
both DP groups to enter compatible forward steps. It cannot simply be removed.
However, two serialized collectives are unnecessary for DP2: one all-gather
can validate wave/quantum equality and compute unfinished OR.

Observed cost:

- full-run BS192: 26.40 ms per engine quantum on average;
- full-run BS256: 43.45 ms per engine quantum on average;
- steady-state 60-360 s: about 48.42 ms/q for BS192 and 94.34 ms/q for
  BS256, equivalent to 3.03 and 5.90 ms/token.

These numbers are rendezvous time, not pure Gloo wire time. Faster engines wait
for the slower engine, so CPU admission stalls, batch imbalance, and GPU launch
jitter all inflate this metric.

## Root cause 4: slower service creates a batch/latency feedback loop

Across the newest runs, interval batch size and executor ITL have correlations
of about 0.99. The current BS256 run demonstrates the feedback directly:

- capacity contribution decreases from 6.643 to 3.337 ms/token;
- execution contribution increases from 94.610 to 105.290 ms/token;
- executor ITL increases from 82.885 to 89.877 ms/token;
- net TPOT becomes 7.374 ms/token worse;
- minimum observed free blocks fall to 691, close to KV pressure;
- no preemption explains the result.

The late BS256 interval reaches roughly 3,832 aggregate active requests and
103.41-ms executor ITL. Increasing BS from 192 to 256 therefore trades a
smaller frontend queue for a much worse GPU/synchronization regime.

The current BS192 placement is already well balanced by output work:

- two-engine output-token CV is about 0.09%;
- 16-rank mastered-decode-token CV is about 0.50%.

Prompt-plus-output placement is less balanced, about 6.03% CV across the two
engines, so KV pressure can still amplify queueing. It does not explain the
primary decode-output slowdown.

## Confirmed capacity-planner bug

The frontend admission shadow double-counts current-batch master reservations
on the legacy planning path:

- `apply_reservation()` increments both `master_counts` and, for the current
  batch, `batch_master_counts`:
  `nanodeploy/router/admission_planner.py:164-192`;
- `_check_legacy_placement()` later sums both:
  `nanodeploy/router/admission_planner.py:327-358`.

Thus each prior reservation in the same frontend batch is counted twice when
checking reserved master blocks. This makes the planner over-conservative near
capacity and can add global capacity queueing. It is a real correctness/
efficiency bug, but current free-block traces do not support treating it as the
sole cause of the 20.8-ms/token regression.

## What Ray and DLSlime actually contribute

Both scheduler architectures use the same hybrid transport:

- Ray actor RPC triggers `worker.run.remote(...)`;
- DLSlime `send_seqs()` carries sequence payloads;
- Ray returns token results.

The decentralized path is in
`nanodeploy/engine/local_executor.py:99-205`; the centralized path is in
`nanodeploy/engine/ray_executor.py:243-329`.

Latest BS192 means:

| Boundary stage | Central | Decentralized |
| --- | ---: | ---: |
| actor submit | 1.433 ms/q | 5.977 ms/q |
| DLSlime `send_seqs` | 19.803 ms/q | 23.392 ms/q |
| worker-observed critical | 1007.626 ms/q | 1022.483 ms/q |
| worker finish to `ray.get` return | 2.553 ms/q | 2.825 ms/q |

Important interpretation:

- `ray_get` near one second is almost entirely waiting for model execution.
  The pure result-return tail is only about 2.8 ms/q and is nearly unchanged.
- `actor_submit` is misnamed as a pure RPC metric. Before submitting, the
  decentralized executor repeatedly filters the full batch by eight ranks:
  `nanodeploy/engine/local_executor.py:104-154`.
- The two LocalEngines sent about 1% more total serialized payload than the
  centralized run, not 2x. Their sends are concurrent and must not be summed.
- `send_seqs` does have 58-68-ms small-message outliers consistent with delayed
  receive posting/RNR or CPU scheduling, but its central-to-decentral delta is
  only about 3.6 ms/q, around 0.22 ms/token.
- Result rebuild is roughly 0.03 ms/token and is not worth prioritizing.

Therefore Ray/DLSlime is a secondary optimization area, not the current root
cause.

## Architectural baseline versus the current regression

The cleanest existing low-load comparison is rate 20, BS192:

- central TPOT: 48.823 ms/token;
- decentralized TPOT: 50.589 ms/token;
- difference: +1.766 ms/token;
- capacity queue is negligible in both.

That approximately 3.6% cost is the genuine baseline architecture overhead:
slightly slower worker boundary plus per-quantum control work.

The old rate-50 decentralized result showing 79.362 ms/token versus central
80.464 ms/token is not evidence that decentralized was faster. Its frontend
could not dispatch the offered load:

- throughput was 6.26% lower;
- runtime was 6.68% longer;
- mean arrival E2E was 72.72 s versus 48.82 s;
- dispatch/ACK stalls smoothed the GPU input and artificially lowered execution
  TPOT.

## Fix order

All first-line fixes can be Python-only:

1. Make planned admission `O(active + batch)`:
   compute active master/receiver counts once per batch, update them
   incrementally, and maintain an incremental active-request count.
2. Move terminal records out of active hot-path storage. Preserve duplicate-ID
   semantics with a compact tombstone/seen-ID structure.
3. Maintain waiting/running/master/receiver/token load counters incrementally so
   `load_snapshot()` does not scan lifetime history.
4. Give normal-command admission drain a request-count or time budget so it
   cannot starve the next decode quantum.
5. Fix frontend `master_counts`/`batch_master_counts` double counting and add a
   capacity-boundary CPU regression test.
6. Fuse the two Gloo reductions into one DP2 all-gather while preserving
   wave/quantum validation and global unfinished semantics.
7. Cache per-rank mastered sequences and result indices in
   `LocalDecodeBatch`; reuse them in executor and postprocess.
8. Only after these, investigate receive pre-posting and QP1/QP4 DLSlime A/B.

No C++ change is required for the first validation. A later C++ batch fast path
may be useful, but should be considered only after the Python complexity
regression is removed.

## Minimum controlled validation

Before claiming the final central/decentral gap:

1. record clean git SHA, dirty state, compiled-extension build identity,
   `SLIME_QP_NUM`, NIC choice, and Gloo interface in every artifact;
2. set `SLIME_QP_NUM=4` as required by `AGENTS.md`;
3. use one clean SHA/build, BS192, qdiag off, identical dataset and routing;
4. run central/decentral at rate 20 and rate 50, at least twice each;
5. retain per-quantum execute, consensus, admission-drain, inter-quantum gap,
   active history size, admission batch size, and per-rank token/KV load.

The rate-20 pair verifies the approximately 1.8-ms/token architecture floor.
The rate-50 pair verifies whether removing quadratic admission prevents the
batch/queue feedback loop.

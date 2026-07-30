# Development progress

## 2026-07-29 — decentralized admission redesign

### Goal

Make hierarchical/decentralized scheduling follow the same logical split as
the legacy centralized scheduler:

- the frontend load balancer owns one global pending-admission queue and
  performs DP/node selection;
- each `LocalScheduler` owns only the selected node's SP placement and
  execution work;
- `least_batch` selection uses live running/admitted load plus tentative
  admissions, rather than sticky request-owner counts over the entire request
  lifetime;
- ownership is assigned only after local admission/ingress is accepted.

This first change targets centralized-compatible `least_batch` semantics.
Cost-aware routing is intentionally deferred, and the inner SP selector should
remain `LeastBatch`.

### Recovery point

- Branch: `decentralized-july`
- Baseline commit before this development:
  `ed1f63c010dee2041a21ce03fce149137cca64bc`
- The worktree was already dirty at the start. Preserve all pre-existing
  modified and untracked files; do not use a destructive reset.

### Findings

- `RequestRouter.submit_async()` currently chooses an engine immediately and
  stores `PENDING_INGRESS` with a sticky `engine_id`.
- Hierarchical `least_batch` currently sorts by `_owner_counts()`, which
  includes requests for their full lifetime and therefore differs from the
  centralized scheduler.
- The centralized C++ scheduler keeps a single pending queue. For each pending
  request it orders DP candidates by current running sequences plus tentative
  placements in the same scheduling pass, then asks the selected DP/SP planner
  for feasibility before committing placement.
- `LocalScheduler` already owns the SP-specific planner and must stay
  single-writer inside the `LocalEngine` loop.

### Evidence from the preceding experiments

- In the comparable DP2/SP8, rate-40, six-minute run, decentralized TPOT was
  about 6.5% slower although total prompt/output work matched.
- The decentralized engines had about 23% prompt-work skew; the busier engine
  showed higher KV occupancy and decode ITL.
- External/control-plane overhead was not the dominant difference.
- Outer arrival-time `least_batch` routing was strongly phase-locked to input
  ordering. Switching the inner SP selector to `LeastCache` made the result
  worse, so the inner selector should remain `LeastBatch`.

### Current status

- Baseline recorded.
- Relevant router, LLM frontend, local scheduler, local engine, deployment
  transport, centralized C++ scheduler, and control-plane tests inspected.
- The first centralized-compatible `least_batch` implementation is complete:
  - `PENDING_GLOBAL` keeps requests unowned before DP selection.
  - `RequestRouter` drains one global admission queue using cached live
    `running` plus versioned tentative admission counts, with DP id as the
    centralized tie-breaker.
  - `LocalEngineCore.admit_add()` executes inside the scheduler's single-writer
    loop. Admissions from one command drain are combined into one local batch.
  - `LocalScheduler.try_admit_batch()` reuses the existing C++ SP planner,
    commits feasible requests, and removes transiently infeasible candidates
    from local waiting state so the global coordinator can fall back or retry.
  - Successful ACKs carry a monotonic admission version. Router-side tentative
    charges are removed only after an authoritative load snapshot includes
    that version.
  - Queue-full and SP/KV-deferred candidates fall back across DPs. If every DP
    defers, the request returns to the global FIFO head and waits for a changed
    load generation instead of spinning.
  - Admission attempts, commits, deferrals, queue-full responses, fallbacks,
    global retries, and per-engine projected batches are exposed through
    hierarchical metrics.
- `round_robin` and `least_cache` retain their existing ingress behavior. This
  change intentionally targets `least_batch` first.
- No C++ source was changed, so an editable reinstall was not required.

### Verification

- `tests/test_hierarchical_control_plane.py`,
  `tests/test_hierarchical_contract.py`, and
  `tests/test_hierarchical_serving_ingress.py`: 64 passed.
- Remaining explicit CPU suites
  (`test_decode_rpc_optimization.py`, `test_routing_config.py`,
  `test_sp_backend.py`): 25 passed.
- `python -m compileall -q nanodeploy tests`: passed.
- `git diff --check`: passed.
- A blanket pytest collection reaches the pre-existing `tests/test_fa.py`,
  which initializes CUDA at import time. It was not run because GPU operations
  require explicit elevation.

### Six-minute DP2/SP8 E2E result

The targeted two-node run completed successfully:

- Run tag:
  `global_admission_dp2sp8_r40_6min_lb_lb_qdiag_20260729_0443`
- Configuration: DP=2, SP=8, 16 GPUs, rate=40, 360-second arrival
  window, 14,400 requests, `ROUTING_STRATEGY=LeastBatch`,
  `ROUTER_POLICY=least_batch`, `SP_MASTER_SELECTOR=LeastBatch`, quantum
  diagnostics enabled.
- Dataset work closed exactly at 66,382,719 prompt tokens and 8,644,024
  output tokens. All 14,400 requests finished and none failed.
- Benchmark runtime, including tail drain, was 450.884 seconds.
- Mean E2E was 42,100.402 ms; p50/p90/p95/p99 were
  41,778.843/56,862.754/61,996.704/74,935.728 ms.
- Mean queue-inclusive TPOT was 70.462 ms. The originally reported
  1,670.134 ms "TTFT" was the first model-token observation and did not use
  the same bootstrap-ready endpoint as the legacy centralized metric.

Admission behavior was clean for the whole run:

- engine admission commits were 7,331 vs 7,325;
- deferrals, queue-full responses, fallbacks, global retries, ingress
  rejections, and request failures were all zero;
- at representative load points, running-count splits were 758/758 at 73
  seconds, 902/900 at 300 seconds, and the final cumulative commit split
  differed by only six requests;
- local waiting stayed at zero.

Compared with the prior qdiag-enabled decentralized LeastBatch run
(`decentral_dp2sp8_r40_6min_qdiag_20260728_retry1`):

- runtime improved by 3.932 seconds (0.864%);
- mean E2E improved by 427.314 ms (1.005%);
- p90/p95 E2E improved by 790.690/1,022.873 ms
  (1.371%/1.623%);
- mean queue-inclusive TPOT improved by 0.709 ms (0.996%);
- internal weighted hierarchical decode ITL improved from 66.047 to
  65.091 ms (1.45%);
- steady-window paired-engine running-count imbalance was essentially
  unchanged (mean absolute difference 5.25 before vs 5.29 now), but mean
  minimum-free-block separation narrowed from about 2,289 to 1,286 blocks.

Compared with the centralized DP2/SP8 run from
`dp2sp8_r40_6min_maxreq910k_20260728_072713`:

- total runtime is now almost closed: 450.884 vs 449.724 seconds
  (0.258% slower);
- mean E2E remains 2,060.949 ms (5.147%) slower;
- mean queue-inclusive TPOT remains 3.622 ms (5.419%) slower;
- the p50/p90/p95/p99 E2E gaps are 5.119%/4.566%/4.221%/4.003%;
- the old centralized and hierarchical TTFT values are not directly
  comparable: legacy dummy-prefill records the bootstrap token, while the
  original hierarchical benchmark observed the first model-generated token.
  The earlier attribution of 80.9% of the E2E gap to TTFT is therefore
  withdrawn.

The admission ACK latency changed meaning in this implementation: it now
waits for authoritative local planner admission, so its mean rose from about
5.45 ms to 603.79 ms. This is observability of the admission barrier, not
additional E2E regression: E2E and TPOT both improved, while the old
router-to-local ingress queue time moved into admission.

Per-engine completed request counts and output tokens are nearly equal
(7,203/7,197 requests and 4,323,820/4,320,204 output tokens), but LeastBatch
does not balance total prompt tokens. This run assigned
37,794,842/28,587,877 prompt tokens. That is consistent with the centralized
LeastBatch definition, which selects by running sequence count rather than
prompt-token cost; it is a separate cost-aware-routing issue.

The result supports two conclusions:

1. The global admission/node-selection split is logically correct and stable,
   and yields a reproducible but modest performance improvement.
2. The remaining centralized-vs-hierarchical gap is not explained by DP
   request-count imbalance or external control-plane overhead. TTFT must use
   a common endpoint before it can be used to attribute the remaining gap.

### Bootstrap-compatible TTFT change and rerun

The benchmark now reports two separate hierarchical metrics without changing
runtime scheduling:

- `ttft_ms` and `bootstrap_ttft_ms` are T3-T1: actual dispatch until the
  authoritative local admission ACK produced after bootstrap placement. This
  matches the legacy centralized dummy-prefill endpoint.
- `model_ttft_ms` is T4-T1: actual dispatch until the benchmark observes the
  first model-generated token.
- Each request records `ttft_source`; centralized-compatible hierarchical
  admission uses `authoritative_admission_ack`.

This change only touched Python benchmark accounting and its CPU test. No C++
source changed and no reinstall was required. The relevant CPU suites passed:
65 control-plane/contract/serving tests and 25 remaining explicit CPU tests.
Compileall and `git diff --check` also passed.

The same two-node six-minute workload was rerun under:
`ttft_t3_t1_dp2sp8_r40_6min_lb_lb_qdiag_20260729_0646`.

- 14,400/14,400 requests completed, with zero failures, rejections,
  deferrals, queue-full responses, fallbacks, or global retries.
- Dataset work again closed exactly at 66,382,719 prompt tokens and
  8,644,024 output tokens.
- All 14,400 records used `authoritative_admission_ack`; no bootstrap or
  model TTFT sample was missing.
- Bootstrap-compatible TTFT mean/p50/p90/p95/p99:
  601.883/590.220/1,053.235/1,125.134/1,211.361 ms.
- Model TTFT mean/p50/p90/p95/p99:
  1,667.718/1,663.961/2,160.328/2,254.301/2,392.759 ms.
- Mean E2E was 42,080.915 ms, mean queue-inclusive TPOT was 70.437 ms,
  and internal weighted decode ITL was 65.089 ms.
- Runtime was 451.911 seconds and admission commits were 7,329/7,327.

The accounting-only change did not alter performance: versus the immediately
preceding decentralized run, mean E2E changed by -19.487 ms (-0.046%), mean
TPOT by -0.025 ms (-0.035%), and internal decode ITL by -0.002 ms.

With the common T3-T1 endpoint, centralized TTFT is 2.587 ms and the new
hierarchical TTFT is 601.883 ms. This difference is now semantically valid for
server-side dispatch-to-bootstrap latency. The separate T0-T1 arrival/dispatch
effect is intentionally deferred to the next investigation.

### Next steps

1. Discuss and define the T0-T1 arrival/dispatch comparison before attributing
   the remaining latency difference.
2. If server-side bootstrap latency is the next target, split T1-T3 into
   command-queue wait and local admission work before changing quantum size.
3. Keep cost-aware DP routing as a separate experiment. LeastBatch now matches
   centralized semantics, but it cannot by definition balance prompt-token or
   KV work.

### Worktree note

The worktree still contains substantial pre-existing, unrelated experiment and
diagnostic changes, including overlapping edits in several files touched by
this implementation. No commit was created automatically because a path-level
commit would also capture those earlier changes. The rollback baseline remains
`ed1f63c010dee2041a21ce03fce149137cca64bc`.

## 2026-07-29 07:28:56 UTC — GPU-capacity queue metric

The requested queue definition was narrowed: only time during which a request
is eligible for scheduling but cannot be placed because of DP/SP/KV/GPU
capacity should count. The initial wait for a LocalEngine to finish its
previous 16-step quantum and pick up the admission command must not count.

The Python control plane and benchmark now expose:

- `global_capacity_queue_ms`: time spent in the global admission FIFO after
  all candidate LocalEngines have explicitly returned transient capacity
  failure. Admission RPC time is paused/excluded.
- `local_scheduler_queue_ms`: time from successful insertion into
  `LocalScheduler` until immediately before the first `executor.run` that
  contains the request.
- `first_schedule_latency_ms`: the sum of those two components.

`FirstScheduleEvent` is emitted before `executor.run`, routed back through
`DeploymentManager` and `LLMEngine`, and persisted per request and in summary
percentiles by `scripts/sp_ablation/bench_serving_overhead.py`. For legacy
centralized scheduling, the same output field uses the existing
`SequenceMetric.queueing_time_ms`; hierarchical runs do not silently fall back
to that older metric if the new event is missing.

The implementation changed Python only. No C++ files changed, so reinstalling
the editable package was unnecessary. Validation completed:

- 65 hierarchical contract/control-plane/serving tests passed;
- 12 routing/sequence-proxy/decode-RPC CPU tests passed;
- `compileall` and `git diff --check` passed.

At implementation time, no prior GPU log could reconstruct this metric
exactly because those logs did not contain the new first-forward event.

### GPU-capacity queue metric six-minute validation

The same DP=2, SP=8, 16-GPU, rate-40 workload was run under:
`first_schedule_dp2sp8_r40_6min_lb_lb_qdiag_20260729_0730`.

- Configuration remained `ROUTING_STRATEGY=LeastBatch`,
  `ROUTER_POLICY=least_batch`, and `SP_MASTER_SELECTOR=LeastBatch`, with
  quantum diagnostics enabled.
- All 14,400 requests completed, with zero request failures, ingress or
  scheduler rejections, queue-full responses, and preemptions. Runtime,
  including tail drain, was 453.06 seconds.
- The dataset again closed at exactly 66,382,719 prompt tokens and 8,644,024
  output tokens.
- The per-request JSONL contains exactly 14,400 records. Every record contains
  `first_schedule_latency_ms`, `global_capacity_queue_ms`, and
  `local_scheduler_queue_ms`.
- First-schedule latency mean/p50/p90/p95/p99/max was
  32.040/25.510/60.537/71.659/111.370/332.294 ms.
- Every request had `global_capacity_queue_ms=0`. Consequently the
  first-schedule distribution is identical to the local-scheduler component.
- There were 73 transient admission deferrals on engine 0. Every one could
  immediately fall back to engine 1, so none entered the global capacity FIFO;
  global retries and final global pending depth remained zero.

The new result distinguishes three intervals that were previously conflated:

1. Bootstrap-compatible T3-T1 remained much larger at 746.469 ms mean because
   it still includes waiting for the LocalEngine to pick up the admission
   command.
2. After successful local insertion, the request reached its first real
   `executor.run` in 32.040 ms mean.
3. No request waited because both DP engines lacked GPU/KV scheduling
   capacity in this workload, so the user-defined capacity queue time was
   exactly zero.

Mean E2E and queue-inclusive TPOT were 42,185.229 ms and 70.646 ms. Versus
the immediately preceding hierarchical run, these changed by +104.314 ms
(+0.248%) and +0.209 ms (+0.297%), while weighted internal decode ITL changed
from 65.089 to 64.961 ms. One rerun cannot separate this small end-to-end
difference from run-to-run variance. Bootstrap-compatible and model TTFT rose
to 746.469 and 1,807.517 ms respectively, but the first-schedule measurement
shows only 32.040 ms after insertion and zero true global capacity queue.

## 2026-07-29 08:21:48 UTC — Default TPOT scheduling-boundary correction

The default benchmark TPOT boundary was changed to match the requested queue
semantics. `tpot_with_queue_ms` no longer uses benchmark-observed
`dispatch -> completion / generated tokens`.

The new default is:

- centralized:
  `SequenceMetric.avg_tpot_with_queueing`, which is local scheduler arrival
  through local completion and therefore contains the centralized scheduler's
  capacity wait but not benchmark dispatch lag;
- hierarchical:
  `(first_forward_to_terminal_ms + global_capacity_queue_ms) /
  generated_tokens`.

`LocalRequestRecord` now retains the `perf_counter()` timestamp immediately
before the request's first `executor.run`. `FinishEvent` carries the exact
same-process `first_forward_to_terminal_ms`. `RequestRouter.finish()` decorates
the terminal event with the accumulated global capacity queue time before
`LLMEngine.poll()` returns it to the benchmark. This avoids reconstructing the
boundary from admission ACK delivery or benchmark polling timestamps.

The previous metric remains available as
`dispatch_normalized_latency_ms`, with the explicit definition
`benchmark-observed dispatch -> completion / generated tokens`. It is emitted
per request, summarized, included in `BENCH_RESULT`, and printed in a separate
legacy diagnostic section. Default goodput now uses the corrected
`tpot_with_queue_ms`.

Human-readable output now:

- prints corrected TPOT percentiles under the existing
  `TPOT With Queueing Time` section so downstream section parsers continue to
  work;
- states that queueing means GPU-capacity queue only;
- prints the legacy dispatch-normalized distribution separately;
- uses the request-summary TTFT/E2E values rather than a conflicting
  secondary sequence-metric aggregate.

This implementation changed Python only; no C++ source changed and no package
reinstall was required. Validation completed:

- 66 hierarchical contract/control-plane/serving tests passed, including
  explicit centralized and hierarchical default-TPOT source tests;
- 12 routing/sequence-proxy/decode-RPC tests passed;
- `compileall` and `git diff --check` passed;
- `scripts/sp_ablation/bench_serving_overhead.py` contains neither `getattr`
  nor `hasattr`.

No GPU rerun has been performed with the corrected default output yet. Existing
six-minute logs retain their historical `tpot_with_queue_ms` values; the exact
new hierarchical percentiles require the new terminal-duration event and
therefore must come from a new run. The rollback baseline remains
`ed1f63c010dee2041a21ce03fce149137cca64bc`.

## 2026-07-29 08:36:39 UTC — Corrected default TPOT six-minute validation

The corrected metric was validated with the same DP=2, SP=8, 16-GPU,
rate-40 workload under
`adjusted_tpot_dp2sp8_r40_6min_lb_lb_qdiag_20260729_0825`.
Routing remained `LeastBatch` at the load balancer and `least_batch` at the
hierarchical router/local-scheduler selection boundary.

- All 14,400 requests completed with zero failures, rejections, deferrals,
  queue-full responses, fallbacks, global retries, or preemptions.
- Both local engines were balanced: 7,227 and 7,173 completed requests.
- The per-request JSONL has exactly 14,400 records. Every record has a non-null
  `first_forward_to_terminal_ms`, `global_capacity_queue_ms`,
  `dispatch_normalized_latency_ms`, and corrected `tpot_with_queue_ms`.
  All 14,400 corrected values report
  `tpot_with_queue_source=hierarchical_first_forward_to_terminal`.
- No request encountered GPU/KV capacity pressure on both DP engines, so
  `global_capacity_queue_ms` is exactly zero for the full distribution.
- Corrected TPOT mean/p50/p90/p95/p99/max is
  68.720/69.644/74.459/75.179/76.878/85.345 ms/token.
- The retained legacy dispatch-normalized distribution is
  69.898/70.855/75.797/76.799/80.336/97.003 ms/token. The corrected boundary
  therefore removes 1.178 ms/token from the mean in this run.
- Weighted hierarchical executor decode ITL is 64.533 ms/token, while
  first-forward-to-terminal time is 41,154.398 ms mean.
- Benchmark runtime including tail drain was 451.677 seconds, and corrected
  TPOT goodput below 100 ms/token was 14,400/14,400.

The comparable centralized C++ scheduler-boundary TPOT from
`dp2sp8_r40_6min_maxreq910k_20260728_072713` is
66.810/66.750/72.840/73.540/74.920 ms/token at
mean/p50/p90/p95/p99. The corrected hierarchical result is therefore higher by
1.910/2.894/1.619/1.639/1.958 ms/token respectively. Its mean weighted
executor decode ITL, 64.533 ms/token, is only 0.073 ms/token above the
centralized 64.460 ms/token. This localizes most of the remaining mean TPOT
gap to time between executor quantums/local completion boundaries rather than
GPU model execution itself or external dispatch/RPC pickup before first
forward.

The request summary is:
`bench_logs/adjusted_tpot_dp2sp8_r40_6min_lb_lb_qdiag_20260729_0825/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n14400_r40_bs192_LB_hier_rpLB_maxreq910000_hao_basic_qdiag_adjusted_tpot_lb_lb_rate40_6min_qdiag_decentralized_dp2sp8/20260729_162546.summary.json`.

## 2026-07-29 09:29:09 UTC — Rate-20 and rate-50 architecture comparison

Four DP=2, SP=8 six-minute runs used the same `LeastBatch` load-balancer,
`least_batch` hierarchical router, and `LeastBatch` SP-master policy. The
rate-20 pair each processed 7,200 identical requests containing 35,631,781
prompt and 4,313,369 output tokens. The rate-50 pair each processed 18,000
identical requests containing 78,836,913 prompt and 10,801,614 output tokens.
All four successful runs completed every request without a request failure or
rejection.

Corrected TPOT results in ms/token:

| Rate | Architecture | Mean | P50 | P90 | P95 | P99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | centralized | 48.823 | 49.051 | 50.858 | 51.293 | 52.446 |
| 20 | hierarchical | 50.589 | 50.669 | 52.468 | 53.293 | 54.529 |
| 50 | centralized | 80.464 | 83.624 | 87.834 | 88.710 | 90.605 |
| 50 | hierarchical | 79.362 | 81.611 | 84.785 | 85.387 | 87.265 |

At rate 20, hierarchical TPOT is higher by
1.766/1.618/1.610/2.000/2.083 ms/token at
mean/p50/p90/p95/p99. Its weighted executor decode ITL is 48.568 ms/token,
versus centralized ITL-with-decode-queue at 47.440 ms/token. Mean scheduled
arrival-to-completion latency is 30,684.870 ms hierarchical versus
29,591.549 ms centralized. Hierarchical GPU-capacity queue time is exactly
zero; centralized local scheduler queue time averages 0.445 ms/request.

At rate 50, the corrected first-forward-to-terminal TPOT appears lower for
hierarchical by 1.102/2.013/3.049/3.323/3.340 ms/token. This is not evidence
that hierarchical sustains rate 50 better. The hierarchical frontend could
not dispatch the scheduled arrivals at 50 requests/s:

- hierarchical dispatch lag mean/p50/p90/p95/p99 was
  7,691.649/2,753.870/23,379.108/30,476.484/41,663.092 ms, with a
  44,864.905 ms maximum;
- centralized dispatch lag was
  628.658/615.337/1,158.988/1,247.504/1,343.011 ms;
- hierarchical ingress-ACK latency averaged 17,136.263 ms and ADD acceptance
  averaged 31,548.608 ms, despite zero GPU-capacity queue time;
- hierarchical scheduled arrival-to-completion latency averaged
  72,716.960 ms, versus 48,819.811 ms centralized;
- benchmark runtime including tail drain was 485.166 seconds hierarchical
  versus 454.804 seconds centralized.

Consequently the rate-50 hierarchical executor saw a temporally smoothed,
lower effective load during part of the scheduled arrival window. Its
weighted executor ITL was 73.905 ms/token, below centralized
ITL-with-decode-queue at 77.320 ms/token, which explains the apparently better
corrected TPOT. The end-to-end result shows the actual problem: the
hierarchical async admission/command-pickup path saturates before GPU/KV
capacity admission reports both DPs blocked. Optimizing only the
first-forward-to-terminal interval would miss this bottleneck.

### Rate-50 early-terminal race and repair

The first hierarchical rate-50 attempt stopped after 8,331 completions with:
`terminal event owner mismatch: request=10927, engine=1,
owner=PENDING_INGRESS`. A short request could finish locally before the
frontend observed its admission ACK and ADD result. This is valid
cross-actor event reordering, not an engine ownership change or GPU OOM.

`RequestRouter.record_finish_events()` now buffers a terminal event received
while the matching owner is `PENDING_INGRESS` or `PENDING_ADD`, then performs
the existing strict owner validation and terminal accounting after the owner
becomes `OWNED`. Mismatched engines and invalid owner transitions still fail
strictly. `LLMEngine.poll()` now consumes these ordered terminal events.
A regression test reproduces terminal-before-ACK-before-ADD ordering. All 79
related tests passed, along with `compileall` and `git diff --check`. This
repair changed Python only and required no package reinstall.

Successful summaries:

- rate-20 centralized:
  `bench_logs/tpot_compare_r20_dp2sp8_6min_lb_lb_20260729_0839/centralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n7200_r20_bs192_LB_legacy_rpLB_maxreq910000_hao_basic_tpot_compare_r20_6min_lb_lb_centralized_dp2sp8/20260729_163849.summary.json`;
- rate-20 hierarchical:
  `bench_logs/tpot_compare_r20_decentralized_dp2sp8_6min_lb_lb_20260729_0848/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n7200_r20_bs192_LB_hier_rpLB_maxreq910000_hao_basic_qdiag_tpot_compare_r20_6min_lb_lb_decentralized_dp2sp8/20260729_164827.summary.json`;
- rate-50 centralized:
  `bench_logs/tpot_compare_r50_centralized_dp2sp8_6min_lb_lb_20260729_0857/centralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n18000_r50_bs192_LB_legacy_rpLB_maxreq910000_hao_basic_tpot_compare_r50_6min_lb_lb_centralized_dp2sp8/20260729_165749.summary.json`;
- rate-50 hierarchical:
  `bench_logs/tpot_compare_r50_decentralized_dp2sp8_6min_lb_lb_retry_20260729_0915/decentralized_dp2sp8/deepseek-v3/sharegpt4o-random_geminiissue_r0.01_n60000_60k/dp2sp8_seg64k_n18000_r50_bs192_LB_hier_rpLB_maxreq910000_hao_basic_qdiag_tpot_compare_r50_6min_lb_lb_retry_decentralized_dp2sp8/20260729_171655.summary.json`.

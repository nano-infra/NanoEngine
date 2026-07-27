# DP2SP8 Hierarchical Rate-20 Benchmark

Started: 2026-07-27 06:30:13 UTC

Stopped: 2026-07-27 06:40:36 UTC, at the user's request

## Goal

Run the two-node DeepSeek-V3 hierarchical benchmark against
`10.102.206.14:7789` with 7,200 requests arriving at 20 requests/s.
The arrival window is nominally six minutes; total wall time also includes the
fixed warmup and draining all requests after the final arrival.

## Configuration

- Topology: `DP2 SP8 TP1 EP16`
- Scheduler: `hierarchical`
- Routing: `RoundRobin`
- Loop count: `16`
- SP backend: `hao_basic`
- Execution: CUDA Graph, `full` mode; eager is disabled
- Batch size: `192`
- GPU memory utilization: `0.9`
- Dataset:
  `/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv`
- Benchmark artifacts:
  `bench_logs/hierarchical_dp2sp8_rate20_6min`

## Logging

- `start_bench.sh` enables unbuffered output and `RAY_DEDUP_LOGS=0`.
- The benchmark log records model configuration, per-rank input-transfer
  latency, DLSlime send overhead, warmup timing, request throughput, useful
  completion-token throughput, TTFT/E2E, TPOT/ITL percentiles, queue latency,
  decode-queue latency, and goodput.
- Per-request ITL samples are written as JSON Lines by the benchmark.
- The full outer console stream is duplicated to `console.log` in this
  directory.
- Hierarchical execution traces and decode A2A mask dumps remain disabled
  because their volume and synchronization overhead would contaminate the
  performance result.

## Observed data

- CUDA Graph capture succeeded on all 16 ranks. Each rank reported 16 local
  and 48 SP graphs (64 total).
- The fixed 256-request warmup completed in 7.51 seconds, or 34.08 requests/s.
- The formal workload started at 06:34:10 UTC and ran for approximately
  386 seconds before interruption.
- The last rendered client progress was 56/7,200 at 50.44 seconds, with
  35.46 seconds average E2E latency. This is not a trustworthy final
  completion count because the interrupted client did not emit a final
  summary or ITL JSON.
- Every rank logged about 971 worker `run()` calls; the two LocalEngine actors
  logged 970 and 971 DLSlime sends. The ranks were still executing at the
  time of interruption, so this was not a CUDA Graph crash or dead worker.
- No traceback, CUDA error, OOM, queue-full rejection, or C++ preemption line
  appeared in the captured log.

## Diagnostic gap and follow-up

The client previously exposed progress only through `tqdm`. It did not
periodically record submitted/accepted/rejected/completed counts, ADD latency,
or the hierarchical load snapshots that already contain waiting/running
counts, useful tokens, preemptions, quantum counts, and stage timings.
Consequently, the existing log cannot distinguish a stale client display from
ADD-side blocking or scheduler backlog.

Low-overhead structured diagnostics were added for the next reproduction:

- `[BENCH_DIAG]` once per configured interval, including client counters,
  ADD latency distribution, aggregate and per-engine hierarchical metrics,
  interval useful-token rate, quantum timing, and preemption deltas.
- `[BENCH_SLOW_ADD]` for individual slow `engine.add_request` calls.
- `[BENCH_ADD_REJECTED]` for every rejected request.
- `[BENCH_INTERRUPTED]` plus a forced final diagnostic snapshot on Ctrl-C.
- Optional per-rank hierarchical execution trace JSONL. It is deliberately
  reserved for a second-stage diagnosis because it adds material overhead.

## Diagnostic reproduction

A second run used the same DP2/SP8 topology, `hao_basic`, full CUDA Graph,
and rate-20 workload, with a one-second diagnostic interval and a 20 ms
slow-ADD threshold. The outer console stream is in
`diagnostic_console.log`. It was stopped after 81.59 seconds of formal
traffic once the root cause was reproducible.

Observed at the final diagnostic snapshot:

- 98 requests were attempted and accepted in 81.59 seconds: only 1.20
  requests/s instead of the requested 20 requests/s.
- 97 of 98 ADD calls exceeded 20 ms. Steady-state ADD latency was about
  0.48-0.51 seconds, with no rejected requests.
- The two LocalEngine actors had completed 868 decode quantums in aggregate.
  Aggregate execute time was 419,903.89 ms, or 483.76 ms per
  engine-quantum.
- Aggregate command queue delay was 250,950.44 ms over 522 handled commands,
  or 480.75 ms per command. This matches one decode quantum almost exactly.
- Both engines remained healthy and decoding. The last interval reported
  788.71 useful decode tokens/s, 24 running requests, zero waiting requests,
  and zero preemptions.
- The client reported zero completions even though only 24 of the 98 accepted
  requests were still running. The completion events were not being polled
  by the client while its submission loop was permanently catching up.

## Root cause

The rate-controlled benchmark submits every due request synchronously inside
one catch-up loop. Hierarchical `add_request` performs a synchronous Ray
round trip and waits for `LocalEngineCore.submit_add` to complete.
`LocalEngineCore` drains normal commands only at the beginning of a decode
quantum; after that it runs the whole 16-forward quantum before checking the
normal command queue again.

With a roughly 0.48-0.51 second quantum and a target inter-arrival time of
0.05 seconds, the first delayed ADD makes the sender fall behind. Its inner
catch-up loop then never catches the arrival schedule and never returns to
`engine.step()` to drain completion events. This creates all three observed
symptoms:

1. effective offered load collapses from 20 requests/s to about 1.2
   requests/s;
2. the progress display appears frozen even while both engines execute;
3. a six-minute run cannot produce valid rate-20 performance data.

This is a benchmark/control-plane serialization problem, not a CUDA Graph
failure, GPU OOM, scheduler backlog, request rejection, or preemption issue.
The fix should decouple timed request production from synchronous ADD
acknowledgements (for example, bounded asynchronous/batched submission) and
continue polling completions independently. Reducing `loop_count` would only
shorten the blocking interval; it would not remove the coupling.

The complete follow-up design is documented in
`../hierarchical_serving_ingress_design.md`. It keeps arrival/rate generation
in the Bench Load Generator, uses RequestRouter as the Load Balancer, adds a
non-blocking LocalEngine ingress boundary, and leaves LocalScheduler responsible
only for its local waiting/admission/decode state.

## Status

Interrupted intentionally; not a valid performance result. Cleanup verified
after both runs. The final Ray check at 06:58:56 UTC reported 0/16 GPUs in
use, no pending demands, and both nodes healthy.

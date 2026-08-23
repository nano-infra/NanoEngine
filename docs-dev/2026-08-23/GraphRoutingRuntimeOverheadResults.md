# Graph Routing Runtime Overhead Results

Date: 2026-08-23

## 1. Status

The P0 component measurements are complete:

1. Graph-bucket selection CPU time;
2. routing-only and all-Graph metadata injection time;
3. production CUDA Graph and padding-kernel attribution;
4. non-profiled `graph.replay()` host-observed submission time;
5. three final-code non-profiled Issue1% serving repeats.

The experiment does not recompute the paper's static/AOT control. A net DCP
delta must reuse an existing result only if it has the same final-code workload
and topology. Otherwise a matched control remains necessary. The component
times below must not be added together to manufacture a net overhead.

## 2. Experiment identity

- Ray GCS: `10.102.252.174:6380`
- topology: 4 nodes, 8 H200 GPUs per node, DP4 SP8 TP1 EP32
- model: DeepSeek-V3 snapshot
  `e815299b0bcbac849fa540c768ef21845365c9eb`
- workload: Issue1%, rate 60 requests/s, 10,800 requests, 180-second send
  window, batch-size cap 192, segment size 65,536
- production instrumentation commit:
  `0963dd2cadc064fb71eba9814a7dd24659049b9c`
- corrected metadata-host benchmark commit:
  `32ad1b3573d34f276bbee8c0a2de0b4202bc1773`
- Python 3.12.13, PyTorch 2.10.0+cu129, CUDA runtime 12.9

Raw logs, traces, and JSON outputs are under
`bench_logs/graph_runtime_overhead/` and are intentionally not committed.

## 3. Production profiler calibration and accepted trace

The first profiler run used step 320. It was rejected for paper attribution:
both selected ranks were still on a local Graph with shape `(6, 8, 6, 8)`, so
the trace did not execute SP routing metadata or padding.

The accepted run used `PROFILER_START_STEP=3200` for 8 inner steps:

| Rank | Actual master | Graph master | Actual attention | Graph attention |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 70 | 80 | 76 | 80 |
| 17 | 70 | 80 | 70 | 80 |

These shapes are close to the historical Issue1% 74/78-row snapshot. Each
trace contains 9 correlated `cudaGraphLaunch` events, no unmatched launches,
and all expected ranges: bucket selection, q-destination rows, actual-attention
scalar, padding, all metadata, MLA preparation, and Graph replay.

Accepted artifacts:

- `issue1_trace_run_2/*.pt.trace.json`
- `issue1_trace_run_2/runtime_overhead_rank_{1,17}.json`
- `trace_summary_run_2.json`

## 4. Graph-bucket selection

One repeat contains 1,000,000 calls to the production selection and lookup
helper. Seven repeats were collected after 10,000 warmup calls.

| Case | Shape | P50 | P95 |
| --- | --- | ---: | ---: |
| Issue1 typical | `(70, 76) -> (80, 80)` | 1.707 us | 1.713 us |
| valid boundary | `(192, 224) -> (192, 224)` | 1.832 us | 1.854 us |

Primary artifact: `bucket.json`.

The production profiler reports much larger CPU-range durations because of
profiler perturbation. Those durations are attribution evidence only and are
not used as the bucket timing result.

## 5. Routing metadata injection

The benchmark uses the production helper and production capacities/dtypes.
Host enqueue samples start from an idle CUDA stream; synchronization after each
operation is outside the timed interval. Device spans use batched CUDA Events.
Each cell reports the P50/P95 across seven repeats of 2,000 operations.

| Scope and case | Host P50/P95 | Device P50/P95 | Bytes touched estimate |
| --- | ---: | ---: | ---: |
| routing-only, no padding | 14.028 / 14.427 us | 13.410 / 13.577 us | 12,292 |
| routing-only, Issue1 padded | 67.280 / 69.566 us | 67.565 / 68.783 us | 34,652 |
| all Graph metadata, no padding | 199.821 / 202.955 us | 196.607 / 202.065 us | 14,517,964 |
| all Graph metadata, Issue1 padded | 271.659 / 272.011 us | 262.750 / 268.852 us | 14,517,796 |

Primary artifact: `metadata.json`.

The routing-only result is the direct answer to the reviewer question. The
all-Graph number is context and includes metadata updates that are not newly
introduced by dynamic routing. FlashMLA preparation and Q-payload staging are
excluded from both scopes.

## 6. Graph and padding attribution

| Rank | Device Graph span P50/P95 | Padding kernels/replay | Individual padding kernel P50/P95 | Padding sum/replay P50/P95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56.256 / 56.428 ms | 61 | 1.600 / 1.793 us | 99.072 / 99.916 us |
| 17 | 56.242 / 59.425 ms | 61 | 1.856 / 2.080 us | 115.325 / 115.967 us |

The padding aggregate is about 0.176% and 0.205% of the corresponding median
full-Graph spans. This is attribution, not a directly additive critical-path
delta: padding nodes may overlap other device work.

The profiler's median host `cudaGraphLaunch` API duration was 48.656 ms on
rank 1 and 48.791 ms on rank 17. These values reflect profiler perturbation and
CUDA queue backpressure; they must not be described as intrinsic Graph launch
cost or Graph execution time. Device Graph execution is represented by the
separate Graph span above.

## 7. Non-profiled replay submission

The production timing wraps the exact `graph.replay()` call without an inserted
synchronization. It therefore measures host-observed submission under the real
serving queue state, including times when CUDA backpressure blocks submission.

| Run | Rank 1 P50/P95 | Rank 17 P50/P95 | Samples/rank |
| ---: | ---: | ---: | ---: |
| 1 | 31.293 us / 45.182 ms | 568.231 us / 45.193 ms | 7,390 |
| 2 | 427.204 us / 47.172 ms | 4.837 ms / 47.167 ms | 7,390 |
| 3 | 390.601 us / 47.698 ms | 365.603 us / 47.593 ms | 7,422 |

Primary artifact: `replay_submit.json`.

The wide rank/run variation is a real warning against presenting one fixed
"launch overhead" number. For the paper, label this as production-observed
submission exposure, retain the P50/P95 distribution, and use the trace only
to attribute the underlying `cudaGraphLaunch -> device Graph` relationship.

## 8. Non-profiled serving repeats

| Run | Total time | Throughput | ITL avg | ITL P50 | ITL P95 | ITL P99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 305.88 s | 21,210.71 tok/s | 53.17 ms | 53.89 ms | 55.73 ms | 56.10 ms |
| 2 | 306.01 s | 21,201.41 tok/s | 54.52 ms | 54.74 ms | 59.26 ms | 59.63 ms |
| 3 | 305.68 s | 21,224.43 tok/s | 54.32 ms | 53.90 ms | 59.34 ms | 59.69 ms |
| Mean | 305.86 s | 21,212.18 tok/s | 54.00 ms | 54.18 ms | 58.11 ms | 58.47 ms |

All three runs completed 10,800/10,800 requests with 100% goodput under the
100-ms TPOT-with-queueing SLO. These are absolute final-code results, not a net
DCP-overhead delta. A matched static/AOT value is required before making a net
impact claim.

## 9. Paper-ready interpretation

- Graph selection and lookup takes about 1.71 us on the Issue1 typical path and
  1.83 us at the tested valid boundary.
- Routing metadata takes 13.41 us of device time without padding and 67.57 us
  for the observed `(70, 80, 76, 80)` padded shape.
- All Graph metadata updates take 196.61 us and 262.75 us respectively; this is
  context, not routing-specific overhead.
- The captured padding nodes account for about 0.10--0.12 ms of aggregate
  kernel duration per replay in the representative production trace.
- Component attribution is not net critical-path overhead. Do not add bucket,
  metadata, padding, and replay-submission values.
- The final-code serving runs average 21.21k tokens/s and 54.00-ms ITL. Map the
  existing matched AOT/static control, or rerun that one control, before stating
  the net runtime impact.

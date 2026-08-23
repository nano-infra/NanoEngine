# Routing Metadata Fusion Prototype Results

Date: 2026-08-23

## Status

Phase 1 passes its correctness, operation-count, and single-H200 performance
criteria. The prototype remains standalone and does not modify NanoDeploy's
serving path.

The measured implementation is commit
`cb32dea92c9c4140c80e082b73521d8c9e615726` (`feat: prototype SP graph
metadata fusion`).

## Implementation

The prototype is in:

```text
scripts/benchmark_sp_graph_metadata_fusion.py
```

It contains:

- an unchanged production-helper reference path;
- one Triton launch that copies the full q-destination mapping, writes the
  actual attention batch scalar, and materializes attention/master padding;
- setup-time validation and Triton JIT warmup outside measured operations;
- a 54-case bit-exact correctness matrix;
- alternating reference/fused host and device timing;
- optional Chrome trace export with automatic CUDA-operation attribution;
- JSON output with raw repeats, summaries, shapes, and environment identity.

The timed fused closure uses Triton's already compiled kernel runner. Calling
the high-level JIT dispatcher on every iteration added enough Python dispatch
overhead to regress the no-padding microbenchmark, even though the underlying
kernel was small. Capturing the compiled runner after warmup removes cache-key
and binding work from every measured launch. Phase 2 should preserve this
property and explicitly handle the serving stream when preparing the runner.

## Environment and command

The final run used one NVIDIA H200 (compute capability 9.0), PyTorch
`2.10.0+cu129`, CUDA runtime `12.9`, Triton `3.6.0`, and Python `3.12.13`.
The representative shape was:

```text
(M_actual, N_actual) -> (M_graph, N_graph)
(70, 76)             -> (80, 80)
SP=8, sp_rank=1, max_num_seqs=192, block-table width=1384
```

The final command was:

```bash
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
CUDA_VISIBLE_DEVICES=0 python3 \
  scripts/benchmark_sp_graph_metadata_fusion.py \
  --shape-json \
  bench_logs/graph_runtime_overhead/issue1_trace_run_2/runtime_overhead_rank_1.json \
  --cases no_padding,issue1_padded \
  --warmup 200 \
  --iterations 2000 \
  --repeats 7 \
  --output \
  bench_logs/graph_runtime_overhead/metadata_fusion_prototype.json \
  --trace-output \
  bench_logs/graph_runtime_overhead/metadata_fusion_prototype.trace.json
```

The generated JSON and trace are retained locally under `bench_logs/` and are
not committed.

## Correctness

All 54 cases passed exact comparison of all nine complete persistent metadata
tensors. The matrix was the Cartesian product of:

- no padding, Issue1 padding, attention-only padding, master-only padding,
  one-row tails, and larger 16-row tails;
- SP ranks 0, 1, and 7;
- block-table widths 1, 17, and 1384.

The q-destination sources contained nontrivial values mixed with inactive `-1`
sentinels. Randomized base destinations made writes outside the selected Graph
bucket or block-table width observable.

## Performance

Times are microseconds per call. P50 and P95 are computed across seven repeats
of 2,000 calls after 200 untimed calls of each implementation.

| Case | Metric | Reference P50 | Fused P50 | Fused P95 | P50 speedup | P50 delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no padding | device span | 11.392 | 5.667 | 6.448 | 2.01x | 5.725 |
| no padding | host enqueue | 12.304 | 6.480 | 6.829 | 1.90x | 5.824 |
| Issue1 padded | device span | 55.756 | 5.815 | 6.498 | 9.59x | 49.941 |
| Issue1 padded | host enqueue | 54.488 | 5.720 | 6.371 | 9.53x | 48.768 |

The final raw device repeats were:

```text
no-padding reference:
  10.7043, 11.2326, 11.3917, 12.2579, 10.7732, 12.8397, 13.1290
no-padding fused:
  5.7740, 5.5752, 5.6669, 5.7816, 5.6155, 5.6148, 6.7332
Issue1 reference:
  60.2509, 61.9744, 52.9227, 55.7562, 58.7080, 52.8962, 51.6554
Issue1 fused:
  5.8147, 6.0666, 5.4853, 5.5528, 6.5874, 6.2896, 5.5903
```

The current reference P50 values are about 15% lower for no padding and 17%
lower for Issue1 padding than the earlier 13.410-us and 67.565-us measurements.
This is shared-node/run drift rather than a changed reference implementation:
the prototype calls the same production helpers, the relative fragmentation
pattern remains, and profiler operation counts match the inventory.

## Profiler attribution

The warmed Issue1 invocation showed:

```text
reference: 10 CUDA operations (9 kernels and 1 DtoD copy)
fused:      1 routing_metadata_fusion_kernel launch
```

Profiler durations are used only for attribution, not as performance results.

## Acceptance review

| Criterion | Result |
| --- | --- |
| Required bit-exact cases | Pass: 54/54 |
| JIT, validation, allocation, `.item()`, sync outside timed fused call | Pass |
| Reference reasonably consistent with earlier baseline | Pass; drift documented above |
| One fused launch in trace | Pass |
| Issue1 device P50 at least 2x faster and no more than 30 us | Pass: 9.59x, 5.815 us |
| No-padding device P50 does not regress | Pass: 2.01x faster |
| Raw repeats and environment identity in JSON | Pass |

This is a component-only microbenchmark result. It must not be presented as a
9.59x serving-latency improvement. The earlier 67.57-us routing component was
only about 0.12% of the representative 56.25-ms full Graph span.

## Validation

CPU-safe checks:

```text
python3 -m pytest tests/test_sp_graph_runtime_overhead.py -q
12 passed in 2.21s

python3 -m py_compile scripts/benchmark_sp_graph_metadata_fusion.py
python3 scripts/benchmark_sp_graph_metadata_fusion.py --help
git diff --check
```

GPU validation is the formal command above. No Ray, RDMA communication, model
weights, C++ rebuild, or multi-node serving run was used.

## Phase 2 recommendation

Proceed with Phase 2 as a separate change. Move the kernel into
`nanodeploy/kernels/`, add focused CUDA tests, connect it behind a clear
supported-path/fallback boundary in `sp_graph_policy.py`, and rerun the
production metadata benchmark plus one-node SP8 dynamic-routing correctness.
Keep the conservative full dummy block-table-row copy unchanged.

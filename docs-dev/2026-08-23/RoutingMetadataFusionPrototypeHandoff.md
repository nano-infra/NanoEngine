# Routing Metadata Fusion Prototype: Implementation Handoff

Date: 2026-08-23

## 1. Objective

Implement and evaluate a **standalone Triton prototype** that fuses NanoDeploy's
per-iteration SP routing-metadata updates. The prototype must compare the
current production implementation with the fused implementation on one GPU in
the same process.

Phase 1 is an engineering feasibility study. It must **not change the
NanoDeploy serving path**. If the prototype passes correctness and performance
criteria, Phase 2 will move the kernel into `nanodeploy/kernels/` and connect it
to the production path.

The prototype answers:

1. Can the padded Issue1 routing path be reduced from roughly ten CUDA
   operations to one fused kernel launch?
2. Does the fused kernel produce bit-exactly the same persistent Graph metadata
   as the current Python/PyTorch implementation?
3. How much do host enqueue and CUDA-stream device span improve on one H200?

This work does not need Ray, RDMA communication, model weights, or a multi-node
serving run.

## 2. Baseline identity and existing results

Use repository commit
`6145917dae7b7f8598faca1bdff3d534ab875858` as the handoff baseline. The
relevant earlier commits are:

```text
0963dd2  feat: measure graph routing runtime overhead
32ad1b3  fix: isolate metadata host enqueue timing
6145917  docs: record graph runtime overhead results
```

The worktree currently has a user-owned `.gitignore` modification. Do not edit,
discard, or stage it.

The accepted final-code production trace used DP4/SP8 on four 8xH200 nodes. Its
rank-1 representative tuple is:

```text
actual master batch:       70
Graph master bucket:       80
actual attention batch:    76
Graph attention bucket:    80
attention SP:               8
max_num_seqs:             192
block-table width:       1384
```

In paper notation, this is:

```text
(M_actual, N_actual) -> (M_graph, N_graph)
(70, 76)             -> (80, 80)
```

The raw shape source is generated data and is not committed:

```text
bench_logs/graph_runtime_overhead/
  issue1_trace_run_2/runtime_overhead_rank_1.json
```

The current single-component measurements are:

| Case | Actual `(M,N)` | Graph `(M_g,N_g)` | Routing device P50 | Routing host P50 |
| --- | ---: | ---: | ---: | ---: |
| no padding | `(80,80)` | `(80,80)` | 13.410 us | 14.028 us |
| Issue1 padded | `(70,76)` | `(80,80)` | 67.565 us | 67.280 us |

These are per decode iteration and per GPU rank. They are not per-layer
numbers. Full results and measurement definitions are in:

```text
docs-dev/2026-08-23/GraphRoutingRuntimeOverheadResults.md
docs-dev/2026-08-23/GraphRoutingRuntimeOverheadExperimentPlan.md
```

## 3. Why padding is slower at a fixed Graph bucket

At a fixed `(80,80)` Graph bucket, Graph tensor shapes and the main model
computation are fixed. The padded path is slower before `graph.replay()` because
the implementation first copies real metadata prefixes and then patches the
dummy tails with several small CUDA operations.

The all-Graph metadata byte estimates demonstrate that total data volume is
almost unchanged:

| Case | All-Graph metadata bytes touched |
| --- | ---: |
| no padding | 14,517,964 |
| Issue1 padded | 14,517,796 |

The padded case is slower because of operation fragmentation and launch/enqueue
latency, not because it transfers substantially more data.

The routing-only benchmark intentionally isolates routing-specific updates. It
does not count the baseline real-prefix copies as routing overhead.

## 4. Exact current operation inventory

The reference boundary is `_routing_metadata_once()` in
`scripts/benchmark_sp_graph_runtime_overheads.py`. It calls the production
helpers:

```python
copy_graph_q_dst_rows(...)
copy_graph_actual_attn_bs(...)
materialize_sp_graph_padding(...)
```

The production implementation is in
`nanodeploy/worker/sp_graph_policy.py`.

### 4.1 Operations common to both cases

```text
1. graph_q_dst_row_indices.fill_(-1)
2. graph_q_dst_row_indices.copy_(q_dst_row_indices)
3. graph_actual_attn_bs.fill_(actual_attn_bs)
```

The first operation is redundant when the source is present. C++ constructs
`q_dst_row_indices_flat` at full `SP * max_num_seqs` capacity and initializes
inactive entries to `-1` in
`csrc/nanodeploy/worker/model_runner_utils.cpp`. A full destination copy
overwrites every entry. Phase 1's fused implementation should exploit that
fact, but the reference implementation must remain unchanged.

### 4.2 Additional operations in the Issue1 padded case

Attention tail, `N_actual=76 -> N_graph=80`:

```text
4. context_lens_for_attn[76:80].fill_(1)
5. block_tables[76:80, :1384].copy_(block_tables[0:1].expand(4, -1))
```

Master tail, `M_actual=70 -> M_graph=80`:

```text
6. context_lens[sp_rank, 70:80].fill_(1)
7. global_context_lens[sp_rank, 70:80].fill_(1)
8. res_slice_get_to_buffer_output[70:80].fill_(dummy_attention_row)
9. torch.arange(..., out=res_slice_fill_to_buffer_output[70:80])
10. res_to_buffer_output_mask[70:80].fill_(1)
```

These are ten logical CUDA operations in the padded routing-only path versus
three in the no-padding path. Depending on PyTorch dispatch, an operation may
appear as a CUDA kernel or a copy API; count both as CUDA operations.

## 5. Phase 1 scope and files

Add one standalone script:

```text
scripts/benchmark_sp_graph_metadata_fusion.py
```

The script should contain or import:

1. a reference wrapper that calls the current production helpers unchanged;
2. a prototype fused Triton kernel and a thin validating Python wrapper;
3. benchmark-only tensor allocation based on the production shape JSON;
4. bit-exact reference-versus-fused correctness checks;
5. host and device timing for both implementations;
6. optional one-invocation profiler trace for operation-count attribution;
7. a JSON result writer with environment and Git identity.

Do not modify these production files in Phase 1:

```text
nanodeploy/worker/sp_graph_policy.py
nanodeploy/worker/model_runner.py
nanodeploy/layers/attention.py
csrc/
```

No C++ change or extension rebuild is required. Triton JIT compilation must be
completed during warmup and excluded from timing.

It is acceptable to reuse benchmark-only parsing/allocation helpers from
`scripts/benchmark_sp_graph_runtime_overheads.py`. Do not duplicate production
semantics for the reference: call the current production helpers so the oracle
remains authoritative.

## 6. Required fused semantics

Phase 1 tests launch fusion only. It must preserve all current metadata values,
including the conservative copy of the complete dummy block-table row.

In particular, **do not yet optimize the 1,384-column dummy block table down to
one entry**. Although `context_len=1` suggests that only the first KV block ID
is consumed, changing that behavior requires a separate FlashAttention
correctness argument. Mixing it into the fusion experiment would make failures
and speedups difficult to attribute.

The fused kernel must perform:

```text
copy all q_dst source entries to the persistent q_dst destination
write the actual_attn_bs device scalar
for each attention-tail row:
    write context_lens_for_attn = 1
    copy all block_table_width entries from actual_block_tables row 0
for each master-tail row:
    write local context_lens = 1
    write local global_context_lens = 1
    write res_get = dummy_attention_row
    write res_fill = sp_rank * max_num_seqs + master_row
    write res_mask = 1
```

The dummy attention row follows current production semantics:

```python
dummy_attention_row = (
    actual_attn_bs if actual_attn_bs < graph_attn_bs else 0
)
```

### 6.1 Suggested one-launch Triton mapping

Use a one-dimensional grid with `BLOCK_SIZE=256`. Define:

```text
q_work      = q_dst_numel
attn_work   = (graph_attn_bs - actual_attn_bs) * block_table_width
master_work = graph_master_bs - actual_master_bs
total_work  = max(q_work, attn_work, master_work, 1)
```

Each Triton offset conditionally handles all applicable segments:

```text
offset < q_work:
    q_dst[offset] = q_dst_source[offset]

offset == 0:
    graph_actual_attn_bs[0] = actual_attn_bs

offset < attention_tail:
    context_lens_for_attn[actual_attn_bs + offset] = 1

offset < attn_work:
    tail_row = offset // block_table_width
    column   = offset % block_table_width
    graph_block_tables[actual_attn_bs + tail_row, column] =
        actual_block_tables[0, column]

offset < master_tail:
    master_row = actual_master_bs + offset
    context_lens[sp_rank, master_row] = 1
    global_context_lens[sp_rank, master_row] = 1
    res_get[master_row] = dummy_attention_row
    res_fill[master_row] = sp_rank * max_num_seqs + master_row
    res_mask[master_row] = 1
```

Pass actual and Graph row counts as runtime kernel arguments rather than
`tl.constexpr` values. Only `BLOCK_SIZE` needs to be constexpr. This avoids
compiling a new kernel for every observed batch tuple. Pass tensor strides
explicitly; do not assume that the Graph block-table row stride equals the
actual block-table width.

The measured wrapper must not allocate tensors, call `.item()`, synchronize,
or perform validation inside the timed operation.

### 6.2 Wrapper preconditions

Validate before timing that:

```text
all tensors are CUDA int32 tensors on one device
q_dst source and destination have equal numel
q_dst source is full capacity and contains inactive sentinels
actual_block_tables is two-dimensional and non-empty
0 < actual_attn_bs <= graph_attn_bs
0 <= actual_master_bs <= graph_master_bs
result mapping capacity covers graph_master_bs
all required destination tensors are contiguous or have supported strides
```

Do not put these Python checks in the repeated timed lambda.

## 7. Correctness oracle

For every test case:

1. create one randomized/sentinel-filled base set of destination tensors;
2. clone it into independent `reference` and `fused` destinations;
3. use the same source tensors and scalar shape arguments;
4. run the current production helper on `reference`;
5. run the fused kernel on `fused`;
6. synchronize once;
7. compare the complete destination tensors with exact equality.

Compare all of:

```text
q_dst_row_indices
actual_attn_bs
context_lens
global_context_lens
context_lens_for_attn
block_tables
res_slice_get_to_buffer_output
res_slice_fill_to_buffer_output
res_to_buffer_output_mask
```

Compare complete tensors, not only updated slices. This detects accidental
writes outside the selected Graph bucket or block-table width.

Timed performance cases are only:

```text
no_padding:    (80,80) -> (80,80)
issue1_padded: (70,76) -> (80,80), width=1384, SP8, max_num_seqs=192
```

Additional shapes are correctness cases, not a performance sweep:

```text
attention-only tail: (80,76) -> (80,80)
master-only tail:    (70,80) -> (80,80)
minimal tails:       (79,79) -> (80,80)
larger valid tails:  (64,64) -> (80,80)
sp_rank:             0, 1, 7
block-table width:   1, 17, 1384
```

The test must include nontrivial q-destination values mixed with `-1`; an
all-zero source would not detect sentinel/copy bugs.

## 8. Timing methodology

Use the same timing definitions as the existing metadata benchmark.

### 8.1 Device span

For each implementation and repeat:

```python
begin_event.record()
for _ in range(iterations):
    operation()
end_event.record()
end_event.synchronize()
us_per_call = begin_event.elapsed_time(end_event) * 1000 / iterations
```

Run 200 untimed warmup calls before collecting seven repeats of 2,000 calls.
Warm both implementations before either is measured. Alternate reference/fused
measurement order across repeats when practical to reduce clock and thermal
bias.

### 8.2 Host enqueue

For each host sample:

```python
torch.cuda.synchronize()
t0_ns = time.perf_counter_ns()
operation()
t1_ns = time.perf_counter_ns()
torch.cuda.synchronize()  # outside the measured interval
```

The synchronization before each sample ensures that CUDA queue backpressure is
not mislabeled as host submission cost. Do not synchronize between `t0_ns` and
`t1_ns`.

### 8.3 Profiler attribution

Profiler timing is not a performance result. Optionally profile one warmed
reference invocation and one warmed fused invocation under separate
`record_function` ranges:

```text
nanodeploy.prototype.routing_metadata.reference
nanodeploy.prototype.routing_metadata.fused
```

Export one Chrome trace and verify the logical operation-count change:

```text
Issue1 reference: approximately 10 CUDA operations
Issue1 fused:      1 Triton kernel launch
```

## 9. Result schema

Write a JSON artifact containing at least:

```text
schema_version
git_sha
torch version
triton version
CUDA runtime
GPU name and index
shape JSON source
warmup, iterations, repeats
correctness cases and pass/fail status
for each timed case and implementation:
    raw host us/call repeats
    host p50/p95
    raw device us/call repeats
    device p50/p95
speedup and absolute delta
optional profiler trace path
```

Do not commit bulky raw profiler traces. Store generated outputs under
`bench_logs/graph_runtime_overhead/`.

## 10. Single-GPU command

Before every GPU run, configure the repository-required driver environment:

```bash
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
```

The target CLI is:

```bash
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

Ray is not used. Do not connect to `10.102.252.174:6380` for this phase.

## 11. Acceptance criteria

Phase 1 passes only if:

1. all required correctness cases are bit-exact;
2. no timed call includes Triton compilation, allocation, validation, `.item()`,
   or synchronization;
3. the reference measurement is reasonably consistent with the existing
   13.41-us no-padding and 67.57-us Issue1-padded device P50 values; document
   shared-node drift instead of forcing exact agreement;
4. the profiler shows one fused kernel launch for the routing update;
5. Issue1-padded device P50 improves by at least 2x, with a target of 30 us or
   less;
6. no-padding device P50 does not materially regress;
7. the JSON contains raw repeats and full environment identity.

The expected, not guaranteed, outcome is:

```text
reference padded: approximately 67.6 us and 10 CUDA operations
fused padded:     approximately 10-25 us and 1 CUDA operation
```

Do not present this component speedup as a net serving latency improvement.
The current 67.57 us is about 0.12% of the representative 56.25-ms full Graph
span, and component durations must not simply be added to other asynchronous
costs.

## 12. Explicit exclusions

Phase 1 does not include:

- a Fig. 17-style batch-size sweep;
- a broader Graph-bucket or padding-tail performance sweep;
- historical before/candidate commit regression testing as a paper result;
- changing the dummy block-table semantics;
- modifying FlashAttention or any external dependency;
- optimizing `prepare_decode_mla_metadata`;
- optimizing the per-layer Graph-internal `zero_padded_rows_kernel`;
- Ray or multi-node serving.

The accepted trace contains 61 `zero_padded_rows_kernel` instances per replay.
Those kernels execute inside the captured model Graph and are separate from the
once-per-iteration routing-metadata update measured here.

## 13. Phase 2, only after prototype acceptance

If Phase 1 passes:

1. move the kernel to `nanodeploy/kernels/sp_graph_metadata.py`;
2. update `nanodeploy/worker/sp_graph_policy.py` to use it on the supported CUDA
   Hao/SP path while retaining a clear fallback/reference path;
3. add focused CUDA correctness tests and keep existing CPU policy tests;
4. rerun `scripts/benchmark_sp_graph_runtime_overheads.py --component metadata`
   through the production helper;
5. run the one-node SP8 dynamic-routing correctness test;
6. optionally perform one final four-node production smoke before using the
   optimized value in the paper.

Phase 2 remains Python/Triton-only unless the design intentionally moves
bucket-sized metadata construction into `prepare_decode_cpp`. Any C++ change
would require `python3 -m pip install -v -e .` before project validation.

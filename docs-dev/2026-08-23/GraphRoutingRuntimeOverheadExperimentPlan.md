# ASPLOS Revision: Graph Routing Runtime Overhead Experiment Plan

## 1. Scope and conclusion

This plan measures the remaining runtime-overhead gaps in the ASPLOS review:

1. Graph-bucket selection on CPU;
2. routing-metadata injection on GPU;
3. CUDA Graph replay submission and the padding kernels contained in a replay;
4. one production trace showing that the microbenchmarks correspond to a
   real Issue1% serving shape.

The primary subject is the **final integration HEAD used by the paper**. The
experiment does not include a historical commit comparison.

No C++ change is required. All three boundaries are visible from Python:

- bucket selection and Graph lookup are Python operations;
- routing metadata is updated by PyTorch/Triton calls issued from Python;
- `graph.replay()` returns after host submission and can be timed with
  `time.perf_counter_ns()` without synchronizing inside the measured interval.

C++ instrumentation would only be justified if the paper later asks for an
internal breakdown of a native helper such as `prepare_decode_cpp`. That is not
part of this experiment.

## 2. Measurement principles

Use two complementary sources of evidence:

- **stable timing:** non-profiled microbenchmarks and non-profiled serving runs;
- **attribution:** one or two short `torch.profiler` traces with CPU and CUDA
  activities enabled.

Profiler durations are not the primary microsecond-scale result. In particular,
`record_function`, stack collection, shape recording, and memory profiling can
perturb short CPU operations.

The following values are separate and must not be added together:

```text
host metadata enqueue time
device metadata-operation time/span
host graph.replay() submission time
padding-kernel device time
full Graph device span
```

Asynchronous copies, kernels, and host work may overlap. The paper should show
component costs for attribution and use a separate non-profiled serving result
for net runtime impact.

## 3. Exact measurement boundaries

### 3.1 Graph-bucket selection

Measure the production operation from the first bucket search through the Graph
dictionary lookup:

```text
master_bs = smallest captured master bucket >= actual master bs
graph_attn_bs = smallest captured attention bucket >= actual attention bs
graph = sp_graphs[(master_bs, graph_attn_bs)]
```

The local-Graph branch performs only the master-bucket selection and
`local_graphs[master_bs]` lookup. The primary paper number is the dynamic SP
branch exercised by Issue1%.

Exclude metadata injection, FlashMLA metadata preparation, and Graph replay.
Report the absolute `selection + lookup` p50 and p95 in nanoseconds. A direct
known-key lookup may be retained as an internal control, but an adjusted value
is not needed in the paper.

### 3.2 Routing-metadata injection

Report two scopes:

```text
routing metadata
  q_dst_row_indices fill/copy
  actual_attn_bs scalar update
  materialize_sp_graph_padding(...)

all Graph metadata
  complete _copy_decode_context_to_graph_vars(...)
```

Exclude both of the following:

- `prepare_decode_mla_metadata`, which is an existing FlashMLA preparation
  phase;
- `copy_batch_indexed_triton`, which stages Q payload and is not metadata.

For each scope report host enqueue p50/p95 and CUDA-stream elapsed p50/p95. Also
record the copied/initialized byte count and actual versus captured row count.
When operations overlap, report both the sum of individual device durations and
the enclosing device span; label them explicitly.

`q_dst_row_indices` currently copies its allocated capacity. Its byte count is
therefore based on the allocated tensor, not only the active rows.

### 3.3 Graph replay and padding kernels

The stable host-submission measurement is:

```python
t0_ns = time.perf_counter_ns()
graph.replay()
t1_ns = time.perf_counter_ns()
replay_submit_ns.append(t1_ns - t0_ns)
```

There must be no `torch.cuda.synchronize()` between `t0_ns` and `t1_ns`.
Synchronize before warmup and only after a measured batch when a completed
device result is required. Measure and report an empty `perf_counter_ns()` pair
as timer overhead, but do not silently subtract it from the headline value.

The profiler trace is used to attribute:

- `cudaGraphLaunch` host API duration;
- full device Graph span associated with its correlation ID;
- `zero_padded_rows_kernel` count per replay;
- individual padding-kernel device p50/p95;
- aggregate padding-kernel device duration per replay.

Do not call `cudaGraphLaunch` host API duration “Graph execution time.” The
padding cost reported for a token/replay is the aggregate across all captured
attention layers, not one kernel instance.

## 4. Real Issue1% reference shape

Use the following existing production log as a read-only workload reference:

```text
/mnt/nvme1n1/ml_research/chenjiefei/nano_logs/
  issue003_then_issue001_rate5_chain_20260405_v2/
  issue001_deepseek_v3_rate5/deepseek-v3/
  sharegpt4o-random_geminiissue_r0.01_n60000_60k/
  dp4sp8_seg64k_n36000_r60_bs192_LB_cen_maxin1000k/
  20260406_091601.log
```

The selected decode record is identifiable by:

```text
timestamp:        2026-04-06 01:24:00
ITL:              62.82 ms
scheduler:         6.42 ms
post-scheduler:    2.62 ms
```

Its per-DP/per-SP active attention-row counts are:

```python
sp_batch_sizes = [
    [70, 74, 76, 74, 75, 73, 72, 75],
    [69, 73, 75, 76, 77, 72, 73, 73],
    [77, 78, 76, 64, 78, 74, 73, 68],
    [74, 77, 73, 77, 71, 70, 71, 75],
]
```

The current scheduler code defines each entry as
`len(filtered_dp_sp_seqs[dp_idx * sp_size + sp_idx])`. In this log, every
entry also equals the length of the corresponding `sp_seq_lens` list. Across
the 32 attention workers:

```text
minimum = 64
median  = 74
mean    = 73.53125
maximum = 78
```

Use this snapshot as a production-derived shape target:

- 74 is the candidate representative attention-row count;
- 78 is the observed upper-end candidate for the padded case;
- the whole `64..78` range is the production-relevance acceptance window.

The old log does **not** contain enough information to assert the final code's
exact `context.attention_compute_bs`, `master_bs`, or selected `graph_attn_bs`.
`sp_batch_sizes` is a strong proxy for the active attention rows, but do not
equate it with any of those values without confirmation. The new final-HEAD
trace must record this tuple explicitly on each worker:

```text
(actual_master_bs, master_bs, actual_attn_bs, graph_attn_bs)
```

The microbenchmark must use a tuple observed from that final-HEAD run. If the
new serving run does not enter the historical `64..78` range, report the
mismatch and choose the nearest steady-state tuple rather than claiming an
exact replay of the old snapshot.

The reference configuration encoded by the log path is DP4/SP8, batch limit
192, segment size 64K, request rate 60, LeastBatch/central scheduling, maximum
input length 1M, and the Issue1% ShareGPT-derived dataset. Preserve these
settings for the production validation unless the final paper configuration
has intentionally changed; record any difference.

## 5. P0 experiment matrix

| Experiment | Cases | Repeats | Primary output |
| --- | --- | ---: | --- |
| CPU bucket microbench | observed typical tuple; valid boundary tuple | 7 | selection + lookup ns, p50/p95 |
| GPU metadata microbench | no padding; observed padded tuple near 74/78 rows | 5–7 | routing-only and all-metadata host/device us |
| Graph replay submission | representative final-HEAD model Graph | 5–7 | non-profiled host submission us |
| Production attribution trace | Issue1% dynamic case; optional exact-bucket control | 1–2 | APIs, Graph span, padding kernels, observed tuples |
| Final-HEAD non-profiled serving | same Issue1% workload | 3–5 | ITL/throughput net result |

### 5.1 Bucket cases

Run 10,000 untimed warmup calls, followed by 1,000,000 measured calls per
repeat. Pin the process to one CPU core if the host is shared. Use:

1. the most frequent tuple observed in the production trace;
2. a valid boundary case immediately above a captured bucket boundary.

Timing must use the same pure selection helper called by production code. A
pytest test verifies exact selection and failure semantics, but pytest must not
contain a latency threshold.

### 5.2 Metadata cases

Use 200 warmup iterations and 2,000 measured iterations per repeat unless GPU
memory or runtime makes that impractical. Record one CUDA Event pair around a
batch of iterations and synchronize after the end event, outside the host
enqueue timing interval.

Cases:

1. `no_padding`: `actual_attn_bs == graph_attn_bs` and
   `actual_master_bs == master_bs`, exposing fixed metadata cost;
2. `issue1_padded`: the observed final-HEAD tuple nearest the historical
   median/upper range, exposing realistic padding cost.

The paper table only needs `routing metadata` and `all Graph metadata`.
Individual q-destination/scalar/padding subcomponents may be saved in JSON for
debugging or an appendix.

### 5.3 Replay submission

Collect `perf_counter_ns()` samples around the actual model Graph's
`graph.replay()` in a non-profiled run. Discard capture/warmup and the first two
decode steps. Report per-call p50/p95 over normal serving queue state. Padding
kernel attribution comes directly from the production trace.

### 5.4 Net runtime impact

Use the paper's corresponding final-code control, such as the already defined
static/AOT configuration, for end-to-end comparison. Do not define net DCP
overhead as the sum of component timings. If the existing paper result was
collected from the same final implementation and workload, it may be reused;
otherwise run 3–5 non-profiled repeats.

## 6. Minimal code changes

Keep all instrumentation behind disabled-by-default options.

### 6.1 `nanodeploy/worker/sp_graph_policy.py`

Extract the current selection logic into a pure helper that returns
`(master_bs, graph_attn_bs)` and performs the existing validation. Both
`ModelRunner.run_model` and the CPU benchmark must call this helper. This is a
behavior-preserving refactor, not a new selection algorithm.

Keep `materialize_sp_graph_padding` reusable and move the complete Graph
metadata copy into a shared helper, so both metadata scopes execute the same
operations as production.

### 6.2 `nanodeploy/config.py`

Add these disabled-by-default controls:

```python
profiler_mode: Literal["default", "runtime_overhead"] = "default"
runtime_overhead_timing: bool = False
profiler_ranks: tuple[int, ...] | None = None
```

In `runtime_overhead` profiler mode use:

```python
activities = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]
record_shapes = False
profile_memory = False
with_stack = False
```

The default profiler behavior must remain unchanged. `profiler_ranks=None`
keeps the existing all-rank behavior; the production overhead trace should use
only the representative and upper-end attention ranks to avoid writing 32
large traces.

### 6.3 `nanodeploy/worker/model_runner.py`

Add profiler-only ranges around:

```text
nanodeploy.graph.bucket_select
nanodeploy.graph.metadata.all
nanodeploy.graph.metadata.q_dst_rows
nanodeploy.graph.metadata.actual_attn_bs
nanodeploy.graph.metadata.padding
nanodeploy.graph.prepare_decode_mla_metadata
nanodeploy.graph.replay
```

Emit the four-value shape tuple in the replay range name and a compact per-rank
JSON summary while profiling. Do not log it every step in normal serving.

When `runtime_overhead_timing=True` and the profiler is off, collect
`perf_counter_ns()` host samples around metadata injection and
`graph.replay()`. Buffer samples in memory and write only the final aggregate or
one compact JSON file; file I/O must not occur in the timed decode path.

The profiler range durations are attribution only. The buffered non-profiled
samples are the stable host numbers.

### 6.4 `scripts/benchmark_sp_graph_runtime_overheads.py`

Add one focused script with three components:

```text
--component bucket
--component metadata
--component replay-submit
```

The script should:

- accept an observed shape tuple via CLI or JSON;
- emit raw repeat results and summary p50/p95 as JSON;
- record warmup/iteration/repeat counts, device, software versions, git SHA,
  and `nanodeploy.__file__`;
- reject impossible tuples rather than silently clipping them.

The metadata path should allocate tensors with production dtypes and capacities
and call the same routing-padding helper used by `ModelRunner`.

### 6.5 `scripts/run_issue001_deepseek_v3_issue001_bucket.sh`

Forward these environment-controlled arguments to
`scripts/issue003/bench_serving_overhead.py`:

```text
ENABLE_PROFILER
PROFILER_MODE
PROFILER_START_STEP
PROFILING_STEP
PROFILER_DIR
PROFILER_RANKS
RUNTIME_OVERHEAD_TIMING
DURATION_SECONDS
```

`DURATION_SECONDS` should replace the hard-coded `600` only in request-count
calculation and default to 600, preserving existing behavior.

Also replace the script's existing `python -u` launch with `python3 -u` to
follow the repository command policy.

### 6.6 Trace parser

Add one lightweight script under `utils_analysis/` to correlate
`cudaGraphLaunch` with device events and aggregate `zero_padded_rows_kernel`
count/duration per replay. The reference-log shape has already been validated
for this plan; a general reference-log parser is unnecessary.

Do not build a general profiler framework for this revision.

### 6.7 Tests

Add CPU tests for:

- typical, exact-boundary, next-boundary, and overflow bucket selection;
- profiler-mode defaults, rank filtering, and validation.

Manually inspect a few parsed launches against the profiler UI before accepting
the aggregate output.

CUDA timing assertions are not unit tests. After any Python-only change, no
editable rebuild is required. Reinstall with `python3 -m pip install -v -e .`
only if the implementation unexpectedly changes C++ or bindings.

## 7. Commands

Run Section 7.3 first. Its per-rank summary supplies the final-HEAD shape JSON
consumed by Sections 7.1 and 7.2.

### 7.1 CPU bucket benchmark

```bash
python3 scripts/benchmark_sp_graph_runtime_overheads.py \
  --component bucket \
  --shape-json bench_logs/graph_runtime_overhead/issue1_trace/runtime_overhead_rank_1.json \
  --warmup 10000 \
  --iterations 1000000 \
  --repeats 7 \
  --output bench_logs/graph_runtime_overhead/bucket.json
```

### 7.2 GPU metadata microbenchmark

Before every GPU command, configure the required DLSlime/RDMA environment and
request elevated execution permission:

```bash
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4

python3 scripts/benchmark_sp_graph_runtime_overheads.py \
  --component metadata \
  --shape-json bench_logs/graph_runtime_overhead/issue1_trace/runtime_overhead_rank_1.json \
  --cases no_padding,issue1_padded \
  --warmup 200 \
  --iterations 2000 \
  --repeats 7 \
  --output bench_logs/graph_runtime_overhead/metadata.json
```

### 7.3 Production Issue1% profiler run

Ray is the exception to the repository proxy rule, so remove proxy variables
before connecting to the cluster:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
export NANODEPLOY_LOG_DECODE_STEP_DETAIL=1

MODEL_PATH=/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
DP=4 SP=8 EP=32 BATCH_SIZE=192 SEG=65536 LOOP_COUNT=16 \
SCHEDULER_ARCH=legacy_global ROUTING=LeastBatch \
DYNAMIC_SP_SIZE_STRATEGY=bucket DYNAMIC_SP_BUCKET_PRESET=deepseek_v3 \
MAX_MODEL_LEN=1000000 MAX_INPUT_LEN=1000000 \
DURATION_SECONDS=180 ENABLE_PROFILER=1 PROFILER_MODE=runtime_overhead \
PROFILER_START_STEP=320 PROFILING_STEP=8 \
PROFILER_RANKS=1,17 \
PROFILER_DIR=bench_logs/graph_runtime_overhead/issue1_trace \
RUN_TAG=graph_runtime_overhead_issue1_profile \
bash scripts/run_issue001_deepseek_v3_issue001_bucket.sh 60
```

Historical flattened attention ranks 1 and 17 correspond respectively to a
74-row worker and one of the 78-row workers in the supplied snapshot. The
final-HEAD tuple recorded on those ranks is authoritative; change the rank
filter for a retry if their new shapes are not representative.

`PROFILER_START_STEP=320` means 20 outer decode batches with the current
`LOOP_COUNT=16`; verify this assumption in the run log. If the profiled window
does not reach a steady decode phase, change only the start step and record it.
Keep `PROFILING_STEP` at 4–8 inner steps to limit trace size.

Run this once. A second identical run is only needed if the trace is incomplete
or the selected window is not representative.

The production profile must run before the bucket and metadata commands because
its per-rank summary supplies the authoritative final-HEAD shape and captured
Graph candidates.

### 7.4 Non-profiled replay submission and serving result

Use the same command and workload with:

```text
ENABLE_PROFILER=0
RUNTIME_OVERHEAD_TIMING=1
PROFILER_DIR=bench_logs/graph_runtime_overhead/issue1_nonprofiled_run_1
```

Run 3–5 repeats for the end-to-end result. The compact timing output may report
metadata enqueue and replay submission, but the end-to-end ITL/throughput must
come from the normal serving metrics.

Summarize the stable replay-submission measurements with:

```bash
python3 scripts/benchmark_sp_graph_runtime_overheads.py \
  --component replay-submit \
  --timing-json \
    bench_logs/graph_runtime_overhead/issue1_nonprofiled_run_1/runtime_overhead_rank_1.json \
  --output bench_logs/graph_runtime_overhead/replay_submit.json
```

### 7.5 Production trace parser

```bash
python3 utils_analysis/analyze_sp_graph_runtime_trace.py \
  bench_logs/graph_runtime_overhead/issue1_trace/*/*.pt.trace.json \
  --output bench_logs/graph_runtime_overhead/trace_summary.json
```

## 8. Trace analysis and reporting

For every `cudaGraphLaunch` event:

1. read its correlation ID;
2. collect device events belonging to that launch;
3. compute full Graph span as `latest_end - earliest_start`;
4. select events whose name contains `zero_padded_rows_kernel`;
5. report their count, individual duration distribution, and summed duration;
6. retain the associated shape tuple and worker rank.

The parser reports unmatched launches when a trace does not expose usable
correlation IDs. Do not match kernels solely by nearest timestamp.

The paper-facing result can be two compact tables:

| Component | Case | p50 | p95 | Unit |
| --- | --- | ---: | ---: | --- |
| bucket selection + lookup | Issue1 typical / boundary | ... | ... | ns |
| routing metadata host enqueue | no padding / Issue1 padded | ... | ... | us |
| routing metadata device span | no padding / Issue1 padded | ... | ... | us |
| all Graph metadata | no padding / Issue1 padded | ... | ... | us |
| Graph replay host submission | Issue1 representative | ... | ... | us |

| Production attribution | Value |
| --- | ---: |
| observed `(actual_master, graph_master, actual_attn, graph_attn)` | ... |
| full Graph device span | ... us |
| padding kernels per replay | ... |
| aggregate padding-kernel time per replay | ... us |
| non-profiled ITL / throughput | ... |

Suggested paper wording:

> Graph selection takes X ns at the median (Y ns at P95). Routing-specific
> metadata injection takes A us of host enqueue time and B us of device-stream
> span for an Issue1% production-derived shape. CUDA Graph replay submission
> takes C us on the host. The additional padding nodes execute K times per
> replay and account for D us in aggregate. These component values are used for
> attribution; net runtime impact is measured separately by the non-profiled
> serving experiment.

## 9. Acceptance criteria

- Primary numbers come from final integration HEAD.
- Bucket and metadata microbenchmarks cover only typical and boundary/padded
  cases.
- The production profile has only 1–2 repeats and uses the real Issue1%
  workload configuration.
- The final trace records the actual/selected master and attention sizes; no
  bucket is inferred from the April log.
- Profiler timing is used for attribution, not as the sole stable result.
- Host submission is not mislabeled as device execution.
- Overlapping component times are not summed into net overhead.
- No C++ source or external dependency is modified.

Raw logs and profiler traces belong under `bench_logs/` and should not be
committed. Save the reviewed numerical summary as
`docs-dev/2026-08-23/GraphRoutingRuntimeOverheadResults.md`.

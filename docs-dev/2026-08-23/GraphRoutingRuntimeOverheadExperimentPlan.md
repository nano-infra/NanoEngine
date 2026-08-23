# SP CUDA Graph Routing Runtime Overhead Experiment Plan

## 1. Objective

This experiment measures the three runtime costs requested during review of
the unified Hao SP CUDA Graph routing change:

1. CUDA Graph bucket-selection CPU overhead;
2. routing-metadata injection overhead before Graph replay;
3. the additional padding-kernel and CUDA Graph launch/replay overhead.

The target change is `a5a01b2` (`fix: unify Hao SP graph routing`). Its direct
parent, `6689f5d`, is the primary before-change baseline. Use these two commits
for the isolated before/after claim. A final smoke on the current integration
HEAD is required, but later scheduler and serving changes must not be folded
into the overhead attributed to `a5a01b2`.

The experiment must produce both:

- stable timing numbers obtained without `torch.profiler`; and
- production traces that explain which CPU APIs, copies, kernels, and Graph
  nodes account for those numbers.

Profiler durations alone are not used as the end-to-end regression result,
because CPU tracing, shape recording, stack collection, and memory profiling
can materially perturb microsecond-scale host work.

## 2. Definitions and measurement boundaries

### 2.1 Graph bucket selection

The measured production operation starts at the `bs` lookup in
`ModelRunner.run_model` and ends after looking up the selected Graph object. It
includes:

- selection of `master_bs` from `graph_master_rank_bs`;
- selection of `attn_bs` from `sp_graph_map[master_bs]` for SP execution; and
- lookup of `sp_graphs[(master_bs, attn_bs)]` or
  `local_graphs[master_bs]`.

It excludes metadata copying, FlashMLA metadata construction, and Graph
replay. The primary number is the CPU time per decode step. The reported
overhead is:

```text
bucket overhead = (selection + graph lookup) - direct known-key graph lookup
```

Report the absolute production-path time as well as this adjusted value. A
pytest unit test validates selection semantics, but a timing assertion must not
be placed in pytest.

### 2.2 Routing-metadata injection

Use two nested scopes so that the review can distinguish the new routing state
from the pre-existing full Graph-variable update:

```text
graph_metadata_injection.total
  input IDs / positions / slot mapping
  q_dst_row_indices fill and copy
  context/block/Q/Res/LSE metadata updates
  actual_attn_bs scalar update
  SP Graph padding metadata materialization
```

The routing-specific component is the sum of:

```text
graph_metadata_injection.q_dst_rows
graph_metadata_injection.actual_attn_bs
graph_metadata_injection.padding_metadata
```

`prepare_decode_mla_metadata` is measured in its own scope and is not included
in routing-metadata injection. The Q payload staging kernel
`copy_batch_indexed_triton` is also excluded: it copies Q data into the
all-to-all buffer and is not metadata injection.

For both the routing-only and total scopes report:

- host enqueue time per decode step;
- device-stream elapsed time per decode step;
- bytes copied or initialized;
- before/after delta where both versions implement a valid case.

The persistent `q_dst_row_indices` copy currently covers its allocated
capacity, so its reported byte count is
`attention_sp * max_num_seqs * sizeof(int32)`, not the number of active rows.
Report active-row count separately.

### 2.3 Padding kernel and Graph launch/replay

The new device operation is `zero_padded_rows_kernel`. It is captured inside
the full model CUDA Graph and executes once per SP attention layer, even when
`actual_attn_bs == graph_attn_bs` and the writable tail is empty.

Report all of the following; they are different quantities:

- `cudaGraphLaunch` host API duration;
- total device Graph span, from the earliest to latest device event carrying
  the launch correlation ID;
- `zero_padded_rows_kernel` count per replay;
- single-kernel p50/p95 device duration;
- sum of padding-kernel device durations per replay;
- before/after full-Graph span and non-profiled decode latency.

Do not describe `cudaGraphLaunch` host API duration as Graph execution time.
Do not report only one padding-kernel duration: the per-token cost is the
aggregate across all attention layers.

## 3. Experimental controls

### 3.1 Software versions

Create independent worktrees rather than checking out over the user's current
dirty tree:

```text
before:    6689f5d + the profiling-only instrumentation patch
candidate: a5a01b2 + the identical profiling-only instrumentation patch
head:      current integration HEAD, smoke only
```

The profiling patch must not change bucket candidates, tensor sizes,
collective arguments, stream placement, or synchronization. Record the exact
instrumentation commit in every result manifest. If the patch needs a small
context adjustment on the old implementation, keep the range boundaries
semantically identical and record the difference in the result note.

No C++ source is changed by the proposed instrumentation. Each worktree should
still be made importable explicitly, and the run manifest must record
`nanodeploy.__file__` so that a Ray worker cannot silently load the other
worktree. If editable installation is used, install and run the two worktrees
sequentially.

### 3.2 Hardware and runtime controls

- Use the same eight GPUs, node, clocks/power policy, Ray allocation, CUDA,
  PyTorch, Triton, FlashMLA, and DLSlime build for every matched pair.
- Use `hao_basic`, attention DP1/SP8/TP1, FFN DP1/EP8/TP1, and full CUDA Graph.
- Use the repository DeepSeek-V3 snapshot:

  ```text
  /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb
  ```

- Use dummy prefill and dummy weights for the component profile so model I/O
  and checkpoint loading do not dominate turnaround. Use the same MoE routing
  simulation and seed in every run.
- Exclude Graph capture, Triton compilation, allocator warmup, and the first
  two profiled decode steps from statistics.
- Run at least five matched repeats. Alternate version order, for example
  `before, candidate, candidate, before`, instead of running every baseline
  many hours before every candidate.
- Save raw artifacts under
  `bench_logs/graph_routing_overhead_<UTC timestamp>/`; do not commit raw
  traces. Save the reviewed summary under
  `docs-dev/2026-08-23/GraphRoutingRuntimeOverheadResults.md`.

Before every GPU run, export all required DLSlime settings:

```bash
export SLIME_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SLIME_GID_INDEX=3
export SLIME_QP_NUM=4
```

Before Ray commands or a run that connects to Ray, remove HTTP proxies:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

Every GPU command must be run with elevated permission according to the
repository instructions. CPU-only bucket tests do not require elevation.

## 4. Required code changes

Keep the implementation in one profiling-only commit so it can be applied to
both worktrees.

### 4.1 `nanodeploy/config.py`

Add a profiler mode with a compatibility-preserving default:

```python
profiler_mode: Literal["default", "runtime_overhead"] = "default"
```

Validate the two allowed values. `runtime_overhead` is valid only when
`enable_profiler=True`; either reject or ignore it otherwise, but use one
behavior consistently and cover it with a CPU config test.

Do not change the existing default profiler behavior. In runtime-overhead mode
the ModelRunner profiler must use:

```python
activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]
record_shapes=False
profile_memory=False
with_stack=False
```

CPU activity is needed for `record_function`, `cudaGraphLaunch`,
`cudaMemcpyAsync`, and kernel-launch API events. Disabling shapes, memory, and
stacks reduces trace size and perturbation.

### 4.2 `nanodeploy/worker/model_runner.py`

#### Track whether ranges should be emitted

Add a lightweight helper backed by `contextlib.nullcontext`:

```python
def _runtime_overhead_range(self, name: str):
    if self._runtime_overhead_profiler_active:
        return torch.profiler.record_function(name)
    return nullcontext()
```

Initialize `_runtime_overhead_profiler_active=False`, set it only after
`self.profiler.start()` succeeds, and clear it immediately before or after
`self.profiler.stop()`. This prevents `record_function` overhead during normal
serving and during non-profiled warmup.

#### Make the bucket path directly benchmarkable

Refactor the existing selection statements, without changing their search
algorithm, into:

```python
def _select_decode_graph(
    self,
    bs: int,
    context,
) -> tuple[int, int, torch.cuda.CUDAGraph]:
    ...
```

The helper must contain the existing master-bucket selection, attention-bucket
selection, and Graph dictionary lookup. `run_model` calls this helper once.
Keeping the existing ordered linear search avoids mixing an optimization into
the measurement patch.

The production trace may wrap this helper with
`nanodeploy::graph_bucket_selection` for correlation, but its traced duration
must not be used as the primary bucket timing because the annotation itself is
larger than the operation. The standalone CPU benchmark described below is
the source of record.

#### Add metadata ranges

At the full-Graph call site in `run_model`, add the following static ranges:

```text
nanodeploy::graph_metadata_injection.total
nanodeploy::flash_mla_metadata
nanodeploy::cuda_graph_replay
```

Inside `_copy_decode_context_to_graph_vars`, add:

```text
nanodeploy::graph_metadata_injection.q_dst_rows
nanodeploy::graph_metadata_injection.actual_attn_bs
nanodeploy::graph_metadata_injection.padding_metadata
```

Also wrap the complete post-selection region with a dynamic, outer annotation
that records shape context without copying tensor contents to the CPU:

```text
nanodeploy::decode_graph_step/
bs=<int>/master_bs=<int>/actual_attn_bs=<int>/graph_attn_bs=<int>
```

All four values already exist as Python integers on this path. Do not call
`.item()`, `.cpu()`, or synchronize to build the annotation.

Do not add CUDA Events or a `torch.cuda.synchronize()` to production
`run_model`. Device timing for the production path comes from trace
correlations; synchronized CUDA-Event timing belongs in the standalone
microbenchmark.

### 4.3 `scripts/benchmark_sp_graph_runtime_overheads.py` (new)

Implement three subcommands and write machine-readable JSON.

#### `bucket`

- CPU only; no Ray and no CUDA initialization.
- Construct a lightweight `ModelRunner` with fake Graph objects and the real
  production bucket lists/maps.
- Call `_select_decode_graph` so the benchmark cannot drift from production.
- Pre-generate the input cases outside the timed loop.
- Disable Python GC during samples and restore it afterward.
- Warm up for at least 10,000 calls.
- Measure at least 1,000,000 calls per repeat and seven repeats.
- Run a direct-known-key lookup control with the same input iteration and
  result consumption.
- Include two distributions:
  - `steady`: the same hot bucket on each call;
  - `boundary_mix`: cycle exact boundaries, just-over-boundary values, and the
    largest legal bucket.
- Cover local Graph, fixed-full SP, and dynamic sparse SP maps.
- Output raw repeat samples, p50/p95, absolute ns/call, and control-adjusted
  ns/call.

#### `metadata`

- One GPU; no Ray or communication is required.
- Construct representative graph variables and context tensors with the same
  dtypes, devices, shapes, and contiguous layouts used by ModelRunner.
- Invoke the production `_copy_decode_context_to_graph_vars` method through a
  lightweight runner and a fake rank context; do not duplicate its tensor
  operations in the benchmark.
- Time `total` and the three routing-only components separately.
- Measure host enqueue time in batches with `time.perf_counter_ns()` and one
  synchronization after the timed batch.
- Measure device elapsed time with CUDA Event pairs around the timed batch and
  divide by the number of iterations.
- Warm up first, use at least 1,000 measured iterations per repeat, and collect
  seven repeats.
- Cases:
  - `no_padding`: `actual_attn_bs == graph_attn_bs` and
    `actual_master_bs == graph_master_bs`;
  - `attn_padding`: a non-empty attention tail;
  - `master_padding`: both attention and local-master tails;
  - capacities `max_num_seqs=8` and the production upper bound
    `max_num_seqs=256`.

Output host and device p50/p95 independently. Include `q_dst_row_indices`
allocated bytes, active rows, actual and Graph attention rows, master batch
sizes, and block-table width.

#### `padding-graph`

- One GPU; no Ray or communication is required.
- Use DeepSeek-V3 Q row geometry: BF16,
  `row_numel = 128 * (512 + 64) = 73,728` elements.
- Capture two otherwise identical CUDA Graphs:
  - control Graph with one fixed anchor operation;
  - candidate Graph with the same anchor followed by
    `zero_padded_rows_triton`.
- Use a replay-mutable device `actual_rows` scalar exactly like production.
- Measure non-profiled replay latency with CUDA Events over batches of replays.
- Separately collect a lightweight CPU+CUDA profiler trace to obtain
  `cudaGraphLaunch` and `zero_padded_rows_kernel` events.
- Cases:
  - `no_tail`: `actual_rows == graph_rows`, isolating the unavoidable extra
    Graph node/kernel execution;
  - `small_tail`: one padded row;
  - `eos_tail`: one rank-local request has disappeared;
  - `worst_tail`: smallest legal actual row count for the selected Graph.
- Sweep representative `graph_rows` values `8`, `16`, and `64`.

The isolated Graph delta is:

```text
padding Graph replay - control Graph replay
```

It is the cleanest estimate of the new node itself, while the production
profile below captures interaction with communication and attention.

### 4.4 `scripts/profile_sp_graph_runtime_overheads.py` (new)

Add a deterministic one-node DP1/SP8/EP8 driver based on the setup in
`examples/dummy_prefill.py`. It must accept:

```text
--case {fixed_dense,fixed_tail,dynamic_sparse}
--model-path
--master-address
--ray-address
--output-dir
--profiler-start-step
--profiling-step
--enable-profiler
--profiler-mode
--seed
```

It must save `manifest.json` before engine creation with:

- Git commit and dirty status;
- `nanodeploy.__file__` on the driver;
- model path and model type;
- CUDA/PyTorch/Triton versions;
- Ray and master addresses;
- topology and all Graph/SP config values;
- request prompt/output lengths and seed;
- relevant `SLIME_*` values;
- profiler settings.

The ModelRunner logs must also print their resolved `nanodeploy.__file__` once
in runtime-overhead mode so worktree mistakes are visible.

Use these deterministic cases:

| Case | Requests | Prompt length | Output length | Policy | Purpose |
|---|---:|---:|---:|---|---|
| `fixed_dense` | 16 | 2,048 | 96 each | fixed SP8 | two master rows/rank, no attention tail |
| `fixed_tail` | 16 | 2,048 | one request 16, others 96 | fixed SP8 | stable post-completion `actual < graph` tail |
| `dynamic_sparse` | 8 | 65,536 | 96 each | DeepSeek-V3 bucket, SP5 interval | sparse destination rows and rounded attention Graph bucket |

Set `loop_count=1` in all three cases so one ModelRunner `run_count` is one
decode token step and the profiler start/stop steps below are unambiguous. Use
RoundRobin master placement. Use `max_num_seqs=2` for the two fixed cases so
16 requests yield two real master rows on every rank; use `max_num_seqs=1` for
the eight-request dynamic case. Keep `max_num_recv_seqs=16` and record the
resulting captured `master_bs` and `attn_bs` candidates in the manifest.

For `fixed_tail`, start profiling after step 24 so the short request has
completed and the surviving batch is stable. This is deterministic and does
not depend on sampling an EOS token. Verify from the dynamic Graph-step
annotations that `actual_attn_bs < graph_attn_bs` on at least one rank; reject
the run otherwise.

For `fixed_dense` and `dynamic_sparse`, start profiling after step 32. Profile
16 decode steps and generate enough tokens for the profiler to stop and flush.
Use `ignore_eos=True`, `dummy_prefill=True`, `dummy_weight=True`,
`moe_routing_simulation_strategy="perfect_eplb"`, and seed 0.

In addition to profiled runs, support `--enable-profiler` being absent. These
non-profiled runs are used for the net decode-step/ITL comparison; the profile
run is used only for breakdown.

### 4.5 `utils_analysis/summarize_sp_graph_overhead_trace.py` (new)

Parse all `*.pt.trace.json` files below one run directory and emit:

```text
summary.json
rank_components.csv
step_samples.csv
```

Parsing rules:

1. Locate static and dynamic `nanodeploy::*` user annotations.
2. For a metadata annotation, find CUDA runtime events on the same CPU thread
   whose timestamps lie inside the annotation. Collect their correlation IDs,
   then match GPU memcpy/memset/kernel events carrying those IDs.
3. For every `cudaGraphLaunch`, collect device events with the same
   `args.correlation`. Compute:
   - host launch duration;
   - sum of device durations;
   - device span `max(end) - min(start)`;
   - padding-kernel count and duration sum.
4. Match launches to the enclosing `nanodeploy::cuda_graph_replay` and dynamic
   Graph-step annotation.
5. Discard the first two valid Graph steps in each trace.
6. Fail loudly when:
   - a production full-Graph trace has no `cudaGraphLaunch`;
   - a candidate SP trace has no `zero_padded_rows_kernel`;
   - a fixed-tail or dynamic-sparse run has no `actual < graph` annotation;
   - rank counts, replay counts, or kernel counts vary unexpectedly.

For synchronous decode, report both the median across ranks and the slowest
rank. The slowest-rank value is the primary production number. For potentially
overlapping device events, report both sum and span; do not present summed
kernel durations as a critical-path latency.

### 4.6 Tests

Add or extend these CPU tests:

- `tests/test_sp_graph_bucket_selection.py`:
  exact boundaries, just-over-boundary cases, fixed-full candidates, dynamic
  candidates, local Graph, and overflow errors for `_select_decode_graph`;
- `tests/test_routing_config.py`:
  valid/default/invalid `profiler_mode` behavior;
- parser unit tests with a tiny synthetic Chrome trace containing one user
  annotation, two runtime correlations, one Graph launch, and one padding
  kernel;
- keep the existing padding semantics in
  `tests/test_pd_decode_sp_batch_semantics.py` unchanged except for reuse of a
  shared test builder if the metadata microbenchmark needs it.

Run the CPU regression before any GPU experiment:

```bash
python3 -m pytest \
  tests/test_sp_graph_bucket_selection.py \
  tests/test_routing_config.py \
  tests/test_pd_decode_sp_batch_semantics.py
```

## 5. Experiment matrix

### 5.1 CPU bucket measurement

Run on both `before` and `candidate` worktrees if the extraction applies
cleanly; otherwise run the production-equivalent helper on the candidate and
record that the selection algorithm is unchanged across the target commit.

```bash
python3 scripts/benchmark_sp_graph_runtime_overheads.py bucket \
  --iterations 1000000 \
  --repeats 7 \
  --output "$RUN_ROOT/candidate/bucket.json"
```

The minimum reported rows are:

```text
local/steady
fixed_full/steady
fixed_full/boundary_mix
dynamic_sparse/steady
dynamic_sparse/boundary_mix
```

### 5.2 One-GPU component microbenchmarks

After configuring the three required `SLIME_*` variables, run with elevated
permission:

```bash
python3 scripts/benchmark_sp_graph_runtime_overheads.py metadata \
  --device cuda:0 \
  --iterations 1000 \
  --repeats 7 \
  --output "$RUN_ROOT/candidate/metadata.json"

python3 scripts/benchmark_sp_graph_runtime_overheads.py padding-graph \
  --device cuda:0 \
  --warmup 100 \
  --iterations 10000 \
  --repeats 7 \
  --output "$RUN_ROOT/candidate/padding_graph.json" \
  --trace-dir "$RUN_ROOT/candidate/padding_graph_traces"
```

If 10,000 worst-tail replays make one repeat excessively long, reduce only
that case to 2,000 and record the per-case iteration count. Do not silently use
different counts.

### 5.3 Start Ray for the production profiles

Use explicit site-specific addresses rather than committing a node IP:

```bash
export PROFILE_NODE_IP=<eight-GPU-node-IP>
export PROFILE_RAY_PORT=<free-Ray-GCS-port>
export PROFILE_MASTER_PORT=<free-torch-master-port>
export PROFILE_RAY_ADDRESS="$PROFILE_NODE_IP:$PROFILE_RAY_PORT"
export PROFILE_MASTER_ADDRESS="$PROFILE_NODE_IP:$PROFILE_MASTER_PORT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ray start --head \
  --node-ip-address "$PROFILE_NODE_IP" \
  --port "$PROFILE_RAY_PORT" \
  --num-gpus 8
ray status
```

Ray commands and every following GPU run require elevated permission. Reuse
the allocated Ray cluster across matched runs only after verifying that the
previous engine released all actors and placement groups. Stop it at the end
with the same proxy-free environment:

```bash
ray stop
```

### 5.4 Production profile commands

For each instrumented worktree and each case:

```bash
python3 -u scripts/profile_sp_graph_runtime_overheads.py \
  --case fixed_dense \
  --model-path /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
  --master-address "$PROFILE_MASTER_ADDRESS" \
  --ray-address "$PROFILE_RAY_ADDRESS" \
  --enable-profiler \
  --profiler-mode runtime_overhead \
  --profiler-start-step 32 \
  --profiling-step 16 \
  --output-dir "$RUN_ROOT/candidate/fixed_dense/repeat_0"

python3 -u scripts/profile_sp_graph_runtime_overheads.py \
  --case fixed_tail \
  --model-path /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
  --master-address "$PROFILE_MASTER_ADDRESS" \
  --ray-address "$PROFILE_RAY_ADDRESS" \
  --enable-profiler \
  --profiler-mode runtime_overhead \
  --profiler-start-step 24 \
  --profiling-step 16 \
  --output-dir "$RUN_ROOT/candidate/fixed_tail/repeat_0"

python3 -u scripts/profile_sp_graph_runtime_overheads.py \
  --case dynamic_sparse \
  --model-path /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
  --master-address "$PROFILE_MASTER_ADDRESS" \
  --ray-address "$PROFILE_RAY_ADDRESS" \
  --enable-profiler \
  --profiler-mode runtime_overhead \
  --profiler-start-step 32 \
  --profiling-step 16 \
  --output-dir "$RUN_ROOT/candidate/dynamic_sparse/repeat_0"
```

Replace `candidate` with `before` for the matched baseline and repeat indices
`0..4`. Use a new master port, or prove that the previous process group is
fully gone, before starting another engine.

Run the parser after every profile so a missing annotation or kernel is found
before an expensive matrix completes:

```bash
python3 utils_analysis/summarize_sp_graph_overhead_trace.py \
  "$RUN_ROOT/candidate/fixed_dense/repeat_0" \
  --output-dir "$RUN_ROOT/candidate/fixed_dense/repeat_0/analysis"
```

### 5.5 Non-profiled paired latency run

Repeat `fixed_dense` and `fixed_tail` at least five times on `before` and
`candidate` with the same driver but without `--enable-profiler`. Save per-step
latency after warmup. The primary net result is the median of paired run
medians:

```text
delta_us = candidate decode-step median - before decode-step median
delta_pct = delta_us / before decode-step median * 100
```

The production profile explains this delta; it does not replace it. If a
profiled delta and non-profiled delta disagree, trust the non-profiled result
for end-to-end impact and investigate profiler perturbation.

## 6. Result tables

The final note must include at least these tables.

### 6.1 Bucket selection

| Topology/case | Distribution | Absolute p50 ns | p95 ns | Direct-lookup control ns | Adjusted overhead ns |
|---|---|---:|---:|---:|---:|

### 6.2 Metadata injection

| Version | Case | max_num_seqs | Scope | Host p50 us | Device p50 us | p95 us | Bytes | Active rows |
|---|---|---:|---|---:|---:|---:|---:|---:|

Include separate rows for `q_dst_rows`, `actual_attn_bs`,
`padding_metadata`, routing-only sum, and total injection. Add a before/after
delta column for the total production scope.

### 6.3 Kernel and Graph launch/replay

| Version | Case | Rank aggregate | cudaGraphLaunch host p50 us | Graph device span p50 us | Padding kernels/replay | Padding sum/replay us | Non-profiled step p50 us |
|---|---|---|---:|---:|---:|---:|---:|

For the isolated padding Graph, add control, no-tail, small-tail, EOS-tail, and
worst-tail rows. For production, report the median-rank and slowest-rank rows.

## 7. Validation and rejection criteria

Reject and rerun a sample when any of the following holds:

- the imported NanoDeploy path or Git commit is not the intended worktree;
- fewer than eight worker traces are produced;
- the trace contains capture/compilation in the measured window;
- no `cudaGraphLaunch` or candidate padding kernel is found;
- Graph-step annotations do not show the intended actual/Graph row relation;
- the number of model Graph replays differs across ranks;
- a request completes inside a supposedly stable measured window, except for
  completion that occurs before the fixed-tail window;
- CUDA errors, Ray actor restarts, OOM, request rejection, or scheduler backlog
  are observed;
- before and candidate use different Graph shapes or request topology for a
  matched case.

The parser must also verify that the padding-kernel count per replay is stable
and explain it using the model's attention-layer count. A count mismatch is a
measurement failure, not a value to average away.

## 8. Interpretation rules

- A CPU unit test is evidence of bucket-selection correctness, not runtime
  overhead. Cite the standalone benchmark number.
- A single D2D copy duration is not the whole routing-metadata cost. Cite both
  the routing-only sum and complete injection scope.
- The existing `profiler_traces/dst_rows_perf_20260821` traces measure a
  communication-only Graph. They do not contain `zero_padded_rows_kernel` and
  must not be cited for the model-level padding/Graph-launch overhead.
- Absolute kernel time answers “how expensive is the new primitive”; the
  before/after Graph span and non-profiled decode delta answer “how much did
  the system slow down.” Report both.
- If the candidate's total metadata injection is faster than the old dense
  fixed-SP path, report the negative overhead rather than presenting only the
  new copy cost.
- If the end-to-end median changes by less than 1% and remains within the
  measured run-to-run spread, describe it as within noise and still provide
  the component microseconds.

## 9. Implementation and execution sequence

1. Add the profiler mode, guarded ranges, bucket-selection helper, deterministic
   driver, component benchmark, parser, and CPU tests in one profiling-only
   commit.
2. Run the focused CPU tests and the CPU bucket benchmark.
3. Run a one-GPU smoke of `metadata` and `padding-graph`; verify JSON and trace
   parsing before increasing iterations.
4. Apply the same instrumentation commit to `before` and `candidate`
   worktrees.
5. Run one `fixed_dense` production profile on each version and compare Graph
   shapes, annotations, trace counts, and import paths.
6. Run the remaining five-repeat profiled and non-profiled matrix in paired
   order.
7. Run one smoke on current integration HEAD.
8. Write and commit the concise result note, exact commands, commit hashes,
   topology, profiler settings, tables, and interpretation. Keep raw trace and
   benchmark artifacts uncommitted under `bench_logs/`.

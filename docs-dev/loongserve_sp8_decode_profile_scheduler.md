# LoongServe-style SP<=8 Pure Decode Profile and Scheduler Plan

## Scope

This document defines the profile plan and runtime scheduler policy for choosing
LoongServe-style decode attention DoP from:

```text
d_attn in {1, 2, 4, 8}
```

The intended benchmark mode is pure decode only:

```text
dummy_prefill=True
mode=decode
loop_count=1
cuda_graph_mode=full
```

Prefill latency and prefill scale-down are out of scope. Expert routing skew is
also out of scope under forced uniform EP. The scheduler decision here is only:

```text
given an active decode batch and existing KV placement,
which intra-node attention SP/DoP should be used for the next decode step?
```

Normal operation must not use cross-node attention SP. `d_attn=8` is the maximum
normal target because it maps to one 8-GPU node.

The concrete execution checklist for collecting the missing profile data is:

```text
docs-dev/loongserve_sp8_decode_profile_execution_plan.md
```

## Paper Basis

LoongServe's paper is "LoongServe: Efficiently Serving Long-Context Large
Language Models with Elastic Sequence Parallelism":

```text
https://arxiv.org/abs/2404.09526
```

The relevant design points are:

- Elastic Sequence Parallelism changes sequence-parallel DoP at iteration
  granularity, without repartitioning model parameters.
- The global manager uses a Scaling Information Base (SIB), populated by
  profiling, to guide runtime DoP, batching, placement, and scaling decisions.
- Decode often runs at smaller DoP because communication overhead can dominate.
- Decode scale-up happens when GPU memory is insufficient or when decode becomes
  compute-bound.
- In LoongServe, decode compute-bound detection is based on a pre-profiled batch
  size threshold because FFN becomes the bottleneck in their target setting.
- Multi-master distributed decoding is the mechanism used after scale-up: each
  master owns a subset of requests, writes newly generated KV locally, and
  shares attention work with peer ranks without migrating historical KV.

For our forced-uniform EP setting, the paper's method should be preserved but
the profiled threshold key must change:

```text
LoongServe original:
    profile decode batch-size threshold for FFN/local-layer compute-bound scale-up

NanoDeploy uniform EP path:
    profile full-layer decode latency as a function of attention work and d_attn
```

The reason is that FFN/expert work is already distributed by uniform EP. The
remaining elastic-SP decision is dominated by attention work, attention
communication, scheduler overhead, and memory fit.

## LoongServe Code Mapping

Local reference repo:

```text
/mnt/nvme1n1/ml_research/linbinbin1/LoongServe
```

Important code concepts:

- `Batch.occupied_instances` is the active ESP group for a decode batch.
- `Req.cur_kv_len_list` is per-rank KV ownership and maps to NanoDeploy
  `Sequence.block_ctx(...).num_dispatched_tokens`.
- `_schedule_decode_batch_list()` first checks whether the current occupied
  ranks have enough token capacity for the next decode step.
- If memory is insufficient, it scales up by adding idle instances or merges
  batches to share capacity.
- It then creates `num_sp_master_ranks` and `mini_batch_size_list`, driven by
  `min_comp_bound_decoding_batch_size`, to enable multi-master decoding.
- `_decode_batch()` sends `occupied_instances`, logical SP rank,
  `num_sp_master_ranks`, and `mini_batch_size_list` to the workers.

NanoDeploy already has related pieces:

- `loongserve_decode_scheduler`
- `loongserve_min_comp_bound_batch_size`
- `loongserve_occupied_instances`
- per-SP dispatched KV token counters
- `sp_send_counts`, `sp_recv_counts`, `sp_size_hist_per_dp`
- dynamic SP bucket/long-short policies

But the existing bucket policy is length-rule based. The scheduler proposed here
should become profile-table based, matching LoongServe's SIB idea.

## DPSK Decode Workload

Workload reference datasets:

```text
DPSK-issue1%:
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv

DPSK-issue5%:
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv
```

Dataset summary:

| dataset | rows | long rows | prompt mean | prompt p99 | prompt p99.9 | total p99 | total p99.9 | max total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DPSK-issue1 | 60000 | 600 | 4908.7 | 12655 | 819589 | 12893 | 820235 | 996462 |
| DPSK-issue5 | 60000 | 3000 | 22890.0 | 659519 | 970551 | 660008 | 971230 | 996596 |

Long-context request counts:

| dataset | total>=64K | total>=128K | total>=256K | total>=512K | total>=786K | total>=900K |
|---|---:|---:|---:|---:|---:|---:|
| DPSK-issue1 | 600 | 561 | 406 | 190 | 72 | 47 |
| DPSK-issue5 | 3000 | 2814 | 2107 | 1059 | 408 | 232 |

Output lengths are not very long:

| dataset | output p50 | output p90 | output p99 | output>=1024 | output>=2048 |
|---|---:|---:|---:|---:|---:|
| DPSK-issue1 | 596 | 805 | 1079 | 897 | 13 |
| DPSK-issue5 | 594 | 806 | 1084 | 910 | 14 |

Implication for pure decode:

- The important decode variable is `L_i = prompt_len_i + generated_tokens_i`.
- Most decode steps are short requests, but long prompt requests stay active for
  hundreds of output steps.
- For batch size `B`, the chance of at least one long request is meaningful:

| long ratio | B=16 | B=32 | B=64 | B=128 |
|---:|---:|---:|---:|---:|
| 1% | 14.9% | 27.5% | 47.4% | 72.4% |
| 5% | 56.0% | 80.6% | 96.2% | 99.9% |

So profile must include mixed batches, not only uniform-length batches.

These datasets are used to choose meaningful controlled profile buckets and to
validate the final scheduler on realistic decode mixtures. They are not the
primary source of the low-level performance threshold. The threshold itself
should come from controlled SIB profiles over `(B, W_attn, L distribution,
d_attn)`, following the LoongServe paper's profile-first method.

## Existing 16-GPU Anchor Data

Existing files:

```text
docs-dev/profile-results/loongserve_16g_pilot_20260709.csv
docs-dev/profile-results/loongserve_16g_longctx_640k_20260709.csv
docs-dev/profile-results/loongserve_16g_longctx_1m_20260709.csv
```

Run properties:

```text
SP/EP: 16/16
mode: pure decode
dummy_weight=True
dummy_prefill=True
cuda_graph_mode=full
RPC path: DLSlime RPC
```

Single local GPU raw KV capacity observed in this setup:

```text
num_local_kvcache_blocks = 14273
kvcache_block_size = 64
raw capacity = 913472 tokens
```

Current full-step mean latency, limited to `d in {1,2,4,8}`:

| B | L | W_attn | d=1 | d=2 | d=4 | d=8 | best under SP<=8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 40960 | 655360 | 111.18 | 1743.64 | 989.89 | 575.14 | 1 |
| 16 | 49152 | 786432 | 111.38 | 1777.76 | 1019.23 | 565.60 | 1 |
| 16 | 53248 | 851968 | 113.09 | 1750.04 | 1002.84 | 568.66 | 1 |
| 16 | 65536 | 1048576 | OOM/skip | 1673.00 | 1038.36 | 538.84 | 8 |

Interpretation:

- At `B=16`, performance-only scale-up is not justified while `d=1` fits.
- When memory forces `d>1`, the best capped choice in the current data is `d=8`.
- `d=2` and `d=4` are currently poor served-path choices, likely due to static
  full-world NCCL/synchronization overhead. They should not be selected for
  performance unless future subgroup profiling proves otherwise.
- The existing `d=16` result is faster for the 1M point, but it crosses the
  proposed normal SP cap and should not be used for the intra-node policy.

## What To Profile

The profile source of truth must be full-layer or full-step pure decode latency,
not only the partial attention kernel.

Required key:

```text
(B, W_attn_bucket, L_p90_bucket, L_max_bucket, d_attn) -> latency distribution
```

where:

```text
B = active decode token count in this step
L_i = current KV/context length of request i
W_attn = sum_i L_i
d_attn in {1,2,4,8}
```

Record at least:

```text
B
L_avg
L_p50
L_p90
L_max
W_attn
d_attn
step_mean_ms
step_p50_ms
step_p90_ms
model_mean_ms
scheduler_mean_ms
postprocess_mean_ms
sp_size_hist
master_counts
occupied_instances
append_instances
sp_send_counts
sp_recv_counts
```

When worker-side timers are available, also record:

```text
T_q_dispatch
T_partial_attn
T_res_lse_return
T_lse_merge
T_attn_total
T_ep_dispatch
T_expert_compute
T_ep_combine
T_ep_total
T_layer_total
```

The scheduler table should be generated from full-layer or full-step latency.
Attention-only timers are diagnostic.

## Profile Matrix

Use two profile families.

### 1. Controlled Uniform Work Points

These points isolate `B`, `L`, `W_attn`, and `d_attn`.

```text
d_attn in {1,2,4,8}
B in {1,2,4,8,16,32,64,128}
```

Comparable memory-fit points:

```text
W_attn in {64K,128K,256K,512K,640K,768K,850K}
L = ceil(W_attn / B)
skip points where L is too small to split across d_attn
```

Memory-forced points:

```text
W_attn in {1M,2M,4M,6M,7M}
L = ceil(W_attn / B)
skip any d_attn that cannot fit
```

For the current 16-GPU H200 setup, `d=1` should be skipped when the predicted
per-rank KV tokens exceed the usable single-rank capacity. With the observed raw
capacity of 913472 tokens, use a safety margin instead of the raw number:

```text
usable_tokens_per_rank = floor(raw_capacity * safety_margin)
safety_margin = 0.90 initially
```

The 640K minimum long-context requirement is covered by `W_attn>=640K`.

### 2. DPSK Dataset-replay Validation Points

These points validate the controlled SIB on realistic length mixtures. They
should not directly overwrite the controlled threshold table. If replay exposes
a mismatch, add the missing controlled bucket or shape variant, then regenerate
the SIB.

Build synthetic pure-decode batches from the CSVs:

```text
L_i(step) = prompt_len_i + step
active if step < output_len_i
```

No prefill is run. KV cache is preconstructed to `L_i(step)`.

Use these batch families for each dataset:

```text
short-only:
    sample rows with total_len < 8K

mixed-natural:
    sample rows from the full CSV distribution

mixed-forced-long:
    force 1 long request plus B-1 short requests

long-heavy:
    sample only rows with total_len >= 64K

tail-stress:
    sample rows with total_len >= 512K
```

Use active batch sizes:

```text
B in {16,32,64,128}
```

Use decode step offsets:

```text
step in {0,128,512,1024}
```

Skip `step` values exceeding a sampled request's output length.

For each profile point:

```text
warmup = 20
measure = 100
repeat = 3 independent samples for dataset-replay points
```

For fast pilot/debug:

```text
warmup = 3
measure = 10
```

Only full warmup/measure runs should be used to produce production thresholds.

## Threshold Extraction

Build a profile table:

```text
ProfileRow:
    workload_kind
    dataset
    B_bucket
    W_attn_bucket
    L_p90_bucket
    L_max_bucket
    d_attn
    viable
    step_p50_ms
    step_p90_ms
    model_p50_ms
    model_p90_ms
```

For a runtime batch, compute candidate set:

```text
D = {1,2,4,8}
D = D intersect ranks available inside the current node-local SP group
D = D intersect d values that fit memory
```

Then choose by profile:

```text
best_latency = min(T_profile(batch, d) for d in D)
d_profile = smallest d in D such that:
    T_profile(batch, d) <= 1.05 * best_latency
```

Use p90 for latency-sensitive scheduling and p50 for throughput-oriented
scheduling. The first production policy should use p90 because long-context
decode is tail-sensitive.

Pairwise thresholds can be derived for debugging and simpler configs:

```text
W_1to2(B_bucket, L_p90_bucket):
    first W where d=2 beats d=1 by >=5%

W_2to4(...):
    first W where d=4 beats d=2 by >=5%

W_4to8(...):
    first W where d=8 beats d=4 by >=5%
```

But runtime should prefer table lookup over hard-coded pairwise thresholds.

## Runtime Scheduler Policy

### Step 1: Compute Runtime Features

For each decode batch:

```text
B = number of scheduled real sequences
L_i = sum(num_dispatched_tokens_i)
W_attn = sum_i L_i
L_avg, L_p50, L_p90, L_max
current_d = number of active KV owner ranks for this batch
```

Use current block accounting rather than prompt length if available:

```text
Sequence.block_ctx(ACTIVE).num_dispatched_tokens
```

### Step 2: Enforce Node-local Candidate Groups

Define local SP groups:

```text
node0: ranks [0,1,2,3,4,5,6,7]
node1: ranks [8,9,10,11,12,13,14,15]
```

For 32 GPUs, define four such 8-rank groups.

The normal scheduler must only pick participants from one local group. If the
batch already owns KV on one rank, choose that rank's local group. If it owns KV
on multiple groups because of legacy placement, do not expand cross-node; either
drain toward one group or mark the batch as legacy/cross-node.

### Step 3: Compute Memory-viable DoP

For each candidate `d`:

```text
simulate placement of existing KV owners plus append targets
fit(d) = every selected rank remains below usable token capacity
```

Use actual block/KV ownership for mixed lengths. A simple upper-bound fallback
is:

```text
tokens_per_rank_after = ceil(W_attn / d) + ceil(B / d)
fit(d) = tokens_per_rank_after <= usable_tokens_per_rank
```

Then:

```text
d_mem = smallest d in {1,2,4,8} where fit(d)
```

If no `d<=8` fits:

```text
normal policy: do not cross node
action: preempt/offload/queue/reject according to serving policy
optional emergency flag: allow cross-node SP only when explicitly enabled
```

### Step 4: Pick Profile Target

Only evaluate profile candidates that are memory viable:

```text
D_viable = {d in {1,2,4,8} | d >= d_mem and fit(d)}
d_target = profile_near_optimal(D_viable)
```

Current empirical rule until more data is collected:

```text
if d=1 fits:
    choose d=1 unless the profile table shows d>1 improves p90 by >=5%
elif d=8 fits:
    choose d=8
elif d=4 fits:
    choose d=4
elif d=2 fits:
    choose d=2
else:
    no normal intra-node SP plan exists
```

This rule is intentionally conservative. It matches the current 16-GPU data:
`d=1` is fastest up to 852K total KV tokens at B=16, and `d=8` is the best
SP<=8 choice at the 1M memory-forced point.

### Step 5: Hysteresis and Scale Events

Scale up immediately when memory requires it:

```text
if current_d does not fit next step:
    scale up to d_target
```

For performance-only scale-up:

```text
scale up only if:
    predicted_p90(current_d) - predicted_p90(d_target) >= max(5%, 0.1 ms/layer)
and the same target is preferred for >= 3 consecutive decode steps
```

For scale-down:

```text
scale down only if:
    lower d fits for >= 16 consecutive decode steps
or scheduler has node-local resource pressure
```

If lower `d` is predicted faster, scale down sooner after a short cooldown:

```text
cooldown = 4 decode steps
```

The scale-down target is the lowest near-optimal memory-viable `d`.

### Step 6: Multi-master Planning

For `d>1`, split the active requests across masters as evenly as possible:

```text
masters = selected local ranks
per_master_new_kv ~= ceil(B / d)
```

Tie-breakers:

```text
1. prefer ranks that already own KV for the batch
2. then idle ranks in the same local group
3. then ranks with most free KV capacity
4. keep rank order stable to avoid graph/metadata churn
```

Historical KV does not need to be migrated for scale-up. Newly generated KV is
written to the selected master/append ranks, matching LoongServe's multi-master
decode idea.

## Important Implementation Constraint

The scheduler policy cannot deliver intra-node SP<=8 performance if the
communication backend still launches full-world SP16 collectives for a `d=8`
batch.

For the no-cross-node policy to be real, the execution path needs one of:

```text
1. active subgroup NCCL/P2P for the selected local ranks, or
2. DLSlime RPC packing that only communicates among occupied local ranks.
```

Validation must check:

```text
occupied_instances are within one local group
sp_send_counts/sp_recv_counts outside the group are zero
NCCL/RPC traces do not include cross-node ranks for d<=8
```

Until then, profile results with `d=8` may still include hidden SP16 overhead.

## Evaluation Plan

Run three scheduler modes:

```text
fixed_d1:
    baseline single-master when memory fits

fixed_d8:
    intra-node long-request baseline

profile_scheduler:
    SIB/profile-table driven d in {1,2,4,8}
```

For each mode:

```text
dataset in {DPSK-issue1, DPSK-issue5}
traffic rate sweep matching existing DPSK scripts
dummy_prefill=True
pure decode only
```

Metrics:

```text
decode step p50/p90/p99
model step p50/p90/p99
throughput tokens/s
request output latency p50/p90/p99
scale-up count
scale-down count
time spent at each d
number of no-fit d<=8 events
node-locality violations
```

Acceptance criteria:

```text
1. Scheduler never chooses d>8 in normal mode.
2. Scheduler never crosses node for d<=8.
3. d=1 is selected for B=16, W_attn<=852K unless new profile data disproves it.
4. d=8 is selected for memory-forced 1M-token B=16 cases under the SP<=8 cap.
5. DPSK issue1/issue5 end-to-end pure-decode runs show the scheduler is no worse
   than fixed_d1 on normal short-heavy traffic, and better than fixed_d1 on
   memory-forced long-context traffic where fixed_d1 cannot run.
```

## Implementation Tasks

1. Extend the profile runner to generate dataset-replay decode batches from the
   DPSK CSVs.
2. Save controlled profile rows to `docs-dev/profile-results/` and derive the
   threshold/SIB table as
   `docs-dev/profile-results/loongserve_sp8_decode_sib_*.json`.
   Dataset replay rows are saved beside them as validation artifacts.
3. Add scheduler config:

```text
loongserve_max_local_decode_sp = 8
loongserve_decode_profile_path = ""
loongserve_decode_profile_near_optimal_ratio = 1.05
loongserve_decode_profile_abs_gain_ms = 0.1
loongserve_decode_cross_node_sp = false
```

4. Replace performance-only use of `loongserve_min_comp_bound_batch_size` with
   profile-table lookup. Keep the existing min-batch knob only as a fallback.
5. Add node-local rank group selection.
6. Add active subgroup communication or validate DLSlime occupied-rank-only
   communication.
7. Add unit tests for:

```text
d_mem selection
near-optimal d selection
scale-up hysteresis
scale-down hysteresis
node-local candidate filtering
d>8 rejection
```

8. Add end-to-end dummy-prefill pure-decode tests for:

```text
B=16,W=640K -> d=1
B=16,W=850K -> d=1
B=16,W=1M   -> d=8 under SP<=8 cap
DPSK-issue1 mixed-natural replay
DPSK-issue5 mixed-natural replay
```

## Current Recommendation

Until the expanded profile table is available, use this provisional scheduler:

```text
if d=1 fits:
    d_target = 1
elif d=8 fits:
    d_target = 8
elif d=4 fits:
    d_target = 4
elif d=2 fits:
    d_target = 2
else:
    no normal intra-node SP<=8 plan
```

This is not the final LoongServe-style threshold policy. It is the safest
interim policy because it is consistent with existing 16-GPU served-path data
and avoids choosing `d=2`/`d=4` for performance when current measurements show
they are much slower.

The final scheduler must be generated from the profile/SIB table described
above.

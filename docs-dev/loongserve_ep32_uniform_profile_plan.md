# LoongServe-style Decode DoP Profile Plan for Uniform EP32

## Goal

This document defines the profiling plan for LoongServe-style decode attention
parallelism when expert parallelism is fixed to uniform EP32.

With forced uniform EP32, expert routing skew is not a scheduling dimension. The
remaining performance question is:

```text
For a fixed EP32 uniform FFN path, when does increasing attention DoP reduce
full-layer decode latency enough to justify the extra communication and
synchronization?
```

This differs from the original LoongServe threshold, which uses a decode
compute-bound batch-size threshold for scale-up. In the EP32 uniform setting, the
threshold should be based on attention work, not FFN routing or expert skew.

## Variables

Profile these variables:

```text
B: active decode tokens in this step
L: current KV/context length per request
W_attn: sum_i L_i, or B * L for uniform length
d_attn: LoongServe-style attention degree of parallelism
```

For uniform synthetic decode:

```text
W_attn = B * L
```

The primary lookup table is:

```text
(B, L, d_attn) -> decode latency
```

For runtime scheduling, the more stable threshold key is:

```text
(W_attn, d_attn) -> decode latency
```

## Required Tables

### Attention-only Table

Measure:

```text
T_attn(B, L, d_attn)
```

This must include the complete attention data path:

```text
Q dispatch
partial attention kernel
Res/LSE return
LSE merge
attention-side synchronization
```

Do not use the partial attention kernel alone as the threshold source. Increasing
`d_attn` reduces per-rank attention compute, but increases Q/Res/LSE
communication and synchronization.

### EP32 Uniform Table

Measure:

```text
T_ep32_uniform(B)
```

This table should be close to a function of B under forced uniform routing. Its
role is not to choose attention DoP directly. It tells us whether attention
optimization is visible at full-layer scope. For example, if attention improves
by 0.4 ms but EP32 FFN costs 5 ms, the layer-level gain is small.

### Full-layer Table

Measure:

```text
T_layer(B, L, d_attn, EP32-uniform)
```

This is the source of truth for runtime thresholds. Attention communication and
EP dispatch/combine can contend for the same fabric and may have barriers or
stream dependencies, so independent component timings are not sufficient.

## Threshold Extraction

For each `(B, L)`, find:

```text
d_best(B, L) = argmin_d T_layer(B, L, d)
```

At runtime, prefer the smallest near-optimal DoP instead of the numerically best
DoP:

```text
d_perf = min d such that T_layer(B, L, d) <= 1.05 * min_d T_layer(B, L, d)
```

Pairwise thresholds can then be written as:

```text
B_th(L, 1->2): smallest B where d=2 beats d=1 by at least 5%
B_th(L, 2->4): smallest B where d=4 beats d=2 by at least 5%
B_th(L, 4->8): smallest B where d=8 beats d=4 by at least 5%
```

Prefer an attention-work version for production:

```text
if W_attn < W_1to2:
    d_attn = 1
elif W_attn < W_2to4:
    d_attn = 2
elif W_attn < W_4to8:
    d_attn = 4
else:
    d_attn = 8 or higher
```

The final target also needs the memory constraint:

```text
d_mem = smallest attention DoP that fits current KV cache
d_perf = smallest near-optimal DoP from profile table
d_target = max(d_mem, d_perf)
```

Add hysteresis before actually changing DoP:

```text
scale only if predicted improvement is > 5% or > 0.1 ms/layer
```

## Benchmark Matrix

### 16-GPU Pilot

Use this matrix to validate the harness and detect obvious current-path
regressions:

```text
B in {16, 32, 64}
L in {1K, 4K, 16K}
d_attn in {1, 2, 4, 8, 16}
warmup = 5 to 20 steps
measure = 20 to 100 steps
```

The current 16-GPU cluster can only approximate the final EP32 setting. It runs
SP16/EP16 and validates the LoongServe-style attention scheduling path, DLSlime
RPC transport, full CUDA graph decode, and metadata consistency.

### 32-GPU Final

Use this matrix for the actual EP32 threshold table:

```text
B in {1, 2, 4, 8, 16, 32, 64, 128, 256}
L in {1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K}
d_attn in {1, 2, 4, 8, 16, 32}
warmup = 20 steps
measure = 100 steps
```

The decode benchmark must preconstruct KV cache and execute decode steps only.
Prefill must not be included.

## Metrics to Record

Minimal full-layer fields:

```text
B
L_avg
L_p90
L_max
W_attn
d_attn
T_attn_total
T_ep_total
T_layer_total
T_decode_step_total
```

Preferred breakdown:

```text
T_q_dispatch
T_partial_attn
T_res_lse_return
T_lse_merge
T_attn_total

T_gate
T_ep_dispatch
T_expert_compute
T_ep_combine
T_ep_total

T_layer_total
T_decode_step_total
```

Driver-side timing through `LLM.executor.run()` is useful for served-path
validation, but final thresholds should be derived from worker-side CUDA event or
profiler measurements around the layer and attention subranges.

## Current 16-GPU Pilot Result

Command:

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --batches 16 \
  --lengths 1024,4096,16384 \
  --dops 1,2,4,8,16 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --out docs-dev/profile-results/loongserve_16g_pilot_20260709.jsonl
```

Run properties:

```text
cluster: 16 H200 GPUs across 2 Ray nodes
ray_address: 10.102.252.174:6380
master_address: 10.102.252.174:29644
SP/EP: 16/16
mode: pure decode
weights/prefill: dummy_weight=True, dummy_prefill=True
CUDA graph: full
RPC path: DLSlime RPC
```

Raw outputs:

```text
docs-dev/profile-results/loongserve_16g_pilot_20260709.jsonl
docs-dev/profile-results/loongserve_16g_pilot_20260709.csv
```

The runner sets `loongserve_min_comp_bound_batch_size = ceil(B / d_attn)`.
Recorded scheduler metadata confirmed the intended DoP:

```text
d=1:  sp_size_hist={1: 16}
d=2:  sp_size_hist={2: 16}
d=4:  sp_size_hist={4: 16}
d=8:  sp_size_hist={8: 16}
d=16: sp_size_hist={16: 16}
```

Step mean latency in milliseconds:

| B | L | W_attn | d=1 | d=2 | d=4 | d=8 | d=16 | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 16384 | 94.79 | 1742.89 | 905.44 | 559.54 | 338.22 | 1 |
| 16 | 4096 | 65536 | 98.90 | 1742.72 | 905.07 | 555.67 | 351.67 | 1 |
| 16 | 16384 | 262144 | 100.30 | 1775.00 | 915.79 | 553.32 | 341.28 | 1 |

Pilot interpretation:

- At `B=16`, attention DoP scale-up is not beneficial in the current served path.
- `d=1` is the smallest near-optimal DoP and also the absolute best point for all
  tested context lengths.
- The weak dependence on L suggests fixed overhead, SP communication,
  synchronization, DLSlime transfer, or CUDA graph scheduling dominates this
  small-B pilot.
- The non-monotonic `d=2/4/8/16` behavior means the runtime policy should not
  infer a threshold from DoP alone. It needs measured `(B, L, d)` or
  `(W_attn, d)` profile data.

Immediate scheduling implication:

```text
For B=16 and L <= 16K on the current 16-GPU SP16/EP16 path,
profile-based compute scale-up should keep d_attn=1.
```

This does not prove `d=1` is globally optimal. The next useful points are larger
`B` and larger `W_attn`, especially `B in {32, 64, 128}` and `L >= 16K`.

## Current 16-GPU Long-context Result

The first pilot only reached `W_attn=262K`, which is too small for evaluating
long-request SP behavior. A larger run was added with `B=16` and total KV tokens
at or above 640K.

Command for the all-DoP comparable run:

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --batches 16 \
  --lengths 40960,49152,53248 \
  --dops 1,2,4,8,16 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --out docs-dev/profile-results/loongserve_16g_longctx_640k_20260709.jsonl
```

Raw outputs:

```text
docs-dev/profile-results/loongserve_16g_longctx_640k_20260709.jsonl
docs-dev/profile-results/loongserve_16g_longctx_640k_20260709.csv
```

Step mean latency in milliseconds:

| B | L | W_attn | d=1 | d=2 | d=4 | d=8 | d=16 | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 40960 | 655360 | 111.18 | 1743.64 | 989.89 | 575.14 | 350.74 | 1 |
| 16 | 49152 | 786432 | 111.38 | 1777.76 | 1019.23 | 565.60 | 300.34 | 1 |
| 16 | 53248 | 851968 | 113.09 | 1750.04 | 1002.84 | 568.66 | 302.87 | 1 |

The 16-GPU run has about:

```text
num_local_kvcache_blocks = 14273
kvcache_block_size = 64
single-GPU raw KV capacity = 913472 tokens
```

The 640K to 852K points still fit in one SP rank, so `d=1` is valid and remains
the fastest served path. However, these points are close to the single-GPU KV
limit and show that, even for long requests, the current multi-SP path is still
dominated by communication/synchronization overhead at `B=16`.

A second run covered a memory-forced point above one GPU's KV capacity. `d=1`
was intentionally not run because `B=16,L=65536` requires `W_attn=1048576`
tokens, exceeding the single-GPU raw KV capacity.

Command:

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --batches 16 \
  --lengths 65536 \
  --dops 2,4,8,16 \
  --warmup 3 \
  --steps 10 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --out docs-dev/profile-results/loongserve_16g_longctx_1m_20260709.jsonl
```

Raw outputs:

```text
docs-dev/profile-results/loongserve_16g_longctx_1m_20260709.jsonl
docs-dev/profile-results/loongserve_16g_longctx_1m_20260709.csv
```

Step mean latency in milliseconds:

| B | L | W_attn | d=2 | d=4 | d=8 | d=16 | best viable |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 65536 | 1048576 | 1673.00 | 1038.36 | 538.84 | 367.23 | 16 |

Long-context interpretation:

- For `W_attn <= 852K`, `d=1` remains fastest because it still fits in one SP
  rank.
- Once memory requires SP (`W_attn=1.05M` in this setup), the best viable point is
  `d=16`.
- The current threshold is therefore not a pure performance threshold at `B=16`.
  It is primarily:

  ```text
  use d=1 while memory allows;
  if memory forces scale-up past one SP rank, prefer the largest tested DoP
  among viable options for this 1.05M-token point.
  ```

This is a served-path result through DLSlime RPC and full CUDA graph. It should
not replace the final EP32 full-layer threshold table, but it gives a concrete
constraint for current 16-GPU scheduling: long request SP is useful for memory
fit first; performance scale-up still needs larger B/W or a lower-overhead SP
path to beat `d=1`.

## Runner

The synthetic decode runner is:

```text
docs-dev/loongserve_decode_profile_runner.py
```

It constructs RUNNING decode sequences with synthetic KV ownership, keeps only
the last token as input, and measures scheduler, executor, and postprocess
latency around one decode step. It does not reinstall the package and does not
modify C++.

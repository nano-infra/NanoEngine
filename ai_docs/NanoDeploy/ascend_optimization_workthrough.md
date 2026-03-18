# Ascend NPU Optimization Workthrough: v3 → v9

**Model**: Qwen3-235B-A22B (94 MoE layers, 128 experts, top-8)
**Hardware**: 8× Ascend 910B (A2), 2 nodes (4+4), HCCL inter-node
**Config evolution**: attention_dp=8/ffn_ep=8 → attention_tp=4/attention_dp=2/ffn_tp=8/ffn_ep=1
**Target**: vllm-ascend ~50ms ITL at bs=2 decode

---

## Summary Table

| Version | Date | ITL (bs=2) | Mode | Key Change | Profiler |
|---------|------|-----------|------|------------|----------|
| v22 | Mar 16 | ~690ms | eager, dp=8/ep=8 | Baseline: first working Ascend run | — |
| v23 | Mar 16 | ~560ms | eager, dp=8/ep=8 | npu_grouped_matmul replaces bmm | — |
| v24 | Mar 16 | ~2600ms | eager | Broken (regression) | — |
| v3 | Mar 16 | — | eager, dp=8/ep=8 | Profiler v3: AllGather dispatch baseline | v3 |
| v25 | Mar 16 | ~130ms | eager, tp=4/dp=2/tp8 | New parallelism: attn_tp=4, ffn_tp=8 | — |
| v26 | Mar 17 | ~122ms | **graph**, tp=4/dp=2/tp8 | ACL graph capture enabled (bs=4) | — |
| v4 | Mar 17 | — | graph, bs=4 | Profiler v4: graph baseline, ConcatD visible | v4 |
| v5 | Mar 17 | — | graph, bs=4 | Replace ConcatD with all_gather_into_tensor | v5 |
| v6 | Mar 17 | — | graph, bs=4 | Regression: ConcatD re-appeared | v6 |
| v27 | Mar 18 | ~59ms | graph, bs=2 | Fused QKV+RMSNorm+RoPE; FusedInferAttention | — |
| v7 | Mar 18 | — | graph, bs=2 | Profiler v7: fused ops, no ConcatD | v7 |
| v8 | Mar 18 | — | graph, bs=2 | MatmulAllReduce fused op; attention fixes | v8 |
| v9/v28 | Mar 18 | ~57ms | graph, bs=2 | Comm overlap (unpermute↔allreduce reorder) | v9 |

---

## Phase 1: Eager Mode Baseline (v22–v24, Mar 16)

### v22 — First Working Run (~690ms ITL)

- **Config**: attention_dp=8, ffn_ep=8, enforce_eager=True
- **Dispatch**: AscendTokenDispatcherNormal (all_to_all, dynamic shapes)
- **Compute**: bmm-based expert compute (torch.bmm per expert)
- **Attention**: npu_incre_flash_attention with block_table
- **Result**: Correct output ("1+1=2"), but 690ms ITL — 14× slower than vllm

### v23 — npu_grouped_matmul (~560ms ITL)

- Replaced per-expert bmm loops with `npu_grouped_matmul` (single fused kernel)
- Added `npu_moe_init_routing_v2` for fused sort+expand+expert counting
- Added `npu_moe_token_unpermute` for fused weighted recombination
- Added `npu_swiglu` replacing manual SiLU+mul
- **Result**: ~560ms → **19% faster**. Output garbled (routing bug, later fixed)

### v24 — Regression (~2600ms)

- Broken run, likely a dispatch shape bug during EP refactoring

---

## Phase 2: AllGather Dispatch + First Profiler (v3, Mar 16)

### v3/profiler_res_v3 — AllGather Dispatch Baseline

**Key insight**: MC2 (`npu_moe_distribute_dispatch`) requires ≥16 NPUs on A2. With 8 NPUs, fallback to `dist.all_to_all` uses dynamic shapes → **blocks ACL graph capture**.

**Solution**: `AscendTokenDispatcherAllGather` — fixed-shape AllGather/ReduceScatter path:
```
hidden [B,H] → AllGather → [B*EP,H] → local argsort+GroupedMatmul → ReduceScatter → [B,H]
```

**Profiler v3 (64 steps, per-step averages)**:

| Metric | Total (us) | Per-step (us) |
|--------|-----------|---------------|
| Computing | 1,115,358 | 17,427 |
| Comm (Not Overlapped) | 643,552 | 10,055 |
| Overlapped | 2,694 | 42 |
| Free | 983,478 | 15,367 |
| **Stage (total)** | **2,745,083** | **42,892** |

**Top ops**:

| Op | Avg (us) | Count/step | % of compute |
|----|---------|------------|-------------|
| GroupedMatmul | 34.8 | 94 | 18.8% |
| MatMulV2 (attn) | 16.2 | 94 | 13.0% |
| Index (KV gather) | 10.6 | 94 | 8.6% |
| ConcatD | 8.0 | 94 | 6.5% |
| Mul (misc) | 2.8 | 406 | 9.6% |
| Cast | 1.3 | 564 | 6.5% |

**Diagnosis**:
- Massive `Free` time (36%) → host-side overhead, no graph capture
- ConcatD (6.5%) → tensor concatenation from all_gather output list
- Index (8.6%) → KV cache pre-gather (block_table → dense)
- Cast/Mul/Pows/ReduceMean → manual RMSNorm (not fused)

---

## Phase 3: New Parallelism Config (v25, Mar 16)

### v25 — attention_tp=4, attention_dp=2, ffn_tp=8, ffn_ep=1 (~130ms)

**Paradigm shift**: Instead of EP=8 (all-to-all dispatch), use TP=8 for FFN:
- **Attention**: TP=4 (each pair of 4 GPUs shares KV), DP=2 (2 independent batch slices)
- **FFN**: TP=8 (all GPUs do all experts, weight-sharded), EP=1 (no dispatch)
- **Transition**: AttnDpToFfnTransition (AllGather bs) + FfnToAttnDpTransition (slice)

**Impact**: Eliminates EP dispatch entirely for FFN. AllReduce replaces AllGather+ReduceScatter.

**Result**: 130ms ITL — **5.3× faster than v22**, but still 2.6× slower than vllm's 50ms.
Output still garbled (correctness bug in transition layers, fixed later).

---

## Phase 4: ACL Graph Capture (v26, v4–v6, Mar 17)

### v26 — Graph Capture Enabled (~122ms)

- ACL graph capture working with AllGather dispatch
- **Graph-safe attention**: Pre-gather KV into dense tensor, bool atten_mask (no device tensor reads)
- **Graph-safe MoE**: Fixed-shape argsort+GroupedMatmul path
- master_bs=4, attn_bs=4
- **Result**: 122ms — modest improvement from graph replay eliminating host overhead

### v4/profiler_res_v4 — Graph Capture Profiler Baseline

**Profiler v4 (63 steps)**:

| Metric | Per-step (us) |
|--------|--------------|
| Computing | 11,522 |
| Comm (Not Overlapped) | 18,602 |
| Overlapped | 7,295 |
| Free | 5,533 |
| **Stage** | **35,660** |

**Key observations**:
- Communication dominates (52% not overlapped)
- ConcatD appeared: 33,814us total, 5.6us/call, 6,051 calls — from `dist.all_gather` returning list
- GroupedMatmul: 35.3us avg, unchanged from eager
- MatMulV2: 16.1us avg, unchanged

### v5/profiler_res_v5 — Fix ConcatD with all_gather_into_tensor

**Change**: Replace `dist.all_gather(tensor_list, ...)` with `dist.all_gather_into_tensor(output, ...)` in AttnDpToFfnTransition. Pre-allocates output buffer → no ConcatD kernel.

**Profiler v5 (64 steps)**:

| Metric | Per-step (us) |
|--------|--------------|
| Computing | 11,180 |
| Comm (Not Overlapped) | 7,697 |
| Overlapped | 30 |
| Free | 16,686 |
| **Stage** | **35,594** |

**Impact**:
- ConcatD **gone** from top ops
- Comm(Not Overlapped) dropped 18,602 → 7,697us (**-59%**)
- But `Free` increased 5,533 → 16,686us — the saved comm time became idle time
- ViewCopy appeared (44,728us total) — likely from `.contiguous()` calls
- Total stage time unchanged (~35.6ms) — bottleneck shifted to scheduling/free time

### v6/profiler_res_v6 — ConcatD Regression

**Profiler v6**: ConcatD re-appeared (31,507us). Likely reverted or a different code path triggered.
Comm(Not Overlapped) back up to 21,277us/step. This version was discarded.

---

## Phase 5: Fused Ops + Correct Config (v27, v7–v8, Mar 18)

### v27 — Fused Triton Kernels + Correct Attention (~59ms)

**Major changes**:
1. **Fused QKV+RMSNorm+RoPE Triton kernel** (`split_qkv_rmsnorm_rope_kernel_1`)
   - Eliminates separate Cast, Mul, Pows, ReduceMean, Rsqrt, RmsNorm ops from attention
   - Single kernel: split QKV → per-head RMSNorm → RoPE rotation

2. **FusedInferAttentionScore** replaces pre-gather + npu_incre_flash_attention
   - Direct paged attention with TND layout
   - Eliminates Index ops (KV pre-gather) and reduces memory bandwidth

3. **master_bs=2** (down from 4) — matches actual decode batch size

4. **Correctness fix** for parallelism transitions (padding/slicing)

**Result**: 59ms ITL — **2× faster than v26**, only 18% above vllm's 50ms

### v7/profiler_res_v7 — Clean Fused Profile

**Profiler v7 (64 steps)**:

| Metric | Per-step (us) |
|--------|--------------|
| Computing | 8,003 |
| Comm (Not Overlapped) | 12,705 |
| Overlapped | 33 |
| Free | 15,122 |
| **Stage** | **35,863** |

**Top ops (clean — no Cast/ConcatD/Index bloat)**:

| Op | Avg (us) | Count/step | Step total (us) |
|----|---------|------------|-----------------|
| GroupedMatmul | 35.0 | 94 | 3,293 |
| MatMulV2 | 16.0 | 94 | 1,506 |
| FusedInferAttentionScore | 10.3 | 32 | 329 |
| MoeInitRoutingV3 | 8.5 | 32 | 272 |
| split_qkv_rmsnorm_rope | 6.7 | 32 | 214 |
| MoeGatingTopKSoftmax | 5.3 | 32 | 170 |
| MoeTokenUnpermute | 4.4 | 32 | 141 |
| SwiGlu | 3.3 | 32 | 107 |

**Diagnosis**:
- Compute is lean — fused ops eliminated all bloat
- Communication still 12.7ms/step not overlapped
- Free time 15.1ms — idle between graph replay and next step
- **No MatmulAllReduce** — allreduce not fused with GroupedMatmul's down proj

### v8/profiler_res_v8 — MatmulAllReduce Fusion

**Key change**: CANN auto-fused `GroupedMatmul(down) + dist.all_reduce` into `MatmulAllReduce` op.

**Profiler v8 (63 steps)**:

| Metric | Per-step (us) |
|--------|--------------|
| Computing | 9,317 |
| Comm (Not Overlapped) | 7,459 |
| Overlapped | 28 |
| Free | 16,784 |
| **Stage** | **33,588** |

**Top ops**:

| Op | Avg (us) | Count/step | Step total (us) |
|----|---------|------------|-----------------|
| GroupedMatmul | 32.5 | 94 | 3,059 |
| **MatmulAllReduce** | **44.9** | **47** | **2,112** |
| MatMulV2 | 16.6 | 94 | 1,560 |
| FusedInferAttentionScore | 10.8 | 47 | 508 |
| MoeInitRoutingV3 | 9.4 | 47 | 441 |
| split_qkv_rmsnorm_rope | 6.7 | 47 | 315 |
| MoeGatingTopKSoftmax | 5.3 | 47 | 249 |
| MoeTokenUnpermute | 4.3 | 47 | 204 |

**Key observation**: `MatmulAllReduce` is now 22.8% of compute — the allreduce for attention's RowParallelLinear got fused. The MoE allreduce was a separate synchronous `dist.all_reduce` — **not fused, not overlapped** (only 28us overlap vs 7,459us not-overlapped comm).

---

## Phase 6: Communication Overlap (v9/v28, Mar 18)

### v28/v9 — Unpermute↔AllReduce Reorder + Async + DP Slice

**Three changes to `experts.py`**:

1. **Reorder allreduce after unpermute**: `GroupedMatmul(down) → unpermute → allreduce` instead of `→ allreduce → unpermute`. Valid because both are linear (commute with elementwise sum).

2. **DP slice before allreduce (decode only)**: When `attention_dp > 1`, slice output to this rank's tokens before allreduce. Halves allreduce volume: `[4, 4096]` → `[2, 4096]` at bs=2.

3. **async_op=True**: `work = dist.all_reduce(..., async_op=True); work.wait()` — in ACL graph capture, trigger and wait become separate graph nodes. The HCCL stream can overlap with other compute during graph replay.

**Supporting changes**:
- `FfnToAttnDpTransition`: Detects already-sliced tensors (decode) and passes through
- `dp_slice` flag: Only slices during decode (uniform batch sizes), not prefill (padded batches)
- MoE block `.view(orig_shape)` guards for changed output shape

**Profiler v9 (128 steps, ITL ~57ms)**:

| Metric | Per-step (us) |
|--------|--------------|
| Computing | — |
| Comm (Not Overlapped) | — |
| **Steady-state ITL** | **~56ms** |

**v28 log ITL progression (post-profiler warmup)**:
```
57.88 → 56.00 → 55.42 → 55.35 → 55.05 → 55.97 → 55.58 → 56.58
→ 56.55 → 55.61 → 56.96 → 56.28 → 57.42 → 58.24 → 57.78 → 57.94
```
**Median: ~56.3ms**, down from 59ms in v27.

---

## Optimization Waterfall

```
v22  690ms  ████████████████████████████████████████████████████████████████████████  (eager, bmm, ep=8)
v23  560ms  ████████████████████████████████████████████████████████████            (npu_grouped_matmul)
v25  130ms  █████████████                                                          (tp=8 replaces ep=8)
v26  122ms  ████████████                                                           (ACL graph capture)
v27   59ms  ██████                                                                 (fused ops)
v28   57ms  █████▊                                                                 (comm overlap)
vllm  50ms  █████                                                                  (target)
```

---

## Remaining Gap Analysis (57ms → 50ms target)

| Source | Est. (ms) | Notes |
|--------|----------|-------|
| GroupedMatmul kernel efficiency | ~2ms | vllm uses optimized CANN GroupedGemm; our 32.5us/layer × 94 = 3.1ms vs ~1ms |
| MatMulV2 (attention proj) | ~0.5ms | Minor; similar kernel |
| MatmulAllReduce overhead | ~2ms | 44.9us/call × 47 = 2.1ms; could be reduced with smaller TP group or pipelining |
| Free/scheduling time | ~2ms | Host-side latency between graph replays |
| **Total gap** | **~7ms** | |

### Next optimization targets:
1. **Reduce Free time**: Pipeline graph replay with scheduling (overlap host work)
2. **GroupedMatmul tuning**: Check CANN op version, tiling config, transB layout
3. **Attention TP reduction**: attn_tp=2 instead of 4 would halve attention allreduce
4. **Larger batch**: bs=4/8 amortizes fixed overhead; currently limited by memory

---

## File Change Log

| File | v3→v5 | v7 | v8 | v9 |
|------|-------|----|----|-----|
| `experts.py` | npu_grouped_matmul, AllGather dispatch | — | — | Reorder unpermute↔allreduce, dp_slice, async_op |
| `token_dispatcher.py` | AscendTokenDispatcherAllGather | — | — | — |
| `attention.py` | Pre-gather KV, bool mask | FusedInferAttentionScore | — | — |
| `layernorm.py` | npu_rms_norm | npu_add_rms_norm | — | — |
| `parallelism_transition.py` | AttnDpToFfnTransition, FfnToAttnDpTransition | all_gather_into_tensor | — | Already-sliced guard |
| `model_runner.py` | Graph capture, _npu_update_stream | — | — | — |
| `qwen3_moe.py` | Transition chain | — | — | .view() guard for dp_slice |
| `fused_qkv_norm_rope.py` | — | Triton kernel | — | — |

# Ascend AllGather Decode + ACL Graph Capture — Design & Walkthrough

## Context

MC2 (`npu_moe_distribute_dispatch/combine`) requires ≥16 NPU cards on A2 (910B).
With 8 NPUs, the previous `dist.all_to_all` fallback worked but had **dynamic
send/recv shapes** that blocked ACL graph capture.

vllm-ascend uses an **AllGather-based MoE dispatch** for this scenario (A2 with EP
but <16 cards): fixed-shape collectives (AllGather/ReduceScatter) enable graph capture.

**Goal**: Replace `dist.all_to_all` fallback → enable graph capture → reduce ITL.

---

## Architecture: AllGather Token Dispatcher

**File**: `layers/token_dispatcher.py` — class `AscendTokenDispatcherAllGather`

### Tensor Flow

```
hidden [B, H]  ─── AllGather ──▶  [B*E, H]    (fixed)
topk_ids [B, K] ── AllGather ──▶  [B*E, K]    (fixed)
topk_wts [B, K] ── AllGather ──▶  [B*E, K]    (fixed)

┌─ Pre-expand: [B*E, H] → [B*E*K, H] (each token repeated K times) ──────────┐
│ argsort(flat_experts, stable=True) → perm [B*E*K]                            │
│ sorted_hidden = x_expanded[perm]   → [B*E*K, H] sorted by expert            │
│ expert_counts via scatter_add_     → [num_experts] (graph-safe, no bincount) │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ Extract local experts (graph-capture-safe, fixed output shape) ─────────────┐
│ index_select + clamp → always [B*E*K, H]                                     │
│ Zero-mask padding rows beyond local count                                     │
│ Pad group_list[-1] so sum = B*E*K (fixed matmul output shape)                │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ npu_grouped_matmul (gate_up + npu_swiglu + down) ──────────────────────────┐
│ group_list_type=1 (count mode), group_list=[local_expert_counts_padded]      │
│ Fixed output shape: [B*E*K, H]                                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ Combine: inverse permutation + weighted sum ────────────────────────────────┐
│ scatter_add_ → full_sorted [B*E*K, H]                                        │
│ full_sorted[inv_perm] → unsorted [B*E*K, H] in original (i*K+k) order       │
│ Multiply by topk_weights, reshape [B*E, K, H], sum(dim=1) → [B*E, H]        │
└──────────────────────────────────────────────────────────────────────────────┘

[B*E, H] ── ReduceScatter(sum) ──▶  [B, H]     (each rank's contribution)
```

### Why Not `npu_moe_init_routing`?

We tried extensively. The 4-arg API available on our CANN version has a
fundamental constraint: **`row_idx` indexes into BOTH `x` AND `expert_idx.flatten()`**.

- `row_idx[i,k]=i` → each token gets the same expert K times (wrong)
- `row_idx[i,k]=i*K+k` → requires `x[T*K, H]` but op enforces `x.rows == row_idx.rows`
- `x[T*K, H]` with `row_idx[T*K, 1]` → works as K'=1, but `expanded_row_idx` output
  is opaque (NOT our input values), making the combine impossible without
  `npu_moe_token_unpermute` (which has graph-capture concerns)

**Solution**: Pure PyTorch `argsort(float32)` runs on AiCore (not AiCpu) and gives
us an explicit permutation we can invert for the combine. Clean, debuggable, correct.

### Graph-Capture Safety

All operations are device-resident:
- `argsort(float32)` → AiCore (not AiCpu; int32/int64 falls to AiCpu)
- `index_select`, `scatter_add_`, `clamp` → NPU device ops
- `AllGather`, `ReduceScatter` → HCCL collectives (graph-capturable)
- No `.item()`, `bincount`, `print`, or other host-sync operations
- All tensor shapes fixed for a given `max_num_seqs` (decode batch size)

---

## torch.compile Fix

**Files**: `layers/rotary_embedding.py`, `layers/activation.py`

The `_maybe_torch_compile` decorator only checked `NANO_BACKEND` env var. When
`--backend_type ascend` is passed as CLI arg (not env var), `torch.compile` was
applied on NPU, generating inductor/triton kernels that crash with OOB memory
access during graph capture.

**Fix**: Also check `torch.npu.is_available()`:
```python
def _maybe_torch_compile(fn):
    if os.environ.get("NANO_BACKEND", "") == "ascend":
        return fn
    try:
        import torch_npu
        if torch.npu.is_available():
            return fn
    except ImportError:
        pass
    return torch.compile(fn)
```

---

## Expert Compute

**File**: `backends/ascend/layers/experts.py`

`_compute_decode_ep` uses the AllGather dispatcher with `npu_grouped_matmul`:

```python
(recv_hidden, _, _, expert_token_nums, group_list_type
) = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)

gate_up_out = torch_npu.npu_grouped_matmul(
    x=[recv_hidden], weight=[gate_up_proj.T],
    split_item=2, group_list_type=1, group_type=0,
    group_list=expert_token_nums,
)[0]
gate_up_out = torch_npu.npu_swiglu(gate_up_out)
down_output = torch_npu.npu_grouped_matmul(
    x=[gate_up_out], weight=[down_proj.T],
    split_item=2, group_list_type=1, group_type=0,
    group_list=expert_token_nums,
)[0]
result = dispatcher.combine(down_output)
```

---

## Known Latency Gap vs vllm-ascend (~3x)

### Root Causes Identified

| Bottleneck | NanoDeploy | vllm-ascend | Impact |
|-----------|-----------|-------------|--------|
| **Attention KV pre-gather** | Dense gather `k_flat[slots]` creates `[B, max_kv_len, nkv, D]` every step | Direct `block_table` in `npu_fused_infer_attention_score` TND mode | **High** — massive memory bandwidth per step |
| **Stream sync before replay** | `torch_npu.npu.current_stream().synchronize()` before every `graph.replay()` | No explicit sync (graph replay handles it) | **Medium** — host-device roundtrip per decode step |
| **MoE argsort** | PyTorch `argsort(float32)` on AiCore | `npu_moe_init_routing` native NPU op | **Low-Medium** — kernel launch overhead |
| **Attention op choice** | `npu_incre_flash_attention` BNSD | `npu_fused_infer_attention_score` TND paged | **Low** — similar compute |

### Fix Priority

1. **Remove stream sync** (`model_runner.py:1014`): Test without the explicit
   `synchronize()` before `graph.replay()`. vllm-ascend doesn't do this.
   NPU graph replay should handle stream ordering internally.

2. **Direct paged attention**: Switch decode to `npu_fused_infer_attention_score`
   TND with `block_table` if CANN version supports it without copy_stream sync.
   Alternatively, investigate if the copy_stream sync only happens with
   `actual_seq_lengths` as a Python list (not tensor) — using a pre-allocated
   fixed-size tensor might avoid it.

3. **Replace argsort with `npu_moe_init_routing`**: Use the K'=1 approach
   (reshape to `[T*K, 1]`) which avoids the row-count mismatch. Even if
   `expanded_row_idx` is opaque, we can use `npu_moe_token_unpermute` for the
   combine and validate correctness.

---

## Files Changed

| File | Change |
|------|--------|
| `layers/token_dispatcher.py` | Added `AscendTokenDispatcherAllGather` (argsort dispatch/combine) |
| `backends/ascend/layers/experts.py` | `_get_decode_dispatcher` → AllGather; removed debug prints |
| `worker/model_runner.py` | Removed MC2 group creation |
| `context/expert_context.py` | Removed `mc2_group` from `ascend_warmup` |
| `layers/rotary_embedding.py` | `_maybe_torch_compile` NPU detection fix |
| `layers/activation.py` | `_maybe_torch_compile` NPU detection fix |

---

## Validation

### Qwen3-30B-A3B (attention_dp=8, ffn_ep=8)
```
Prompt: '1+1=?'
Completion: '1 + 1 = 2<|im_end|>'
```
- Eager mode: ✅ correct output
- Graph capture: ✅ captures and runs

### Qwen3-235B-A22B (attention_tp=4, attention_dp=2, ffn_ep=8)
```
Prompt: '1+1=?'
Completion: correct
```
- Graph capture: ✅ passes
- Latency: ~3x vs vllm-ascend (see gap analysis above)

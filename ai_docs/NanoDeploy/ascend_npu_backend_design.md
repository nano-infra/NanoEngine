# Ascend NPU Backend — Design

## Overview

This document describes the architecture of NanoDeploy's Ascend NPU backend,
added to support running Qwen3-MoE (and similar models) natively on Huawei
Ascend NPUs via `torch_npu` + HCCL, without CUDA/Triton/NVSHMEM dependencies.

---

## Key Design Decisions

### 1. No NanoCCL on Ascend
NanoCCL is a CUDA/NVSHMEM library. On Ascend, all collective communication
uses `torch.distributed` with the HCCL backend. This includes:
- TP all-reduce in linear layers
- EP all-to-all in token dispatchers
- EP all-gather (not used in current path)

### 2. No Triton on Ascend
Triton does not run on NPU. KV-cache store operations are rewritten as pure
PyTorch index-scatter ops (`kv_ops.py`).

### 3. BF16 First, FP8 Deferred
The initial implementation is BF16-only. No FP8/W8A8/DeepGEMM. Expert
grouped matmul uses `torch.bmm`.

### 4. SP=1 Only
Sequence parallelism is deferred. The `attn_sp` dimension must be 1.

### 5. enforce_eager=True
No CUDA graph capture on Ascend in this version. Eager mode is mandatory.

### 6. Explicit Backend Detection
Backend selection priority:
```
explicit backend_type arg > NANO_BACKEND env var > hardware auto-detect
```
Auto-detect: tries `torch_npu.npu.is_available()`, then CUDA capability.

---

## Component Map

```
backends/
  detect.py                  hardware auto-detection
  __init__.py                init_backend() factory + NANO_BACKEND env var
  ascend/
    __init__.py              AscendBackendFactory — wires all layers
    layers/
      linear.py              BF16 parallel linear (RowParallel, ColParallel, etc.)
      attention.py           GQA via npu_fused_infer_attention_score (TND)
      experts.py             MoE EP experts (bmm, no DeepGEMM)
    ops/
      kv_ops.py              KV-cache store via index-scatter (no Triton)

layers/
  token_dispatcher.py        AscendTokenDispatcherNormal  (prefill EP)
                             AscendTokenDispatcherLowLatency (decode EP)

context/
  expert_context.py          ascend_warmup() — bypasses DeepEP buffer init
  cache.py                   device-agnostic mem_get_info / synchronize
  distributed.py             device_type-aware mesh init
```

---

## Attention: npu_fused_infer_attention_score (TND)

### Why Not npu_prompt_flash_attention?
`npu_prompt_flash_attention` with BNSD + `sparse_mode=0` + explicit bool mask
did not correctly apply causal masking on tested Ascend hardware. The unified
`npu_fused_infer_attention_score` API (same one vllm-ascend uses) works
correctly with:

- **Layout: TND** — `[total_tokens, num_heads, head_dim]` — no per-sequence
  reshape needed for varlen
- **sparse_mode=3** — causal masking built-in (uses upper-triangle bias)
- **atten_mask** — pre-computed 2048×2048 bool upper-triangle tensor
- **actual_seq_lengths** — **cumulative** (not per-seq): e.g., for seqs of
  length 3 and 4 → `[3, 7]`

### Prefill
```python
torch_npu.npu_fused_infer_attention_score(
    query=q[:T],            # [T, N, D]
    key=k[:T],              # [T, Nkv, D]
    value=v[:T],            # [T, Nkv, D]
    atten_mask=causal_mask, # [2048, 2048] bool
    block_table=None,
    input_layout="TND",
    block_size=128,
    actual_seq_lengths=cu_seqlens_q[1:].tolist(),   # cumulative
    actual_seq_lengths_kv=cu_seqlens_k[1:].tolist(),
    num_heads=N, num_key_value_heads=Nkv,
    scale=scale, sparse_mode=3,
)
```

### Decode
```python
# Reshape paged cache for TND paged mode
k_flat = k_cache.view(num_blocks, block_size, Nkv * D)
v_flat = v_cache.view(num_blocks, block_size, Nkv * Dv)

torch_npu.npu_fused_infer_attention_score(
    query=q,                            # [B, N, D] (one token per seq)
    key=k_flat, value=v_flat,
    atten_mask=causal_mask,
    block_table=block_tables,
    input_layout="TND",
    block_size=block_size,
    actual_seq_lengths=[1, 2, ..., B],  # cumulative single tokens
    actual_seq_lengths_kv=context_lens.tolist(),
    num_heads=N, num_key_value_heads=Nkv,
    scale=scale, sparse_mode=3,
)
```

---

## Expert Parallelism: Token Dispatchers

Two dispatchers are provided, sharing the same dispatch/combine interface
as the DeepEP dispatchers used on CUDA.

### AscendTokenDispatcherNormal (prefill EP)

Uses `torch.distributed.all_to_all` (HCCL):

1. Sort tokens by expert index; compute per-rank send counts
2. All-to-all to exchange token counts
3. All-to-all to exchange (hidden_states, expert_indices) in expert-sorted order
4. Local expert compute (masking per expert)
5. All-to-all to send results back
6. Unsort + weighted scatter-add to reconstruct per-token output

### AscendTokenDispatcherLowLatency (decode EP)

Primary path: `torch_npu.npu_moe_distribute_dispatch` /
`npu_moe_distribute_combine` (MC2 fused all-to-all, Ascend-native).

Fallback (when MC2 unavailable): same `torch.distributed.all_to_all`
approach as Normal dispatcher, but packs results into `[L, max_m, H]`
grouped-matmul format expected by `_compute_decode_ep`.

Key correctness points:
- **Send `sorted_x [N*K, H]`**, not `hidden_states [N, H]`. Each token is
  expanded once per top-k expert before dispatch.
- **`max_m = max(per-expert receive counts)`**, not average. Individual
  experts can receive more than the average.
- **Combine reverse all-to-all** splits by `recv_splits` (tokens received
  from each rank), sends back to each rank in the same expert-sorted order.

---

## KV Cache

`store_kvcache_npu` writes new K/V tokens into paged slots via index-scatter:
```python
k_flat = k_cache.view(-1, Nkv, D)
k_flat[slot_mapping[valid]] = key[valid]
```
No Triton, no custom CUDA kernel.

---

## ExpertContext on Ascend

`expert_context.ascend_warmup()` replaces the standard `warmup()` (which
requires DeepEP + CUDA). It sets `warmup_called = True` and `buffer = None`
without allocating any NVSHMEM buffers. Ascend dispatchers handle their own
all-to-all via HCCL.

---

## Ray Executor Changes

- Workers use `resources={"NPU": 1}` placement instead of `num_gpus=1`
- Placement groups use `{"CPU": 0.1, "NPU": 1.0}` bundles
- `init_cudagraph_buffer` / `capture_cudagraph` are skipped when
  `device_type == "npu"`

---

## Config Fields Added

| Field | Default | Purpose |
|-------|---------|---------|
| `device_type` | `"cuda"` | `"npu"` for Ascend; propagated from `NANO_DEVICE_TYPE` env |
| `backend_type` | `""` | Override backend; empty = auto-detect |
| `enforce_eager` | `False` | Must be `True` for Ascend (no CUDA graph) |

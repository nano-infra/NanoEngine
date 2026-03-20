# Ascend NPU Backend — Implementation Walkthrough

Porting NanoDeploy to run natively on Huawei Ascend NPUs. Target model:
**Qwen3-30B-A3B-Instruct** (MoE) in hybrid (prefill+decode) mode with
`attention_dp=8, ffn_ep=8`.

---

## Running

```bash
python examples/non_disagg.py \
  --ray_address <head>:6379 \
  --master_address <head>:6006 \
  --model /path/to/Qwen3-30B-A3B-Instruct \
  --attention_dp 8 --ffn_ep 8 \
  --kvcache_block_size 256 --max_num_seqs 32 \
  --temperature 0 --prompt "1+1=?" --max_tokens 16 \
  --max_num_batched_tokens 4096 --max_model_len 4096 \
  --device_type npu --backend_type ascend --enforce_eager true
```

Expected output: `Completion: '1 + 1 = 2<|im_end|>'`

---

## Files Changed / Created

### New: `backends/detect.py`
Hardware auto-detection. Tries `torch_npu` first, then CUDA capability.

### Modified: `backends/__init__.py`
Added `"ascend"` to valid backends; calls `AscendBackendFactory` when selected.

### New: `backends/ascend/`

| File | What it does |
|------|-------------|
| `__init__.py` | `AscendBackendFactory` — returns Ascend layer implementations |
| `layers/linear.py` | BF16 parallel linears (Row/Col/Merged/QKV/Replicated). Reuses generic logic; RowParallel all-reduce uses HCCL |
| `layers/attention.py` | GQA via `npu_fused_infer_attention_score` (TND, sparse_mode=3) |
| `layers/experts.py` | `AscendDistributedRoutedExperts`: bmm-based compute, uses Ascend token dispatchers for EP |
| `ops/kv_ops.py` | `store_kvcache_npu`: pure PyTorch index-scatter to write KV slots |

### Modified: `layers/token_dispatcher.py`
Added two new dispatcher classes at the bottom:
- `AscendTokenDispatcherNormal` — prefill EP via `dist.all_to_all`
- `AscendTokenDispatcherLowLatency` — decode EP via MC2 (with `dist.all_to_all` fallback)

### Modified: `context/expert_context.py`
- Guarded `import deep_ep` with `try/except`
- Added `ascend_warmup()` method — sets `warmup_called=True` without DeepEP

### Modified: `context/cache.py`
Replaced hardcoded `torch.cuda.*` calls with device-agnostic helpers:
- `_get_device_mem_info()` — uses `torch_npu.npu.mem_get_info()` on NPU
- `_device_synchronize()` — dispatches to npu or cuda

### Modified: `context/distributed.py`
`init_device_mesh` now uses `config.device_type` instead of hardcoded `"cuda"`.

### Modified: `config.py`
Added `device_type: str = "cuda"` and `backend_type: str = ""` fields, read
from `NANO_DEVICE_TYPE` / `NANO_BACKEND` env vars.

### Modified: `engine/ray_executor.py`
- NPU workers use `resources={"NPU": 1}` placement
- Placement group bundles use `"NPU"` key when `device_type == "npu"`
- Skips CUDA graph capture for NPU

### Modified: `worker/model_runner.py`
- Calls `ExpertContext.get_instance().ascend_warmup(...)` instead of `warmup()` for Ascend EP
- Skips `init_cudagraph_buffer` / `capture_cudagraph` when `enforce_eager=True`

### Modified: `layers/layernorm.py`
Fast path via `torch_npu.npu_rms_norm` when available, falls back to PyTorch.

### Modified: `examples/non_disagg.py`
Added `--device_type` and `--backend_type` arguments.

---

## Key Bug Fixes During Development

### 1. Decode attention layout (BSND → BNSD)
`npu_incre_flash_attention` requires query in BNSD layout.
Wrong: `q.unsqueeze(1)` → `[B, 1, N, D]` (BSND)
Fixed: `q.unsqueeze(2)` → `[B, N, 1, D]` (BNSD)

### 2. EP dispatch sends expanded tokens
Wrong: sent `hidden_states [N, H]` (original, not expanded per expert)
Fixed: sort `hidden_states[flat_row_idx[sort_perm]]` → `sorted_x [N*K, H]`,
one copy per expert selection.

### 3. max_m must be actual max, not average
Wrong: `max_m = (total_recv + L - 1) // L`
Fixed: `max_m = max(counts_per_expert)` where counts come from
`(recv_expert_flat == global_expert_id).sum()` per expert.

### 4. Prefill must use fresh K/V, not paged k_cache
The paged `k_cache` is in `[num_blocks, block_size, Nkv, D]` block format —
cannot be indexed by token position. Prefill must pass `k, v` directly.

### 5. Causal attention: switch to npu_fused_infer_attention_score
`npu_prompt_flash_attention` (BNSD + sparse_mode=0 + bool mask) produced
incorrect attention (future tokens visible). Switching to
`npu_fused_infer_attention_score` with TND + sparse_mode=3 fixed it.
This is the same API used by vllm-ascend.

### 6. actual_seq_lengths is CUMULATIVE for TND
For two sequences of length 3 and 4:
Wrong:  `actual_seq_lengths=[3, 4]`
Fixed:  `actual_seq_lengths=[3, 7]`  (cumulative, like cu_seqlens[1:])

---

## Validation

Tested: Qwen3-30B-A3B-Instruct-2507, `attention_dp=8, ffn_ep=8`, 8× Ascend NPU

```
Prompt: '1+1=?'
Completion: '1 + 1 = 2<|im_end|>'
```

Also validated with `attention_dp=4, ffn_ep=4` and `attention_dp=1, ffn_ep=1`.

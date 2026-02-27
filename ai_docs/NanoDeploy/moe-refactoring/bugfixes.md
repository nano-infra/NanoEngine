# MoE Refactoring — Bugs & Fixes Log

Complete chronological record of all issues encountered and resolved during the MoE refactoring.

---

## Error 1: `TypeError: cannot unpack non-iterable NoneType`

**Location**: `deep_ep/buffer.py:442` via `token_dispatcher.py:207`
**Symptom**: `Buffer.combine()` receives `handle=None`

**Root Cause**: Duplicate `dispatch` method in `DeepEPTokenDispatcherNormal` was shadowing the correct one. The shadowing method didn't set `self.handle`, so `combine()` received `None`.

**Fix**: Removed the duplicate `dispatch` method from `token_dispatcher.py`.

---

## Error 2: Incorrect TP-sliced dimensions in decode

**Location**: `distributed_routed_experts.py:_compute_decode_ep`
**Symptom**: Tensor shape mismatch during masked GEMM

**Root Cause**: Code used `self.num_experts_per_rank` (unset) instead of `self.num_local_experts`, and `self.intermediate_size` instead of `self.local_intermediate_size` for TP-sliced operations.

**Fix**: Changed `num_groups = self.num_experts_per_rank` → `self.num_local_experts` and `self.intermediate_size` → `self.local_intermediate_size`.

---

## Error 3: Wrong `is_prefill` heuristic

**Location**: Model forward methods
**Symptom**: Prefill path used during decode (small batch)

**Root Cause**: `is_prefill = (num_tokens > 8)` was a heuristic that broke for small batches.

**Fix**: Replaced with `is_prefill = get_context().is_prefill` — uses the scheduler's explicit mode flag.

---

## Error 4: `DeepEP RDMA buffer too small`

**Location**: `expert_context.py:warmup` → `Buffer.__init__`
**Symptom**: Assertion `layout.total_bytes <= num_rdma_bytes` fails during low-latency dispatch

**Root Cause** (3 sub-issues):
1. `hidden_size_bytes` was passed to `get_low_latency_rdma_size_hint` instead of raw `hidden_size`
2. `max_num_sequence` (16) used for buffer sizing instead of `num_max_dispatch_tokens_per_rank` (128)
3. `num_sms` was set to `min(4, ...)` instead of `Buffer.num_sms` (20)

**Fix**:
- In `expert_context.py`: Used `hidden_size` (not bytes) for hint, set `num_sms = Buffer.num_sms`, calculated `num_qps_per_rank = num_local_experts`
- In `token_dispatcher.py`: `DeepEPTokenDispatcherLowLatency.__init__` reads `num_max_dispatch_tokens_per_rank` from `ExpertContext.get_instance()`

---

## Error 5: Missing `clean_low_latency_buffer` on mode transition

**Location**: `expert_context.py`
**Symptom**: DeepEP internal assertion on prefill→decode transition

**Root Cause**: DeepEP requires `Buffer.clean_low_latency_buffer()` when switching between Normal and Low Latency modes. No mode tracking existed.

**Fix**: Added `_latest_mode` attribute + `transition_to_low_latency()` / `transition_to_normal()` methods in `ExpertContext`. Called from `_compute_prefill_ep` and `_compute_decode_ep` in `DistributedRoutedExperts`.

---

## Error 6: `AssertionError: refcount=4` in `DisposibleTensor`

**Location**: `distributed_routed_experts.py:314`, `layers/utils.py:189`
**Symptom**: `DisposibleTensor.dispose()` assertion fails because PyTorch holds extra references

**Root Cause**: `DisposibleTensor` expects exactly 2 references (the variable + the `sys.getrefcount` call), but PyTorch's internal graph/autograd holds additional references.

**Fix**: **Removed all `DisposibleTensor` usage entirely.** Replaced with direct tensor handling and `del` for explicit cleanup. Removed all `.value` and `.dispose()` calls from `token_dispatcher.py` and `distributed_routed_experts.py`.

---

## Error 7: Missing `silu_and_mul_masked_post_quant_fwd`

**Location**: `nanoexpert/kernels/fp8.py`
**Symptom**: `ImportError: cannot import name 'silu_and_mul_masked_post_quant_fwd'`

**Root Cause**: Kernel was not ported from DLBlas.

**Fix**: Ported `_silu_and_mul_post_quant_kernel` Triton kernel and its Python wrapper from `dlblas/kernels/moe.py` to `nanodeploy/kernels/fp8.py`.

---

## Error 8: NVL buffer too small for prefill combine

**Location**: `expert_context.py` → `Buffer.combine()`
**Symptom**: `RuntimeError: num_channels * num_ranks * sizeof(int) * 2 + ... <= num_nvl_bytes`

**Root Cause**: `hidden_size_bytes` was calculated using FP8 size for NVL buffer, but `Buffer.combine()` always outputs BF16.

**Fix**: Changed `hidden_size_bytes = hidden_size * 2` (always BF16) for NVL buffer sizing in `expert_context.py`.

---

## Error 9: Duplicate tensor construction in `prepare_decode`

**Location**: `model_runner.py:509-579`
**Symptom**: Lines 545-579 duplicated lines 509-543 exactly.

**Fix**: Removed the redundant block.

---

## Error 10: `AttributeError: 'NoneType' object has no attribute 'numel'`

**Location**: `nanodeploy/context/cache.py` during `start_peer_agent()`
**Symptom**: KV cache is None when `start_peer_agent` is called

**Root Cause**: `start_peer_agent()` was called from `preallocate_kvcache()` before actual KV cache allocation.

**Fix**: Moved `cache_context.start_peer_agent()` to the end of `allocate_kvcache()` in `model_runner.py`.

---

## Error 11: BF16 weight size mismatch during loading

**Location**: `loader.py:_handle_packed_expert_weight`
**Symptom**: `RuntimeError: The size of tensor a (64) must match the size of tensor b (512)`

**Root Cause**: Packed expert weight handler didn't slice by EP rank (dim 0) and TP rank (intermediate dim) for BF16 weights.

**Fix**: Modified `_handle_packed_expert_weight` to correctly slice EP and TP dimensions.

---

## Error 12: BF16 decode — wrong unpack of `packed_recv_hidden`

**Location**: `distributed_routed_experts.py:_compute_decode_ep`
**Symptom**: `ValueError: not enough values to unpack (expected 3, got 2)`

**Root Cause**: In BF16 mode, `low_latency_dispatch` returns `packed_recv_hidden` directly (not a tuple), but code tried `packed_recv_hidden[0]`.

**Fix**: Changed `recv_x = packed_recv_hidden[0]` → `recv_x = packed_recv_hidden` in BF16 path.

---

## Error 13: BF16 prefill — missing `fused_moe_v3_bf16`

**Location**: `nanoexpert/kernels/fused_moe_v3.py`
**Symptom**: `ValueError: not enough values to unpack (expected 2, got 0)` when calling `fused_moe_v3` with BF16 weights

**Root Cause**: `fused_moe_v3` was FP8-specific, no BF16 equivalent existed.

**Fix**: Implemented `fused_moe_v3_bf16` in `fused_moe_v3.py` for BF16 prefill, with `ep_scatter_bf16` Triton kernel. Updated call sites in `distributed_routed_experts.py`.

---

## Error 14: FP8 decode scale dimension assertion

**Location**: `deep_gemm` C++ code, `layout.hpp:75`
**Symptom**: `RuntimeError: sf.dim() == static_cast<int>(num_groups.has_value()) + 2`

**Root Cause**: `gate_up_scale_inv` and `down_scale_inv` parameters were not being loaded for FP8 models — they remained as empty 1D tensors (`torch.empty(0)`). DeepGEMM expects 3D scale tensors.

**Fix** (first instance — Qwen3 MoE): Added `self.config = config` to `Qwen3MoeForCausalLM.__init__()`. Made `loader.py` robust to both `num_experts` and `n_routed_experts` config attribute names.

**Fix** (second instance — Qwen3.5 MoE packed scales): Added `PACKED_EXPERT_SCALE_RE` regex and `load_packed_expert_scale` function to `loader.py`. Updated `qwen3_5_moe_loader.py` to handle packed expert scale tensors.

---

## Error 15: `ModuleNotFoundError: No module named 'transformers_modules'`

**Location**: `config.py` during Ray serialization
**Symptom**: DeepSeek V3 uses a custom config class from `transformers_modules.*` which isn't available on remote Ray workers.

**Root Cause**: Dynamic config class from `trust_remote_code=True` can't be pickled/unpickled by Ray if the `transformers_modules` package isn't installed on workers.

**Fix**: In `config.py:validate_config`, added conversion of dynamic config classes to standard `PretrainedConfig` by copying all attributes, enabling Ray serialization.

---

## Error 16: Stale `dlblas` imports

**Location**: Model definition files
**Symptom**: `ImportError: No module named 'dlblas'`

**Fix**: Removed `from dlblas.layers.moe.ep_moe import build_deepep_moe` from all model files. Updated imports to `nanodeploy.layers.distributed_routed_experts`.

---

## Error 17: Qwen3.5 MoE — Expert weights not loading (per-expert vs packed format mismatch)

**Location**: `qwen3_5_moe_loader.py`
**Symptom**: Expert weights silently skipped; FP8 GEMM fails at runtime due to uninitialized weights/scales

**Root Cause**: The loader only handled packed 3D format (`PACKED_EXPERT_RE`: `experts.gate_up_proj`), but the actual Qwen3.5-397B-A17B-FP8 checkpoint uses per-expert format (`experts.E.gate_proj.weight`).

**Fix**: Added `EXPERT_RE` + `load_per_expert_weight` as primary handler (checked first), keeping packed format as fallback. Added `config = model.config` for `num_experts` lookup.

---

## Summary

| # | Category | Severity | Effort |
|---|----------|----------|--------|
| 1 | Duplicate method | Critical | Small |
| 2 | Wrong attribute name | Critical | Small |
| 3 | Bad heuristic | Medium | Small |
| 4 | Buffer sizing (3 issues) | Critical | Medium |
| 5 | Missing mode transition | Critical | Medium |
| 6 | DisposibleTensor refcount | Critical | Medium |
| 7 | Missing kernel port | Critical | Large |
| 8 | NVL buffer sizing | Critical | Small |
| 9 | Duplicate code | Low | Small |
| 10 | Init ordering | Critical | Small |
| 11 | EP/TP slice missing | Critical | Medium |
| 12 | Wrong unpack | Critical | Small |
| 13 | Missing BF16 kernel | Critical | Large |
| 14 | Scale dim assertion (×2) | Critical | Medium |
| 15 | Config serialization | Critical | Medium |
| 16 | Stale imports | Critical | Small |
| 17 | Checkpoint format mismatch | Critical | Medium |

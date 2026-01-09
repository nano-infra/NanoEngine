# Qwen3 Debugging Status Report (Legacy)

**Date:** 2026-01-08
**Objective:** Resolve repetitive garbage output in Qwen3 generation using NanoDeploy.

## 1. Issue Summary

The `Qwen3` model running on `NanoDeploy` (C++ backend) generates coherent output for likely the first token but then degenerates into repetitive garbage (e.g., "QuestionQuestion...").
We suspected issues in:

- Rotary Embedding (RoPE) application (interleaved vs. split layout).
- Attention Mechanism (SDPA vs. FlashInfer usage).
- KV Cache data layout and retrieval.
- Prefill execution (potentially missing or incorrect).

## 2. Work Done & Hypotheses

### Hypothesis 1: RoPE Layout Mismatch

- **Observation:** Qwen/Llama weights from HuggingFace are often interleaved (`cos1, sin1, ...`), while our `RotaryEmbedding` implementation expects a split format (`cos... | sin...`).
- **Action:** Implemented a `permute_for_rope` helper to convert interleaved inputs to split format before RoPE.
- **Result:** The first generated token changed (from `151644` to `198`), but subsequent output remained garbage. This suggests RoPE layout might be *part* of the issue but not the sole cause, or the permutation was incorrect. **Status: Reverted for clean debugging.**

### Hypothesis 2: Attention Logic (Decode vs Prefill)

- **Observation:** We validated a "Slow Path" using PyTorch's `scaled_dot_product_attention` (SDPA) for both Prefill and Decode to isolate issues from `FlashInfer`.
- **Finding:** The `is_causal` flag in SDPA was initially set to `true` for Decode steps (`seq_len=1`). This is incorrect because for a single decode query, attending to the full separate KV history requires `is_causal=false` (assuming `attn_mask` handles causality or causal logic isn't needed for single-token query attending to past).
- **Action:** Attempted to set `is_causal = (seq_len > 1)` (True for Prefill, False for Decode).
- **Status:** Partially implemented but blocked by compilation errors.

### Hypothesis 3: Missing Prefill

- **Observation:** Logs consistently showed "Attn Mode: DECODE" but never "Attn Mode: PREFILL".
- **Implication:** The prefill step might be skipped entirely, or failing to log. If prefill (processing the prompt) doesn't happen, the KV cache remains empty or uninitialized, leading to garbage for any subsequent decode steps.
- **Status:** **Critical Open Issue.** Needs verification in `SimpleEngine`.

## 3. Current State & Legacy Issues

### A. Compilation Instability (`qwen3.h`)

- **Issue:** The template parameter `template <QuantType Q>` conflicts with a Torch internal macro `c10::attr::Q`, causing `template argument invalid` errors (e.g., `std::unique_ptr<layers::RowParallelLinear<Q>>`).
- **Fix Attempt:** Renamed `Q` to `Quant`.
- **Current Status:** The file is currently in a mixed/broken state due to manual reverts and syntax errors (e.g., missing includes, brace mismatches).
- **Action Required:** Clean up `qwen3.h`:
  1. Rename `template <QuantType Q>` to `template <QuantType Quant>` globally in the file.
  2. Ensure `<cstdio>` is included.
  3. Verify closing braces for namespaces.

### B. Garbage Output

- **Issue:** Model generates repetitive "QuestionQuestion...".
- **Investigation:**
  - Verify **Prefill** is actually running.
  - Verify **KV Cache** is being populated correctly (check layout `[B, H, S, D]` vs `[B, S, H, D]` for `set_kv`).
  - Verify **RoPE** correctness (compare against Python reference).

## 4. Next Steps (Legacy Handover)

1. **Fix Compilation**: Correct `qwen3.h` syntax and template parameters.
2. **Verify Prefill**: Add logging in `SimpleEngine` to ensure the first `run` call has `seq_len > 1`.
3. **Debug Attention**: Once compiling, re-enable the "Slow Path" (SDPA) with `is_causal=(seq_len>1)` and checking tensor stats.

# NanoExpert MoE Refactoring — Complete Reference

This directory documents the full MoE (Mixture of Experts) refactoring effort that decouples NanoDeploy from DLBlas, introducing a unified `DistributedRoutedExperts` layer and per-model weight loading architecture.

## Documents

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | System architecture: DistributedRoutedExperts, ExpertContext, TokenDispatcher |
| [weight-loading.md](weight-loading.md) | Per-model weight loading design and `loader.py` utilities |
| [checkpoint-formats.md](checkpoint-formats.md) | Checkpoint key analysis for all supported models |
| [bugfixes.md](bugfixes.md) | All bugs encountered and their fixes during the refactoring |

## Overview

### Goal
Refactor the MoE layer in NanoExpert to:
1. **Decouple from DLBlas** — zero dependency on `dlblas.*` namespace
2. **Unified API** — single `DistributedRoutedExperts` layer replaces scattered EP/TP logic
3. **Native BF16 support** — alongside existing FP8 paths
4. **Centralized buffer management** — `ExpertContext` singleton for DeepEP buffers
5. **Per-model weight loading** — clean, model-specific loaders replacing monolithic logic

### Supported Models

| Model | Config `model_type` | Loader | Expert Format |
|-------|-------------------|--------|---------------|
| DeepSeek V2/V3 | `deepseek_v3` | `deepseek_v2_loader.py` | Per-expert FP8 |
| Qwen3 (dense) | `qwen3` | `qwen3_loader.py` | N/A (no MoE) |
| Qwen3-235B-A22B MoE | `qwen3_moe` | `qwen3_moe_loader.py` | Per-expert BF16 |
| Qwen3.5-397B-A17B MoE | `qwen3_5_moe` | `qwen3_5_moe_loader.py` | Per-expert FP8 |

### Key Dependencies (after refactoring)
- **DeepGEMM** — FP8/BF16 grouped GEMM kernels (`m_grouped_fp8_gemm_nt_masked`, `m_grouped_bf16_gemm_nt_contiguous`, etc.)
- **DeepEP** — Expert parallel communication (`Buffer.dispatch()`, `Buffer.combine()`, `Buffer.low_latency_dispatch()`, `Buffer.low_latency_combine()`)
- **Triton** — Custom kernels (`per_token_group_quant_fp8`, `fused_moe_v3`, `silu_and_mul_masked_post_quant_fwd`)

# implementation_plan.md

# C++ ModelRunner Migration Plan

This plan outlines the steps to migrate the Python `ModelRunner` and associated layers/models to a high-performance C++ implementation using libtorch, FlashInfer, DeepGEMM, and DeepEP.

## User Review Required

> \[!IMPORTANT\]
> This migration involves deep integration with NVIDIA-specific libraries (DeepGEMM, DeepEP). Ensure the target environment has CUDA 12.3+ and compatible RDMA drivers.

## Proposed Changes

### \[Phase 0\] Architecture & Infrastructure

Establish the foundation for the C++ codebase.

#### \[NEW\] \[config.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/core/config.h)

Definition of `ModelConfig` and JSON loader for HF `config.json`.

#### \[NEW\] \[weight_mapping.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/core/weight_mapping.h)

Mapping logic from internal layer names to HuggingFace parameter names.

______________________________________________________________________

### \[Phase 1\] Weight Loading & KV Cache

Implement efficient data handling components.

#### \[NEW\] \[weight_loader.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/weight_loader.h)

`SafeTensorLoader` using `mmap` for fast weight access.

#### \[NEW\] \[kv_cache.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/kv_cache.h)

C++ wrapper for FlashInfer's paged KV cache.

______________________________________________________________________

### \[Phase 2\] Fundamental Operators

Port basic layers using FlashInfer and DeepGEMM.

#### \[NEW\] \[linear.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/layers/linear.h)

Linear layer templates with DeepGEMM FP8 support.

#### \[NEW\] \[attention.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/layers/attention.h)

Attention layer utilizing FlashInfer MLA/FA.

______________________________________________________________________

### \[Phase 3\] MoE & DeepEP

Implement the Sparse MoE block with эксперт parallelism.

#### \[NEW\] \[moe.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/layers/moe.h)

Integration with DeepEP for expert communication.

______________________________________________________________________

### \[Phase 4\] Model & Striped Parallelism

Assemble the full model and implement advanced parallelism.

#### \[NEW\] \[qwen3_moe.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/models/qwen3_moe.h)

Full Qwen3-MoE model assembly.

#### \[MODIFY\] \[attention.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/layers/attention.h)

Add Striped SP logic using `dlslime::AllToAllIntraLLBuffer::allToAllLL2D` for low-latency communication.

______________________________________________________________________

### \[Phase 5\] ModelRunner & Protocol

Final integration with the scheduler and Spoke.

#### \[NEW\] \[model_runner.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/model_runner.h)

The main execution loop, implementing **Local/Global Dual Batch Capture** for CUDA Graph compatibility.

#### \[NEW\] \[model_runner_ipc.h\](file:///d:/src/NanoDeploy/csrc/nanodeploy/worker/model_runner_ipc.h)

Spoke RPC message definitions for `RunReq` and `RunResp`.

## Verification Plan

### Automated Tests

- `test_weight_loader`: Verify HF parity.
- `test_layers`: Compare C++ vs Python output tensors.
- `test_distributed_moe`: Cross-node communication validation.

### Manual Verification

- Run end-to-end inference on 8-GPU node and compare generated tokens with `nanodeploy/worker/model_runner.py`.

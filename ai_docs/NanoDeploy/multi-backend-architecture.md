# Multi-Backend Adaptation Architecture for NanoDeploy

## 1. Goal Description & Background

Currently, the NanoDeploy framework is deeply coupled with the NVIDIA Hopper architecture (e.g., `fused_moe_v3`, TMA, FP8 features) and DeepSeek AI Infra (primarily DeepEP for MoE routing communication).

To support multiple backends (such as generic CUDA architecture, AMD ROCm, or Ascend NPUs) while maintaining maximum performance on Hopper, we are introducing a **Hardware Abstraction Layer (HAL)**. This layer decouples explicit hardware-dependent compute and communication interfaces from the high-level neural network topology (e.g., `qwen3_moe.py`).

### 1.1 The "From-Scratch" Design Dividend

Compared to legacy frameworks that evolved organically:

- **vLLM**: Highly component-based but suffers from historical debt, leading to bloated `layers` directories and complex dispatch mechanisms where new backends require patching across numerous files.
- **SGLang**: Initially mirrored vLLM's monolith approach but is now pivoting to a `hardware_backend` directory (e.g., for NPU), highlighting the necessity of physical isolation for different hardware.
- **NanoDeploy**: We benefit from clear foresight. By physically isolating backends into top-level functional domains (`backends/<platform>`) from day one, we avoid codebase pollution and simplify onboarding new hardware environments.

## 2. Core Design Philosophy: The Three-Layer Separation

The new architecture strictly adheres to the following call chain from top to bottom:

`ModelRunner` -> `Model Definition` -> `Backend Interface (Factory)` -> `Backend Layer (Hardware-Specific)` -> `Backend Kernel (C++/Triton)`

### 2.1 Model & ModelRunner (Unified)

- **Model (e.g., `qwen3_moe.py`)**: There is strictly only ONE set of model definition code. It only describes the "topology" (e.g., Attention followed by MLP). When initializing, it delegates the creation of specific computation layers to `get_backend()`. It MUST NOT contain any hardware-specific imports.
- **ModelRunner**: The core scheduling logic (batching, KV cache lifecycle) remains unified. Hardware-specific operations (like different KV block alignment or CUDA Graph capture logic) are abstracted and delegated to the Backend.

### 2.2 Concept Definitions & Boundaries

- **Quantization / Data Type**:
  - **Role**: Global configuration (e.g., `QuantizationConfig`). It merely *describes* the precision (BF16, FP8 W8A8) as a data structure without any compute logic.
  - **Flow**: Passed from the Model to the Backend Factory during initialization to select the right Layer instantiation.
- **Layer (Carrier of Business Logic & Weights)**:
  - **Role**: Subclasses of `nn.Module`. Responsible for managing `nn.Parameter`, handling Distributed Tensor slice logic (Row/Col Parallel), dealing with Quantization scales (e.g. `Scale Inv`), and organizing the computation graph in `forward()`.
  - **Pain Point Solved**: FP8 Linear (needs Scale Inverse tensors) and BF16 Linear manage parameters entirely differently. They should NOT be forced into a single file with `if/else` checks. They are now distinct `Layer` classes provided by different Backends.
- **Kernel (Pure Compute)**:
  - **Role**: Extremely low-level C++/CUDA/Triton function wrappers (e.g., `fbgemm_fp8_gemm`, `fused_moe_v3`). They have no state, no `nn.Parameter`—only black-box tensor I/O.

## 3. Directory Structure: Clean Physical Isolation

We physically isolate backends at the `nanodeploy/backends/` directory level:

```text
nanodeploy/
├── backends/
│   ├── base_backend.py           # Abstract factory interface (RowParallelLinearBase, etc.)
│   │
│   ├── hopper/                   # ---- Hopper Exclusive Domain (FP8, DeepEP) ----
│   │   ├── __init__.py           # HopperBackendFactory module
│   │   ├── layers/               # Hopper-specific Layers (manages FP8 Weights and Scales)
│   │   │   ├── linear.py         # Contains FP8RowParallelLinear. `forward()` calls kernels.
│   │   │   └── experts.py        # Contains DistributedRoutedExperts_Hopper (DeepEP wrappers)
│   │   └── kernels/              # Hopper-coupled compute kernels
│   │       ├── fused_moe_v3.py   # Triton FP8 MoE Kernel
│   │       └── triton_gemm.py    # TMA WGMMA Kernel
│   │
│   └── gpu_generic/              # ---- Generic GPU Domain (BF16, No special infra dependency) ----
│       ├── __init__.py           # GenericBackendFactory module
│       └── layers/               # Generic PyTorch layers (No FP8 scales)
│           ├── linear.py         # BF16RowParallelLinear (wraps standard torch.matmul)
│           └── experts.py        # Explicitly raises NotImplementedError for MoE routed experts.
│
├── layers/                       # ---- Generic Math & Control Logic ----
│   ├── layernorm.py              # (RMSNorm) High-frequency common layer, hardware agnostic.
│   ├── rotary_embedding.py       # (RoPE) Coordinate transformation.
│   └── activation.py             # (SiLU) Generic activation functions.
│
└── models/                       # ---- Pure Business Logic ----
    ├── quant_config.py           # Describes e.g., {"dtype": "fp8", "block_size": [128,128]}
    └── qwen3_moe/
        └── qwen3_moe.py          # Pure topology. Asks Backend Factory to instantiate layers.
```

## 4. Implicit vs. Explicit Distributed Layers

A key architectural decision is classifying which `Layers` belong in the top-level generic `nanodeploy/layers/` directory versus the backend-specific `nanodeploy/backends/<name>/layers/` directory.

Although all layers in the engine operate on tensors sliced by Tensor Parallelism (TP) or Sequence Parallelism (SP) — meaning all layers are distributed in a macro sense — the boundary is whether the communication is **Implicit** or **Explicit**:

1. **Explicit Distributed Layers MUST sink to Backend**:
   - **Characteristics**: In addition to local computation, they **must communicate across devices via NICs (NVLink/RoCE/IB)**. They contain explicit `dist.all_reduce()`, `all_to_all()`, or rely on underlying communication libraries like `DeepEP`.
   - **Examples**: `FP8RowParallelLinear` (requires AllReduce after GEMM), `DistributedRoutedExperts` (EP slicing requires routing tokens across nodes). These are highly sensitive to hardware topology and must be backend-specific.
2. **Implicit Distributed Layers STAY in Top-Level `layers/`**:
   - **Characteristics**: Only process local sliced data (Local Tensor) with **zero cross-card communication logic**.
   - **Examples**: `RMSNorm`, `SiLU`. Though they operate in a TP/SP context, each GPU independently calculates standard element-wise math along the hidden dimension of its local portion.
   - **Conclusion**: Since their mathematical formulation and parameter shapes (`weight`) remain identical regardless of Hopper or generic GPU, isolating them as shared infrastructure prevents over-engineering and eliminates redundant code.

## 5. IDE Navigation & Type Hinting (Developer Experience)

Dynamic factory creation (`get_backend().get_linear()`) typically breaks IDE "Go to Definition" features. To solve this enterprise pain point, we strictly use Python Type Hinting with abstract base classes. This forces the framework to define a rigorous contract for every layer.

```python
# nanodeploy/backends/base_backend.py
class RowParallelLinearBase(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class BackendFactory:
    def get_linear(self, type: str, ...) -> RowParallelLinearBase: ...
```

When building models:

```python
self.linear = get_backend().get_linear("row_parallel", ...)
# IDE autocomplete works because it statically knows this is a RowParallelLinearBase.
# 'Find Implementations' resolves directly to the Backend specific implementations.
```

## 6. Implementation Timeline & Resources

- **Resource Allocation**: 1 Core AI Infra Architect.
- **Estimated Duration**: 1 Week (approx. 5 working days).

### Phase 1: Foundation & Base Migration (2 Days)

- Define global Interfaces (`BackendFactory`, `Layer` bases).
- Create `hopper` and `gpu_generic` backend directories.
- Move existing kernels in `nanodeploy/kernels/` to `backends/hopper/kernels/`.

### Phase 2: Hopper Layer Refactoring & DeepEP Isolation (1.5 Days)

- Refactor `distributed_routed_experts.py` into `HopperRoutedExperts`, isolating DeepEP logic.
- Encapsulate FP8 Triton GEMM kernels into `FP8Row/ColumnParallelLinear`.

### Phase 3: Generic GPU Fallback & Model Decoupling (1.5 Days)

- Implement `gpu_generic/layers/linear.py` utilizing standard `torch.matmul`.
- Refactor Qwen3 MoE topology (`qwen3_moe.py`) to remove hardcoded imports, transitioning to dynamic interface instantiations (`get_backend()`).
- Validate end-to-end exact match against existing benchmarks/logs.

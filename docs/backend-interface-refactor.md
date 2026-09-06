# Layer interface and backend implementation refactor

This document describes the planned cleanup tracked by [Issue #325](https://github.com/JimyMa/NanoDeploy/issues/325). It is a design and migration plan; it does not change runtime behavior by itself.

## Motivation

The current `runtime/layers` tree mixes three different responsibilities:

1. layer contracts and distributed-parallel interfaces;
2. hardware policy and capability selection; and
3. concrete implementations and kernel adapters.

GDN is the clearest example: projection setup, causal convolution, recurrent state management, recurrence math, output normalization, and kernel selection are currently coupled in the generic implementation and then partially overridden by specialized implementations. This makes reference execution, hardware fallback, and new model-specific shapes harder to reason about.

## Target model

The refactor uses one abstract interface layer, a policy-driven registry, and backend implementations:

```text
model topology
    -> layer interface / abstract contract
    -> backend policy
    -> backend registry
    -> concrete implementation
```

The interface is responsible only for the operation contract and policy-based implementation lookup. It must not contain FlashInfer, FLA, DeepGEMM, or model-type-specific execution branches.

Concrete implementations live below `runtime/layers/backends/`:

```text
runtime/layers/
    base_backend.py          # contracts and parallel layer interfaces
    backend_policy.py        # policy schema and registry entry points
    backends/
        generic/             # portable GPU implementations
        ref/                 # correctness-first reference implementations
        flashinfer/          # FlashInfer implementations
        fla/                 # FLA implementations
        deepseek/             # DeepSeek-specific optimized implementations
        fa/                  # Flash-Attention implementations
```

`generic` and `ref` are implementation families, not top-level interfaces. A separate `torch` backend is not required: reference implementations may use PyTorch internally without exposing that fact as a backend category.

## Policy selection

Hardware-specific modules such as Hopper and Blackwell provide policy, not model-layer implementations. A policy describes the preferred implementation, allowed fallbacks, and capability constraints:

```python
BackendPolicy(
    preferred="flashinfer",
    fallback="generic",
    ref_fallback_allowed=True,
    supports_fp8=True,
)
```

Configuration and environment overrides select the execution family explicitly:

```text
backend_mode=native|generic|ref
fallback_backend=generic|ref
```

The normal native policy may select an optimized implementation, while an unsupported shape (for example, GLM-5.3's special MLA dimensions) can resolve to `generic` or `ref` without model code importing a hardware backend.

## GDN decomposition

GDN/KDA should be composed from independently testable parts:

- projection bundle: q/k/v/g/b/forget projections and decode packing;
- causal depthwise convolution: prefill and decode update paths;
- recurrent state: layout, allocation, slot mapping, reset, and continuation handling;
- recurrence: prefill and decode delta-rule implementations;
- output transform: RMS normalization, gate activation, and output projection;
- backend composition: selects the implementations above according to policy.

The reference, generic, FlashInfer, and FLA implementations can then share contracts and state handling without inheriting one large `GenericGatedDeltaNet` class. Kernel-specific code should implement only the relevant component or adapter.

## Migration plan

1. Introduce the policy schema and backend registry without changing the default selection.
2. Move the current `runtime/layers/generic` implementations under `backends/generic`, retaining compatibility imports temporarily.
3. Add `backends/ref` for deterministic correctness paths and precision alignment.
4. Decompose GDN and migrate its state/projection/recurrence components independently.
5. Convert Hopper, Blackwell, and native selection code to policy providers.
6. Migrate model topologies to the abstract interfaces and remove model-specific branches from implementation classes.
7. Delete compatibility shims only after all model families and focused tests use the new paths.

Each implementation step should preserve existing focused tests and add a reference-vs-backend token or tensor comparison where the operation is numerically sensitive.

## Compatibility and review boundaries

This plan intentionally separates correctness cleanup from performance work. Existing optimized kernels remain available during migration, and the reference backend is an explicit fallback rather than an implicit global mode. Individual migration PRs should be limited to one operation family or one compatibility layer so they can be reviewed and reverted independently.

Follow-up implementation PRs should reference #325 and the GLM-5.3 workstream #322; this planning PR does not close either issue.

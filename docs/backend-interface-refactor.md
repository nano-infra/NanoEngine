# Layer interface and backend implementation refactor

This document describes the planned cleanup tracked by [Issue #325](https://github.com/JimyMa/NanoDeploy/issues/325). It is a design and migration plan; it does not change runtime behavior by itself.

## Motivation

The current `dlengine/runtime/layers` tree mixes three different responsibilities:

1. layer contracts and distributed-parallel interfaces;
2. hardware policy and capability selection; and
3. concrete implementations and kernel adapters.

GDN is the clearest example: projection setup, causal convolution, recurrent state management, recurrence math, output normalization, and kernel selection are all coupled inside a single `GenericGatedDeltaNet` class (`generic/gated_delta_net.py`), which is then subclassed by `FlashInferGatedDeltaNet`, `FLAGatedDeltaNet`, and `TorchGatedDeltaNet` mostly by toggling internal `_HAS_*` capability flags. KDA (`FlashInferKDA`) escapes this hierarchy entirely and is imported directly by models. This makes reference execution, hardware fallback, and new model-specific shapes hard to reason about.

### Named design problems

The `model -> get_backend() -> *Base` contract (`base_backend.py`) is sound: model
topology depends on abstract interfaces, not on concrete kernels. That edge is fine.
The problems are below it, inside the selection-and-implementation layers. To be
precise about what is wrong (and to avoid mislabelling), the concrete issues are:

- **Miscut abstraction boundary.** The tree is split by *hardware tier* (`hopper/`,
  `blackwell/`) when the natural axis is *vendor/kernel* (DeepGEMM, FlashInfer, FLA,
  NVFP4, ...). A DeepGEMM GEMM does not *belong to* Hopper; it *requires* SM90+. Storing
  it under `hopper/` conflates a capability constraint with implementation ownership.

- **Inverted dependency direction (concrete-on-concrete).** The vendor package depends
  on the hardware-tier package rather than the reverse: `backends/deepseek/linear.py`
  imports the real classes from `hopper/linear.py` and re-exports them as aliases. The
  intuitive direction ("a tier selects vendor implementations") is reversed, and two
  concrete modules are coupled directly with no abstraction between them. Note this is
  *not* a violation of the Dependency Inversion Principle at the model boundary — that
  boundary is correct; it is a wrong-way coupling internal to the backend layer.

- **Single Responsibility violation.** `hopper/` carries two jobs at once: policy
  ("which implementation for this tier") and implementation (the actual FP8/DeepGEMM
  kernels). These change for different reasons and should not live together.

- **Duplicated selection logic (DRY).** Two independent selection systems coexist (the
  hardware factory chooses linear/experts inside its `get_*` methods; `selector.py`
  chooses attention/GDN). As a direct symptom, `BackendPlan.linear` and
  `BackendPlan.experts` are dead fields never used at runtime.

- **Inheritance used for configuration.** `BlackwellBackendFactory(HopperBackendFactory)`
  subclasses only to override two `get_*` methods. A per-tier difference in *which*
  implementation to pick is data (a policy table), not a type relationship; this is a
  "prefer composition over inheritance" case.

- **Escaped special case.** `FlashInferKDA` bypasses both the factory and the selector
  and is imported directly by models, with hard-coded no-fallback error paths — an
  implementation detail leaking into topology code.

These are real maintainability problems, not stylistic preferences: they are why
adding a new shape (e.g. GLM-5.3 MLA dims), a new vendor kernel, or an explicit
reference fallback currently requires touching several coupled places at once. The
refactor targets each named problem directly (see Target model and Migration plan).

## Current layout (as-is)

```text
dlengine/runtime/layers/
    __init__.py              # process-local backend holder: init/get/set/reset_backend
    base_backend.py          # ABCs: BackendFactory + Base classes per layer family
    backend_selection.py     # hardware resolution + factory construction
    backends/
        selector.py          # capability-aware plans + create_attention / create_gdn
        deepseek/            # aliases to hopper/* (linear, experts)
        fa/                  # fa2 / fa3 / fa4 attention adapters
        fla/                 # FLA GDN adapter
        flashinfer/          # FlashInfer attention + GDN adapters
        kda/                 # FlashInferKDA (bypasses selector/factory)
        megamoe/             # DeepGEMM MXFP4 experts
        nvfp4/               # Blackwell NVFP4 experts
        torch/               # aliases to generic/* (attention, experts, gdn, linear)
    generic/                 # BF16 base implementations (the real classes)
    hopper/                  # FP8 / DeepGEMM / DeepEP implementations
    blackwell/               # SM100+ implementations (extends Hopper)
```

Two selection systems coexist today:

- **Hardware factory**: `backend_selection.resolve_backend_selection()` picks
  `blackwell`/`hopper`/`gpu_generic` and `create_backend()` constructs the matching
  `*BackendFactory`. The factory chooses linear/experts implementations directly in
  its `get_*` methods.
- **Composable selector**: `backends/selector.py` resolves per-family *plans*
  (`AttentionBackendPlan`, `GDNBackendPlan`) and `create_attention` / `create_gdn`
  instantiate the concrete adapter. The factory delegates attention and GDN to this
  selector, but the `linear` and `experts` fields of `BackendPlan` are currently
  unused at runtime.

`FlashInferKDA` is a hard-wired special case: models import it directly
(`models/glm5_next`, `models/kimi_k3`) and it has explicit no-fallback error paths.

### Implementation ownership is inverted today

The most confusing part of the current tree is that the hardware-tier packages own
the real implementations while the vendor/kernel packages are empty aliases:

- `hopper/linear.py` holds the *real* DeepGEMM/FP8 linear classes
  (`HopperColumnParallelLinear`, ...).
- `backends/deepseek/linear.py` merely re-exports those Hopper classes under aliases
  (`HopperColumnParallelLinear as ColumnParallelLinear`); it has no logic of its own.
- `blackwell/__init__.py` has no linear/experts implementations at all — it subclasses
  `HopperBackendFactory` and only overrides `get_distributed_routed_experts`
  (nvfp4 / mxfp4 branch) and `get_attention`.

So `deepseek` (a vendor/kernel name) is a shell, while `hopper` (a hardware tier)
carries the implementation. Conceptually a DeepGEMM FP8 GEMM does not *belong to*
Hopper — it *requires* SM90+. Naming and storing it under `hopper/` conflates a
capability requirement with implementation ownership, and it is why `hopper` and the
`backends/deepseek` alias look duplicated.

## Target model (to-be)

The refactor keeps one abstract interface layer, a policy-driven selection stage,
and backend implementations:

```text
model topology
    -> layer interface / abstract contract   (base_backend.py)
    -> backend policy                         (backend_selection.py + backends/selector.py)
    -> backend registry / factory
    -> concrete implementation
```

The interface (`base_backend.py`) is responsible only for the operation contract and
policy-based implementation lookup. It must not contain FlashInfer, FLA, DeepGEMM, or
model-type-specific execution branches.

The key structural change is to **invert the current ownership**: implementations are
named after the vendor/kernel that provides them and live under `backends/`, while the
hardware-tier packages become policy-only.

- `backends/` owns every concrete implementation, named after its vendor/kernel and
  declaring its own capability requirement (e.g. "requires SM90+"):
  - `backends/deepseek/` — the FP8 + DeepGEMM linear/experts implementations that
    currently live in `hopper/linear.py` and `hopper/experts.py`. After migration the
    real classes live here and are no longer re-exported aliases.
  - `backends/fa/`, `backends/flashinfer/`, `backends/fla/`, `backends/kda/`,
    `backends/megamoe/`, `backends/nvfp4/` — the remaining kernel adapters.
  - `backends/generic/` (a `ref` / portable BF16 family) — the correctness-first and
    non-optimized-GPU implementations that currently live in `generic/`. A separate
    top-level `torch` backend is not required; `backends/torch/*` stay as thin
    compatibility aliases only during migration.
- `hopper/` and `blackwell/` become **policy providers**, not implementation
  containers. They hold a policy table for their capability tier (preferred
  implementation per layer family, allowed fallbacks, capability constraints) and no
  longer own kernel classes. `BlackwellBackendFactory` subclassing
  `HopperBackendFactory` just to override two `get_*` methods is exactly the pattern
  that a policy table replaces.
- `backends/selector.py` remains the single place that turns policy + capability into
  a concrete `BackendPlan` and instantiates the implementations.

`generic` and `ref` are implementation *families*, not top-level interfaces.
Reference paths may use PyTorch internally without exposing that as a backend
category.

The end state is a **single selection stage**: today there are two systems (the
hardware factory in `backend_selection.py` choosing linear/experts inside its `get_*`
methods, and the composable `selector.py` choosing attention/GDN). Because linear and
experts are chosen inside the factory, `BackendPlan.linear` and `BackendPlan.experts`
are currently dead fields. After the refactor the factory delegates *all* families to
`resolve_backend_plan()`, so those fields become live and the hardware tiers contribute
only policy.

## Policy selection

Hardware tiers (Hopper, Blackwell) provide policy, not just implementations. A policy
describes the preferred implementation, allowed fallbacks, and capability constraints.
Conceptually:

```python
BackendPolicy(
    preferred="flashinfer",
    fallback="generic",
    ref_fallback_allowed=True,
    supports_fp8=True,
)
```

The existing config fields already select the execution family explicitly and are
kept as-is:

```text
hardware_backend  = auto | blackwell | hopper | gpu_generic
attention_backend = auto | fa2 | fa3 | fa4 | flashinfer | torch
gdn_backend       = auto | flashinfer | fla | torch
```

The normal policy may select an optimized implementation, while an unsupported shape
(for example, GLM-5.3's special MLA dimensions) can resolve to `generic`/`ref`
without model code importing a hardware backend.

## Reference fallback configuration

Today "fallback" is expressed only two ways: ad-hoc `_HAS_*` probes inside
`GenericGatedDeltaNet`, and the narrow `enable_mla_reference_fallback` flag. There is
no unified, config-level fallback control.

This refactor introduces a single explicit switch rather than three separate
per-family flags. Instead of `--gdn_fallback`, `--attention_fallback`, and
`--expert_fallback`, the three are collapsed into one policy field:

```text
ref_fallback_allowed = true | false   (default: false)
```

Semantics:

- When `true`, the selection stage may degrade a preferred/native implementation to a
  `generic` or `ref` implementation when the native path cannot serve the requested
  shape or capability (e.g. an MLA head-dim FlashMLA does not instantiate, or a
  FlashInfer/FLA GDN kernel is unavailable). This replaces the implicit `_HAS_*`
  probe-and-degrade behaviour with an explicit, policy-gated decision.
- When `false`, an unsupported shape or missing kernel raises instead of silently
  falling back, preserving deterministic performance expectations.

`ref_fallback_allowed` maps to the `ref_fallback_allowed` field of `BackendPolicy`
and applies uniformly across attention and GDN. Expert fallback is a **future goal**:
`generic` experts do not currently support `ep_size > 1`, and there is no `ref`
experts implementation yet, so expert ref-fallback is out of scope for the initial
landing and is documented as a follow-up (build `backends/ref/experts` plus generic
EP support first).

New config field (added in the implementation PR, not this planning PR):

```python
# dlengine/config.py, runner config section
ref_fallback_allowed: bool = False
```

It is threaded into `resolve_backend_selection()` alongside the existing
`requested_hardware`/`requested_attention`/`requested_gdn` arguments and recorded in
`BackendSelection` so the chosen fallback and its reason are observable.

## GDN decomposition

GDN/KDA should be composed from independently testable parts instead of one large
`GenericGatedDeltaNet` class:

- projection bundle: q/k/v/g/b/forget projections and decode packing;
- causal depthwise convolution: prefill and decode update paths (currently
  `_apply_conv1d`, guarded by `_HAS_CAUSAL_CONV1D`);
- recurrent state: layout, allocation, slot mapping, reset (`_zero_fresh_slots`), and
  continuation handling;
- recurrence: prefill and decode delta-rule implementations;
- output transform: RMS normalization (`RMSNormGated` / `SigmoidRMSNormGated`), gate
  activation, and output projection;
- backend composition: selects the implementations above according to policy.

The reference, generic, FlashInfer, FLA, and KDA implementations can then share
contracts and state handling without inheriting one monolithic class. In particular,
`FlashInferKDA` — which today reimplements `nn.Module.__init__` and only borrows
`_apply_conv1d` / `_zero_fresh_slots` by inheritance — should be expressible through
the same shared components and brought under the factory/selector rather than being
imported directly by models.

## Migration plan

1. Introduce a `BackendPolicy` schema (preferred/fallback/`ref_fallback_allowed`/
   capability flags) and thread it through `backend_selection.py` and
   `backends/selector.py` without changing the default selection.
2. Add the `ref_fallback_allowed` config field and wire it into
   `resolve_backend_selection()`; record the effective fallback in `BackendSelection`.
3. Formalise `generic/` as the `ref`/portable family and add deterministic
   correctness paths where operations are numerically sensitive.
4. Decompose GDN into projection/conv/state/recurrence/output components and migrate
   them independently; bring `FlashInferKDA` under the shared contracts.
5. Invert implementation ownership: move the real FP8/DeepGEMM classes from
   `hopper/linear.py` and `hopper/experts.py` into `backends/deepseek/`, and delete the
   `backends/deepseek` re-export aliases. Move `generic/*` implementations into
   `backends/generic/` (the `ref` family). Each moved implementation declares its own
   capability requirement.
6. Convert Hopper and Blackwell into policy providers (a policy table per tier), remove
   their implementation classes and the `BlackwellBackendFactory(HopperBackendFactory)`
   override pattern, and drive linear/experts through `resolve_backend_plan` so the
   `BackendPlan.linear`/`BackendPlan.experts` fields become live instead of dead. This
   collapses the two selection systems into one.
7. Migrate model topologies to the abstract interfaces and remove model-specific
   branches (including direct KDA imports) from implementation classes.
8. Add `ref` experts and generic EP support, then extend `ref_fallback_allowed` to the
   experts family.
9. Delete compatibility shims (`backends/torch/*` aliases, `backends/deepseek/*`
   aliases, legacy `init_backend` env reads) only after all model families and
   focused tests use the new paths.

Each implementation step should preserve existing focused tests and add a
reference-vs-backend token or tensor comparison where the operation is numerically
sensitive.

## Compatibility and review boundaries

This plan intentionally separates correctness cleanup from performance work. Existing
optimized kernels remain available during migration, and the reference family is an
explicit fallback (gated by `ref_fallback_allowed`) rather than an implicit global
mode. Individual migration PRs should be limited to one operation family or one
compatibility layer so they can be reviewed and reverted independently.

Follow-up implementation PRs should reference #325 and the GLM-5.3 workstream #322;
this planning PR does not close either issue.

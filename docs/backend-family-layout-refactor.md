# Backend family layout and attention ownership refactor

This document is the follow-up plan to [Issue #325](https://github.com/JimyMa/NanoDeploy/issues/325)
(see `docs/backend-interface-refactor.md`). #325 separated interfaces, policy, and
implementations, and inverted ownership for **linear**, **experts**, and **GDN**.
This plan finishes the job for the remaining family — **attention** — and unifies the
whole `backends/` tree around a single "family / vendor" convention so the residual
hardware-tier packages (`hopper/`, `blackwell/`, `generic/`) can be deleted.

It is a design and migration plan; it does not change runtime behavior by itself.

## Motivation

After #325 the tree is half-migrated:

- `backends/deepseek/{linear,experts}.py` and `backends/generic/{linear,experts,gated_delta_net}.py`
  hold real implementations (good).
- But `hopper/attention.py` and `blackwell/attention.py` still hold the real attention
  implementations, and `hopper/`, `blackwell/`, `generic/` still exist only because of
  them plus the three thin `*BackendFactory` shells. `backends/fa/fa3.py` and
  `backends/fa/fa4.py` are alias-only re-exports pointing back at those hardware-tier
  modules — the same inverted ownership #325 removed for linear/experts, still present
  for attention.

Concretely, the attention family still exhibits every problem #325 named:

- **Miscut boundary / inverted ownership**: `hopper/attention.py` owns `HopperAttention`
  (which is really FA3 + FlashMLA) and `blackwell/attention.py` owns `BlackwellAttention`
  / `BlackwellMLAAttention` (really FA4 + TRTLLM-GEN MLA). `backends/fa/fa3.py` /
  `fa4.py` import *from* the hardware-tier package.
- **Hardware words in type names**: `HopperAttention`, `BlackwellAttention`,
  `BlackwellMLAAttention`, and the mislabeled `_FA2AttentionImpl` (it is the generic
  QK·softmax·V impl with FA2/FlashInfer/SDPA dispatch, not "FA2-only").
- **Inheritance for configuration**: `BlackwellMLAAttention(HopperAttention)` and
  `BlackwellAttention(HopperAttention)` subclass across tiers to reuse code.
- **Leaked shared helpers**: `topk_indices_to_physical`, `_gather_cache_cached_only`,
  `_interleave_cached_fresh`, `_gather_kv_cached_concat`, and
  `_hisparse_prefill_fresh_slot_mapping` live inside `hopper/attention.py` /
  `generic/attention.py` and are imported directly by `models/deepseek_v2`,
  `runner/mtp_runner`, and tests.

There is also no first-class home for the **DSA/NSA sparse-attention** path (the
DeepSeek/GLM indexer + sparse MLA decode). It is currently threaded through
`deepseek_v2.py` plus `nsa_index_topk`/`index_topk` flags and the `Indexer` in
`layers/indexer.py`, with no explicit backend family.

## Target: one family/vendor convention

Every operation family becomes a directory under `backends/`, and every concrete
implementation is a file named after its **vendor/kernel** (never a hardware tier):

```text
dlengine/runtime/layers/backends/
    selector.py            # resolve_*_plan + create_* for every family
    attention/             # dense GQA attention
        __init__.py
        base.py            # GqaAttentionBase + shared GQA cache/gather scaffolding
        mla_utils.py       # shared paged-cache gather/scatter + topk->physical
        generic.py         # GenericAttention (pure SDPA reference; no FA/FlashInfer)
        fa2.py             # Fa2Attention        (FlashAttention-2)
        fa3.py             # Fa3Attention        (FlashAttention-3, SM90)
        flashinfer.py      # FlashInferAttention (paged FlashInfer decode)
    mla/                   # dense MLA (compressed-KV, absorbed projection)
        __init__.py
        base.py            # MlaAttentionBase
        flash_mla.py       # FlashMlaAttention   (Hopper FlashMLA prefill+decode)
        trtllm.py          # TrtllmMlaAttention  (Blackwell TRTLLM-GEN decode)
        reference.py       # RefMlaAttention     (varlen/decode reference fallback)
    dsa/                   # DeepSeek/GLM sparse attention (own family; see
                           # docs/dsa-backend-family.md) — indexer + sparse MLA
    experts/
        __init__.py
        generic.py         # GenericExperts      (BF16 + EP reference)      <- backends/generic/experts
        deep_gemm.py       # DeepGemmExperts     (FP8 DeepGEMM/DeepEP)      <- backends/deepseek/experts
        mega_moe.py        # MegaMoEExperts      (MXFP4)                    <- backends/megamoe
        nvfp4.py           # NvFp4Experts        (Blackwell NVFP4)          <- backends/nvfp4
    linear/
        __init__.py
        generic.py         # Generic*Linear      (BF16)                     <- backends/generic/linear
        deep_gemm.py       # DeepGemm*Linear     (FP8 DeepGEMM)             <- backends/deepseek/linear
    delta_net/
        __init__.py
        base.py            # GatedDeltaNetBase re-export
        components/        # conv / state / recurrence / output / kernels   <- backends/generic/gdn
        generic.py         # GenericGatedDeltaNet                           <- backends/generic/gated_delta_net
        flashinfer.py      # FlashInferGatedDeltaNet                        <- backends/flashinfer/gdn
        fla.py             # FlaGatedDeltaNet                               <- backends/fla/gdn
        torch.py           # TorchGatedDeltaNet                             <- backends/torch/gdn
        kda.py             # FlashInferKda (Kimi Delta Attention)           <- backends/kda
```

Naming convention for classes: **`<Vendor><Family>`**, no hardware tier words.

Attention-family renames already landed (stage 2, PR #343):
`HopperAttention→Fa3Attention`, `BlackwellAttention→Fa4Attention`,
`BlackwellMLAAttention→Fa4MlaAttention`, `FlashAttentionImpl→Fa3AttentionImpl`,
`FlashMLAImpl→FlashMlaAttentionImpl`, `BlackwellAttentionImpl→Fa4AttentionImpl`,
`_FA2AttentionImpl→_GenericAttentionImpl`, `FA2Attention→Fa2Attention`.

Remaining renames (later stages):

| old name | new name | family |
| --- | --- | --- |
| `FlashMlaAttentionImpl` (in `attention/fa3.py`) | `FlashMlaAttention` (in `mla/flash_mla.py`) | mla |
| `Fa4MlaAttention` (in `attention/fa4.py`) | `TrtllmMlaAttention` (in `mla/trtllm.py`) | mla |
| MLA reference (in `deepseek_v2`) | `RefMlaAttention` (in `mla/reference.py`) | mla |
| `HopperDistributedRoutedExperts` | `DeepGemmExperts` | experts |
| `GenericDistributedRoutedExperts` | `GenericExperts` | experts |
| `Hopper*Linear` | `DeepGemm*Linear` | linear |
| `Generic*Linear` | `Generic*Linear` (unchanged) | linear |
| `FlashInferKDA` | `FlashInferKda` | delta_net |

### `generic` attention must be pure SDPA (single responsibility)

The attention-family migration relocated the files but left `GenericAttention`
(`attention/generic.py`) as a **multi-strategy dispatcher**: its
`_GenericAttentionImpl` carries `use_fa2` / `use_flashinfer_decode` /
`use_flashinfer_prefill` flags and branches into FlashAttention-2, FlashInfer
paged, and SDPA at runtime. `attention/fa2.py`, `attention/flashinfer.py`, and
`attention/torch.py` are then all thin subclasses that just flip those flags on
the *same* impl. That is a single class owning four backends — the opposite of
the family/vendor split.

Target:

- **`generic.py`** owns only the **pure SDPA / naive** reference path (no
  `flash_attn`, no `flashinfer`). It is the portable correctness backend and the
  `ref_fallback_allowed` target for GQA.
- **`fa2.py`** owns the FlashAttention-2 kernel path; **`flashinfer.py`** owns the
  FlashInfer paged prefill/decode path. They no longer subclass `GenericAttention`
  by toggling flags — they contain their own kernel calls.
- **`torch.py` is deleted.** "Torch attention" was just `Fa2Attention` with all
  accelerated flags turned off, i.e. the SDPA path — which is exactly what
  `generic` now is. The debug/correctness backend is `generic`; a `torch`
  attention backend is redundant.
- Shared GQA plumbing (KV-cache store, paged gather, HiSparse SWA, FlashInfer
  metadata/plan caching) moves to `attention/base.py` (`GqaAttentionBase`) so the
  three kernel backends reuse it instead of duplicating or sharing one god-impl.

The `attention_backend` config value `torch` is remapped to `generic` (kept as a
deprecated alias for one release), and `resolve_attention_plan` drops the `torch`
name.

### The DSA/NSA sparse-attention family

DeepSeek-V3.2 and GLM-5.3 use sparse MLA: an **indexer** selects top-k keys per query,
then a sparse MLA kernel attends only those. Because this is a *compose of* an indexer
(with paged-FP8/reference/pooled/fused-topk strategies) and a sparse kernel
(FlashMLA/TRTLLM-GEN/reference), it is promoted to its **own top-level family**
`backends/dsa/` rather than a single `attention/dsa.py` file.

The full design — layout, the indexer × sparse-kernel selection matrix, the factory
contract (`get_dsa_attention` / `resolve_dsa_plan` / `create_dsa`), and its own 6-stage
migration — is specified in **`docs/dsa-backend-family.md`**. It supersedes the
single-file framing here.

### Shared MLA helpers

`topk_indices_to_physical`, `_gather_cache_cached_only`, `_interleave_cached_fresh`,
`_gather_kv_cached_concat`, and `_compute_cached_split` move to
`backends/attention/mla_utils.py`. `_hisparse_prefill_fresh_slot_mapping` and the
HiSparse SWA helpers move alongside the generic attention impl. Consumers
(`deepseek_v2.py`, `mtp_runner.py`, tests) import from the new module; the old
`hopper.attention` / `generic.attention` import paths are removed once repointed.

These are paged-cache gather/scatter utilities, so an alternative is to push them down
to `runtime/kernel/`. This plan keeps them in `attention/mla_utils.py` (a backend-layer
concern, not a raw kernel) unless review prefers the kernel layer.

## Selection

`backends/selector.py` remains the single selection stage. `resolve_attention_plan`
gains explicit vendor names and an MLA/DSA axis:

```text
attention plan = (family, prefill_impl, decode_impl)
  GQA  -> generic(SDPA) | fa2 | fa3 | flashinfer
  MLA  -> flash_mla (Hopper) | trtllm (Blackwell) | reference (ref_fallback_allowed)
  DSA  -> dsa (indexer + sparse MLA), reference fallback when ref_fallback_allowed
```

`create_attention` dispatches on `(attention_type, plan)` and instantiates the
`attention/`, `mla/`, or `dsa/` implementation directly — no import from `hopper/` or
`blackwell/`. The `hardware_backend` argument is replaced by the capability-derived
plan, so the `hardware_backend == "blackwell"/"hopper"` branches in `create_attention`
disappear. The GQA `torch` name is dropped (`generic` is the SDPA backend).

## Deleting the hardware-tier packages

Once attention is migrated and linear/experts/delta_net already live under
`backends/`, the only remaining content of `generic/`, `hopper/`, `blackwell/` is the
three `*BackendFactory` shells. Those collapse into `policy_backend.PolicyBackendFactory`
(already the single real factory; #329). `backend_selection.create_backend()` then
constructs `PolicyBackendFactory(quant_config, tier=selection.hardware)` directly, and
`generic/`, `hopper/`, `blackwell/` are deleted. `TIER_POLICIES` keeps the tier names
as **policy keys** — the hardware concept survives as policy data, not as packages.

## Migration plan (each stage is an independent, reviewable PR)

1. **This doc PR.** Terminology, target tree, naming table, DSA family definition.
2. **Attention migration.** *(done — PR #343)* Moved `hopper/attention.py`,
   `blackwell/attention.py`, `backends/generic/attention.py` into
   `backends/attention/`, split `mla_utils.py`, renamed classes, repointed consumers.
3. **`generic` SDPA split + drop `torch`.** Extract shared GQA plumbing into
   `attention/base.py` (`GqaAttentionBase`); make `generic.py` a pure SDPA reference;
   give `fa2.py` / `flashinfer.py` their own kernel calls instead of flag-toggling one
   impl; delete `attention/torch.py` and remap the `torch` attention name to `generic`.
4. **MLA family.** Extract MLA decode impls from `attention/fa3.py`
   (`FlashMlaAttentionImpl`) and `attention/fa4.py` (`Fa4MlaAttention`) into
   `backends/mla/{flash_mla,trtllm}.py` (`FlashMlaAttention` / `TrtllmMlaAttention`),
   add `mla/reference.py`, and add `resolve_mla_plan` / `create_mla` to the selector.
   (MLA prefill currently in `DeepseekV2Attention.forward` is pulled into the backend
   in the DSA stage, since DSA reuses the MLA prefill/decode path.)
5. **DSA family.** Per `docs/dsa-backend-family.md`: extract indexer + sparse kernels +
   state into `backends/dsa/`, move sparse orchestration (and MLA prefill) out of
   `DeepseekV2Attention.forward` into `DsaAttention`, wire `get_dsa_attention` /
   `resolve_dsa_plan` / `create_dsa`, fold `enable_mla_reference_fallback` into
   `ref_fallback_allowed`.
6. **Family regrouping for the settled families.** Move `backends/deepseek/*` ->
   `backends/{linear,experts}/deep_gemm.py`, `backends/generic/{linear,experts}` ->
   `backends/{linear,experts}/generic.py`, `backends/{megamoe,nvfp4}` ->
   `backends/experts/{mega_moe,nvfp4}.py`, and `backends/{generic/gdn,flashinfer/gdn,
   fla/gdn,torch/gdn,kda}` -> `backends/delta_net/*`. Apply the class renames. Update
   `selector.create_linear/experts/gdn/kda`. Keep temporary re-export shims.
7. **Collapse factories and delete hardware-tier packages.** Route
   `create_backend()` through `PolicyBackendFactory`; delete `generic/`, `hopper/`,
   `blackwell/`; keep `TIER_POLICIES` keys.
8. **Remove shims.** Delete the temporary re-export modules and old import paths after
   all consumers and focused tests use the new locations.

Each stage preserves existing focused tests and adds a reference-vs-backend token or
tensor comparison where an operation is numerically sensitive. The GLM-5.3-Flash
alignment run (`examples/glm5_next_alignment.py`) is the end-to-end regression gate,
since it exercises DSA sparse MLA, KDA, and the FP8 fallbacks together.

## Compatibility and review boundaries

No behavior change is intended in any stage; classes are relocated and renamed, and
selection results per capability tier stay identical. Individual PRs are limited to one
family (or the final factory collapse) so they can be reviewed and reverted
independently. This plan references #325; follow-up implementation PRs reference this
document.

# DSA (sparse attention) backend family

This document specifies promoting **DSA** — DeepSeek/GLM sparse attention (the
NSA "Lightning Indexer" + sparse MLA decode/prefill) — into its own top-level
backend family under `backends/dsa/`, a sibling of `attention/`, `experts/`,
`linear/`, and `delta_net/`. It is part of the workstream in
[Issue #339](https://github.com/JimyMa/NanoDeploy/issues/339) and follows
`docs/backend-family-layout-refactor.md`.

It is a design and migration plan; it does not change runtime behavior by itself.

## Why DSA is its own family, not a file under `attention/`

DSA is not a single attention kernel. It is a **compose of an indexer plus a
sparse attention kernel**, and each half has several independent implementation
strategies that are selected by capability, shape, and config:

- **Indexer** (top-k key selection): FP8 paged MQA-logits (`deep_gemm`), a
  reference BF16/FP32 eager path, a **pooled** GLM-5.3 variant (`index_kpool`),
  and fused vs non-fused top-k transforms.
- **Sparse attention** (attend the selected keys): FlashMLA sparse (Hopper),
  TRTLLM-GEN sparse MLA (Blackwell), and a reference fallback.
- **Top-k → physical translation**: fused (inside the top-k transform) or
  standalone (`topk_indices_to_physical`).
- **State sharing**: single-layer, cross-layer "shared" reuse, and MTP-iteration
  reuse (`_IndexerTopKState`).

Today all of this is orchestrated inline inside `DeepseekV2Attention.forward`
(~560 lines across the prefill and decode branches in
`models/deepseek_v2/deepseek_v2.py`), with the indexer in `layers/indexer.py`,
the sparse consume split across `hopper/attention.py` and `blackwell/attention.py`,
and shared helpers (`topk_indices_to_physical`, `_gather_cache_cached_only`,
`_interleave_cached_fresh`) leaking out of `hopper/attention.py` into models,
`mtp_runner`, and tests. There is no single owner and no policy-driven selection —
exactly the coupling this refactor removes for the other families.

Because DSA has this indexer × sparse-kernel matrix of strategies, it warrants a
family directory of its own rather than one `attention/dsa.py` file.

## Terminology

- **DSA** — the overall DeepSeek/GLM sparse-attention operation (indexer + sparse
  MLA). "NSA" (Native Sparse Attention) is the model-side name; "DSA" is the
  backend-family name here.
- **Indexer** — the Lightning-Indexer MLP that scores keys and returns top-k
  logical indices per query.
- **Sparse MLA** — the MLA attention kernel that attends only the selected keys.
- **Pooled indexer** — GLM-5.3's `index_kpool > 1` variant that scores compressed
  key pools and expands the selected pools back to token indices.

## Target layout

```text
dlengine/runtime/layers/backends/dsa/
    __init__.py
    base.py            # DsaAttentionBase (contract) + shared dataclasses
    indexer/
        __init__.py
        cache.py       # IndexerCache               <- layers/indexer.py
        lightning.py   # Indexer (paged FP8 + eager reference)  <- layers/indexer.py
        pooled.py      # pooled (index_kpool) scoring + tail     <- layers/indexer.py
        topk.py        # fused/non-fused topk + topk_indices_to_physical
    sparse/
        __init__.py
        flash_mla.py   # FlashMLA sparse prefill + decode    <- backends/mla/flash_mla.py
        trtllm.py      # TRTLLM-GEN sparse MLA decode         <- backends/mla/trtllm.py
        reference.py   # reference/naive sparse attention     <- backends/mla/reference.py
    state.py           # _IndexerTopKState (+ PP/MTP transfer) <- deepseek_v2.py
    dsa_attention.py   # DsaAttention: composes indexer + sparse + state
```

DSA sparse attention is **dense MLA plus an index mask**: the `sparse/` kernels are
the `backends/mla/` kernels invoked with the indexer's selected keys. They build on
the `mla/` family (added in the MLA stage) rather than re-deriving from the
hardware-tier packages, so the MLA family lands before DSA.

`DsaAttention` is the family entry point. It holds an indexer implementation, a
sparse-kernel implementation, and the top-k state, and exposes the same
`forward(q, k, v, ...)`-style contract the model calls today, so the ~560 lines
of sparse orchestration move out of `DeepseekV2Attention.forward` into the
backend.

The **indexer stays a first-class sub-component** (it is also consumed directly by
`mtp_runner`, `graph_runner`, and the MTP model for state seeding/sharing), so
`backends/dsa/indexer/` re-exports the public names (`Indexer`, `IndexerCache`,
`pool_indexer_topk`, `_expand_decode_context_lens`, `topk_indices_to_physical`,
`_uses_linear_mtp_indexer_path`) that those consumers import.

## Implementation strategies (the selection matrix)

`backends/selector.py` gains `resolve_dsa_plan(...)` and `create_dsa(...)`. The
plan chooses an indexer strategy and a sparse-kernel strategy from capability +
config; DSA is selected when `attention_type == "MLA" and nsa_index_topk > 0`.

| axis | strategies | selection condition |
| --- | --- | --- |
| indexer scoring | `paged_fp8` (deep_gemm) / `reference` (eager BF16) | FP8 KV cache + kernels present → paged_fp8; else reference (gated by `ref_fallback_allowed`) |
| indexer variant | `plain` / `pooled` | `index_kpool == 1` → plain; `> 1` → pooled (GLM-5.3) |
| topk transform | `fused` / `torch` | `fused_kernels_enabled() and index_topk in {512,2048} and ctx ≤ 16384` → fused; else torch.topk + `topk_indices_to_physical` |
| sparse kernel | `flash_mla` / `trtllm` / `reference` | Hopper → flash_mla; Blackwell → trtllm; unsupported shape + `ref_fallback_allowed` → reference |
| indexer mode | `full` / `shared` / `none` | `_get_indexer_mode(config, layer_idx)` (unchanged logic, moved into the family) |
| state reuse | single / cross-layer shared / MTP-iteration | `_IndexerTopKState`, `index_share_for_mtp_iteration` |

These match the current runtime decisions exactly; the refactor only moves them
behind `resolve_dsa_plan` instead of scattering them across model code and env
gates. The existing env gates (`DLENGINE_DSV4_DEBUG_NSA_SPARSE_PREFILL`,
`DLENGINE_FORCE_MLA_REFERENCE`) are preserved during migration.

## Selection and the factory contract

`base_backend.BackendFactory` gains:

```python
def get_dsa_attention(self, layer_idx, config, *, nsa_index_topk, **kwargs) -> DsaAttentionBase: ...
```

`PolicyBackendFactory.get_dsa_attention` calls `create_dsa(...)`, which builds the
indexer + sparse + state composition from `resolve_dsa_plan`. `DeepseekV2Attention`
(and the DSv4 / GLM-5.3 paths) then request DSA through the factory instead of
constructing `Indexer` and branching on `is_v32`/`indexer_mode`/`sparse_indices`
inline. Dense MLA (no indexer) continues to use the `attention/` family.

## Shared helpers

`topk_indices_to_physical` moves into `backends/dsa/indexer/topk.py`. The
paged-cache gather/scatter helpers it shares with dense MLA
(`_gather_cache_cached_only`, `_interleave_cached_fresh`, `_gather_kv_cached_concat`,
`_compute_cached_split`) move to `backends/attention/mla_utils.py` (per the parent
plan) and are imported by both the `attention/` and `dsa/` families, so there is a
single definition instead of the current hopper-vs-generic duplication.

## Migration plan (independent, reviewable PRs)

This supersedes stage 3 of `docs/backend-family-layout-refactor.md` (which framed
DSA as a single `attention/dsa.py`); DSA becomes its own family. Ordering assumes
the attention-family migration (stage 2 of the parent plan) lands first so
`attention/mla_utils.py` exists.

1. **This doc PR.**
2. **Extract the indexer.** Move `layers/indexer.py` into `backends/dsa/indexer/`
   (`cache.py`, `lightning.py`, `pooled.py`, `topk.py`), keep a thin
   `layers/indexer.py` re-export shim so `mtp_runner`/`graph_runner`/model/tests
   keep working. No behavior change.
3. **Extract sparse kernels.** Move FlashMLA sparse (from `hopper/attention.py`)
   and TRTLLM-GEN sparse MLA (from `blackwell/attention.py`) into
   `backends/dsa/sparse/{flash_mla,trtllm}.py`, plus the reference sparse path into
   `sparse/reference.py`. Behavior-preserving.
4. **Extract state + orchestration.** Move `_IndexerTopKState` to
   `backends/dsa/state.py`; move the sparse prefill/decode orchestration out of
   `DeepseekV2Attention.forward` into `DsaAttention` (`dsa_attention.py`).
5. **Wire the factory + selector.** Add `get_dsa_attention` / `resolve_dsa_plan` /
   `create_dsa`; switch `DeepseekV2Attention`, DSv4, and GLM-5.3 to request DSA
   through the factory. Fold `enable_mla_reference_fallback` into
   `ref_fallback_allowed` for the DSA reference strategy.
6. **Remove shims.** Delete the `layers/indexer.py` re-export and old import paths
   after all consumers and tests use `backends/dsa/`.

Each stage preserves existing focused tests
(`test_indexer*.py`, `test_shared_indexer.py`, `test_sparse_decode.py`,
`test_glm5_next_pool_indexer.py`, `test_mtp_linear.py`) and keeps the GLM-5.3-Flash
and DeepSeek-V3.2 alignment runs as the end-to-end gate (DSA sparse MLA + indexer,
plus for GLM-5.3 the KDA and FP8 fallbacks, exercised together).

## Compatibility and review boundaries

No behavior change is intended in any stage. The indexer's public API is preserved
via the `layers/indexer.py` shim until stage 6, so MTP and graph-capture consumers
are unaffected during migration. Each PR is limited to one extraction step so it
can be reviewed and reverted independently. This plan references #339 and #325.

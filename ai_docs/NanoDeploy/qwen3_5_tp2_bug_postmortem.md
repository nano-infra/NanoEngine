# Qwen3.5 MoE FFN TP=2 Divergence Bug Postmortem

## Overview

When running the Qwen3.5 MoE model with Tensor Parallelism (`TP=2`, `DP=4`), the model was observed to output incorrect logits compared to the baseline (`TP=1`, `DP=8`). The divergence uniquely appeared only in TP configurations greater than 1, while TP=1 worked perfectly.

## Root Cause Analysis

The bug was traced to the interaction between two systems:

1. **WideEP `AttnToFfnTransition` (Batch Scattering)**
2. **`GenericBackendFactory` kwargs dropping (Parallel Context mismatch)**

### 1. `AttnToFfnTransition` Scattering

In the `Qwen3.5` architecture config, we employ an asymmetric Parallelism config: Attention uses `attn_tp=2`, while the MoE FFN uses Expert Parallelism `ffn_ep=8` and `ffn_tp=1`.
When transforming from `attn` to `ffn`, the `AttnToFfnTransition` layer takes the input batch and chunks it across the TP dimension so that each GPU processes a different disjoint subset of the batch for FFN routing.

- Example: With Batch Size 1 and `TP=2`, the layer pads the batch to 2, and then scatters it.
  - Rank 0 receives the real token.
  - Rank 1 receives the padded ALL-ZERO token.

### 2. The Context Mismatch

The FFN layer (`Qwen3_5MoeSparseMoeBlock`) consists of two parts: the `routed_experts` and a `shared_expert`.
The `shared_expert` is a normal MLP instantiated via `Qwen3_5MoeMLP`, which uses our unified `GenericBackendFactory` (and `HopperBackendFactory`) to spawn its native parallel linear layers.
When `Qwen3_5MoeMLP` initializes its inner linear layers, it historically relied on a string tag `parallel_context="ffn"` to select the communication group.

**The Bug:** The underlying Python factories (`get_column_parallel_linear`, `get_row_parallel_linear`, etc.) failed to propagate `**kwargs` into the linear class constructors. It completely dropped the configuration.
Because the context was dropped, the linear layers inside the `shared_expert` defaulted to `attn`, falling back to using the `attn_tp_group` (size 2).

### The Fatal Collision

Since `shared_expert.down_proj` believed it was running in `attn_tp=2`, it automatically executed an `AllReduce` across Rank 0 and Rank 1 at the end of its forward pass.
However, because of the `AttnToFfnTransition`, Rank 0 was holding the processed features for the real token, and Rank 1 was holding the processed features for a padded zero-token.
The illicit `AllReduce` forcibly summed the real token's output with the padded token's output across GPUs. Consequently, both devices received equally corrupted (summed) activations, fully diverging the mathematically correct outputs.

## Resolution

1. **Refactor Parallelism Group Passing (`tp_group`):** We replaced the ambiguous string-based `parallel_context` tag with a direct `tp_group` parameter of type `dist.ProcessGroup`. `Qwen3_5MoeMLP` now explicitly passes `tp_group=get_dist_context().ffn_tp_group` to the `GenericBackendFactory` and `HopperBackendFactory`. The factories and the underlying linear layers (`RowParallelLinear`, `ColumnParallelLinear`, etc.) were updated to explicitly accept and utilize this `tp_group` parameter over `**kwargs`.
2. **Deterministic Fallback (Defense in Depth):** Added `torch.manual_seed(0)` during `_complete_dist_init` in `ModelRunner`. This ensures that any randomly initialized variables not loaded from `safetensors` behave identically across all TP ranks regardless of the degree of parallelism chunking.

With the `tp_group` properly passed and preserved, the `shared_expert` correctly executes as completely replicated (since `ffn_tp=1`) and refrains from cross-GPU `AllReduce`, isolating the real tokens from padding tokens.

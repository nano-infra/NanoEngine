# Design Proposal: Phase-Level Backend Orchestration

**Status**: Proposal
**Date**: 2026-03-18
**Context**: Ascend comm/compute overlap work exposed that op-level backend dispatch cannot express cross-op optimizations (DP-slice before allreduce, async overlap windows, shared expert pipelining). This doc proposes moving backend awareness from op-level to phase/device-level.

---

## Problem

The current architecture dispatches at the **op level**:

```
Model.forward()
  → get_backend().get_experts()        → AscendExperts / HopperExperts
  → get_backend().get_attention()      → AscendAttention / HopperAttention
  → get_backend().get_row_parallel()   → AscendRowParallel / HopperRowParallel
```

Each op is a black box. The model code sequences them but cannot express:

1. **Cross-op data dependencies**: DP-slice output of experts BEFORE their allreduce to halve volume — requires the MoE block (model code) to know about DP slicing, which is Ascend-specific.

2. **Overlap windows**: Launch async allreduce after experts, overlap with shared expert compute, then wait — requires the MoE block to orchestrate timing between routed and shared expert paths.

3. **Phase-specific strategies**: Prefill vs decode may need different op orderings (e.g., DP-slice only during decode when batch sizes are uniform).

When we tried to express these in the current architecture, backend-specific guards leaked into shared model code:
```python
# qwen3_moe.py — polluted with Ascend-specific shape detection
if final_hidden_states.shape[0] != num_tokens:  # Ascend DP-sliced?
    return final_hidden_states
```

---

## Proposed Architecture

### Layer 1: Device Strategy (top level)

A **DeviceStrategy** owns the full decode/prefill phase and decides the orchestration pattern. Selected once at init time based on `backend_type`.

```python
# nanodeploy/strategies/base.py
class DeviceStrategy(ABC):
    @abstractmethod
    def build_decoder_layer(self, config, layer_idx) -> nn.Module:
        """Return a device-optimized decoder layer."""

    @abstractmethod
    def build_moe_block(self, config) -> nn.Module:
        """Return a device-optimized MoE block."""
```

```python
# nanodeploy/strategies/ascend.py
class AscendStrategy(DeviceStrategy):
    def build_moe_block(self, config):
        return AscendMoeBlock(config)  # owns overlap logic

# nanodeploy/strategies/hopper.py
class HopperStrategy(DeviceStrategy):
    def build_moe_block(self, config):
        return HopperMoeBlock(config)  # uses fused DeepGEMM path
```

### Layer 2: Device-Specific MoE Block

Each backend provides its own MoE block that knows the optimal execution order:

```python
# Ascend: async allreduce with overlap window
class AscendMoeBlock(nn.Module):
    def forward(self, hidden_states, is_prefill):
        # 1. Route
        topk_ids, topk_weights = self.gate(hidden_states)

        # 2. Shared expert (can overlap with routed allreduce)
        shared_out = self.shared_expert(hidden_states)

        # 3. Routed experts: compute → unpermute → dp_slice → async allreduce
        routed_out = self.routed_experts(hidden_states, topk_ids, topk_weights)

        if not is_prefill and self.dp_world_size > 1:
            routed_out = self._dp_slice(routed_out)

        work = dist.all_reduce(routed_out, group=self.tp_group, async_op=True)

        # 4. Shared expert gate (overlaps with HCCL transfer)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_out = shared_out * shared_gate

        if not is_prefill and self.dp_world_size > 1:
            shared_out = self._dp_slice(shared_out)

        # 5. Wait for allreduce, combine
        work.wait()
        return routed_out + shared_out
```

```python
# Hopper: fused matmul+allreduce, no explicit overlap needed
class HopperMoeBlock(nn.Module):
    def forward(self, hidden_states, is_prefill):
        topk_ids, topk_weights = self.gate(hidden_states)
        shared_out = self.shared_expert(hidden_states)  # allreduce fused inside
        routed_out = self.routed_experts(hidden_states, topk_ids, topk_weights)  # allreduce fused inside
        return routed_out + shared_out * torch.sigmoid(self.shared_expert_gate(hidden_states))
```

### Layer 3: Op-Level Backend (unchanged)

Individual ops (attention, linear, GroupedMatmul) remain backend-specific via the factory. The strategy layer orchestrates them; the op layer executes them.

```
DeviceStrategy          → orchestration (what order, what overlaps)
  └─ DeviceMoeBlock     → cross-op coordination (async allreduce, dp_slice)
      └─ BackendFactory → single-op implementation (GroupedMatmul, attention)
```

---

## What This Enables

### 1. DP-Slice Before AllReduce (currently removed to avoid pollution)

The Ascend MoE block directly owns the dp_slice + allreduce sequence. No shape guards in shared code:

```python
# Inside AscendMoeBlock — NOT in shared Qwen3MoeSparseMoeBlock
routed_out = self.routed_experts(hidden_states, ...)  # returns [bs*dp, H]
routed_out = self._dp_slice(routed_out)               # → [bs, H]
work = dist.all_reduce(routed_out, async_op=True)      # half volume
```

### 2. Shared Expert Overlap

Qwen3.5-MoE has a shared expert whose compute can fill the allreduce wait window:

```
Timeline (current):
  [routed experts] → [allreduce WAIT] → [shared expert] → [add]

Timeline (proposed):
  [routed experts] → [async allreduce START]
                      [shared expert compute]  ← overlaps with HCCL
                      [allreduce WAIT]
                      [add]
```

This is only possible when the MoE block orchestrates both paths. With op-level dispatch, each expert is a black box that does its own allreduce internally.

### 3. Backend-Specific Transition Logic

The decoder layer's attn↔ffn transitions can also be device-specific:

```python
class AscendDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        hidden_states = self.attn(positions, hidden_states)
        hidden_states, residual = self.post_norm(hidden_states, residual)

        # Ascend: AllGather for DP, no slice after (MoE handles it)
        hidden_states = self.dp_gather(hidden_states)
        hidden_states = self.moe(hidden_states, is_prefill)
        # MoE already DP-sliced during decode; only slice during prefill
        if is_prefill:
            hidden_states = self.dp_slice(hidden_states)

        return hidden_states, residual
```

No shape-detection heuristics. The strategy explicitly controls the flow.

---

## Migration Path

### Phase 1: Strategy Selection (minimal)
- Add `DeviceStrategy` base class and `get_strategy()` factory
- Strategy wraps existing model classes, delegates to them
- No behavior change — just the wiring

### Phase 2: Move MoE Block (per-model)
- Create `AscendMoeBlock` / `HopperMoeBlock` with device-specific forward
- Move dp_slice + async allreduce + shared expert overlap into AscendMoeBlock
- Remove op-level allreduce from experts (experts return pre-allreduce output)
- Shared model code becomes thinner

### Phase 3: Move Decoder Layer
- Create device-specific decoder layers that own transition logic
- Remove AttnDpToFfnTransition/FfnToAttnDpTransition from shared code
- Ascend decoder layer explicitly orchestrates gather/slice/overlap

### Phase 4: Move Full Model (optional)
- If needed, device-specific model classes (like vllm-ascend's approach)
- Enables fully custom forward pass per device

---

## What We Keep from Current Work

Even without the phase-level design, these optimizations are **already landed** and are purely within the Ascend backend (no shared-code pollution):

| Optimization | Status | Where |
|-------------|--------|-------|
| Unpermute before allreduce | Landed | `experts.py` (Ascend-only) |
| Async allreduce (`async_op=True`) | Landed | `experts.py` (Ascend-only) |
| Fused QKV+RMSNorm+RoPE | Landed | `fused_qkv_norm_rope.py` (Ascend-only) |
| FusedInferAttentionScore | Landed | `attention.py` (Ascend-only) |
| all_gather_into_tensor | Landed | `parallelism_transition.py` (all backends) |

**Deferred to phase-level design**:

| Optimization | Est. gain | Requires |
|-------------|----------|----------|
| DP-slice before allreduce | ~1-2ms/step | AscendMoeBlock |
| Shared expert ↔ allreduce overlap | ~2-3ms/step | AscendMoeBlock |
| Phase-specific transition logic | ~0.5ms/step | AscendDecoderLayer |

---

## Summary

```
Current:   Model → [shared MoE block] → get_backend().get_experts() → device op
Proposed:  Model → get_strategy().build_moe_block() → [device MoE block] → device op
                                                       ↑ owns overlap, dp_slice, transitions
```

The key insight: **overlap is an orchestration concern, not an op concern.** It belongs in the device strategy layer, not in shape guards scattered across shared model code.

# Checkpoint Format Analysis

Analysis of weight key patterns for all supported models, verified against actual checkpoint files.

---

## 1. Qwen3.5-397B-A17B-FP8

**Path**: `/models/models--Qwen--Qwen3.5-397B-A17B-FP8`
**Architecture**: `Qwen3_5MoeForConditionalGeneration`
**Config Type**: `qwen3_5_moe` (VLM with `text_config`)
**Quantization**: FP8 (`float8_e4m3fn`)
**Expert Count**: 512

### Weight Key Patterns

```
# VLM prefix — stripped by _strip_vlm_prefix → "model.layers.N..."
model.language_model.layers.N.*

# Full Attention (every 4th layer)
model.language_model.layers.N.self_attn.q_proj.weight                    # FP8
model.language_model.layers.N.self_attn.q_proj.weight_scale_inv          # float32
model.language_model.layers.N.self_attn.k_proj.weight                    # FP8
model.language_model.layers.N.self_attn.k_proj.weight_scale_inv
model.language_model.layers.N.self_attn.v_proj.weight                    # FP8
model.language_model.layers.N.self_attn.v_proj.weight_scale_inv
model.language_model.layers.N.self_attn.o_proj.weight                    # FP8
model.language_model.layers.N.self_attn.o_proj.weight_scale_inv
model.language_model.layers.N.self_attn.q_norm.weight                    # float32
model.language_model.layers.N.self_attn.k_norm.weight                    # float32

# Linear Attention (GatedDeltaNet, other layers)
model.language_model.layers.N.linear_attn.in_proj_qkv.weight            # FP8
model.language_model.layers.N.linear_attn.in_proj_qkv.weight_scale_inv
model.language_model.layers.N.linear_attn.in_proj_z.weight               # FP8
model.language_model.layers.N.linear_attn.in_proj_z.weight_scale_inv
model.language_model.layers.N.linear_attn.out_proj.weight                # FP8
model.language_model.layers.N.linear_attn.out_proj.weight_scale_inv
model.language_model.layers.N.linear_attn.in_proj_a.weight               # BF16
model.language_model.layers.N.linear_attn.in_proj_b.weight               # BF16
model.language_model.layers.N.linear_attn.conv1d.weight                  # BF16
model.language_model.layers.N.linear_attn.A_log                          # BF16
model.language_model.layers.N.linear_attn.dt_bias                        # BF16
model.language_model.layers.N.linear_attn.norm.weight                    # float32

# Routed Experts — PER-EXPERT format (NOT packed 3D)
model.language_model.layers.N.mlp.experts.E.gate_proj.weight             # FP8
model.language_model.layers.N.mlp.experts.E.gate_proj.weight_scale_inv   # float32
model.language_model.layers.N.mlp.experts.E.up_proj.weight               # FP8
model.language_model.layers.N.mlp.experts.E.up_proj.weight_scale_inv
model.language_model.layers.N.mlp.experts.E.down_proj.weight             # FP8
model.language_model.layers.N.mlp.experts.E.down_proj.weight_scale_inv

# Router
model.language_model.layers.N.mlp.gate.weight                           # BF16

# Shared Expert (singular "shared_expert", not "shared_experts")
model.language_model.layers.N.mlp.shared_expert.gate_proj.weight         # FP8
model.language_model.layers.N.mlp.shared_expert.gate_proj.weight_scale_inv
model.language_model.layers.N.mlp.shared_expert.up_proj.weight           # FP8
model.language_model.layers.N.mlp.shared_expert.up_proj.weight_scale_inv
model.language_model.layers.N.mlp.shared_expert.down_proj.weight         # FP8
model.language_model.layers.N.mlp.shared_expert.down_proj.weight_scale_inv

# Shared Expert Gate (sigmoid)
model.language_model.layers.N.mlp.shared_expert_gate.weight              # BF16

# Layer Norms
model.language_model.layers.N.input_layernorm.weight
model.language_model.layers.N.post_attention_layernorm.weight

# Embeddings & Head
model.language_model.embed_tokens.weight
model.language_model.norm.weight
lm_head.weight

# MTP layers (skipped)
mtp.layers.N.*
mtp.fc.weight
mtp.norm.weight
mtp.pre_fc_norm_*.weight

# Visual encoder (skipped)
model.visual.*
```

### Loader Mapping

| Weight Pattern | Handler | Target Parameter |
|---|---|---|
| `experts.E.gate_proj.weight[_scale_inv]` | `EXPERT_RE` → `load_per_expert_weight` | `routed_experts.gate_up_proj` (first half) |
| `experts.E.up_proj.weight[_scale_inv]` | `EXPERT_RE` → `load_per_expert_weight` | `routed_experts.gate_up_proj` (second half) |
| `experts.E.down_proj.weight[_scale_inv]` | `EXPERT_RE` → `load_per_expert_weight` | `routed_experts.down_proj` |
| `self_attn.q/k/v_proj.*` | `_PACKED_MODULES_MAPPING` | `self_attn.qkv_proj.*` |
| `shared_expert.gate/up_proj.*` | `_PACKED_MODULES_MAPPING` | `shared_expert.gate_up_proj.*` |
| `shared_expert.down_proj.*` | Default loader | `shared_expert.down_proj.*` |
| `shared_expert_gate.*` | Default loader | `shared_expert_gate.*` |
| `linear_attn.*` | Default loader | `linear_attn.*` |
| `mtp.*`, `visual.*` | Skipped | N/A |

---

## 2. DeepSeek-V3

**Path**: `/models/deepseek-v3`
**Architecture**: `DeepseekV3ForCausalLM`
**Config Type**: `deepseek_v3`
**Quantization**: FP8
**Expert Count**: 256 (`n_routed_experts`)
**Hidden Layers**: 61

### Weight Key Patterns

```
# Standard prefix
model.layers.N.*

# MLA Attention
model.layers.N.self_attn.q_a_proj.weight                    # FP8
model.layers.N.self_attn.q_a_proj.weight_scale_inv
model.layers.N.self_attn.q_b_proj.weight                    # FP8
model.layers.N.self_attn.q_b_proj.weight_scale_inv
model.layers.N.self_attn.q_a_layernorm.weight
model.layers.N.self_attn.kv_a_proj_with_mqa.weight          # FP8
model.layers.N.self_attn.kv_a_proj_with_mqa.weight_scale_inv
model.layers.N.self_attn.kv_b_proj.weight                   # FP8 → decomposed to kc + vc
model.layers.N.self_attn.kv_b_proj.weight_scale_inv
model.layers.N.self_attn.kv_a_layernorm.weight
model.layers.N.self_attn.o_proj.weight                      # FP8
model.layers.N.self_attn.o_proj.weight_scale_inv

# Routed Experts — PER-EXPERT format
model.layers.N.mlp.experts.E.gate_proj.weight                # FP8
model.layers.N.mlp.experts.E.gate_proj.weight_scale_inv
model.layers.N.mlp.experts.E.up_proj.weight                  # FP8
model.layers.N.mlp.experts.E.up_proj.weight_scale_inv
model.layers.N.mlp.experts.E.down_proj.weight                # FP8
model.layers.N.mlp.experts.E.down_proj.weight_scale_inv

# Router (with correction bias)
model.layers.N.mlp.gate.weight
model.layers.N.mlp.gate.e_score_correction_bias

# Shared Experts (plural "shared_experts")
model.layers.N.mlp.shared_experts.gate_proj.weight           # FP8
model.layers.N.mlp.shared_experts.gate_proj.weight_scale_inv
model.layers.N.mlp.shared_experts.up_proj.weight             # FP8
model.layers.N.mlp.shared_experts.up_proj.weight_scale_inv
model.layers.N.mlp.shared_experts.down_proj.weight           # FP8
model.layers.N.mlp.shared_experts.down_proj.weight_scale_inv

# Dense MLP layers (first few layers, non-MoE)
model.layers.N.mlp.gate_proj.weight                          # FP8
model.layers.N.mlp.gate_proj.weight_scale_inv
model.layers.N.mlp.up_proj.weight                            # FP8
model.layers.N.mlp.up_proj.weight_scale_inv
model.layers.N.mlp.down_proj.weight                          # FP8
model.layers.N.mlp.down_proj.weight_scale_inv

# MTP layers (skipped)
model.layers.N.eh_proj.weight
model.layers.N.embed_tokens.weight
model.layers.N.enorm.weight
model.layers.N.hnorm.weight
model.layers.N.shared_head.head.weight
model.layers.N.shared_head.norm.weight

# Layer Norms
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight

# Embeddings & Head
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

### Loader Mapping

| Weight Pattern | Handler | Target Parameter |
|---|---|---|
| `experts.E.gate/up/down_proj.*` | `EXPERT_RE` → `load_per_expert_weight` | `routed_experts.gate_up_proj` / `down_proj` |
| `kv_b_proj.weight` | `_handle_kv_b_proj` (deferred) | `kc.weight` + `vc.weight` |
| `kv_b_proj.weight_scale_inv` | Buffered for `_handle_kv_b_proj` | (consumed during dequant) |
| `shared_experts.gate/up_proj.*` | `_PACKED_MODULES_MAPPING` (`gate_proj` key) | `shared_experts.gate_up_proj.*` |
| `mlp.gate/up_proj.*` (dense) | `_PACKED_MODULES_MAPPING` | `mlp.gate_up_proj.*` |
| `gate.e_score_correction_bias` | Default loader | Direct copy |

---

## 3. Qwen3-235B-A22B (Qwen3 MoE)

**Path**: `/models/Qwen3-235B-A22B-Instruct-2507`
**Architecture**: `Qwen3MoeForCausalLM`
**Config Type**: `qwen3_moe`
**Quantization**: None (BF16)
**Expert Count**: 128
**Hidden Layers**: 94

### Weight Key Patterns

```
# GQA Attention
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight
model.layers.N.self_attn.k_norm.weight

# Routed Experts — PER-EXPERT format (BF16, no scales)
model.layers.N.mlp.experts.E.gate_proj.weight
model.layers.N.mlp.experts.E.up_proj.weight
model.layers.N.mlp.experts.E.down_proj.weight

# Router
model.layers.N.mlp.gate.weight

# Layer Norms
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight

# Embeddings & Head
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

### Loader Mapping

| Weight Pattern | Handler | Target Parameter |
|---|---|---|
| `experts.E.gate/up/down_proj.weight` | `EXPERT_RE` → `load_per_expert_weight` | `routed_experts.gate_up_proj` / `down_proj` |
| `q/k/v_proj.weight` | `_PACKED_MODULES_MAPPING` | `qkv_proj.weight` |
| All others | Default loader | Direct copy |

---

## 4. Qwen3-8B (Dense)

**Path**: `/models/qwen3-8b-deepseek-r1`
**Architecture**: `Qwen3ForCausalLM`
**Config Type**: `qwen3`
**Quantization**: None (BF16)
**Hidden Layers**: 36

### Weight Key Patterns

```
# GQA Attention
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.self_attn.q_norm.weight
model.layers.N.self_attn.k_norm.weight

# Dense MLP (no MoE)
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight

# Layer Norms
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight

# Embeddings & Head
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

### Loader Mapping

| Weight Pattern | Handler | Target Parameter |
|---|---|---|
| `q/k/v_proj.weight` | `_PACKED_MODULES_MAPPING` | `qkv_proj.weight` |
| `gate/up_proj.weight` | `_PACKED_MODULES_MAPPING` | `gate_up_proj.weight` |
| All others | Default loader | Direct copy |

---

## Key Differences Between Models

| Feature | DeepSeek V3 | Qwen3 | Qwen3 MoE | Qwen3.5 MoE |
|---------|------------|-------|-----------|-------------|
| Attention | MLA (`kv_b_proj`) | GQA | GQA | Full + GatedDeltaNet |
| Expert Format | Per-expert | N/A | Per-expert | Per-expert |
| Shared Expert | `shared_experts` (plural) | N/A | None | `shared_expert` (singular) |
| Shared Expert Gate | None | N/A | None | `shared_expert_gate` (sigmoid) |
| Quantization | FP8 | BF16 | BF16 | FP8 |
| VLM Prefix | No | No | No | Yes (`model.language_model.`) |
| Config Expert Key | `n_routed_experts` | N/A | `num_experts` | `num_experts` |
| MTP Layers | Yes | No | No | Yes |
| Router Bias | `e_score_correction_bias` | No | No | No |

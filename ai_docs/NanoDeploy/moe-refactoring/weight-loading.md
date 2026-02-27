# Per-Model Weight Loading Architecture

## Design

The weight loading system is split into two layers:

1. **`nanodeploy/worker/loader.py`** — Shared infrastructure (iteration, regex, utilities)
2. **`nanodeploy/models/<model>/<model>_loader.py`** — Per-model loading logic

### Flow

```
loader.py:load_model(model, path)
  │
  ├─ model.load_weights(weights_generator)  ← if model defines it
  │    │
  │    └─ <model>_loader.py:load_weights(model, weights)
  │         ├─ Expert weights → load_per_expert_weight() / load_packed_expert_weight()
  │         ├─ Packed modules → _PACKED_MODULES_MAPPING dispatch
  │         └─ Default → param.weight_loader(param, tensor)
  │
  └─ _load_model_generic(model, weights)    ← fallback if no load_weights()
```

### Weight Name Transform Pipeline

```
Raw checkpoint key
  e.g. "model.language_model.layers.0.mlp.experts.5.gate_proj.weight"
       │
       ▼
_strip_vlm_prefix()   — removes "language_model." for VLM models
  → "model.layers.0.mlp.experts.5.gate_proj.weight"
       │
       ▼
_should_skip_weight()  — skips visual, mtp, non-language weights
       │
       ▼
iterate_weights()      — yields (weight_name, raw_weight_name, tensor)
       │
       ▼
Per-model loader handles matching and loading
```

---

## Shared Utilities (loader.py)

### Regex Patterns

```python
# Per-expert format: experts.{index}.{proj}.weight[_scale_inv]
EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(\d+)\.(\w+)\.(weight(?:_scale_inv)?)"
)
# Captures: (mlp_prefix, expert_idx, proj_name, suffix)

# Packed 3D format: experts.{gate_up_proj|down_proj}
PACKED_EXPERT_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$"
)

# Packed 3D scale format: experts.{gate_up_proj|down_proj}_scale_inv
PACKED_EXPERT_SCALE_RE = re.compile(
    r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)_scale_inv$"
)
```

### Helper Functions

#### `load_per_expert_weight(model, weight_name, tensor, config) -> bool`

Loads a **single expert's** weight into the packed 3D `DistributedRoutedExperts` parameter.

- Determines expert's EP rank → skips if not assigned to this rank
- Determines TP slice → slices the intermediate dimension
- For `gate_proj` / `up_proj`: writes into `routed_experts.gate_up_proj[local_idx]`
  - `gate_proj` → first half of dim 0 (`[:I]`)
  - `up_proj` → second half of dim 0 (`[I:]`)
- For `down_proj`: writes into `routed_experts.down_proj[local_idx]`
- Handles `weight_scale_inv` tensors: initializes combined scale param on first encounter, then copies slices

#### `load_packed_expert_weight(model, weight_name, tensor) -> bool`

Loads an **already-packed 3D** expert tensor (all experts stacked).

- EP slicing: `tensor[expert_start:expert_end]`
- TP slicing: For `gate_up_proj`, splits gate/up halves and slices each by TP rank. For `down_proj`, slices dim 2.

#### `load_packed_expert_scale(model, weight_name, tensor) -> bool`

Loads a **packed 3D** expert FP8 scale tensor.

- Same EP/TP slicing logic as `load_packed_expert_weight`
- Ensures `float32` dtype
- Handles device placement: `param.data = tensor.to(device=param.data.device)`

#### `default_weight_loader(param, tensor, *args, **kwargs)`

Simple `param.data.copy_(tensor)` fallback.

---

## Per-Model Loaders

### DeepSeek V2/V3 (`deepseek_v2_loader.py`)

```python
_PACKED_MODULES_MAPPING = {
    "gate_proj": ("gate_up_proj", 0),    # shared expert + dense MLP
    "up_proj":   ("gate_up_proj", 1),
}
```

**Special handling:**
- `kv_b_proj` decomposition → `kc.weight` + `vc.weight` (FP8 dequant if needed)
- Buffers scale tensors for deferred `kv_b_proj` processing
- Expert weights: `EXPERT_RE` → `load_per_expert_weight`

**Config fields used:** `n_routed_experts`, `num_attention_heads`, `qk_nope_head_dim`, `v_head_dim`, `kv_lora_rank`

### Qwen3 (`qwen3_loader.py`)

```python
_PACKED_MODULES_MAPPING = {
    "q_proj":    ("qkv_proj", "q"),
    "k_proj":    ("qkv_proj", "k"),
    "v_proj":    ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj":   ("gate_up_proj", 1),
}
```

Dense model, no expert handling needed. Simplest loader.

### Qwen3 MoE (`qwen3_moe_loader.py`)

```python
_PACKED_MODULES_MAPPING = {
    "q_proj":         ("qkv_proj", "q"),
    "k_proj":         ("qkv_proj", "k"),
    "v_proj":         ("qkv_proj", "v"),
    "gate_proj":      ("gate_up_proj", 0),
    "up_proj":        ("gate_up_proj", 1),
    "gate_scale_inv": ("gate_up_scale_inv", 0),
    "up_scale_inv":   ("gate_up_scale_inv", 1),
}
```

**Expert handling:** `EXPERT_RE` → `load_per_expert_weight`

**Config fields used:** `num_experts`

### Qwen3.5 MoE (`qwen3_5_moe_loader.py`)

```python
_PACKED_MODULES_MAPPING = {
    "self_attn.q_proj":       ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj":       ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj":       ("self_attn.qkv_proj", "v"),
    "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
    "shared_expert.up_proj":   ("shared_expert.gate_up_proj", 1),
}
```

**Expert handling** (supports two checkpoint formats):
1. `EXPERT_RE` → `load_per_expert_weight` (primary — Qwen3.5-397B-A17B-FP8 uses this)
2. `PACKED_EXPERT_RE` → `load_packed_expert_weight` (fallback — alternate checkpoints)
3. `PACKED_EXPERT_SCALE_RE` → `load_packed_expert_scale` (fallback — alternate checkpoints)

**Config fields used:** `num_experts`

**Notes:**
- Uses more specific mapping keys (`self_attn.q_proj` instead of `q_proj`) to avoid false matches with `linear_attn` weights
- VLM prefix (`model.language_model.`) stripped by `_strip_vlm_prefix` in `loader.py`
- `linear_attn.*` weights (GatedDeltaNet) load via default path

---

## FP8 Scale Handling in Linear Layers

The `LinearBase` class registers `weight_scale_inv` as an `nn.Parameter` with the same `weight_loader` as the module. The weight_loader detects scale tensors by checking `"inv" in weight_name`.

| Linear Class | Scale Handling |
|---|---|
| `QKVParallelLinear` | Adjusts shard offset/size by `block_size[0]` for scales |
| `MergedColumnParallelLinear` | Same block_size adjustment for output_sizes |
| `ColumnParallelLinear` | Standard TP slice on dim 0 |
| `RowParallelLinear` | Standard TP slice on dim 1 |

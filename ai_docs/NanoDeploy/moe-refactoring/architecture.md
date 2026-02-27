# MoE Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Layer (e.g. Qwen3_5MoeSparseMoeBlock)                   │
│                                                                 │
│  ┌──────────────┐   ┌───────────────────────────────────┐       │
│  │ Router (gate)│   │ DistributedRoutedExperts           │       │
│  │ nn.Linear    │──▶│  ┌─────────────────────────────┐   │       │
│  └──────────────┘   │  │ gate_up_proj [E, 2I, H]     │   │       │
│                     │  │ down_proj    [E, H, I]       │   │       │
│                     │  │ gate_up_scale_inv (FP8 only) │   │       │
│                     │  │ down_scale_inv    (FP8 only) │   │       │
│                     │  └─────────────────────────────┘   │       │
│                     │                                     │       │
│                     │  Execution paths:                   │       │
│                     │  ├─ _compute_local()    (EP=1)      │       │
│                     │  ├─ _compute_prefill_ep() (Normal)  │       │
│                     │  └─ _compute_decode_ep()  (LowLat)  │       │
│                     └───────────────────────────────────┘       │
│                                                                 │
│  ┌────────────────────┐                                         │
│  │ shared_expert (MLP)│  (optional, e.g. DeepSeek/Qwen3.5)     │
│  └────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

## DistributedRoutedExperts

**File**: `nanodeploy/layers/distributed_routed_experts.py`

### Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `gate_up_proj` | `[num_local_experts, local_intermediate_size * 2, hidden_size]` | Packed gate + up projection (FP8 or BF16) |
| `down_proj` | `[num_local_experts, hidden_size, local_intermediate_size]` | Down projection (FP8 or BF16) |
| `gate_up_scale_inv` | varies (FP8 only) | Per-block scale inverse for gate_up |
| `down_scale_inv` | varies (FP8 only) | Per-block scale inverse for down |

Where:
- `num_local_experts = num_experts // ep_size`
- `local_intermediate_size = intermediate_size // tp_size`

### Constructor Signature

```python
DistributedRoutedExperts(
    hidden_size: int,           # e.g. 4096 (Qwen3.5), 7168 (DSv3)
    intermediate_size: int,     # e.g. 1024 (Qwen3.5), 2048 (DSv3)
    num_experts: int,           # e.g. 512 (Qwen3.5), 256 (DSv3)
    top_k: int,                 # e.g. 10 (Qwen3.5), 8 (DSv3)
    ep_size: int,               # Expert Parallel world size
    tp_size: int,               # Tensor Parallel world size
    ep_group,                   # EP process group
    tp_group,                   # TP process group
    n_group: int = 1,           # Expert grouping for routing
    topk_group: int = 1,        # TopK within each group
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "softmax",
    quantization_config = None,
)
```

### Execution Paths

#### Path 1: `_compute_local()` — EP=1 (Single-rank or Pure TP)

Used when `ep_size == 1`. No DeepEP communication needed.

- **FP8**: Uses `fused_moe_v3()` Triton kernel
- **BF16**: Uses `fused_moe_v3_bf16()` Triton kernel
- If `tp_size > 1`: performs `all_reduce` on output

#### Path 2: `_compute_prefill_ep()` — Normal Mode

Used during prefill (long sequences) with EP > 1.

```
Input → [FP8: quant_fp8] → dispatch_layout → dispatch → 
  gate_up GEMM → SiLU+Mul → [FP8: quant] → down GEMM → combine → Output
```

- **FP8**: `per_token_group_quant_fp8` → `Buffer.dispatch(fp8)` → `fused_moe_v3` → `Buffer.combine`
- **BF16**: `Buffer.dispatch(bf16)` → `fused_moe_v3_bf16` → `Buffer.combine`

Uses `DeepEPTokenDispatcherNormal` for token routing.

#### Path 3: `_compute_decode_ep()` — Low Latency Mode

Used during decode (single-token generation) with EP > 1.

```
Input → low_latency_dispatch → 
  gate_up masked GEMM → SiLU+Mul[+PostQuant] → down masked GEMM → 
  low_latency_combine → Output
```

- **FP8**: `Buffer.low_latency_dispatch(use_fp8=True)` → `deep_gemm.m_grouped_fp8_gemm_nt_masked` → `silu_and_mul_masked_post_quant_fwd` → `deep_gemm.m_grouped_fp8_gemm_nt_masked` → `Buffer.low_latency_combine`
- **BF16**: `Buffer.low_latency_dispatch` → `deep_gemm.m_grouped_bf16_gemm_nt_masked` → SiLU+Mul → `deep_gemm.m_grouped_bf16_gemm_nt_masked` → `Buffer.low_latency_combine`

Uses `DeepEPTokenDispatcherLowLatency` for token routing.

---

## ExpertContext

**File**: `nanodeploy/context/expert_context.py`

Singleton managing DeepEP buffer lifecycle.

### Responsibilities

1. **Buffer allocation**: One-time `warmup()` creates `deep_ep.Buffer` sized for both Normal and Low Latency modes
2. **Mode transitions**: `transition_to_normal()` / `transition_to_low_latency()` calls `clean_low_latency_buffer` when switching modes
3. **Configuration**: Centralizes `num_sms`, `num_max_dispatch_tokens_per_rank`, buffer sizes

### Buffer Sizing

```python
# Normal mode (prefill)
num_nvl_bytes = Buffer.get_dispatch_layout_size_hint(...)
num_rdma_bytes = 0  # Normal mode uses NVL only

# Low Latency mode (decode)
num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank,  # default 128
    hidden_size,                       # raw dimension (e.g. 4096)
    num_local_experts,
    num_experts,
)

# Final buffer uses max of both
buffer = deep_ep.Buffer(
    ep_group, num_nvl_bytes, num_rdma_bytes,
    num_qps_per_rank=num_local_experts,
    low_latency_mode=True,
    num_sms=Buffer.num_sms,  # typically 20
)
```

### Key Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `num_max_dispatch_tokens_per_rank` | Default 128 | Max tokens dispatched per rank in low-latency mode |
| `num_sms` | `deep_ep.Buffer.num_sms` (20) | SMs reserved for communication |
| `num_qps_per_rank` | `num_local_experts` | RDMA queue pairs per rank |
| `hidden_size_bytes` | `hidden_size * 2` | BF16 output size for NVL buffer |

---

## TokenDispatcher

**File**: `nanodeploy/layers/token_dispatcher.py`

### DeepEPTokenDispatcherNormal

For prefill mode. Wraps `Buffer.dispatch()` / `Buffer.combine()`.

```python
dispatcher = DeepEPTokenDispatcherNormal(...)
recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens, handle = \
    dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
# ... compute on recv_x ...
out = dispatcher.combine(result)
```

### DeepEPTokenDispatcherLowLatency

For decode mode. Wraps `Buffer.low_latency_dispatch()` / `Buffer.low_latency_combine()`.

```python
dispatcher = DeepEPTokenDispatcherLowLatency(...)
packed_recv_hidden, masked_m, expected_m, handle = \
    dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
# ... compute with masked GEMM using masked_m ...
out = dispatcher.combine(result, handle)
```

Reads `num_max_dispatch_tokens_per_rank` from `ExpertContext.get_instance()`.

---

## Ported Kernels

All previously DLBlas-dependent kernels now live under `nanodeploy/kernels/`:

| Kernel | File | Usage |
|--------|------|-------|
| `per_token_group_quant_fp8` | `kernels/fp8.py` | FP8 token quantization before dispatch |
| `silu_and_mul_masked_post_quant_fwd` | `kernels/fp8.py` | Fused SiLU+Mul+FP8Quant for decode |
| `fused_moe_v3` | `kernels/fused_moe_v3.py` | FP8 MoE for local/prefill path |
| `fused_moe_v3_bf16` | `kernels/fused_moe_v3.py` | BF16 MoE for local/prefill path |
| `ep_scatter_bf16` | `kernels/fused_moe_v3.py` | BF16 scatter for EP combine |
| `tma_align_input_scale` | `kernels/fused_moe_v3.py` | TMA alignment for FP8 scales |
| `silu_and_mul` | `kernels/fused_moe_v3.py` | Standard SiLU+Mul activation |

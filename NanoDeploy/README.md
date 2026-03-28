# NanoDeploy

LLM inference engine with Prefill-Decode disaggregation and Wide Expert Parallelism.

## Key Features

- **Disaggregated Prefill/Decode** — Maximize throughput by separating computation phases across nodes
- **Automated Service Discovery** — Redis-based control plane for dynamic engine registration and heartbeat
- **Load Balancing** — Intelligent routing with multiple strategies (round-robin, least-batch, least-cache)
- **KV Cache Migration** — Automatic prefill→decode migration via RDMA for zero-copy transfer
- **Distributed Execution** — Ray-based worker management across multiple nodes
- **OpenAI-Compatible API** — Standard HTTP `/v1/completions` endpoints for easy integration

## Supported Models

| Model                                                         | Architecture                         | Attention          | FFN      | Quantization | KV Cache Block Size |
| ------------------------------------------------------------- | ------------------------------------ | ------------------ | -------- | ------------ | ------------------- |
| [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)                 | `Qwen3ForCausalLM`                   | GQA                | Dense    | BF16         | 256                 |
| [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-235B-A22B)      | `Qwen3MoeForCausalLM`                | GQA                | MoE (EP) | FP8 (E4M3)   | 256                 |
| [Qwen3.5-MoE](https://huggingface.co/Qwen/Qwen3.5-MoE)        | `Qwen3_5MoeForConditionalGeneration` | GQA + GDN (hybrid) | MoE (EP) | FP8 (E4M3)   | 256                 |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | `DeepseekV3ForCausalLM`              | MLA (FlashMLA)     | MoE (EP) | FP8 (E4M3)   | 64                  |

**Notes:**

- **GQA** (Grouped-Query Attention): Standard multi-head attention with KV sharing across query groups.
- **GDN** (GatedDeltaNet): Linear attention with gated delta updates. Qwen3.5-MoE uses a hybrid architecture — full GQA attention for ~25% of layers (every 4th) and GDN linear attention for the rest. GDN layers maintain recurrent states instead of KV cache.
- **MLA** (Multi-head Latent Attention): Compressed KV projection into low-rank space. Decode uses FlashMLA on compressed cache; prefill uses Flash Attention 3 with explicit Q/K/V expansion.
- **MoE** (Mixture of Experts): Expert parallelism distributes experts across GPUs via `ffn_ep`.
- **FP8 (E4M3)**: Block-wise FP8 quantization with per-block scale factors.

## Parallelism Strategies

NanoDeploy supports independent parallelism configuration for attention and FFN layers:

| Strategy                  | Config Key                | Description                                       | Use Case                                |
| ------------------------- | ------------------------- | ------------------------------------------------- | --------------------------------------- |
| Data Parallelism (DP)     | `attention_dp` / `ffn_dp` | Replicate layers, split batch across GPUs         | Increase throughput                     |
| Sequence Parallelism (SP) | `attention_sp`            | Split long sequences across GPUs during attention | Reduce per-GPU memory for long contexts |
| Tensor Parallelism (TP)   | `attention_tp` / `ffn_tp` | Split weight matrices across GPUs                 | Fit large models on multiple GPUs       |
| Expert Parallelism (EP)   | `ffn_ep`                  | Distribute MoE experts across GPUs                | MoE models (DeepSeek-V3, Qwen3-MoE)     |

**Constraint:** `attention_dp × attention_sp × attention_tp == ffn_dp × ffn_ep × ffn_tp` (total world size must match)

### Example Configurations

| Setup          | GPUs | attention_dp | attention_sp | attention_tp | ffn_dp | ffn_ep | ffn_tp |
| -------------- | ---- | ------------ | ------------ | ------------ | ------ | ------ | ------ |
| Qwen3-8B       | 1    | 1            | 1            | 1            | 1      | 1      | 1      |
| Qwen3-MoE-235B | 8    | 8            | 1            | 1            | 1      | 8      | 1      |
| DeepSeek-V3    | 8    | 8            | 1            | 1            | 1      | 8      | 1      |

## Third-Party GPU Kernels

| Library                                             | Version | Description                                                     | Source      |
| --------------------------------------------------- | ------- | --------------------------------------------------------------- | ----------- |
| [DeepEP](https://github.com/deepseek-ai/DeepEP)     | 1.2.1   | Expert-parallel all-to-all communication (MoE dispatch/combine) | deepseek-ai |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 2.1.1   | FP8 GEMM kernels with fine-grained scaling (JIT compiled)       | deepseek-ai |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 1.0.0   | Multi-head Latent Attention decode kernels for Hopper GPUs      | deepseek-ai |

All three require SM90+ (NVIDIA Hopper) GPUs. Install from source:

```bash
cd DeepEP && pip install .
cd DeepGEMM && pip install .
cd FlashMLA && pip install .
```

## Quick Start

### Prerequisites

- NVIDIA GPUs with CUDA 12.1+ (SM90+ for FlashMLA / Flash Attention 3)
- RDMA-capable NICs (for multi-node KV cache migration)
- Python 3.10+, Ray, Redis, Rust 1.70+

```bash
pip install torch transformers accelerate
pip install ray redis zmq flatbuffers httpx pydantic jsonargparse
```

### Single-Node Example (Non-Disaggregated)

```bash
ray start --head --port=7078 --dashboard-host=0.0.0.0 --dashboard-port=8265
# examples/non_disagg.py
python examples/non_disagg.py \
    --ray_address <node0-ip>:7078 \
    --model /models/deepseek-v3 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 64 \
    --max_tokens 64 --temperature 0.1 \
    --prompt "What is 1+1?"
```

### PD Disaggregated Example

> **Prerequisite:** Start NanoCtrl before running this example (see [Deployment Guide](../docs/deployment.md) Step 1).

```bash
cd ../NanoCtrl; cargo run --release; cd -;
ray start --head --port=7078 --dashboard-host=0.0.0.0 --dashboard-port=8265
ray start --address <ray-head-ip>:7078
# examples/disagg.py — common config + per-role overlay
python examples/disagg.py \
    --model /models/deepseek-v3 \
    --ray_address <node0-ip>:7078 \
    --nanoctrl_address <node0-ip>:3000 \
    --attention_dp 8 --ffn_ep 8 \
    --kvcache_block_size 64 \
    --prefill.master_address 10.0.0.2:6006 \
    --decode.master_address 10.0.0.1:6006 \
    --decode.loop_count 16
```

## Configuration Reference

### Engine Parameters

| Parameter                | Type  | Default            | Description                                |
| ------------------------ | ----- | ------------------ | ------------------------------------------ |
| `model`                  | str   | Required           | Model path or HuggingFace ID               |
| `mode`                   | str   | `"hybrid"`         | Engine mode: `prefill`, `decode`, `hybrid` |
| `host`                   | str   | `"0.0.0.0"`        | Bind address                               |
| `port`                   | int   | `5000`             | ZMQ port                                   |
| `max_model_len`          | int   | `16384`            | Maximum sequence length                    |
| `max_num_batched_tokens` | int   | `16384`            | Max tokens per batch                       |
| `max_num_seqs`           | int   | `256`              | Max concurrent sequences                   |
| `kvcache_block_size`     | int   | `256`              | KV cache block size (64 for MLA models)    |
| `gpu_memory_utilization` | float | `0.9`              | GPU memory usage fraction                  |
| `enforce_eager`          | bool  | `False`            | Disable CUDA Graph (for debugging)         |
| `ray_address`            | str   | `"127.0.0.1:6379"` | Ray cluster address                        |
| `nanoctrl_address`       | str   | `None`             | NanoCtrl address for PD disaggregation     |
| `log_level`              | str   | `"CRITICAL"`       | Logging level                              |
| `enable_profiler`        | bool  | `False`            | Enable torch profiler                      |
| `profiler_start_step`    | int   | `40`               | Step number to start profiling             |
| `profiling_step`         | int   | `16`               | Number of steps to profile                 |
| `profiler_dir`           | str   | `"./profiler_res"` | Output directory for profiler traces       |

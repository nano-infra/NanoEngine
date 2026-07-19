# Feature Matrix

DLEngine focuses on distributed inference for state-of-the-art large models. This page separates implemented capabilities from roadmap work so the README can stay focused on the production deployment path.

## Status definitions

| Status | Meaning |
| --- | --- |
| **Available** | Implemented in the repository and usable for the supported model/configuration combinations. |
| **Experimental** | Implemented, but still has narrower validation coverage or operational constraints. |
| **Coming soon** | Planned or under active development; do not treat it as production-ready yet. |

## Distributed execution

| Feature | Status | Description |
| --- | --- | --- |
| Ray-based resource management | Available | Discovers cluster GPU resources and places distributed DLEngine workers across nodes. |
| Attention data parallelism | Available | Replicates attention while distributing requests across attention ranks. |
| Wide expert parallelism | Available | Spreads MoE experts across the full GPU set and composes with attention data parallelism. |
| Tensor parallelism | Available | Shards model tensors within supported model and kernel configurations. |
| Prefill/decode disaggregation | Available | Runs prefill and decode as separate services with independent scaling policies. |
| GPUDirect RDMA KV migration | Available | Transfers prefill KV state directly to decode workers through DLSlime. |
| Service discovery and control plane | Available | Uses dlslime-ctrl and Redis for engine registration, heartbeat, scope isolation, and peer metadata. |
| Pipeline parallelism (PP) | **Coming soon** | Splits decoder layers into pipeline stages; production support, correctness coverage, and performance tuning are still being completed. |

## Scheduling, cache, and long context

| Feature | Status | Description |
| --- | --- | --- |
| Continuous batching | Available | Dynamically batches active requests with paged cache allocation. |
| Chunked prefill | Available | Processes long prompts in bounded chunks to control memory and scheduling pressure. |
| Prefix caching | Available | Reuses cached KV state for shared prompt prefixes. |
| FP8 KV cache | Available | Stores supported paged KV caches in FP8 to reduce device-memory use. |
| HiSparse | Available | Keeps bounded hot attention state on device and a larger cold tier on host for supported sparse and sliding-attention models. |
| Million-token context | Experimental | GLM-family long-context serving is supported, but capacity and latency depend strongly on topology, Indexer behavior, and HiSparse configuration. |
| 3FS/L3 cache integration | Experimental | Extends cache capacity beyond local device and host memory for deployments configured with the optional storage path. |

## Model execution

| Feature | Status | Description |
| --- | --- | --- |
| Multi-head Latent Attention (MLA) | Available | Uses compressed latent KV representations for DeepSeek- and GLM-family models. |
| Native Sparse Attention / DSA | Available | Uses model-native Indexer selection and sparse attention for supported DeepSeek and GLM architectures. |
| Gated Delta Net (GDN) | Available | Executes hybrid linear/full-attention layers used by Qwen3.5-family models. |
| Multi-Token Prediction (MTP) | Experimental | Uses model-native prediction heads for speculative decoding where the model and cache configuration permit it. |
| CUDA Graph decode | Available | Captures supported decode paths to reduce launch overhead and improve token latency. |
| FP8 model kernels | Available | Runs Hopper-optimized FP8 attention, GEMM, MoE, and cache kernels for supported architectures. |

## Serving APIs

| Feature | Status | Description |
| --- | --- | --- |
| OpenAI-compatible API | Available | Serves chat completions, completions, streaming responses, and model discovery through dlengine-router. |
| Anthropic-compatible API | Available | Serves Messages and token-counting endpoints, including the Claude Code workflow. |
| Tool calling | Available | Parses model-family tool-call formats and returns OpenAI/Anthropic-compatible tool events. |
| Grammar-constrained decoding | Available | Restricts generation to a supplied grammar or structured-output contract for supported serving requests. |
| Dynamic engine discovery | Available | Routes only to healthy engines registered in the selected dlslime-ctrl scope. |
| Streaming | Available | Streams generated tokens and tool events through the public router API. |

## Scope and constraints

Feature availability is configuration-specific. Model architecture, GPU generation, attention layout, cache mode, and parallel topology may rule out combinations that are individually supported. DLEngine validates many incompatible combinations during configuration, but production deployments should still validate accuracy, memory capacity, and throughput with their exact checkpoint and topology.

See [Supported Models](./supported-models.md) for model-specific coverage and [the production workflow](../README.md#quick-start-ray--dlengine-pd--router) for a complete deployment example.

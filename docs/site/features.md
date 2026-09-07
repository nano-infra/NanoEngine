# Feature Matrix

DLEngine focuses on distributed inference for state-of-the-art large models. This page separates implemented capabilities from roadmap work so the README can stay focused on the production deployment path.

## Status definitions

| Status           | Meaning                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------- |
| **Available**    | Implemented in the repository and usable for the supported model/configuration combinations. |
| **Experimental** | Implemented, but still has narrower validation coverage or operational constraints.          |
| **Coming soon**  | Planned or under active development; do not treat it as production-ready yet.                |

## Distributed execution

| Feature                             | Status       | Description                                                                                                                               |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Ray-based resource management       | Available    | Discovers cluster GPU resources and places distributed DLEngine workers across nodes.                                                     |
| Attention data parallelism          | Available    | Replicates attention while distributing requests across attention ranks.                                                                  |
| Wide expert parallelism             | Available    | Spreads MoE experts across the full GPU set and composes with attention data parallelism.                                                 |
| Tensor parallelism                  | Available    | Shards model tensors within supported model and kernel configurations.                                                                    |
| Prefill/decode disaggregation       | Available    | Runs prefill and decode as separate services with independent scaling policies.                                                           |
| GPUDirect RDMA KV migration         | Available    | Transfers prefill KV state directly to decode workers through DLSlime.                                                                    |
| Service discovery and control plane | Available    | Uses dlslime-ctrl and Redis for engine registration, heartbeat, scope isolation, and peer metadata.                                       |
| Pipeline parallelism (PP)           | Experimental | Supports bounded-microbatch prefill pipelines. Decode remains single-stage, and long-context performance validation is still in progress. |

## Scheduling, cache, and long context

| Feature                  | Status       | Description                                                                                                                                       |
| ------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Continuous batching      | Available    | Dynamically batches active requests with paged cache allocation.                                                                                  |
| Chunked prefill          | Available    | Processes long prompts in bounded chunks to control memory and scheduling pressure.                                                               |
| Prefix caching           | Available    | Reuses cached KV state for shared prompt prefixes.                                                                                                |
| FP8 KV cache             | Available    | Stores supported paged KV caches in FP8 to reduce device-memory use.                                                                              |
| HiSparse                 | Available    | Keeps bounded hot attention state on device and a larger cold tier on host for supported sparse and sliding-attention models.                     |
| Million-token context    | Experimental | GLM-family long-context serving is supported, but capacity and latency depend strongly on topology, Indexer behavior, and HiSparse configuration. |
| 3FS/L3 cache integration | Experimental | Extends cache capacity beyond local device and host memory for deployments configured with the optional storage path.                             |

### Kimi-K3 MLA FP8 cache on Blackwell

Append `--kv_cache_dtype fp8_e4m3` to the existing Kimi-K3 serving command to store
its MLA KV pages as E4M3 FP8. `auto` (the default) retains the previous policy:
K3 dense MLA uses the model dtype, while supported NSA/DSA models select FP8.
`bfloat16` explicitly selects BF16 MLA storage; on sparse models this also
requires `--disable_nsa true`, and HiSparse still requires FP8.

The explicit FP8 option supports compressed rank 512 and a 64-wide positional
component on Blackwell dense/sparse MLA, or the existing Hopper sparse MLA
path. Unsupported hardware/shapes and reference decode fail before checkpoint
weights are loaded. Explicit KV dtype selection currently applies only to MLA.

For K3, an MLA token occupies **576 bytes in FP8 versus 1152 bytes in BF16**.
This halves the MLA page storage. KDA recurrent/conv state and model weights
retain their existing precision, so total process memory is not halved.
Both chunked prefill cache reuse and CUDA Graph decode use the selected format.

K3 automatically sizes MegaMoE's per-rank capacity for the configured prefill
chunk and decode batch, including attention-TP padding. With TP8, chunks of
8192, 16384, and 32768 tokens require capacities of 1024, 2048, and 4096.
Attention-DP routing can put a full chunk on one group, so DP does not further
divide this bound. Longer prompts reuse the same buffer across chunks.
`--mega_moe_max_tokens_per_rank 0` selects automatic sizing (the default);
explicit smaller capacities fail at startup. Larger chunks require more
workspace memory independently of the selected KV-cache precision.

Run the checkpoint-attention validator from an installed checkout:

```bash
python -m examples.kimi_k3_fp8_kv_validation \
  --model /path/to/Kimi-K3 --layers all \
  --prefill-tokens 257 --chunk-tokens 63 --decode-steps 8 \
  --output k3-fp8-validation.json
```

It loads real K3 MLA weights one layer at a time and uses normalized synthetic
activations to compare BF16/FP8 against causal BF16 prefill. It checks cache
writes across noncontiguous pages, cached prefill, repeated decode, graph
replay, and idle-DP cache preservation. This is attention-level validation;
full-model generation quality and distributed serving require separate checks.

## Model execution

| Feature                           | Status       | Description                                                                                                                   |
| --------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Multi-head Latent Attention (MLA) | Available    | Uses compressed latent KV representations for DeepSeek- and GLM-family models.                                                |
| Native Sparse Attention / DSA     | Available    | Uses model-native Indexer selection and sparse attention for supported DeepSeek and GLM architectures.                        |
| Gated Delta Net (GDN)             | Available    | Executes hybrid linear/full-attention layers used by Qwen3.5-family models.                                                   |
| Multi-Token Prediction (MTP)      | Experimental | Linear model-native speculation; GLM supports one predictor recurrently run five times with exact stochastic target sampling. |
| CUDA Graph decode                 | Available    | Captures supported decode paths to reduce launch overhead and improve token latency.                                          |
| FP8 model kernels                 | Available    | Runs Hopper-optimized FP8 attention, GEMM, MoE, and cache kernels for supported architectures.                                |

## Serving APIs

| Feature                      | Status    | Description                                                                                              |
| ---------------------------- | --------- | -------------------------------------------------------------------------------------------------------- |
| OpenAI-compatible API        | Available | Serves chat completions, completions, streaming responses, model discovery, and the OpenCode workflow.   |
| Anthropic-compatible API     | Available | Serves Messages and token-counting endpoints, including the Claude Code workflow.                        |
| Tool calling                 | Available | Parses model-family tool-call formats and returns OpenAI/Anthropic-compatible tool events.               |
| Grammar-constrained decoding | Available | Restricts generation to a supplied grammar or structured-output contract for supported serving requests. |
| Dynamic engine discovery     | Available | Routes only to healthy engines registered in the selected dlslime-ctrl scope.                            |
| Streaming                    | Available | Streams generated tokens and tool events through the public router API.                                  |

## Scope and constraints

Feature availability is configuration-specific. Model architecture, GPU generation, attention layout, cache mode, and parallel topology may rule out combinations that are individually supported. DLEngine validates many incompatible combinations during configuration, but production deployments should still validate accuracy, memory capacity, and throughput with their exact checkpoint and topology.

See [Supported Models](./supported-models.md) for model-specific coverage and [the production workflow](./online-serving.md) for a complete deployment example.

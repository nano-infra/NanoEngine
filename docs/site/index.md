# DLEngine

<div class="hero">
  <div>
    <p class="eyebrow">Distributed inference for state-of-the-art large models</p>
    <h1>Scale prefill, decode, attention, and experts across the cluster.</h1>
    <p>
      DLEngine combines Ray-managed GPU resources, prefill/decode disaggregation,
      wide expert parallelism, sparse attention, and long-context cache management
      behind OpenAI- and Anthropic-compatible APIs.
    </p>
  </div>
</div>

## Start here

<div class="feature-grid">
  <a href="installation/"><strong>Installation</strong><br>Use the aligned development image or prepare a local developer build.</a>
  <a href="online-serving/"><strong>Production serving</strong><br>Launch Ray, dlslime-ctrl, prefill/decode engines, and the public router.</a>
  <a href="supported-models/"><strong>Supported models</strong><br>Check architecture strings, model-family coverage, and constraints.</a>
  <a href="offline-inference/"><strong>Offline inference</strong><br>Validate checkpoints and distributed execution without an HTTP gateway.</a>
</div>

## Designed for large-model serving

| Layer                | Responsibility                                                                          |
| -------------------- | --------------------------------------------------------------------------------------- |
| Ray                  | Manages cluster GPU resources and places distributed workers.                           |
| dlslime-ctrl + Redis | Provides node/service discovery, liveness, scope isolation, and control-plane metadata. |
| DLEngine             | Executes model prefill and decode, owns cache state, and transfers KV through DLSlime.  |
| dlengine-router      | Exposes OpenAI and Anthropic APIs and orchestrates the prefill-to-decode request flow.  |

## Core capabilities

- Attention data parallelism and wide expert parallelism for large MoE models.
- Prefill/decode disaggregation with GPUDirect RDMA KV migration.
- MLA, NSA/DSA, GDN, MTP, FP8 cache, HiSparse, chunked prefill, and prefix reuse.
- Long-context inference for supported DeepSeek- and GLM-family checkpoints.
- OpenAI, Anthropic, tool-calling, grammar-constrained, streaming, Claude Code, and OpenCode workflows.

See the [Feature Matrix](features.md) for implementation status. Pipeline parallelism is listed there as **Coming soon** until its production correctness and performance work is complete.

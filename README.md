# DLEngine: Distributed Inference for State-of-the-Art Large Models

DLEngine is a distributed inference system built primarily for serving state-of-the-art large models. It combines cluster-wide GPU resource management, prefill/decode disaggregation, wide expert parallelism, sparse attention, and long-context inference behind OpenAI- and Anthropic-compatible APIs.

Start with the focused documentation:

- [Installation](./docs/site/installation.md) — image-based and local development setup.
- [Feature matrix](./docs/site/features.md) — distributed inference, serving, attention, cache, and model-execution capabilities.
- [Supported models](./docs/site/supported-models.md) — model families, architectures, and current constraints.

## Components

| Component                            | Language   | Description          | Key Features                                                                                    |
| ------------------------------------ | ---------- | -------------------- | ----------------------------------------------------------------------------------------------- |
| [dlengine](./dlengine)               | Python/C++ | LLM inference engine | Prefill/decode engines, KV cache management, continuous batching, Ray-based distributed workers |
| [dlengine-router](./rust/src/router) | Rust       | HTTP API gateway     | OpenAI and Anthropic APIs, PD routing, dynamic engine discovery, streaming                      |

## Installation

Use the prebuilt CUDA development image for the recommended setup. DeepSeek-family kernels require SM90+ NVIDIA Hopper GPUs. See the [installation guide](./docs/site/installation.md) for image-based setup, local editable installs, component extras, and developer builds.

## Production Deployment

DLEngine production serving combines Ray-managed GPU resources, the dlslime-ctrl discovery/control plane, separate prefill and decode engines, and dlengine-router as the public OpenAI/Anthropic gateway. Follow the complete [Production Serving guide](./docs/site/online-serving.md), including the Claude Code and OpenCode workflows and operational checks.

The router supports foreground and managed background operation. Use `dlengine-router --daemonize yes` to start it in the background, `dlengine-router status` to inspect process health and discovered prefill/decode/hybrid engines, and `dlengine-router stop` for a graceful shutdown. The detailed deployment workflow is documented in the [Production Serving guide](./docs/site/online-serving.md).

For checkpoint validation, debugging, and batch jobs without a public HTTP gateway, use the [Offline Inference guide](./docs/site/offline-inference.md).

______________________________________________________________________

## 🤝 Contribution Workflow

We use GitHub Issues, formal Sub-issues, and pull requests as one hierarchy:

```text
Epic
└── Workstream
    └── Task / Bug
        └── Pull Request
```

- Search for and reuse an existing Issue before creating a new one.
- Track project outcomes as Epics, implementation areas as Workstreams, and concrete deliverables as Task/Bug Sub-issues.
- Keep one primary review objective per PR; split correctness, performance, and large refactors when they can be reviewed or reverted independently.
- Use `Refs #N` for Epics, Workstreams, and partially addressed Issues. Use `Closes #N` only when merging fully satisfies a leaf Issue's acceptance criteria.
- Target the `Pure_dp` integration branch unless the PR documents an explicit exception.
- Create and link follow-up Issues before merging; a note left only in a PR is not considered scheduled work.

See the complete [GitHub Issue and Pull Request Workflow](./docs/github-workflow.zh.md) for Issue templates, branch conventions, review requirements, labels, and weekly triage rules.

Release maintainers should use the guarded [DLEngine release workflow](./docs/releasing.md) to synchronize versions and create tags.

______________________________________________________________________

## 📄 License

See individual component [license](./LICENSE).

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JimyMa/NanoDeploy/issues)
- **Documentation**: Start with the [documentation map](./docs/README.md)
- **Contribution workflow**: [GitHub Issue and Pull Request Workflow](./docs/github-workflow.zh.md)

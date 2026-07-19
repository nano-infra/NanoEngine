# Installation

The recommended way to run DLEngine is the prebuilt CUDA development image. It keeps the Python, Rust, communication, and GPU-kernel versions aligned across every Ray node.

## Requirements

- NVIDIA SM90+ GPUs (Hopper or newer) for the DeepSeek-family optimized kernels.
- CUDA- and driver-compatible hosts on every node.
- RDMA-capable networking for the production prefill/decode KV-migration path.
- The same DLEngine build and model checkpoint on all participating nodes.

## Recommended: development image

The CUDA 12.8 development image bundles PyTorch, the DLEngine kernels, communication libraries, DLSlime, the Rust toolchain, and an optional 3FS USRBIO variant.

See [the Docker guide](https://github.com/JimyMa/NanoDeploy/tree/Pure_dp/docker#readme) for pinned versions, image builds, container startup, mounts, and the 3FS image.

## Local Python installation

Use a local installation for development, unit tests, and code navigation when the required CUDA toolchain and native libraries are already available on the host.

Install the full project:

```bash
pip install ".[all]"
```

Or install only the required Python component:

```bash
pip install ".[dlengine]"   # inference engine
pip install ".[dlenginevl]" # inference engine plus vision-language extras
```

For an editable developer build:

```bash
pip install -e .
```

GPU kernels are maintained under `dlengine/kernel`; users should not separately install the vendored DeepEP, DeepGEMM, or FlashMLA source trees.

## Router build

Build the Rust extension and `dlengine-router`:

```bash
cargo build --release
```

## DLSlime control plane

DLEngine uses DLSlime for peer communication, service discovery, and the control plane. Install the packaged data-plane client and control-plane binary:

```bash
pip install dlslime
pip install dlslime-ctrl
```

For container deployment and external Redis configuration, use the official [DLSlime Docker deployment guide](https://github.com/DeepLink-org/DLSlime/blob/main/docker/README.md).

After installation, continue with the [Ray + DLEngine PD + Router workflow](./online-serving.md). For local Python-only validation, see [Offline Inference](./offline-inference.md).

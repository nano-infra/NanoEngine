# Build the AE image

The [Dockerfile](Dockerfile) builds the AE software environment for Linux
x86_64 and H200 / SM90. It starts from
`vllm/vllm-openai:v0.18.0`, pinned by image digest, and installs the modified
vLLM at commit `5dcf9eec3c55309824a89b996bc57687d4aa4ac8` as a regular wheel.
The base image supplies Python 3.12, PyTorch 2.10.0+cu129, and CUDA 12.9.

## Source inputs

Preparing the build context requires Python 3, Git, and `tar` on a Linux host.
It does not require an existing Docker container. DLSlime is fetched by the
Dockerfile from the repository listed in the
[root README](../../README.md#pinned-dependencies). The Dockerfile uses HTTPS
for the same repository, so this download does not require an SSH key.

| Input | Required revision | Source |
|---|---|---|
| NanoDeploy | `a0752b7b0eb6d08b69cf4c9962552650c742529b` | This repository |
| DLSlime | `4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e` | [nano-infra/DLSlime](https://github.com/nano-infra/DLSlime), branch `feat/hao-basic-alltoall-offsets` |

The pinned DLSlime commit belongs to `feat/hao-basic-alltoall-offsets`, rather
than `main`. The Dockerfile clones that branch, checks out the fixed commit,
and verifies the source revision before installation.

`prepare_context.py` exports NanoDeploy from this repository at its pinned
commit even when this README comes from a newer documentation revision.
Git metadata, untracked files, and uncommitted changes are excluded. No other
local source checkout is required; the Dockerfile downloads its dependencies
from their pinned upstream sources.

The NanoDeploy commit must be available locally. If using a shallow clone
that lacks it, fetch the `asplos27-ae-v1` release tag first:

```zsh
git fetch --depth 1 origin tag asplos27-ae-v1
```

## Prepare and build

From `ae_scripts/` on the host, select a new output directory:

```zsh
python3 docker/prepare_context.py \
  --output /absolute/path/to/ae-image-context

docker build --platform linux/amd64 \
  -t nanodeploy-ae:vllm0180 /absolute/path/to/ae-image-context
```

The helper refuses to overwrite an existing context. Docker needs access to
the base-image registry, Ubuntu package mirrors, PyPI, GitHub, and GitHub's
source-archive service. Configure your site's Docker build proxy if needed.
`--build-arg BUILD_JOBS=8` controls native build parallelism.

The recipe adds the packages in [requirements-extra.txt](requirements-extra.txt)
while constraining the existing runtime stack with
[constraints-base.txt](constraints-base.txt). vLLM reuses the base image's
precompiled extensions. FlashAttention 3 uses the fixed Windreamer wheel;
FlashMLA and CUTLASS use pinned official source archives. DLSlime is built
with `CMAKE_ARGS="-DBUILD_INTRA_OPS=ON"`.

## Run and validate

Follow the [AE container instructions](../README.md#ae-container) to distribute
the image, create a container on every worker, and mount the experiment
repository, checkpoints, and datasets. The image sets
`VLLM_WORKDIR=/opt/ae/vllm-workdir`; use the same value for `AE_VLLM_ROOT` in
`paths.env`. This directory is separate from source checkouts so Python loads
the installed vLLM package.

The Dockerfile checks package versions and required native imports with
`python3 /opt/ae/verify_environment.py`. These checks do not run the paper's
GPU experiments or validate performance. The complete Dockerfile has not yet
been validated by an end-to-end image build.

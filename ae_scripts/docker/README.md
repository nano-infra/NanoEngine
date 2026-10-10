# Build the AE image

The [Dockerfile](Dockerfile) reconstructs the observed `ae_merged` software
stack for Linux x86_64 and H200 / SM90. It starts from
`vllm/vllm-openai:v0.18.0`, pinned by image digest, and installs the modified
vLLM at commit `5dcf9eec3c55309824a89b996bc57687d4aa4ac8` as a regular wheel.
The base image supplies Python 3.12, PyTorch 2.10.0+cu129, and CUDA 12.9.

## Source inputs

Preparing the build context requires Python 3, Git, and `tar` on a Linux host.
It does not require an existing Docker container. Obtain the two native
source checkouts from the artifact maintainers; these checkouts are separate
build inputs and are not bundled in this directory.

| Input | Required revision |
|---|---|
| NanoDeploy (this repository) | `a0752b7b0eb6d08b69cf4c9962552650c742529b` |
| DLSlime | `4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e` |
| `nano_intra_alltoall` | `e070174a7e4b2c412bd9c78beafaa489df2720de` |

`prepare_context.py` checks each native checkout's HEAD, copies its tracked
working-tree files, and records tracked local changes in
`SOURCE_WORKTREE.patch`. Preserve the supplied native checkouts' build fixes;
for `nano_intra_alltoall`, the supplied CMake file also installs
`intra_alltoall/__init__.py`. The helper exports NanoDeploy at its pinned
commit even when this README comes from a newer documentation revision.
Git metadata and untracked local files are excluded.

The NanoDeploy commit must be available locally. If using a shallow clone
that lacks it, fetch the `asplos27-ae-v1` release tag first:

```zsh
git fetch --depth 1 origin tag asplos27-ae-v1
```

## Prepare and build

From `ae_scripts/` on the host, select a new output directory:

```zsh
python3 docker/prepare_context.py \
  --dlslime /absolute/path/to/DLSlime \
  --nano-intra-alltoall /absolute/path/to/nano_intra_alltoall \
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

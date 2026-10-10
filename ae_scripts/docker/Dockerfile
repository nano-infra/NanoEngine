# ae_merged / NanoDeploy environment, sampled on 2026-10-09.
# linux/amd64; H200 / SM90. See README.md before building.
ARG BASE_IMAGE=vllm/vllm-openai:v0.18.0@sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05
FROM ${BASE_IMAGE}

USER root
# Bootstrap zsh; this first step uses POSIX sh, not bash.
SHELL ["/bin/sh", "-ec"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        zsh ca-certificates curl git build-essential pkg-config openssh-client \
        libnuma-dev libibverbs-dev ibverbs-providers \
        libcublas-dev-12-9 libcurand-dev-12-9 \
        libcusolver-dev-12-9 libcusparse-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/usr/bin/zsh", "-eu", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=8
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Torch, Triton, FlashInfer, DeepEP, DeepGEMM and NVSHMEM already exist
# in the pinned base image. Constrain them while adding Python packages.
COPY requirements-extra.txt constraints-base.txt /opt/ae/
RUN python3 -m pip install --no-cache-dir \
        --constraint /opt/ae/constraints-base.txt \
        --requirement /opt/ae/requirements-extra.txt

# Replace the base vLLM with the pinned modified source, as a normal wheel.
# Reuse this image's matching native libraries and generated Python interfaces.
ARG VLLM_REPOSITORY=https://github.com/FirwoodLin/vllm.git
ARG VLLM_BRANCH=ae-repro-clean
ARG VLLM_COMMIT=5dcf9eec3c55309824a89b996bc57687d4aa4ac8
COPY export_vllm_precompiled.py /opt/ae/
RUN git clone --depth 1 --single-branch --branch "${VLLM_BRANCH}" \
        "${VLLM_REPOSITORY}" /tmp/vllm-source \
    && git -C /tmp/vllm-source fetch --depth 1 origin "${VLLM_COMMIT}" \
    && git -C /tmp/vllm-source checkout --detach "${VLLM_COMMIT}" \
    && test "$(git -C /tmp/vllm-source rev-parse HEAD)" = "${VLLM_COMMIT}" \
    && python3 /opt/ae/export_vllm_precompiled.py \
        /tmp/vllm-base-precompiled.whl /opt/ae/vllm-precompiled.json \
    && VLLM_USE_PRECOMPILED=1 \
        VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-base-precompiled.whl \
        VLLM_VERSION_OVERRIDE=0.18.0 \
        python3 -m pip wheel --no-index --no-cache-dir --no-build-isolation --no-deps \
        --wheel-dir /tmp/vllm-wheel /tmp/vllm-source \
    && python3 -m pip install --no-index --no-cache-dir --no-deps --force-reinstall \
        /tmp/vllm-wheel/vllm-*.whl \
    && print -r -- "${VLLM_COMMIT}" > /opt/ae/vllm-source-commit.txt \
    && rm -rf /tmp/vllm-source /tmp/vllm-wheel /tmp/vllm-base-precompiled.whl

# Exact wheel found in the user's cu129_torch2100 index.
ARG FLASH_ATTN3_URL=https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.03.16-4d76052/flash_attn_3-3.0.0%2B20260316.cu129torch2100cxx11abitrue.71bf77-cp39-abi3-linux_x86_64.whl#sha256=186595796ffab2449f57c429ca9da94ebf081a5c84bad9f4ca9c49c36d3ebe48
RUN python3 -m pip install --no-cache-dir --no-deps "${FLASH_ATTN3_URL}"

# FlashMLA: official pinned source archives, without git clone.
# CUTLASS is a submodule and is not contained in GitHub's source archive.
ARG FLASH_MLA_COMMIT=1408756a88e52a25196b759eaf8db89d2b51b5a1
ARG FLASH_MLA_CUTLASS_COMMIT=e94e888df3551224738bfa505787b515eae8352f
ARG FLASH_MLA_DISABLE_SM100=1
COPY prepare_flashmla.py /opt/ae/
RUN mkdir -p /opt/src/FlashMLA/csrc/cutlass \
    && curl --fail --location --retry 3 \
        "https://codeload.github.com/deepseek-ai/FlashMLA/tar.gz/${FLASH_MLA_COMMIT}" \
        | tar -xz --strip-components=1 -C /opt/src/FlashMLA \
    && curl --fail --location --retry 3 \
        "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${FLASH_MLA_CUTLASS_COMMIT}" \
        | tar -xz --strip-components=1 -C /opt/src/FlashMLA/csrc/cutlass \
    && python3 /opt/ae/prepare_flashmla.py /opt/src/FlashMLA "${FLASH_MLA_COMMIT}" \
    && FLASH_MLA_DISABLE_SM100="${FLASH_MLA_DISABLE_SM100}" NVCC_THREADS=4 \
        python3 -m pip install --verbose --no-cache-dir --no-build-isolation \
        --no-deps /opt/src/FlashMLA

# Export these local, potentially private checkouts using prepare_context.py.
ARG DLSLIME_COMMIT=4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e
COPY sources/DLSlime/ /opt/src/DLSlime/
RUN test "$(cat /opt/src/DLSlime/SOURCE_COMMIT)" = "${DLSLIME_COMMIT}" \
    && CMAKE_ARGS="-DBUILD_INTRA_OPS=ON" \
        python3 -m pip install --verbose --no-cache-dir --no-build-isolation \
        --no-deps --editable /opt/src/DLSlime

# Fig. 17's standalone extension; included in the observed environment.
ARG NANO_INTRA_COMMIT=e070174a7e4b2c412bd9c78beafaa489df2720de
COPY sources/nano_intra_alltoall/ /opt/src/nano_intra_alltoall/
RUN test "$(cat /opt/src/nano_intra_alltoall/SOURCE_COMMIT)" = "${NANO_INTRA_COMMIT}" \
    && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" \
        python3 -m pip install --verbose --no-cache-dir --no-build-isolation \
        --no-deps /opt/src/nano_intra_alltoall

ARG NANODEPLOY_COMMIT=a0752b7b0eb6d08b69cf4c9962552650c742529b
COPY sources/NanoDeploy/ /opt/src/NanoDeploy/
RUN test "$(cat /opt/src/NanoDeploy/SOURCE_COMMIT)" = "${NANODEPLOY_COMMIT}" \
    && python3 -m pip install --verbose --no-cache-dir --no-build-isolation \
        --no-deps --editable /opt/src/NanoDeploy

COPY verify_environment.py /opt/ae/
# Import native extensions and check required symbols; no GPU allocation.
RUN python3 /opt/ae/verify_environment.py

# Keep runtime imports separate from any vLLM source checkout.
RUN mkdir -p /opt/ae/vllm-workdir
ENV VLLM_WORKDIR=/opt/ae/vllm-workdir

WORKDIR /opt/src/NanoDeploy
ENTRYPOINT ["/usr/bin/zsh"]
CMD ["-l"]

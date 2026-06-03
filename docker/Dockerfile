FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG PYTORCH_VERSION=2.10.0
ARG PYTORCH_INDEX_URL
ARG EXPECTED_TORCH_CUDA=12.8
ARG NANODEPLOY_VERSION=0.2.0
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG RUST_TOOLCHAIN=1.95.0
ARG DEEP_EP_REF=567632dd59810d77b3cc05553df953cc0f779799
ARG DEEP_EP_VERSION=v1.2.1-25-g567632d
ARG FLASH_MLA_REF=1408756a88e52a25196b759eaf8db89d2b51b5a1
ARG FLASH_MLA_VERSION=1408756
ARG DEEP_GEMM_REF=891d57b4db1071624b5c8fa0d1e51cb317fa709f
ARG DEEP_GEMM_VERSION=v2.1.1.post3-16-g891d57b
ARG FLASHINFER_REF=v0.6.9
ARG DLSLIME_REF=v0.1.16
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG OH_MY_ZSH_REF=c86ba78e2ff5c5a3e9282a84c0cc220dd3d5f253
ARG ZSH_AUTOSUGGESTIONS_REF=v0.7.1
ARG ZSH_SYNTAX_HIGHLIGHTING_REF=0.8.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --default-timeout=600 --retries=20 --progress-bar off --ignore-installed setuptools wheel packaging build \
    && if [[ -n "${PYTORCH_INDEX_URL}" ]]; then \
        python3 -m pip install --break-system-packages --default-timeout=600 --retries=20 --progress-bar off --index-url "${PYTORCH_INDEX_URL}" torch==${PYTORCH_VERSION}; \
      else \
        python3 -m pip install --break-system-packages --default-timeout=600 --retries=20 --progress-bar off torch==${PYTORCH_VERSION}; \
      fi \
    && EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA}" python3 -c 'import os, torch; expected = os.environ.get("EXPECTED_TORCH_CUDA", "12.8"); actual = torch.version.cuda; print(f"torch={torch.__version__} cuda={actual}"); assert actual == expected, f"expected torch CUDA {expected}, got {actual}"'

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        curl \
        git \
        ibverbs-providers \
        iproute2 \
        less \
        libasio-dev \
        libibverbs-dev \
        libmlx5-1 \
        libnuma-dev \
        librdmacm-dev \
        ninja-build \
        openssh-client \
        pkg-config \
        procps \
        python3-dev \
        python3-venv \
        rdma-core \
        vim \
        zsh \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --default-timeout=600 --retries=20 --progress-bar off \
        "cmake>=3.26" \
        "ninja" \
        "pybind11>=2.12" \
        "scikit-build-core>=0.10" \
        "maturin>=1.0,<2.0"

COPY NanoDeploy/pyproject.toml /tmp/nanodeploy-pyproject.toml

RUN python3 -c 'import subprocess, sys, tomllib; data = tomllib.load(open("/tmp/nanodeploy-pyproject.toml", "rb")); deps = list(data["project"]["dependencies"]); deps += data["project"].get("optional-dependencies", {}).get("cuda-devel", []); skip = {"dlslime"}; deps = [d for d in deps if d.split(";", 1)[0].split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].strip().lower().replace("_", "-") not in skip]; subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "--default-timeout=600", "--retries=20", "--progress-bar", "off", *deps])'

RUN mkdir -p /opt/src

RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/src/DeepEP \
    && cd /opt/src/DeepEP \
    && git checkout ${DEEP_EP_REF} \
    && unset NVSHMEM_DIR \
    && export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    && export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
    && export TORCH_CUDA_ARCH_LIST=9.0 \
    && export MAX_JOBS=64 \
    && python3 -m pip install --break-system-packages --no-build-isolation --no-cache-dir -v .

RUN git clone https://github.com/deepseek-ai/FlashMLA.git /opt/src/FlashMLA \
    && cd /opt/src/FlashMLA \
    && git checkout ${FLASH_MLA_REF} \
    && git submodule update --init --recursive \
    && export FLASH_MLA_DISABLE_SM100=1 \
    && export NVCC_THREADS=32 \
    && export MAX_JOBS=64 \
    && python3 -m pip install --break-system-packages --no-build-isolation --no-cache-dir -v .

RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /opt/src/DeepGEMM \
    && cd /opt/src/DeepGEMM \
    && git checkout ${DEEP_GEMM_REF} \
    && git submodule update --init --recursive \
    && export DG_FORCE_BUILD=1 \
    && export DG_USE_LOCAL_VERSION=1 \
    && export MAX_JOBS=64 \
    && python3 -m pip install --break-system-packages --no-build-isolation --no-cache-dir -v .

RUN git clone --recursive --depth 1 --branch ${FLASHINFER_REF} https://github.com/flashinfer-ai/flashinfer.git /opt/src/flashinfer \
    && cd /opt/src/flashinfer \
    && export FLASHINFER_CUDA_ARCH_LIST="9.0a" \
    && export FLASHINFER_NVCC_THREADS=32 \
    && python3 -m pip install --break-system-packages --no-build-isolation --no-cache-dir -v .

RUN python3 -m pip install --break-system-packages --no-cache-dir "${FLASH_ATTN_WHEEL}"

RUN git clone --depth 1 --branch ${DLSLIME_REF} https://github.com/Deeplink-org/DLSlime.git /opt/src/DLSlime \
    && cd /opt/src/DLSlime \
    && python3 -m pip install --break-system-packages --no-build-isolation --no-cache-dir -v ./dlslime

RUN mkdir -p /workspace

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rustc --version \
    && cargo --version

RUN git init /root/.oh-my-zsh \
    && cd /root/.oh-my-zsh \
    && git remote add origin https://github.com/ohmyzsh/ohmyzsh.git \
    && git fetch --depth 1 origin "${OH_MY_ZSH_REF}" \
    && git checkout --detach FETCH_HEAD \
    && git clone --depth 1 --branch "${ZSH_AUTOSUGGESTIONS_REF}" https://github.com/zsh-users/zsh-autosuggestions.git /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && git clone --depth 1 --branch "${ZSH_SYNTAX_HIGHLIGHTING_REF}" https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting \
    && { \
        echo 'export ZSH=/root/.oh-my-zsh'; \
        echo 'ZSH_THEME="ys"'; \
        echo 'plugins=(git docker python pip zsh-autosuggestions zsh-syntax-highlighting)'; \
        echo 'source $ZSH/oh-my-zsh.sh'; \
        echo 'export CUDA_HOME=/usr/local/cuda'; \
        echo 'export NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem'; \
        echo 'export RUSTUP_HOME=/opt/rustup'; \
        echo 'export CARGO_HOME=/opt/cargo'; \
        echo 'export PIP_BREAK_SYSTEM_PACKAGES=1'; \
        echo 'export PIP_ROOT_USER_ACTION=ignore'; \
        echo 'export PATH=/opt/cargo/bin:/usr/local/cuda/bin:$PATH'; \
        echo 'export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}'; \
        echo 'export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}'; \
        echo 'export TORCH_CUDA_ARCH_LIST=9.0'; \
        echo 'export FLASHINFER_CUDA_ARCH_LIST=9.0a'; \
        echo 'export TVM_FFI_CUDA_ARCH_LIST=9.0'; \
        echo 'export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0'; \
      } > /root/.zshrc

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="NanoDeploy" \
      org.opencontainers.image.version="${NANODEPLOY_VERSION}" \
      org.opencontainers.image.description="NanoDeploy CUDA 12.8 devel container with PyTorch, DeepEP, FlashMLA, DeepGEMM, FlashInfer, FlashAttention, DLSlime, and build toolchains. NanoDeploy source is intentionally not included." \
      org.opencontainers.image.source="https://github.com/DeepLink-org/NanoDeploy" \
      org.opencontainers.image.url="https://github.com/DeepLink-org/NanoDeploy" \
      org.opencontainers.image.documentation="https://github.com/DeepLink-org/NanoDeploy#docker-development-image" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.ref.name="nanodeploy:${NANODEPLOY_VERSION}-cu128-devel" \
      org.opencontainers.image.vendor.deep_ep="${DEEP_EP_VERSION}@${DEEP_EP_REF}" \
      org.opencontainers.image.vendor.flash_mla="${FLASH_MLA_VERSION}@${FLASH_MLA_REF}" \
      org.opencontainers.image.vendor.deep_gemm="${DEEP_GEMM_VERSION}@${DEEP_GEMM_REF}" \
      org.opencontainers.image.vendor.flashinfer="${FLASHINFER_REF}" \
      org.opencontainers.image.vendor.dlslime="${DLSLIME_REF}" \
      org.opencontainers.image.vendor.flash_attn="v2.8.1+cu12torch2.10cxx11abiTRUE" \
      org.opencontainers.image.vendor.rust="${RUST_TOOLCHAIN}" \
      org.opencontainers.image.vendor.oh_my_zsh="${OH_MY_ZSH_REF}" \
      org.opencontainers.image.vendor.zsh_autosuggestions="${ZSH_AUTOSUGGESTIONS_REF}" \
      org.opencontainers.image.vendor.zsh_syntax_highlighting="${ZSH_SYNTAX_HIGHLIGHTING_REF}"

CMD ["/usr/bin/zsh"]

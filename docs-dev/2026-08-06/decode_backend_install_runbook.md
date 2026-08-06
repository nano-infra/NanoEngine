# Decode 后端固定版本安装手册

日期：2026-08-06

本文记录当前 NanoDeploy DeepSeek-V3/Kimi-K2 decode 路径已经验证通过的
dlBLAS、DeepGEMM 和 DeepEP 卸载、构建、安装及验收命令。

## 1. 固定版本和安装形态

| 组件 | worktree tag | commit | 安装形态 | Python 版本号 |
| --- | --- | --- | --- | --- |
| dlBLAS | `v0.0.7` | `6bc37b092d96531b5cdcb2bf5d9f9b84df3d960c` | editable | `0.0.7` |
| DeepGEMM | `v2.1.1.post3` | `c9f8b34dcdacc20aa746b786f983492c51072870` | 本地 wheel | `2.1.1+c9f8b34` |
| DeepEP | `v1.2.1` | `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` | editable | `1.2.1+9af0e0d` |

对应路径：

```text
/mnt/nvme1n1/ml_research/linbinbin1/dlBLAS-aug
/mnt/nvme1n1/ml_research/linbinbin1/DeepGEMM-aug
/mnt/nvme1n1/ml_research/linbinbin1/DeepEP-aug
```

DeepGEMM 不使用 `pip install -e`。该版本的 editable 安装不会把
`third-party/cutlass` 和 CuTe 头文件复制到运行时 `deep_gemm/include`，第一次
JIT 编译会报：

```text
fatal error: cutlass/arch/barrier.h: No such file or directory
```

从相同 worktree 构建本地 wheel 会把这些头文件一起打包，同时保留 NanoDeploy
启动检查要求的版本号 `2.1.1+c9f8b34`。

## 2. 完整重装命令

应在没有 NanoDeploy/Ray actor 使用这些 Python 包时执行。以下命令只读取本地
worktree，并通过 `--no-deps`/`--no-build-isolation` 和 `DG_FORCE_BUILD=1`
避免访问网络或隐式更换依赖。

```bash
set -euo pipefail

ML_ROOT=/mnt/nvme1n1/ml_research/linbinbin1
DLBLAS_DIR="${ML_ROOT:?}/dlBLAS-aug"
DEEPGEMM_DIR="${ML_ROOT:?}/DeepGEMM-aug"
DEEPEP_DIR="${ML_ROOT:?}/DeepEP-aug"
NVSHMEM_DIR=/sgl-workspace/nvshmem/install

DLBLAS_COMMIT=6bc37b092d96531b5cdcb2bf5d9f9b84df3d960c
DEEPGEMM_COMMIT=c9f8b34dcdacc20aa746b786f983492c51072870
DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee

test "$(git -C "${DLBLAS_DIR:?}" rev-parse HEAD)" = "${DLBLAS_COMMIT:?}"
test "$(git -C "${DEEPGEMM_DIR:?}" rev-parse HEAD)" = "${DEEPGEMM_COMMIT:?}"
test "$(git -C "${DEEPEP_DIR:?}" rev-parse HEAD)" = "${DEEPEP_COMMIT:?}"
test -z "$(git -C "${DLBLAS_DIR:?}" status --porcelain)"
test -z "$(git -C "${DEEPGEMM_DIR:?}" status --porcelain)"
test -z "$(git -C "${DEEPEP_DIR:?}" status --porcelain)"
test -f "${DEEPGEMM_DIR:?}/third-party/cutlass/include/cutlass/arch/barrier.h"
test -f "${NVSHMEM_DIR:?}/include/nvshmem.h"

python -m pip uninstall -y dlblas deep-gemm deep-ep

PIP_NO_INDEX=1 python -m pip install \
  --no-deps --no-build-isolation -e "${DLBLAS_DIR:?}"

DG_WHEEL_DIR="$(mktemp -d /tmp/nanodeploy-deep-gemm-wheel.XXXXXX)"
test -n "${DG_WHEEL_DIR:?}"
(
  cd "${DEEPGEMM_DIR:?}"
  DG_FORCE_BUILD=1 \
  DG_USE_LOCAL_VERSION=1 \
  MAX_JOBS=16 \
    python setup.py bdist_wheel --dist-dir "${DG_WHEEL_DIR:?}"
)

shopt -s nullglob
DG_WHEELS=("${DG_WHEEL_DIR:?}"/deep_gemm-2.1.1+c9f8b34-*.whl)
if (( ${#DG_WHEELS[@]} != 1 )) || [[ ! -f "${DG_WHEELS[0]}" ]]; then
  echo "Expected exactly one DeepGEMM wheel under ${DG_WHEEL_DIR:?}" >&2
  exit 1
fi
PIP_NO_INDEX=1 python -m pip install \
  --force-reinstall --no-deps "${DG_WHEELS[0]}"

NVSHMEM_DIR="${NVSHMEM_DIR:?}" \
TORCH_CUDA_ARCH_LIST=9.0 \
MAX_JOBS=16 \
PIP_NO_INDEX=1 \
  python -m pip install -v \
    --no-deps --no-build-isolation -e "${DEEPEP_DIR:?}"

echo "DeepGEMM wheel kept at: ${DG_WHEELS[0]}"
```

说明：

- `TORCH_CUDA_ARCH_LIST=9.0` 对应当前 H200/SM90 节点。
- DeepEP 必须使用 `/sgl-workspace/nvshmem/install`，否则构建脚本会禁用
  NVSHMEM、internode 和 low-latency 功能。
- `MAX_JOBS=16` 是当前节点使用过的保守并行度，可按构建机 CPU/内存调整。
- wheel 输出保留在新建的 `/tmp/nanodeploy-deep-gemm-wheel.*` 目录中，便于在
  同一节点重复安装；系统清理 `/tmp` 后需重新构建。

## 3. 安装验收

先检查 pip 元数据和 editable 路径：

```bash
python -m pip show dlblas deep-gemm deep-ep
```

预期：

- dlBLAS 显示 `0.0.7`，editable project location 指向 `dlBLAS-aug`。
- DeepGEMM 显示 `2.1.1+c9f8b34`，location 位于 Python `site-packages`，不应
  显示 editable project location。
- DeepEP 显示 `1.2.1+9af0e0d`，editable project location 指向
  `DeepEP-aug`。

然后在 NanoDeploy 仓库中执行版本、API 和 DeepGEMM JIT 头文件检查。必须先
导入 `torch`，使 `libc10.so` 等 PyTorch 动态库进入进程：

```bash
cd /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July

python - <<'PY'
from importlib import metadata
from pathlib import Path

import torch
import deep_gemm
import deep_ep

from nanodeploy.worker.decode_backend_compat import validate_decode_backend_compat

versions = validate_decode_backend_compat(rank=0)
assert versions == {
    "dlblas": "0.0.7",
    "deep_gemm": "2.1.1+c9f8b34",
    "deep_ep": "1.2.1+9af0e0d",
}

barrier_header = (
    Path(deep_gemm.__file__).parent
    / "include/cutlass/arch/barrier.h"
)
assert barrier_header.is_file(), barrier_header

print("versions:", versions)
print("deep_gemm:", deep_gemm.__file__)
print("deep_ep:", deep_ep.__file__)
print("JIT header:", barrier_header)
print("decode backend installation: OK")
PY
```

安装验收只证明版本、Python/C++ 扩展和必要 API 正确。部署前仍应设置
`SLIME_QP_NUM=4`，并运行一次实际 DP8/EP8 decode smoke test，以覆盖
DeepGEMM 首次 JIT 及 DeepEP dispatch/combine。

## 4. 当前节点已验证结果

2026-08-06 使用上述版本组合在 Ray `10.102.252.174:6379` 的单节点
8 张 H200 上完成 DeepSeek-V3 DP8/EP8 eager decode smoke test：8 个请求均完成，
共生成 16 个 decode token；8 个 rank 均显式销毁 DeepEP，测试结束后 8 张 GPU
全部回收。

成功日志：
`bench_logs/decode_backend_smoke/deepseek_dp8_ep8_eager_retry.log`。

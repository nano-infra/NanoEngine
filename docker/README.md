# DLEngine Docker Images

This directory holds the container build definitions for DLEngine.

| File                                     | Image                        | Purpose                                                                                                         |
| ---------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`Dockerfile`](./Dockerfile)             | `dlengine:0.2.0-cu128-devel` | CUDA 12.8 development image (PyTorch + DeepEP/DeepGEMM/FlashMLA/FlashInfer/flash-attn/DLSlime + Rust toolchain) |
| [`Dockerfile.hf3fs`](./Dockerfile.hf3fs) | `dlengine:cu128-devel-3fs`   | The dev image **plus 3FS USRBIO** (`hf3fs_py_usrbio`) support                                                   |

______________________________________________________________________

## Development image (`Dockerfile`)

Built from NVIDIA CUDA 12.8 devel (Ubuntu 24.04, Python 3.12), PyTorch 2.10 CUDA 12.8,
source-built DeepEP/DeepGEMM/FlashMLA/FlashInfer, release-wheel flash-attn, rustup-managed
Rust, and the build toolchains needed for DLEngine. The image intentionally **does not
include the DLEngine source tree**; mount or clone DLEngine inside the container and
install it there. This keeps the expensive dependency layers reusable across source changes.

### Pinned third-party dependencies

The development image pins every external build dependency. Prefer tags when upstream
provides a usable tag; otherwise pin the exact commit that has been smoke-tested.

| Library                                                    | Pinned version / ref                    | Notes                                                               |
| ---------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------- |
| PyTorch                                                    | `2.10.0+cu128`                          | CUDA 12.8 wheel.                                                    |
| [DeepEP](https://github.com/deepseek-ai/DeepEP)            | `567632dd` (`v1.2.1-25-g567632d`)       | Nearest tag: `v1.2.1`; pinned commit is the tested post-tag build.  |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)        | `891d57b4` (`v2.1.1.post3-16-g891d57b`) | Nearest tag: `v2.1.1.post3`; pinned commit reports package `2.5.0`. |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA)        | `1408756a`                              | Upstream currently has no tags; pinned by commit.                   |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer)  | `v0.6.9`                                | Built from source.                                                  |
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | `v2.8.1` wheel for `cu12` / `torch2.10` | Uses the release wheel.                                             |
| [DLSlime](https://github.com/Deeplink-org/DLSlime)         | `v0.1.16`                               | Builds `dlslime`; `dlslime-ctrl` is not built in this image.        |
| Rust                                                       | `1.95.0` via rustup                     | Minimal rustup toolchain; not installed from apt.                   |

The DeepSeek kernels require SM90+ (NVIDIA Hopper) GPUs.

### Build

```bash
docker build --network host \
  -f docker/Dockerfile \
  -t dlengine:0.2.0-cu128-devel \
  .
```

Private mirrors or proxies can be passed with Docker build args in local environments;
the image does not require them.

### Run for local development

```bash
docker run --gpus all --rm -it --network host --ipc=host \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --device=/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v $PWD:/workspace/DLEngine \
  -w /workspace/DLEngine/dlengine \
  dlengine:0.2.0-cu128-devel
```

Inside the container, install DLEngine from the mounted checkout (the GPU
compute kernels ship inside the `dlengine.kernel` subpackage):

```bash
python3 -m pip install --break-system-packages --no-build-isolation -v -e .
```

______________________________________________________________________

## 3FS USRBIO image (`Dockerfile.hf3fs`)

Adds 3FS USRBIO support (`hf3fs_py_usrbio` + `hf3fs_fuse`) on top of the dev image, for
testing 3FS access (FUSE/POSIX and USRBIO) from inside a DLEngine container.

### Why it is built this way

- The dev image is **Ubuntu 24.04 (noble), Python 3.12** → use the **cp312** wheel.
- The prebuilt wheel is compiled on **Ubuntu 22.04 (jammy)**; its native lib
  `libhf3fs_api_shared.so` depends on **boost 1.74 / ICU 70 / glog / gflags / ...**, whose
  sonames differ from noble's (boost 1.83 / ICU 74). So instead of `apt install`, we
  **copy the exact runtime `.so` from the `hf3fs:dev-py312` image** into `/opt/hf3fs/lib`
  and add it (plus the wheel's install dir) to `LD_LIBRARY_PATH`.
- RDMA userspace libs (`libibverbs`/`librdmacm`/`libmlx5`/`rdma-core`) are already present
  in the dev image, which USRBIO needs.

### Prerequisites

- Base image `dlengine:0.2.0-cu128-devel` built (see above).
- Runtime image `hf3fs:dev-py312` available locally (provides the jammy `.so`).
- The **cp312** wheel copied into this `docker/` directory (it is git-ignored):

```bash
cp /path/to/3FS/dist/hf3fs_py_usrbio-1.2.9+22fca04-cp312-cp312-linux_x86_64.whl \
   DLEngine/docker/
```

### Build

```bash
cd DLEngine
docker build -f docker/Dockerfile.hf3fs -t dlengine:cu128-devel-3fs docker/
```

Build args (optional): `DLENGINE_BASE`, `HF3FS_RUNTIME_IMAGE`, `HF3FS_WHEEL`.

The Dockerfile runs `python3 -c "import hf3fs_py_usrbio, hf3fs_fuse.io"` as the last step,
so a successful build **guarantees the import works** (import needs neither RDMA hardware
nor a mount).

> Note: keep the original wheel filename. Newer pip (Ubuntu 24.04) rejects a renamed
> `*.whl` with "is not a valid wheel filename", so the wheel is copied into a directory and
> installed via a glob.

### Run (to actually exercise USRBIO)

`import` works anywhere, but real USRBIO read/write needs RDMA + the 3FS mount visible in
the container.

**Recommended: bind-mount the host's existing FUSE mount (consumer mode).**

The host already runs `hf3fs_fuse_main` and mounts at `/3fs/mnt`. You only need the
container to *see* that mount — no need to start FUSE again inside the container.

```bash
docker run --gpus all --rm -it \
  --network host \
  --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --ipc=host \
  --mount type=bind,source=/3fs,target=/3fs,bind-propagation=rslave \
  dlengine:cu128-devel-3fs zsh
```

Equivalent shorter form:

```bash
docker run ... -v /3fs/mnt:/3fs/mnt:rslave ...
# or bind the parent tree:
docker run ... -v /3fs:/3fs:rslave ...
```

Notes:

- **`bind-propagation=rslave` (or `:rslave`) is the important part.** FUSE is a mount on
  top of a directory; plain `-v /3fs/mnt:/3fs/mnt` without propagation often shows an
  empty directory inside the container. `rslave` lets the host FUSE mount propagate in
  one direction (host → container). Prefer `rslave` over `rshared`.
- **`--device /dev/fuse` is NOT required** for this consumer pattern. `/dev/fuse` is only
  needed when you run `hf3fs_fuse_main` *inside* the container to create a new mount.
- Host mount must use `allow_other` (yours does) so non-root users in the container can
  access files.
- RDMA still needs `--network host`, `--device /dev/infiniband`, `IPC_LOCK`, `memlock`.

**Alternative: mount 3FS inside the container (heavier).**

Only if you cannot bind-mount the host mount:

```bash
docker run ... \
  --device /dev/fuse \
  --privileged \   # or CAP_SYS_ADMIN + /dev/fuse
  ...
# then run hf3fs_fuse_main --launcher_cfg /opt/3fs/etc/hf3fs_fuse_main_launcher.toml
```

This requires shipping `/opt/3fs/bin` + configs into the image or mounting them separately.

Verify inside the container:

```bash
df -h | grep hf3fs          # should show /3fs/mnt
ls /3fs/mnt                 # should list cluster dirs, not empty
python3 tools/3fs/benchmark/bench_3fs.py --engine both --op read \
  --dir /3fs/mnt/<writable> --bs 1M --size 64M --numjobs 1
```

See the 3FS docs for demos, benchmark and troubleshooting: `docs/3FS/README.md`.

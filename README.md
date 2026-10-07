# NanoDeploy

A lightweight LMDeploy implementation built from scratch.

## Pinned dependencies

The native MoE backends and the collective library are built from specific
commits. Rebuild or reinstall them with
[`scripts/reinstall_deepgemm_deepep_aug.zsh`](scripts/reinstall_deepgemm_deepep_aug.zsh),
which verifies each checkout is at the expected commit and asserts the
installed versions after building.

| Component | Version | Source |
|---|---|---|
| DLSlime | `4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e` | `git@github.com:DeepLink-org/DLSlime.git` |
| DeepEP | `1.2.1+73b6ea4` | commit `73b6ea4a439ba03a695563f9fd242c8e4b02b37c` |
| DeepGEMM | `2.3.0+477618c` | commit `477618cd51baffca09c4b0b87e97c03fe827ef03` |
| FlashMLA | `1.0.0+1408756` | `git@github.com:deepseek-ai/FlashMLA.git` |

Notes:

- The pinned DLSlime commit lives on branch `hao-basic-alltoall-offsets`, not
  on `main`. Fetch that branch explicitly; a plain clone of `main` will not
  contain it.
- DLSlime must be compiled with CUDA and `BUILD_INTRA_OPS=ON`. The default
  PyPI build does not provide the intra-node all-to-all operators.
- DeepEP must be built against the NVSHMEM installation on the host, otherwise
  the build disables NVSHMEM, internode, and low-latency support.

## Result reproduction

To reproduce the paper figures and end-to-end experiments, see
[`ae_scripts/`](ae_scripts/). Start with
[`ae_scripts/README.md`](ae_scripts/README.md); each `fig*/` directory has its
own README with the exact commands, hardware requirements, and expected
runtimes.

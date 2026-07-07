from __future__ import annotations

import torch

from .utils import cache_once, load_jit


@cache_once
def _jit_nccl_gin_module():
    return load_jit(
        "nccl_gin_trans",
        cuda_files=["kvcacheio/nccl_gin_trans.cuh"],
        cuda_wrappers=[
            ("hbm_to_remote_dram", "dlengine::nano::gin::hbm_to_remote_dram"),
            ("remote_dram_to_hbm", "dlengine::nano::gin::remote_dram_to_hbm"),
        ],
        extra_ldflags=["-lnccl"],
    )


def _anchor(device: torch.device | int | None = None) -> torch.Tensor:
    if device is None:
        device = torch.cuda.current_device()
    return torch.empty((), device=device)


def hbm_to_remote_dram(
    *,
    dev_comm: int,
    dev_window: int,
    host_window: int,
    nbytes: int,
    peer: int,
    device: torch.device | int | None = None,
) -> None:
    """Launch GPU-initiated RDMA write from local HBM window to peer HOST_NUMA window.

    The NCCL handles are opaque addresses owned by the caller. This thin JIT
    wrapper intentionally does not create or register NCCL resources; it only
    gives Python/HiSparse code a stable launch surface.
    """

    module = _jit_nccl_gin_module()
    module.hbm_to_remote_dram(
        int(dev_comm),
        int(dev_window),
        int(host_window),
        int(nbytes),
        int(peer),
        _anchor(device),
    )


def store_remote_hicache_region(
    *,
    dev_comm: int,
    local_hbm_window: int,
    remote_host_window: int,
    nbytes: int,
    peer: int,
    device: torch.device | int | None = None,
) -> None:
    """Store one contiguous local HBM KV region into a peer HOST_NUMA window."""

    hbm_to_remote_dram(
        dev_comm=dev_comm,
        dev_window=local_hbm_window,
        host_window=remote_host_window,
        nbytes=nbytes,
        peer=peer,
        device=device,
    )


def remote_dram_to_hbm(
    *,
    dev_comm: int,
    host_window: int,
    dev_window: int,
    nbytes: int,
    peer: int,
    device: torch.device | int | None = None,
) -> None:
    """Launch GPU-initiated RDMA read from peer HOST_NUMA window into local HBM window."""

    module = _jit_nccl_gin_module()
    module.remote_dram_to_hbm(
        int(dev_comm),
        int(host_window),
        int(dev_window),
        int(nbytes),
        int(peer),
        _anchor(device),
    )


def load_remote_hicache_region(
    *,
    dev_comm: int,
    remote_host_window: int,
    local_hbm_window: int,
    nbytes: int,
    peer: int,
    device: torch.device | int | None = None,
) -> None:
    """Load one contiguous remote HOST_NUMA KV region into a local HBM window."""

    remote_dram_to_hbm(
        dev_comm=dev_comm,
        host_window=remote_host_window,
        dev_window=local_hbm_window,
        nbytes=nbytes,
        peer=peer,
        device=device,
    )


__all__ = [
    "hbm_to_remote_dram",
    "load_remote_hicache_region",
    "remote_dram_to_hbm",
    "store_remote_hicache_region",
]

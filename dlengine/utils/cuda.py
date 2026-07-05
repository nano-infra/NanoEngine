"""CUDA device capability helpers."""

from __future__ import annotations

from typing import Union

import torch

CudaDevice = Union[torch.device, int, str]


def get_cuda_compute_capability(
    device: CudaDevice | None = None,
) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None

    try:
        return torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError):
        return None


def is_hopper(device: CudaDevice | None = None) -> bool:
    capability = get_cuda_compute_capability(device)
    return capability is not None and capability[0] == 9

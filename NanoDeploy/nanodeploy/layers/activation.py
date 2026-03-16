import os

import torch
import torch.nn.functional as F
from torch import nn


def _maybe_torch_compile(fn):
    """Apply torch.compile on CUDA; skip on Ascend NPU (inductor not supported)."""
    if os.environ.get("NANO_BACKEND", "") == "ascend":
        return fn
    try:
        import torch_npu
        if torch.npu.is_available():
            return fn
    except ImportError:
        pass
    return torch.compile(fn)


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @_maybe_torch_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

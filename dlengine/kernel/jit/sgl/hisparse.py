from __future__ import annotations

import torch

from .utils import cache_once, load_jit


@cache_once
def _jit_hisparse_module():
    return load_jit(
        "hisparse_ring_mapping",
        cuda_files=["hisparse/hisparse.cuh"],
        cuda_wrappers=[("build_ring_slot_mapping", "HiSparseRingMappingKernel::run")],
    )


def build_ring_slot_mapping(
    slots: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    num_real_reqs: torch.Tensor,
    max_num_seqs: int,
    tokens_per_seq: int,
) -> torch.Tensor:
    """Build graph-safe per-request ring-buffer physical slot indices."""
    _jit_hisparse_module().build_ring_slot_mapping(
        slots,
        positions,
        output,
        num_real_reqs,
        max_num_seqs,
        tokens_per_seq,
    )
    return output


__all__ = ["build_ring_slot_mapping"]

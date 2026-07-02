from dataclasses import dataclass

import torch

from dlengine.context_v2 import BaseContext


@dataclass
class HiSparseContext(BaseContext):
    enabled: bool = False
    max_num_seqs: int = 0
    dummy_slot: int = -1
    device_buffer_size: int = 0
    num_real_reqs: torch.Tensor | None = None

    @classmethod
    def get_context_type(cls) -> str:
        return "hisparse"

    @classmethod
    def get_context_name(cls) -> str:
        return "HiSparseContext"

    def clear_context(self) -> None:
        self.enabled = False
        self.max_num_seqs = 0
        self.dummy_slot = -1
        self.device_buffer_size = 0
        self.num_real_reqs = None

    def reset_context(self) -> None:
        self.clear_context()


_HISPARSE_CONTEXT = HiSparseContext()


def get_hisparse_context() -> HiSparseContext:
    return _HISPARSE_CONTEXT


def reset_hisparse_context() -> None:
    global _HISPARSE_CONTEXT
    _HISPARSE_CONTEXT = HiSparseContext()


def initialize_hisparse_context(max_num_seqs: int, device, device_buffer_size: int) -> HiSparseContext:
    ctx = get_hisparse_context()
    ctx.enabled = True
    ctx.max_num_seqs = max_num_seqs
    ctx.dummy_slot = max_num_seqs
    ctx.device_buffer_size = device_buffer_size
    ctx.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
    return ctx


def remap_slot_mapping(slot_mapping: torch.Tensor | None) -> torch.Tensor | None:
    # Phase 1 dummy-prefill keeps the full FP8 MLA cache resident on device, so
    # logical physical slots and HiSparse device slots are identical. The hook is
    # intentionally present so the model path is already wired for Phase 2.
    return slot_mapping


def remap_sparse_indices(sparse_indices: torch.Tensor | None) -> torch.Tensor | None:
    return sparse_indices


__all__ = [
    "HiSparseContext",
    "get_hisparse_context",
    "initialize_hisparse_context",
    "remap_slot_mapping",
    "remap_sparse_indices",
    "reset_hisparse_context",
]

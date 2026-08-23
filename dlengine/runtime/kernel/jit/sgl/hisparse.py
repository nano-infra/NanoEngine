from __future__ import annotations

import torch

from .utils import cache_once, load_jit


@cache_once
def _jit_hisparse_module():
    return load_jit(
        "hisparse_ring_mapping",
        cuda_files=["hisparse/hisparse.cuh"],
        cuda_wrappers=[
            ("build_ring_slot_mapping", "HiSparseRingMappingKernel::run"),
            ("load_mla_slot", "HiSparseMLASlotLoadKernel::run"),
            ("writeback_mla_slot", "HiSparseMLASlotWritebackKernel::run"),
        ],
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


def load_mla_slot(
    logical_indices: torch.Tensor,
    indices: torch.Tensor,
    request_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    resident_tokens: torch.Tensor,
    cold: torch.Tensor,
    hot: torch.Tensor,
    output: torch.Tensor,
    hot_output_slots: torch.Tensor,
    num_real_reqs: torch.Tensor,
    max_num_seqs: int,
    hot_capacity: int,
    slot_stride_tokens: int,
    union_hash_entries: torch.Tensor,
    union_hash_values: torch.Tensor,
    union_hash_capacity: int,
    num_tokens_per_seq: int,
    layer_id: int,
    phase_id: int,
) -> None:
    _jit_hisparse_module().load_mla_slot(
        logical_indices,
        indices,
        request_slots,
        seq_lens,
        resident_tokens,
        cold,
        hot,
        output,
        hot_output_slots,
        num_real_reqs,
        max_num_seqs,
        hot_capacity,
        slot_stride_tokens,
        union_hash_entries,
        union_hash_values,
        union_hash_capacity,
        num_tokens_per_seq,
        layer_id,
        phase_id,
    )


def writeback_mla_slot(
    logical_slots: torch.Tensor,
    hot_slots: torch.Tensor,
    hot: torch.Tensor,
    cold: torch.Tensor,
    num_real_reqs: torch.Tensor,
    num_tokens_per_seq: int,
) -> None:
    _jit_hisparse_module().writeback_mla_slot(
        logical_slots, hot_slots, hot, cold, num_real_reqs, num_tokens_per_seq
    )


__all__ = [
    "build_ring_slot_mapping",
    "load_mla_slot",
    "writeback_mla_slot",
]

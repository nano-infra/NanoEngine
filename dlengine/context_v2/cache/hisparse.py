from dataclasses import dataclass

import torch

from dlengine.context_v2 import BaseContext


@dataclass
class HiSparseContext(BaseContext):
    enabled: bool = False
    max_num_seqs: int = 0
    dummy_slot: int = -1
    device_buffer_size: int = 0
    tokens_per_seq: int = 0
    hot_kv_cache: torch.Tensor | None = None
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
        self.tokens_per_seq = 0
        self.hot_kv_cache = None
        self.num_real_reqs = None

    def reset_context(self) -> None:
        self.clear_context()


_HISPARSE_CONTEXT = HiSparseContext()


def get_hisparse_context() -> HiSparseContext:
    return _HISPARSE_CONTEXT


def reset_hisparse_context() -> None:
    global _HISPARSE_CONTEXT
    _HISPARSE_CONTEXT = HiSparseContext()


def initialize_hisparse_context(
    max_num_seqs: int, device, device_buffer_size: int
) -> HiSparseContext:
    ctx = get_hisparse_context()
    ctx.enabled = True
    ctx.max_num_seqs = max_num_seqs
    ctx.dummy_slot = max_num_seqs
    # ``device_buffer_size`` is a per-sequence capacity. Keeping this invariant
    # makes a configured sliding window independent of scheduler concurrency.
    ctx.device_buffer_size = device_buffer_size
    ctx.tokens_per_seq = device_buffer_size
    ctx.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
    return ctx


def allocate_gqa_hot_buffer(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor | None:
    ctx = get_hisparse_context()
    if not ctx.enabled or ctx.device_buffer_size <= 0:
        ctx.hot_kv_cache = None
        return None

    total_tokens = ctx.tokens_per_seq * max(1, ctx.max_num_seqs)
    ctx.hot_kv_cache = torch.empty(
        2,
        num_layers,
        total_tokens,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    return ctx.hot_kv_cache


def remap_slot_mapping(slot_mapping: torch.Tensor | None) -> torch.Tensor | None:
    # Phase 1 dummy-prefill keeps the full FP8 MLA cache resident on device, so
    # logical physical slots and HiSparse device slots are identical. The hook is
    # intentionally present so the model path is already wired for Phase 2.
    return slot_mapping


def build_hot_slot_mapping(
    hisparse_slots: torch.Tensor | None,
    positions: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    ctx = get_hisparse_context()
    if hisparse_slots is None or positions is None or ctx.tokens_per_seq <= 0:
        return None
    slots = hisparse_slots.to(torch.int64)
    pos = positions[: slots.numel()].to(torch.int64)
    if (
        slots.is_cuda
        and pos.is_cuda
        and ctx.num_real_reqs is not None
        and slots.dtype == torch.int64
        and pos.dtype == torch.int64
    ):
        if output is None:
            output = torch.empty(slots.numel(), dtype=torch.int32, device=slots.device)
        try:
            from dlengine.kernel.jit.sgl.hisparse import build_ring_slot_mapping

            return build_ring_slot_mapping(
                slots,
                pos,
                output[: slots.numel()],
                ctx.num_real_reqs,
                ctx.max_num_seqs,
                ctx.tokens_per_seq,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "tvm_ffi":
                raise
    hot = slots * ctx.tokens_per_seq + (pos % ctx.tokens_per_seq)
    hot = torch.where(slots < ctx.max_num_seqs, hot, -1)
    hot = hot.to(torch.int32)
    if output is not None:
        output[: hot.numel()].copy_(hot)
        return output[: hot.numel()]
    return hot


def remap_sparse_indices(sparse_indices: torch.Tensor | None) -> torch.Tensor | None:
    return sparse_indices


__all__ = [
    "HiSparseContext",
    "allocate_gqa_hot_buffer",
    "build_hot_slot_mapping",
    "get_hisparse_context",
    "initialize_hisparse_context",
    "remap_slot_mapping",
    "remap_sparse_indices",
    "reset_hisparse_context",
]

from dataclasses import dataclass

import torch

from dlengine.runtime.context import BaseContext


@dataclass
class HiSparseContext(BaseContext):
    enabled: bool = False
    max_num_seqs: int = 0
    dummy_slot: int = -1
    device_buffer_size: int = 0
    tokens_per_seq: int = 0
    hot_kv_cache: torch.Tensor | None = None
    cold_kv_cache: torch.Tensor | None = None
    block_size: int = 0
    hot_blocks_per_seq: int = 0
    resident_tokens: torch.Tensor | None = None
    slot_owner_ids: list[int] | None = None
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
        self.cold_kv_cache = None
        self.block_size = 0
        self.hot_blocks_per_seq = 0
        self.resident_tokens = None
        self.slot_owner_ids = None
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


def initialize_mla_hisparse_cache(
    cold_kv_cache: torch.Tensor,
    *,
    max_num_seqs: int,
    device_buffer_size: int,
) -> torch.Tensor:
    """Allocate the decode GPU hot tier using the MLA cache's exact layout.

    The first implementation repacks every layer's selected pages on demand.
    A request owns a fixed, disjoint hot-page range, which makes simultaneous
    batches safe without requiring a global allocator.
    """
    if cold_kv_cache is None or cold_kv_cache.ndim != 6:
        raise ValueError("MLA HiSparse requires a 6-D host cold KV cache")
    ctx = get_hisparse_context()
    block_size = int(cold_kv_cache.shape[3])
    # One extra page is reserved for the newly generated token, matching the
    # per-request HiSparse slot layout used by SGLang.
    slot_stride_tokens = device_buffer_size + block_size
    hot_blocks_per_seq = max(1, (slot_stride_tokens + block_size - 1) // block_size)
    total_hot_blocks = max_num_seqs * hot_blocks_per_seq

    # FP8 MLA uses one padding row between physical blocks. Mirror that layout
    # because FlashMLA derives its block stride from the tensor view.
    padded = torch.empty(
        cold_kv_cache.shape[0],
        cold_kv_cache.shape[1],
        total_hot_blocks,
        block_size + 1,
        cold_kv_cache.shape[4],
        cold_kv_cache.shape[5],
        dtype=cold_kv_cache.dtype,
        device=ctx.num_real_reqs.device,
    )
    ctx.cold_kv_cache = cold_kv_cache
    ctx.hot_kv_cache = padded[:, :, :, :block_size, :, :]
    ctx.block_size = block_size
    ctx.hot_blocks_per_seq = hot_blocks_per_seq
    ctx.tokens_per_seq = hot_blocks_per_seq * block_size
    ctx.resident_tokens = torch.zeros(
        cold_kv_cache.shape[1],
        max_num_seqs,
        ctx.device_buffer_size,
        dtype=torch.uint8,
        device=ctx.num_real_reqs.device,
    )
    ctx.slot_owner_ids = [-1] * max_num_seqs
    return ctx.hot_kv_cache


def stage_mla_sparse_indices(
    layer_idx: int,
    logical_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    hisparse_slots: torch.Tensor,
    output_slots: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load top-k tokens into their request slot using a graph-safe CUDA kernel."""
    ctx = get_hisparse_context()
    if ctx.hot_kv_cache is None or ctx.cold_kv_cache is None:
        raise RuntimeError("MLA HiSparse hot/cold cache is not initialized")
    if sparse_indices.ndim != 2 or hisparse_slots.numel() != sparse_indices.shape[0]:
        raise ValueError("HiSparse slots must contain one entry per sparse-index row")

    if sparse_indices.shape[1] > ctx.device_buffer_size:
        raise RuntimeError(
            f"HiSparse top-k={sparse_indices.shape[1]} exceeds slot capacity "
            f"{ctx.device_buffer_size}"
        )
    if not sparse_indices.is_cuda:
        raise RuntimeError("MLA HiSparse slot loading requires CUDA")

    from dlengine.runtime.kernel.jit.sgl.hisparse import load_mla_slot

    result = torch.empty_like(sparse_indices, dtype=torch.int32)
    hot_output_slots = torch.empty_like(output_slots, dtype=torch.int32)
    load_mla_slot(
        logical_indices.to(torch.int32),
        sparse_indices.to(torch.int32),
        hisparse_slots,
        seq_lens.to(torch.int32),
        ctx.resident_tokens[layer_idx],
        ctx.cold_kv_cache[0, layer_idx],
        ctx.hot_kv_cache[0, layer_idx],
        result,
        hot_output_slots,
        ctx.num_real_reqs,
        ctx.max_num_seqs,
        ctx.device_buffer_size,
        ctx.tokens_per_seq,
    )
    return result, hot_output_slots


def reset_mla_hisparse_slots(slots: list[int]) -> None:
    ctx = get_hisparse_context()
    if ctx.resident_tokens is None:
        return
    valid = sorted({int(slot) for slot in slots if 0 <= int(slot) < ctx.max_num_seqs})
    if valid:
        ctx.resident_tokens[:, valid] = 0


def update_mla_hisparse_slot_owners(slots: list[int], seq_ids: list[int]) -> None:
    """Clear residency when a scheduler slot is assigned to another sequence."""
    ctx = get_hisparse_context()
    if ctx.slot_owner_ids is None:
        return
    changed = []
    for slot, seq_id in zip(slots, seq_ids):
        slot = int(slot)
        seq_id = int(seq_id)
        if 0 <= slot < ctx.max_num_seqs and ctx.slot_owner_ids[slot] != seq_id:
            ctx.slot_owner_ids[slot] = seq_id
            changed.append(slot)
    reset_mla_hisparse_slots(changed)


def writeback_mla_output_pages(
    layer_idx: int,
    logical_output_slots: torch.Tensor,
    hot_output_slots: torch.Tensor,
) -> None:
    """Persist freshly appended decode KV from the hot tier to host cold KV."""
    ctx = get_hisparse_context()
    if ctx.hot_kv_cache is None or ctx.cold_kv_cache is None:
        return
    if not logical_output_slots.is_cuda:
        raise RuntimeError("MLA HiSparse writeback requires CUDA")
    from dlengine.runtime.kernel.jit.sgl.hisparse import writeback_mla_slot

    writeback_mla_slot(
        logical_output_slots.to(torch.int32),
        hot_output_slots.to(torch.int32),
        ctx.hot_kv_cache[0, layer_idx],
        ctx.cold_kv_cache[0, layer_idx],
        ctx.num_real_reqs,
    )


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
            from dlengine.runtime.kernel.jit.sgl.hisparse import build_ring_slot_mapping

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
    "initialize_mla_hisparse_cache",
    "remap_slot_mapping",
    "remap_sparse_indices",
    "reset_hisparse_context",
    "reset_mla_hisparse_slots",
    "stage_mla_sparse_indices",
    "update_mla_hisparse_slot_owners",
    "writeback_mla_output_pages",
]

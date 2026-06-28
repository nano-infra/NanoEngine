from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from dlengine.context_v2 import BaseContext


@dataclass
class BatchInContext:
    is_prefill: bool = False
    max_bs: int | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_dummy: bool = False
    num_tokens_per_seq: int = 1
    sampling_token_indices: torch.Tensor | None = None
    sampling_seq_indices: torch.Tensor | None = None

    # TODO(context_v2): move these backend-specific fields to their own
    # attention contexts after all call sites use batch_in / batch_out directly.
    gdn_conv_states: torch.Tensor | None = None
    gdn_recurrent_states: torch.Tensor | None = None
    gdn_state_slots: torch.Tensor | None = None
    dsv4_state_slots: torch.Tensor | None = None
    dsv4_compressed_block_tables: dict[int, torch.Tensor] | None = None


@dataclass
class BatchOutContext:
    token_ids: list[torch.Tensor] = field(default_factory=list)
    step_logprobs: list[torch.Tensor] | None = None

    # TODO(context_v2): move scheduler metadata to HSA/CSA/DSA context.
    tile_scheduler_metadata: Any = None
    sparse_tile_scheduler_metadata: Any = None

    # Used by MTP to force low-latency EP while attention still runs prefill.
    use_low_latency_ep: bool = False


_INPUT_FIELDS = set(BatchInContext.__dataclass_fields__)
_OUTPUT_FIELDS = set(BatchOutContext.__dataclass_fields__)


@dataclass
class BatchContext(BaseContext):
    batch_in: BatchInContext = field(default_factory=BatchInContext)
    batch_out: BatchOutContext = field(default_factory=BatchOutContext)

    @classmethod
    def get_context_type(cls) -> str:
        return "batch"

    @classmethod
    def get_context_name(cls) -> str:
        return "BatchContext"

    def __getattr__(self, name: str):
        if name in _INPUT_FIELDS:
            return getattr(self.batch_in, name)
        if name in _OUTPUT_FIELDS:
            return getattr(self.batch_out, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name in _INPUT_FIELDS:
            setattr(self.batch_in, name, value)
            return
        if name in _OUTPUT_FIELDS:
            setattr(self.batch_out, name, value)
            return
        object.__setattr__(self, name, value)

    def clear_context(self) -> None:
        self.__dict__.clear()
        object.__setattr__(self, "batch_in", BatchInContext())
        object.__setattr__(self, "batch_out", BatchOutContext())

    def reset_context(self) -> None:
        self.clear_context()


Context = BatchContext

_CONTEXT = BatchContext()


def get_batch_context() -> BatchContext:
    return _CONTEXT


def set_batch_context(
    is_prefill: bool,
    max_bs: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: torch.Tensor | None = None,
    block_tables: Optional[torch.Tensor] = None,
    is_dummy: bool = False,
    tile_scheduler_metadata: Any = None,
    sparse_tile_scheduler_metadata: Any = None,
    gdn_conv_states: Optional[torch.Tensor] = None,
    gdn_recurrent_states: Optional[torch.Tensor] = None,
    gdn_state_slots: Optional[torch.Tensor] = None,
    dsv4_state_slots: Optional[torch.Tensor] = None,
    dsv4_compressed_block_tables: Optional[dict[int, torch.Tensor]] = None,
    num_tokens_per_seq: int = 1,
    use_low_latency_ep: bool = False,
    sampling_token_indices: Optional[torch.Tensor] = None,
    sampling_seq_indices: Optional[torch.Tensor] = None,
) -> BatchContext:
    global _CONTEXT
    _CONTEXT = BatchContext(
        batch_in=BatchInContext(
            is_prefill=is_prefill,
            max_bs=max_bs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            is_dummy=is_dummy,
            num_tokens_per_seq=num_tokens_per_seq,
            sampling_token_indices=sampling_token_indices,
            sampling_seq_indices=sampling_seq_indices,
            gdn_conv_states=gdn_conv_states,
            gdn_recurrent_states=gdn_recurrent_states,
            gdn_state_slots=gdn_state_slots,
            dsv4_state_slots=dsv4_state_slots,
            dsv4_compressed_block_tables=dsv4_compressed_block_tables,
        ),
        batch_out=BatchOutContext(
            token_ids=[],
            step_logprobs=None,
            tile_scheduler_metadata=tile_scheduler_metadata,
            sparse_tile_scheduler_metadata=sparse_tile_scheduler_metadata,
            use_low_latency_ep=use_low_latency_ep,
        ),
    )
    return _CONTEXT


def reset_batch_context() -> None:
    global _CONTEXT
    _CONTEXT = BatchContext()


def get_context() -> BatchContext:
    return get_batch_context()


def set_context(*args, **kwargs) -> BatchContext:
    return set_batch_context(*args, **kwargs)


def reset_context() -> None:
    reset_batch_context()


__all__ = [
    "BatchContext",
    "BatchInContext",
    "BatchOutContext",
    "Context",
    "get_batch_context",
    "get_context",
    "reset_batch_context",
    "reset_context",
    "set_batch_context",
    "set_context",
]

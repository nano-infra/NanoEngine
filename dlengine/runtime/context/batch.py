from dataclasses import dataclass
from typing import Optional

import torch

from dlengine.runtime.context import BaseContext
from dlengine.runtime.context.graph import PagedAttentionStrategy


@dataclass
class BatchContext(BaseContext):
    is_prefill: bool = False
    max_bs: int | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    is_dummy: bool = False
    num_tokens_per_seq: int = 1
    sampling_token_indices: torch.Tensor | None = None
    sampling_seq_indices: torch.Tensor | None = None
    paged_attention_strategy: PagedAttentionStrategy | None = None
    graph_attention_strategy: PagedAttentionStrategy | None = None
    decode_page_plan_key: tuple[int, ...] | None = None
    mtp_draft_safe: bool = True

    # TODO(context): move these backend-specific fields to their own
    # attention contexts after all call sites use BatchContext directly.
    block_tables: torch.Tensor | None = None
    gdn_conv_states: torch.Tensor | None = None
    gdn_recurrent_states: torch.Tensor | None = None
    gdn_state_slots: torch.Tensor | None = None
    gdn_state_slots_i32: torch.Tensor | None = None
    dsv4_state_slots: torch.Tensor | None = None
    dsv4_compressed_block_tables: dict[int, torch.Tensor] | None = None
    hisparse_slots: torch.Tensor | None = None
    hisparse_slot_mapping: torch.Tensor | None = None
    hisparse_num_real_reqs: torch.Tensor | None = None
    indexer_schedule_meta: torch.Tensor | tuple[torch.Tensor, ...] | None = None

    @classmethod
    def get_context_type(cls) -> str:
        return "batch"

    @classmethod
    def get_context_name(cls) -> str:
        return "BatchContext"

    def clear_context(self) -> None:
        self.__dict__.clear()
        self.__dict__.update(BatchContext().__dict__)

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
    gdn_conv_states: Optional[torch.Tensor] = None,
    gdn_recurrent_states: Optional[torch.Tensor] = None,
    gdn_state_slots: Optional[torch.Tensor] = None,
    dsv4_state_slots: Optional[torch.Tensor] = None,
    dsv4_compressed_block_tables: Optional[dict[int, torch.Tensor]] = None,
    hisparse_slots: Optional[torch.Tensor] = None,
    hisparse_slot_mapping: Optional[torch.Tensor] = None,
    hisparse_num_real_reqs: Optional[torch.Tensor] = None,
    num_tokens_per_seq: int = 1,
    sampling_token_indices: Optional[torch.Tensor] = None,
    sampling_seq_indices: Optional[torch.Tensor] = None,
    paged_attention_strategy: PagedAttentionStrategy | None = None,
    graph_attention_strategy: PagedAttentionStrategy | None = None,
    decode_page_plan_key: tuple[int, ...] | None = None,
    mtp_draft_safe: bool = True,
) -> BatchContext:
    global _CONTEXT
    gdn_state_slots_i32 = None
    if gdn_state_slots is not None:
        gdn_state_slots_i32 = (
            gdn_state_slots
            if gdn_state_slots.dtype == torch.int32
            else gdn_state_slots.to(torch.int32)
        )
    _CONTEXT = BatchContext(
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
        paged_attention_strategy=paged_attention_strategy,
        graph_attention_strategy=graph_attention_strategy,
        decode_page_plan_key=decode_page_plan_key,
        mtp_draft_safe=mtp_draft_safe,
        gdn_conv_states=gdn_conv_states,
        gdn_recurrent_states=gdn_recurrent_states,
        gdn_state_slots=gdn_state_slots,
        gdn_state_slots_i32=gdn_state_slots_i32,
        dsv4_state_slots=dsv4_state_slots,
        dsv4_compressed_block_tables=dsv4_compressed_block_tables,
        hisparse_slots=hisparse_slots,
        hisparse_slot_mapping=hisparse_slot_mapping,
        hisparse_num_real_reqs=hisparse_num_real_reqs,
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
    "Context",
    "get_batch_context",
    "get_context",
    "reset_batch_context",
    "reset_context",
    "set_batch_context",
    "set_context",
]

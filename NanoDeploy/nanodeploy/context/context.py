from dataclasses import dataclass, field
from typing import Optional

import torch
from nanodeploy.logging import get_logger

# Initialize logger with NANODEPLOY namespace
logger = get_logger()


@dataclass
class Context:
    is_prefill: bool = False
    max_bs: int | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    global_context_lens: torch.Tensor | None = None

    # Decode Sequence Parallel
    q_mask: torch.Tensor | None = None
    res_lse_mask: torch.Tensor | None = None

    is_dummy: bool = False

    tile_scheduler_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None

    token_ids: list[torch.Tensor] = field(default_factory=list)

    q_slice_get: torch.Tensor | None = None
    q_slice_fill: torch.Tensor | None = None
    q_copy_mask: torch.Tensor | None = None
    res_slice_get_to_buffer_output: torch.Tensor | None = None
    res_slice_fill_to_buffer_output: torch.Tensor | None = None
    res_to_buffer_output_mask: Optional[torch.Tensor] = None
    res_slice_get_to_buffer_input: torch.Tensor | None = None
    res_slice_fill_to_buffer_input: torch.Tensor | None = None
    res_to_buffer_input_mask: Optional[torch.Tensor] = None
    attention_compute_bs: Optional[int] = None

    # used for all2all q transfer
    q_offsets: torch.Tensor | None = None
    context_lens_for_attn: torch.Tensor | None = None

    # GatedDeltaNet state buffers (for mixed attention models like Qwen3.5-MoE)
    # conv_states: [num_layers, num_slots, conv_dim, kernel_size]
    gdn_conv_states: torch.Tensor | None = None
    # recurrent_states: [num_layers, num_slots, num_v_heads, head_k_dim, head_v_dim]
    gdn_recurrent_states: torch.Tensor | None = None
    # Per-sequence GDN slot indices: [num_seqs], maps batch position i -> slot index
    gdn_state_slots: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    max_bs: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: torch.Tensor | None = None,
    block_tables: Optional[torch.Tensor] = None,
    global_context_lens: Optional[torch.Tensor] = None,
    q_mask: Optional[torch.Tensor] = None,
    res_lse_mask: Optional[torch.Tensor] = None,
    is_dummy: bool = False,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    q_slice_get: Optional[torch.Tensor] = None,
    q_slice_fill: Optional[torch.Tensor] = None,
    q_copy_mask: Optional[torch.Tensor] = None,
    res_slice_get_to_buffer_output: Optional[torch.Tensor] = None,
    res_slice_fill_to_buffer_output: Optional[torch.Tensor] = None,
    res_to_buffer_output_mask: Optional[torch.Tensor] = None,
    res_slice_get_to_buffer_input: Optional[torch.Tensor] = None,
    res_slice_fill_to_buffer_input: Optional[torch.Tensor] = None,
    res_to_buffer_input_mask: Optional[torch.Tensor] = None,
    attention_compute_bs: Optional[int] = None,
    q_offsets: Optional[torch.Tensor] = None,
    context_lens_for_attn: Optional[torch.Tensor] = None,
    gdn_conv_states: Optional[torch.Tensor] = None,
    gdn_recurrent_states: Optional[torch.Tensor] = None,
    gdn_state_slots: Optional[torch.Tensor] = None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        max_bs,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        global_context_lens,
        q_mask=q_mask,
        res_lse_mask=res_lse_mask,
        is_dummy=is_dummy,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        q_slice_get=q_slice_get,
        q_slice_fill=q_slice_fill,
        q_copy_mask=q_copy_mask,
        res_slice_get_to_buffer_output=res_slice_get_to_buffer_output,
        res_slice_fill_to_buffer_output=res_slice_fill_to_buffer_output,
        res_to_buffer_output_mask=res_to_buffer_output_mask,
        res_slice_get_to_buffer_input=res_slice_get_to_buffer_input,
        res_slice_fill_to_buffer_input=res_slice_fill_to_buffer_input,
        res_to_buffer_input_mask=res_to_buffer_input_mask,
        attention_compute_bs=attention_compute_bs,
        q_offsets=q_offsets,
        context_lens_for_attn=context_lens_for_attn,
        gdn_conv_states=gdn_conv_states,
        gdn_recurrent_states=gdn_recurrent_states,
        gdn_state_slots=gdn_state_slots,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

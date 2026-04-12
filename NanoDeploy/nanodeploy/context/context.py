from dataclasses import dataclass, field
from typing import Any, Optional

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

    is_dummy: bool = False

    # FlashMLASchedMeta (new API) or None; kernel manages internal tensors lazily.
    tile_scheduler_metadata: Any = None
    # Separate FlashMLASchedMeta for sparse decode (NSA V3.2); managed by graph runner.
    sparse_tile_scheduler_metadata: Any = None

    token_ids: list[torch.Tensor] = field(default_factory=list)

    # GatedDeltaNet state buffers (for mixed attention models like Qwen3.5-MoE)
    # conv_states: [num_layers, num_slots, conv_dim, kernel_size]
    gdn_conv_states: torch.Tensor | None = None
    # recurrent_states: [num_layers, num_slots, num_v_heads, head_v_dim, head_k_dim] (K-last)
    gdn_recurrent_states: torch.Tensor | None = None
    # Per-sequence GDN slot indices: [num_seqs], maps batch position i -> slot index
    gdn_state_slots: torch.Tensor | None = None

    # Number of tokens per sequence in decode mode (1 = normal decode,
    # 2 = lazy verify with [token_{K-1}, draft] per seq).
    num_tokens_per_seq: int = 1

    # Force low-latency (decode) EP path even when is_prefill=True.
    # Used by MTP: attention needs prefill mode (flash_attn_varlen) but MoE
    # must use low-latency dispatch for CUDAGraph compatibility.
    use_low_latency_ep: bool = False

    # Chunked prefill: indices into the Q (hidden_states) tensor for the last token of each
    # final-chunk sequence. Shape: [n_final]. None means all sequences are final chunks.
    sampling_token_indices: torch.Tensor | None = None
    # Which sequence index (0-based) each sampling_token_indices entry
    # corresponds to. Shape: [n_final]. None when sampling_token_indices is None.
    sampling_seq_indices: torch.Tensor | None = None


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
    is_dummy: bool = False,
    tile_scheduler_metadata: Any = None,
    sparse_tile_scheduler_metadata: Any = None,
    gdn_conv_states: Optional[torch.Tensor] = None,
    gdn_recurrent_states: Optional[torch.Tensor] = None,
    gdn_state_slots: Optional[torch.Tensor] = None,
    num_tokens_per_seq: int = 1,
    use_low_latency_ep: bool = False,
    sampling_token_indices: Optional[torch.Tensor] = None,
    sampling_seq_indices: Optional[torch.Tensor] = None,
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
        is_dummy=is_dummy,
        tile_scheduler_metadata=tile_scheduler_metadata,
        sparse_tile_scheduler_metadata=sparse_tile_scheduler_metadata,
        num_tokens_per_seq=num_tokens_per_seq,
        use_low_latency_ep=use_low_latency_ep,
        gdn_conv_states=gdn_conv_states,
        gdn_recurrent_states=gdn_recurrent_states,
        gdn_state_slots=gdn_state_slots,
        sampling_token_indices=sampling_token_indices,
        sampling_seq_indices=sampling_seq_indices,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

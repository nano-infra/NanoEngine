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
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

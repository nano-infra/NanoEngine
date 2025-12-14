from dataclasses import dataclass

import torch

from dlslime.buffer.intra.all_to_all_intra_ll_buffer import AllToAllIntraLLBuffer

from nanodeploy.logger import get_logger
from nanodeploy.worker.distributed import get_dist_context


logger = get_logger()


@dataclass
class SPContext:
    max_num_seqs: int

    head_size: int
    num_attention_heads: int

    dtype: torch.dtype

    rank: int
    sp_size: int

    q_buffer: AllToAllIntraLLBuffer | None = None
    res_buffer: AllToAllIntraLLBuffer | None = None
    lse_buffer: AllToAllIntraLLBuffer | None = None

    def __post_init__(self):

        self.msg_size = (self.head_size + 1) * self.num_attention_heads

        sp_rank = get_dist_context().attn_sp_rank
        sp_world_size = get_dist_context().attn_sp_world_size

        q_res_lse_buffer_size = AllToAllIntraLLBuffer.get_buffer_size_hint(
            sp_world_size,
            self.max_num_seqs,
            self.msg_size,
            self.dtype.itemsize,
        )

        self.q_buffer = AllToAllIntraLLBuffer(
            sp_world_size,
            self.max_num_seqs,
            sp_rank,
            sp_world_size,
            q_res_lse_buffer_size,
        )

        self.res_buffer = AllToAllIntraLLBuffer(
            1,
            self.max_num_seqs,
            sp_rank,
            sp_world_size,
            q_res_lse_buffer_size,
        )

        self.lse_buffer = AllToAllIntraLLBuffer(
            1,
            self.max_num_seqs,
            sp_rank,
            sp_world_size,
            q_res_lse_buffer_size,
        )

        self.q_buffer.connect_full_mesh(get_dist_context().attn_sp_group)
        self.res_buffer.connect_full_mesh(get_dist_context().attn_sp_group)
        self.lse_buffer.connect_full_mesh(get_dist_context().attn_sp_group)


_SP_CONTEXT: SPContext


def get_sp_context() -> SPContext:
    return _SP_CONTEXT


def set_sp_context(
    max_num_seqs: int,
    head_size: int,
    num_attention_heads: int,
    dtype: torch.dtype,
    rank: int,
    sp_size: int,
):
    global _SP_CONTEXT
    _SP_CONTEXT = SPContext(
        max_num_seqs, head_size, num_attention_heads, dtype, rank, sp_size
    )


def reset_sp_context():
    global _SP_CONTEXT
    raise AttributeError("SP Buffer Context is immutable")

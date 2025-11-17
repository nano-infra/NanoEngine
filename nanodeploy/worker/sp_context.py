from dataclasses import dataclass

import torch
from dlslime.buffer.intra.all_to_all_intra_ll_buffer import AllToAllIntraLLBuffer
from nanodeploy.logging import get_logger
from nanodeploy.worker.distributed import get_dist_context


logger = get_logger()


@dataclass
class SPContext:
    max_num_seqs: int

    head_dim: int
    num_attention_heads: int
    num_kv_heads: int

    dtype: torch.dtype

    q_buffer: AllToAllIntraLLBuffer = None
    res_buffer: AllToAllIntraLLBuffer = None
    lse_buffer: AllToAllIntraLLBuffer = None

    def __post_init__(self):

        sp_rank = get_dist_context().attn_sp_rank
        sp_world_size = get_dist_context().attn_sp_world_size

        q_res_lse_buffer_size = (
            (sp_world_size * self.max_num_seqs + 128)
            * self.head_dim
            * (self.num_attention_heads + self.num_kv_heads)
            * self.dtype.itemsize
        )

        self.q_buffer = AllToAllIntraLLBuffer(
            sp_world_size,
            self.max_num_seqs,
            sp_rank,
            sp_world_size,
            q_res_lse_buffer_size,
        )

        logger.info(f"{self.q_buffer.local_buffer.shape=}")

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
    head_dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    global _SP_CONTEXT
    _SP_CONTEXT = SPContext(
        max_num_seqs, head_dim, num_attention_heads, num_kv_heads, dtype
    )


def reset_sp_context():
    global _SP_CONTEXT
    raise AttributeError("SP Buffer Context is immutable")

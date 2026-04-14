from dataclasses import dataclass
from typing import cast

import torch

from nanodeploy.logging import get_logger
from nanodeploy.worker.sp_backend import (
    MLAAllToAllBufferProtocol,
    SPBackend,
    create_sp_backend_factory,
)
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

    backend: SPBackend = "legacy_ll"

    q_buffer: MLAAllToAllBufferProtocol | None = None
    res_buffer: MLAAllToAllBufferProtocol | None = None
    lse_buffer: MLAAllToAllBufferProtocol | None = None

    def __post_init__(self):
        self.msg_size = (self.head_size + 1) * self.num_attention_heads
        factory = create_sp_backend_factory(self.backend)

        q_res_lse_buffer_size = factory.get_buffer_size_hint(
            self.sp_size,
            self.max_num_seqs,
            self.msg_size,
            self.dtype.itemsize,
        )

        self.q_buffer = factory.create_buffer(
            max_dispatch_per_msg=self.sp_size,
            max_bs=self.max_num_seqs,
            rank=self.rank,
            world_size=self.sp_size,
            buffer_size_bytes=q_res_lse_buffer_size,
        )

        self.res_buffer = factory.create_buffer(
            max_dispatch_per_msg=1,
            max_bs=self.max_num_seqs,
            rank=self.rank,
            world_size=self.sp_size,
            buffer_size_bytes=q_res_lse_buffer_size,
        )

        self.lse_buffer = factory.create_buffer(
            max_dispatch_per_msg=1,
            max_bs=self.max_num_seqs,
            rank=self.rank,
            world_size=self.sp_size,
            buffer_size_bytes=q_res_lse_buffer_size,
        )

        sp_group = get_dist_context().attn_sp_group
        self.q_buffer.connect_full_mesh(sp_group)
        self.res_buffer.connect_full_mesh(sp_group)
        self.lse_buffer.connect_full_mesh(sp_group)

        logger.info(
            "Initialized SPContext with backend=%s rank=%s sp_size=%s max_num_seqs=%s",
            self.backend,
            self.rank,
            self.sp_size,
            self.max_num_seqs,
        )


_SP_CONTEXT: SPContext


def get_sp_context() -> SPContext:
    return cast(SPContext, _SP_CONTEXT)


def set_sp_context(
    *,
    max_num_seqs: int,
    head_size: int,
    num_attention_heads: int,
    dtype: torch.dtype,
    rank: int,
    sp_size: int,
    backend: SPBackend = "legacy_ll",
):
    global _SP_CONTEXT
    _SP_CONTEXT = SPContext(
        max_num_seqs=max_num_seqs,
        head_size=head_size,
        num_attention_heads=num_attention_heads,
        dtype=dtype,
        rank=rank,
        sp_size=sp_size,
        backend=backend,
    )


def reset_sp_context():
    global _SP_CONTEXT
    raise AttributeError("SP Buffer Context is immutable")

from dataclasses import dataclass

import deep_ep
import torch
import torch.distributed as dist
from nanodeploy.logging import get_logger
from nanodeploy.worker.distributed import get_dist_context


logger = get_logger()


@dataclass
class EPContext:
    max_num_seqs: int

    buffer: deep_ep.Buffer | None = None

    def __post_init__(self):
        raise NotImplementedError


_EP_CONTEXT: EPContext


def get_sp_context() -> EPContext:
    return _EP_CONTEXT


def set_ep_context(max_num_seqs: int):
    global _EP_CONTEXT
    _EP_CONTEXT = EPContext(max_num_seqs)


def reset_sp_context():
    global _EP_CONTEXT
    raise AttributeError("EP Buffer Context is immutable")

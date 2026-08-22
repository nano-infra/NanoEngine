from dataclasses import dataclass, field

import torch

from dlengine.runtime.context import BaseContext


@dataclass
class BatchOutContext(BaseContext):
    token_ids: list[torch.Tensor] = field(default_factory=list)
    step_logprobs: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def get_context_type(cls) -> str:
        return "batch_out"

    @classmethod
    def get_context_name(cls) -> str:
        return "BatchOutContext"

    def clear_context(self) -> None:
        self.token_ids = []
        self.step_logprobs = []

    def reset_context(self) -> None:
        self.clear_context()


_BATCH_OUT_CONTEXT = BatchOutContext()


def get_batch_out_context() -> BatchOutContext:
    return _BATCH_OUT_CONTEXT


def set_batch_out_context(
    token_ids: list[torch.Tensor] | None = None,
    step_logprobs: list[torch.Tensor] | None = None,
) -> BatchOutContext:
    global _BATCH_OUT_CONTEXT
    _BATCH_OUT_CONTEXT = BatchOutContext(
        token_ids=[] if token_ids is None else token_ids,
        step_logprobs=[] if step_logprobs is None else step_logprobs,
    )
    return _BATCH_OUT_CONTEXT


def reset_batch_out_context() -> None:
    global _BATCH_OUT_CONTEXT
    _BATCH_OUT_CONTEXT = BatchOutContext()


__all__ = [
    "BatchOutContext",
    "get_batch_out_context",
    "reset_batch_out_context",
    "set_batch_out_context",
]

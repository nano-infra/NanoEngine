import uuid
from itertools import count
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from nanodeploy.metrics import SequenceMetric

from nanodeploy._cpp import (
    BlockContext as _CppBlockContext,
    Sequence as _CppSequence,
    SequenceStatus as _CppSequenceStatus,
)
from nanodeploy.sampling_params import SamplingParams


class SequenceConfig(BaseModel):
    temperature: float | None = None


class Sequence(_CppSequence):
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        engine_id: str = "",
        master_sp_rank: int = 0,
    ):
        sampling_params = sampling_params or SamplingParams()
        super().__init__(
            token_ids,
            sampling_params.temperature,
            sampling_params.max_tokens,
            sampling_params.ignore_eos,
            engine_id,
            master_sp_rank,
        )


BlockContext = _CppBlockContext
SequenceStatus = _CppSequenceStatus

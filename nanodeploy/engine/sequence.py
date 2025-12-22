import uuid
from itertools import count
from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanodeploy.metrics import SequenceMetric

from nanodeploy.sampling_params import SamplingParams
from nanodeploy._cpp import (
    Sequence as _CppSequence,
    BlockContext as _CppBlockContext,
    SequenceStatus as _CppSequenceStatus,
)

class SequenceConfig(BaseModel):
    temperature: float | None = None

class Sequence(_CppSequence):
    block_size = 256
    counter = count()
    
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        engine_id: str | None = None,
        master_sp_rank: int = 0,
    ):
        sampling_params = sampling_params or SamplingParams()
        super().__init__(
            token_ids,
            sampling_params.temperature,
            sampling_params.max_tokens,
            sampling_params.ignore_eos,
            engine_id,
            master_sp_rank
        )

BlockContext = _CppBlockContext
SequenceStatus = _CppSequenceStatus


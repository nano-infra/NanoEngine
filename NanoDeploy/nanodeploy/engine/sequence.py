import uuid
from itertools import count
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from nanodeploy.metrics import SequenceMetric

from nanodeploy._cpp import (
    BlockContext as _CppBlockContext,
    SamplingParams as _CppSamplingParams,
    Sequence as _CppSequence,
    SequenceStatus as _CppSequenceStatus,
)
from nanodeploy.sampling_params import SamplingParams

Sequence = _CppSequence
BlockContext = _CppBlockContext
SequenceStatus = _CppSequenceStatus

"""
NanoSequence: Sequence management library for NanoDeploy
"""

try:
    from nanosequence._nanosequence_cpp import (
        BlockContext,
        BlockContextSlot,
        BlockIdList,
        BlockLocationList,
        DefaultIntDict,
        DefaultListDict,
        deserialize,
        SamplingParams,
        Sequence,
        SequenceMetric,
        SequenceStatus,
        serialize,
        set_log_level,
    )
except ImportError as e:
    # Provide more helpful error message
    import sys

    error_msg = (
        f"Failed to import nanosequence._nanosequence_cpp: {e}\n"
        f"Please ensure nanosequence is installed: pip install -e /path/to/NanoSequence\n"
        f"Python path: {sys.path}"
    )
    raise ImportError(error_msg) from e

# Also export enum values for convenience
__all__ = [
    "BlockContext",
    "BlockContextSlot",
    "BlockIdList",
    "BlockLocationList",
    "DefaultIntDict",
    "DefaultListDict",
    "SamplingParams",
    "Sequence",
    "SequenceMetric",
    "SequenceStatus",
    "deserialize",
    "serialize",
    "set_log_level",
]

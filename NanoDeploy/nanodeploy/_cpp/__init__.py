import importlib.util
import os
import sys

# Try to find the compiled module
# It should be named _nanodeploy_cpp.cp3x-win_amd64.pyd on Windows or .so on Linux
# We can just import it if it's in the path or in this directory

try:
    from nanodeploy._nanodeploy_cpp import *
except ImportError as e:
    # Propagate the error so that the caller can see why the import failed
    # This is crucial for debugging (e.g. missing dependencies, symbol errors)
    raise e

# Sequence-related types are now directly available from _nanodeploy_cpp
# (they are bound in module.cpp from NanoSequence bindings)
# Re-export them with convenient names
__all__ = [
    "BlockContext",
    "BlockContextSlot",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "SequenceMetric",
    "deserialize",  # renamed from deserialize_sequences for convenience
    "serialize",  # renamed from serialize for convenience
    "BlockIdList",
    "BlockLocationList",
    "DefaultIntDict",
    "DefaultListDict",
]

# Alias for backward compatibility
deserialize_sequences = deserialize
serialize_sequences = serialize

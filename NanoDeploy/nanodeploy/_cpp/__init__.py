import importlib.util
import os
import sys

from nanodeploy.config import Config
from nanodeploy.logging import get_logger


logger = get_logger("nanodeploy")


# Try to find the compiled module
# It should be named _nanodeploy_cpp.cp3x-win_amd64.pyd on Windows or .so on Linux
# We can just import it if it's in the path or in this directory


try:
    from nanodeploy._nanodeploy_cpp import *
except ImportError as e:
    # Propagate the error so that the caller can see why the import failed
    # This is crucial for debugging (e.g. missing dependencies, symbol errors)
    logger.error(f"Failed to import nanodeploy._nanodeploy_cpp: {e}")
    logger.error(
        f"Please check if the compiled module is in the path or in this directory"
    )
    logger.error(f"If not, please compile the module using the following command:")
    logger.error(f"pip install './NanoDeploy[nanodeploy]'")
    raise e


def init_scheduler(config: Config) -> Scheduler:
    return Scheduler(
        config.engine_id,
        config.loop_count,
        config.max_num_seqs,
        config.max_num_batched_tokens,
        config.max_model_len,
        config.eos,
        config.attention_dp,
        config.attention_sp,
        config.num_kvcache_blocks,
        config.kvcache_block_size,
        config.mode,
    )


__all__ = [
    "BlockContext",
    "BlockContextSlot",
    "Scheduler",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "SequenceMetric",
    "init_scheduler",
]

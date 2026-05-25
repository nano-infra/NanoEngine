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
    sched = Scheduler(
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

    # DSv4: configure per-compression-ratio compressed-cache pools.
    # The compress_ratios attribute lives on the HF config; pools are sized
    # either from explicit Config overrides (--dsv4_compressed_pool_pages_*)
    # or from the worst-case default (max_num_seqs * max_model_len / ratio).
    if config.hf_config.architectures[0] == "DeepseekV4ForCausalLM":
        compress_ratios = getattr(config.hf_config, "compress_ratios", None) or []
        unique_ratios = sorted({r for r in compress_ratios if r > 0})
        if unique_ratios:
            page_size = 2  # 16-byte alignment for flash_mla 128-bit loads
            cfgs = []
            for ratio in unique_ratios:
                # Worst-case: max_num_seqs * (max_model_len / ratio) tokens.
                max_compressed = (config.max_model_len // ratio + 63) // 64 * 64
                max_blocks_per_seq = (max_compressed + page_size - 1) // page_size
                worst_case_pages = config.max_num_seqs * max_blocks_per_seq
                override = 0
                if ratio == 4:
                    override = config.dsv4_compressed_pool_pages_ratio4
                elif ratio == 128:
                    override = config.dsv4_compressed_pool_pages_ratio128
                num_pages = override if override > 0 else worst_case_pages
                cfgs.append(
                    CompressedPoolConfig(
                        ratio=ratio,
                        num_pages=num_pages,
                        page_size=page_size,
                        max_blocks_per_seq=max_blocks_per_seq,
                    )
                )
            sched.configure_compressed_pools(cfgs)
            logger.info(
                f"DSv4: configured compressed pools: "
                + ", ".join(
                    f"ratio={c.ratio} pages={c.num_pages} max_blocks={c.max_blocks_per_seq}"
                    for c in cfgs
                )
            )
    return sched


__all__ = [
    "BlockContext",
    "BlockContextSlot",
    "CompressedPoolConfig",
    "Scheduler",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "SequenceMetric",
    "init_scheduler",
]

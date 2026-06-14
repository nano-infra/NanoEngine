import importlib.util
import os
import sys
from typing import Any

from dlengine.config import Config
from dlengine.logging import get_logger

logger = get_logger("dlengine")


# Try to find the compiled module
# It should be named _dlengine_cpp.cp3x-win_amd64.pyd on Windows or .so on Linux
# We can just import it if it's in the path or in this directory


try:
    from dlengine._dlengine_cpp import *
except ImportError as e:
    # Propagate the error so that the caller can see why the import failed
    # This is crucial for debugging (e.g. missing dependencies, symbol errors)
    logger.error(f"Failed to import dlengine._dlengine_cpp: {e}")
    logger.error(
        f"Please check if the compiled module is in the path or in this directory"
    )
    logger.error(f"If not, please compile the module using the following command:")
    logger.error(f"pip install './DLEngine[dlengine]'")
    raise e


def init_scheduler(config: Config) -> Scheduler:
    sched = Scheduler(
        config.engine_id,
        config.num_speculative_tokens,
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

    # Apply the configured DP routing strategy (default RoundRobin). This is a
    # public, runtime-mutable field on the scheduler; the strategy object is
    # (re)built lazily on the next scheduling pass.
    sched.routing_strategy = RoutingStrategy[config.routing_strategy]

    # Linear-attention / GatedDeltaNet models: cross-request prefix caching is
    # unsafe because the recurrent (conv + state) cache is not captured by the
    # shared KV blocks. A prefix-cache hit would skip recomputing those tokens
    # and start the linear-attention state from a stale slot, corrupting output
    # non-deterministically under concurrency. Disable it so every request
    # recomputes its full prompt from a zero state.
    is_linear_attention = _has_linear_attention(config.hf_config)
    if is_linear_attention:
        sched.set_prefix_caching_enabled(False)
        logger.info(
            "Linear-attention model detected (layer_types contains "
            "'linear_attention'): disabled cross-request prefix caching to "
            "keep the GatedDeltaNet recurrent state correct."
        )

    # Session-scoped GatedDeltaNet state caching: retain a finished turn's KV
    # blocks + GDN recurrent-state slot and reuse them on the next turn from the
    # same session as a chunked-prefill continuation.
    #
    # MUTED: correct GDN continuation needs the next turn to be a token-exact
    # extension of the parked context, but agent clients (e.g. Claude Code)
    # re-render each turn — the assistant generation prompt injects a transient
    # ``<think>`` that vanishes once the turn becomes history, and tool/system
    # blocks are periodically rewritten — so the parked context is essentially
    # never an exact prefix and adoption always rejects. We therefore keep the
    # feature off regardless of the ``--gdn_state_cache_slots`` flag. The C++
    # machinery stays in place (inert) for workloads that append verbatim.
    cache_slots = max(0, getattr(config, "gdn_state_cache_slots", 0))
    if cache_slots > 0:
        logger.warning(
            "gdn_state_cache_slots=%d requested but session-scoped GDN state "
            "caching is muted: it requires token-exact prompt continuation, "
            "which re-rendering agent clients do not provide. Ignoring.",
            cache_slots,
        )
    return sched


def _has_linear_attention(hf_config: Any) -> bool:
    """Whether ``hf_config`` (or any nested sub-config) declares a
    ``linear_attention`` layer.

    Hybrid models such as Qwen3.5-MoE nest ``layer_types`` under a sub-config
    (e.g. ``text_config``) rather than at the top level, so we must walk nested
    PretrainedConfig children -- otherwise the linear-attention guard silently
    misses and cross-request prefix caching stays (unsafely) enabled.
    """
    seen: set[int] = set()

    def visit(cfg: Any) -> bool:
        if cfg is None or id(cfg) in seen:
            return False
        seen.add(id(cfg))
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types and any(lt == "linear_attention" for lt in layer_types):
            return True
        # Recurse into nested PretrainedConfig children (text_config, etc.).
        sub = getattr(cfg, "sub_configs", None)
        names = list(sub.keys()) if isinstance(sub, dict) else []
        for name in ("text_config", "thinker_config", "decoder_config"):
            if name not in names:
                names.append(name)
        for name in names:
            child = getattr(cfg, name, None)
            if hasattr(child, "to_dict") or getattr(child, "layer_types", None):
                if visit(child):
                    return True
        return False

    return visit(hf_config)


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

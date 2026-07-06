import json

from dlengine._rust.config import CachePlan, RoutingStrategy, SchedulerConfig
from dlengine._rust.core import Scheduler as RustScheduler
from dlengine.config import Config
from dlengine.context_v2.cache.plan import (
    deepseek_mla_cache_plan,
    gqa_cache_plan,
    gqa_hisparse_cache_plan,
    hca_csa_cache_plan,
    qwen35_cache_plan,
)
from dlengine.logging import get_logger
from dlengine.models.trait import has_gdn_component, has_hca_csa_cache

logger = get_logger("dlengine")


def init_scheduler(config: Config) -> RustScheduler:
    cache_plan = ensure_cache_plan(config)
    scheduler_config = build_scheduler_config(config, cache_plan)
    return RustScheduler(scheduler_config)


def build_scheduler_config(
    config: Config, cache_plan: CachePlan | None = None
) -> SchedulerConfig:
    cache_plan = cache_plan or ensure_cache_plan(config)
    return SchedulerConfig(
        engine_id=config.engine_id or "",
        num_speculative_tokens=config.num_speculative_tokens,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_model_len=config.max_model_len,
        eos_ids=config.eos,
        attention_dp=config.attention_dp,
        group_size=config.attention_sp,
        num_kvcache_blocks=config.num_kvcache_blocks,
        num_host_kvcache_blocks=config.num_host_kvcache_blocks,
        kvcache_block_size=config.kvcache_block_size,
        mode=config.mode,
        routing_strategy=RoutingStrategy[config.routing_strategy],
        gdn_state_cache_slots=max(0, getattr(config, "gdn_state_cache_slots", 0)),
        enable_prefix_cache=bool(getattr(config, "enable_prefix_cache", True)),
        cache_plan=cache_plan,
    )


def ensure_cache_plan(config: Config) -> CachePlan:
    cache_plan = getattr(config, "cache_plan", None)
    if cache_plan is None:
        cache_plan, reason = _build_cache_plan_with_reason(config)
        config.cache_plan = cache_plan
        logger.info(
            "Selected cache plan: %s, reason=%s",
            _format_cache_plan(cache_plan, config),
            reason,
        )
    else:
        logger.debug(
            "Using preconfigured cache plan: %s", _format_cache_plan(cache_plan, config)
        )
    return cache_plan


def build_cache_plan(config: Config) -> CachePlan:
    cache_plan, _ = _build_cache_plan_with_reason(config)
    return cache_plan


def _build_cache_plan_with_reason(config: Config) -> tuple[CachePlan, str]:
    hf_config = config.hf_config
    arch = (getattr(hf_config, "architectures", None) or [""])[0]

    if has_hca_csa_cache(hf_config):
        plan = hca_csa_cache_plan()
        _configure_hca_csa_cache_plan(plan, config)
        return plan, "hf_config.compress_ratios declares HCA/CSA compressed cache"

    if has_gdn_component(hf_config):
        return qwen35_cache_plan(), "layer_types contains linear_attention"

    if arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration") and bool(
        getattr(config, "enable_hisparse", False)
    ):
        return (
            gqa_hisparse_cache_plan(),
            "Gemma4 enable_hisparse requires SWA hot-buffer cache",
        )

    kv_lora_rank = getattr(hf_config, "kv_lora_rank", 0) or 0
    qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 0) or 0
    if kv_lora_rank or qk_rope_head_dim:
        use_indexer = getattr(hf_config, "index_head_dim", 0) > 0 and not getattr(
            config, "disable_nsa", False
        )
        return (
            deepseek_mla_cache_plan(
                use_indexer=use_indexer,
                use_hisparse=bool(getattr(config, "enable_hisparse", False)),
            ),
            (
                "MLA config detected "
                f"(kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
                f"use_indexer={use_indexer}, enable_hisparse={bool(getattr(config, 'enable_hisparse', False))})"
            ),
        )

    return gqa_cache_plan(), "default GQA cache"


def _format_cache_plan(cache_plan: CachePlan, config: Config) -> str:
    hf_config = config.hf_config
    architectures = getattr(hf_config, "architectures", None) or []
    architecture = architectures[0] if architectures else "unknown"
    payload = json.loads(cache_plan.to_json())
    payload["architecture"] = architecture
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _configure_hca_csa_cache_plan(plan: CachePlan, config: Config) -> None:
    compress_ratios = getattr(config.hf_config, "compress_ratios", None) or []
    unique_ratios = sorted({r for r in compress_ratios if r > 0})

    page_size = 2  # 16-byte alignment for flash_mla 128-bit loads
    for ratio in unique_ratios:
        max_compressed = (config.max_model_len // ratio + 63) // 64 * 64
        max_blocks_per_seq = (max_compressed + page_size - 1) // page_size
        worst_case_pages = config.max_num_seqs * max_blocks_per_seq
        override = 0
        if ratio == 4:
            override = config.dsv4_compressed_pool_pages_ratio4
        elif ratio == 128:
            override = config.dsv4_compressed_pool_pages_ratio128
        spec = plan.hca if ratio >= 64 else plan.csa
        spec.compression_ratio = ratio
        spec.num_pages = override if override > 0 else worst_case_pages
        spec.page_size = page_size
        spec.max_blocks_per_seq = max_blocks_per_seq
        if ratio >= 64:
            plan.hca = spec
        else:
            plan.csa = spec


def Scheduler(config: Config) -> RustScheduler:
    return init_scheduler(config)


__all__ = [
    "RoutingStrategy",
    "Scheduler",
    "build_cache_plan",
    "build_scheduler_config",
    "ensure_cache_plan",
    "init_scheduler",
]

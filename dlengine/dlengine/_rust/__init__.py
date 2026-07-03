from dlengine.logging import get_logger

logger = get_logger("dlengine")

try:
    from dlengine._dlengine_rust import *
except ImportError as e:
    logger.error(f"Failed to import dlengine._dlengine_rust: {e}")
    logger.error("Build it with: maturin develop")
    raise e


__all__ = [
    "CachePlan",
    "CachePlanFlag",
    "CsaCacheSpec",
    "GqaCacheSpec",
    "GdnCacheSpec",
    "HcaCacheSpec",
    "HiSparseCacheSpec",
    "IndexerCacheSpec",
    "BlockContextSlot",
    "BlockContext",
    "BlockManager",
    "MlaCacheSpec",
    "RoutingStrategy",
    "ScheduleResult",
    "Scheduler",
    "SchedulerMetricSnapshot",
    "SchedulerConfig",
    "SamplingParams",
    "Sequence",
    "SequenceMetric",
    "SequenceStatus",
    "ServerMetric",
    "StepMetricSnapshot",
    "cache_plan_flag",
    "decode_run_result",
    "deserialize",
    "encode_run_result",
    "extract_aux_from_bytes",
    "extract_vision_slots_from_bytes",
    "parse_migrate_batch",
    "prepare_decode_from_bytes",
    "prepare_prefill_from_bytes",
    "serialize",
    "serialize_dummy_run_batch",
    "serialize_migrate_batch",
    "serialize_run_batch",
]

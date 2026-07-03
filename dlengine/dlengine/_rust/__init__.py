from dlengine.logging import get_logger

logger = get_logger("dlengine")

try:
    from dlengine import _dlengine_rust as _native
except ImportError as e:
    logger.error(f"Failed to import dlengine._dlengine_rust: {e}")
    logger.error("Build it with: maturin develop")
    raise e


CachePlan = _native.CachePlan
CachePlanFlag = _native.CachePlanFlag
CsaCacheSpec = _native.CsaCacheSpec
GqaCacheSpec = _native.GqaCacheSpec
GdnCacheSpec = _native.GdnCacheSpec
HcaCacheSpec = _native.HcaCacheSpec
HiSparseCacheSpec = _native.HiSparseCacheSpec
IndexerCacheSpec = _native.IndexerCacheSpec
MlaCacheSpec = _native.MlaCacheSpec

BlockContext = _native.BlockContext
BlockContextSlot = _native.BlockContextSlot
BlockManager = _native.BlockManager

RoutingStrategy = _native.RoutingStrategy
ScheduleResult = _native.ScheduleResult
Scheduler = _native.Scheduler
SchedulerConfig = _native.SchedulerConfig
SchedulerMetricSnapshot = _native.SchedulerMetricSnapshot
StepMetricSnapshot = _native.StepMetricSnapshot

SamplingParams = _native.SamplingParams
SequenceMetric = _native.SequenceMetric
SequenceStatus = _native.SequenceStatus
ServerMetric = _native.ServerMetric

cache_plan_flag = _native.cache_plan_flag
decode_migration_metadata = _native.decode_migration_metadata
decode_run_result = _native.decode_run_result
encode_add_request = _native.encode_add_request
encode_run_result = _native.encode_run_result
extract_aux_from_bytes = _native.extract_aux_from_bytes
extract_vision_slots_from_bytes = _native.extract_vision_slots_from_bytes
parse_migrate_batch = _native.parse_migrate_batch
prepare_decode_from_bytes = _native.prepare_decode_from_bytes
prepare_prefill_from_bytes = _native.prepare_prefill_from_bytes
serialize_dummy_run_batch = _native.serialize_dummy_run_batch
set_log_level = _native.set_log_level

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
    "SequenceMetric",
    "SequenceStatus",
    "ServerMetric",
    "StepMetricSnapshot",
    "cache_plan_flag",
    "decode_migration_metadata",
    "decode_run_result",
    "encode_add_request",
    "encode_run_result",
    "extract_aux_from_bytes",
    "extract_vision_slots_from_bytes",
    "parse_migrate_batch",
    "prepare_decode_from_bytes",
    "prepare_prefill_from_bytes",
    "serialize_dummy_run_batch",
    "set_log_level",
]

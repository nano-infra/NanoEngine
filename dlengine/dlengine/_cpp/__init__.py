"""Compatibility import path for the Rust runtime.

Historically Python imported pybind11 symbols from ``dlengine._cpp``. The C++
runtime has been replaced by the Rust/PyO3 extension, but keeping this package
lets existing Python code migrate without touching every import site at once.
"""

from dlengine._rust import *

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
    "SchedulerConfig",
    "SchedulerMetricSnapshot",
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
    "set_log_level",
]

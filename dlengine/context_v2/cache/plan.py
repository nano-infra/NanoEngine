"""Model-declared cache requirements."""

from __future__ import annotations

from dlengine._rust.config import cache_plan_flag, CachePlan, CachePlanFlag


def cache_plan(
    flags: tuple[CachePlanFlag, ...],
) -> CachePlan:
    raw_flags = 0
    for flag in flags:
        raw_flags |= cache_plan_flag(flag)
    return CachePlan(flags=raw_flags)


def gqa_cache_plan() -> CachePlan:
    return cache_plan(flags=(CachePlanFlag.Gqa,))


def qwen35_cache_plan() -> CachePlan:
    return cache_plan(flags=(CachePlanFlag.Gqa, CachePlanFlag.Gdn))


def deepseek_mla_cache_plan(
    *, use_indexer: bool = False, use_hisparse: bool = False
) -> CachePlan:
    flags = [CachePlanFlag.Mla]
    if use_indexer:
        flags.append(CachePlanFlag.Indexer)
    if use_hisparse:
        flags.append(CachePlanFlag.Hisparse)
    return cache_plan(flags=tuple(flags))


def hca_csa_cache_plan() -> CachePlan:
    return cache_plan(flags=(CachePlanFlag.Hca, CachePlanFlag.Csa))


def deepseek_v4_cache_plan() -> CachePlan:
    return hca_csa_cache_plan()


__all__ = [
    "CachePlan",
    "CachePlanFlag",
    "cache_plan",
    "deepseek_mla_cache_plan",
    "deepseek_v4_cache_plan",
    "gqa_cache_plan",
    "hca_csa_cache_plan",
    "qwen35_cache_plan",
]

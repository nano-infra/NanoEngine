"""Model-declared cache requirements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePlan:
    primary: str
    components: tuple[str, ...]

    def has(self, component: str) -> bool:
        return component in self.components


def gqa_cache_plan() -> CachePlan:
    return CachePlan(primary="gqa", components=("gqa",))


def qwen35_cache_plan() -> CachePlan:
    return CachePlan(primary="gqa", components=("gqa", "gdn"))


def deepseek_mla_cache_plan(
    *, use_indexer: bool = False, use_hisparse: bool = False
) -> CachePlan:
    components = ["mla"]
    if use_indexer:
        components.append("indexer")
    if use_hisparse:
        components.append("hisparse")
    return CachePlan(primary="mla", components=tuple(components))


def deepseek_v4_cache_plan() -> CachePlan:
    return CachePlan(primary="dsv4", components=("hca", "csa"))


__all__ = [
    "CachePlan",
    "deepseek_mla_cache_plan",
    "deepseek_v4_cache_plan",
    "gqa_cache_plan",
    "qwen35_cache_plan",
]

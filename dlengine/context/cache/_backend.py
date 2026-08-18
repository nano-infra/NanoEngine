"""Explicit primary-cache backend selection.

The supported primary cache layouts are a closed set. Resolve them directly
instead of relying on importing implementation modules for registration side
effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class CacheKind(StrEnum):
    GQA = "gqa"
    MLA = "mla"
    DSV4 = "dsv4"


@dataclass(frozen=True)
class CacheBackend:
    kind: CacheKind
    configure: Callable
    get_block_bytes: Callable
    allocate: Callable


def resolve_cache_backend(kind: CacheKind | str) -> CacheBackend:
    """Return the implementation for a supported primary cache layout."""
    try:
        resolved_kind = CacheKind(kind)
    except ValueError as exc:
        known = ", ".join(item.value for item in CacheKind)
        raise ValueError(
            f"Unknown cache mode: {kind}. Supported modes: {known}"
        ) from exc

    # Keep implementation imports lazy to avoid a cycle through CacheContext,
    # while making every possible dispatch target visible in this function.
    if resolved_kind is CacheKind.GQA:
        from dlengine.context.cache.gqa import GQA_CACHE_BACKEND

        return GQA_CACHE_BACKEND
    if resolved_kind is CacheKind.MLA:
        from dlengine.context.cache.mla import MLA_CACHE_BACKEND

        return MLA_CACHE_BACKEND
    if resolved_kind is CacheKind.DSV4:
        from dlengine.context.cache.hca import DSV4_CACHE_BACKEND

        return DSV4_CACHE_BACKEND
    raise AssertionError(f"Unhandled cache kind: {resolved_kind}")


def supported_cache_kinds() -> tuple[CacheKind, ...]:
    return tuple(CacheKind)


__all__ = [
    "CacheBackend",
    "CacheKind",
    "resolve_cache_backend",
    "supported_cache_kinds",
]

"""Cache backend registry used by CacheContext facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CacheBackend:
    name: str
    configure: Callable
    get_block_bytes: Callable
    allocate: Callable


_CACHE_BACKENDS: dict[str, CacheBackend] = {}


def register_cache_backend(
    name: str,
    *,
    configure: Callable,
    get_block_bytes: Callable,
    allocate: Callable,
) -> CacheBackend:
    backend = CacheBackend(
        name=name,
        configure=configure,
        get_block_bytes=get_block_bytes,
        allocate=allocate,
    )
    _CACHE_BACKENDS[name] = backend
    return backend


def get_cache_backend(name: str) -> CacheBackend:
    try:
        return _CACHE_BACKENDS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_CACHE_BACKENDS)) or "<none>"
        raise ValueError(
            f"Unknown cache mode: {name}. Registered modes: {known}"
        ) from exc


def registered_cache_backends() -> dict[str, CacheBackend]:
    return dict(_CACHE_BACKENDS)


__all__ = [
    "CacheBackend",
    "get_cache_backend",
    "register_cache_backend",
    "registered_cache_backends",
]

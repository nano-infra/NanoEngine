from dataclasses import dataclass
from typing import Any

from dlengine.context_v2 import BaseContext


@dataclass
class DSAContext(BaseContext):
    sparse_tile_scheduler_metadata: Any = None

    @classmethod
    def get_context_type(cls) -> str:
        return "dsa"

    @classmethod
    def get_context_name(cls) -> str:
        return "DSAContext"

    def clear_context(self) -> None:
        self.sparse_tile_scheduler_metadata = None

    def reset_context(self) -> None:
        self.clear_context()


_DSA_CONTEXT = DSAContext()


def get_dsa_context() -> DSAContext:
    return _DSA_CONTEXT


def set_dsa_context(sparse_tile_scheduler_metadata: Any = None) -> DSAContext:
    global _DSA_CONTEXT
    _DSA_CONTEXT = DSAContext(
        sparse_tile_scheduler_metadata=sparse_tile_scheduler_metadata
    )
    return _DSA_CONTEXT


def reset_dsa_context() -> None:
    global _DSA_CONTEXT
    _DSA_CONTEXT = DSAContext()


__all__ = [
    "DSAContext",
    "get_dsa_context",
    "reset_dsa_context",
    "set_dsa_context",
]

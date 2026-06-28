from dataclasses import dataclass
from typing import Any

from dlengine.context_v2 import BaseContext


@dataclass
class HSAContext(BaseContext):
    tile_scheduler_metadata: Any = None

    @classmethod
    def get_context_type(cls) -> str:
        return "hsa"

    @classmethod
    def get_context_name(cls) -> str:
        return "HSAContext"

    def clear_context(self) -> None:
        self.tile_scheduler_metadata = None

    def reset_context(self) -> None:
        self.clear_context()


_HSA_CONTEXT = HSAContext()


def get_hsa_context() -> HSAContext:
    return _HSA_CONTEXT


def set_hsa_context(tile_scheduler_metadata: Any = None) -> HSAContext:
    global _HSA_CONTEXT
    _HSA_CONTEXT = HSAContext(tile_scheduler_metadata=tile_scheduler_metadata)
    return _HSA_CONTEXT


def reset_hsa_context() -> None:
    global _HSA_CONTEXT
    _HSA_CONTEXT = HSAContext()


__all__ = [
    "HSAContext",
    "get_hsa_context",
    "reset_hsa_context",
    "set_hsa_context",
]

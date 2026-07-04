"""Native PyO3 extension loader and symbol export helper."""

from __future__ import annotations

from dlengine.logging import get_logger

logger = get_logger("dlengine")

try:
    from dlengine import _engine as _native
except ImportError as e:
    logger.error(f"Failed to import dlengine._engine: {e}")
    logger.error("Build it with: maturin develop")
    raise e


def export(names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(_native, name) for name in names}

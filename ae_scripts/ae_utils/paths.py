"""Machine-specific paths, loaded from ``paths.env``.

``paths.env`` is the only file in this tree that may contain absolute machine
paths. Every script resolves its external dependencies through :func:`get` or
:func:`require` so that a missing entry fails loudly instead of silently
falling back to another host's data.

Shell scripts read the same file directly::

    source "$AE_ROOT/paths.env"
"""

from __future__ import annotations

import os
from pathlib import Path


AE_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(os.environ.get("AE_PATHS_ENV", AE_ROOT / "paths.env"))


def _load() -> dict[str, str]:
    """Parse ``paths.env`` into a mapping, ignoring comments and blank lines."""
    if not ENV_FILE.is_file():
        raise SystemExit(
            f"{ENV_FILE} not found.\n"
            f"Copy paths.env.example to paths.env and fill in the paths for "
            f"this host, or point AE_PATHS_ENV at an existing file."
        )

    values: dict[str, str] = {}
    for lineno, raw in enumerate(ENV_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise SystemExit(f"{ENV_FILE}:{lineno}: expected KEY=VALUE, got {raw!r}")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_VALUES = _load()


def get(key: str, default: str | None = None) -> str | None:
    """Return the configured value for ``key``.

    Environment variables take precedence over ``paths.env``, so a single
    experiment can override one path without editing the file.
    """
    return os.environ.get(key) or _VALUES.get(key) or default


def require(key: str) -> str:
    """Return the configured value for ``key`` or fail with actionable guidance."""
    value = get(key)
    if value:
        return value
    known = "\n".join(f"  {name}" for name in sorted(_VALUES))
    raise SystemExit(
        f"{key} is not configured.\n"
        f"Add it to {ENV_FILE}, or export it in the environment.\n"
        f"Configured keys:\n{known}"
    )


def require_path(key: str) -> Path:
    """Like :func:`require` but returns a :class:`~pathlib.Path`."""
    return Path(require(key))
"""Shared plotting defaults for artifact figures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager


FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_PATHS = (
    FONT_DIR / "LinBiolinum_R.otf",
    FONT_DIR / "LinBiolinum_RI.otf",
    FONT_DIR / "LinBiolinum_RB.otf",
)
FALLBACK_FONT = "DejaVu Sans"


@lru_cache(maxsize=1)
def get_plot_font_family() -> str:
    """Return Linux Biolinum when usable, otherwise DejaVu Sans."""
    try:
        for font_path in FONT_PATHS:
            font_manager.fontManager.addfont(str(font_path))
        return font_manager.FontProperties(fname=str(FONT_PATHS[0])).get_name()
    except (OSError, RuntimeError):
        return FALLBACK_FONT

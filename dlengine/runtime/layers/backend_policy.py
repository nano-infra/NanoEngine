"""Backend policy schema.

A ``BackendPolicy`` describes, for a hardware capability tier, which
implementation family is preferred, what fallback is permitted, whether a
correctness-first reference implementation may be substituted, and the
capability constraints that gate those choices.

This module only defines the schema. Selection logic
(``backend_selection.py`` and ``backends/selector.py``) consumes it. Keeping the
schema separate from the selectors lets policy (which implementation, and when
to fall back) evolve independently from implementation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BackendPolicy:
    """Preferred implementation, fallbacks, and capability constraints.

    Attributes
    ----------
    preferred:
        Name of the preferred implementation family (e.g. ``"flashinfer"``,
        ``"deepseek"``). ``None`` means "let the capability-aware selector
        decide" (equivalent to ``auto``).
    fallback:
        Name of the non-reference fallback implementation family, if any. Used
        when the preferred family cannot serve a request but a faster path than
        the reference implementation is still available.
    ref_fallback_allowed:
        When ``True``, selection may degrade to a ``generic``/``ref``
        implementation if neither the preferred nor the ``fallback`` family can
        serve the requested shape or capability. When ``False``, an unsupported
        shape or missing kernel raises instead of silently falling back.
    supports_fp8:
        Whether the tier supports FP8 execution. Advisory capability flag used
        by policy consumers to gate FP8-only implementations.
    """

    preferred: Optional[str] = None
    fallback: Optional[str] = None
    ref_fallback_allowed: bool = False
    supports_fp8: bool = False

    def with_ref_fallback(self, allowed: bool) -> "BackendPolicy":
        """Return a copy with ``ref_fallback_allowed`` overridden."""
        return BackendPolicy(
            preferred=self.preferred,
            fallback=self.fallback,
            ref_fallback_allowed=allowed,
            supports_fp8=self.supports_fp8,
        )


__all__ = ["BackendPolicy"]

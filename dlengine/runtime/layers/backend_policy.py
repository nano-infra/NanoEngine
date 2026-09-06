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


@dataclass(frozen=True)
class TierPolicy:
    """Per-capability-tier selection policy.

    A tier (``gpu_generic``, ``hopper``, ``blackwell``) is described entirely by
    data: which implementation family to use for each layer family, whether the
    tier supports FP8, and whether experts selection consults the quantization
    config (Blackwell picks NVFP4/MXFP4 experts based on the checkpoint format).

    This replaces the previous pattern where hardware-tier factories carried
    per-family construction logic and Blackwell subclassed Hopper only to
    override two methods.

    Attributes
    ----------
    hardware:
        Tier name, matching ``BackendSelection.hardware``.
    linear:
        Linear implementation family (``"generic"`` or ``"deepseek"``).
    experts:
        Default routed-experts family (``"generic"`` or ``"deepseek"``).
    attention / gdn:
        Requested attention/GDN backend for the tier. ``"auto"`` lets the
        capability-aware selector decide.
    supports_fp8:
        Whether the tier supports FP8 execution.
    experts_quant_override:
        When ``True``, experts selection first inspects the quantization config
        and may pick a quant-specific family (NVFP4 or MXFP4) before falling
        back to ``experts``.
    """

    hardware: str
    linear: str
    experts: str
    attention: str = "auto"
    gdn: str = "auto"
    supports_fp8: bool = False
    experts_quant_override: bool = False


# Static policy table. The linear/experts families match what the previous
# hardware-tier factories hardcoded (generic tier -> generic/BF16; hopper and
# blackwell tiers -> deepseek/FP8), so default selection is unchanged.
TIER_POLICIES = {
    "gpu_generic": TierPolicy(
        hardware="gpu_generic",
        linear="generic",
        experts="generic",
        supports_fp8=False,
    ),
    "hopper": TierPolicy(
        hardware="hopper",
        linear="deepseek",
        experts="deepseek",
        supports_fp8=True,
    ),
    "blackwell": TierPolicy(
        hardware="blackwell",
        linear="deepseek",
        experts="deepseek",
        supports_fp8=True,
        experts_quant_override=True,
    ),
}


__all__ = ["BackendPolicy", "TierPolicy", "TIER_POLICIES"]

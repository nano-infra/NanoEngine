"""Small compatibility layer for the DeepGEMM APIs used by NanoDeploy.

NanoDeploy's Hopper path uses FP32 ("natural") scaling factors.  Newer
DeepGEMM releases can also consume packed UE8M0 scales, so every FP8 entry
point in this module explicitly disables that conversion.  Keeping this in
one module prevents dense and MoE call sites from silently choosing different
scale semantics when DeepGEMM changes API names.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


_deep_gemm: ModuleType | None = None


def _load_deep_gemm() -> ModuleType:
    global _deep_gemm
    if _deep_gemm is None:
        os.environ.setdefault(
            "DG_JIT_CACHE_DIR",
            os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "nanodeploy",
                "deep_gemm",
            ),
        )
        try:
            _deep_gemm = importlib.import_module("deep_gemm")
        except ImportError as exc:
            raise ImportError(
                "DeepGEMM is required by NanoDeploy's Hopper FP8 backend"
            ) from exc
    return _deep_gemm


def _resolve(name: str):
    module = _load_deep_gemm()
    function = getattr(module, name, None)
    if function is None:
        raise RuntimeError(f"DeepGEMM is missing required API: {name}")
    return function


def _call_with_natural_scales(function, *args, **kwargs):
    """Call a DeepGEMM FP8 API with Hopper FP32-scale semantics.

    NanoDeploy targets DeepGEMM 2.3.0 and deliberately does not fall back to
    older aliases.  A signature mismatch must fail loudly instead of silently
    changing the scale representation.
    """

    kwargs["disable_ue8m0_cast"] = True
    return function(*args, **kwargs)


def ceil_div(x: int, y: int) -> int:
    """Return ``ceil(x / y)`` without requiring DeepGEMM to be imported."""

    if y == 0:
        raise ZeroDivisionError("ceil_div divisor must be non-zero")
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout() -> int:
    """Return the target DeepGEMM release's M/K contiguous alignment."""

    function = _resolve("get_mk_alignment_for_contiguous_layout")
    return int(function())


def fp8_gemm_nt(a, b, d, *args: Any, **kwargs: Any):
    """Compatibility wrapper matching ``deep_gemm.fp8_gemm_nt``."""

    function = _resolve("fp8_gemm_nt")
    return _call_with_natural_scales(function, a, b, d, *args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(
    a,
    b,
    d,
    m_indices,
    *args: Any,
    **kwargs: Any,
):
    """Run an M-grouped contiguous FP8 GEMM using natural FP32 scales."""

    function = _resolve("m_grouped_fp8_gemm_nt_contiguous")
    return _call_with_natural_scales(
        function,
        a,
        b,
        d,
        m_indices,
        *args,
        **kwargs,
    )


def m_grouped_fp8_gemm_nt_masked(
    a,
    b,
    d,
    masked_m,
    expected_m: int,
    *args: Any,
    **kwargs: Any,
):
    """Run an M-grouped masked FP8 GEMM using natural FP32 scales."""

    function = _resolve("m_grouped_fp8_gemm_nt_masked")
    return _call_with_natural_scales(
        function,
        a,
        b,
        d,
        masked_m,
        expected_m,
        *args,
        **kwargs,
    )


def _reset_for_testing() -> None:
    """Clear the lazy module cache (intentionally private, used by tests)."""

    global _deep_gemm
    _deep_gemm = None


__all__ = [
    "ceil_div",
    "fp8_gemm_nt",
    "get_m_alignment_for_contiguous_layout",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
]

"""Pure runtime-backend selection and explicit backend construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dlengine.runtime.layers.base_backend import BackendFactory

HardwareBackendName = Literal["blackwell", "hopper", "gpu_generic"]
RequestedHardwareBackend = Literal["auto", "blackwell", "hopper", "gpu_generic"]


@dataclass(frozen=True)
class BackendSelection:
    hardware: HardwareBackendName
    attention: str
    gdn: str
    hardware_source: Literal["config", "environment", "auto"]
    hardware_reason: str
    # When True, the selector may degrade a preferred/native implementation to a
    # generic/reference implementation when the native path cannot serve the
    # requested shape or capability. Defaults to False to preserve deterministic
    # behavior. See dlengine/config.py:ref_fallback_allowed and docs/
    # backend-interface-refactor.md.
    ref_fallback_allowed: bool = False
    # Human-readable explanation of the fallback decision, for observability.
    fallback_reason: str = "ref fallback disabled"


def resolve_backend_selection(
    *,
    requested_hardware: RequestedHardwareBackend = "auto",
    requested_attention: str = "auto",
    requested_gdn: str = "auto",
    cuda_capability: tuple[int, int] | None,
    legacy_hardware_backend: str | None = None,
    ref_fallback_allowed: bool = False,
) -> BackendSelection:
    """Resolve backend names without reading environment or touching CUDA."""
    valid_hardware = {"auto", "blackwell", "hopper", "gpu_generic"}
    if requested_hardware not in valid_hardware:
        raise ValueError(
            f"Unknown hardware backend: {requested_hardware!r}. "
            f"Valid values: {', '.join(sorted(valid_hardware))}."
        )

    fallback_reason = (
        "ref fallback allowed by config"
        if ref_fallback_allowed
        else "ref fallback disabled"
    )

    if requested_hardware != "auto":
        return BackendSelection(
            hardware=requested_hardware,
            attention=requested_attention,
            gdn=requested_gdn,
            hardware_source="config",
            hardware_reason=f"explicit hardware_backend={requested_hardware}",
            ref_fallback_allowed=ref_fallback_allowed,
            fallback_reason=fallback_reason,
        )

    if legacy_hardware_backend:
        if legacy_hardware_backend not in valid_hardware - {"auto"}:
            raise ValueError(
                f"Unknown NANO_BACKEND value: {legacy_hardware_backend!r}. "
                "Valid values: blackwell, hopper, gpu_generic."
            )
        return BackendSelection(
            hardware=legacy_hardware_backend,
            attention=requested_attention,
            gdn=requested_gdn,
            hardware_source="environment",
            hardware_reason=f"NANO_BACKEND={legacy_hardware_backend}",
            ref_fallback_allowed=ref_fallback_allowed,
            fallback_reason=fallback_reason,
        )

    if cuda_capability is None:
        hardware: HardwareBackendName = "gpu_generic"
        reason = "CUDA capability unavailable"
    elif cuda_capability[0] >= 10:
        hardware = "blackwell"
        reason = f"CUDA capability {cuda_capability[0]}.{cuda_capability[1]}"
    elif cuda_capability[0] >= 9:
        hardware = "hopper"
        reason = f"CUDA capability {cuda_capability[0]}.{cuda_capability[1]}"
    else:
        hardware = "gpu_generic"
        reason = f"CUDA capability {cuda_capability[0]}.{cuda_capability[1]}"

    return BackendSelection(
        hardware=hardware,
        attention=requested_attention,
        gdn=requested_gdn,
        hardware_source="auto",
        hardware_reason=reason,
        ref_fallback_allowed=ref_fallback_allowed,
        fallback_reason=fallback_reason,
    )


def create_backend(
    selection: BackendSelection,
    quant_config,
) -> BackendFactory:
    """Construct the policy-driven backend factory without changing global state.

    The hardware tier (``blackwell``/``hopper``/``gpu_generic``) is a *policy
    key* into ``TIER_POLICIES``; a single ``PolicyBackendFactory`` implements
    every tier, so there are no per-tier factory classes.
    """
    from dlengine.runtime.layers.policy import TIER_POLICIES
    from dlengine.runtime.layers.factory import PolicyBackendFactory

    if selection.hardware not in TIER_POLICIES:
        raise AssertionError(f"Unhandled hardware backend: {selection.hardware}")

    backend = PolicyBackendFactory(quant_config, tier=selection.hardware)
    backend.attention_backend = selection.attention
    backend.gdn_backend = selection.gdn
    backend.ref_fallback_allowed = selection.ref_fallback_allowed
    return backend


__all__ = [
    "BackendSelection",
    "HardwareBackendName",
    "RequestedHardwareBackend",
    "create_backend",
    "resolve_backend_selection",
]

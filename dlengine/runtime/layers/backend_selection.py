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


def resolve_backend_selection(
    *,
    requested_hardware: RequestedHardwareBackend = "auto",
    requested_attention: str = "auto",
    requested_gdn: str = "auto",
    cuda_capability: tuple[int, int] | None,
    legacy_hardware_backend: str | None = None,
) -> BackendSelection:
    """Resolve backend names without reading environment or touching CUDA."""
    valid_hardware = {"auto", "blackwell", "hopper", "gpu_generic"}
    if requested_hardware not in valid_hardware:
        raise ValueError(
            f"Unknown hardware backend: {requested_hardware!r}. "
            f"Valid values: {', '.join(sorted(valid_hardware))}."
        )

    if requested_hardware != "auto":
        return BackendSelection(
            hardware=requested_hardware,
            attention=requested_attention,
            gdn=requested_gdn,
            hardware_source="config",
            hardware_reason=f"explicit hardware_backend={requested_hardware}",
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
    )


def create_backend(
    selection: BackendSelection,
    quant_config,
) -> BackendFactory:
    """Construct the selected hardware factory without changing global state."""
    if selection.hardware == "blackwell":
        from dlengine.runtime.layers.blackwell import BlackwellBackendFactory

        backend = BlackwellBackendFactory(quant_config)
    elif selection.hardware == "hopper":
        from dlengine.runtime.layers.hopper import HopperBackendFactory

        backend = HopperBackendFactory(quant_config)
    elif selection.hardware == "gpu_generic":
        from dlengine.runtime.layers.generic import GenericBackendFactory

        backend = GenericBackendFactory(quant_config)
    else:
        raise AssertionError(f"Unhandled hardware backend: {selection.hardware}")

    backend.attention_backend = selection.attention
    backend.gdn_backend = selection.gdn
    return backend


__all__ = [
    "BackendSelection",
    "HardwareBackendName",
    "RequestedHardwareBackend",
    "create_backend",
    "resolve_backend_selection",
]

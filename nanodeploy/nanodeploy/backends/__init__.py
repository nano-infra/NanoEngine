"""NanoDeploy Hardware Abstraction Layer — backend factory singleton.

Usage
-----
    # In model_runner.py, before model creation:
    from nanodeploy.backends import init_backend
    init_backend(quant_config=quant_config)   # auto-detects GPU capability

    # In model topology files:
    from nanodeploy.backends import get_backend
    self.qkv_proj = get_backend().get_qkv_parallel_linear(...)

Backend selection order
-----------------------
1. ``NANO_BACKEND`` environment variable (``"hopper"`` | ``"gpu_generic"``)
2. Auto-detect from ``torch.cuda.get_device_capability()``:
   - compute capability >= 9.x  → ``"hopper"``
   - otherwise                  → ``"gpu_generic"``
"""

import os
from typing import Optional

from .base_backend import BackendFactory

_backend: Optional[BackendFactory] = None


def init_backend(
    quant_config=None,
    backend_type: Optional[str] = None,
) -> None:
    """Initialise the global backend factory.

    Parameters
    ----------
    quant_config:
        A ``QuantizationConfig`` instance (or ``None`` for BF16).
        If ``None``, a default (no-quantization) config is created.
    backend_type:
        Explicit backend name.  When ``None`` the backend is resolved via
        the ``NANO_BACKEND`` env var or GPU capability auto-detection.
    """
    global _backend

    # --- resolve backend type ---
    if backend_type is None:
        backend_type = os.environ.get("NANO_BACKEND")

    if backend_type is None:
        try:
            import torch

            cap = torch.cuda.get_device_capability()
            backend_type = "hopper" if cap[0] >= 9 else "gpu_generic"
        except Exception:
            backend_type = "gpu_generic"

    # --- default quant config ---
    if quant_config is None:
        from nanodeploy.models.quant_config import QuantizationConfig

        quant_config = QuantizationConfig()

    # --- instantiate factory ---
    if backend_type == "hopper":
        from .hopper import HopperBackendFactory

        _backend = HopperBackendFactory(quant_config)
    elif backend_type == "gpu_generic":
        from .gpu_generic import GenericBackendFactory

        _backend = GenericBackendFactory(quant_config)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            "Valid values: 'hopper', 'gpu_generic'."
        )


def get_backend() -> BackendFactory:
    """Return the global backend factory.

    Raises ``RuntimeError`` if ``init_backend()`` has not been called yet.
    """
    if _backend is None:
        raise RuntimeError(
            "Backend not initialised. Call nanodeploy.backends.init_backend() "
            "before creating model layers."
        )
    return _backend

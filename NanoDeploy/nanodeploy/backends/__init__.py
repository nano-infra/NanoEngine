"""NanoDeploy Hardware Abstraction Layer — backend factory singleton.

Usage
-----
    # In model_runner.py, before model creation:
    from nanodeploy.backends import init_backend
    init_backend(quant_config=quant_config)   # auto-detects hardware

    # In model topology files:
    from nanodeploy.backends import get_backend
    self.qkv_proj = get_backend().get_qkv_parallel_linear(...)

Backend selection order
-----------------------
1. Explicit ``backend_type`` argument to ``init_backend()``
2. ``NANO_BACKEND`` environment variable
   (``"hopper"`` | ``"gpu_generic"`` | ``"ascend"``)
3. Hardware auto-detect via ``backends.detect.detect_backend()``:
   - torch_npu available + NPU device present → ``"ascend"``
   - CUDA compute capability >= 9.x           → ``"hopper"``
   - CUDA available but not Hopper            → ``"gpu_generic"``
"""

import os
from typing import Optional

from .base_backend import BackendFactory

VALID_BACKENDS = ("hopper", "gpu_generic", "ascend")

_backend: Optional[BackendFactory] = None
_backend_type: Optional[str] = None


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
        the ``NANO_BACKEND`` env var or hardware auto-detection.
    """
    global _backend, _backend_type

    # --- resolve backend type ---
    # Priority: explicit arg > NANO_BACKEND env > hardware auto-detect
    if backend_type is None:
        backend_type = os.environ.get("NANO_BACKEND")

    if backend_type is None:
        from .detect import detect_backend

        backend_type = detect_backend()

    if backend_type not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Valid values: {VALID_BACKENDS}."
        )

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
    elif backend_type == "ascend":
        from .ascend import AscendBackendFactory

        _backend = AscendBackendFactory(quant_config)

    _backend_type = backend_type


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


def get_backend_type() -> Optional[str]:
    """Return the currently selected backend type string (e.g. ``"ascend"``)."""
    return _backend_type

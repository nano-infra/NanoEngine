"""DLEngine Hardware Abstraction Layer.

Usage
-----
    # In model_runner.py, before model creation:
    from dlengine.runtime.layers import init_backend
    init_backend(quant_config=quant_config)   # auto-detects GPU capability

    # In model topology files:
    from dlengine.runtime.layers import get_backend
    self.qkv_proj = get_backend().get_qkv_parallel_linear(...)

Backend selection order
-----------------------
1. Config.hardware_backend when explicitly set.
2. Legacy NANO_BACKEND environment variable.
3. Auto-detect from torch.cuda.get_device_capability():
   - compute capability >= 10.x → ``"blackwell"``
   - compute capability >= 9.x  → ``"hopper"``
   - otherwise                  → ``"gpu_generic"``

Attention and GDN kernels are selected independently by
``Config.attention_backend`` and ``Config.gdn_backend``. Their ``auto`` modes
resolve from CUDA capability and installed kernel support.
"""

from typing import Optional

from .backend_selection import create_backend, resolve_backend_selection
from .base_backend import BackendFactory

_backend: Optional[BackendFactory] = None


def init_backend(
    quant_config=None,
    backend_type: Optional[str] = None,
    attention_backend: Optional[str] = None,
    gdn_backend: Optional[str] = None,
) -> BackendFactory:
    """Compatibility entry point that selects, constructs, and stores a backend.

    New runtime code should resolve and construct explicitly, then call
    set_backend.
    """
    import os

    if quant_config is None:
        from dlengine.runtime.models.quant_config import QuantizationConfig

        quant_config = QuantizationConfig()

    try:
        import torch

        capability = torch.cuda.get_device_capability()
    except Exception:
        capability = None

    selection = resolve_backend_selection(
        requested_hardware=backend_type or "auto",
        requested_attention=attention_backend or "auto",
        requested_gdn=gdn_backend or "auto",
        cuda_capability=capability,
        legacy_hardware_backend=os.environ.get("NANO_BACKEND"),
    )
    backend = create_backend(selection, quant_config)
    set_backend(backend)
    return backend


def set_backend(backend: BackendFactory) -> None:
    """Store the already-resolved backend for the current worker process."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Clear process backend state. Intended for worker teardown and tests."""
    global _backend
    _backend = None


def get_backend() -> BackendFactory:
    """Return the global backend factory.

    Raises ``RuntimeError`` if ``init_backend()`` has not been called yet.
    """
    if _backend is None:
        raise RuntimeError(
            "Backend not initialised. Call dlengine.runtime.layers.init_backend() "
            "before creating model layers."
        )
    return _backend


__all__ = [
    "create_backend",
    "get_backend",
    "init_backend",
    "reset_backend",
    "resolve_backend_selection",
    "set_backend",
]

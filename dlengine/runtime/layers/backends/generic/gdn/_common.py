"""Shared low-level helpers for GDN components.

Kept in a tiny leaf module so the component mixins and the composing class can
all import it without creating an import cycle through
``gated_delta_net.py``.
"""

import torch

from dlengine.runtime.compile_utils import maybe_compile


# Module-level lazily-compiled L2 norm. Compiling at module-import time (e.g.
# via @torch.compile on a class method) attaches ConfigModuleInstance refs to
# the class object, which breaks cloudpickle in Ray actors on torch >= 2.10.
# Using a module-level function + lazy compile keeps the wrapper out of any
# class dict and out of module globals at import time.
def _l2norm_impl(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


_l2norm_compiled_fn = None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization matching FLA's l2norm."""
    global _l2norm_compiled_fn
    if _l2norm_compiled_fn is None:
        _l2norm_compiled_fn = maybe_compile(_l2norm_impl)
    return _l2norm_compiled_fn(x, dim, eps)


__all__ = ["l2norm"]

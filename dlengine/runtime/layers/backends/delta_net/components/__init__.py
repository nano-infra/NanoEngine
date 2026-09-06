"""Decomposed GatedDeltaNet components.

Each component is an independently-testable mixin composed by
``GenericGatedDeltaNet``:

- ``kernels``:    optional-kernel capability probes and callables.
- ``conv``:       causal depthwise convolution (prefill + decode).
- ``state``:      recurrent-state slot hygiene (fresh-slot zeroing, keep mask).
- ``recurrence``: delta-rule recurrence (FlashInfer / FLA / naive) for prefill
                  and decode.
- ``output``:     gated RMSNorm + output projection, and ``RMSNormGated``.
"""

from .conv import CausalConvMixin
from .output import OutputTransformMixin, RMSNormGated
from .recurrence import RecurrenceMixin
from .state import StateMixin

__all__ = [
    "CausalConvMixin",
    "OutputTransformMixin",
    "RMSNormGated",
    "RecurrenceMixin",
    "StateMixin",
]

"""Decomposed GatedDeltaNet components.

Independently-testable mixins composed by the GDN backends:

- ``kernels``:     optional-kernel capability probes and callables.
- ``conv``:        causal depthwise convolution (prefill + decode).
- ``state``:       recurrent-state slot hygiene (fresh-slot zeroing, keep mask).
- ``recurrence``:  ``NaiveRecurrenceMixin`` (pure PyTorch reference) + the shared
                   ``RecurrenceStateMixin`` pool load/store.
- ``recurrence_flashinfer`` / ``recurrence_fla``: the FlashInfer and FLA
                   recurrence strategies (each a ``RecurrenceStateMixin`` mixin).
- ``output``:      gated RMSNorm + output projection, and ``RMSNormGated``.

Recurrence strategy is chosen by *which mixin a backend composes*, not by
runtime ``_has_*`` flags.
"""

from .conv import CausalConvMixin
from .output import OutputTransformMixin, RMSNormGated
from .recurrence import NaiveRecurrenceMixin, RecurrenceMixin, RecurrenceStateMixin
from .recurrence_fla import FlaRecurrenceMixin
from .recurrence_flashinfer import FlashInferRecurrenceMixin
from .state import StateMixin

__all__ = [
    "CausalConvMixin",
    "OutputTransformMixin",
    "RMSNormGated",
    "NaiveRecurrenceMixin",
    "RecurrenceMixin",
    "RecurrenceStateMixin",
    "FlashInferRecurrenceMixin",
    "FlaRecurrenceMixin",
    "StateMixin",
]

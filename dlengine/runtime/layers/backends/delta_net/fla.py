"""Flash-Linear-Attention (FLA) GatedDeltaNet backend.

Reuses ``GenericGatedDeltaNet``'s projection/conv/state/output but swaps the
recurrence to FLA's chunk-prefill + fused-recurrent-decode kernels. Requires
flash-linear-attention; raises at construction otherwise.
"""

from dlengine.runtime.layers.backends.delta_net.components import kernels
from dlengine.runtime.layers.backends.delta_net.components.recurrence_fla import (
    FlaRecurrenceMixin,
)
from dlengine.runtime.layers.backends.delta_net.generic import GenericGatedDeltaNet


class FlaGatedDeltaNet(FlaRecurrenceMixin, GenericGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kernels.HAS_FLA_GDN:
            raise RuntimeError(
                "FLA GDN was selected but flash-linear-attention is unavailable."
            )


__all__ = ["FlaGatedDeltaNet"]

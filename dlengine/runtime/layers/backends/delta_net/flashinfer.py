"""FlashInfer GatedDeltaNet backend (SM90 native).

Reuses ``GenericGatedDeltaNet``'s projection/conv/state/output but swaps the
recurrence to the FlashInfer chunk-prefill + fused-decode kernels. Requires the
FlashInfer GDN kernels; raises at construction otherwise.
"""

from dlengine.runtime.layers.backends.delta_net.components import kernels
from dlengine.runtime.layers.backends.delta_net.components.recurrence_flashinfer import (
    FlashInferRecurrenceMixin,
)
from dlengine.runtime.layers.backends.delta_net.generic import GenericGatedDeltaNet
from dlengine.utils.cuda import get_cuda_compute_capability


class FlashInferGatedDeltaNet(FlashInferRecurrenceMixin, GenericGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        capability = get_cuda_compute_capability()
        sm_major = capability[0] if capability is not None else 0
        has_prefill = kernels.HAS_FLASHINFER_GDN_PREFILL and sm_major >= 9
        self._has_flashinfer_pretranspose = (
            kernels.HAS_FLASHINFER_GDN_PRETRANSPOSE and sm_major >= 9
        )
        self._has_flashinfer_nontranspose = (
            kernels.HAS_FLASHINFER_GDN_NONTRANSPOSE and sm_major == 9
        )
        if not has_prefill or not (
            self._has_flashinfer_pretranspose or self._has_flashinfer_nontranspose
        ):
            raise RuntimeError(
                "FlashInfer GDN was selected but its prefill/decode kernels are unavailable."
            )


__all__ = ["FlashInferGatedDeltaNet"]

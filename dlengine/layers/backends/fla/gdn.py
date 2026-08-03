"""Flash Linear Attention GatedDeltaNet backend."""

from dlengine.layers.generic.gated_delta_net import GenericGatedDeltaNet


class FLAGatedDeltaNet(GenericGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._has_fla:
            raise RuntimeError(
                "FLA GDN was selected but flash-linear-attention is unavailable."
            )
        self._has_flashinfer_prefill = False
        self._has_flashinfer_pretranspose = False
        self._has_flashinfer_nontranspose = False


__all__ = ["FLAGatedDeltaNet"]

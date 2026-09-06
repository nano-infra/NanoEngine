"""FlashInfer GatedDeltaNet backend."""

from dlengine.runtime.layers.backends.generic.gated_delta_net import GenericGatedDeltaNet


class FlashInferGatedDeltaNet(GenericGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._has_flashinfer_prefill or not (
            self._has_flashinfer_pretranspose or self._has_flashinfer_nontranspose
        ):
            raise RuntimeError(
                "FlashInfer GDN was selected but its prefill/decode kernels are unavailable."
            )
        self._has_fla = False


__all__ = ["FlashInferGatedDeltaNet"]

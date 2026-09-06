"""Explicit correctness/debug Torch GatedDeltaNet backend."""

from dlengine.runtime.layers.backends.delta_net.generic import GenericGatedDeltaNet


class TorchGatedDeltaNet(GenericGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_flashinfer_prefill = False
        self._has_flashinfer_pretranspose = False
        self._has_flashinfer_nontranspose = False
        self._has_fla = False


__all__ = ["TorchGatedDeltaNet"]

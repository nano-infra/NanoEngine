"""Explicit correctness/debug Torch attention backend."""

from dlengine.runtime.layers.backends.attention.fa2 import Fa2Attention


class TorchAttention(Fa2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_flashinfer_decode=False, **kwargs)
        self.impl.use_fa2 = False


__all__ = ["TorchAttention"]

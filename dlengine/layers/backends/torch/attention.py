"""Explicit correctness/debug Torch attention backend."""

from ..fa.fa2 import FA2Attention


class TorchAttention(FA2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_flashinfer_decode=False, **kwargs)
        self.impl.use_fa2 = False


__all__ = ["TorchAttention"]

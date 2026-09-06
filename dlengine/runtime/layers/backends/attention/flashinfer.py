"""FlashInfer paged attention backend."""

from dlengine.runtime.layers.backends.attention.fa2 import Fa2Attention


class FlashInferAttention(Fa2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_flashinfer_decode=True, **kwargs)
        if not self.impl.has_flashinfer:
            raise RuntimeError("FlashInfer attention was selected but is unavailable.")


__all__ = ["FlashInferAttention"]

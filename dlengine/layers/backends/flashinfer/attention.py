"""FlashInfer paged attention backend."""

from ..fa.fa2 import FA2Attention


class FlashInferAttention(FA2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_flashinfer_decode=True, **kwargs)
        if not self.impl.has_flashinfer:
            raise RuntimeError("FlashInfer attention was selected but is unavailable.")


__all__ = ["FlashInferAttention"]

"""FlashInfer paged attention backend (GQA).

FlashInfer paged prefill + decode, with FlashAttention-2 as the ragged-prefill
and no-FlashInfer fallback. Built on ``Fa2Attention`` with the FlashInfer path
forced on; requires FlashInfer to be installed.
"""

from dlengine.runtime.layers.backends.attention.fa2 import Fa2Attention


class FlashInferAttention(Fa2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_flashinfer_decode=True, **kwargs)
        if not self.has_flashinfer:
            raise RuntimeError("FlashInfer attention was selected but is unavailable.")


__all__ = ["FlashInferAttention"]

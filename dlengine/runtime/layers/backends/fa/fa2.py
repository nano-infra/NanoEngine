"""FlashAttention-2 backend for Ampere/Ada and compatible GPUs."""

from dlengine.runtime.layers.generic.attention import GenericAttention


class FA2Attention(GenericAttention):
    def __init__(self, *args, force_flashinfer_decode: bool = False, **kwargs):
        # The existing FA2 implementation uses FlashInfer decode in auto mode.
        # Keep that proven hybrid path for auto selection; explicit ``fa2``
        # disables it through the instance dispatch flags added below.
        super().__init__(*args, **kwargs)
        self.impl.use_flashinfer_decode = force_flashinfer_decode
        self.impl.use_flashinfer_prefill = force_flashinfer_decode


__all__ = ["FA2Attention"]

"""FA4 prefill + FlashInfer TRTLLM decode backend for Blackwell."""

from dlengine.layers.blackwell.attention import BlackwellAttention as FA4Attention

__all__ = ["FA4Attention"]

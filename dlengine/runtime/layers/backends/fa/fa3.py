"""FlashAttention-3 backend for Hopper."""

from dlengine.runtime.layers.hopper.attention import HopperAttention as FA3Attention

__all__ = ["FA3Attention"]

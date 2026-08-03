"""FlashAttention-3 backend for Hopper."""

from dlengine.layers.hopper.attention import HopperAttention as FA3Attention

__all__ = ["FA3Attention"]

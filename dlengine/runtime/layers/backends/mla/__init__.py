"""MLA (Multi-head Latent Attention) backend family.

Dense compressed-KV MLA decode backends, named by vendor/kernel:

- ``flash_mla`` : FlashMlaAttention   (Hopper FlashMLA decode; dense + FP8 sparse)
- ``trtllm``    : TrtllmMlaAttention  (Blackwell TRTLLM-GEN decode)

Selected by ``selector.create_mla`` / ``resolve_mla_plan`` from the hardware
tier. DSA sparse attention (``backends/dsa/``) reuses these kernels with an
index mask. MLA prefill currently lives in ``DeepseekV2Attention.forward`` and
is pulled into the family with the DSA stage.
"""

from dlengine.runtime.layers.backends.mla.base import MlaAttentionBase

__all__ = ["MlaAttentionBase"]

"""Attention backend family.

Concrete dense-attention implementations, named by vendor/kernel:

- ``generic``   : GenericAttention (QK·softmax·V; FA2/FlashInfer/SDPA dispatch)
- ``fa2``       : Fa2Attention        (FlashAttention-2)
- ``fa3``       : Fa3Attention        (FlashAttention-3 + FlashMLA, SM90)
- ``fa4``       : Fa4Attention / Fa4MlaAttention (FA4 + TRTLLM-GEN, SM100+)
- ``flashinfer``: FlashInferAttention (paged FlashInfer decode)
- ``torch``     : TorchAttention      (SDPA correctness/debug)

Shared paged-cache helpers live in ``mla_utils``. Sparse MLA (DeepSeek/GLM DSA)
is a separate family under ``backends/dsa/``.
"""

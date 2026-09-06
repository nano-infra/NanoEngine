"""Attention backend family.

Concrete dense-GQA attention implementations, named by vendor/kernel. Shared
GQA plumbing (KV store, HiSparse SWA, cached-prefill gather, FlashInfer plan
caching, SDPA kernels) lives in ``base`` (``GqaAttentionBase``); each backend
implements only its attend kernels.

- ``generic``   : GenericAttention   (pure SDPA reference; the GQA ref fallback
                  and the debug backend, formerly named ``torch``)
- ``fa2``       : Fa2Attention        (FlashAttention-2, optional FlashInfer path)
- ``fa3``       : Fa3Attention        (FlashAttention-3 + FlashMLA, SM90)
- ``fa4``       : Fa4Attention / Fa4MlaAttention (FA4 + TRTLLM-GEN, SM100+)
- ``flashinfer``: FlashInferAttention (paged FlashInfer prefill+decode)

Shared paged-cache helpers live in ``mla_utils``. Sparse MLA (DeepSeek/GLM DSA)
is a separate family under ``backends/dsa/``.
"""

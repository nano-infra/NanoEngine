"""Routed-experts (MoE) backend family (named by vendor/kernel).

Implementations:

- ``generic``   : GenericExperts       (BF16 + EP reference)
- ``deep_gemm`` : DeepGemmExperts       (FP8 DeepGEMM/DeepEP, SM90+)
- ``mega_moe``  : MegaMoEExperts        (MXFP4)
- ``nvfp4``     : ModelOptNvFp4Experts  (Blackwell NVFP4)

MoE-specific machinery shared by the implementations:

- ``local_dispatch``   : LocalPaddedDispatcher (CUDA-graph-safe local EP=1 path)
- ``token_dispatcher`` : DeepEP normal / low-latency token dispatchers
- ``eplb``             : expert-parallel load-balancing (logical->physical maps)
"""

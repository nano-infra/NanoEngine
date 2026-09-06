"""Routed-experts (MoE) backend family (named by vendor/kernel).

- ``generic``   : GenericExperts       (BF16 + EP reference)
- ``deep_gemm`` : DeepGemmExperts       (FP8 DeepGEMM/DeepEP, SM90+)
- ``mega_moe``  : MegaMoEExperts        (MXFP4)
- ``nvfp4``     : ModelOptNvFp4Experts  (Blackwell NVFP4)
"""

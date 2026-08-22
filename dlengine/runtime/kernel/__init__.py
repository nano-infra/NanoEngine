"""GPU compute kernels for DLEngine, grouped by implementation technology.

- ``kernel.triton``: hand-written Triton kernels (``generic`` and ``hopper``).
- ``kernel.jit.sgl``: vendored sglang tvm-ffi JIT CUDA kernels (Apache-2.0).
- ``kernel.tilelang.sgl``: vendored sglang TileLang kernels (Apache-2.0).
"""

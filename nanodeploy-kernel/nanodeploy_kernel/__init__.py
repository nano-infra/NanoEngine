"""nanodeploy-kernel: Triton and tvm-ffi JIT compute kernels for NanoDeploy.

Standalone package holding the GPU kernels used by ``nanodeploy``:

- ``nanodeploy_kernel.gpu_generic``: portable Triton kernels.
- ``nanodeploy_kernel.hopper``: Hopper (sm_90) specific kernels.
- ``nanodeploy_kernel.sglang_jit_kernel`` / ``sglang_mhc``: vendored slices of
  ``sglang`` JIT kernels (Apache-2.0). See each subpackage for attribution.
"""

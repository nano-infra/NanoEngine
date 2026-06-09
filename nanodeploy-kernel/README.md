# nanodeploy-kernel

Standalone GPU compute kernels used by [`nanodeploy`](../nanodeploy). Split out
so the kernels can be versioned, tested, and depended on independently of the
serving engine.

Pure-Python distribution: the kernels are [Triton](https://github.com/triton-lang/triton)
kernels plus a few CUDA kernels JIT-compiled at runtime via
[`tvm-ffi`](https://pypi.org/project/apache-tvm-ffi/) `load_inline`. There is no
C++ build step at install time.

## Layout

| Import path                           | Contents                                             |
| ------------------------------------- | ---------------------------------------------------- |
| `nanodeploy_kernel.gpu_generic`       | Portable Triton kernels (copy, moe, rmsnorm, ...)    |
| `nanodeploy_kernel.hopper`            | Hopper (sm_90) specific kernels (fp8, fused MoE v3)  |
| `nanodeploy_kernel.sglang_jit_kernel` | Vendored slice of `sglang` JIT kernels (Apache-2.0)  |
| `nanodeploy_kernel.sglang_mhc`        | Vendored `sglang` multi-head compressor (Apache-2.0) |

The vendored `sglang_jit_kernel` ships its `include/` and `csrc/` headers as
package data; they are resolved at runtime relative to the installed module.

## Runtime dependencies

- `torch` and `triton` (provided by the CUDA runtime environment, as in `nanodeploy`)
- `apache-tvm-ffi` (declared dependency; provides the `load_inline` JIT path)

## Install (editable, for development)

```bash
pip install -e nanodeploy-kernel
```

`nanodeploy` declares a dependency on this package, so installing the engine
pulls it in automatically.

"""Pruned vendor of https://github.com/sgl-project/sglang
   python/sglang/jit_kernel/deepseek_v4.py

Tier 1 + Tier 2 fused kernels exposed here:
  * ``fused_rope``                  — single-kernel Q+K RoPE (replaces ~10
                                      elementwise launches per call)
  * ``rmsnorm_self``                — per-head RMSNorm in one kernel
  * ``fused_norm_rope_inplace``     — RMSNorm + RoPE fused, in-place on the
                                      kv_a buffer (no slice-write-back, no
                                      explicit clone)
  * ``silu_and_mul_clamp``          — SwiGLU with clamp in one kernel
                                      (drop-in for the activation in MLP /
                                      shared-expert paths)

The upstream module also ships topk / mega-moe / kv-store kernels; those
depend on ``sglang.srt.debug_utils`` and the compressor plan helpers
which would drag in more of the upstream surface. Re-vendor selectively
if/when those paths are needed.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union

import torch

from .utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


# ─── JIT module loaders (each compiled once on first use) ───────────────────


@cache_once
def _jit_rmsnorm_head_module(head_dim: int, dtype: torch.dtype):
    args = make_cpp_args(head_dim, dtype, is_arch_support_pdl())
    kernel_class = f"RMSNormKernel<{args}>"
    return load_jit(
        _make_name("rmsnorm_head"),
        *args,
        cuda_files=["deepseek_v4/rmsnorm.cuh"],
        cuda_wrappers=[("run_self", f"{kernel_class}::run_self")],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
):
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        _make_name("fused_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        _make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_fused_rope_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


# ─── Public API ─────────────────────────────────────────────────────────────


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply DSV4 interleaved RoPE to ``q`` (and optionally ``k``) in-place.

    Parameters
    ----------
    q : Tensor
        Shape ``[B, num_q_heads, head_dim]``, contiguous along the last
        dim, dtype must match the kernel build (currently bfloat16).
    k : Optional[Tensor]
        Same layout, ``[B, num_k_heads, head_dim]``. Pass ``None`` to
        skip the K rotation.
    freqs_cis : Tensor
        ``[max_pos, head_dim/2]`` complex tensor (cos+i·sin). Must live
        on the same device as q. The kernel internally indexes this by
        ``positions``.
    positions : Tensor
        ``[B]`` int32 or int64.
    inverse : bool
        Apply the inverse rotation (used in DSV4's MLA output projection).
    """
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """RMSNorm + interleaved RoPE in a single kernel, in-place on ``kv``.

    Replaces the pattern::

        kv = kv_norm(kv)
        kv[..., -rope_dim:] = _apply_rotary_interleaved(rotary_emb, positions, kv[..., -rope_dim:])

    with one kernel launch. Avoids the residual-clone my Tier-1
    rotary patch had to do.

    Parameters
    ----------
    kv : Tensor
        Shape ``[B, head_dim]`` (2D — different from ``fused_rope`` which
        is 3D), bfloat16, contiguous along last dim. ``head_dim`` must
        be one of the kernel-supported sizes (e.g. 128, 256, 512).
        Mutated in place.
    weight : Tensor
        RMSNorm weight, shape ``[head_dim]``.
    eps : float
    freq_cis : Tensor
        Complex64 ``[max_pos, rope_dim/2]``, same as for ``fused_rope``.
    positions : Tensor
        ``[B]`` int64 (this kernel is int64-only, unlike ``fused_rope``
        which also accepts int32).
    """
    freq_cis_real = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis_real.shape[-1])
    # mode=2 means "norm + rope in one pass, no compressor plan"
    module.forward(kv, weight, positions, freq_cis_real, 2, eps, 0)


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Single-kernel SwiGLU with clamp.

    ``output[i] = silu(input[i, :D]) * input[i, D:]`` then clamped to
    ``[-swiglu_limit, swiglu_limit]``. Drop-in for the
    chunk + silu + mul + clamp + cast sequence currently used by
    the activation modules.

    Parameters
    ----------
    input : Tensor
        Shape ``[*, 2*D]`` (the two halves go to silu(x) and y in
        ``silu(x)*y``). Bfloat16.
    output : Tensor
        Shape ``[*, D]``. Bfloat16.
    swiglu_limit : float
        Pass ``float('inf')`` to skip clamp.
    """
    module = _jit_silu_and_mul_clamp_module(input.dtype)
    module.run(input, output, float(swiglu_limit))


def rmsnorm_self(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm in a single kernel.

    Parameters
    ----------
    q : Tensor
        Shape ``[batch, num_heads, head_dim]``.
    eps : float
        Epsilon for the rsqrt.

    Returns
    -------
    Tensor
        Same shape and dtype as ``q``, freshly allocated.
    """
    module = _jit_rmsnorm_head_module(q.shape[-1], q.dtype)
    out = q.new_empty(q.shape)
    module.run_self(q, out, eps)
    return out


# ─── AOT precompile helper (call once at process start to avoid first-
#     request JIT pause). Matches upstream's ``compile_aot``. ────────────────


def _compile_one(*input_tuple) -> None:
    name, job_fn, *args = input_tuple
    print(f"Compiling {name}...", flush=True)
    job_fn(*args)
    print(f"Finished compiling {name}.", flush=True)


def compile_aot():
    """Build all vendored kernels in parallel (multiprocessing). Call at
    worker start to avoid the first-request JIT pause.

    Note on ``rmsnorm_self`` head_dim: the kernel is templated on
    ``kHeadDim`` and statically asserts ``kHeadDim % (kWarpThreads *
    kVecSize) == 0``, i.e. head_dim must be a multiple of 128 (warp
    size 32 × 4-element bf16 vector). DSV4 production paths that fit:
    ``kv_lora_rank=512`` and ``q_lora_rank=1536``. Smaller head dims
    (e.g. 64 for the qk_rope sub-projection) are NOT supported by this
    kernel — keep the eager RMSNorm there.
    """
    jobs = [
        ("rope", _jit_fused_rope_module),
        ("rmsnorm_head_512_bf16", _jit_rmsnorm_head_module, 512, torch.bfloat16),
    ]
    import multiprocessing

    max_parallel_jobs = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        pool.starmap(_compile_one, jobs)


if __name__ == "__main__":
    compile_aot()

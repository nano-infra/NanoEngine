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
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    swizzle: bool = False,
    apply_swiglu_limit: bool = True,
):
    """sglang's silu_and_mul + per-token-group FP8 quant (contiguous variant).

    Fuses ``silu(gate) * up`` (with optional clamp), then per-128-element
    UE8M0 FP8 quant, into one CUDA kernel. Replaces the eager chain
    ``silu_and_mul_post_quant_kernel + _quant_fp8_kernel`` (~2 launches +
    intermediate tensor → 1 launch, no intermediate).
    """
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        _make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_mul_quant_varlen_module(
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    swizzle: bool = False,
    apply_swiglu_limit: bool = True,
):
    """sglang's silu_and_mul + per-token-group FP8 quant (masked / varlen).

    For the masked-MoE expert compute path: input is
    ``[num_experts, num_tokens_padded, 2*hidden_dim]`` and ``masked_m``
    indicates per-expert valid token count. Output is
    ``[num_experts, num_tokens_padded, hidden_dim]`` fp8.
    """
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        _make_name("silu_mul_quant_varlen"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
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


@cache_once
def _jit_topk_module():
    """sglang's NSA top-K=512 + page-table-translation kernel (radix-256
    in shared memory). Replaces masked_fill + topk + where + (optional
    page transform) chain."""
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        _make_name("topk"),
        *args,
        cuda_files=["deepseek_v4/topk.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK512Kernel<{args}>::transform")],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: int,
):
    """sglang's compressor kernel — fuses scatter-update + softmax-weighted-
    sum into one CUDA pass. ``ratio`` is 4 (overlap) or 128 (non-overlap).
    Replaces the per-decode chain:
        kv_state[idx] = kv;  score_state[idx] = score + ape
        compressed = (kv_state * score_state.softmax(dim=1)).sum(dim=1)
    """
    assert ratio in (4, 128)
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        _make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
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


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """sglang's NSA top-K=512 + page-table-translation kernel.

    Single CUDA kernel that:
      1. Treats positions ``>= seq_lens[b]`` as ``-inf`` (implicit mask).
      2. Selects the 512 highest-scoring positions per batch entry via a
         radix-256 sort in shared memory.
      3. Translates those raw token positions into physical page slots
         via ``page_tables`` (when caller wants page indices).

    Replaces the masked_fill + topk + where + (optional page transform)
    chain with a single launch. Drop-in for DLEngine's NSA Indexer's
    final selection step.

    Parameters
    ----------
    scores : [bs, max_context_len] float32 (the deep_gemm
        ``fp8_paged_mqa_logits`` output works directly here).
    seq_lens : [bs] int32 — actual context length per sequence.
    page_tables : [bs, max_pages] int32 — physical page id for each
        logical page slot.
    out_page_indices : [bs, 512] int32 — pre-allocated output buffer
        for page-translated slot indices (set to ``-1`` for invalid
        positions).
    page_size : int — must be a power of 2.
    out_raw_indices : optional [bs, 512] int32 — if supplied, also
        receives the pre-translation raw token positions. Pass ``None``
        when only page indices are needed.
    """
    module = _jit_topk_module()
    module.topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


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


def silu_mul_quant_contig(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    swiglu_limit: float = float("inf"),
    transposed: bool = False,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    swizzle: bool = False,
) -> None:
    """Fused silu_and_mul + per-token-group FP8 quant (contiguous).

    Replaces the eager chain ``silu_and_mul_post_quant_kernel +
    _quant_fp8_kernel`` with a single CUDA launch (no intermediate
    bfloat16 buffer materialized).

    Parameters
    ----------
    input : Tensor
        Shape ``[M, 2*D]`` bfloat16. Last dim is the concatenated
        ``[gate, up]`` pair; ``silu(gate) * up`` is computed.
    output : Tensor
        Shape ``[M, D]`` float8_e4m3fn. Pre-allocated.
    output_scale : Tensor
        Shape ``[M, D/quant_group_size]`` float32 (when ``transposed=False``).
        For ``transposed=True``, see sglang's docs for the col-major int32
        layout. Pre-allocated.
    swiglu_limit : float
        Pre-clamp limit for ``silu(gate) * up``. ``inf`` to skip clamp.
        DSV4 production uses ``10.0``.
    transposed : bool
        Whether the scale is in transposed (col-major int32) layout.
    quant_group_size : int
        Must be 128.
    scale_ue8m0 : bool
        UE8M0 (power-of-2-rounded) scaling. Matches DLEngine's existing
        UE8M0 quant path.
    swizzle : bool
        Layout swizzle for sm100+ TMA paths. Default False on H200.
    """
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size=quant_group_size,
        scale_ue8m0=scale_ue8m0,
        swizzle=swizzle,
        apply_swiglu_limit=(swiglu_limit != float("inf")),
    )
    module.run(input, output, output_scale, transposed, float(swiglu_limit))


def silu_mul_quant_masked(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    topk: int,
    swiglu_limit: float = float("inf"),
    transposed: bool = False,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    swizzle: bool = False,
) -> None:
    """Fused silu_and_mul + per-token-group FP8 quant (masked / per-expert).

    For the MoE expert compute path. ``input`` shape
    ``[num_experts, num_tokens_padded, 2*hidden_dim]`` bf16, ``masked_m``
    [num_experts] int32 indicates valid tokens per expert. Replaces the
    existing ``silu_and_mul_masked_post_quant_fwd`` Triton kernel chain.
    """
    module = _jit_silu_mul_quant_varlen_module(
        quant_group_size=quant_group_size,
        scale_ue8m0=scale_ue8m0,
        swizzle=swizzle,
        apply_swiglu_limit=(swiglu_limit != float("inf")),
    )
    module.run(
        input, output, output_scale, masked_m, topk, transposed, float(swiglu_limit)
    )


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

"""Fused tilelang kernels for DSV4 compressor decode compute.

Replaces the cat-rearrange + softmax + weighted-sum + cast chain in
``DeepseekV4Compressor.forward_decode_batched`` with a single tilelang
kernel per case (overlap=True for ratio=4 layers, overlap=False for
ratio=128 layers). Single launch instead of 5-7 elementwise + 1 spatial
softmax + 1 reduce per call.

Tilelang (rather than Triton) is used because v23 bench showed that
Triton-with-warp-reductions + tilelang's pre-compiled HC kernels
(`_SGL_MHC_PRE`) trigger a runtime conflict on first co-launch — the HC
fast path silently raises and the eager fallback fires (~12k extra
kernels/step). Two tilelang kernels in the same process don't have this
problem.

The compressor's persistent state buffers ``_kv_states`` and
``_score_states`` are not modified by these kernels — they only consume
gathered slices ``kv_st = _kv_states[seq_slots]`` and produce the
bfloat16-cast compressed output that feeds into ``_maybe_fused_norm_rope``.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from dlengine.kernel.jit.sgl.utils import is_arch_support_pdl

tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP32 = "float32"
BF16 = "bfloat16"


# ─── Overlap (ratio=4) softmax + weighted-sum ──────────────────────────────


@functools.lru_cache(maxsize=8)
def _make_overlap_kernel(ratio: int, head_dim: int, block_hd: int):
    """Factory: tilelang JIT-compiled kernel for the overlap case.

    For each (bs_id, h_chunk) program:
      * Load 2*ratio = ``n_rows`` rows × ``block_hd`` head_dim elements.
        Rows in [0, ratio) take from the LEFT half of the last dim
        (``[:, k, :head_dim]``); rows in [ratio, 2*ratio) take from the
        RIGHT half (``[:, k, head_dim:]``).
      * Per-element softmax along the row axis.
      * Weighted sum and store as bfloat16.
    """
    n_rows = 2 * ratio
    two_hd = 2 * head_dim
    assert head_dim % block_hd == 0, "head_dim must be a multiple of block_hd"
    n_chunks = head_dim // block_hd
    enable_pdl = is_arch_support_pdl()

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _kernel():
        bs = T.symbolic("bs")

        @T.prim_func
        def main(
            kv_st: T.Tensor[(bs, n_rows, two_hd), FP32],
            score_st: T.Tensor[(bs, n_rows, two_hd), FP32],
            out: T.Tensor[(bs, head_dim), BF16],
        ):
            with T.Kernel(bs, n_chunks, threads=128) as (i, chunk):
                if enable_pdl:
                    T.pdl_sync()

                h_start = chunk * block_hd

                kv_frag = T.alloc_fragment((n_rows, block_hd), FP32)
                score_frag = T.alloc_fragment((n_rows, block_hd), FP32)

                # Gather overlap rows (k < ratio): left half of last dim
                for k, hh in T.Parallel(ratio, block_hd):
                    kv_frag[k, hh] = kv_st[i, k, h_start + hh]
                    score_frag[k, hh] = score_st[i, k, h_start + hh]
                # Gather current rows (k >= ratio): right half (head_dim + h)
                for k, hh in T.Parallel(ratio, block_hd):
                    kv_frag[ratio + k, hh] = kv_st[
                        i, ratio + k, head_dim + h_start + hh
                    ]
                    score_frag[ratio + k, hh] = score_st[
                        i, ratio + k, head_dim + h_start + hh
                    ]

                # Per-element softmax along the row axis
                score_max = T.alloc_fragment(block_hd, FP32)
                T.reduce_max(score_frag, score_max, dim=0)
                for k, hh in T.Parallel(n_rows, block_hd):
                    score_frag[k, hh] = T.exp(score_frag[k, hh] - score_max[hh])
                sum_exp = T.alloc_fragment(block_hd, FP32)
                T.reduce_sum(score_frag, sum_exp, dim=0)

                # Weighted sum
                out_frag = T.alloc_fragment(block_hd, FP32)
                T.clear(out_frag)
                for k in T.serial(n_rows):
                    for hh in T.Parallel(block_hd):
                        out_frag[hh] += kv_frag[k, hh] * (
                            score_frag[k, hh] / sum_exp[hh]
                        )

                # Store as bf16
                for hh in T.Parallel(block_hd):
                    out[i, h_start + hh] = T.cast(out_frag[hh], BF16)

        return main

    return _kernel()


def compress_overlap_softmax_sum(
    kv_st: torch.Tensor,  # [bs, 2*ratio, 2*head_dim] fp32
    score_st: torch.Tensor,  # [bs, 2*ratio, 2*head_dim] fp32
    ratio: int,
) -> torch.Tensor:
    """Drop-in replacement for the overlap-case softmax+weighted-sum chain.

    Returns ``[bs, head_dim]`` bfloat16. The caller can skip the
    ``.to(dtype).contiguous()`` cast that fed into ``_maybe_fused_norm_rope``.
    """
    bs, n_rows, two_hd = kv_st.shape
    assert n_rows == 2 * ratio
    head_dim = two_hd // 2
    out = torch.empty(bs, head_dim, dtype=torch.bfloat16, device=kv_st.device)
    block_hd = 128
    kernel = _make_overlap_kernel(ratio, head_dim, block_hd)
    kernel(kv_st.contiguous(), score_st.contiguous(), out)
    return out


# ─── No-overlap (ratio=128) softmax + weighted-sum ─────────────────────────


@functools.lru_cache(maxsize=8)
def _make_no_overlap_kernel(ratio: int, head_dim: int, block_hd: int):
    """Factory: tilelang JIT-compiled kernel for the no-overlap case.

    Direct layout — kv_st/score_st are [bs, ratio, head_dim] without
    rearrangement. Per-element softmax along the ratio axis, weighted sum,
    bf16 cast. ratio=128 in production DSV4 config.
    """
    assert head_dim % block_hd == 0
    n_chunks = head_dim // block_hd
    enable_pdl = is_arch_support_pdl()

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _kernel():
        bs = T.symbolic("bs")

        @T.prim_func
        def main(
            kv_st: T.Tensor[(bs, ratio, head_dim), FP32],
            score_st: T.Tensor[(bs, ratio, head_dim), FP32],
            out: T.Tensor[(bs, head_dim), BF16],
        ):
            with T.Kernel(bs, n_chunks, threads=128) as (i, chunk):
                if enable_pdl:
                    T.pdl_sync()

                h_start = chunk * block_hd

                kv_frag = T.alloc_fragment((ratio, block_hd), FP32)
                score_frag = T.alloc_fragment((ratio, block_hd), FP32)

                for ki, hh in T.Parallel(ratio, block_hd):
                    kv_frag[ki, hh] = kv_st[i, ki, h_start + hh]
                    score_frag[ki, hh] = score_st[i, ki, h_start + hh]

                # Per-element softmax along ratio axis
                score_max = T.alloc_fragment(block_hd, FP32)
                T.reduce_max(score_frag, score_max, dim=0)
                for kj, hh in T.Parallel(ratio, block_hd):
                    score_frag[kj, hh] = T.exp(score_frag[kj, hh] - score_max[hh])
                sum_exp = T.alloc_fragment(block_hd, FP32)
                T.reduce_sum(score_frag, sum_exp, dim=0)
                for kl, hh in T.Parallel(ratio, block_hd):
                    score_frag[kl, hh] = score_frag[kl, hh] / sum_exp[hh]

                # Weighted sum: kv * softmax, then reduce along ratio
                for km, hh in T.Parallel(ratio, block_hd):
                    kv_frag[km, hh] = kv_frag[km, hh] * score_frag[km, hh]
                out_frag = T.alloc_fragment(block_hd, FP32)
                T.reduce_sum(kv_frag, out_frag, dim=0)

                for hh in T.Parallel(block_hd):
                    out[i, h_start + hh] = T.cast(out_frag[hh], BF16)

        return main

    return _kernel()


def compress_no_overlap_softmax_sum(
    kv_st: torch.Tensor,  # [bs, ratio, head_dim] fp32
    score_st: torch.Tensor,  # [bs, ratio, head_dim] fp32
) -> torch.Tensor:
    """Drop-in replacement for the no-overlap softmax+weighted-sum chain.

    Returns ``[bs, head_dim]`` bfloat16.
    """
    bs, ratio, head_dim = kv_st.shape
    out = torch.empty(bs, head_dim, dtype=torch.bfloat16, device=kv_st.device)
    # ratio=128 with block_hd=64 keeps the per-program tile at 128*64 fp32 = 32 KB.
    block_hd = 64 if ratio >= 64 else head_dim
    kernel = _make_no_overlap_kernel(ratio, head_dim, block_hd)
    kernel(kv_st.contiguous(), score_st.contiguous(), out)
    return out

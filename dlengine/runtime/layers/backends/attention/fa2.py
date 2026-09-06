"""FlashAttention-2 GQA attention backend (with optional FlashInfer paged path).

Owns the FA2 kernel calls (``flash_attn_varlen_func`` prefill,
``flash_attn_with_kvcache`` paged decode). When constructed with
``force_flashinfer_decode`` (the ``auto`` hybrid on pre-SM90 GPUs), it uses the
FlashInfer paged prefill/decode kernels where available and falls back to FA2.
"""

import torch

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _HAS_FA2 = True
except ImportError:
    flash_attn_varlen_func = None  # type: ignore
    flash_attn_with_kvcache = None  # type: ignore
    _HAS_FA2 = False

from dlengine.runtime.layers.backends.attention.base import (
    GqaAttentionBase,
    _flashinfer_decode,
    _flashinfer_enabled,
    _flashinfer_prefill_enabled,
    _flashinfer_prefill_paged,
    _sdpa_varlen_func,
)


class Fa2Attention(GqaAttentionBase):
    """GQA attention using FlashAttention-2, with an optional FlashInfer path."""

    def __init__(self, *args, force_flashinfer_decode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if not _HAS_FA2:
            raise RuntimeError(
                "FlashAttention-2 is required for the fa2 attention backend. "
                "Install ``flash_attn`` (FA2) or select the generic (SDPA) "
                "backend."
            )
        # ``auto`` selection on pre-SM90 pairs FA2 prefill with FlashInfer
        # decode; explicit ``fa2`` disables the FlashInfer path.
        self.use_flashinfer_decode = force_flashinfer_decode
        self.use_flashinfer_prefill = force_flashinfer_decode

    def _attend_prefill_paged(
        self, q, k_cache, v_cache, block_tables, num_seqs, context
    ):
        if (
            self.has_flashinfer
            and self.use_flashinfer_prefill
            and _flashinfer_prefill_enabled()
        ):
            return _flashinfer_prefill_paged(
                q,
                k_cache,
                v_cache,
                block_tables,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                num_seqs,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                k_cache.shape[1],
                self.scale,
            )
        return None

    def _attend_prefill_varlen(self, q, k, v, context):
        attn_kwargs = {}
        if self.sliding_window is not None:
            attn_kwargs["window_size"] = (int(self.sliding_window) - 1, 0)
        # FA2 does not instantiate head_dim > 256; use the SDPA reference there.
        if self.head_dim > 256:
            return _sdpa_varlen_func(
                q,
                k,
                v,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                self.scale,
                sliding_window=self.sliding_window,
            )
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            **attn_kwargs,
        )

    def _attend_decode_paged(
        self, q, k_cache, v_cache, block_tables, context_lens, bs, ntps, context
    ):
        total_tokens, num_head, head_dim = q.shape
        if (
            self.has_flashinfer
            and self.use_flashinfer_decode
            and _flashinfer_enabled()
            and ntps == 1
        ):
            return _flashinfer_decode(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                bs,
                ntps,
                num_head,
                self.num_kv_heads,
                head_dim,
                k_cache.shape[1],
                self.scale,
            )

        out = flash_attn_with_kvcache(
            q.reshape(bs, ntps, num_head, head_dim),
            k_cache,
            v_cache,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=self.scale,
            causal=ntps > 1,
            # NOTE: ``return_softmax_lse`` intentionally omitted — the LSE is
            # unused here (only ``out`` is consumed) and some flash-attn builds
            # (e.g. the PPU runtime) reject that keyword.
        )
        o = out[0] if isinstance(out, tuple) else out
        if ntps > 1:
            o = o.reshape(total_tokens, num_head, head_dim)
        return o


__all__ = ["Fa2Attention"]

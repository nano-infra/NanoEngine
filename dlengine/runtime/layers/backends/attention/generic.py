"""Generic (pure SDPA) GQA attention backend.

The portable, correctness-first GQA backend: it uses only
``torch.nn.functional.scaled_dot_product_attention`` — no ``flash_attn`` and no
``flashinfer``. It is the ``ref_fallback_allowed`` target for GQA and the debug
backend selected by ``attention_backend=generic`` (the former ``torch`` alias).

MLA is not implemented here — MLA lives in ``backends/mla/``. Shared GQA
plumbing (KV store, HiSparse SWA, cached-prefill gather) is in
``attention/base.py``.
"""

from dlengine.runtime.layers.backends.attention.base import (
    GqaAttentionBase,
    _sdpa_fixed_decode,
    _sdpa_varlen_func,
)
from dlengine.runtime.layers.backends.attention.mla_utils import (
    _gather_kv_cached_concat,  # noqa: F401  (re-exported for back-compat)
)


class GenericAttention(GqaAttentionBase):
    """Pure-SDPA GQA attention (no FlashAttention / FlashInfer kernels)."""

    def _attend_prefill_paged(
        self, q, k_cache, v_cache, block_tables, num_seqs, context
    ):
        # No paged prefill kernel: fall back to gather + SDPA varlen.
        return None

    def _attend_prefill_varlen(self, q, k, v, context):
        return _sdpa_varlen_func(
            q,
            k,
            v,
            context.cu_seqlens_q,
            context.cu_seqlens_k,
            self.scale,
            sliding_window=self.sliding_window,
        )

    def _attend_decode_paged(
        self, q, k_cache, v_cache, block_tables, context_lens, bs, ntps, context
    ):
        # Pure SDPA has no paged decode kernel; the base class raises for paged
        # decode. Serving decode must select fa2 or flashinfer.
        return None


__all__ = ["GenericAttention"]

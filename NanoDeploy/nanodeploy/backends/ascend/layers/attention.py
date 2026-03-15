"""Ascend NPU GQA attention implementation.

Uses torch_npu npu_fused_infer_attention_score (TND layout, sparse_mode=3):
  - Prefill: TND, block_table=None, actual_seq_lengths = cumulative q seqlens
  - Decode:  TND, k/v_cache reshaped to [blocks, block_size, nkv*dim],
             actual_seq_lengths = cumulative decode tokens (1 per seq)

This matches the calling convention used by vllm-ascend.
Sequence Parallelism (sp > 1) is not supported in this version.
"""

import torch
from torch import nn

from nanodeploy.backends.base_backend import AttentionBase
from nanodeploy.backends.ascend.ops.kv_ops import store_kvcache_npu
from nanodeploy.context.context import get_context
from nanodeploy.logging import get_logger

logger = get_logger()

# Pre-computed 2048×2048 causal mask (upper triangle = 1, lower = 0).
# sparse_mode=3 uses this as an additive bias; positions where mask=1 are
# effectively masked out. Shape matches vllm-ascend convention.
_CAUSAL_MASK_CACHE: torch.Tensor | None = None


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    global _CAUSAL_MASK_CACHE
    if _CAUSAL_MASK_CACHE is None or _CAUSAL_MASK_CACHE.device != device:
        _CAUSAL_MASK_CACHE = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(device)
        )
    return _CAUSAL_MASK_CACHE


class AscendAttention(AttentionBase):
    """Ascend NPU GQA attention using npu_fused_infer_attention_score (TND)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        # Placeholders: assigned by model_runner.allocate_kvcache
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()

        # Store new tokens into paged KV cache (skip for dummy warmup)
        if self.k_cache.numel() and self.v_cache.numel() and not context.is_dummy:
            store_kvcache_npu(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            return self._forward_prefill(q, k, v, context)
        else:
            return self._forward_decode(q, context)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Prefill via npu_fused_infer_attention_score, TND layout, sparse_mode=3.

        Args:
            q: [total_tokens, num_heads, head_dim]   (TND — T=total, N=heads, D=dim)
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
        """
        import torch_npu

        # cu_seqlens_q: [num_seqs+1] cumulative token offsets.
        # actual_seq_lengths for TND must be CUMULATIVE (not per-seq lengths).
        cu_seqlens_q = context.cu_seqlens_q  # [num_seqs+1]
        actual_seq_lengths_q = cu_seqlens_q[1:].tolist()

        if context.cu_seqlens_k is not None:
            actual_seq_lengths_kv = context.cu_seqlens_k[1:].tolist()
        else:
            actual_seq_lengths_kv = actual_seq_lengths_q

        total_tokens = int(cu_seqlens_q[-1].item())
        atten_mask = _get_causal_mask(q.device)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q[:total_tokens].contiguous(),
            key=k[:total_tokens].contiguous(),
            value=v[:total_tokens].contiguous(),
            atten_mask=atten_mask,
            block_table=None,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        # attn_output: [total_tokens * num_heads * head_dim] or [total_tokens, num_heads, head_dim]
        return attn_output.view(total_tokens, self.num_heads, self.head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Decode via npu_fused_infer_attention_score, TND layout, sparse_mode=3.

        Args:
            q: [bs, num_heads, head_dim]  (TND: one query token per sequence)
        """
        import torch_npu

        bs = q.shape[0]
        compute_bs = getattr(context, "attention_compute_bs", bs) or bs
        q_compute = q[:compute_bs]

        context_lens = context.context_lens_for_attn[:compute_bs]
        block_tables = context.block_tables[:compute_bs]

        # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        # Reshape to [num_blocks, block_size, num_kv_heads*head_dim] for TND paged mode
        num_blocks, block_size, nkv, kdim = self.k_cache.shape
        k_flat = self.k_cache.view(num_blocks, block_size, nkv * kdim)
        v_flat = self.v_cache.view(num_blocks, block_size, nkv * self.v_head_dim)

        # actual_seq_lengths for TND decode: cumulative (1 query per sequence)
        actual_seq_lengths_q = list(range(1, compute_bs + 1))
        actual_seq_lengths_kv = context_lens.tolist()

        atten_mask = _get_causal_mask(q_compute.device)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q_compute.contiguous(),
            key=k_flat,
            value=v_flat,
            atten_mask=atten_mask,
            block_table=block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        # attn_output: [compute_bs, num_heads, head_dim]
        out = attn_output.view(compute_bs, self.num_heads, self.head_dim)

        # Pad back to original bs if needed (SP padding)
        if compute_bs < bs:
            pad = torch.zeros(
                bs - compute_bs,
                self.num_heads,
                self.head_dim,
                dtype=out.dtype,
                device=out.device,
            )
            out = torch.cat([out, pad], dim=0)

        return out

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
        """Decode attention via pre-gathered KV + npu_incre_flash_attention (no block_table).

        Both npu_fused_infer_attention_score and npu_incre_flash_attention trigger
        AclrtSynchronizeStreamWithTimeout on the copy_stream when using block_table,
        which is forbidden inside ACL graph capture (CANN limitation).

        Work-around: pre-gather the paged KV blocks into a dense [B, N, S_kv, D]
        tensor using pure index ops (all static output shapes) and call the attention
        op without block_table.  Every intermediate shape is known at compile time
        from tensor .shape attributes → fully ACL-graph safe.

        Args:
            q: [bs, num_heads, head_dim]
        """
        import torch_npu

        bs = q.shape[0]
        compute_bs = getattr(context, "attention_compute_bs", bs) or bs
        q_compute = q[:compute_bs]                             # [B, num_heads, head_dim]

        context_lens = context.context_lens_for_attn[:compute_bs]  # [B], stays on NPU
        block_tables = context.block_tables[:compute_bs]            # [B, max_blocks]

        n_blks_total, block_size, nkv, kdim = self.k_cache.shape
        max_blocks = block_tables.shape[1]                     # compile-time const
        max_kv_len = max_blocks * block_size                   # compile-time const

        # Pre-gather KV from paged cache — all output shapes are static
        # positions: [max_kv_len] — reused for both slot computation and mask
        positions = torch.arange(max_kv_len, device=q_compute.device)
        blk_idx = positions // block_size                      # [max_kv_len]
        blk_off = positions % block_size                       # [max_kv_len]

        # slots[i, p] = flat cache index for (sequence i, KV position p)
        # block_tables[:, blk_idx]: [B, max_kv_len] — static shape
        slots = (
            block_tables[:, blk_idx] * block_size + blk_off.unsqueeze(0)
        ).clamp(0, n_blks_total * block_size - 1)              # [B, max_kv_len]

        # Gather: [B, max_kv_len, nkv, kdim] — static shape
        k_flat = self.k_cache.view(-1, nkv, kdim)
        v_flat = self.v_cache.view(-1, nkv, self.v_head_dim)
        k_seq = k_flat[slots]                                  # [B, max_kv_len, nkv, kdim]
        v_seq = v_flat[slots]                                  # [B, max_kv_len, nkv, v_head_dim]

        # Transpose to BNSD: [B, nkv, max_kv_len, kdim]
        k_bnsd = k_seq.permute(0, 2, 1, 3).contiguous()
        v_bnsd = v_seq.permute(0, 2, 1, 3).contiguous()

        # query BNSD: [B, num_heads, 1, head_dim]
        q_bnsd = q_compute.unsqueeze(2)

        # Build boolean attention mask instead of using actual_seq_lengths.
        # Passing actual_seq_lengths as a device tensor causes CANN to read it via the
        # copy_stream (host-side kernel configuration), which triggers
        # AclrtSynchronizeStreamWithTimeout — forbidden inside ACL graph capture.
        # A bool atten_mask is a normal on-device tensor: no copy_stream sync.
        # npu_incre_flash_attention accepts bool/int8/uint8 for atten_mask;
        # True = masked out (invalid/padding), False = attend to.
        # valid[i, p] = True if position p is within sequence i's KV history.
        valid = positions.unsqueeze(0) < context_lens.to(torch.int64).view(compute_bs, 1)
        # atten_mask: [B, 1, 1, max_kv_len] bool — True for padding positions
        atten_mask = (~valid).unsqueeze(1).unsqueeze(2)

        attn_output = torch_npu.npu_incre_flash_attention(
            query=q_bnsd,
            key=k_bnsd,
            value=v_bnsd,
            atten_mask=atten_mask,
            num_heads=self.num_heads,
            scale_value=self.scale,
            input_layout="BNSD",
            num_key_value_heads=self.num_kv_heads,
            # actual_seq_lengths intentionally omitted: device tensor → copy_stream sync
        )
        # attn_output: [B, num_heads, 1, head_dim]
        out = attn_output.squeeze(2)                           # [B, num_heads, head_dim]

        # Pad back to original bs if needed (SP padding)
        if compute_bs < bs:
            pad = torch.zeros(
                bs - compute_bs, self.num_heads, self.head_dim,
                dtype=out.dtype, device=out.device,
            )
            out = torch.cat([out, pad], dim=0)

        return out

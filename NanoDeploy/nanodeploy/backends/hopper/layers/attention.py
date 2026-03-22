import flash_mla
import torch
from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanodeploy.backends.base_backend import AttentionBase
from nanodeploy.backends.gpu_generic.kernels.copy import copy_batch_indexed_triton
from nanodeploy.backends.gpu_generic.kernels.kv_combine import (
    inter_rank_gqa_fwd_batch_decode_combine_kv,
)
from nanodeploy.backends.gpu_generic.kernels.kv_store import store_kcache, store_kvcache
from nanodeploy.context.context import get_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.context.sp_context import get_sp_context
from nanodeploy.logging import get_logger

logger = get_logger()


def _build_paged_gather_indices(
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build flat linear indices for gathering tokens from a paged cache.

    Args:
        block_table: [num_seqs, max_num_blocks] — int32 physical block IDs
        cu_seqlens_k:[num_seqs + 1] — cumulative K lengths (int32)
        block_size:  tokens per cache block

    Returns:
        linear_indices: [total_k_tokens] — index into cache.reshape(-1, ...)
    """
    num_seqs = block_table.shape[0]
    total_k = int(cu_seqlens_k[-1].item())
    device = block_table.device

    linear_indices = torch.empty(total_k, dtype=torch.int64, device=device)

    for i in range(num_seqs):
        start = int(cu_seqlens_k[i].item())
        end = int(cu_seqlens_k[i + 1].item())
        seqlen = end - start
        if seqlen == 0:
            continue
        t = torch.arange(seqlen, dtype=torch.int64, device=device)
        block_ids = block_table[i, t // block_size].to(torch.int64)
        offsets = t % block_size
        linear_indices[start:end] = block_ids * block_size + offsets

    return linear_indices


def _gather_kv_paged(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV cache into contiguous ragged tensors for flash_attn_varlen_func.

    Args:
        k_cache:     [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache:     [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [num_seqs, max_num_blocks] — int32 physical block IDs (-1 = padding)
        cu_seqlens_k:[num_seqs + 1] — cumulative K lengths (int32)
        block_size:  tokens per KV block

    Returns:
        k_gathered: [total_k_tokens, num_kv_heads, head_dim]
        v_gathered: [total_k_tokens, num_kv_heads, head_dim]
    """
    linear_indices = _build_paged_gather_indices(block_table, cu_seqlens_k, block_size)

    _, _, num_kv_heads, head_dim = k_cache.shape
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim)
    return k_flat[linear_indices], v_flat[linear_indices]


def _gather_cache_paged(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather a single paged cache into a contiguous ragged tensor.

    Works for any cache with shape [num_blocks, block_size, ...].

    Returns:
        gathered: [total_k_tokens, ...] (remaining dims preserved)
    """
    linear_indices = _build_paged_gather_indices(block_table, cu_seqlens_k, block_size)
    trailing_shape = cache.shape[2:]
    flat = cache.reshape(-1, *trailing_shape)
    return flat[linear_indices]


class FlashAttentionImpl:

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        context = get_context()
        if k_cache.numel() and v_cache.numel() and not get_context().is_dummy:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        if context.is_prefill:
            if context.block_tables is not None:
                num_seqs = context.cu_seqlens_k.shape[0] - 1
                bt = context.block_tables[sp_rank, :num_seqs, :]
                k, v = _gather_kv_paged(
                    k_cache,
                    v_cache,
                    bt,
                    context.cu_seqlens_k,
                    k_cache.shape[1],  # block_size
                )
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode
            bs, num_head, head_dim = q.shape
            if sp_size > 1:
                max_num_seqs = get_sp_context().max_num_seqs
                q_buffer = get_sp_context().q_buffer

                # Q copy
                local_q_buffer_3d = q_buffer.local_buffer.view(get_sp_context().dtype)[
                    : sp_size * max_num_seqs * num_head * head_dim
                ].view(sp_size * max_num_seqs, num_head, head_dim)
                copy_batch_indexed_triton(
                    q,
                    local_q_buffer_3d,
                    context.q_slice_get,
                    context.q_slice_fill,
                    context.q_copy_mask,
                )

                # Compute offsets from q_output_stride for contiguous layout
                # Compute offsets from q_output_stride for contiguous layout
                # q_offsets = torch.zeros(sp_size + 1, dtype=torch.int32, device=context.q_output_stride.device)
                # q_offsets[1:] = torch.cumsum(context.q_output_stride, dim=0)

                q = q_buffer.all_to_all_ll(
                    q.view([bs, -1]),
                    mask=context.q_mask,
                    offsets=context.q_offsets,
                ).view([sp_size * max_num_seqs, num_head, head_dim])

                q = q[: context.attention_compute_bs]
                context_lens = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]
            else:
                context_lens = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]

            o, lse = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                page_table=block_tables,
                softmax_scale=self.scale,
                causal=False,
                return_softmax_lse=True,
            )[:2]

            if sp_size > 1:
                res_buffer = get_sp_context().res_buffer
                lse_buffer = get_sp_context().lse_buffer
                lse = lse.to(torch.bfloat16)
                gathered_o = o.view([context.attention_compute_bs, num_head, head_dim])
                gathered_lse = lse.view([context.attention_compute_bs, num_head, 1])

                # 1. 拷贝 gathered_o 到 res_local_buffer
                res_local_buffer_3d = res_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * head_dim].view(
                    sp_size * max_num_seqs, num_head, head_dim
                )
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 2. Copy gathered_lse to lse_local_buffer
                lse_local_buffer_3d = lse_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * 1].view(
                    sp_size * max_num_seqs, num_head, 1
                )
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 3. Allocate All-to-All Input Buffer
                res_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, head_dim),
                    dtype=gathered_o.dtype,
                    device=gathered_o.device,
                )
                lse_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, 1),
                    dtype=gathered_lse.dtype,
                    device=gathered_lse.device,
                )

                # 4. Copy gathered_o to res_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                # 5. Copy gathered_lse to lse_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                all_ranks_res_output_combine = res_buffer.all_to_all_ll(
                    res_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, head_dim)
                all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
                    lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_res_output_combine,
                    all_ranks_lse_output_combine,
                    context.global_context_lens,
                    num_head,
                    head_dim,
                    get_sp_context().max_num_seqs,
                    sp_size,
                ).view([max_num_seqs, num_head, head_dim])[:bs]

        return o


class FlashMLAImpl:
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        causal: bool = True,
        **kwargs,
    ):
        if scale is None:
            scale = 1.0 / (head_size**0.5)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if v_head_size is None:
            v_head_size = head_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_size = v_head_size
        self.causal = causal

        assert num_kv_heads == 1, "MLA requires num kv heads equal to 1"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):

        context = get_context()
        if k_cache.numel() and not get_context().is_dummy:
            store_kcache(k, k_cache, context.slot_mapping)

        # --- DEBUG removed (too noisy with 61 layers) ---

        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size

        if context.is_prefill:
            # NOTE: MLA prefill is handled directly in DeepseekV2Attention.forward
            # using the non-absorbed approach (expanded K/V). This path should not
            # be reached for MLA models.
            raise RuntimeError(
                "FlashMLAImpl.forward should not be called during prefill. "
                "MLA prefill is handled in DeepseekV2Attention.forward."
            )

        else:  # decode
            bs, num_head, head_dim = q.shape
            if sp_size > 1:
                max_num_seqs = get_sp_context().max_num_seqs
                q_buffer = get_sp_context().q_buffer

                local_q_buffer_3d = q_buffer.local_buffer.view(get_sp_context().dtype)[
                    : sp_size * max_num_seqs * num_head * head_dim
                ].view(sp_size * max_num_seqs, num_head, head_dim)
                copy_batch_indexed_triton(
                    q.view(bs, num_head, head_dim),
                    local_q_buffer_3d,
                    context.q_slice_get,
                    context.q_slice_fill,
                    context.q_copy_mask,
                )

                q = q_buffer.all_to_all_ll(
                    q.view([bs, -1]),
                    mask=context.q_mask,
                ).view([sp_size * max_num_seqs, num_head, head_dim])

                q = q[: context.attention_compute_bs]
                context_lens = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]
            else:
                q = q[: context.attention_compute_bs]
                context_lens = context.context_lens_for_attn[
                    : context.attention_compute_bs
                ]
                block_tables = context.block_tables[: context.attention_compute_bs]

            batch_size = q.shape[0]

            if context.tile_scheduler_metadata is not None:
                # Use precomputed metadata from prepare_decode (CUDA graph compatible)
                tile_scheduler_metadata = context.tile_scheduler_metadata
                # num_splits must have shape (batch_size + 1); slice buffer to match
                num_splits = context.num_splits[: batch_size + 1]
            else:
                # Fallback: compute inline
                tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
                    context_lens,
                    self.num_heads // self.num_kv_heads,
                    self.num_kv_heads,
                )

            o, lse = flash_mla.flash_mla_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                block_tables,
                context_lens,
                self.v_head_size,
                tile_scheduler_metadata,
                num_splits,
                self.scale,
                self.causal,
            )

            o = o.squeeze(1)

            if sp_size > 1:
                _, num_head, v_head_dim = o.shape

                res_buffer = get_sp_context().res_buffer
                lse_buffer = get_sp_context().lse_buffer
                lse = lse.to(torch.bfloat16)
                gathered_o = o.view(
                    [context.attention_compute_bs, num_head, v_head_dim]
                )
                gathered_lse = lse.view([context.attention_compute_bs, num_head, 1])

                # 1. 拷贝 gathered_o 到 res_local_buffer
                res_local_buffer_3d = res_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * v_head_dim].view(
                    sp_size * max_num_seqs, num_head, v_head_dim
                )

                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, v_head_dim),
                    res_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 2. 拷贝 gathered_lse 到 lse_local_buffer
                lse_local_buffer_3d = lse_buffer.local_buffer.view(
                    get_sp_context().dtype
                )[: sp_size * max_num_seqs * num_head * 1].view(
                    sp_size * max_num_seqs, num_head, 1
                )

                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    context.res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 3. 分配 All-to-All Input Buffer
                res_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, v_head_dim),
                    dtype=gathered_o.dtype,
                    device=gathered_o.device,
                )
                lse_all_to_all_input_buffer = torch.empty(
                    (sp_size * max_num_seqs, num_head, 1),
                    dtype=gathered_lse.dtype,
                    device=gathered_lse.device,
                )

                # 4. 拷贝 gathered_o 到 res_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, v_head_dim),
                    res_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                # 5. 拷贝 gathered_lse 到 lse_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    context.res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                all_ranks_res_output_combine = res_buffer.all_to_all_ll(
                    res_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, v_head_dim)
                all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
                    lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
                    mask=context.res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, max_num_seqs, num_head, 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_res_output_combine,
                    all_ranks_lse_output_combine,
                    context.global_context_lens,
                    num_head,
                    v_head_dim,
                    get_sp_context().max_num_seqs,
                    sp_size,
                ).view([max_num_seqs, num_head, v_head_dim])[:bs]

        return o


class HopperAttention(AttentionBase):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        v_head_dim,
        attention_type: str = "MLA",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.forward_method = None

        if attention_type == "MLA":
            self.impl = FlashMLAImpl(
                num_heads, head_dim, scale, num_kv_heads, v_head_dim
            )
        elif attention_type == "GQA":
            self.impl = FlashAttentionImpl(
                num_heads,
                head_dim,
                scale,
                num_kv_heads,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """forward."""
        return self.impl.forward(q, k, v, self.k_cache, self.v_cache)

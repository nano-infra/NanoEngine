import flash_mla
import torch

from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from nanodeploy.kernels.attention import inter_rank_gqa_fwd_batch_decode_combine_kv
from nanodeploy.kernels.copy import (
    copy_batch_indexed_triton,
    zero_padded_rows_triton,
)
from nanodeploy.kernels.kvcache import store_kcache, store_kvcache
from nanodeploy.logging import get_logger
from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from nanodeploy.worker.sp_context import get_sp_context

from torch import nn


logger = get_logger()


def _uses_nccl_comm_bs(sp_context) -> bool:
    return sp_context.backend == "nccl"


def _get_sp_comm_bs(sp_context, context) -> int:
    if _uses_nccl_comm_bs(sp_context) and context.sp_comm_bs is not None:
        return int(context.sp_comm_bs)
    return sp_context.max_num_seqs


def _narrow_sp_matrix_for_comm(tensor: torch.Tensor, comm_bs: int) -> torch.Tensor:
    if tensor.size(1) == comm_bs:
        return tensor
    return tensor[:, :comm_bs]


def _initialize_hao_graph_q_padding(q: torch.Tensor, context) -> None:
    actual_attn_bs = context.actual_attn_bs
    if actual_attn_bs is None:
        return
    zero_padded_rows_triton(
        q,
        actual_attn_bs,
        context.attention_compute_bs,
    )


def _remap_sp_stride_indices(
    indices: torch.Tensor,
    *,
    old_stride: int,
    new_stride: int,
) -> torch.Tensor:
    if old_stride == new_stride:
        return indices
    valid = indices >= 0
    ranks = torch.div(indices, old_stride, rounding_mode="floor")
    seq_ids = indices - ranks * old_stride
    remapped = ranks * new_stride + seq_ids
    return torch.where(valid, remapped, indices)


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
        use_sp_a2a = sp_size > 1 and context.use_sp_a2a
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
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
            if use_sp_a2a:
                sp_context = get_sp_context()
                max_num_seqs = sp_context.max_num_seqs
                comm_bs = _get_sp_comm_bs(sp_context, context)
                q_mask = (
                    _narrow_sp_matrix_for_comm(context.q_mask, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.q_mask
                )
                q_dst_row_indices = (
                    context.q_dst_row_indices
                    if sp_context.backend == "hao_basic"
                    else None
                )
                q_buffer = sp_context.q_buffer

                # Q copy
                local_q_buffer_3d = q_buffer.local_buffer.view(sp_context.dtype)[
                    : sp_size * comm_bs * num_head * head_dim
                ].view(sp_size * comm_bs, num_head, head_dim)
                if _uses_nccl_comm_bs(sp_context):
                    local_q_buffer_3d.zero_()
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
                    mask=None if q_dst_row_indices is not None else q_mask,
                    offsets=None if q_dst_row_indices is not None else context.q_offsets,
                    dst_row_indices=q_dst_row_indices,
                ).view([sp_size * comm_bs, num_head, head_dim])

                _initialize_hao_graph_q_padding(q, context)

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

            if use_sp_a2a:
                sp_context = get_sp_context()
                comm_bs = _get_sp_comm_bs(sp_context, context)
                res_lse_mask = (
                    _narrow_sp_matrix_for_comm(context.res_lse_mask, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_lse_mask
                )
                global_context_lens = (
                    _narrow_sp_matrix_for_comm(context.global_context_lens, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.global_context_lens
                )
                res_slice_fill_to_buffer_output = (
                    _remap_sp_stride_indices(
                        context.res_slice_fill_to_buffer_output,
                        old_stride=max_num_seqs,
                        new_stride=comm_bs,
                    )
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_slice_fill_to_buffer_output
                )
                res_slice_fill_to_buffer_input = (
                    _remap_sp_stride_indices(
                        context.res_slice_fill_to_buffer_input,
                        old_stride=max_num_seqs,
                        new_stride=comm_bs,
                    )
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_slice_fill_to_buffer_input
                )
                res_buffer = sp_context.res_buffer
                lse_buffer = sp_context.lse_buffer
                # FlashAttention returns LSE in FP32.  Preserve that precision
                # across SP communication because the inter-rank combine uses
                # LSE to reconstruct the global softmax weights.
                gathered_o = o.view([context.attention_compute_bs, num_head, head_dim])
                gathered_lse = lse.view([context.attention_compute_bs, num_head, 1])

                # 1. 拷贝 gathered_o 到 res_local_buffer
                res_local_buffer_3d = res_buffer.local_buffer.view(
                    sp_context.dtype
                )[: sp_size * comm_bs * num_head * head_dim].view(
                    sp_size * comm_bs, num_head, head_dim
                )
                if _uses_nccl_comm_bs(sp_context):
                    res_local_buffer_3d.zero_()
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 2. Copy gathered_lse to lse_local_buffer
                lse_local_buffer_3d = lse_buffer.local_buffer.view(
                    gathered_lse.dtype
                )[: sp_size * comm_bs * num_head * 1].view(
                    sp_size * comm_bs, num_head, 1
                )
                if _uses_nccl_comm_bs(sp_context):
                    lse_local_buffer_3d.zero_()
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 3. Allocate All-to-All Input Buffer
                res_all_to_all_input_buffer = torch.empty(
                    (sp_size * comm_bs, num_head, head_dim),
                    dtype=gathered_o.dtype,
                    device=gathered_o.device,
                )
                lse_all_to_all_input_buffer = torch.empty(
                    (sp_size * comm_bs, num_head, 1),
                    dtype=gathered_lse.dtype,
                    device=gathered_lse.device,
                )

                # 4. Copy gathered_o to res_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, head_dim),
                    res_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                # 5. Copy gathered_lse to lse_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                all_ranks_res_output_combine = res_buffer.all_to_all_ll(
                    res_all_to_all_input_buffer.view(sp_size * comm_bs, -1),
                    mask=res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, comm_bs, num_head, head_dim)
                all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
                    lse_all_to_all_input_buffer.view(sp_size * comm_bs, -1),
                    mask=res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, comm_bs, num_head, 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_res_output_combine,
                    all_ranks_lse_output_combine,
                    global_context_lens,
                    num_head,
                    head_dim,
                    comm_bs,
                    sp_size,
                ).view([comm_bs, num_head, head_dim])[:bs]
        
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

    def forward_prefill_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_k: torch.Tensor,
        k_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run non-absorbed MLA prefill and store compressed KV for decode."""
        context = get_context()
        if k_cache.numel() and not context.is_dummy:
            store_kcache(cache_k, k_cache, context.slot_mapping)

        if context.is_dummy:
            return q.new_zeros((q.shape[0], self.num_heads, v.shape[-1]))
        if context.prefill_has_prefix:
            raise RuntimeError(
                "Native MLA prefill only supports batches without cached prefixes"
            )
        if context.cu_seqlens_q is None or context.cu_seqlens_k is None:
            raise RuntimeError("Native MLA prefill requires varlen sequence offsets")

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output

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

        sp_rank = get_dist_context().attn_sp_rank
        sp_size = get_dist_context().attn_sp_world_size
        use_sp_a2a = sp_size > 1 and context.use_sp_a2a

        if context.is_prefill:
            if context.is_dummy:
                # Empty DP lanes still execute the transformer/MoE collectives,
                # but they own no KV blocks and their sampled output is ignored.
                return q.new_zeros((q.shape[0], self.num_heads, self.v_head_size))

            if context.block_tables is None:
                raise RuntimeError("MLA prefill requires local paged block tables")
            cu_seqlens_q_host = context.prefill_cu_seqlens_q_host
            if cu_seqlens_q_host is None:
                raise RuntimeError("MLA prefill requires host query offsets")

            num_seqs = len(cu_seqlens_q_host) - 1
            if context.block_tables.ndim != 2 or context.block_tables.shape[0] != num_seqs:
                raise RuntimeError(
                    "MLA prefill block-table batch mismatch: "
                    f"block_tables={tuple(context.block_tables.shape)}, "
                    f"num_seqs={num_seqs}"
                )

            context_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
            output_by_seq: list[torch.Tensor | None] = [None] * num_seqs
            groups: dict[int, list[tuple[int, int, int]]] = {}
            for seq_idx, (start, end) in enumerate(
                zip(
                    cu_seqlens_q_host[:-1],
                    cu_seqlens_q_host[1:],
                    strict=True,
                )
            ):
                query_len = end - start
                if query_len <= 0:
                    raise RuntimeError(
                        f"MLA prefill sequence {seq_idx} has no query tokens"
                    )
                groups.setdefault(query_len, []).append((seq_idx, start, end))

            for query_len, group in groups.items():
                seq_indices = [item[0] for item in group]
                q_batch = torch.stack([q[start:end] for _, start, end in group])
                group_context_lens = context_lens[seq_indices].contiguous()
                group_block_tables = context.block_tables[seq_indices].contiguous()
                tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
                    group_context_lens,
                    query_len * self.num_heads // self.num_kv_heads,
                    self.num_kv_heads,
                )
                group_output, _ = flash_mla.flash_mla_with_kvcache(
                    q_batch,
                    k_cache,
                    group_block_tables,
                    group_context_lens,
                    self.v_head_size,
                    tile_scheduler_metadata,
                    num_splits,
                    self.scale,
                    self.causal,
                )
                for group_idx, seq_idx in enumerate(seq_indices):
                    output_by_seq[seq_idx] = group_output[group_idx]

            if any(output is None for output in output_by_seq):
                raise RuntimeError("MLA prefill did not produce every sequence output")
            return torch.cat(output_by_seq, dim=0)

        else:  # decode
            bs, num_head, head_dim = q.shape
            if use_sp_a2a:
                sp_context = get_sp_context()
                max_num_seqs = sp_context.max_num_seqs
                comm_bs = _get_sp_comm_bs(sp_context, context)
                q_mask = (
                    _narrow_sp_matrix_for_comm(context.q_mask, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.q_mask
                )
                q_dst_row_indices = (
                    context.q_dst_row_indices
                    if sp_context.backend == "hao_basic"
                    else None
                )
                q_buffer = sp_context.q_buffer

                local_q_buffer_3d = q_buffer.local_buffer.view(sp_context.dtype)[
                    : sp_size * comm_bs * num_head * head_dim
                ].view(sp_size * comm_bs, num_head, head_dim)
                if _uses_nccl_comm_bs(sp_context):
                    local_q_buffer_3d.zero_()
                copy_batch_indexed_triton(
                    q.view(bs, num_head, head_dim),
                    local_q_buffer_3d,
                    context.q_slice_get,
                    context.q_slice_fill,
                    context.q_copy_mask,
                )

                q = q_buffer.all_to_all_ll(
                    q.view([bs, -1]),
                    mask=None if q_dst_row_indices is not None else q_mask,
                    offsets=None if q_dst_row_indices is not None else context.q_offsets,
                    dst_row_indices=q_dst_row_indices,
                ).view([sp_size * comm_bs, num_head, head_dim])

                _initialize_hao_graph_q_padding(q, context)

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

            tile_scheduler_metadata = context.tile_scheduler_metadata
            num_splits = context.num_splits
            if tile_scheduler_metadata is None or num_splits is None:
                raise RuntimeError(
                    "FlashMLA decode metadata must be prepared once before "
                    "running transformer layers"
                )
            expected_num_splits = context_lens.numel() + 1
            if num_splits.numel() != expected_num_splits:
                raise RuntimeError(
                    "FlashMLA decode split metadata has the wrong batch shape: "
                    f"got={num_splits.numel()} expected={expected_num_splits}"
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

            if use_sp_a2a:
                _, num_head, v_head_dim = o.shape

                sp_context = get_sp_context()
                comm_bs = _get_sp_comm_bs(sp_context, context)
                res_lse_mask = (
                    _narrow_sp_matrix_for_comm(context.res_lse_mask, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_lse_mask
                )
                global_context_lens = (
                    _narrow_sp_matrix_for_comm(context.global_context_lens, comm_bs)
                    if _uses_nccl_comm_bs(sp_context)
                    else context.global_context_lens
                )
                res_slice_fill_to_buffer_output = (
                    _remap_sp_stride_indices(
                        context.res_slice_fill_to_buffer_output,
                        old_stride=max_num_seqs,
                        new_stride=comm_bs,
                    )
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_slice_fill_to_buffer_output
                )
                res_slice_fill_to_buffer_input = (
                    _remap_sp_stride_indices(
                        context.res_slice_fill_to_buffer_input,
                        old_stride=max_num_seqs,
                        new_stride=comm_bs,
                    )
                    if _uses_nccl_comm_bs(sp_context)
                    else context.res_slice_fill_to_buffer_input
                )
                res_buffer = sp_context.res_buffer
                lse_buffer = sp_context.lse_buffer
                # FlashMLA returns LSE in FP32.  Quantizing it to the model
                # dtype perturbs the weights used to merge partial attention
                # outputs from different SP ranks.
                gathered_o = o.view(
                    [context.attention_compute_bs, num_head, v_head_dim]
                )
                gathered_lse = lse.view([context.attention_compute_bs, num_head, 1])

                # 1. 拷贝 gathered_o 到 res_local_buffer
                res_local_buffer_3d = res_buffer.local_buffer.view(
                    sp_context.dtype
                )[: sp_size * comm_bs * num_head * v_head_dim].view(
                    sp_size * comm_bs, num_head, v_head_dim
                )
                if _uses_nccl_comm_bs(sp_context):
                    res_local_buffer_3d.zero_()

                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, v_head_dim),
                    res_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 2. 拷贝 gathered_lse 到 lse_local_buffer
                lse_local_buffer_3d = lse_buffer.local_buffer.view(
                    gathered_lse.dtype
                )[: sp_size * comm_bs * num_head * 1].view(
                    sp_size * comm_bs, num_head, 1
                )
                if _uses_nccl_comm_bs(sp_context):
                    lse_local_buffer_3d.zero_()

                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_local_buffer_3d,
                    context.res_slice_get_to_buffer_output,
                    res_slice_fill_to_buffer_output,
                    context.res_to_buffer_output_mask,
                )

                # 3. 分配 All-to-All Input Buffer
                res_all_to_all_input_buffer = torch.empty(
                    (sp_size * comm_bs, num_head, v_head_dim),
                    dtype=gathered_o.dtype,
                    device=gathered_o.device,
                )
                lse_all_to_all_input_buffer = torch.empty(
                    (sp_size * comm_bs, num_head, 1),
                    dtype=gathered_lse.dtype,
                    device=gathered_lse.device,
                )

                # 4. 拷贝 gathered_o 到 res_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_o.view(-1, num_head, v_head_dim),
                    res_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                # 5. 拷贝 gathered_lse 到 lse_all_to_all_input_buffer
                copy_batch_indexed_triton(
                    gathered_lse.view(-1, num_head, 1),
                    lse_all_to_all_input_buffer,
                    context.res_slice_get_to_buffer_input,
                    res_slice_fill_to_buffer_input,
                    context.res_to_buffer_input_mask,
                )

                all_ranks_res_output_combine = res_buffer.all_to_all_ll(
                    res_all_to_all_input_buffer.view(sp_size * comm_bs, -1),
                    mask=res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, comm_bs, num_head, v_head_dim)
                all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
                    lse_all_to_all_input_buffer.view(sp_size * comm_bs, -1),
                    mask=res_lse_mask,
                    is_transpose=True,
                ).view(sp_size, comm_bs, num_head, 1)

                o = inter_rank_gqa_fwd_batch_decode_combine_kv(
                    all_ranks_res_output_combine,
                    all_ranks_lse_output_combine,
                    global_context_lens,
                    num_head,
                    v_head_dim,
                    comm_bs,
                    sp_size,
                ).view([comm_bs, num_head, v_head_dim])[:bs]

        return o


class Attention(nn.Module):

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

    def forward_mla_prefill_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> torch.Tensor:
        return self.impl.forward_prefill_native(q, k, v, cache_k, self.k_cache)

"""
性能测试：SP Attention with CUDA Graph

这个测试用例用于测试Sequence Parallel (SP) Attention在使用CUDA Graph时的性能。
支持两种Attention类型：
1. GQA (Grouped Query Attention) - 使用FlashAttention
2. MLA (Multi-head Latent Attention) - 使用FlashMLA

注意：测试用例假设整个系统中只有1个请求（batch_size=1），默认Master Rank=0。

测试包括：
1. Q的all-to-all通信
2. Attention计算（FlashAttention或FlashMLA）
3. 结果的all-to-all通信和combine
4. CUDA Graph的capture和replay

使用方法：
    # 测试GQA
    python tests/test_sp_attention_cudagraph.py --attention_type GQA --num_kv_heads 8
    
    # 测试MLA（要求num_kv_heads=1, head_dim=576, v_head_dim=512）
    torchrun --nproc_per_node=2 tests/test_sp_attention_cudagraph.py \
        --attention_type MLA \
        --num_kv_heads 1 \
        --head_dim 576 \
        --v_head_dim 512 \
        --attention_sp 2
"""

import os
import time
import csv
from datetime import datetime
from typing import Optional, Literal

import torch
import torch.distributed as dist
from einops import rearrange
from flash_attn_interface import flash_attn_with_kvcache
from triton.testing import do_bench, do_bench_cudagraph

try:
    import flash_mla
except ImportError:
    flash_mla = None

from nanodeploy.kernels.attention import inter_rank_gqa_fwd_batch_decode_combine_kv
from nanodeploy.kernels.copy import copy_batch_indexed_triton
from nanodeploy.worker.distributed import get_dist_context
from nanodeploy.worker.sp_context import get_sp_context, set_sp_context


def setup_distributed_context(
    attention_sp: int = 1,
    attention_tp: int = 1,
    attention_dp: int = 1,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """设置分布式上下文
    
    如果 attention_sp > 1 但没有分布式环境，会自动初始化一个本地分布式环境。
    对于测试用例，这允许在单机上测试 SP attention。
    """
    from nanodeploy.worker.distributed import set_dist_context, get_dist_context
    
    # 计算需要的总world_size
    total_world_size = attention_dp * attention_sp * attention_tp
    
    # 设置CUDA设备 - 必须在初始化分布式之前设置
    # torchrun会自动设置LOCAL_RANK环境变量
    if not dist.is_initialized():
        # 先尝试从环境变量获取LOCAL_RANK（torchrun会自动设置）
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                print(f"Setting CUDA device to {local_rank} (from LOCAL_RANK)")
        elif "RANK" in os.environ:
            # 如果有RANK但没有LOCAL_RANK，从RANK推断
            rank_from_env = int(os.environ["RANK"])
            if torch.cuda.is_available():
                device_id = rank_from_env % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                print(f"Setting CUDA device to {device_id} (inferred from RANK={rank_from_env})")
        elif rank is not None and torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            print(f"Setting CUDA device to {device_id} (inferred from rank={rank})")
    
    if not dist.is_initialized():
        # 尝试初始化分布式环境
        if rank is not None and world_size is not None:
            # 用户提供了rank和world_size
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # 从环境变量读取
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(backend="nccl")
        elif total_world_size > 1:
            # 需要多进程但环境变量未设置
            # 注意：真正的分布式需要多个进程，单进程无法模拟
            # 我们需要至少设置一个默认的dist_context以便测试能够继续
            print(f"Warning: attention_sp={attention_sp} requires distributed environment.")
            print(f"Currently running in single-process mode.")
            print(f"\nFor true multi-rank SP testing, please use one of:")
            print(f"  1. Use torchrun (recommended):")
            print(f"     torchrun --nproc_per_node={total_world_size} tests/test_sp_attention_cudagraph.py ...")
            print(f"  2. Set environment variables manually:")
            print(f"     export MASTER_ADDR=localhost")
            print(f"     export MASTER_PORT=29500")
            print(f"     export RANK=0  # or 1, 2, ... for each process")
            print(f"     export WORLD_SIZE={total_world_size}")
            print(f"\nFalling back to single rank mode (attention_sp=1) for testing...")
            rank = 0
            world_size = 1
            attention_sp = 1  # 强制设置为1
        else:
            # 单进程模式
            rank = 0
            world_size = 1

    # 确定rank和world_size
    if dist.is_initialized():
        if rank is None:
            rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
        
        # 再次确认CUDA设备设置（分布式初始化后）
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            if torch.cuda.current_device() != local_rank:
                torch.cuda.set_device(local_rank)
                print(f"[Rank {rank}] Corrected CUDA device to {local_rank}")
        elif torch.cuda.is_available() and world_size > 1:
            # 如果没有LOCAL_RANK，根据rank分配设备
            device_id = rank % torch.cuda.device_count()
            if torch.cuda.current_device() != device_id:
                torch.cuda.set_device(device_id)
                print(f"[Rank {rank}] Setting CUDA device to {device_id}")
        
        # 打印当前设备信息
        if torch.cuda.is_available():
            print(f"[Rank {rank}] Current CUDA device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name()}")
    else:
        rank = 0
        world_size = 1
        if attention_sp > 1:
            print(f"Warning: attention_sp={attention_sp} but no distributed environment. Setting attention_sp=1")
            attention_sp = 1

    # 检查设备数量是否足够
    required_devices = attention_dp * attention_sp * attention_tp
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size < required_devices:
        print(f"Warning: Required {required_devices} devices but only have {world_size}. Adjusting parallelism...")
        # 按比例调整，但不能超过world_size
        if attention_sp > 1 and world_size == 1:
            attention_sp = 1
        if attention_dp > 1 and attention_dp * attention_sp * attention_tp > world_size:
            attention_dp = max(1, world_size // (attention_sp * attention_tp))
        if attention_tp > 1 and attention_dp * attention_sp * attention_tp > world_size:
            attention_tp = max(1, world_size // (attention_dp * attention_sp))
        print(f"Adjusted: attention_dp={attention_dp}, attention_sp={attention_sp}, attention_tp={attention_tp}")

    # 使用set_dist_context，它会自动创建device mesh
    # 如果创建device mesh失败，会捕获异常并给出更清晰的错误信息
    try:
        set_dist_context(
            rank=rank,
            world_size=world_size,
            attention_dp=attention_dp,
            attention_sp=attention_sp,
            attention_tp=attention_tp,
        )
        dist_context = get_dist_context()
        return dist_context
    except Exception as e:
        error_msg = str(e)
        if "device" in error_msg.lower() or "mesh" in error_msg.lower() or "cuda" in error_msg.lower() or "nccl" in error_msg.lower():
            print(f"Error: Failed to create device mesh: {e}")
            print(f"This may be because:")
            print(f"  - Not enough CUDA devices (need {required_devices}, have {num_gpus} GPUs, {world_size} processes)")
            print(f"  - Multiple ranks assigned to the same GPU (duplicate GPU detected)")
            print(f"  - CUDA not available")
            print(f"  - NCCL initialization failed")
            print(f"\nSolution:")
            print(f"  1. Make sure you have at least {attention_sp} GPUs available")
            print(f"  2. Use torchrun which automatically handles device assignment:")
            print(f"     torchrun --nproc_per_node={attention_sp} tests/test_sp_attention_cudagraph.py ...")
            print(f"  3. Or manually set CUDA_VISIBLE_DEVICES to specify which GPUs to use:")
            print(f"     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node={attention_sp} tests/test_sp_attention_cudagraph.py ...")
            raise RuntimeError(
                f"Cannot create device mesh: need {required_devices} devices but have {num_gpus} GPUs and {world_size} processes. "
                f"For attention_sp={attention_sp}, please use torchrun with --nproc_per_node={attention_sp}. "
                f"If you have multiple GPUs, make sure each rank uses a different GPU."
            ) from e
        else:
            raise


def setup_sp_context(
    max_num_seqs: int,
    head_size: int,
    num_attention_heads: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """设置SP上下文"""
    dist_context = get_dist_context()
    rank = dist_context.attn_sp_rank
    sp_size = dist_context.attn_sp_world_size

    set_sp_context(
        max_num_seqs=max_num_seqs,
        head_size=head_size,
        num_attention_heads=num_attention_heads,
        dtype=dtype,
        rank=rank,
        sp_size=sp_size,
    )


def sp_attention_forward_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    global_context_lens: torch.Tensor,
    q_mask_adjusted: torch.Tensor,
    res_lse_mask_adjusted: torch.Tensor,
    q_slice_get: torch.Tensor,
    q_slice_fill: torch.Tensor,
    q_copy_mask: torch.Tensor,
    res_slice_get_to_buffer_output: torch.Tensor,
    res_slice_fill_to_buffer_output: torch.Tensor,
    res_to_buffer_output_mask: torch.Tensor,
    res_slice_get_to_buffer_input: torch.Tensor,
    res_slice_fill_to_buffer_input: torch.Tensor,
    res_to_buffer_input_mask: torch.Tensor,
    q_offsets: torch.Tensor,
    attention_compute_bs: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    sp_size: int,
) -> torch.Tensor:
    """
    SP Attention Forward Pass (GQA with FlashAttention)
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor
        v_cache: Value cache tensor
        context_lens: Context lengths for attention computation
        block_tables: Block tables for paged attention
        global_context_lens: Global context lengths [sp_size, batch_size]
        q_mask_adjusted: Adjusted Q mask for all-to-all [sp_size, batch_size] (precomputed)
        res_lse_mask_adjusted: Adjusted Result/LSE mask for all-to-all [sp_size, batch_size] (precomputed)
        q_slice_get: Q slice get indices
        q_slice_fill: Q slice fill indices
        q_copy_mask: Q copy mask
        res_slice_get_to_buffer_output: Result slice get indices for output buffer
        res_slice_fill_to_buffer_output: Result slice fill indices for output buffer
        res_to_buffer_output_mask: Result copy mask for output buffer
        res_slice_get_to_buffer_input: Result slice get indices for input buffer
        res_slice_fill_to_buffer_input: Result slice fill indices for input buffer
        res_to_buffer_input_mask: Result copy mask for input buffer
        q_offsets: Q offsets for all-to-all
        attention_compute_bs: Attention computation batch size
        num_heads: Number of attention heads
        head_dim: Head dimension
        scale: Attention scale
        sp_size: SP world size
        
    Returns:
        Output tensor [batch_size, num_heads, head_dim]
    """
    bs, num_head, head_dim = q.shape
    max_num_seqs = get_sp_context().max_num_seqs
    q_buffer = get_sp_context().q_buffer
    res_buffer = get_sp_context().res_buffer
    lse_buffer = get_sp_context().lse_buffer

    # Q copy to local buffer
    local_q_buffer_3d = q_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * head_dim
    ].view(sp_size * max_num_seqs, num_head, head_dim)
    copy_batch_indexed_triton(
        q,
        local_q_buffer_3d,
        q_slice_get,
        q_slice_fill,
        q_copy_mask,
    )

    # Q all-to-all
    # 对于只有1个请求，Master Rank=0的情况：
    #   - Rank 0: q_mask_adjusted[0]=0（不发送给自己），q_mask_adjusted[1]=1（发送给Rank 1）
    #   - Rank 1: q_mask_adjusted全0（没有Q，不需要发送）
    q = q_buffer.all_to_all_ll(
        q.view([bs, -1]),
        mask=q_mask_adjusted,
        offsets=q_offsets,
    ).view([sp_size * max_num_seqs, num_head, head_dim])

    q = q[:attention_compute_bs]
    context_lens_for_attn = context_lens[:attention_compute_bs]
    block_tables_for_attn = block_tables[:attention_compute_bs]

    # Attention computation
    o, lse = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens_for_attn,
        page_table=block_tables_for_attn,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
    )[:2]

    # Convert LSE to bfloat16
    lse = lse.to(torch.bfloat16)
    gathered_o = o.view([attention_compute_bs, num_head, head_dim])
    gathered_lse = lse.view([attention_compute_bs, num_head, 1])

    # Copy gathered_o to res_local_buffer
    res_local_buffer_3d = res_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * head_dim
    ].view(sp_size * max_num_seqs, num_head, head_dim)
    copy_batch_indexed_triton(
        gathered_o.view(-1, num_head, head_dim),
        res_local_buffer_3d,
        res_slice_get_to_buffer_output,
        res_slice_fill_to_buffer_output,
        res_to_buffer_output_mask,
    )

    # Copy gathered_lse to lse_local_buffer
    lse_local_buffer_3d = lse_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * 1
    ].view(sp_size * max_num_seqs, num_head, 1)
    copy_batch_indexed_triton(
        gathered_lse.view(-1, num_head, 1),
        lse_local_buffer_3d,
        res_slice_get_to_buffer_output,
        res_slice_fill_to_buffer_output,
        res_to_buffer_output_mask,
    )

    # Allocate All-to-All Input Buffers
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

    # Copy gathered_o to res_all_to_all_input_buffer
    copy_batch_indexed_triton(
        gathered_o.view(-1, num_head, head_dim),
        res_all_to_all_input_buffer,
        res_slice_get_to_buffer_input,
        res_slice_fill_to_buffer_input,
        res_to_buffer_input_mask,
    )

    # Copy gathered_lse to lse_all_to_all_input_buffer
    copy_batch_indexed_triton(
        gathered_lse.view(-1, num_head, 1),
        lse_all_to_all_input_buffer,
        res_slice_get_to_buffer_input,
        res_slice_fill_to_buffer_input,
        res_to_buffer_input_mask,
    )

    # All-to-all for results
    all_ranks_res_output_combine = res_buffer.all_to_all_ll(
        res_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask_adjusted,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, head_dim)
    all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
        lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask_adjusted,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, 1)

    # Combine results from all ranks
    o = inter_rank_gqa_fwd_batch_decode_combine_kv(
        all_ranks_res_output_combine,
        all_ranks_lse_output_combine,
        global_context_lens,
        num_head,
        head_dim,
        max_num_seqs,
        sp_size,
    )
    
    # Extract output for the actual batch size
    o = o[:bs]
    return o


def sp_attention_forward_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    global_context_lens: torch.Tensor,
    q_mask_adjusted: torch.Tensor,
    res_lse_mask_adjusted: torch.Tensor,
    q_slice_get: torch.Tensor,
    q_slice_fill: torch.Tensor,
    q_copy_mask: torch.Tensor,
    res_slice_get_to_buffer_output: torch.Tensor,
    res_slice_fill_to_buffer_output: torch.Tensor,
    res_to_buffer_output_mask: torch.Tensor,
    res_slice_get_to_buffer_input: torch.Tensor,
    res_slice_fill_to_buffer_input: torch.Tensor,
    res_to_buffer_input_mask: torch.Tensor,
    attention_compute_bs: int,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    scale: float,
    sp_size: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    debug: bool = False,
) -> torch.Tensor:
    """
    SP Attention Forward Pass (MLA with FlashMLA)
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor (MLA only needs k_cache, not v_cache)
        context_lens: Context lengths for attention computation
        tile_scheduler_metadata: Precomputed MLA metadata (computed once during initialization)
        num_splits: Precomputed MLA num_splits (computed once during initialization)
        block_tables: Block tables for paged attention
        global_context_lens: Global context lengths [sp_size, batch_size]
        q_mask_adjusted: Adjusted Q mask for all-to-all [sp_size, batch_size] (precomputed)
        res_lse_mask_adjusted: Adjusted Result/LSE mask for all-to-all [sp_size, batch_size] (precomputed)
        q_slice_get: Q slice get indices
        q_slice_fill: Q slice fill indices
        q_copy_mask: Q copy mask
        res_slice_get_to_buffer_output: Result slice get indices for output buffer
        res_slice_fill_to_buffer_output: Result slice fill indices for output buffer
        res_to_buffer_output_mask: Result copy mask for output buffer
        res_slice_get_to_buffer_input: Result slice get indices for input buffer
        res_slice_fill_to_buffer_input: Result slice fill indices for input buffer
        res_to_buffer_input_mask: Result copy mask for input buffer
        attention_compute_bs: Attention computation batch size
        num_heads: Number of attention heads
        head_dim: Head dimension (for Q and K)
        v_head_dim: V head dimension (for output)
        scale: Attention scale
        sp_size: SP world size
        tile_scheduler_metadata: Precomputed MLA metadata (computed once during initialization)
        num_splits: Precomputed MLA num_splits (computed once during initialization)
        debug: Enable debug logging
        
    Returns:
        Output tensor [batch_size, num_heads, v_head_dim]
    """
    if flash_mla is None:
        raise ImportError("flash_mla is required for MLA attention. Please install it.")
    
    bs, num_head, head_dim_q = q.shape
    max_num_seqs = get_sp_context().max_num_seqs
    q_buffer = get_sp_context().q_buffer
    res_buffer = get_sp_context().res_buffer
    lse_buffer = get_sp_context().lse_buffer

    # Q copy to local buffer
    local_q_buffer_3d = q_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * head_dim_q
    ].view(sp_size * max_num_seqs, num_head, head_dim_q)
    copy_batch_indexed_triton(
        q,
        local_q_buffer_3d,
        q_slice_get,
        q_slice_fill,
        q_copy_mask,
    )

    # Q all-to-all (MLA doesn't use offsets)
    # 使用调整后的mask
    q = q_buffer.all_to_all_ll(
        q.view([bs, -1]),
        mask=q_mask_adjusted,
    ).view([sp_size * max_num_seqs, num_head, head_dim_q])

    q = q[:attention_compute_bs]
    context_lens_for_attn = context_lens[:attention_compute_bs]
    block_tables_for_attn = block_tables[:attention_compute_bs]

    # tile_scheduler_metadata and num_splits are precomputed during initialization
    # and passed as parameters to avoid recalculating on each forward pass

    # Attention computation with FlashMLA
    # Note: For MLA:
    # - head_dim = 576 = kv_lora_rank (512) + qk_rope_head_dim (64) - used for Q and K cache
    # - v_head_dim = 512 = kv_lora_rank (512) - used for output V
    # The k_cache should have head_dim (576) for K dimension
    assert v_head_dim == 512, f"v_head_dim must be 512 (kv_lora_rank) for MLA, but got {v_head_dim}"
    assert head_dim == 576, f"head_dim must be 576 (kv_lora_rank + qk_rope_head_dim) for MLA, but got {head_dim}"
    
    # Ensure v_head_dim is int (not float or tensor)
    if not isinstance(v_head_dim, int):
        v_head_dim = int(v_head_dim)
        print(f"[TEST] Converted v_head_dim to int: {v_head_dim}")
    
    # Ensure q.unsqueeze(1) has the correct shape before calling FlashMLA
    q_for_attn = q.unsqueeze(1)
    
    # Debug logs - match format with FlashMLAImpl
    if debug:
        print(f"[TEST] Before flash_mla_with_kvcache:")
        print(f"  q.shape: {q.shape}, q_for_attn.shape: {q_for_attn.shape}")
        print(f"  k_cache.shape: {k_cache.shape}")
        print(f"  k_cache last dimension: {k_cache.shape[-1]}")
        print(f"  block_tables_for_attn.shape: {block_tables_for_attn.shape}")
        print(f"  context_lens_for_attn.shape: {context_lens_for_attn.shape}")
        print(f"  context_lens_for_attn: {context_lens_for_attn}")
        print(f"  v_head_dim parameter: {v_head_dim} (type: {type(v_head_dim)})")
        print(f"  head_dim parameter: {head_dim} (type: {type(head_dim)})")
        print(f"  num_heads: {num_heads}, num_kv_heads: 1")
        print(f"  scale: {scale}, causal: True")
        print(f"  tile_scheduler_metadata type: {type(tile_scheduler_metadata)}")
        print(f"  num_splits.shape: {num_splits.shape if hasattr(num_splits, 'shape') else type(num_splits)}")
    
    # For MLA: k_cache uses head_dim (576) for K, but v_head_dim (512) is passed to flash_mla_with_kvcache
    # This matches the working environment: k_cache.shape[-1]=576, but v_head_size=512
    assert k_cache.shape[-1] == head_dim, (
        f"k_cache last dimension ({k_cache.shape[-1]}) must equal head_dim ({head_dim}) for K storage"
    )
    
    o, lse = flash_mla.flash_mla_with_kvcache(
        q_for_attn,
        k_cache,  # k_cache shape: [batch * num_pages, page_size, num_kv_heads=1, v_head_dim=576]
        block_tables_for_attn,
        context_lens_for_attn,
        v_head_dim,  # v_head_size - must be 576 (int)
        tile_scheduler_metadata,
        num_splits,
        scale,
        causal=True,
    )
    
    if debug:
        print(f"[TEST] After flash_mla_with_kvcache:")
        print(f"  o.shape (before squeeze): {o.shape}")
        print(f"  lse.shape: {lse.shape}")

    o = o.squeeze(1)
    if debug:
        print(f"  o.shape (after squeeze): {o.shape}")
    # Verify output shape matches expected v_head_dim
    if o.shape[-1] != v_head_dim:
        print(f"[WARNING] Output shape mismatch: o.shape[-1]={o.shape[-1]}, expected v_head_dim={v_head_dim}")

    # Convert LSE to bfloat16
    lse = lse.to(torch.bfloat16)
    gathered_o = o.view([attention_compute_bs, num_head, v_head_dim])
    gathered_lse = lse.view([attention_compute_bs, num_head, 1])

    # Copy gathered_o to res_local_buffer
    res_local_buffer_3d = res_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * v_head_dim
    ].view(sp_size * max_num_seqs, num_head, v_head_dim)
    copy_batch_indexed_triton(
        gathered_o.view(-1, num_head, v_head_dim),
        res_local_buffer_3d,
        res_slice_get_to_buffer_output,
        res_slice_fill_to_buffer_output,
        res_to_buffer_output_mask,
    )

    # Copy gathered_lse to lse_local_buffer
    lse_local_buffer_3d = lse_buffer.local_buffer.view(get_sp_context().dtype)[
        : sp_size * max_num_seqs * num_head * 1
    ].view(sp_size * max_num_seqs, num_head, 1)
    copy_batch_indexed_triton(
        gathered_lse.view(-1, num_head, 1),
        lse_local_buffer_3d,
        res_slice_get_to_buffer_output,
        res_slice_fill_to_buffer_output,
        res_to_buffer_output_mask,
    )

    # Allocate All-to-All Input Buffers
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

    # Copy gathered_o to res_all_to_all_input_buffer
    copy_batch_indexed_triton(
        gathered_o.view(-1, num_head, v_head_dim),
        res_all_to_all_input_buffer,
        res_slice_get_to_buffer_input,
        res_slice_fill_to_buffer_input,
        res_to_buffer_input_mask,
    )

    # Copy gathered_lse to lse_all_to_all_input_buffer
    copy_batch_indexed_triton(
        gathered_lse.view(-1, num_head, 1),
        lse_all_to_all_input_buffer,
        res_slice_get_to_buffer_input,
        res_slice_fill_to_buffer_input,
        res_to_buffer_input_mask,
    )

    # All-to-all for results
    all_ranks_res_output_combine = res_buffer.all_to_all_ll(
        res_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask_adjusted,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, v_head_dim)
    all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
        lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask_adjusted,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, 1)

    # Combine results from all ranks
    o = inter_rank_gqa_fwd_batch_decode_combine_kv(
        all_ranks_res_output_combine,
        all_ranks_lse_output_combine,
        global_context_lens,
        num_head,
        v_head_dim,
        max_num_seqs,
        sp_size,
    )
    
    # Extract output for the actual batch size
    o = o[:bs]
    return o


def create_test_data(
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    seq_len: int,
    max_num_seqs: int,
    sp_size: int,
    attention_type: Literal["GQA", "MLA"] = "GQA",
    v_head_dim: Optional[int] = None,
    page_size: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """创建测试数据
    注意：整个系统中只有1个请求（batch_size=1），Master Rank=0
    """
    batch_size = 1  # 固定为1，整个系统只有一个请求
    
    # Q tensor
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)

    # K cache
    # For MLA: k_cache stores K, so it should use head_dim (576), not v_head_dim (512)
    # head_dim = 576 = kv_lora_rank (512) + qk_rope_head_dim (64) - used for Q and K
    # v_head_dim = 512 = kv_lora_rank (512) - used for output V only
    k_cache = torch.randn(
        batch_size * seq_len // page_size,
        page_size,
        num_kv_heads,
        head_dim,  # k_cache uses head_dim (576) for MLA, not v_head_dim (512)
        dtype=dtype,
        device=device,
    )
    
    # V cache (only for GQA, MLA doesn't need v_cache storage)
    if attention_type == "GQA":
        v_cache = torch.randn(
            batch_size * seq_len // page_size,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
    else:
        # MLA doesn't use v_cache, but we'll create a dummy tensor for compatibility
        v_cache = torch.tensor([], dtype=dtype, device=device)

    # Global context lengths [sp_size, batch_size]
    # 表示每个rank处理每个序列的token数量
    # 每个序列的KVCache被均分到所有SP ranks
    global_context_lens = torch.zeros(
        sp_size, max_num_seqs, dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        # 模拟将序列分配到不同的SP rank
        for sp_idx in range(sp_size):
            # 简单的分配策略：均匀分配
            if i < max_num_seqs:
                segment_len = seq_len // sp_size
                if sp_idx < seq_len % sp_size:
                    segment_len += 1
                global_context_lens[sp_idx, i] = segment_len

    # Block tables
    # 每个rank的block数量应该按照该rank处理的token数量来计算
    # 对于每个序列，计算每个rank需要的pages数量
    # 注意：所有ranks都创建相同的block_tables结构，但每个rank实际只使用自己负责的部分
    # 实际上，block_tables应该是每个rank按照自己处理的segment_len来创建
    # 但由于所有ranks都创建相同的test_data结构，我们使用segment_len（第一个rank的长度）来创建
    segment_len_per_rank = seq_len // sp_size  # 每个rank处理的token数量（均匀分配时）
    num_pages_per_rank = (segment_len_per_rank + page_size - 1) // page_size
    block_tables = torch.arange(
        batch_size * num_pages_per_rank, device=device, dtype=torch.int32
    ).view(batch_size, num_pages_per_rank)

    # Context lengths
    # context_lens应该设置为每个rank实际处理的token数量
    # 对于均匀分配：每个rank处理 seq_len // sp_size 个tokens
    segment_len = seq_len // sp_size
    context_lens = torch.full(
        (max_num_seqs,), segment_len, dtype=torch.int32, device=device
    )

    # Q mask: Q需要发送到所有有KVCache的ranks
    # 根据model_runner.py的实现逻辑:
    #   q_mask = global_context_lens.clone()
    #   q_mask[sp_rank].fill_(0)  # 不发送给自己
    #   q_mask[q_mask != 0] = 1   # 其他有KVCache的ranks设为1
    # 含义：q_mask[target_rank, seq_id] = 1 表示当前rank需要向target_rank发送seq_id的Q
    # 
    # 对于测试用例：只有1个请求（seq_id=0），Master Rank=0
    # - Q在Master Rank (Rank 0)，需要发送到所有有KVCache的ranks（Rank 0-7）
    # - 但在实际执行时，Rank 0会将自己的位置fill_(0)，所以只发送给Rank 1-7
    # - Rank 1-7没有Q，所以它们的q_mask应该全0（不会发送）
    # 注意：当前代码中所有ranks都使用相同的q_mask（基于global_context_lens），
    # 但只有Master Rank (Rank 0)会真正使用q_mask发送Q，其他ranks虽然有相同的q_mask值，
    # 但由于它们没有Q，实际上不会执行Q的发送操作
    q_mask = global_context_lens.clone()
    # 运行时，每个rank会将自己的rank位置fill_(0)（不发送给自己）

    # Res/LSE mask: 结果只发送回Master Rank
    # 根据model_runner.py的实现逻辑:
    #   res_lse_mask = context_lens.clone()
    #   res_lse_mask[sp_rank].fill_(0)  # 不发送给自己
    #   res_lse_mask[res_lse_mask != 0] = 1
    # 含义：res_lse_mask[target_rank, seq_id] = 1 表示当前rank需要向target_rank发送seq_id的结果
    # 
    # 对于测试用例：只有1个请求（seq_id=0），Master Rank=0
    # - 所有ranks的结果都发送回Master Rank (Rank 0)
    # - Rank 0: res_lse_mask[0, 0]会被fill_(0)，但res_lse_mask其他位置都是0（不需要发送给其他ranks）
    # - Rank 1: res_lse_mask[0, 0] = context_len（非0），表示需要发送给Rank 0
    # 使用context_lens作为基础，因为所有ranks都有完整的context_len
    res_lse_mask = context_lens[:max_num_seqs].unsqueeze(0).expand(sp_size, -1).clone()
    # 运行时，每个rank会将自己的rank位置fill_(0)（不发送给自己）
    # 但我们需要确保只有Master Rank (Rank 0)接收结果
    # 实际上，应该设置为：只有Rank 0的位置为非0，其他都为0
    res_lse_mask.zero_()
    for i in range(batch_size):
        if i < max_num_seqs:
            # 只有Master Rank (Rank 0)接收结果
            # 设置为非0值（context_len），表示所有ranks都需要向Rank 0发送
            res_lse_mask[0, i] = context_lens[i]

    q_slice_get = torch.arange(batch_size, dtype=torch.int32, device=device)
    q_slice_fill = torch.arange(batch_size, dtype=torch.int32, device=device)
    q_copy_mask = torch.ones(batch_size, dtype=torch.int32, device=device)

    res_slice_get_to_buffer_output = torch.arange(
        batch_size, dtype=torch.int32, device=device
    )
    res_slice_fill_to_buffer_output = torch.arange(
        batch_size, dtype=torch.int32, device=device
    )
    res_to_buffer_output_mask = torch.ones(
        batch_size, dtype=torch.int32, device=device
    )

    max_num_recv_seqs = max_num_seqs - batch_size
    res_slice_get_to_buffer_input = torch.full(
        (max_num_recv_seqs,), -1, dtype=torch.int32, device=device
    )
    res_slice_fill_to_buffer_input = torch.full(
        (max_num_recv_seqs,), -1, dtype=torch.int32, device=device
    )
    res_to_buffer_input_mask = torch.zeros(
        max_num_recv_seqs, dtype=torch.int32, device=device
    )

    # Q offsets
    q_offsets = torch.zeros(sp_size + 1, dtype=torch.int32, device=device)
    # 简单的均匀分配
    for i in range(1, sp_size + 1):
        q_offsets[i] = (batch_size * i + sp_size - 1) // sp_size

    result = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "context_lens": context_lens,
        "block_tables": block_tables,
        "global_context_lens": global_context_lens,
        "q_mask": q_mask,
        "res_lse_mask": res_lse_mask,
        "q_slice_get": q_slice_get,
        "q_slice_fill": q_slice_fill,
        "q_copy_mask": q_copy_mask,
        "res_slice_get_to_buffer_output": res_slice_get_to_buffer_output,
        "res_slice_fill_to_buffer_output": res_slice_fill_to_buffer_output,
        "res_to_buffer_output_mask": res_to_buffer_output_mask,
        "res_slice_get_to_buffer_input": res_slice_get_to_buffer_input,
        "res_slice_fill_to_buffer_input": res_slice_fill_to_buffer_input,
        "res_to_buffer_input_mask": res_to_buffer_input_mask,
    }
    
    # Q offsets (only for GQA, MLA doesn't use it)
    if attention_type == "GQA":
        result["q_offsets"] = q_offsets
    
    return result


def benchmark_sp_attention_with_cudagraph(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    attention_sp: int = 2,
    attention_type: Literal["GQA", "MLA"] = "GQA",
    v_head_dim: Optional[int] = None,
    use_cudagraph: bool = True,
    num_warmup: int = 100,
    num_iterations: int = 200,
    debug: bool = False,
    enable_profiler: bool = False,
    profiler_iterations: int = 50,
):
    """
    使用CUDA Graph对SP Attention进行性能测试
    
    注意：整个系统中只有1个请求（batch_size=1），Master Rank=0
    
    Args:
        seq_len: Sequence length
        num_heads: Number of query heads
        head_dim: Head dimension (for Q and K)
        num_kv_heads: Number of key/value heads (must be 1 for MLA)
        attention_sp: SP parallelism size
        attention_type: Attention type ("GQA" or "MLA")
        v_head_dim: V head dimension (for MLA output, if None, uses head_dim)
        use_cudagraph: Whether to use CUDA Graph
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        debug: Enable debug logging for tensor shapes
        enable_profiler: Whether to run profiler after benchmark
        profiler_iterations: Number of iterations to profile (default: 50)
    """
    batch_size = 1  # 固定为1，整个系统只有一个请求
    
    # Validate attention type
    if attention_type == "MLA":
        if num_kv_heads != 1:
            raise ValueError("MLA requires num_kv_heads == 1")
        if flash_mla is None:
            raise ImportError("flash_mla is required for MLA attention. Please install it.")
        # For MLA: head_dim = kv_lora_rank (512) + qk_rope_head_dim (64) = 576
        # This is used for Q and K cache
        if head_dim != 576:
            raise ValueError(
                f"MLA requires head_dim (kv_lora_rank + qk_rope_head_dim) == 576, but got {head_dim}. "
                f"Please set --head_dim 576 when using --attention_type MLA"
            )
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1  # 固定为1，整个系统只有一个请求
    max_num_seqs = 2  # 留一些空间，但实际上只有1个请求
    
    # MLA mode requires block_size=64, GQA can use page_size=16
    if attention_type == "MLA":
        page_size = 64  # MLA mode only supports block_size=64
        # Also need to ensure seq_len is divisible by page_size
        if seq_len % page_size != 0:
            # Round up to next multiple of page_size
            seq_len = ((seq_len + page_size - 1) // page_size) * page_size
            print(f"Warning: seq_len adjusted to {seq_len} to be divisible by page_size={page_size}")
    else:
        page_size = 16  # GQA can use smaller page size
    
    scale = 1.0 / (head_dim**0.5)
    
    # For MLA, v_head_dim is kv_lora_rank = 512, not 576!
    # head_dim = 576 = kv_lora_rank (512) + qk_rope_head_dim (64) - used for Q and K
    # v_head_dim = 512 = kv_lora_rank (512) - used for output V
    if v_head_dim is None:
        if attention_type == "MLA":
            # Default for MLA: v_head_dim = kv_lora_rank = 512
            v_head_dim = 512
        else:
            v_head_dim = head_dim
    
    # Validate v_head_dim for MLA
    if attention_type == "MLA":
        # FlashMLA: v_head_dim should be 512 (kv_lora_rank), not 576!
        # The error message "Only head_size_v == 576 is supported" is misleading
        # Actually v_head_size should be 512, as seen in the working environment
        if v_head_dim != 512:
            raise ValueError(
                f"MLA requires v_head_dim (kv_lora_rank) == 512, but got {v_head_dim}. "
                f"Note: head_dim=576 is for Q/K (512+64), but v_head_dim=512 is for output V. "
                f"Please set --v_head_dim 512 when using --attention_type MLA"
            )
        # Print validation info for debugging
        print(f"Validation passed: head_dim={head_dim} (for Q/K), v_head_dim={v_head_dim} (for output V)")

    print(f"\n{'='*80}")
    print(f"SP Attention Performance Test with CUDA Graph")
    print(f"{'='*80}")
    print(f"Attention Type: {attention_type}")
    print(f"Batch Size: {batch_size} (fixed, single request in entire system)")
    print(f"Sequence Length: {seq_len}")
    print(f"Num Heads: {num_heads}, Num KV Heads: {num_kv_heads}")
    print(f"Head Dim: {head_dim}")
    if attention_type == "MLA":
        print(f"V Head Dim: {v_head_dim}")
    print(f"SP Size: {attention_sp}")
    print(f"Master Rank: 0 (fixed)")
    print(f"Use CUDA Graph: {use_cudagraph}")
    print(f"{'='*80}\n")

    # 设置分布式上下文
    try:
        dist_context = setup_distributed_context(
            attention_sp=attention_sp,
            attention_tp=1,
            attention_dp=1,
        )
        # 获取实际的attention_sp（可能在单rank模式下被调整为1）
        actual_attention_sp = dist_context.attention_sp
        if actual_attention_sp != attention_sp:
            print(f"Note: attention_sp adjusted from {attention_sp} to {actual_attention_sp}")
            attention_sp = actual_attention_sp
    except Exception as e:
        print(f"Error: Failed to setup distributed context: {e}")
        print("\nPossible solutions:")
        print("  1. For single rank testing, use attention_sp=1:")
        print("     python tests/test_sp_attention_cudagraph.py --attention_sp 1 ...")
        print("  2. For multi-rank testing, use torchrun:")
        print(f"     torchrun --nproc_per_node={attention_sp} tests/test_sp_attention_cudagraph.py ...")
        return

    # 设置SP上下文
    try:
        setup_sp_context(
            max_num_seqs=max_num_seqs,
            head_size=head_dim,
            num_attention_heads=num_heads,
            dtype=dtype,
        )
    except Exception as e:
        print(f"Error: Failed to setup SP context: {e}")
        print("Please ensure distributed context is properly initialized.")
        # Check if this is because rank is not in SP group
        dist_context = get_dist_context()
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        attn_sp_rank = getattr(dist_context, 'attn_sp_rank', -1)
        if attn_sp_rank < 0:
            print(f"[Rank {current_rank}] This rank is not part of SP group. Skipping benchmark but will wait at barrier.")
            # Skip all benchmark operations but continue to barrier
            # Set a flag to skip rest of execution
            import sys
            # We'll add a barrier at the end to ensure all ranks synchronize
            pass
        else:
            # Real error, should return
            return

    # 创建测试数据（batch_size固定为1）
    # Only create test data if this rank is in SP group
    test_data = create_test_data(
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        max_num_seqs=max_num_seqs,
        sp_size=attention_sp,
        attention_type=attention_type,
        v_head_dim=v_head_dim,
        page_size=page_size,
        device=device,
        dtype=dtype,
    )

    # Precompute MLA metadata if needed (only once during initialization)
    tile_scheduler_metadata = None
    num_splits = None
    if attention_type == "MLA":
        # For MLA: num_query_heads_per_kv = num_heads // num_kv_heads = num_heads // 1 = num_heads
        num_query_heads_per_kv = num_heads // 1  # For MLA, num_kv_heads is always 1
        context_lens_for_mla_metadata = test_data["context_lens"][:batch_size]  # Only need first batch_size entries
        if debug:
            print(f"[TEST] Precomputing get_mla_metadata: context_lens_for_mla_metadata.shape={context_lens_for_mla_metadata.shape}, "
                  f"context_lens_for_mla_metadata={context_lens_for_mla_metadata}, "
                  f"num_query_heads_per_kv={num_query_heads_per_kv}, num_kv_heads=1")
        tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
            context_lens_for_mla_metadata,
            num_query_heads_per_kv,
            1,  # num_kv_heads (MLA requires num_kv_heads == 1)
        )
    
    # Precompute adjusted masks (only compute once, not during forward)
    sp_rank = get_dist_context().attn_sp_rank if attention_sp > 1 else 0
    q_mask_adjusted = test_data["q_mask"].clone()
    q_mask_adjusted[sp_rank].fill_(0)  # 不发送给自己
    q_mask_adjusted[q_mask_adjusted != 0] = 1  # 非0值设为1
    
    res_lse_mask_adjusted = test_data["res_lse_mask"].clone()
    res_lse_mask_adjusted[sp_rank].fill_(0)  # 不发送给自己
    res_lse_mask_adjusted[res_lse_mask_adjusted != 0] = 1  # 非0值设为1
    
    # Debug print before creating forward_fn (to avoid printing during CUDA Graph capture)
    if debug:
        print("[TEST] Debug info (before CUDA Graph capture):")
        print(f"  context_lens: {test_data['context_lens']}")
        print(f"  block_tables.shape: {test_data['block_tables'].shape}")
        print(f"  global_context_lens: {test_data['global_context_lens']}")
        if attention_type == "MLA":
            print(f"  tile_scheduler_metadata type: {type(tile_scheduler_metadata)}")
            if hasattr(num_splits, 'shape'):
                print(f"  num_splits.shape: {num_splits.shape}")
    
    # Create a sync tensor for all_reduce synchronization at the start of each forward
    # This ensures all ranks are aligned before all_to_all operations
    sync_tensor = None
    if attention_sp > 1 and dist.is_initialized():
        sync_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    
    # 创建forward函数（注意：在CUDA Graph capture期间，debug打印会被禁用以避免CUDA错误）
    if attention_type == "GQA":
        def forward_fn():
            # Synchronize all ranks at the start of each forward using all_reduce
            # This ensures all_to_all communication is aligned and avoids long waits
            if sync_tensor is not None:
                dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            
            return sp_attention_forward_gqa(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                v_cache=test_data["v_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                global_context_lens=test_data["global_context_lens"],
                q_mask_adjusted=q_mask_adjusted,
                res_lse_mask_adjusted=res_lse_mask_adjusted,
                q_slice_get=test_data["q_slice_get"],
                q_slice_fill=test_data["q_slice_fill"],
                q_copy_mask=test_data["q_copy_mask"],
                res_slice_get_to_buffer_output=test_data["res_slice_get_to_buffer_output"],
                res_slice_fill_to_buffer_output=test_data["res_slice_fill_to_buffer_output"],
                res_to_buffer_output_mask=test_data["res_to_buffer_output_mask"],
                res_slice_get_to_buffer_input=test_data["res_slice_get_to_buffer_input"],
                res_slice_fill_to_buffer_input=test_data["res_slice_fill_to_buffer_input"],
                res_to_buffer_input_mask=test_data["res_to_buffer_input_mask"],
                q_offsets=test_data["q_offsets"],
                attention_compute_bs=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                sp_size=attention_sp,
            )
    else:  # MLA
        def forward_fn():
            # Synchronize all ranks at the start of each forward using all_reduce
            # This ensures all_to_all communication is aligned and avoids long waits
            if sync_tensor is not None:
                dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            
            return sp_attention_forward_mla(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                global_context_lens=test_data["global_context_lens"],
                q_mask_adjusted=q_mask_adjusted,
                res_lse_mask_adjusted=res_lse_mask_adjusted,
                q_slice_get=test_data["q_slice_get"],
                q_slice_fill=test_data["q_slice_fill"],
                q_copy_mask=test_data["q_copy_mask"],
                res_slice_get_to_buffer_output=test_data["res_slice_get_to_buffer_output"],
                res_slice_fill_to_buffer_output=test_data["res_slice_fill_to_buffer_output"],
                res_to_buffer_output_mask=test_data["res_to_buffer_output_mask"],
                res_slice_get_to_buffer_input=test_data["res_slice_get_to_buffer_input"],
                res_slice_fill_to_buffer_input=test_data["res_slice_fill_to_buffer_input"],
                res_to_buffer_input_mask=test_data["res_to_buffer_input_mask"],
                attention_compute_bs=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                scale=scale,
                sp_size=attention_sp,
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
                debug=False,  # 在forward_fn中禁用debug，避免CUDA Graph capture期间的tensor打印
            )

    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = forward_fn()
    torch.cuda.synchronize()

    # Initialize graph variable for potential profiler use
    graph = None
    
    # Benchmark with or without CUDA Graph
    if use_cudagraph:
        print(f"Benchmarking with CUDA Graph ({num_iterations} iterations)...")
        torch.cuda.synchronize()
        
        # Capture CUDA Graph
        graph = torch.cuda.CUDAGraph()
        graph_pool = None
        
        # Warmup for graph capture
        for _ in range(10):
            _ = forward_fn()
        torch.cuda.synchronize()
        
        # Capture graph
        with torch.cuda.graph(graph, graph_pool):
            output = forward_fn()
        
        if graph_pool is None:
            graph_pool = graph.pool()
        
        # Benchmark with graph replay
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations
        avg_time_us = avg_time_ms * 1000
        
        print(f"Average time with CUDA Graph: {avg_time_us:.2f} us")
    else:
        print(f"Benchmarking without CUDA Graph ({num_iterations} iterations)...")
        elapsed_time = do_bench(forward_fn, warmup=10, rep=num_iterations)
        avg_time_us = elapsed_time * 1e6
        print(f"Average time without CUDA Graph: {avg_time_us:.2f} us")

    # 计算吞吐量和理论性能
    total_seqlen = seq_len * batch_size
    if attention_type == "GQA":
        mem_io = (
            total_seqlen * num_kv_heads * head_dim * 2  # K, V cache
            + batch_size * num_heads * head_dim * 2  # Q
            + batch_size * num_heads * head_dim * 2  # Output
        )
    else:  # MLA
        # MLA only stores K cache, not V cache
        mem_io = (
            total_seqlen * num_kv_heads * head_dim  # K cache
            + batch_size * num_heads * head_dim * 2  # Q
            + batch_size * num_heads * v_head_dim * 2  # Output
        )
    
    if attention_type == "GQA":
        flops = (
            batch_size
            * total_seqlen
            * num_heads
            * head_dim
            * 2
        )
    else:  # MLA
        # MLA has different compute pattern
        flops = (
            batch_size
            * total_seqlen
            * num_heads
            * v_head_dim
            * 2
        )
    
    throughput_gbs = mem_io * 1e-9 / (avg_time_us * 1e-6)
    throughput_tflops = flops * 1e-12 / (avg_time_us * 1e-6)
    
    # 理论性能（假设H100）
    ideal_h100_time_mem = mem_io / 3.35e12 * 1e6  # microseconds
    ideal_h100_time_flop = flops / 989e12 * 1e6  # microseconds
    ideal_h100_time = max(ideal_h100_time_mem, ideal_h100_time_flop)
    
    print(f"\n{'='*80}")
    print(f"Performance Results:")
    print(f"{'='*80}")
    print(f"Average Latency: {avg_time_us:.2f} us")
    print(f"Throughput: {throughput_gbs:.2f} GB/s")
    print(f"Compute: {throughput_tflops:.2f} TFLOPS/s")
    print(f"Arithmetic Intensity: {flops / mem_io:.2f}")
    print(f"Ideal H100 Time: {ideal_h100_time:.2f} us")
    print(f"Efficiency: {ideal_h100_time / avg_time_us * 100:.1f}%")
    print(f"{'='*80}\n")

    # Profiler (if enabled)
    if enable_profiler:
        sp_rank = get_dist_context().attn_sp_rank if attention_sp > 1 else 0
        print(f"[Rank {sp_rank}] Running profiler ({profiler_iterations} iterations)...")
        torch.cuda.synchronize()
        
        # Note: We don't need a barrier before profiler because profiler runs independently on each rank
        # Each rank will profile its own operations, and forward_fn() will handle distributed communication internally
        
        # Generate output filename with timestamp, rank info, and mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "eager" if not use_cudagraph else "graph"
        output_dir = f"profiler_traces/sp/{mode_str}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"sp_attention_{attention_type.lower()}_sp{attention_sp}_rank{sp_rank}_seq{seq_len}_{mode_str}_{timestamp}.json"
        )
        
        print(f"[Rank {sp_rank}] Profiler output file: {output_file}")
        
        # Create profiler
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        try:
            if use_cudagraph:
                # For CUDA Graph, profile the graph replay
                print(f"[Rank {sp_rank}] Starting CUDA Graph profiler...")
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    for iter_idx in range(profiler_iterations):
                        # Mark iteration boundary in profiler trace
                        with torch.profiler.record_function(f"iteration_{iter_idx}"):
                            graph.replay()
                    torch.cuda.synchronize()
            else:
                # For non-graph mode, profile the forward function
                print(f"[Rank {sp_rank}] Starting eager mode profiler...")
                # For very long sequences, reduce profiler overhead
                use_stack = seq_len < 100000  # Only use stack trace for shorter sequences
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=use_stack,  # Disable stack trace for very long sequences to reduce overhead
                ) as prof:
                    for iter_idx in range(profiler_iterations):
                        # Mark iteration boundary in profiler trace
                        with torch.profiler.record_function(f"iteration_{iter_idx}"):
                            _ = forward_fn()
                        # Print progress every 10 iterations for long runs
                        if (iter_idx + 1) % 10 == 0:
                            print(f"[Rank {sp_rank}] Profiler progress: {iter_idx + 1}/{profiler_iterations} iterations", flush=True)
                    torch.cuda.synchronize()
                print(f"[Rank {sp_rank}] Profiler iterations completed", flush=True)
            
            print(f"[Rank {sp_rank}] Profiler capture completed, exporting trace...", flush=True)
            
            # Export chrome trace
            prof.export_chrome_trace(output_file)
            print(f"[Rank {sp_rank}] Profiler trace saved to: {output_file}", flush=True)
        except Exception as e:
            print(f"[Rank {sp_rank}] Error during profiler: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Get all kernel statistics
        print(f"[Rank {sp_rank}] Processing kernel statistics...", flush=True)
        key_averages = prof.key_averages(group_by_input_shape=True)
        print(f"[Rank {sp_rank}] Found {len(key_averages)} kernel events", flush=True)
        
        # Save kernel statistics to CSV
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Name',
                'Self CPU Time (us)',
                'CPU Time (us)',
                'Self CUDA Time (us)',
                'CUDA Time (us)',
                'Avg Self CUDA Time (us)',
                'Self CPU Memory (bytes)',
                'CPU Memory (bytes)',
                'Self CUDA Memory (bytes)',
                'CUDA Memory (bytes)',
                'Input Shapes',
                'Call Count',
            ])
            
            # Write data rows
            for event in key_averages:
                # Extract input shapes if available
                input_shapes = ''
                if hasattr(event, 'input_shapes') and event.input_shapes:
                    input_shapes = str(event.input_shapes)
                
                # Get time values (already in microseconds for torch.profiler)
                self_cpu_time = getattr(event, 'self_cpu_time_total', 0)
                cpu_time = getattr(event, 'cpu_time_total', 0)
                self_cuda_time = getattr(event, 'self_cuda_time_total', 0)
                cuda_time = getattr(event, 'cuda_time_total', 0)
                count = getattr(event, 'count', 1)
                avg_self_cuda_time = self_cuda_time / count if count > 0 else 0
                
                # Get memory values (in bytes)
                self_cpu_mem = getattr(event, 'self_cpu_memory_usage', 0)
                cpu_mem = getattr(event, 'cpu_memory_usage', 0)
                self_cuda_mem = getattr(event, 'self_cuda_memory_usage', 0)
                cuda_mem = getattr(event, 'cuda_memory_usage', 0)
                
                writer.writerow([
                    event.key,  # Kernel/operator name
                    f'{self_cpu_time:.2f}',
                    f'{cpu_time:.2f}',
                    f'{self_cuda_time:.2f}',
                    f'{cuda_time:.2f}',
                    f'{avg_self_cuda_time:.2f}',
                    self_cpu_mem,
                    cpu_mem,
                    self_cuda_mem,
                    cuda_mem,
                    input_shapes,
                    count,
                ])
        
        print(f"[Rank {sp_rank}] Kernel statistics saved to: {csv_file}", flush=True)
        
        # Print summary
        print(f"\n[Rank {sp_rank}] {'='*80}", flush=True)
        print(f"[Rank {sp_rank}] Profiler Summary (top 10 operators):", flush=True)
        print(f"[Rank {sp_rank}] {'='*80}", flush=True)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"[Rank {sp_rank}] {'='*80}\n", flush=True)
        
        # Ensure all ranks finish before exiting
        # Use barrier across all processes in the process group (not just SP group)
        # This ensures all ranks (even those not in SP group) reach the same point
        if attention_sp > 1 and dist.is_initialized():
            print(f"[Rank {sp_rank}] Waiting for all ranks to finish profiler...", flush=True)
            # Synchronize CUDA first, then barrier across all ranks
            torch.cuda.synchronize()
            dist.barrier()
        print(f"[Rank {sp_rank}] Profiler completed successfully", flush=True)


def main():
    """主函数：运行一系列测试"""
    import argparse

    parser = argparse.ArgumentParser(description="SP Attention CUDA Graph Performance Test (single request)")
    # Note: batch_size is fixed to 1 (single request in entire system), Master Rank=0
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length (single value)")
    parser.add_argument("--seq_lens", type=str, default=None, help="Comma-separated sequence lengths for batch testing (e.g., '1024,2048,4096,8192')")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads (must be 1 for MLA)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (for Q and K). For MLA, must be 576 (512+64)")
    parser.add_argument("--v_head_dim", type=int, default=None, help="V head dimension. For MLA, must be 512 (kv_lora_rank), not 576! Default: 512 for MLA, head_dim for GQA")
    parser.add_argument("--attention_sp", type=int, default=2, help="SP parallelism size")
    parser.add_argument("--attention_type", type=str, default="GQA", choices=["GQA", "MLA"], help="Attention type (GQA or MLA)")
    parser.add_argument("--no_cudagraph", action="store_true", help="Disable CUDA Graph")
    parser.add_argument("--eager", action="store_true", help="Use eager mode (equivalent to --no_cudagraph)")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for tensor shapes")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable profiler to save trace data")
    parser.add_argument("--profiler_iterations", type=int, default=50, help="Number of iterations to profile (default: 50)")
    
    args = parser.parse_args()
    
    # Parse seq_lens: support both --seq_len (single) and --seq_lens (comma-separated)
    if args.seq_lens:
        # Parse comma-separated values
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(',')]
        if args.seq_len is not None:
            print("Warning: Both --seq_len and --seq_lens provided. Using --seq_lens.")
    elif args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        # Default to 4096 if neither is provided
        seq_lens = [4096]
    
    # Run benchmarks for each seq_len
    total_runs = len(seq_lens)
    for idx, seq_len in enumerate(seq_lens, 1):
        print(f"\n{'='*80}")
        print(f"Running benchmark {idx}/{total_runs}: seq_len={seq_len}")
        print(f"{'='*80}\n")
        
        benchmark_sp_attention_with_cudagraph(
            seq_len=seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_kv_heads=args.num_kv_heads,
            attention_sp=args.attention_sp,
            attention_type=args.attention_type,
            v_head_dim=args.v_head_dim,
            use_cudagraph=not (args.no_cudagraph or args.eager),
            num_iterations=args.iterations,
            debug=args.debug,
            enable_profiler=args.enable_profiler,
            profiler_iterations=args.profiler_iterations,
        )
        
        if idx < total_runs:
            print(f"\nCompleted {idx}/{total_runs} runs. Continuing to next seq_len...\n")
    
    print(f"\n{'='*80}")
    print(f"All benchmarks completed! Total runs: {total_runs}")
    print(f"Sequence lengths tested: {seq_lens}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

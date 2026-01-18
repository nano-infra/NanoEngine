"""
性能测试：SP Attention with CUDA Graph

这个测试用例用于测试Sequence Parallel (SP) Attention在使用CUDA Graph时的性能。
支持两种Attention类型：
1. GQA (Grouped Query Attention) - 使用FlashAttention
2. MLA (Multi-head Latent Attention) - 使用FlashMLA

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
    q_mask: torch.Tensor,
    res_lse_mask: torch.Tensor,
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
        q_mask: Q mask for all-to-all [sp_size, batch_size]
        res_lse_mask: Result/LSE mask for all-to-all [sp_size, batch_size]
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
    q = q_buffer.all_to_all_ll(
        q.view([bs, -1]),
        mask=q_mask,
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
        causal=False,
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
        mask=res_lse_mask,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, head_dim)
    all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
        lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask,
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
    q_mask: torch.Tensor,
    res_lse_mask: torch.Tensor,
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
) -> torch.Tensor:
    """
    SP Attention Forward Pass (MLA with FlashMLA)
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor (MLA only needs k_cache, not v_cache)
        context_lens: Context lengths for attention computation
        block_tables: Block tables for paged attention
        global_context_lens: Global context lengths [sp_size, batch_size]
        q_mask: Q mask for all-to-all [sp_size, batch_size]
        res_lse_mask: Result/LSE mask for all-to-all [sp_size, batch_size]
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
    q = q_buffer.all_to_all_ll(
        q.view([bs, -1]),
        mask=q_mask,
    ).view([sp_size * max_num_seqs, num_head, head_dim_q])

    q = q[:attention_compute_bs]
    context_lens_for_attn = context_lens[:attention_compute_bs]
    block_tables_for_attn = block_tables[:attention_compute_bs]

    # Get MLA metadata
    # For MLA: num_query_heads_per_kv = num_heads // num_kv_heads = num_heads // 1 = num_heads
    # This matches the project implementation: self.num_heads // self.num_kv_heads
    num_query_heads_per_kv = num_heads // 1  # For MLA, num_kv_heads is always 1
    print(f"[TEST] get_mla_metadata: context_lens_for_attn.shape={context_lens_for_attn.shape}, "
          f"num_query_heads_per_kv={num_query_heads_per_kv}, num_kv_heads=1")
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        context_lens_for_attn,
        num_query_heads_per_kv,  # num_heads // num_kv_heads (same as project: self.num_heads // self.num_kv_heads)
        1,  # num_kv_heads (MLA requires num_kv_heads == 1)
    )

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
    print(f"[TEST] Before flash_mla_with_kvcache:")
    print(f"  q.shape: {q.shape}, q_for_attn.shape: {q_for_attn.shape}")
    print(f"  k_cache.shape: {k_cache.shape}")
    print(f"  k_cache last dimension: {k_cache.shape[-1]}")
    print(f"  block_tables_for_attn.shape: {block_tables_for_attn.shape}")
    print(f"  context_lens_for_attn.shape: {context_lens_for_attn.shape}")
    print(f"  v_head_dim parameter: {v_head_dim} (type: {type(v_head_dim)})")
    print(f"  head_dim parameter: {head_dim} (type: {type(head_dim)})")
    print(f"  num_heads: {num_heads}, num_kv_heads: 1")
    print(f"  scale: {scale}, causal: False")
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
        causal=False,
    )
    
    print(f"[TEST] After flash_mla_with_kvcache:")
    print(f"  o.shape (before squeeze): {o.shape}")
    print(f"  lse.shape: {lse.shape}")

    o = o.squeeze(1)
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
        mask=res_lse_mask,
        is_transpose=True,
    ).view(sp_size, max_num_seqs, num_head, v_head_dim)
    all_ranks_lse_output_combine = lse_buffer.all_to_all_ll(
        lse_all_to_all_input_buffer.view(sp_size * max_num_seqs, -1),
        mask=res_lse_mask,
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
    batch_size: int,
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
    """创建测试数据"""
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

    # Block tables
    num_pages = seq_len // page_size
    block_tables = torch.arange(
        batch_size * num_pages, device=device, dtype=torch.int32
    ).view(batch_size, num_pages)

    # Context lengths
    context_lens = torch.full(
        (max_num_seqs,), seq_len, dtype=torch.int32, device=device
    )

    # Global context lengths [sp_size, batch_size]
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

    # Masks and indices
    q_mask = torch.zeros(sp_size, max_num_seqs, dtype=torch.int32, device=device)
    q_mask[:, :batch_size] = 1

    res_lse_mask = torch.zeros(sp_size, max_num_seqs, dtype=torch.int32, device=device)
    res_lse_mask[:, :batch_size] = 1

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
    batch_size: int,
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
):
    """
    使用CUDA Graph对SP Attention进行性能测试
    
    Args:
        batch_size: Batch size
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
    """
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
    max_num_seqs = batch_size * 2  # 留一些空间给recv sequences
    
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
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Num Heads: {num_heads}, Num KV Heads: {num_kv_heads}")
    print(f"Head Dim: {head_dim}")
    if attention_type == "MLA":
        print(f"V Head Dim: {v_head_dim}")
    print(f"SP Size: {attention_sp}")
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
        return

    # 创建测试数据
    test_data = create_test_data(
        batch_size=batch_size,
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

    # 创建forward函数
    if attention_type == "GQA":
        def forward_fn():
            return sp_attention_forward_gqa(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                v_cache=test_data["v_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                global_context_lens=test_data["global_context_lens"],
                q_mask=test_data["q_mask"],
                res_lse_mask=test_data["res_lse_mask"],
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
            return sp_attention_forward_mla(
                q=test_data["q"],
                k_cache=test_data["k_cache"],
                context_lens=test_data["context_lens"],
                block_tables=test_data["block_tables"],
                global_context_lens=test_data["global_context_lens"],
                q_mask=test_data["q_mask"],
                res_lse_mask=test_data["res_lse_mask"],
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
            )

    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = forward_fn()
    torch.cuda.synchronize()

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


def main():
    """主函数：运行一系列测试"""
    import argparse

    parser = argparse.ArgumentParser(description="SP Attention CUDA Graph Performance Test")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads (must be 1 for MLA)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (for Q and K). For MLA, must be 576 (512+64)")
    parser.add_argument("--v_head_dim", type=int, default=None, help="V head dimension. For MLA, must be 512 (kv_lora_rank), not 576! Default: 512 for MLA, head_dim for GQA")
    parser.add_argument("--attention_sp", type=int, default=2, help="SP parallelism size")
    parser.add_argument("--attention_type", type=str, default="GQA", choices=["GQA", "MLA"], help="Attention type (GQA or MLA)")
    parser.add_argument("--no_cudagraph", action="store_true", help="Disable CUDA Graph")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    
    args = parser.parse_args()

    benchmark_sp_attention_with_cudagraph(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads,
        attention_sp=args.attention_sp,
        attention_type=args.attention_type,
        v_head_dim=args.v_head_dim,
        use_cudagraph=not args.no_cudagraph,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()

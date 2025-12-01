import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_D": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_D": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_D": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_D": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    ],
    key=["D", "M", "H"],  # 根据 input shape 自动选择最佳配置
)
@triton.jit
def copy_batch_indexed_kernel_opt(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    mask_ptr,
    B,
    H,
    D,
    M,
    strideB_src,
    strideH_src,
    strideD_src,
    strideB_dst,
    strideH_dst,
    strideD_dst,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_h = H

    GROUP_SIZE_M = 8
    num_pid_m = grid_m
    num_pid_in_group = GROUP_SIZE_M * grid_h
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_h = (pid % num_pid_in_group) // group_size_m

    if pid_m >= grid_m or pid_h >= grid_h:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
    dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)
    m_mask_val = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)

    is_active = (
        (m_mask_val == 1)
        & (src_idx >= 0)
        & (src_idx < B)
        & (dst_idx >= 0)
        & (dst_idx < B)
    )
    p_valid = mask_m & is_active

    src_base = (src_idx * strideB_src + pid_h * strideH_src)[:, None]
    dst_base = (dst_idx * strideB_dst + pid_h * strideH_dst)[:, None]

    num_full_blocks = D // BLOCK_D

    for i in range(num_full_blocks):
        d_offset = i * BLOCK_D
        offs_d = d_offset + tl.arange(0, BLOCK_D)

        curr_src_ptr = src_ptr + src_base + offs_d[None, :] * strideD_src
        curr_dst_ptr = dst_ptr + dst_base + offs_d[None, :] * strideD_dst

        val = tl.load(curr_src_ptr, mask=p_valid[:, None], other=0.0)
        tl.store(curr_dst_ptr, val, mask=p_valid[:, None])

    if num_full_blocks * BLOCK_D < D:
        d_offset = num_full_blocks * BLOCK_D
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        curr_src_ptr = src_ptr + src_base + offs_d[None, :] * strideD_src
        curr_dst_ptr = dst_ptr + dst_base + offs_d[None, :] * strideD_dst

        curr_mask = p_valid[:, None] & mask_d[None, :]

        val = tl.load(curr_src_ptr, mask=curr_mask, other=0.0)
        tl.store(curr_dst_ptr, val, mask=curr_mask)


# Specialized kernel for D=1 (Keep fast path for extremely small D)
@triton.jit
def copy_batch_indexed_kernel_D1(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    mask_ptr,
    B,
    H,
    M,
    strideB_src,
    strideH_src,
    strideD_src,
    strideB_dst,
    strideH_dst,
    strideD_dst,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)

    pid_m = pid % grid_m
    pid_h = pid // grid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
    dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)
    m_mask = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)

    p_valid = (
        mask_m
        & (m_mask == 1)
        & (src_idx >= 0)
        & (src_idx < B)
        & (dst_idx >= 0)
        & (dst_idx < B)
    )

    src_offsets = src_idx * strideB_src + pid_h * strideH_src
    dst_offsets = dst_idx * strideB_dst + pid_h * strideH_dst

    val = tl.load(src_ptr + src_offsets, mask=p_valid, other=0.0)
    tl.store(dst_ptr + dst_offsets, val, mask=p_valid)


def copy_batch_indexed_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    mask: torch.Tensor,
    # block_d/num_warps deprecated, handled by autotuner
    **kwargs,
):
    assert src.is_cuda and dst.is_cuda
    assert src.is_contiguous() and dst.is_contiguous()

    B, H, D = src.shape
    M = src_idx.numel()

    # 极小 D 优化
    if D == 1:
        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M) * H,)
        copy_batch_indexed_kernel_D1[grid](
            src,
            dst,
            src_idx,
            dst_idx,
            mask,
            B,
            H,
            M,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            dst.stride(0),
            dst.stride(1),
            dst.stride(2),
            BLOCK_M=BLOCK_M,
        )
        return

    # 普通情况 (D > 1)
    def grid_fn(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * H,)

    copy_batch_indexed_kernel_opt[grid_fn](
        src,
        dst,
        src_idx,
        dst_idx,
        mask,
        B,
        H,
        D,
        M,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
    )

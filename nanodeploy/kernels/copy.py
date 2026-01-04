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
    key=["D", "H"],
)
@triton.jit
def copy_batch_indexed_kernel_opt(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    mask_ptr,
    B_src,
    B_dst,
    H,
    D,
    M,
    strideB_src,
    strideH_src,
    strideD_src,
    strideB_dst,
    strideH_dst,
    strideD_dst,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
    dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)
    m_mask_val = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)

    is_active = (
        (m_mask_val == 1)
        & (src_idx >= 0)
        & (src_idx < B_src)
        & (dst_idx >= 0)
        & (dst_idx < B_dst)
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

        mask_curr = p_valid[:, None]
        val = tl.load(curr_src_ptr, mask=mask_curr, other=0.0)
        tl.store(curr_dst_ptr, val, mask=mask_curr)

    if num_full_blocks * BLOCK_D < D:
        d_offset = num_full_blocks * BLOCK_D
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        curr_src_ptr = src_ptr + src_base + offs_d[None, :] * strideD_src
        curr_dst_ptr = dst_ptr + dst_base + offs_d[None, :] * strideD_dst

        curr_mask = p_valid[:, None] & mask_d[None, :]

        val = tl.load(curr_src_ptr, mask=curr_mask, other=0.0)
        tl.store(curr_dst_ptr, val, mask=curr_mask)


@triton.jit
def copy_batch_indexed_kernel_D1(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    mask_ptr,
    B_src,
    B_dst,
    H,
    M,
    strideB_src,
    strideH_src,
    strideD_src,
    strideB_dst,
    strideH_dst,
    strideD_dst,
    dtype: tl.constexpr,
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
        & (src_idx < B_src)
        & (dst_idx >= 0)
        & (dst_idx < B_dst)
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
    **kwargs,
):
    B_src, H, D = src.shape
    B_dst, _, _ = dst.shape
    M = src_idx.numel()

    dtype_mapping = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_dtype = dtype_mapping.get(src.dtype, tl.float32)

    # --- Case 1: 极小 D 优化 (LSE Copy) ---
    if D == 1:
        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M) * H,)
        copy_batch_indexed_kernel_D1[grid](
            src,
            dst,
            src_idx,
            dst_idx,
            mask,
            B_src,
            B_dst,
            H,
            M,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            dst.stride(0),
            dst.stride(1),
            dst.stride(2),
            dtype=triton_dtype,
            BLOCK_M=BLOCK_M,
        )
        return

    # --- Case 2: 普通情况 (D > 1) ---
    def grid_fn(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), H)

    copy_batch_indexed_kernel_opt[grid_fn](
        src,
        dst,
        src_idx,
        dst_idx,
        mask,
        B_src,
        B_dst,
        H,
        D,
        M,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dtype=triton_dtype,
    )


def warmup_copy_kernel(shapes=None, dtype=torch.float16):
    """
    Warm up the ``copy_batch_indexed_triton`` kernel.

    Rationale:
    Since the autotune key only contains ["D", "H"], we only need to invoke the kernel
    once for each unique (H, D) pair to trigger compilation and populate the cache.
    For subsequent real inference requests, regardless of batch size, as long as H and D
    hit the cache, there will be no additional compilation overhead.

    Args:
        shapes: Optional. A list of (B, H, D, ...) tuples. If not provided, commonly used
            configurations hard-coded in this function will be used.
        dtype: Data type used during warmup. It is recommended to keep it the same as the
            one used during inference (default: float16).
    """
    if shapes is None:
        configs = [
            # ds3 attn 系列
            (8, 128, 576, "ds3 small attn"),
            (64, 128, 576, "ds3 medium attn"),
            (128, 128, 576, "ds3 large attn"),
            # ds3 q 系列
            (8, 128, 512, "ds3 small q"),
            (64, 128, 512, "ds3 medium q"),
            (128, 128, 512, "ds3 large q"),
            # ds3 lse 系列
            (8, 128, 1, "ds3 small lse"),
            (64, 128, 1, "ds3 medium lse"),
            (128, 128, 1, "ds3 large lse"),
            # qwen3 q 系列
            (8, 64, 128, "qwen3 small q"),
            (64, 64, 128, "qwen3 medium q"),
            (128, 64, 128, "qwen3 large q"),
            # qwen3 lse 系列
            (8, 64, 1, "qwen3 small lse"),
            (64, 64, 1, "qwen3 medium lse"),
            (128, 64, 1, "qwen3 large lse"),
        ]
        unique_shapes = list(set((c[1], c[2]) for c in configs))
    else:
        unique_shapes = list(set((s[1], s[2]) for s in shapes))

    unique_shapes.sort()

    print(f"[Warmup] Starting warmup for {len(unique_shapes)} unique (H, D) configs...")

    M_warmup = 32
    device = torch.device("cuda")

    max_h = max(s[0] for s in unique_shapes)
    max_d = max(s[1] for s in unique_shapes)

    B_src_dst = M_warmup * 2

    try:
        src_buffer = torch.randn((B_src_dst, max_h, max_d), device=device, dtype=dtype)
        dst_buffer = torch.randn((B_src_dst, max_h, max_d), device=device, dtype=dtype)

        src_idx = torch.arange(M_warmup, device=device, dtype=torch.int32)
        dst_idx = torch.arange(M_warmup, device=device, dtype=torch.int32)
        mask = torch.ones(M_warmup, device=device, dtype=torch.int32)

        for H, D in unique_shapes:
            curr_src = src_buffer[:, :H, :D].contiguous()
            curr_dst = dst_buffer[:, :H, :D].contiguous()

            print(f"[Warmup] Compiling/Running Copy Kernel for H={H}, D={D} ...")

            copy_batch_indexed_triton(curr_src, curr_dst, src_idx, dst_idx, mask)

        torch.cuda.synchronize()
        print(f"[Warmup] Completed. Triton kernels are cached.")

    except Exception as e:
        import traceback

        print(f"[Warmup] Failed! Error: {e}")
        traceback.print_exc()

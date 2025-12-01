import math
import random
import time

import torch
import triton
import triton.language as tl


# =========================
# Kernel A: 3D sub-block copy
# =========================
@triton.jit
def copy_3d_block_kernel(
    src_ptr,
    dst_ptr,
    b0,
    h0,
    d0,  # source start offsets
    b1,
    h1,
    d1,  # dest start offsets
    b_len,
    h_len,
    d_len,
    strideB_src,
    strideH_src,
    strideD_src,
    strideB_dst,
    strideH_dst,
    strideD_dst,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    # bounds
    if pid_b >= b_len or pid_h >= h_len:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_len

    # compute absolute positions
    b_src = b0 + pid_b
    h_src = h0 + pid_h
    d_src = d0 + offs_d

    b_dst = b1 + pid_b
    h_dst = h1 + pid_h
    d_dst = d1 + offs_d

    # linearized pointer offsets
    src_offsets = b_src * strideB_src + h_src * strideH_src + d_src * strideD_src
    dst_offsets = b_dst * strideB_dst + h_dst * strideH_dst + d_dst * strideD_dst

    vals = tl.load(src_ptr + src_offsets, mask=mask_d, other=0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask_d)


def copy_3d_block_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_start,
    dst_start,
    copy_sizes,
    block_d=128,
    num_warps=None,
):
    """
    src_start: (b0, h0, d0)
    dst_start: (b1, h1, d1)
    copy_sizes: (b_len, h_len, d_len)
    """
    assert src.is_cuda and dst.is_cuda
    assert src.dtype == dst.dtype
    assert src.dim() == 3 and dst.dim() == 3

    (b0, h0, d0) = src_start
    (b1, h1, d1) = dst_start
    (b_len, h_len, d_len) = copy_sizes

    B, H, D = src.shape
    B2, H2, D2 = dst.shape
    assert b0 + b_len <= B and h0 + h_len <= H and d0 + d_len <= D
    assert b1 + b_len <= B2 and h1 + h_len <= H2 and d1 + d_len <= D2

    # pick warps heuristically
    if num_warps is None:
        if d_len >= 512:
            num_warps = 8
        elif d_len >= 128:
            num_warps = 4
        else:
            num_warps = 2

    grid = (b_len, h_len, triton.cdiv(d_len, block_d))

    copy_3d_block_kernel[grid](
        src,
        dst,
        b0,
        h0,
        d0,
        b1,
        h1,
        d1,
        b_len,
        h_len,
        d_len,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


# =========================
# Kernel B: batch-indexed mapping copy
# =========================
@triton.jit
def copy_batch_indexed_kernel(
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
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # request index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # block along D

    if pid_m >= M or pid_h >= H:
        return

    # check mask: if 0, skip
    m = tl.load(mask_ptr + pid_m)  # removed eviction_policy
    if m == 0:
        return

    b_src = tl.load(src_idx_ptr + pid_m)  # removed eviction_policy
    b_dst = tl.load(dst_idx_ptr + pid_m)  # removed eviction_policy

    # bounds guard for batch indices
    if (b_src < 0) | (b_src >= B) | (b_dst < 0) | (b_dst >= B):
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    src_offsets = b_src * strideB_src + pid_h * strideH_src + offs_d * strideD_src
    dst_offsets = b_dst * strideB_dst + pid_h * strideH_dst + offs_d * strideD_dst

    vals = tl.load(src_ptr + src_offsets, mask=mask_d, other=0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask_d)


def copy_batch_indexed_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    mask: torch.Tensor,
    block_d=128,
    num_warps=None,
):
    """
    src, dst: [B, H, D]
    src_idx, dst_idx, mask: shape [M_max], with mask in {0,1}
    Only positions where mask[i]==1 are copied: dst[dst_idx[i], :, :] = src[src_idx[i], :, :]
    """
    assert src.is_cuda and dst.is_cuda
    assert src.dtype == dst.dtype
    assert src.dim() == 3 and dst.dim() == 3
    assert src.is_contiguous(
        memory_format=torch.contiguous_format
    ) and dst.is_contiguous(memory_format=torch.contiguous_format)

    B, H, D = src.shape
    M = src_idx.numel()
    assert dst_idx.numel() == M and mask.numel() == M

    # choose warps
    if num_warps is None:
        if D >= 512:
            num_warps = 8
        elif D >= 128:
            num_warps = 4
        else:
            num_warps = 2

    grid = (M, H, triton.cdiv(D, block_d))

    copy_batch_indexed_kernel[grid](
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
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


# =========================
# Test & Benchmark (with CUDA Graph)
# =========================


def check_correctness_subblock():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B, H, D = 16, 64, 512
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst = torch.zeros_like(src)

    # define copy: a 3D sub-block
    src_start = (2, 3, 128)
    dst_start = (5, 10, 256)
    copy_sizes = (4, 8, 192)  # copy 4 batches × 8 heads × 192 dims

    # baseline using PyTorch slicing
    dst_baseline = torch.zeros_like(dst)
    sb = slice(src_start[0], src_start[0] + copy_sizes[0])
    sh = slice(src_start[1], src_start[1] + copy_sizes[1])
    sd = slice(src_start[2], src_start[2] + copy_sizes[2])

    db = slice(dst_start[0], dst_start[0] + copy_sizes[0])
    dh = slice(dst_start[1], dst_start[1] + copy_sizes[1])
    dd = slice(dst_start[2], dst_start[2] + copy_sizes[2])

    dst_baseline[db, dh, dd].copy_(src[sb, sh, sd])

    # Triton
    dst_triton = torch.zeros_like(dst)
    copy_3d_block_triton(src, dst_triton, src_start, dst_start, copy_sizes)

    # correctness
    diff = (dst_triton - dst_baseline).abs().max().item()
    print(f"[Correctness][Sub-block] max abs diff: {diff:.6f}")
    assert diff == 0.0, "Sub-block copy mismatch!"


def check_correctness_batch_indexed():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B, H, D = 32, 128, 576
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst = torch.zeros_like(src)

    M_max = 256
    src_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    dst_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    mask = torch.zeros((M_max,), device=device, dtype=torch.int32)

    # Example: copy entries 1 and 3 -> to 2 and 4
    active_pairs = [(1, 2), (3, 4)]
    for i, (s, d) in enumerate(active_pairs):
        src_idx_long[i] = s
        dst_idx_long[i] = d
        mask[i] = 1

    # baseline
    dst_baseline = torch.zeros_like(dst)
    active_mask = mask.bool()
    src_active = src_idx_long[active_mask]  # long
    dst_active = dst_idx_long[active_mask]  # long
    dst_baseline.index_copy_(0, dst_active, src.index_select(0, src_active))

    # Triton (convert to int32 for kernel)
    src_idx_i32 = src_idx_long.to(torch.int32)
    dst_idx_i32 = dst_idx_long.to(torch.int32)
    dst_triton = torch.zeros_like(dst)
    copy_batch_indexed_triton(src, dst_triton, src_idx_i32, dst_idx_i32, mask)

    diff = (dst_triton - dst_baseline).abs().max().item()
    print(f"[Correctness][Batch-indexed] max abs diff: {diff:.6f}")
    assert diff == 0.0, "Batch-indexed copy mismatch!"


def benchmark_with_cuda_graph_subblock():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # Use sizes representative of your workloads
    B, H, D = 32, 128, 512
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst_t = torch.zeros_like(src)
    dst_p = torch.zeros_like(src)

    # define copy range
    src_start = (0, 32, 64)
    dst_start = (16, 64, 128)
    copy_sizes = (8, 32, 256)

    # Warmup Triton compilation
    copy_3d_block_triton(src, dst_t, src_start, dst_start, copy_sizes)

    # CUDA Graph: Triton
    stream = torch.cuda.current_stream()
    g1 = torch.cuda.CUDAGraph()
    # allocate static tensors for graph
    static_src = src.clone()
    static_dst = torch.zeros_like(src)

    # capture
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(g1):
        copy_3d_block_triton(static_src, static_dst, src_start, dst_start, copy_sizes)

    # CUDA Graph: PyTorch baseline (slice copy_)
    g2 = torch.cuda.CUDAGraph()
    static_dst2 = torch.zeros_like(src)

    sb = slice(src_start[0], src_start[0] + copy_sizes[0])
    sh = slice(src_start[1], src_start[1] + copy_sizes[1])
    sd = slice(src_start[2], src_start[2] + copy_sizes[2])

    db = slice(dst_start[0], dst_start[0] + copy_sizes[0])
    dh = slice(dst_start[1], dst_start[1] + copy_sizes[1])
    dd = slice(dst_start[2], dst_start[2] + copy_sizes[2])

    torch.cuda.synchronize()
    with torch.cuda.graph(g2):
        static_dst2[db, dh, dd].copy_(static_src[sb, sh, sd])

    # timing by CUDA events
    iters = 200
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    e_start.record()
    for _ in range(iters):
        g1.replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    e_start.record()
    for _ in range(iters):
        g2.replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    # correctness check
    # Run both once to compare
    static_dst.zero_()
    static_dst2.zero_()
    g1.replay()
    g2.replay()
    diff = (static_dst - static_dst2).abs().max().item()

    print(
        f"[Graph][Sub-block] Triton avg ms: {triton_ms:.4f}, Torch slice avg ms: {torch_ms:.4f}, max diff: {diff:.6f}"
    )


def benchmark_with_cuda_graph_batch_indexed():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    B, H, D = 64, 128, 576  # DeepSeek/Qwen-like sizes
    src = torch.randn(B, H, D, device=device, dtype=dtype)

    M_max = 256
    # Triton侧索引与掩码用 int32，无问题
    src_idx = torch.full((M_max,), -1, device=device, dtype=torch.int32)
    dst_idx = torch.full((M_max,), -1, device=device, dtype=torch.int32)
    mask = torch.zeros((M_max,), device=device, dtype=torch.int32)

    # Create a variable-length active set, e.g., 64 pairs
    pairs = [(i, (i + 5) % B) for i in range(64)]
    for i, (s, d) in enumerate(pairs):
        src_idx[i] = s
        dst_idx[i] = d
        mask[i] = 1

    # Warmup Triton compilation
    dst_t = torch.zeros_like(src)
    copy_batch_indexed_triton(src, dst_t, src_idx, dst_idx, mask)

    # CUDA Graph: Triton
    g1 = torch.cuda.CUDAGraph()
    static_src = src.clone()
    static_dst = torch.zeros_like(src)
    static_src_idx = src_idx.clone()
    static_dst_idx = dst_idx.clone()
    static_mask = mask.clone()

    torch.cuda.synchronize()
    with torch.cuda.graph(g1):
        copy_batch_indexed_triton(
            static_src, static_dst, static_src_idx, static_dst_idx, static_mask
        )

    # CUDA Graph: PyTorch baseline using index_copy_
    # 关键修复：在捕获前把索引转为 long
    g2 = torch.cuda.CUDAGraph()
    static_dst2 = torch.zeros_like(src)
    active = static_mask.bool()
    active_src_long = static_src_idx[active].to(torch.long)  # convert to long
    active_dst_long = static_dst_idx[active].to(torch.long)  # convert to long

    torch.cuda.synchronize()
    with torch.cuda.graph(g2):
        static_dst2.index_copy_(
            0, active_dst_long, static_src.index_select(0, active_src_long)
        )

    # timing
    iters = 200
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    e_start.record()
    for _ in range(iters):
        g1.replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    e_start.record()
    for _ in range(iters):
        g2.replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    # correctness
    static_dst.zero_()
    static_dst2.zero_()
    g1.replay()
    g2.replay()
    diff = (static_dst - static_dst2).abs().max().item()

    print(
        f"[Graph][Batch-indexed] Triton avg ms: {triton_ms:.4f}, Torch index_copy_ avg ms: {torch_ms:.4f}, max diff: {diff:.6f}"
    )


# Constants
B_MAX = 256  # 最大 batch，确保能覆盖最多 256 个请求
M_MAX = 256  # 索引缓冲最大容量
# 注意：每个 (H, D) 配置我们会捕获一张 Triton 图，图内形状固定为 [B_MAX, H, D]


def build_triton_graph_for_config(H, D, dtype=torch.float16, device="cuda"):
    """
    为一个 (H, D) 配置捕获一张 Triton CUDA Graph。
    形状固定为 [B_MAX, H, D]，索引容量 M_MAX。
    """
    static_src = torch.zeros(B_MAX, H, D, device=device, dtype=dtype)
    static_dst = torch.zeros_like(static_src)
    static_src_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_dst_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_mask = torch.zeros((M_MAX,), device=device, dtype=torch.int32)

    # 预热编译
    copy_batch_indexed_triton(
        static_src, static_dst, static_src_idx, static_dst_idx, static_mask
    )

    # 捕获图
    g_triton = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g_triton):
        copy_batch_indexed_triton(
            static_src, static_dst, static_src_idx, static_dst_idx, static_mask
        )

    return {
        "g": g_triton,
        "src": static_src,
        "dst": static_dst,
        "src_idx": static_src_idx,
        "dst_idx": static_dst_idx,
        "mask": static_mask,
        "H": H,
        "D": D,
    }


def prepare_case_into_triton_config(static, B, k_requests, mode):
    """
    将一个具体案例写入某个 (H,D) 配置的 Triton 静态缓冲。
    - B: 使用的有效 batch（不超过 B_MAX）
    - k_requests: 请求数量（不超过 M_MAX）
    - mode: 'contig' 或 'noncontig'
    """

    H, D = static["H"], static["D"]

    # 清空缓冲
    static["dst"].zero_()
    static["src"].zero_()
    static["src_idx"].fill_(-1)
    static["dst_idx"].fill_(-1)
    static["mask"].zero_()

    # 构造源数据（只填充前 B 个 batch，其他保持 0，以免影响对比）
    src_case = torch.randn(
        B, H, D, device=static["src"].device, dtype=static["src"].dtype
    )
    static["src"][:B].copy_(src_case)

    # 构建 batch 映射对
    if mode == "contig":
        start_src = 0
        start_dst = B // 2
        src_list = list(range(start_src, start_src + k_requests))
        dst_list = [(start_dst + i) % B for i in range(k_requests)]
    else:
        src_list = random.sample(range(B), k_requests)
        dst_list = random.sample(range(B), k_requests)

    # 写入索引与掩码（激活前 k）
    for i, (s, d) in enumerate(zip(src_list, dst_list)):
        static["src_idx"][i] = s
        static["dst_idx"][i] = d
        static["mask"][i] = 1

    return src_case, src_list, dst_list  # 返回用于 PyTorch 基线的源与索引


def build_torch_graph_for_case(src_case, pairs, device="cuda"):
    """
    为一个具体案例捕获 PyTorch 基线的图。
    使用 index_copy_ + index_select，索引为 long。
    """
    g_torch = torch.cuda.CUDAGraph()
    static_src = src_case.clone()
    static_dst2 = torch.zeros_like(src_case)

    src_idx_long = torch.tensor(
        [s for (s, _) in pairs], device=device, dtype=torch.long
    )
    dst_idx_long = torch.tensor(
        [d for (_, d) in pairs], device=device, dtype=torch.long
    )

    torch.cuda.synchronize()
    with torch.cuda.graph(g_torch):
        static_dst2.index_copy_(
            0, dst_idx_long, static_src.index_select(0, src_idx_long)
        )

    return g_torch, static_dst2


def run_case_on_config(shared_cfg, B, k_requests, mode, iters=200):
    """
    在已经捕获好的 (H,D) Triton 图上运行一个具体案例，
    并为该案例捕获 PyTorch 基线图，返回延迟对比与正确性。
    """
    device = "cuda"
    dtype = torch.float16

    # 准备数据写入 Triton 静态缓冲
    src_case, src_list, dst_list = prepare_case_into_triton_config(
        shared_cfg, B, k_requests, mode
    )
    pairs = list(zip(src_list, dst_list))

    # 捕获该案例的 PyTorch 基线图
    g_torch, static_dst2 = build_torch_graph_for_case(src_case, pairs, device=device)

    # 定时事件
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    # Triton（同一图重放）
    e_start.record()
    for _ in range(iters):
        shared_cfg["g"].replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    # Torch（该案例图重放）
    e_start.record()
    for _ in range(iters):
        g_torch.replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    # 正确性：比较 Triton 的结果（取前 B 批次的切片）与 PyTorch 结果
    triton_slice = shared_cfg["dst"][:B]
    diff = (triton_slice - static_dst2).abs().max().item()

    return triton_ms, torch_ms, diff


def benchmark_suite_per_config_graphs():
    """
    对每个 (H, D) 配置捕获一张 Triton 图；对每个具体案例捕获一张 PyTorch 基线图。
    测不同请求数量与模式，打印汇总表。
    """
    device = "cuda"
    dtype = torch.float16
    random.seed(0)

    configs = [
        (128, 512),
        (128, 576),
        (128, 1),
        (64, 128),
        (128, 1),
    ]
    request_counts = [2, 8, 32, 64, 128, 256]
    modes = ["contig", "noncontig"]

    # 为每个配置捕获一张 Triton 图
    triton_graphs = {}
    for H, D in configs:
        triton_graphs[(H, D)] = build_triton_graph_for_config(
            H, D, dtype=dtype, device=device
        )

    results = []
    for H, D in configs:
        shared_cfg = triton_graphs[(H, D)]
        B = B_MAX  # 保持固定 B=256
        for k in request_counts:
            for mode in modes:
                triton_ms, torch_ms, diff = run_case_on_config(
                    shared_cfg, B=B, k_requests=k, mode=mode, iters=200
                )
                speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
                results.append(
                    {
                        "H": H,
                        "D": D,
                        "k": k,
                        "mode": mode,
                        "torch_ms": torch_ms,
                        "triton_ms": triton_ms,
                        "speedup": speedup,
                        "correct": (diff == 0.0),
                    }
                )
                print(
                    f"[Per-Config] H={H}, D={D}, k={k}, mode={mode}: "
                    f"Triton {triton_ms:.4f} ms, Torch {torch_ms:.4f} ms, "
                    f"Speedup {speedup:.2f}x, Correct={diff==0.0}"
                )

    # 汇总表
    print("\n===== Benchmark Summary (Per-Config Triton Graphs) =====")
    print(
        f"{'H':>5} {'D':>6} {'k':>6} {'mode':>10} {'Torch(ms)':>12} {'Triton(ms)':>12} {'Speedup':>10} {'Correct':>8}"
    )
    for r in results:
        print(
            f"{r['H']:>5} {r['D']:>6} {r['k']:>6} {r['mode']:>10} "
            f"{r['torch_ms']:>12.4f} {r['triton_ms']:>12.4f} {r['speedup']:>10.2f} {str(r['correct']):>8}"
        )


def main():
    # 保留基础正确性与单例演示
    check_correctness_subblock()
    check_correctness_batch_indexed()
    benchmark_with_cuda_graph_subblock()
    benchmark_with_cuda_graph_batch_indexed()

    # 运行每配置一张 Triton 图的测试套件
    benchmark_suite_per_config_graphs()


if __name__ == "__main__":
    main()

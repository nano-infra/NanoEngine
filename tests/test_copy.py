import random

import torch
import triton
import triton.language as tl


# Fused general kernel (handles any D > 1)
# 优化点：
# 1. 移除了内部的 static_range(BLOCK_M) 循环，改为 2D Block 处理。
# 2. 引入了 [BLOCK_M, 1] 和 [1, BLOCK_D] 的广播机制，实现并行搬运。
# 3. 索引只加载一次。
@triton.jit
def copy_batch_indexed_kernel_fused(
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
    BLOCK_M: tl.constexpr,
):
    # 1. 确定当前 Block 处理的 M 范围
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 2. 向量化加载索引 (Vectorized Load Indices)
    # 即使 offs_m 超出 M，通过 mask 保护加载，避免非法内存访问
    mask_m = offs_m < M

    # 加载 mask, src_idx, dst_idx (形状均为 [BLOCK_M])
    m_mask = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)
    src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
    dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)

    # 3. 计算请求的有效性 (Validity Check)
    # 逻辑与原代码一致，但现在是并行计算整个向量
    p_valid = (
        (mask_m)
        & (m_mask == 1)
        & (src_idx >= 0)
        & (src_idx < B)
        & (dst_idx >= 0)
        & (dst_idx < B)
    )

    # 4. 计算 Base 指针 (广播到 [BLOCK_M, 1])
    # src_base: [BLOCK_M, 1]
    src_base = (src_idx * strideB_src + pid_h * strideH_src)[:, None]
    dst_base = (dst_idx * strideB_dst + pid_h * strideH_dst)[:, None]

    # 5. 循环处理 D 维度 (Chunked Loop over D)
    # 使用 tl.range 替代 while，更符合 Triton 风格
    for d_start in tl.range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # 6. 计算 2D 指针矩阵
        # src_base [BLOCK_M, 1] + offs_d [1, BLOCK_D] * stride -> [BLOCK_M, BLOCK_D]
        src_ptrs = src_ptr + src_base + (offs_d[None, :] * strideD_src)
        dst_ptrs = dst_ptr + dst_base + (offs_d[None, :] * strideD_dst)

        # 7. 合并 Mask
        # 请求有效且 D 维度在范围内
        curr_mask = p_valid[:, None] & mask_d[None, :]

        # 8. 块读写
        val = tl.load(src_ptrs, mask=curr_mask, other=0.0)
        tl.store(dst_ptrs, val, mask=curr_mask)


# Specialized kernel for D = 1
# 优化点：
# 1. 同样移除了循环，直接处理长度为 BLOCK_M 的向量。
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
    strideD_src,  # kept for signature consistency
    strideB_dst,
    strideH_dst,
    strideD_dst,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # 1. 向量化处理 M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # 2. 加载索引
    m_mask = tl.load(mask_ptr + offs_m, mask=mask_m, other=0)
    src_idx = tl.load(src_idx_ptr + offs_m, mask=mask_m, other=-1)
    dst_idx = tl.load(dst_idx_ptr + offs_m, mask=mask_m, other=-1)

    # 3. 校验
    p_valid = (
        (mask_m)
        & (m_mask == 1)
        & (src_idx >= 0)
        & (src_idx < B)
        & (dst_idx >= 0)
        & (dst_idx < B)
    )

    # 4. 计算指针 (D=1, 所以不需要 strideD)
    src_offsets = src_idx * strideB_src + pid_h * strideH_src
    dst_offsets = dst_idx * strideB_dst + pid_h * strideH_dst

    # 5. 读写
    val = tl.load(src_ptr + src_offsets, mask=p_valid, other=0.0)
    tl.store(dst_ptr + dst_offsets, val, mask=p_valid)


def copy_batch_indexed_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    mask: torch.Tensor,
    block_d=128,
    num_warps=None,
):
    assert src.is_cuda and dst.is_cuda, "张量必须在 CUDA 设备上"
    assert src.dtype == dst.dtype, "源和目标张量 dtype 必须一致"
    assert src.dim() == 3 and dst.dim() == 3, "张量必须是 3D (B, H, D)"
    assert src.is_contiguous() and dst.is_contiguous(), "张量必须是连续的"

    B, H, D = src.shape
    M = src_idx.numel()

    # 调整了 BLOCK_M 的默认值。
    # 在 2D Tiling 模式下，BLOCK_M * BLOCK_D 决定了寄存器压力。
    # 32 是一个比较平衡的值，既能保证 parallelism，又不会导致寄存器溢出。
    BLOCK_M = 32

    # num_warps heuristic
    if num_warps is None:
        if D >= 512:
            num_warps = 4  # 降低 warp 数，避免小 block_m 下的资源浪费
        elif D >= 128:
            num_warps = 4
        else:
            num_warps = 2

    if D == 1:
        grid = (triton.cdiv(M, BLOCK_M), H)
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
            num_warps=num_warps,
        )
        return

    # D > 1 fused path
    # 动态调整 BLOCK_D，确保 tile 形状合理
    BLOCK_D = 128 if D >= 128 else 64
    if D >= 512:
        BLOCK_D = 256

    # 确保 BLOCK_D 不超过 D 的下一个 2 的幂次太多，虽然 triton handle mask，但太大会浪费
    BLOCK_D = min(BLOCK_D, triton.next_power_of_2(D))

    grid = (triton.cdiv(M, BLOCK_M), H)

    copy_batch_indexed_kernel_fused[grid](
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
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )


# =========================
# 正确性检查 & 扩展测试套件
# =========================
def check_correctness_batch_indexed():
    """检查批量索引复制的正确性（与 PyTorch 原生 index_copy_ 对比）"""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # 测试配置：批量数 B=32，头数 H=128，维度 D=576（大模型常见配置）
    B, H, D = 32, 128, 576
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst = torch.zeros_like(src)

    M_max = 256  # 最大请求数（索引缓冲容量）
    # 初始化索引和掩码（-1 表示无效索引）
    src_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    dst_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    mask = torch.zeros((M_max,), device=device, dtype=torch.int32)

    # 定义有效复制对：(源批量ID, 目标批量ID)
    active_pairs = [(1, 2), (3, 4), (5, 7), (9, 0)]  # 示例：4个有效复制
    for i, (s, d) in enumerate(active_pairs):
        src_idx_long[i] = s
        dst_idx_long[i] = d
        mask[i] = 1

    # PyTorch 原生实现（基线）
    dst_baseline = torch.zeros_like(dst)
    active_mask = mask.bool()
    src_active = src_idx_long[active_mask]  # 有效源批量ID
    dst_active = dst_idx_long[active_mask]  # 有效目标批量ID
    dst_baseline.index_copy_(0, dst_active, src.index_select(0, src_active))

    # Triton 实现（需将索引转为 int32 适配内核）
    src_idx_i32 = src_idx_long.to(torch.int32)
    dst_idx_i32 = dst_idx_long.to(torch.int32)
    dst_triton = torch.zeros_like(dst)
    copy_batch_indexed_triton(src, dst_triton, src_idx_i32, dst_idx_i32, mask)

    # 正确性验证（最大绝对误差）
    diff = (dst_triton - dst_baseline).abs().max().item()
    print(f"[正确性检查][批量索引复制] 最大绝对误差: {diff:.6f}")
    assert diff < 1e-5, f"批量索引复制结果不匹配！误差: {diff}"
    print("[正确性检查] 批量索引复制通过 ✅\n")


# =========================
# 扩展测试套件（多配置、多场景）
# =========================
# 全局常量（适配多场景测试）
B_MAX = 256  # 最大批量数（覆盖多数场景）
M_MAX = 256  # 索引缓冲最大容量（最大请求数）


def build_triton_graph_for_config(H, D, dtype=torch.float16, device="cuda"):
    """为特定 (H, D) 配置捕获 Triton CUDA Graph（一次捕获，多次重放）"""
    # 初始化静态张量（固定形状 [B_MAX, H, D]）
    static_src = torch.zeros(B_MAX, H, D, device=device, dtype=dtype)
    static_dst = torch.zeros_like(static_src)
    static_src_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_dst_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_mask = torch.zeros((M_MAX,), device=device, dtype=torch.int32)

    # 预热编译内核
    copy_batch_indexed_triton(
        static_src, static_dst, static_src_idx, static_dst_idx, static_mask
    )

    # 捕获 CUDA Graph
    g_triton = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g_triton):
        copy_batch_indexed_triton(
            static_src, static_dst, static_src_idx, static_dst_idx, static_mask
        )

    return {
        "graph": g_triton,
        "src": static_src,
        "dst": static_dst,
        "src_idx": static_src_idx,
        "dst_idx": static_dst_idx,
        "mask": static_mask,
        "H": H,
        "D": D,
    }


def prepare_case_into_triton_config(static_config, B, k_requests, mode):
    """将具体测试案例写入静态缓冲（适配 Triton Graph）"""
    H, D = static_config["H"], static_config["D"]

    # 清空缓冲
    static_config["dst"].zero_()
    static_config["src"].zero_()
    static_config["src_idx"].fill_(-1)
    static_config["dst_idx"].fill_(-1)
    static_config["mask"].zero_()

    # 生成测试数据（仅填充前 B 个有效批量）
    src_case = torch.randn(
        B, H, D, device=static_config["src"].device, dtype=static_config["src"].dtype
    )
    static_config["src"][:B].copy_(src_case)

    # 生成批量映射对（连续/非连续两种模式）
    if mode == "contig":
        # 连续模式：源批量连续，目标批量连续
        start_src = 0
        start_dst = B // 2
        src_list = list(range(start_src, start_src + k_requests))
        dst_list = [(start_dst + i) % B for i in range(k_requests)]
    else:  # noncontig
        # 非连续模式：随机选择批量（离散映射）
        src_list = random.sample(range(B), k_requests)
        dst_list = random.sample(range(B), k_requests)

    # 写入索引和掩码（激活前 k_requests 个请求）
    for i, (s, d) in enumerate(zip(src_list, dst_list)):
        static_config["src_idx"][i] = s
        static_config["dst_idx"][i] = d
        static_config["mask"][i] = 1

    return src_case, src_list, dst_list


def build_torch_graph_for_case(src_case, pairs, device="cuda"):
    """为具体案例捕获 PyTorch 原生实现的 CUDA Graph"""
    g_torch = torch.cuda.CUDAGraph()
    static_src = src_case.clone()
    static_dst2 = torch.zeros_like(src_case)

    # 转换为 long 类型（PyTorch index_copy_ 要求）
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


def run_case_on_config(static_config, B, k_requests, mode, iters=200):
    """在特定 (H,D) 配置上运行测试案例，返回性能和正确性结果"""
    # 准备测试数据
    src_case, src_list, dst_list = prepare_case_into_triton_config(
        static_config, B, k_requests, mode
    )
    pairs = list(zip(src_list, dst_list))

    # 捕获 PyTorch 基线 Graph
    g_torch, static_dst2 = build_torch_graph_for_case(src_case, pairs)

    # 性能测试
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    # Triton 性能（重放预捕获的 Graph）
    e_start.record()
    for _ in range(iters):
        static_config["graph"].replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    # PyTorch 性能（重放案例专属 Graph）
    e_start.record()
    for _ in range(iters):
        g_torch.replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    # 正确性验证（仅对比有效批量范围）
    triton_result = static_config["dst"][:B]
    diff = (triton_result - static_dst2).abs().max().item()

    return triton_ms, torch_ms, diff


def benchmark_suite_per_config_graphs():
    """扩展测试套件：多 (H,D) 配置、多请求数、连续/非连续模式"""
    device = "cuda"
    dtype = torch.float16
    random.seed(0)

    # 测试配置组合（覆盖不同模型尺寸）
    test_configs = [
        (128, 512),
        (128, 576),
        (64, 128),
        (128, 1),
        (64, 1),
    ]
    request_counts = [2, 8, 32, 64, 128, 256]  # 不同请求数（稀疏程度）
    modes = ["contig", "noncontig"]  # 连续/非连续映射模式

    # 预捕获所有 (H,D) 配置的 Triton Graph（一次捕获，多次使用）
    triton_graphs = {}
    print(f"[测试套件] 预捕获 {len(test_configs)} 个 (H,D) 配置的 Triton Graph...")
    for H, D in test_configs:
        triton_graphs[(H, D)] = build_triton_graph_for_config(H, D, dtype, device)
    print(f"[测试套件] Triton Graph 预捕获完成 ✅\n")

    # 运行所有测试案例
    results = []
    print(
        f"[测试套件] 开始运行 {len(test_configs) * len(request_counts) * len(modes)} 个测试案例..."
    )
    for H, D in test_configs:
        static_cfg = triton_graphs[(H, D)]
        B = B_MAX  # 固定最大批量数（256）
        for k in request_counts:
            for mode in modes:
                triton_ms, torch_ms, diff = run_case_on_config(
                    static_cfg, B=B, k_requests=k, mode=mode, iters=200
                )
                speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
                is_correct = diff < 1e-5
                results.append(
                    {
                        "H": H,
                        "D": D,
                        "k": k,
                        "mode": mode,
                        "torch_ms": torch_ms,
                        "triton_ms": triton_ms,
                        "speedup": speedup,
                        "correct": is_correct,
                    }
                )

                # 实时打印结果
                print(
                    f"[案例结果] H={H:3d}, D={D:4d}, k={k:3d}, mode={mode:8s} | "
                    f"Triton: {triton_ms:.4f}ms | PyTorch: {torch_ms:.4f}ms | "
                    f"提速: {speedup:.2f}x | 正确: {is_correct}"
                )

    # 打印汇总表
    print("\n" + "=" * 120)
    print(f"{'测试汇总表':^120}")
    print("=" * 120)
    print(
        f"{'H':>5} {'D':>6} {'k':>6} {'模式':>8} {'PyTorch(ms)':>12} {'Triton(ms)':>12} {'提速倍数':>10} {'正确性':>8}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"{r['H']:>5} {r['D']:>6} {r['k']:>6} {r['mode']:>8} "
            f"{r['torch_ms']:>12.4f} {r['triton_ms']:>12.4f} {r['speedup']:>10.2f} {str(r['correct']):>8}"
        )
    print("=" * 120)


# =========================
# 主函数（执行所有测试）
# =========================
def main():
    print("=" * 80)
    print(f"{'批量索引复制（离散复制）测试程序':^80}")
    print("=" * 80)

    # 1. 正确性检查
    print("[1/2] 执行正确性检查...")
    check_correctness_batch_indexed()

    # 2. 扩展测试套件（多配置、多场景）
    print("[2/2] 执行扩展测试套件...")
    benchmark_suite_per_config_graphs()

    print("\n" + "=" * 80)
    print(f"{'所有测试完成！':^80}")
    print("=" * 80)


if __name__ == "__main__":
    main()

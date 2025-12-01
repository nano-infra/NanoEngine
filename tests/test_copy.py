import math
import random
import time

import torch
import triton
import triton.language as tl


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
    m = tl.load(mask_ptr + pid_m)
    if m == 0:
        return

    b_src = tl.load(src_idx_ptr + pid_m)
    b_dst = tl.load(dst_idx_ptr + pid_m)

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
    离散批量映射复制：仅复制 mask[i]==1 的位置
    src, dst: [B, H, D] 3D 张量（批量×头×维度）
    src_idx, dst_idx, mask: shape [M_max]，mask 取值 {0,1}
    功能：dst[dst_idx[i], :, :] = src[src_idx[i], :, :]（仅当 mask[i]==1 时生效）
    """
    assert src.is_cuda and dst.is_cuda, "张量必须在 CUDA 设备上"
    assert src.dtype == dst.dtype, "源和目标张量 dtype 必须一致"
    assert src.dim() == 3 and dst.dim() == 3, "张量必须是 3D (B, H, D)"
    assert src.is_contiguous(
        memory_format=torch.contiguous_format
    ) and dst.is_contiguous(memory_format=torch.contiguous_format), "张量必须是连续的"

    B, H, D = src.shape
    M = src_idx.numel()
    assert dst_idx.numel() == M and mask.numel() == M, "索引和掩码长度必须一致"

    # 自动选择线程束数量（根据 D 维度大小 heuristic）
    if num_warps is None:
        if D >= 512:
            num_warps = 8
        elif D >= 128:
            num_warps = 4
        else:
            num_warps = 2

    # 定义网格维度：(请求数, 头数, 维度块数)
    grid = (M, H, triton.cdiv(D, block_d))

    # 启动 Triton 内核
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
        (128, 512),  # 大模型常见：128头×512维
        (128, 576),  # 大模型常见：128头×576维
        (64, 128),  # 中小模型：64头×128维
        (32, 256),  # 轻量化模型：32头×256维
        (256, 1024),  # 超大型模型：256头×1024维
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

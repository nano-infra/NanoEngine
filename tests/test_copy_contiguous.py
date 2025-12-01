import random

import torch
import triton
import triton.language as tl
from nanodeploy.kernels.copy import copy_batch_indexed_triton


# =========================
# 硬件辅助函数
# =========================
def get_gpu_peak_bw():
    """
    获取当前 GPU 的理论峰值显存带宽 (GB/s)。
    """
    if not torch.cuda.is_available():
        return 1.0

    name = torch.cuda.get_device_name(0)

    # 常见数据中心 GPU 带宽 (GB/s)
    bw_map = {
        "H200": 4800,
        "H100": 3352,
        "A100": 2039,
        "A800": 2039,
        "V100": 900,
        "3090": 936,
        "4090": 1008,
        "A10": 600,
        "T4": 320,
        "L40": 864,
    }

    # 模糊匹配
    for k, v in bw_map.items():
        if k in name:
            print(f"[硬件检测] 识别到 {k} (设备名: {name})，理论峰值带宽: {v} GB/s")
            return float(v)

    # 默认回退
    print(f"[硬件检测] 未识别具体型号 ({name})，默认使用 4800 GB/s 进行计算。")
    return 4800.0


# =========================
# 正确性检查 - 连续拷贝
# =========================
def check_correctness_contiguous_copy():
    """检查连续tensor拷贝的正确性"""
    if not torch.cuda.is_available():
        print("[正确性检查] CUDA不可用，跳过测试")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # 配置：B=32, H=128, D=576
    B, H, D = 32, 128, 576
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst = torch.zeros_like(src)

    # 测试连续拷贝一整块tensor
    M = B  # 拷贝所有行
    src_idx_long = torch.arange(B, device=device, dtype=torch.long)
    dst_idx_long = torch.arange(B, device=device, dtype=torch.long)
    mask = torch.ones((M,), device=device, dtype=torch.int32)

    # PyTorch 基线 - 使用torch.clone()作为连续拷贝的参考
    dst_baseline = torch.clone(src)

    # Triton 实现 - 拷贝所有行
    src_idx_i32 = src_idx_long.to(torch.int32)
    dst_idx_i32 = dst_idx_long.to(torch.int32)
    dst_triton = torch.zeros_like(src)
    copy_batch_indexed_triton(src, dst_triton, src_idx_i32, dst_idx_i32, mask)

    # 验证
    diff = (dst_triton - dst_baseline).abs().max().item()
    print(f"[连续拷贝正确性检查] 最大绝对误差: {diff:.6f}")
    if diff < 1e-4:
        print("[连续拷贝正确性检查] 通过 ✅\n")
    else:
        print("[连续拷贝正确性检查] 失败 ❌ (请检查Kernel实现)\n")


# =========================
# 性能测试 - 连续拷贝
# =========================
def build_contig_triton_graph_for_config(B, H, D, dtype=torch.float16, device="cuda"):
    """预捕获连续拷贝的CUDA Graph"""
    static_src = torch.zeros(B, H, D, device=device, dtype=dtype)
    static_dst = torch.zeros_like(static_src)

    # 连续索引：拷贝整个tensor
    M = B
    static_src_idx = torch.arange(B, device=device, dtype=torch.int32)
    static_dst_idx = torch.arange(B, device=device, dtype=torch.int32)
    static_mask = torch.ones((M,), device=device, dtype=torch.int32)

    # Warmup
    copy_batch_indexed_triton(
        static_src, static_dst, static_src_idx, static_dst_idx, static_mask
    )
    torch.cuda.synchronize()

    # 捕获CUDA Graph
    g_triton = torch.cuda.CUDAGraph()
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
        "B": B,
        "H": H,
        "D": D,
    }


def build_contig_torch_graph_for_config(B, H, D, dtype=torch.float16, device="cuda"):
    """预捕获PyTorch连续拷贝的CUDA Graph"""
    static_src = torch.zeros(B, H, D, device=device, dtype=dtype)
    static_dst = torch.zeros_like(static_src)

    # 捕获PyTorch的clone操作作为连续拷贝的基线
    g_torch = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g_torch):
        static_dst.copy_(static_src)

    return {
        "graph": g_torch,
        "src": static_src,
        "dst": static_dst,
        "B": B,
        "H": H,
        "D": D,
    }


def run_contig_benchmark(triton_config, torch_config, iters=100):
    """运行连续拷贝的性能测试"""
    device = triton_config["src"].device

    # --- 测试 Triton Kernel ---
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    e_start.record()
    for _ in range(iters):
        triton_config["graph"].replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    # --- 测试 PyTorch copy_ ---
    e_start.record()
    for _ in range(iters):
        torch_config["graph"].replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    return triton_ms, torch_ms


def benchmark_contiguous_copy():
    """主测试函数：测试连续拷贝的性能"""
    if not torch.cuda.is_available():
        print("[性能测试] CUDA不可用，跳过测试")
        return

    device = "cuda"
    dtype = torch.float16
    random.seed(0)

    # 1. 获取硬件峰值带宽
    PEAK_BW_GBPS = get_gpu_peak_bw()

    # 2. 测试配置 - 覆盖不同大小的tensor
    test_configs = [
        # (B, H, D, 描述)
        (8, 128, 576, "ds3 small attn"),
        (128, 128, 576, "ds3 large attn"),
        (8, 128, 512, "ds3 small q"),
        (128, 128, 512, "ds3 large q"),
        (8, 128, 1, "ds3 small lse"),
        (128, 128, 1, "ds3 large lse"),
        (8, 32, 128, "qwen3 small q"),
        (128, 32, 128, "qwen3 large q"),
        (8, 32, 1, "qwen3 small lse"),
        (128, 32, 1, "qwen3 large lse"),
    ]

    # 添加D=1的特殊情况
    test_configs.append((32, 128, 1, "D=1特殊情况"))

    print(f"[性能测试] 准备测试 {len(test_configs)} 种配置...")

    results = []

    # 3. 运行测试
    for B, H, D, desc in test_configs:
        print(f"\n[测试配置] B={B}, H={H}, D={D} ({desc})")

        # 预编译CUDA Graphs
        try:
            triton_config = build_contig_triton_graph_for_config(B, H, D, dtype, device)
            torch_config = build_contig_torch_graph_for_config(B, H, D, dtype, device)
        except Exception as e:
            print(f"  跳过配置: 构建CUDA Graph失败 - {e}")
            continue

        # 运行性能测试
        iters = 100 if B * H * D < 1000000 else 50  # 根据数据量调整迭代次数
        triton_ms, torch_ms = run_contig_benchmark(triton_config, torch_config, iters)

        # 计算带宽和利用率
        # 数据量 = B * H * D * 2字节(fp16) * 2(读+写)
        total_bytes = B * H * D * 2 * 2

        if triton_ms > 0:
            triton_gbps = (total_bytes / 1e9) / (triton_ms / 1000.0)
        else:
            triton_gbps = 0

        if torch_ms > 0:
            torch_gbps = (total_bytes / 1e9) / (torch_ms / 1000.0)
        else:
            torch_gbps = 0

        triton_util = (triton_gbps / PEAK_BW_GBPS) * 100 if PEAK_BW_GBPS > 0 else 0
        torch_util = (torch_gbps / PEAK_BW_GBPS) * 100 if PEAK_BW_GBPS > 0 else 0

        speedup = torch_ms / triton_ms if triton_ms > 0 else 0

        results.append(
            {
                "B": B,
                "H": H,
                "D": D,
                "desc": desc,
                "triton_ms": triton_ms,
                "torch_ms": torch_ms,
                "triton_gbps": triton_gbps,
                "torch_gbps": torch_gbps,
                "triton_util": triton_util,
                "torch_util": torch_util,
                "speedup": speedup,
            }
        )

        print(
            f"  Triton: {triton_ms:.3f}ms ({triton_gbps:.1f} GB/s, {triton_util:.1f}%)"
        )
        print(f"  Torch:  {torch_ms:.3f}ms ({torch_gbps:.1f} GB/s, {torch_util:.1f}%)")
        print(f"  Speedup: {speedup:.2f}x")

    # 4. 汇总输出
    print("\n" + "=" * 140)
    print(f"{'连续拷贝性能测试报告':^140}")
    print("=" * 140)
    print(f"GPU: {torch.cuda.get_device_name(0)} | 理论峰值带宽: {PEAK_BW_GBPS} GB/s")
    print("=" * 140)
    header = (
        f"{'B':>4} {'H':>4} {'D':>5} {'描述':<12} "
        f"{'Triton(ms)':>12} {'Torch(ms)':>12} {'Speedup':>10} "
        f"{'Tri_GB/s':>10} {'Tor_GB/s':>10} {'Tri_Util%':>10} {'Tor_Util%':>10}"
    )
    print(header)
    print("-" * 140)

    for r in results:
        print(
            f"{r['B']:>4} {r['H']:>4} {r['D']:>5} {r['desc']:<12} "
            f"{r['triton_ms']:>12.3f} {r['torch_ms']:>12.3f} {r['speedup']:>10.2f} "
            f"{r['triton_gbps']:>10.1f} {r['torch_gbps']:>10.1f} "
            f"{r['triton_util']:>10.1f} {r['torch_util']:>10.1f}"
        )

    # 计算平均加速比
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 140)
    print(f"{'平均加速比':<40} {avg_speedup:>10.2f}x")
    print("=" * 140)


# =========================
# 额外的测试：与index_copy_对比
# =========================
def benchmark_vs_index_copy():
    """对比我们的kernel与PyTorch index_copy_在连续拷贝场景下的性能"""
    if not torch.cuda.is_available():
        return

    device = "cuda"
    dtype = torch.float16

    print("\n" + "=" * 80)
    print("补充测试: 与PyTorch index_copy_的对比")
    print("=" * 80)

    # 测试配置
    B, H, D = 32, 128, 576

    # 准备数据
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst1 = torch.zeros_like(src)
    dst2 = torch.zeros_like(src)

    # 连续索引
    src_idx = torch.arange(B, device=device, dtype=torch.int32)
    dst_idx = torch.arange(B, device=device, dtype=torch.int32)
    mask = torch.ones((B,), device=device, dtype=torch.int32)

    # 构建CUDA Graphs
    # Triton kernel
    g_triton = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_triton):
        copy_batch_indexed_triton(src, dst1, src_idx, dst_idx, mask)

    # PyTorch index_copy_
    g_index_copy = torch.cuda.CUDAGraph()
    src_idx_long = src_idx.to(torch.long)
    dst_idx_long = dst_idx.to(torch.long)
    with torch.cuda.graph(g_index_copy):
        dst2.index_copy_(0, dst_idx_long, src.index_select(0, src_idx_long))

    # 性能测试
    iters = 200
    torch.cuda.synchronize()

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        g_triton.replay()
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / iters

    # index_copy_
    start.record()
    for _ in range(iters):
        g_index_copy.replay()
    end.record()
    torch.cuda.synchronize()
    index_copy_ms = start.elapsed_time(end) / iters

    # 验证结果
    torch.cuda.synchronize()
    g_triton.replay()
    g_index_copy.replay()
    diff = (dst1 - dst2).abs().max().item()

    print(f"配置: B={B}, H={H}, D={D}")
    print(f"Triton kernel: {triton_ms:.3f}ms")
    print(f"PyTorch index_copy_: {index_copy_ms:.3f}ms")
    print(f"Speedup: {index_copy_ms / triton_ms:.2f}x")
    print(f"结果一致性检查: 最大误差 = {diff:.6f}")
    if diff < 1e-4:
        print("一致性检查: 通过 ✅")
    else:
        print("一致性检查: 失败 ❌")
    print("=" * 80)


def main():
    """主函数"""
    print("开始连续拷贝性能测试...")

    # 1. 正确性检查
    check_correctness_contiguous_copy()

    # 2. 主要性能测试
    benchmark_contiguous_copy()

    # 3. 补充测试：与index_copy_对比
    benchmark_vs_index_copy()


if __name__ == "__main__":
    main()

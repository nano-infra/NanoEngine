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
    针对 H200 进行了适配。
    """
    if not torch.cuda.is_available():
        return 1.0

    name = torch.cuda.get_device_name(0)

    # 常见数据中心 GPU 带宽 (GB/s)
    bw_map = {
        "H200": 4800,  # H200 SXM (HBM3e)
        "H100": 3352,  # H100 SXM
        "A100": 2039,  # A100 SXM4 80GB
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
    print(
        f"[硬件检测] 未识别具体型号 ({name})，默认使用 4800 GB/s (H200基准) 进行计算。"
    )
    return 4800.0


# =========================
# 正确性检查
# =========================
def check_correctness_batch_indexed():
    """检查批量索引复制的正确性"""
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # 配置：B=32, H=128, D=576
    B, H, D = 32, 128, 576
    src = torch.randn(B, H, D, device=device, dtype=dtype)
    dst = torch.zeros_like(src)

    M_max = 256
    src_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    dst_idx_long = torch.full((M_max,), -1, device=device, dtype=torch.long)
    mask = torch.zeros((M_max,), device=device, dtype=torch.int32)

    active_pairs = [(1, 2), (3, 4), (5, 7), (9, 0)]
    for i, (s, d) in enumerate(active_pairs):
        src_idx_long[i] = s
        dst_idx_long[i] = d
        mask[i] = 1

    # PyTorch 基线
    dst_baseline = torch.zeros_like(dst)
    active_mask = mask.bool()
    src_active = src_idx_long[active_mask]
    dst_active = dst_idx_long[active_mask]
    dst_baseline.index_copy_(0, dst_active, src.index_select(0, src_active))

    # Triton 实现
    src_idx_i32 = src_idx_long.to(torch.int32)
    dst_idx_i32 = dst_idx_long.to(torch.int32)
    dst_triton = torch.zeros_like(dst)
    copy_batch_indexed_triton(src, dst_triton, src_idx_i32, dst_idx_i32, mask)

    # 验证
    diff = (dst_triton - dst_baseline).abs().max().item()
    print(f"[正确性检查] 最大绝对误差: {diff:.6f}")
    if diff < 1e-4:
        print("[正确性检查] 通过 ✅\n")
    else:
        print("[正确性检查] 失败 ❌ (请检查Kernel实现)\n")


# =========================
# 性能测试套件
# =========================
# 全局常量
B_MAX = 256
M_MAX = 256


def build_triton_graph_for_config(H, D, dtype=torch.float16, device="cuda"):
    """预捕获 Triton Kernel 的 CUDA Graph"""
    static_src = torch.zeros(B_MAX, H, D, device=device, dtype=dtype)
    static_dst = torch.zeros_like(static_src)
    static_src_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_dst_idx = torch.full((M_MAX,), -1, device=device, dtype=torch.int32)
    static_mask = torch.zeros((M_MAX,), device=device, dtype=torch.int32)

    # Warmup
    copy_batch_indexed_triton(
        static_src, static_dst, static_src_idx, static_dst_idx, static_mask
    )

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
    H, D = static_config["H"], static_config["D"]

    # Reset buffers
    static_config["dst"].zero_()
    # 为了性能测试准确性，src 不一定要每次 random，但为了模拟真实数据分布可以保留
    # 这里不重置 src 为 0，避免影响性能（copy 0 和 copy rand 速度一样）
    static_config["src_idx"].fill_(-1)
    static_config["dst_idx"].fill_(-1)
    static_config["mask"].zero_()

    # 只需要生成前 B 个作为数据池
    # 注意：实际 copy 性能和数值无关，只和 index 有关
    src_case = static_config["src"][:B]

    if mode == "contig":
        start_src = 0
        start_dst = B // 2
        src_list = list(range(start_src, start_src + k_requests))
        dst_list = [(start_dst + i) % B for i in range(k_requests)]
    else:
        src_list = random.sample(range(B), k_requests)
        dst_list = random.sample(range(B), k_requests)

    for i, (s, d) in enumerate(zip(src_list, dst_list)):
        static_config["src_idx"][i] = s
        static_config["dst_idx"][i] = d
        static_config["mask"][i] = 1

    return src_case, src_list, dst_list


def build_torch_graph_for_case(src_case, pairs, device="cuda"):
    g_torch = torch.cuda.CUDAGraph()
    static_src = src_case.clone()  # Clone 出一个新的 buffer 避免污染
    static_dst2 = torch.zeros_like(src_case)

    src_idx_long = torch.tensor(
        [s for (s, _) in pairs], device=device, dtype=torch.long
    )
    dst_idx_long = torch.tensor(
        [d for (_, d) in pairs], device=device, dtype=torch.long
    )

    torch.cuda.synchronize()
    with torch.cuda.graph(g_torch):
        # PyTorch 基线逻辑
        static_dst2.index_copy_(
            0, dst_idx_long, static_src.index_select(0, src_idx_long)
        )

    return g_torch, static_dst2


def run_case_on_config(static_config, B, k_requests, mode, iters=100):
    src_case, src_list, dst_list = prepare_case_into_triton_config(
        static_config, B, k_requests, mode
    )
    pairs = list(zip(src_list, dst_list))

    # PyTorch 基线图（每次都要重新录制，因为 indices 变了）
    g_torch, static_dst2 = build_torch_graph_for_case(src_case, pairs)

    # --- 测试 Triton ---
    torch.cuda.synchronize()
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    e_start.record()
    for _ in range(iters):
        static_config["graph"].replay()
    e_end.record()
    torch.cuda.synchronize()
    triton_ms = e_start.elapsed_time(e_end) / iters

    # --- 测试 PyTorch ---
    e_start.record()
    for _ in range(iters):
        g_torch.replay()
    e_end.record()
    torch.cuda.synchronize()
    torch_ms = e_start.elapsed_time(e_end) / iters

    return triton_ms, torch_ms


def benchmark_suite_per_config_graphs():
    device = "cuda"
    dtype = torch.float16
    random.seed(0)

    # 1. 获取硬件峰值带宽 (H200 -> 4800 GB/s)
    PEAK_BW_GBPS = get_gpu_peak_bw()

    # 2. 测试配置
    test_configs = [
        (128, 512),
        (128, 576),  # 常见
        (64, 128),
        (128, 1),  # 极端小
    ]
    request_counts = [32, 64, 128, 256]  # 不同的稀疏度
    modes = ["contig", "noncontig"]

    # 3. 预热 Triton Graphs
    triton_graphs = {}
    print(f"[测试准备] 预编译 {len(test_configs)} 组 Triton CUDA Graphs...")
    for H, D in test_configs:
        triton_graphs[(H, D)] = build_triton_graph_for_config(H, D, dtype, device)
    print(f"[测试准备] 完成。\n")

    results = []

    # 4. 运行 Loop
    for H, D in test_configs:
        static_cfg = triton_graphs[(H, D)]
        B = B_MAX
        for k in request_counts:
            for mode in modes:
                triton_ms, torch_ms = run_case_on_config(
                    static_cfg, B=B, k_requests=k, mode=mode, iters=100
                )

                # --- 带宽计算核心逻辑 ---
                # 数据量 (Bytes) = k个请求 * H * D * 2字节(fp16) * 2(读+写)
                # 注意：Memory Bound Kernel 通常只计算 Payload，忽略 Indices 读取开销
                total_bytes = k * H * D * 2 * 2

                # 吞吐量 (GB/s) = (Bytes / 1e9) / (Seconds)
                if triton_ms > 0:
                    triton_gbps = (total_bytes / 1e9) / (triton_ms / 1000.0)
                else:
                    triton_gbps = 0

                # 利用率 (%)
                utilization = (triton_gbps / PEAK_BW_GBPS) * 100

                speedup = torch_ms / triton_ms if triton_ms > 0 else 0

                results.append(
                    {
                        "H": H,
                        "D": D,
                        "k": k,
                        "mode": mode,
                        "triton_ms": triton_ms,
                        "torch_ms": torch_ms,
                        "speedup": speedup,
                        "gbps": triton_gbps,
                        "util": utilization,
                    }
                )

                print(
                    f"H={H:3d} D={D:4d} k={k:3d} {mode:9s} | "
                    f"Tri: {triton_ms:.3f}ms | "
                    f"BW: {triton_gbps:6.1f} GB/s ({utilization:4.1f}%) | "
                    f"Speedup: {speedup:.2f}x"
                )

    # 5. 汇总输出
    print("\n" + "=" * 120)
    print(f"{f'H200 Performance Report (Peak: {PEAK_BW_GBPS} GB/s)':^120}")
    print("=" * 120)
    header = (
        f"{'H':>5} {'D':>6} {'k':>6} {'Mode':>9} "
        f"{'Time(ms)':>10} {'GB/s':>10} {'Util(%)':>10} {'Speedup':>10}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        print(
            f"{r['H']:>5} {r['D']:>6} {r['k']:>6} {r['mode']:>9} "
            f"{r['triton_ms']:>10.3f} {r['gbps']:>10.1f} {r['util']:>10.1f} {r['speedup']:>10.2f}"
        )
    print("=" * 120)


def main():
    check_correctness_batch_indexed()
    benchmark_suite_per_config_graphs()


if __name__ == "__main__":
    main()

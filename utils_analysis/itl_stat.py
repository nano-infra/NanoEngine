import json
import numpy as np
import argparse
import sys

def print_stats(title, data_array):
    """
    打印通用的统计指标 (均值, P50, P99 等)
    """
    if len(data_array) == 0:
        print(f"\n[{title}] 无数据")
        return

    # 计算指标
    count = len(data_array)
    mean_val = np.mean(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    std_dev = np.std(data_array)
    
    # 百分位数
    p50 = np.percentile(data_array, 50)
    p90 = np.percentile(data_array, 90)
    p95 = np.percentile(data_array, 95)
    p99 = np.percentile(data_array, 99)

    print("=" * 60)
    print(f"统计对象: {title}")
    print(f"样本总量: {count}")
    print("-" * 60)
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 60)
    print(f"{'Mean (均值)':<25} | {mean_val:.4f}")
    print(f"{'Std (标准差)':<25} | {std_dev:.4f}")
    print(f"{'Min':<25} | {min_val:.4f}")
    print(f"{'Max':<25} | {max_val:.4f}")
    print("-" * 60)
    print(f"{'P50 (中位数)':<25} | {p50:.4f}")
    print(f"{'P90':<25} | {p90:.4f}")
    print(f"{'P95':<25} | {p95:.4f}")
    print(f"{'P99':<25} | {p99:.4f}")
    print("=" * 60)
    print() 

def analyze_tail_distribution(data_array, threshold, step_size):
    """
    分析超过阈值的长尾分布情况
    """
    total_count = len(data_array)
    if total_count == 0:
        return

    # 筛选出大于等于阈值的数据
    outliers = data_array[data_array >= threshold]
    outlier_count = len(outliers)
    
    print("=" * 60)
    print(f"长尾分布分析 (Threshold >= {threshold} ms)")
    print("-" * 60)
    print(f"总样本数: {total_count}")
    print(f"超阈值样本数: {outlier_count}")
    print(f"超阈值占比: {outlier_count / total_count * 100:.2f}%")
    print("-" * 60)
    
    if outlier_count == 0:
        print("没有发现超过阈值的记录。")
        print("=" * 60)
        print()
        return

    print(f"{'Range (ms)':<25} | {'Count':<10} | {'% (of outliers)':<15}")
    print("-" * 60)

    # 确定分布区间的上限
    max_val = np.max(outliers)
    
    # 从阈值开始，按步长遍历，直到覆盖最大值
    current_start = threshold
    while current_start <= max_val:
        current_end = current_start + step_size
        
        # 统计落在当前区间 [start, end) 的数量
        if current_start + step_size > max_val:
             # 最后一个区间包含所有剩余的
            mask = (outliers >= current_start)
            label = f"[{current_start}, ∞)"
        else:
            mask = (outliers >= current_start) & (outliers < current_end)
            label = f"[{current_start}, {current_end})"
            
        count_in_bin = np.sum(mask)
        
        if count_in_bin > 0:
            percentage = (count_in_bin / outlier_count) * 100
            print(f"{label:<25} | {count_in_bin:<10} | {percentage:6.2f}%")
        
        current_start += step_size

    print("=" * 60)
    print()

def analyze_itl(file_path, threshold, step):
    all_raw_samples = []    # 存储每一个单独的 token itl
    sequence_means = []     # 存储每一行(每一个序列)的 itl 均值
    normalized_latencies = [] # 存储每一行的归一化延迟
    
    print(f"正在读取并处理文件: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples = data.get('itl_samples', [])
                    queue_time = data.get('queueing_time_ms', 0.0)
                    output_len = data.get('output_len', 0)
                    
                    if isinstance(samples, list) and len(samples) > 0:
                        # 1. 收集原始数据
                        all_raw_samples.extend(samples)
                        
                        # 2. 计算序列 ITL 均值
                        seq_mean = np.mean(samples)
                        sequence_means.append(seq_mean)

                        # 3. 计算归一化延迟 (Normalized Latency)
                        # 公式: (Sum(ITL) + QueueTime) / OutputLen
                        
                        # 数据清洗/兜底: 如果 output_len 为 0，尝试用 samples 长度替代
                        if output_len <= 0:
                            output_len = len(samples)
                        
                        if output_len > 0:
                            total_itl_sum = sum(samples)
                            norm_latency = (total_itl_sum + queue_time) / output_len
                            normalized_latencies.append(norm_latency)
                        
                except json.JSONDecodeError:
                    print(f"Warning: 第 {line_num} 行 JSON 解析失败，已跳过。")
                except Exception as e:
                    print(f"Warning: 第 {line_num} 行处理出错: {e}")

    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}")
        return

    # 转换为 numpy 数组
    arr_raw = np.array(all_raw_samples)
    arr_seq_means = np.array(sequence_means)
    arr_norm_lat = np.array(normalized_latencies)

    # --- 打印报告 ---
    print("\n" + "#" * 25 + " 分析报告 " + "#" * 25)
    
    # 报告 1: 原始 Token 粒度
    print_stats("Token Level ITL (所有 Token 混合)", arr_raw)
    
    # 报告 2: 序列粒度 ITL 均值
    print_stats("Sequence Mean ITL (序列平均值分布)", arr_seq_means)

    # 报告 3: 归一化延迟 (新增)
    # 含义: (Sum ITL + Queue) / Output Tokens
    print_stats("Normalized Latency (ms/token, 含排队)", arr_norm_lat)
    
    # 报告 4: 长尾分布分析
    analyze_tail_distribution(arr_raw, threshold, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 ITL 指标 (Token粒度 & 序列粒度 & 归一化延迟 & 长尾分布)")
    
    # 默认路径
    default_path = "/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy/bench_logs/4o_issue_r0.01_w0_1M/DP32_SP1_EP32_Seg64000_R#40000_Rate80_BS128_LeastBatch_141GB_MEM9_LEN1000000/20260114_025618.json"
    
    parser.add_argument("file", nargs="?", default=default_path, help="日志文件路径")
    parser.add_argument("--threshold", type=float, default=600, help="长尾分析的起始阈值 (默认: 600ms)")
    parser.add_argument("--step", type=float, default=100.0, help="长尾分布统计的区间步长 (默认: 100ms)")
    
    args = parser.parse_args()
    
    analyze_itl(args.file, args.threshold, args.step)
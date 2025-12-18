import re
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def parse_nanodeploy_log(file_path):
    sequence_data = []
    server_data = []
    
    # 正则表达式匹配
    # 匹配 SequenceMetric
    seq_re = re.compile(r"\[(?P<time>.*?)\] .* SequenceMetric \[(?P<id>.*?)\] - TTFT: (?P<ttft>.*?)ms, E2E: (?P<e2e>.*?)ms, Prompt Length: (?P<plen>\d+), Output Length: (?P<olen>\d+), Queueing Time: (?P<qtime>.*?)ms, ITL Wo Queue: (?P<itl>.*?)ms")
    
    # 匹配 Server Metric (LLM Engine step)
    server_re = re.compile(r"\[(?P<time>.*?)\] .* nanodeploy/engine/llm_engine.py:\d+ step - (?P<dict_str>\{.*\})")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 尝试解析 Sequence Metric
            seq_match = seq_re.search(line)
            if seq_match:
                d = seq_match.groupdict()
                sequence_data.append({
                    'timestamp': datetime.strptime(d['time'], '%Y-%m-%d %H:%M:%S'),
                    'ttft': float(d['ttft']),
                    'e2e': float(d['e2e']) / 1000.0, # 转为秒
                    'itl': float(d['itl']),
                    'qtime': float(d['qtime'])
                })
                continue
            
            # 2. 尝试解析 Server Metric
            server_match = server_re.search(line)
            if server_match:
                d = server_match.groupdict()
                try:
                    stats = eval(d['dict_str'])
                    # 这里假设我们关心总的 DP Batch Size 和剩余 Block 数量
                    server_data.append({
                        'timestamp': datetime.strptime(d['time'], '%Y-%m-%d %H:%M:%S'),
                        'total_batch_size': sum(stats['dp_batch_sizes']),
                        'min_free_blocks': min([min(b) if isinstance(b, list) else b for b in stats['free_blocks']])
                    })
                except:
                    continue

    return pd.DataFrame(sequence_data), pd.DataFrame(server_data)

def plot_metrics(df_seq, df_server):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    plt.subplots_adjust(hspace=0.4)

    # 图 1: Sequence Metrics (延迟相关)
    if not df_seq.empty:
        ax1 = axes[0]
        ax1.plot(df_seq['ttft'], label='TTFT (ms)', color='blue', alpha=0.7)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Sequence Latency Metrics')
        
        ax1_e2e = ax1.twinx()
        ax1_e2e.plot(df_seq['e2e'], label='E2E (s)', color='red', linestyle='--')
        ax1_e2e.set_ylabel('E2E Time (seconds)')
        
        # 合并图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_e2e.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

    # 图 2: Server Metrics (负载相关)
    if not df_server.empty:
        ax2 = axes[1]
        ax2.plot(df_server['total_batch_size'], label='Total Batch Size', color='green')
        ax2.set_ylabel('Batch Size')
        ax2.set_title('Server Runtime Status')
        
        ax2_blocks = ax2.twinx()
        ax2_blocks.plot(df_server['min_free_blocks'], label='Min Free Blocks', color='orange', linestyle=':')
        ax2_blocks.set_ylabel('Free Memory Blocks')
        
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_blocks.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('NanoDeploy Performance Analysis', fontsize=16)
    plt.show()

# 使用示例
if __name__ == "__main__":
    LOG_FILE_PATH = "your_log_file.log" # 替换为你的日志路径
    # 模拟生成一个临时文件进行测试，或者直接指定路径
    df_seq, df_server = parse_nanodeploy_log(LOG_FILE_PATH)
    
    if df_seq.empty and df_server.empty:
        print("未解析到有效数据，请检查日志路径及格式。")
    else:
        plot_metrics(df_seq, df_server)
import re
import ast
import matplotlib.pyplot as plt
import sys
import os

def parse_and_plot_log(file_path):
    # --- 1. 检查文件是否存在 ---
    if not os.path.exists(file_path):
        print(f"Error: 文件 '{file_path}' 不存在。")
        sys.exit(1)

    # --- 2. 准备输出文件名 ---
    # 去掉扩展名，加上 .png
    base_name = os.path.splitext(file_path)[0]
    output_png = f"{base_name}.png"

    # --- 3. 读取并清洗日志 ---
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 清除 ANSI 颜色代码 (非常重要，否则无法解析字典)
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_content = ansi_escape.sub('', content)

    # --- 4. 提取数据 ---
    # 查找 "step - " 后面的字典结构
    dict_pattern = re.compile(r"step - (\{.*?\})")
    matches = dict_pattern.findall(clean_content)

    steps = []
    # 动态初始化 rank 数据：sp_history[rank_id] = [val1, val2...]
    sp_history = [] 
    free_history = []
    
    decode_step_count = 0

    print(f"正在分析 {file_path} ...")
    print(f"找到 {len(matches)} 条 step 日志，正在提取 decode 数据...")

    for dict_str in matches:
        try:
            data = ast.literal_eval(dict_str)
            
            # 仅处理 decode 阶段
            if data.get('mode') != 'decode':
                continue

            # 提取并扁平化数据
            # 原始格式 [[34], [43]...] -> 目标格式 [34, 43...]
            raw_sp = data.get('sp_batch_sizes', [])
            flat_sp = [item[0] if isinstance(item, list) else item for item in raw_sp]
            
            raw_free = data.get('free_blocks', [])
            flat_free = [item[0] if isinstance(item, list) else item for item in raw_free]

            # 第一次遇到数据时，初始化 list
            if not sp_history:
                num_ranks = len(flat_sp)
                sp_history = [[] for _ in range(num_ranks)]
                free_history = [[] for _ in range(num_ranks)]

            # 记录数据
            for i in range(len(flat_sp)):
                sp_history[i].append(flat_sp[i])
                free_history[i].append(flat_free[i])

            decode_step_count += 1
            steps.append(decode_step_count)

        except (ValueError, SyntaxError):
            continue

    if decode_step_count == 0:
        print("未找到 decode 阶段的数据，不生成图片。")
        return

    # --- 5. 绘图 (不显示，直接保存) ---
    print(f"解析完成，生成图表中 (共 {decode_step_count} 个数据点)...")

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    num_ranks = len(sp_history)
    # 使用 Tab10 调色板确保区分度
    colors = plt.cm.get_cmap('tab10', max(10, num_ranks))

    # 子图 1: SP Batch Size
    for rank_idx in range(num_ranks):
        ax1.plot(steps, sp_history[rank_idx], 
                 label=f'Rank {rank_idx}', 
                 color=colors(rank_idx),
                 linewidth=1.5, alpha=0.8)
    
    ax1.set_title('SP Batch Size (Decode Phase)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # 图例放在图外侧，避免遮挡数据
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 子图 2: Free Blocks
    for rank_idx in range(num_ranks):
        ax2.plot(steps, free_history[rank_idx], 
                 label=f'Rank {rank_idx}', 
                 color=colors(rank_idx),
                 linewidth=1.5, alpha=0.8)
    
    ax2.set_title('Free Blocks (Decode Phase)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Free Blocks Count', fontsize=12)
    ax2.set_xlabel('Decode Step', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    
    # --- 6. 保存文件 ---
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close() # 关闭图形，释放内存
    
    print(f"成功! 图片已保存为: {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <log_filename>")
    else:
        parse_and_plot_log(sys.argv[1])
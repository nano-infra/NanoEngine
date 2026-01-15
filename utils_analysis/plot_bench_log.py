import re
import ast
import matplotlib.pyplot as plt
import sys
import os
import argparse
import matplotlib.cm as cm
import numpy as np

def parse_and_plot_log(file_path, skip_count=0):
    # --- 1. Check file existence ---
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    # --- 2. Prepare output filename base ---
    base_name = os.path.splitext(file_path)[0]

    # --- 3. Read and clean log ---
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Remove ANSI color codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_content = ansi_escape.sub('', content)

    # --- 4. Extract Data ---
    dict_pattern = re.compile(r"step - (\{.*?\})")
    matches = dict_pattern.findall(clean_content)

    total_found = len(matches)
    print(f"Analyzing {file_path} ...")
    print(f"Found {total_found} raw step logs.")

    # --- Skip Logic ---
    if skip_count > 0:
        if skip_count >= total_found:
            print(f"Error: Skip count ({skip_count}) is larger than total matches. No data left.")
            return
        print(f"Skipping the first {skip_count} records (Warmup)...")
        matches = matches[skip_count:]
        print(f"Remaining records to process: {len(matches)}")
    # ------------------

    steps = []
    
    # Data containers: [step_index][rank_index]
    sp_data_timesteps = []   
    free_data_timesteps = [] 
    
    # 1D Data containers: [step_index]
    waiting_head_data = []
    waiting_total_data = []

    # New metrics
    waiting_reqs_data = []
    itl_data = []  # ITL in ms
    seq_lens_min_data = []
    seq_lens_max_data = []
    seq_lens_avg_data = []
    # Percentiles
    seq_lens_p50_data = []
    seq_lens_p90_data = []
    seq_lens_p95_data = []
    seq_lens_p99_data = []

    # Per-rank seq len data: [step_index][rank_index] -> sum of seq lens for that rank
    seq_lens_sum_per_rank = []
    # Per-rank seq count data: [step_index][rank_index] -> number of sequences for that rank
    seq_count_per_rank = []

    decode_step_count = 0

    print("Extracting decode data...")

    for dict_str in matches:
        try:
            data = ast.literal_eval(dict_str)
            
            if data.get('mode') != 'decode':
                continue

            def flatten(lst):
                flat = []
                for item in lst:
                    if isinstance(item, list):
                        flat.extend(flatten(item))
                    else:
                        flat.append(item)
                return flat

            # Extract existing metrics
            raw_sp = data.get('sp_batch_sizes', [])
            flat_sp = flatten(raw_sp)
            
            raw_free = data.get('free_blocks', [])
            flat_free = flatten(raw_free)
            
            # Extract scalars
            waiting_head = data.get('waiting_head_blocks', 0)
            waiting_total = data.get('waiting_total_blocks', 0)
            
            sp_data_timesteps.append(flat_sp)
            free_data_timesteps.append(flat_free)
            
            waiting_head_data.append(waiting_head)
            waiting_total_data.append(waiting_total)

            # New metrics
            w_reqs = data.get('waiting_reqs', 0)
            waiting_reqs_data.append(w_reqs)
            
            # Extract ITL (format: "12.34ms" -> 12.34)
            itl_str = data.get('itl', '0ms')
            try:
                itl_val = float(itl_str.replace('ms', ''))
            except ValueError:
                itl_val = 0.0
            itl_data.append(itl_val)

            # Use sp_seq_lens for all seq len analysis
            raw_sp_seq_lens = data.get('sp_seq_lens', [])
            flat_seq_lens = flatten(raw_sp_seq_lens)
            if flat_seq_lens:
                seq_lens_min_data.append(np.min(flat_seq_lens))
                seq_lens_max_data.append(np.max(flat_seq_lens))
                seq_lens_avg_data.append(np.mean(flat_seq_lens))
                # Calculate percentiles
                p50, p90, p95, p99 = np.percentile(flat_seq_lens, [50, 90, 95, 99])
                seq_lens_p50_data.append(p50)
                seq_lens_p90_data.append(p90)
                seq_lens_p95_data.append(p95)
                seq_lens_p99_data.append(p99)
            else:
                seq_lens_min_data.append(0)
                seq_lens_max_data.append(0)
                seq_lens_avg_data.append(0)
                seq_lens_p50_data.append(0)
                seq_lens_p90_data.append(0)
                seq_lens_p95_data.append(0)
                seq_lens_p99_data.append(0)

            # Per-GPU seq len analysis using sp_seq_lens (reuse raw_sp_seq_lens from above)
            # sp_seq_lens structure: [dp_idx][sp_idx] -> list of seq lens on that GPU
            rank_sums = []
            rank_counts = []
            for dp_group in raw_sp_seq_lens:
                for sp_rank_data in dp_group:
                    # sp_rank_data is a list of seq lens for this GPU
                    flat_gpu = flatten(sp_rank_data) if isinstance(sp_rank_data, list) else [sp_rank_data]
                    rank_sums.append(sum(flat_gpu))
                    rank_counts.append(len(flat_gpu))
            seq_lens_sum_per_rank.append(rank_sums)
            seq_count_per_rank.append(rank_counts)

            decode_step_count += 1
            steps.append(decode_step_count)

        except (ValueError, SyntaxError):
            continue

    if decode_step_count == 0:
        print("No decode phase data found.")
        return

    # Pivot data to [rank][step]
    if not sp_data_timesteps: 
        return

    num_ranks = len(sp_data_timesteps[0])
    sp_series = list(zip(*sp_data_timesteps))
    free_series = list(zip(*free_data_timesteps))
    
    print(f"Parse complete. Valid Steps: {decode_step_count}, Ranks: {num_ranks}")

    skip_suffix = f"_skip{skip_count}" if skip_count > 0 else ""

    # ==========================================
    # 1. 画一幅所有 Ranks 的 Free Blocks 总览图 (Line Chart)
    # ==========================================
    print("Generating All-Ranks Free Blocks Summary...")
    fig_all, ax_all = plt.subplots(figsize=(16, 8))
    
    # 使用 colors map 防止 rank 太多颜色重复看不清
    colors_all = cm.get_cmap('jet')(np.linspace(0, 1, num_ranks))

    for r_idx in range(num_ranks):
        ax_all.plot(steps, free_series[r_idx], 
                   label=f'Rank {r_idx}', 
                   color=colors_all[r_idx], 
                   linewidth=1, alpha=0.7)
    
    ax_all.set_title('Free Blocks Trend - All Ranks Summary', fontsize=16, fontweight='bold')
    ax_all.set_ylabel('Free Blocks Count', fontsize=14)
    ax_all.set_xlabel('Step', fontsize=14)
    ax_all.grid(True, linestyle='--', alpha=0.5)
    
    if num_ranks <= 16:
        ax_all.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
    else:
        ax_all.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2, title="Ranks (All)")
    
    output_all_free = f"{base_name}{skip_suffix}_ALL_FreeBlocks.png"
    plt.tight_layout()
    plt.savefig(output_all_free, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Summary: {output_all_free}")

    # ==========================================
    # 2. Grouped Plotting
    # ==========================================
    # Rows: 4
    # 1. SP Batch Sizes (Line)
    # 2. Waiting Blocks (System Global)
    # 3. Free Blocks (Stacked Bar) - 宏观总量
    # 4. Free Blocks (Line Chart) - 微观趋势 <--- 修改处：复用 free_series
    
    CHUNK_SIZE = 8
    num_groups = (num_ranks + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Generating {num_groups} grouped images...")
    cmap = plt.get_cmap('tab10')

    for group_idx in range(num_groups):
        start_rank = group_idx * CHUNK_SIZE
        end_rank = min((group_idx + 1) * CHUNK_SIZE, num_ranks)
        current_ranks_count = end_rank - start_rank
        
        output_png = f"{base_name}{skip_suffix}_group_{group_idx}.png"
        print(f"  -> Plotting Group {group_idx}: Ranks {start_rank}-{end_rank - 1} ...")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
        
        group_colors = [cmap(i % 10) for i in range(current_ranks_count)]

        # --- Subplot 1: SP Batch ---
        for i in range(current_ranks_count):
            global_idx = start_rank + i
            ax1.plot(steps, sp_series[global_idx], label=f'R{global_idx}', color=group_colors[i], lw=1.5, alpha=0.8)
        ax1.set_title(f'SP Batch Sizes (Ranks {start_rank}-{end_rank-1})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Batch Size', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        # --- Subplot 2: Waiting Blocks ---
        ax2.plot(steps, waiting_head_data, label='Wait Head', color='orange', lw=2, marker='o', ms=3)
        ax2.plot(steps, waiting_total_data, label='Wait Total', color='red', lw=2, ls='--', marker='x', ms=3)
        ax2.set_title('Waiting Blocks (Global)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        # --- Subplot 3: Free Blocks (Stacked Bar) ---
        bottom = [0] * len(steps)
        for i in range(current_ranks_count):
            global_idx = start_rank + i
            ax3.bar(steps, free_series[global_idx], bottom=bottom, label=f'R{global_idx}', color=group_colors[i], alpha=0.8, width=0.8)
            bottom = [b + v for b, v in zip(bottom, free_series[global_idx])]
        # Overlay line
        ax3.plot(steps, waiting_head_data, color='black', lw=2, zorder=10, label='Wait Head')
        ax3.set_title(f'Free Blocks Stacked (Sum of Ranks {start_rank}-{end_rank-1})', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Total Count', fontsize=12)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        # --- Subplot 4: Free Blocks (Line Chart - Individual Trends) ---
        # 这里的意义在于：Stacked Bar 很难看出单个 Rank 的 Free Blocks 是否在剧烈抖动，折线图可以看得很清楚
        for i in range(current_ranks_count):
            global_idx = start_rank + i
            # 复用 free_series 画折线
            ax4.plot(steps, free_series[global_idx], 
                     label=f'R{global_idx}', 
                     color=group_colors[i], 
                     linewidth=1.5, alpha=0.8)
        
        ax4.set_title(f'Free Blocks Trend (Individual Lines for Ranks {start_rank}-{end_rank-1})', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Free Blocks Count', fontsize=12)
        ax4.set_xlabel('Step', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()

    print("All groups processed successfully.")

    # ==========================================
    # 3. New Plot: Request & Sequence Stats (with ITL)
    # ==========================================
    print("Generating Request & Sequence Stats Summary...")
    fig_req, (ax_itl, ax_req, ax_seq) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # Subplot 1: ITL Trend
    ax_itl.plot(steps, itl_data, label='ITL', color='darkblue', lw=2)
    ax_itl.set_title('Inter-Token Latency (ITL) Trend', fontsize=14, fontweight='bold')
    ax_itl.set_ylabel('ITL (ms)', fontsize=12)
    ax_itl.grid(True, linestyle='--', alpha=0.5)
    ax_itl.legend(loc='upper right')
    # Add average line
    if itl_data:
        avg_itl = np.mean(itl_data)
        ax_itl.axhline(y=avg_itl, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_itl:.2f}ms')
        ax_itl.legend(loc='upper right')

    # Subplot 2: Waiting Requests
    ax_req.plot(steps, waiting_reqs_data, label='Waiting Reqs', color='purple', lw=2)
    ax_req.set_title('Waiting Requests Trend', fontsize=14, fontweight='bold')
    ax_req.set_ylabel('Count', fontsize=12)
    ax_req.grid(True, linestyle='--', alpha=0.5)
    ax_req.legend(loc='upper right')

    # Subplot 3: Sequence Length Stats
    ax_seq.plot(steps, seq_lens_max_data, label='Max', color='red', lw=1.5, linestyle='--')
    ax_seq.plot(steps, seq_lens_p99_data, label='P99', color='darkorange', lw=1.5, linestyle='-')
    ax_seq.plot(steps, seq_lens_p95_data, label='P95', color='orange', lw=1.5, linestyle='-')
    ax_seq.plot(steps, seq_lens_p90_data, label='P90', color='gold', lw=1.5, linestyle='-')
    ax_seq.plot(steps, seq_lens_avg_data, label='Avg', color='blue', lw=2)
    ax_seq.plot(steps, seq_lens_p50_data, label='P50 (Median)', color='cyan', lw=1.5, linestyle='-')
    ax_seq.plot(steps, seq_lens_min_data, label='Min', color='green', lw=1.5, linestyle=':')
    
    ax_seq.set_title('Sequence Length Statistics (Running)', fontsize=14, fontweight='bold')
    ax_seq.set_ylabel('Token Count', fontsize=12)
    ax_seq.set_xlabel('Step', fontsize=12)
    ax_seq.grid(True, linestyle='--', alpha=0.5)
    # Adjust legend to not cover data too much
    ax_seq.legend(loc='upper left', ncol=2, fontsize='small')

    output_req_stats = f"{base_name}{skip_suffix}_RequestStats.png"
    plt.tight_layout()
    plt.savefig(output_req_stats, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Stats: {output_req_stats}")

    # ==========================================
    # 4. New Plot: Per-GPU Seq Len Load Analysis
    # ==========================================
    if seq_lens_sum_per_rank and len(seq_lens_sum_per_rank[0]) > 0:
        print("Generating Per-GPU Seq Len Load Analysis...")
        
        # Pivot data: seq_lens_sum_per_rank[step][rank] -> seq_sum_series[rank][step]
        num_seq_ranks = len(seq_lens_sum_per_rank[0])
        seq_sum_series = list(zip(*seq_lens_sum_per_rank))
        seq_count_series = list(zip(*seq_count_per_rank))
        
        # Figure with 4 subplots
        fig_load, ((ax_load1, ax_load2), (ax_load3, ax_load4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors_load = cm.get_cmap('tab20')(np.linspace(0, 1, num_seq_ranks))
        
        # --- Subplot 1: Per-Rank Seq Len Sum (Line Chart) ---
        for r_idx in range(num_seq_ranks):
            ax_load1.plot(steps, seq_sum_series[r_idx], 
                         label=f'Rank {r_idx}', 
                         color=colors_load[r_idx], 
                         linewidth=1.5, alpha=0.8)
        ax_load1.set_title('Per-GPU Total Seq Len (Token Load)', fontsize=14, fontweight='bold')
        ax_load1.set_ylabel('Total Token Count', fontsize=12)
        ax_load1.set_xlabel('Step', fontsize=12)
        ax_load1.grid(True, linestyle='--', alpha=0.5)
        if num_seq_ranks <= 16:
            ax_load1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
        else:
            ax_load1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2)
        
        # --- Subplot 2: Per-Rank Seq Count (Line Chart) ---
        for r_idx in range(num_seq_ranks):
            ax_load2.plot(steps, seq_count_series[r_idx], 
                         label=f'Rank {r_idx}', 
                         color=colors_load[r_idx], 
                         linewidth=1.5, alpha=0.8)
        ax_load2.set_title('Per-GPU Sequence Count', fontsize=14, fontweight='bold')
        ax_load2.set_ylabel('Sequence Count', fontsize=12)
        ax_load2.set_xlabel('Step', fontsize=12)
        ax_load2.grid(True, linestyle='--', alpha=0.5)
        if num_seq_ranks <= 16:
            ax_load2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
        else:
            ax_load2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2)
        
        # --- Subplot 3: Stacked Bar for Seq Len Sum Distribution ---
        bottom = [0] * len(steps)
        for r_idx in range(num_seq_ranks):
            ax_load3.bar(steps, seq_sum_series[r_idx], bottom=bottom, 
                        label=f'Rank {r_idx}', color=colors_load[r_idx], alpha=0.8, width=0.8)
            bottom = [b + v for b, v in zip(bottom, seq_sum_series[r_idx])]
        ax_load3.set_title('Seq Len Distribution Across Ranks (Stacked)', fontsize=14, fontweight='bold')
        ax_load3.set_ylabel('Total Token Count', fontsize=12)
        ax_load3.set_xlabel('Step', fontsize=12)
        ax_load3.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # --- Subplot 4: Load Imbalance Metrics ---
        # Calculate CV (coefficient of variation) and Max/Avg ratio per step
        load_cv = []
        load_max_avg_ratio = []
        for step_data in seq_lens_sum_per_rank:
            if step_data and sum(step_data) > 0:
                avg = np.mean(step_data)
                std = np.std(step_data)
                max_val = np.max(step_data)
                cv = std / avg if avg > 0 else 0
                ratio = max_val / avg if avg > 0 else 1
                load_cv.append(cv)
                load_max_avg_ratio.append(ratio)
            else:
                load_cv.append(0)
                load_max_avg_ratio.append(1)
        
        ax_load4.plot(steps, load_cv, label='CV (Std/Avg)', color='blue', lw=2)
        ax_load4.plot(steps, load_max_avg_ratio, label='Max/Avg Ratio', color='red', lw=2, linestyle='--')
        ax_load4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Ideal (1.0)')
        ax_load4.set_title('Load Imbalance Metrics', fontsize=14, fontweight='bold')
        ax_load4.set_ylabel('Ratio', fontsize=12)
        ax_load4.set_xlabel('Step', fontsize=12)
        ax_load4.grid(True, linestyle='--', alpha=0.5)
        ax_load4.legend(loc='upper right')
        
        output_load = f"{base_name}{skip_suffix}_PerRankSeqLoad.png"
        plt.tight_layout()
        plt.savefig(output_load, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Saved Per-Rank Load Analysis: {output_load}")
    else:
        print("No per-rank seq_lens data available for load analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot log data.")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--skip", "-s", type=int, default=17, help="Initial steps to skip")
    args = parser.parse_args()

    parse_and_plot_log(args.log_file, args.skip)
import re
import ast
import matplotlib.pyplot as plt
import sys
import os

def parse_and_plot_log(file_path):
    # --- 1. Check file existence ---
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    # --- 2. Prepare output filename ---
    base_name = os.path.splitext(file_path)[0]
    output_png = f"{base_name}.png"

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
    # Find dictionary structure after "step - "
    dict_pattern = re.compile(r"step - (\{.*?\})")
    matches = dict_pattern.findall(clean_content)

    steps = []
    
    # Data containers: [step_index][rank_index]
    sp_data_timesteps = []   
    free_data_timesteps = [] 
    
    # 1D Data containers: [step_index]
    waiting_head_data = []
    waiting_total_data = []

    decode_step_count = 0

    print(f"Analyzing {file_path} ...")
    print(f"Found {len(matches)} step logs. Extracting decode data...")

    for dict_str in matches:
        try:
            data = ast.literal_eval(dict_str)
            
            # Process only decode mode
            if data.get('mode') != 'decode':
                continue

            # Helper to flatten nested lists (arbitrary depth)
            def flatten(lst):
                flat = []
                for item in lst:
                    if isinstance(item, list):
                        flat.extend(flatten(item))
                    else:
                        flat.append(item)
                return flat

            # Extract and flatten sp_batch_sizes
            raw_sp = data.get('sp_batch_sizes', [])
            flat_sp = flatten(raw_sp)
            
            # Extract and flatten free_blocks
            raw_free = data.get('free_blocks', [])
            flat_free = flatten(raw_free)

            # Extract waiting blocks (scalars)
            waiting_head = data.get('waiting_head_blocks', 0)
            waiting_total = data.get('waiting_total_blocks', 0)
            
            # Validate consistency (optional, but good for debugging)
            if sp_data_timesteps and len(flat_sp) != len(sp_data_timesteps[0]):
                # Warning: rank count changed? Proceed anyway by truncating or padding if needed, 
                # but for now assume consistency.
                pass

            sp_data_timesteps.append(flat_sp)
            free_data_timesteps.append(flat_free)
            waiting_head_data.append(waiting_head)
            waiting_total_data.append(waiting_total)

            decode_step_count += 1
            steps.append(decode_step_count)

        except (ValueError, SyntaxError):
            continue

    if decode_step_count == 0:
        print("No decode phase data found. Image will not be generated.")
        return

    # Pivot data to [rank][step] for plotting series
    # Using zip(*) to transpose
    # We assume at least one step exists and ranks are consistent
    if not sp_data_timesteps: 
        return

    num_ranks = len(sp_data_timesteps[0])
    sp_series = list(zip(*sp_data_timesteps))
    free_series = list(zip(*free_data_timesteps))

    print(f"Parse complete. Generating charts for {decode_step_count} steps. Total Ranks found: {num_ranks}")

    # --- 5. Plotting ---
    # Rows: 
    # 1. SP Batch Sizes (Line)
    # 2. Waiting Blocks (Lines) - Head & Total
    # 3. Free Blocks (Stacked Bar)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Color map
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(num_ranks)]

    # --- Plot 1: SP Batch Size ---
    for rank_idx, rank_data in enumerate(sp_series):
        ax1.plot(steps, rank_data, 
                 label=f'Rank {rank_idx}', 
                 color=colors[rank_idx],
                 linewidth=1.5, alpha=0.8)
    
    ax1.set_title('SP Batch Sizes (Decode Phase)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Batch Size', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Legend outside
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Ranks")

    # --- Plot 2: Waiting Blocks ---
    ax2.plot(steps, waiting_head_data, label='Waiting Head Blocks', color='orange', linewidth=2, marker='o', markersize=3)
    ax2.plot(steps, waiting_total_data, label='Waiting Total Blocks', color='red', linewidth=2, linestyle='--', marker='x', markersize=3)
    
    ax2.set_title('Waiting Blocks (Decode Phase)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Block Count', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # --- Plot 3: Free Blocks (Stacked Bar) ---
    bottom = [0] * len(steps)
    for rank_idx, rank_data in enumerate(free_series):
        ax3.bar(steps, rank_data, bottom=bottom, 
                label=f'Rank {rank_idx}', 
                color=colors[rank_idx], alpha=0.9, width=0.8)
        # Manually add to bottom list
        bottom = [b + v for b, v in zip(bottom, rank_data)]
    
    ax3.set_title('Free Blocks (Stacked)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Free Block Count', fontsize=12)
    ax3.set_xlabel('Decode Step', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Ranks")

    plt.tight_layout()
    
    # --- 6. Save ---
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close() # Close to free memory
    
    print(f"Success! Image saved to: {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <log_filename>")
    else:
        parse_and_plot_log(sys.argv[1])
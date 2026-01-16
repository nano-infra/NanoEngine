"""
CSV Dataset Distribution Analysis Script

This script analyzes the distribution of prompt_len and output_len in the benchmark
CSV dataset to investigate potential causes of ITL spikes during inference.

Usage:
    python analyze_csv_dataset.py <csv_path> [--num-requests N] [--max-model-len M] [--window W]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CSV dataset distribution for benchmark.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("--num-requests", "-n", type=int, default=None, 
                        help="Number of requests to analyze (default: all)")
    parser.add_argument("--max-model-len", "-m", type=int, default=4096,
                        help="Max model length for truncation simulation (default: 4096)")
    parser.add_argument("--window", "-w", type=int, default=100,
                        help="Rolling window size for statistics (default: 100)")
    parser.add_argument("--output-prefix", "-o", type=str, default=None,
                        help="Output file prefix (default: based on input filename)")
    return parser.parse_args()


def load_and_preprocess(csv_path, num_requests, max_model_len):
    """Load CSV and apply the same truncation logic as bench_serving.py"""
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    if "prompt_len" not in df.columns or "output_len" not in df.columns:
        print("Error: CSV must contain 'prompt_len' and 'output_len' columns")
        sys.exit(1)
    
    original_len = len(df)
    
    # Apply num_requests limit
    if num_requests is not None:
        if num_requests > len(df):
            print(f"Warning: Requested {num_requests} samples but only {len(df)} available.")
            repeats = (num_requests // len(df)) + 1
            df = pd.concat([df] * repeats, ignore_index=True)
        df = df.head(num_requests)
    
    print(f"Loaded {original_len} rows from CSV, using {len(df)} for analysis.")
    
    # Store original values
    df['original_prompt_len'] = df['prompt_len'].copy()
    df['original_output_len'] = df['output_len'].copy()
    df['original_total_len'] = df['prompt_len'] + df['output_len']
    
    # Apply truncation logic (same as bench_serving.py)
    df['effective_prompt_len'] = df['prompt_len'].copy()
    df['effective_output_len'] = df['output_len'].copy()
    df['truncated'] = False
    
    for idx in df.index:
        prompt_len = df.loc[idx, 'prompt_len']
        output_len = df.loc[idx, 'output_len']
        
        if prompt_len > max_model_len or prompt_len + output_len > max_model_len:
            df.loc[idx, 'truncated'] = True
            if max_model_len - output_len < 4:
                df.loc[idx, 'effective_output_len'] = max_model_len - 4
                df.loc[idx, 'effective_prompt_len'] = 4
            else:
                df.loc[idx, 'effective_prompt_len'] = max_model_len - output_len
    
    df['effective_total_len'] = df['effective_prompt_len'] + df['effective_output_len']
    df['request_idx'] = range(len(df))
    
    return df


def compute_rolling_stats(df, window):
    """Compute rolling statistics for key metrics"""
    df['rolling_prompt_avg'] = df['effective_prompt_len'].rolling(window=window, min_periods=1).mean()
    df['rolling_prompt_max'] = df['effective_prompt_len'].rolling(window=window, min_periods=1).max()
    df['rolling_prompt_std'] = df['effective_prompt_len'].rolling(window=window, min_periods=1).std()
    
    df['rolling_output_avg'] = df['effective_output_len'].rolling(window=window, min_periods=1).mean()
    df['rolling_output_max'] = df['effective_output_len'].rolling(window=window, min_periods=1).max()
    
    df['rolling_total_avg'] = df['effective_total_len'].rolling(window=window, min_periods=1).mean()
    df['rolling_total_max'] = df['effective_total_len'].rolling(window=window, min_periods=1).max()
    
    return df


def detect_spikes(df, column, threshold_std=2.0):
    """Detect data points where the value exceeds threshold_std standard deviations"""
    mean_val = df[column].mean()
    std_val = df[column].std()
    threshold = mean_val + threshold_std * std_val
    
    spikes = df[df[column] > threshold].copy()
    return spikes, threshold


def print_statistics(df):
    """Print detailed statistics about the dataset"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print("\n--- Original Data (before truncation) ---")
    print(f"  Prompt Length:  min={df['original_prompt_len'].min():.0f}, "
          f"max={df['original_prompt_len'].max():.0f}, "
          f"mean={df['original_prompt_len'].mean():.1f}, "
          f"std={df['original_prompt_len'].std():.1f}")
    print(f"  Output Length:  min={df['original_output_len'].min():.0f}, "
          f"max={df['original_output_len'].max():.0f}, "
          f"mean={df['original_output_len'].mean():.1f}, "
          f"std={df['original_output_len'].std():.1f}")
    print(f"  Total Length:   min={df['original_total_len'].min():.0f}, "
          f"max={df['original_total_len'].max():.0f}, "
          f"mean={df['original_total_len'].mean():.1f}, "
          f"std={df['original_total_len'].std():.1f}")
    
    print("\n--- Effective Data (after truncation) ---")
    print(f"  Prompt Length:  min={df['effective_prompt_len'].min():.0f}, "
          f"max={df['effective_prompt_len'].max():.0f}, "
          f"mean={df['effective_prompt_len'].mean():.1f}, "
          f"std={df['effective_prompt_len'].std():.1f}")
    print(f"  Output Length:  min={df['effective_output_len'].min():.0f}, "
          f"max={df['effective_output_len'].max():.0f}, "
          f"mean={df['effective_output_len'].mean():.1f}, "
          f"std={df['effective_output_len'].std():.1f}")
    print(f"  Total Length:   min={df['effective_total_len'].min():.0f}, "
          f"max={df['effective_total_len'].max():.0f}, "
          f"mean={df['effective_total_len'].mean():.1f}, "
          f"std={df['effective_total_len'].std():.1f}")
    
    truncated_count = df['truncated'].sum()
    print(f"\n--- Truncation ---")
    print(f"  Truncated samples: {truncated_count} / {len(df)} ({100*truncated_count/len(df):.2f}%)")
    
    # Percentiles
    print("\n--- Percentiles (Effective Total Length) ---")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(df['effective_total_len'], p)
        print(f"  P{p}: {val:.0f}")
    
    print("=" * 70)


def plot_distribution_analysis(df, output_prefix, window):
    """Generate comprehensive distribution analysis plots"""
    
    fig = plt.figure(figsize=(18, 24))
    
    # === Row 1: Histograms ===
    ax1 = fig.add_subplot(5, 2, 1)
    ax1.hist(df['effective_prompt_len'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(df['effective_prompt_len'].mean(), color='red', linestyle='--', label=f"Mean: {df['effective_prompt_len'].mean():.0f}")
    ax1.set_title('Prompt Length Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Prompt Length (tokens)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(5, 2, 2)
    ax2.hist(df['effective_output_len'], bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(df['effective_output_len'].mean(), color='red', linestyle='--', label=f"Mean: {df['effective_output_len'].mean():.0f}")
    ax2.set_title('Output Length Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Output Length (tokens)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Row 2: Scatter plot by request index ===
    ax3 = fig.add_subplot(5, 2, 3)
    ax3.scatter(df['request_idx'], df['effective_prompt_len'], alpha=0.3, s=5, c='steelblue', label='Prompt')
    ax3.plot(df['request_idx'], df['rolling_prompt_avg'], color='red', lw=2, label=f'Rolling Avg (w={window})')
    ax3.plot(df['request_idx'], df['rolling_prompt_max'], color='darkred', lw=1, linestyle='--', label=f'Rolling Max')
    ax3.set_title('Prompt Length by Request Index', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Request Index')
    ax3.set_ylabel('Prompt Length')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(5, 2, 4)
    ax4.scatter(df['request_idx'], df['effective_output_len'], alpha=0.3, s=5, c='darkorange', label='Output')
    ax4.plot(df['request_idx'], df['rolling_output_avg'], color='red', lw=2, label=f'Rolling Avg (w={window})')
    ax4.plot(df['request_idx'], df['rolling_output_max'], color='darkred', lw=1, linestyle='--', label=f'Rolling Max')
    ax4.set_title('Output Length by Request Index', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Request Index')
    ax4.set_ylabel('Output Length')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # === Row 3: Total length analysis ===
    ax5 = fig.add_subplot(5, 2, 5)
    ax5.scatter(df['request_idx'], df['effective_total_len'], alpha=0.3, s=5, c='green', label='Total')
    ax5.plot(df['request_idx'], df['rolling_total_avg'], color='red', lw=2, label=f'Rolling Avg')
    ax5.plot(df['request_idx'], df['rolling_total_max'], color='darkred', lw=1, linestyle='--', label=f'Rolling Max')
    ax5.set_title('Total Sequence Length (Prompt + Output) by Request Index', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Request Index')
    ax5.set_ylabel('Total Length')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(5, 2, 6)
    ax6.hist(df['effective_total_len'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax6.axvline(df['effective_total_len'].mean(), color='red', linestyle='--', label=f"Mean: {df['effective_total_len'].mean():.0f}")
    for p in [90, 95, 99]:
        val = np.percentile(df['effective_total_len'], p)
        ax6.axvline(val, color='purple', linestyle=':', alpha=0.7)
        ax6.text(val, ax6.get_ylim()[1]*0.9, f'P{p}', fontsize=8)
    ax6.set_title('Total Length Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Total Length (tokens)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # === Row 4: Rolling statistics ===
    ax7 = fig.add_subplot(5, 2, 7)
    ax7.plot(df['request_idx'], df['rolling_prompt_std'].fillna(0), color='steelblue', lw=2, label='Prompt Std')
    ax7.set_title(f'Rolling Std Dev (window={window})', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Request Index')
    ax7.set_ylabel('Std Dev')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Detect spikes
    spikes, threshold = detect_spikes(df, 'effective_total_len', threshold_std=2.0)
    
    ax8 = fig.add_subplot(5, 2, 8)
    ax8.scatter(df['request_idx'], df['effective_total_len'], alpha=0.2, s=5, c='gray', label='Normal')
    if len(spikes) > 0:
        ax8.scatter(spikes['request_idx'], spikes['effective_total_len'], alpha=0.8, s=20, c='red', label=f'Spikes (>{threshold:.0f})')
    ax8.axhline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold (2σ): {threshold:.0f}')
    ax8.set_title('Spike Detection (Total Length > Mean + 2*Std)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Request Index')
    ax8.set_ylabel('Total Length')
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)
    
    # === Row 5: Cumulative and segment analysis ===
    # Divide into segments and analyze variation
    num_segments = 10
    segment_size = len(df) // num_segments
    segment_stats = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < num_segments - 1 else len(df)
        seg_df = df.iloc[start_idx:end_idx]
        segment_stats.append({
            'segment': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'prompt_mean': seg_df['effective_prompt_len'].mean(),
            'output_mean': seg_df['effective_output_len'].mean(),
            'total_mean': seg_df['effective_total_len'].mean(),
            'total_max': seg_df['effective_total_len'].max(),
            'total_std': seg_df['effective_total_len'].std()
        })
    
    seg_df = pd.DataFrame(segment_stats)
    
    ax9 = fig.add_subplot(5, 2, 9)
    x = np.arange(num_segments)
    width = 0.35
    ax9.bar(x - width/2, seg_df['prompt_mean'], width, label='Prompt Mean', color='steelblue', alpha=0.7)
    ax9.bar(x + width/2, seg_df['output_mean'], width, label='Output Mean', color='darkorange', alpha=0.7)
    ax9.set_title('Mean Length by Dataset Segment (10 equal parts)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Segment')
    ax9.set_ylabel('Mean Length')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f"{s['start_idx']}-{s['end_idx']}" for s in segment_stats], rotation=45, ha='right', fontsize=8)
    ax9.legend()
    ax9.grid(True, axis='y', alpha=0.3)
    
    ax10 = fig.add_subplot(5, 2, 10)
    ax10.bar(x, seg_df['total_max'], color='red', alpha=0.5, label='Max')
    ax10.bar(x, seg_df['total_mean'], color='green', alpha=0.7, label='Mean')
    ax10.errorbar(x, seg_df['total_mean'], yerr=seg_df['total_std'], fmt='none', color='black', capsize=3, label='Std')
    ax10.set_title('Total Length Stats by Segment', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Segment')
    ax10.set_ylabel('Total Length')
    ax10.set_xticks(x)
    ax10.set_xticklabels([f"{s['start_idx']}-{s['end_idx']}" for s in segment_stats], rotation=45, ha='right', fontsize=8)
    ax10.legend()
    ax10.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution analysis: {output_file}")
    
    # Print spike info
    if len(spikes) > 0:
        print(f"\n--- Detected {len(spikes)} spikes (total_len > {threshold:.0f}) ---")
        print("Top 10 longest requests:")
        top_spikes = spikes.nlargest(10, 'effective_total_len')[['request_idx', 'effective_prompt_len', 'effective_output_len', 'effective_total_len']]
        print(top_spikes.to_string(index=False))


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print("CSV Dataset Distribution Analysis")
    print(f"{'='*70}")
    print(f"Input: {args.csv_path}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Rolling Window: {args.window}")
    
    # Load and preprocess
    df = load_and_preprocess(args.csv_path, args.num_requests, args.max_model_len)
    
    # Compute rolling stats
    df = compute_rolling_stats(df, args.window)
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    output_prefix = args.output_prefix or os.path.splitext(args.csv_path)[0]
    plot_distribution_analysis(df, output_prefix, args.window)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

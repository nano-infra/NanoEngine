#!/usr/bin/env python3
"""Plot the Attention and DeepEP scaling curves in Figure 3."""

import argparse
import glob
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from ae_utils.plotting import get_plot_font_family

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator


REFERENCE_DIR = SCRIPT_DIR / 'reference_results'
PAPER_RESULT_DIR = REFERENCE_DIR / 'attention'
FONT_NAME = get_plot_font_family()


plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': FONT_NAME,
    'font.weight': 'bold',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.labelweight': 'bold',
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


FIGSIZE = (3.3, 7/6)
MLA_COLORS = ['#4C78A8', '#F28E2B', '#59A14F']
MLA_MARKERS = ['o', 's', '^']
MLA_LINESTYLES = ['-', '--', '-.']
DEEPEP_COLORS = {
    'Dispatch': '#4C78A8',
    'Combine': '#F28E2B',
    'Dispatch + Combine': '#59A14F',
}
KEEP_BATCH_SIZES = {1, 128, 1024}
# KEEP_BATCH_SIZES = {1, 64, 128}
MLA_REQUIRED_COLUMNS = ['seq_len', 'batch_size', 'total_token_num', 'time_us']
DEEPEP_REQUIRED_COLUMNS = ['token', 'dispatch_avg_us', 'combine_avg_us', 'dispatch_combine_avg_us']
MLA_X_TICK_STEP = 256 * 1024
MLA_X_TICK_START = 256 * 1024
MLA_Y_TICK_STEP = 150
DEEPEP_X_TICK_STEP = 64
MLA_MARK_EVERY = 1
DEEPEP_MARK_EVERY = 3
X_TICK_PAD = 1.5
X_LABEL_PAD = 0.0
CAPTION_Y = -0.38
CAPTION_SIZE = 9.0
DEFAULT_MLA_CSV_GLOB = '*flashmla*cudagraph_total_tokens*.csv'


def smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.astype(float)
    if window % 2 == 0:
        window += 1
    return pd.Series(values.astype(float)).rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean().to_numpy()


def format_token_axis(value, _):
    if value >= 1024 ** 2:
        scaled = value / (1024 ** 2)
        return f'{int(scaled)}M' if float(scaled).is_integer() else f'{scaled:.1f}M'
    if value >= 1024:
        scaled = value / 1024
        return f'{int(scaled)}k' if float(scaled).is_integer() else f'{scaled:.1f}k'
    return f'{int(value)}'


def resolve_mla_files(data_dir: str, pattern: str) -> list[str]:
    if pattern == 'latest':
        latest_candidates = glob.glob(os.path.join(data_dir, DEFAULT_MLA_CSV_GLOB))
        if not latest_candidates:
            raise FileNotFoundError(
                f'No MLA CSV files found in {data_dir} with pattern {DEFAULT_MLA_CSV_GLOB}'
            )
        return [max(latest_candidates, key=os.path.getmtime)]

    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f'No MLA CSV files found in {data_dir} with pattern {pattern}')
    return files


def load_mla_data(data_dir: str, pattern: str) -> pd.DataFrame:
    files = resolve_mla_files(data_dir, pattern)

    rows = []
    for path in files:
        df = pd.read_csv(path)
        missing = [column for column in MLA_REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f'{Path(path).name} missing MLA columns: {missing}')

        sub = df[MLA_REQUIRED_COLUMNS].copy()
        for column in MLA_REQUIRED_COLUMNS:
            sub[column] = pd.to_numeric(sub[column], errors='coerce')
        sub = sub.dropna(subset=MLA_REQUIRED_COLUMNS)
        if sub.empty:
            continue
        rows.append(sub)

    if not rows:
        raise ValueError('MLA data is empty after filtering invalid rows.')

    merged = pd.concat(rows, ignore_index=True)
    merged['batch_size'] = merged['batch_size'].astype(int)
    merged['seq_len'] = merged['seq_len'].astype(int)
    merged['total_token_num'] = merged['total_token_num'].astype(int)
    merged = merged[merged['batch_size'].isin(KEEP_BATCH_SIZES)]
    if merged.empty:
        raise ValueError(f'No MLA rows left after filtering batch sizes: {sorted(KEEP_BATCH_SIZES)}')

    grouped = merged.groupby(['batch_size', 'total_token_num'], as_index=False)['time_us'].mean()
    return grouped.sort_values(['batch_size', 'total_token_num']).reset_index(drop=True)


def load_deepep_data(data_dir: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f'No DeepEP CSV files found in {data_dir} with pattern {pattern}. '
            'Use the result directory produced by run_deepep.sh and make sure '
            '--num-nodes matches the directory suffix.'
        )

    rows = []
    for path in files:
        df = pd.read_csv(path)
        missing = [column for column in DEEPEP_REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f'{Path(path).name} missing DeepEP columns: {missing}')

        if 'status' in df.columns:
            df = df[df['status'] == 'ok']

        sub = df[DEEPEP_REQUIRED_COLUMNS].copy()
        for column in DEEPEP_REQUIRED_COLUMNS:
            sub[column] = pd.to_numeric(sub[column], errors='coerce')
        sub = sub.dropna(subset=DEEPEP_REQUIRED_COLUMNS)
        if sub.empty:
            continue
        sub['token'] = sub['token'].astype(int)
        rows.append(sub)

    if not rows:
        raise ValueError('DeepEP data is empty after filtering invalid rows.')

    merged = pd.concat(rows, ignore_index=True)
    merged['token'] = merged['token'].astype(int)
    grouped = merged.groupby('token')[
        ['dispatch_avg_us', 'combine_avg_us', 'dispatch_combine_avg_us']
    ].mean().sort_index()
    grouped.attrs['num_nodes'] = len(files)
    grouped.attrs['num_gpus'] = len(files) * 8
    return grouped


def style_axis(ax):
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_mla(ax, df: pd.DataFrame, smooth: bool, smooth_window: int):
    handles = []
    labels = []

    for idx, batch_size in enumerate(sorted(df['batch_size'].unique().astype(int))):
        sub = df[df['batch_size'] == batch_size].sort_values('total_token_num')
        x = sub['total_token_num'].to_numpy(dtype=int)
        y = sub['time_us'].to_numpy(dtype=float)
        if smooth and len(y) >= 3:
            y = smooth_series(y, smooth_window)

        line, = ax.plot(
            x,
            y,
            marker=MLA_MARKERS[idx % len(MLA_MARKERS)],
            markevery=MLA_MARK_EVERY,
            markersize=2.5,
            linewidth=1.35,
            linestyle=MLA_LINESTYLES[idx % len(MLA_LINESTYLES)],
            color=MLA_COLORS[idx % len(MLA_COLORS)],
            label=f'BS={batch_size}',
            markerfacecolor='white',
            markeredgewidth=0.6,
        )
        handles.append(line)
        labels.append(f'BS={batch_size}')

    x_values = df['total_token_num'].to_numpy(dtype=int)
    x_min = int(np.min(x_values))
    x_max = int(np.max(x_values))
    start_base = max(MLA_X_TICK_START, x_min)
    start_tick = ((start_base + MLA_X_TICK_STEP - 1) // MLA_X_TICK_STEP) * MLA_X_TICK_STEP
    xtick_pos = list(range(start_tick, x_max + 1, MLA_X_TICK_STEP))
    if not xtick_pos:
        xtick_pos = [x_max]
    ax.set_xlim(left=x_min, right=x_max)
    ax.set_xticks(xtick_pos)
    ax.xaxis.set_major_formatter(FuncFormatter(format_token_axis))

    y_arr = df['time_us'].to_numpy(dtype=float)
    y_bottom = max(0.0, float(np.nanmin(y_arr)) * 0.97)
    y_top = float(np.nanmax(y_arr)) * 1.08
    ax.set_ylim(bottom=y_bottom, top=y_top)

    y_tick_min = int(np.floor(y_bottom / MLA_Y_TICK_STEP) * MLA_Y_TICK_STEP)
    y_tick_max = int(np.ceil(y_top / MLA_Y_TICK_STEP) * MLA_Y_TICK_STEP)
    if y_tick_max <= y_tick_min:
        y_tick_max = y_tick_min + MLA_Y_TICK_STEP
    ax.set_yticks(np.arange(y_tick_min, y_tick_max + MLA_Y_TICK_STEP, MLA_Y_TICK_STEP))

    ax.set_xlabel('Total Sequence Length', labelpad=X_LABEL_PAD)
    ax.set_ylabel('Latency (μs)')
    style_axis(ax)
    ax.tick_params(axis='x', pad=X_TICK_PAD)

    ax.legend(
        handles,
        labels,
        loc='upper left',
        frameon=False,
        ncol=1,
        handlelength=1.1,
        handletextpad=0.24,
        borderaxespad=0.2,
        columnspacing=0.08,
        labelspacing=0.15,
        # prop={'size': 8},
    )
    ax.text(
        0.0,
        CAPTION_Y,
        '(a) FlashMLA latency',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=CAPTION_SIZE,
        linespacing=0.80,
        multialignment='left',
    )


def plot_deepep(ax, df: pd.DataFrame, smooth: bool, smooth_window: int):
    tokens = df.index.to_numpy(dtype=int)
    dispatch = df['dispatch_avg_us'].to_numpy(dtype=float)
    combine = df['combine_avg_us'].to_numpy(dtype=float)
    dispatch_combine = df['dispatch_combine_avg_us'].to_numpy(dtype=float)

    if smooth and len(tokens) >= 3:
        dispatch = smooth_series(dispatch, smooth_window)
        combine = smooth_series(combine, smooth_window)
        dispatch_combine = smooth_series(dispatch_combine, smooth_window)

    line_dispatch, = ax.plot(
        tokens,
        dispatch,
        marker='o',
        markevery=DEEPEP_MARK_EVERY,
        markersize=2.5,
        linewidth=1.35,
        color=DEEPEP_COLORS['Dispatch'],
        label='Dispatch',
        markerfacecolor='white',
        markeredgewidth=0.6,
    )
    line_combine, = ax.plot(
        tokens,
        combine,
        marker='s',
        markevery=DEEPEP_MARK_EVERY,
        markersize=2.3,
        linewidth=1.35,
        color=DEEPEP_COLORS['Combine'],
        label='Combine',
        markerfacecolor='white',
        markeredgewidth=0.6,
    )
    line_dispatch_combine, = ax.plot(
        tokens,
        dispatch_combine,
        marker='^',
        markevery=DEEPEP_MARK_EVERY,
        markersize=2.4,
        linewidth=1.45,
        color=DEEPEP_COLORS['Dispatch + Combine'],
        label='Disp.+Comb.',
        markerfacecolor='white',
        markeredgewidth=0.6,
    )

    token_min = int(tokens.min())
    token_max = int(tokens.max())
    start_tick = ((max(token_min, DEEPEP_X_TICK_STEP) + DEEPEP_X_TICK_STEP - 1) // DEEPEP_X_TICK_STEP) * DEEPEP_X_TICK_STEP
    xtick_pos = list(range(start_tick, token_max + 1, DEEPEP_X_TICK_STEP))
    if not xtick_pos:
        xtick_pos = [token_max]
    ax.set_xlim(left=token_min, right=token_max)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([str(value) for value in xtick_pos])

    y_bottom = 0.0
    y_top = float(np.nanmax([dispatch.max(), combine.max(), dispatch_combine.max()])) * 1.08
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, integer=True))

    ax.set_xlabel('Batch Size / GPU', labelpad=X_LABEL_PAD)
    ax.set_ylabel('')
    style_axis(ax)
    ax.tick_params(axis='x', pad=X_TICK_PAD)
    ax.tick_params(axis='y', which='both', left=True, labelleft=True, right=False, labelright=False, pad=1.0)
    ax.legend(
        [line_dispatch_combine, line_dispatch, line_combine],
        ['Disp.+Comb.', 'Dispatch', 'Combine'],
        loc='upper left',
        frameon=False,
        ncol=1,
        handlelength=0.95,
        handletextpad=0.15,
        borderaxespad=0.2,
        columnspacing=0.01,
        labelspacing=0.15,
        # prop={'size': 8},
    )
    num_gpus = df.attrs.get('num_gpus')
    deepep_caption = (
        f'(b) DeepEP on {num_gpus} GPUs'
        if num_gpus is not None
        else '(b) DeepEP latency'
    )
    ax.text(
        0.0,
        CAPTION_Y,
        deepep_caption,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=CAPTION_SIZE,
        linespacing=0.80,
        multialignment='left',
    )


def build_figure(
    mla_df: pd.DataFrame,
    deepep_df: pd.DataFrame,
    output_dir: str,
    output_name: str,
    save_png: bool,
    smooth: bool,
    smooth_window: int,
):
    fig = plt.figure(figsize=FIGSIZE)
    grid = GridSpec(1, 2, figure=fig, wspace=0.13)

    ax_mla = fig.add_subplot(grid[0, 0])
    ax_deepep = fig.add_subplot(grid[0, 1])

    plot_mla(ax_mla, mla_df, smooth=smooth, smooth_window=smooth_window)
    plot_deepep(ax_deepep, deepep_df, smooth=smooth, smooth_window=smooth_window)

    fig.subplots_adjust(left=0.11, right=0.955, top=0.95, bottom=0.29)

    out_base = Path(output_dir) / output_name
    pdf_path = f'{out_base}.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.0)

    saved_paths = [pdf_path]
    if save_png:
        png_path = f'{out_base}.png'
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        saved_paths.append(png_path)

    plt.close(fig)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot MLA and DeepEP varlen latency figures in a single compact figure.'
    )
    parser.add_argument(
        '--mla-data-dir',
        type=str,
        default=str(PAPER_RESULT_DIR),
        help='Directory with MLA benchmark CSV files.',
    )
    parser.add_argument(
        '--mla-pattern',
        type=str,
        default='latest',
        help='Glob pattern for MLA CSV files, or "latest" for the newest MLA CSV in --mla-data-dir.',
    )
    parser.add_argument(
        '--deepep-data-dir',
        type=str,
        default=str(REFERENCE_DIR / 'deepep'),
        help='Directory with DeepEP summary CSV files.',
    )
    parser.add_argument(
        '--deepep-pattern',
        type=str,
        default='node*_summary_rank*.csv',
        help='Glob pattern for DeepEP CSV files.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(SCRIPT_DIR),
        help='Output directory for the figure.',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='fig3',
        help='Output filename prefix without suffix.',
    )
    parser.add_argument(
        '--save-png',
        action='store_true',
        help='Also save a PNG copy.',
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable smoothing on both panels.',
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=3,
        help='Centered moving-average window size, default=3.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mla_df = load_mla_data(os.path.abspath(args.mla_data_dir), args.mla_pattern)
    deepep_df = load_deepep_data(os.path.abspath(args.deepep_data_dir), args.deepep_pattern)
    saved_paths = build_figure(
        mla_df=mla_df,
        deepep_df=deepep_df,
        output_dir=output_dir,
        output_name=args.output_name,
        save_png=args.save_png,
        smooth=not args.no_smooth,
        smooth_window=max(1, args.smooth_window),
    )
    for path in saved_paths:
        print(f'Saved plot: {path}')


if __name__ == '__main__':
    main()

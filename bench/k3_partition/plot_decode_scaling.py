#!/usr/bin/env python3
"""Rebuild section 7.2 figures from measured component CSVs (no GPU required)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'bench/k3_partition/results'
COLORS = {1: '#64748b', 2: '#d97706', 4: '#2563eb', 8: '#059669', 16: '#9333ea'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})

def read(path):
    with (RESULTS / path).open() as f:
        return list(csv.DictReader(f))

def maximum(rows, field='graph_ms', **filters):
    values = [float(r[field]) for r in rows
              if all(r[k] == str(v) for k, v in filters.items()) and r[field]]
    if not values:
        raise ValueError(filters)
    return max(values)

def style(ax, x, labels=None):
    ax.set_xscale('log', base=2)
    ax.set_xticks(x, labels or [str(v) for v in x])
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=10)

def save(fig, name, title, footer):
    fig.suptitle(title, fontsize=15, fontweight='bold', y=.98)
    fig.text(.5, .02, footer, ha='center', va='bottom', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .12, 1, .88))
    for folder in ('docs/blogs/assets', 'docs/site/assets'):
        path = ROOT / folder / (name + '.svg')
        fig.savefig(path, bbox_inches='tight')
        path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines()) + '\n')
    fig.savefig('/tmp/' + name + '.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

rows = read('gb200_graph_kda/kda.csv')
x = [1, 8, 32, 128, 256]
fig, ax = plt.subplots(figsize=(10.5, 4.9)); ax2=ax.twinx()
for context,color in zip((8192,131072,1048576),('#2563eb','#059669','#9333ea')):
    y=[maximum(rows,tp=1,phase='decode',batch_per_dp=b) for b in x]
    label=f'L={context//1024}K' if context<1048576 else 'L=1M'
    ax.plot(x,y,'o-',color=color,lw=2.2,ms=5,label=f'{label} latency')
    ax2.plot(x,[bb/t for bb,t in zip(x,y)],'s--',color=color,alpha=.7,lw=1.8,ms=4,label=f'{label} throughput')
style(ax,x); ax.set(xlabel='Local batch size B (sequences per card)',ylabel='Complete KDA layer latency (ms)')
ax2.set_ylabel('Layer throughput (thousand tokens/s)'); ax2.set_ylim(bottom=0); ax2.spines['top'].set_visible(False)
h,l=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();fig.subplots_adjust(right=.72);ax.legend(h+h2,l+l2,ncol=1,loc='center left',bbox_to_anchor=(1.28,.5),borderaxespad=0.)
save(fig,'gb200-kda-decode-batch','GB200 single card · KDA Decode: batch and context','TP1; context curves overlap because KDA recurrent state is fixed. Left axis latency, right axis throughput; all y axes start at zero.')

rows = read('gb200_mla/mla.csv')
x = [1, 8, 32]
contexts = [1024, 8192, 32768, 131072, 524288, 1048576]
fig, ax = plt.subplots(figsize=(10.5, 4.9))
for total, color in zip((32768, 131072, 1048576), ('#2563eb', '#059669', '#9333ea')):
    y=[]
    for batch in x:
        target=total//batch
        context=min(contexts, key=lambda c: abs(c-target))
        y.append(maximum(rows, tp=1, phase='decode', cache_dtype='fp8_e4m3', context=context, batch_per_dp=batch))
    ax.plot(x, y, 'o-', color=color, label=f'Total context B×L={total//1024}K', lw=2.3, ms=5)
style(ax, x, ['B=1', 'B=8', 'B=32'])
ax.set(xlabel='Local batch size B (sequences per card)', ylabel='Complete MLA layer latency (ms)')
ax.legend(loc='center left', bbox_to_anchor=(1.02, .5))
save(fig, 'gb200-mla-decode-context', 'GB200 single card · MLA Decode at fixed total context',
     'TP1, real-weight complete MLA layer, FP8 KV, CUDA Graph; each point uses L≈(B×L)/B from the measured context grid. All y axes start at zero.')

rows2 = rows
fig, ax = plt.subplots(figsize=(10.5, 4.9))
for context, color in zip((8192, 131072, 1048576), ('#2563eb', '#059669', '#9333ea')):
    y = [maximum(rows2, tp=1, phase='decode', cache_dtype='fp8_e4m3', context=context, batch_per_dp=batch) for batch in (1, 8, 32)]
    label = f'L={context//1024}K' if context < 1048576 else 'L=1M'
    ax.plot((1, 8, 32), y, 'o-', color=color, label=label, lw=2.3, ms=5)
style(ax, (1, 8, 32), ['B=1', 'B=8', 'B=32'])
ax.set(xlabel='Local batch size B (sequences per card)', ylabel='Complete MLA layer latency (ms)')
ax.legend(loc='center left', bbox_to_anchor=(1.02, .5))
save(fig, 'gb200-mla-decode-batch-context', 'GB200 single card · MLA Decode latency at fixed context',
     'TP1, real-weight complete MLA layer, FP8 KV, CUDA Graph; latency only. Legend is outside the plotting area; all y axes start at zero.')

rows = read('gb200_real_moe/real_moe.csv')
x = [16, 128, 2048, 8192]
fig, ax = plt.subplots(figsize=(10.5, 4.9)); ax2=ax.twinx()
y=[maximum(rows,field='graph_ms',ep=4,sources=4,total_tokens_per_ep=b) for b in x]
ax.plot(x,y,'o-',color='#2563eb',lw=2.4,ms=6,label='Latency')
ax2.plot(x,[b/t for b,t in zip(x,y)],'s--',color='#059669',lw=2.2,ms=5,label='Throughput')
style(ax,x,['16','128','2K','8K']); ax.set(xlabel='Token batch per EP group',ylabel='Complete FFN layer latency (ms)')
ax2.set_ylabel('Layer throughput (thousand tokens/s)'); ax2.set_ylim(bottom=0); ax2.spines['top'].set_visible(False)
h,l=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();ax.legend(h+h2,l+l2,loc='upper left')
save(fig,'gb200-moe-decode-batch','GB200 single card proxy · MoE batch scaling','EP4 proxy, synthetic normal inputs; complete FFN CUDA Graph. Left axis latency, right axis throughput; no routing-imbalance claim.')

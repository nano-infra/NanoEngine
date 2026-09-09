#!/usr/bin/env python3
"""Plot GB200 KDA 16K marginal/cumulative model from measured chunk time."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
rows=list(csv.DictReader((ROOT/'results/gb200_graph_kda/kda.csv').open()))
vals=[float(r['ms']) for r in rows if r['tp']=='1' and r['phase']=='prefill' and r['batch_per_dp']=='1' and r['chunk']=='16384']
m=max(vals) # critical-rank latency
n=64;x=list(range(1,n+1));ctx=[i*16 for i in x];cum=[m*i/1000 for i in x]
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(11.5,5.6))
bars=ax.bar(x,[m]*n,color='#16a34a',alpha=.82,width=.82,label=f'Measured 16K chunk (max rank: {m:.2f} ms)')
ax.set_xlabel('16K chunk index (visible context at chunk end)')
ax.set_ylabel('Marginal KDA Prefill time (ms)',color='#15803d');ax.tick_params(axis='y',labelcolor='#15803d')
ax.set_xlim(.2,64.8);ax.set_ylim(0,m*1.35);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
tickpos=[1,8,16,24,32,40,48,56,64]
ax.set_xticks(tickpos,[f'{ctx[i-1]}K' for i in tickpos])
ax2=ax.twinx();line,=ax2.plot(x,cum,color='#ea580c',lw=2.8,marker='o',markersize=3.2,markevery=4,label='Cumulative discrete sum')
ax2.set_ylabel('Cumulative KDA Prefill time (s)',color='#c2410c');ax2.tick_params(axis='y',labelcolor='#c2410c')
ax.legend([bars,line],[bars.get_label(),line.get_label()],loc='upper left',frameon=False)
ax.set_title('GB200 single-card KDA layer · 16K chunks')
fig.text(.5,.01,f'TP1, batch 1, max-rank measured chunk time from gb200_graph_kda; KDA recurrent state makes marginal cost context-independent, so cumulative values are i × {m:.2f} ms. This is a layer model, not full-model TTFT.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.07,1,.96))
for p in [ROOT/'results/gb200_graph_kda/kda_16k_accumulation.svg',Path('docs/blogs/assets/gb200-kda-16k-accumulation.svg')]:
 p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,bbox_inches='tight')
print('chunk_ms',m,'cumulative_1m_s',cum[-1])

#!/usr/bin/env python3
"""Plot measured GB200 single-card 16K-chunk MLA accumulation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).parent
rows=list(csv.DictReader((ROOT/'results/gb200_single_card_trace/trace.csv').open()))
rows.sort(key=lambda r:int(r['chunk_index']))
x=[int(r['chunk_index']) for r in rows]
ctx=[int(r['visible_context'])/1024 for r in rows]
marginal=[float(r['marginal_attention_ms']) for r in rows]
kda_rows=[float(r['ms']) for r in csv.DictReader((ROOT/'results/gb200_graph_kda/kda.csv').open()) if r['tp']=='1' and r['phase']=='prefill' and r['batch_per_dp']=='1' and r['chunk']=='16384']
kda_ms=max(kda_rows)
cumulative=[float(r['cumulative_attention_ms'])/1000 for r in rows]
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(11.5,5.6))
bars=ax.bar(x,marginal,color='#2563eb',alpha=.82,width=.82,label='Marginal time of this 16K chunk')
ax.set_xlabel('16K chunk index (visible context at chunk end)')
ax.set_ylabel('Marginal MLA layer time (ms)',color='#1d4ed8')
ax.tick_params(axis='y',labelcolor='#1d4ed8');ax.axhline(kda_ms,color='#dc2626',lw=2,ls='--',label=f'KDA 16K chunk baseline ({kda_ms:.2f} ms)')
ax.axvline(2,color='#dc2626',lw=1.4,ls=':',alpha=.8)
ax.annotate('KDA becomes faster\nfrom next chunk',xy=(2,kda_ms),xytext=(7,kda_ms*4.2),arrowprops=dict(arrowstyle='->',color='#dc2626'),color='#b91c1c',fontsize=9,ha='left')
ax.set_ylim(0,max(marginal+[kda_ms])*1.12);ax.set_xlim(.2,64.8);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
ax.set_xticks([1,8,16,24,32,40,48,56,64],[f'{v}K' for v in (16,128,256,384,512,640,768,896,1024)])
ax2=ax.twinx();ax2.plot(x,cumulative,color='#9333ea',lw=2.8,marker='o',markersize=3.2,markevery=4,label='Cumulative time from empty cache')
ax2.fill_between(x,cumulative,color='#c4b5fd',alpha=.17);ax2.set_ylabel('Cumulative MLA layer time (s)',color='#7e22ce');ax2.set_ylim(0,max(cumulative)*1.1);ax2.tick_params(axis='y',labelcolor='#7e22ce')
handles=[bars,ax.lines[-1],ax2.lines[0]];ax.legend(handles,[h.get_label() for h in handles],loc='upper left',frameon=False)
ax.set_title('GB200 single-card MLA layer · measured 16K chunks from empty cache')
fig.text(.5,.01,'TP1, MLA layer 3, raw FP8 KV cache, 1,048,576-token endpoint. Each bar is one sequential chunk; purple curve is the direct cumulative sum. This is a representative layer trace, not full-model serving TTFT.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.07,1,.96))
for p in [ROOT/'results/gb200_single_card_trace/mla_16k_accumulation.svg',Path('docs/site/assets/gb200-mla-16k-accumulation.svg')]:p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,bbox_inches='tight')
fig.savefig('/tmp/gb200-mla-16k-accumulation.png',dpi=140,bbox_inches='tight')
print('first marginal ms',marginal[0],'last marginal ms',marginal[-1],'cumulative sec',cumulative[-1])

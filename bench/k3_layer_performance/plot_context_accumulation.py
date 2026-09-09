#!/usr/bin/env python3
"""Plot single-card MLA marginal and cumulative context cost."""
from pathlib import Path
import csv, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
rows=[]
for p in (ROOT/'results').glob('mla_context_*_chunk_16384.csv'):
    rows.extend(csv.DictReader(p.open()))
rows=sorted(rows,key=lambda r:int(r['context']))
# The source rows are measured complete-layer forwards at one final chunk.
xs=[int(r['context']) for r in rows]
ys=[float(r['kernel_pipeline_ms']) for r in rows]
# Interpolate marginal cost in log-context between measured points. This is a
# reconstruction of the per-chunk curve, not a replacement for a trace.
points=[]
for l in range(32768,1048577,16384):
    if l in xs: y=ys[xs.index(l)]
    else:
        j=max(i for i,x in enumerate(xs) if x<l); k=min(i for i,x in enumerate(xs) if x>l)
        u=(math.log2(l)-math.log2(xs[j]))/(math.log2(xs[k])-math.log2(xs[j]))
        y=ys[j]+u*(ys[k]-ys[j])
    points.append((l,y))
cum=[];total=0
for l,y in points:
    total+=y;cum.append(total)
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(12.8,4.8))
# Marginal panel: measured points and interpolation are deliberately distinct.
ax[0].plot([x/1024 for x,y in points],[y for x,y in points],color='#2563eb',lw=1.8,label='Interpolated marginal chunk time')
ax[0].scatter([x/1024 for x in xs],[y for y in ys],color='#dc2626',s=34,zorder=3,label='Measured final-chunk points')
ax[0].set(xscale='log',xlabel='Visible context at chunk end (Ki tokens)',ylabel='Attention pipeline time per 16K chunk (ms)',title='(a) Marginal attention time')
ax[0].grid(alpha=.22,which='both');ax[0].legend(fontsize=8,frameon=False)
# Cumulative panel uses its own color family, not panel-a's blue/red.
ax[1].plot([x/1024 for x,y in points],cum,color='#7c3aed',lw=2.2,label='Discrete cumulative sum')
ax[1].fill_between([x/1024 for x,y in points],cum,color='#c4b5fd',alpha=.25)
ax[1].set(xscale='log',xlabel='Final context (Ki tokens)',ylabel='Cumulative attention pipeline time (s)',title='(b) From empty cache to final context')
ax[1].yaxis.set_major_formatter(lambda x,pos:f'{x/1000:g}')
ax[1].grid(alpha=.22,which='both');ax[1].legend(fontsize=8,frameon=False)
fig.suptitle('GB200-only planning view · MLA layer 3 · 16K fresh chunk · raw context sweep data from one B300 card',fontsize=12)
fig.text(.5,.01,'Red points are measured B300 layer-3 pipeline points; blue/purple curves reconstruct intermediate chunks by log-context interpolation. Validate with a per-chunk GB200 trace before using for SLOs.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.07,1,.93))
for path in [ROOT/'results/mla-context-accumulation-gb200.svg',Path('docs/blogs/assets/mla-context-accumulation-gb200.svg')]:
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,bbox_inches='tight')
fig.savefig('/tmp/mla-context-accumulation-gb200.png',dpi=140,bbox_inches='tight')
print('Measured points:',list(zip(xs,ys)))
print('Cumulative estimate seconds:',[(l,c/1000) for (l,_),c in zip(points,cum) if l in (32768,131072,262144,524288,1048576)])

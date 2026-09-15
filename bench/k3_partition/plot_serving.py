#!/usr/bin/env python3
"""Standalone report figure from the measured complete MLA and paged CP data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bench.k3_partition.summarize_extended import read, maximum


def main():
    mla=read('gb200_mla/mla.csv');cp=read('gb200_mla_final/paged_cp.csv')
    before=read('gb200_mla_before/mla.csv')
    lengths=[1024,8192,32768,131072,524288,1048576]
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    colors={1:'#64748b',2:'#d97706',4:'#16a34a',8:'#2563eb',16:'#9333ea'}
    for tp in [4,8,16]:
        axes[0,0].plot([L/1024 for L in lengths],[maximum(mla,'graph_ms',tp=tp,phase='decode',cache_dtype='fp8_e4m3',context=L,batch_per_dp=1)*1000 for L in lengths],'o-',color=colors[tp],label=f'TP{tp}')
    axes[0,0].set(xscale='log',xlabel='Visible context (Ki tokens)',ylabel='Graph latency (µs)',title='(a) Complete MLA Decode: one local request')
    axes[0,0].legend()
    for tp in [4,8,16]:
        axes[0,1].plot([L/1024 for L in lengths[1:]],[maximum(mla,'eager_ms',tp=tp,phase='prefill',cache_dtype='fp8_e4m3',context=L,chunk=8192) for L in lengths[1:]],'o-',color=colors[tp],label=f'TP{tp}')
    axes[0,1].set(xscale='log',xlabel='Final visible context (Ki tokens)',ylabel='Eager latency (ms)',title='(b) Complete MLA Prefill: final 8K chunk')
    axes[0,1].legend()
    for p in [1,2,4,8]:
        axes[1,0].plot([L/1024 for L in lengths],[maximum(cp,'graph_ms',projection_tp=8,nested_cp=p,batch_per_dp=8,context=L,cache_dtype='torch.float8_e4m3fn')*1000 for L in lengths],'o-',color=colors[p],label=f'CP{p}')
    axes[1,0].set(xscale='log',xlabel='Visible context (Ki tokens)',ylabel='Graph latency (µs)',title='(c) Paged MLA CP: TP8, local batch 8; includes Q exchange')
    axes[1,0].legend()
    labels=['TP8','TP16'];xs=[0,1]
    bv=[maximum(before,'dynamic_peak_bytes',tp=tp,phase='prefill',cache_dtype='bfloat16',context=1048576,chunk=8192)/2**30 for tp in [8,16]]
    av=[maximum(mla,'dynamic_peak_bytes',tp=tp,phase='prefill',cache_dtype='bfloat16',context=1048576,chunk=8192)/2**30 for tp in [8,16]]
    axes[1,1].bar([x-.18 for x in xs],bv,width=.36,color='#94a3b8',label='Whole-prefix expansion')
    axes[1,1].bar([x+.18 for x in xs],av,width=.36,color='#2563eb',label='Bounded prefix expansion')
    for i in xs:
        axes[1,1].text(i-.18,bv[i]+.3,f'{bv[i]:.2f}',ha='center')
        axes[1,1].text(i+.18,av[i]+.3,f'{av[i]:.2f}',ha='center')
    axes[1,1].set(xticks=xs,xticklabels=labels,ylabel='Incremental peak (GiB)',ylim=(0,24),title='(d) BF16 MLA Prefill transient: 1M context / 8K chunk')
    axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GB200: phase, context length and conversion cost change the best layout\nPanels a–c: FP8 cache. Real-weight MLA layer; CP is a component prototype',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    root=Path(__file__).resolve().parents[2]
    fig.savefig(root/'docs/site/assets/k3-gb200-serving.svg',bbox_inches='tight')
    fig.savefig('/tmp/k3-gb200-serving.png',dpi=140,bbox_inches='tight')

if __name__=='__main__':main()

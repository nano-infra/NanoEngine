#!/usr/bin/env python3
"""Render the additional measured evidence and hardware/context planning tables."""
import csv
from pathlib import Path
import json

ROOT=Path(__file__).parent
RESULTS=ROOT/'results'
DOC=ROOT.parents[1]/'docs/blogs/dlengine-kda-evaluation.md'
def read(path):return list(csv.DictReader((RESULTS/path).open()))
def select(rows,**fields):return [r for r in rows if all(r[k]==str(v) for k,v in fields.items())]
def maximum(rows,field,**filters):return max(float(x[field]) for x in select(rows,**filters))
def table(headers, rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def replace(s,name,value):
 begin=f'<!-- BEGIN {name} -->';end=f'<!-- END {name} -->'
 a=s.index(begin)+len(begin);b=s.index(end,a)
 return s[:a]+'\n'+value+'\n'+s[b:]


def main():
 s=DOC.read_text()
 mla=read('gb200_mla/mla.csv');cp=read('gb200_mla_final/paged_cp.csv')
 cold=read('gb200_cold/cold.csv');moe=read('gb200_real_moe/real_moe.csv')
 cap=read('serving_model/capacity.csv');work=read('serving_model/work.csv')
 rows=[]
 for tp in (1,2,4,8,16):
  rows.append([tp,*[f"{maximum(mla,'graph_ms',tp=tp,phase='decode',cache_dtype='fp8_e4m3',context=L,batch_per_dp=1):.3f}" for L in (8192,131072,1048576)],f"{maximum(mla,'eager_ms',tp=tp,phase='prefill',cache_dtype='fp8_e4m3',context=1048576,chunk=8192):.2f}",f"{maximum(mla,'dynamic_peak_bytes',tp=tp,phase='prefill',cache_dtype='bfloat16',context=1048576,chunk=8192)/2**30:.2f}"])
 s=replace(s,'FULL_MLA_TABLE',table(['TP','8K Decode (ms)','128K Decode (ms)','1M Decode (ms)','1M/8K Prefill (ms)','BF16 Prefill transient (GiB)'],rows))
 rows=[]
 for tp in (4,8,16):
  for batch in (1,8,32):
   candidates=sorted({int(x['nested_cp']) for x in select(cp,projection_tp=tp)})
   for L in (8192,131072,1048576):
    values={p:maximum(cp,'graph_ms',projection_tp=tp,nested_cp=p,batch_per_dp=batch,context=L,cache_dtype='torch.float8_e4m3fn') for p in candidates}
    best=min(values,key=values.get)
    if tp==8 or (tp==16 and L==1048576):
     rows.append([tp,batch,f'{L//1024}K',f'{values[1]:.3f}',best,f'{values[best]:.3f}',f'{values[1]/values[best]:.2f}×'])
 s=replace(s,'PAGED_CP_TABLE',table(['Projection TP','Local batch','Context','CP1 (ms)','Best measured CP','Best (ms)','Ratio'],rows))
 rows=[]
 for world in (1,2,4,8,16):
  k=read(f'gb200_world{world}/kda.csv') if world<16 else read('gb200_graph_kda/kda.csv')
  rows.append([world,f"{maximum(k,'ms',tp=world,phase='prefill',chunk=8192):.3f}",*[f"{maximum(k,'graph_ms',tp=world,phase='decode',batch_per_dp=b):.3f}" for b in (1,8,128)]])
 s=replace(s,'WORLD_KDA_TABLE',table(['Actual world size = TP','8K Prefill (ms)','Decode B1 (ms)','Decode B8 (ms)','Decode B128 (ms)'],rows))
 rows=[]
 for ep in (4,8,16):
  rows.append([ep,*[f"{maximum(moe,'graph_ms',ep=ep,total_tokens_per_ep=n,sources=ep):.3f}" for n in (16,128,2048,8192)],f"{maximum(moe,'expert_max_mean_load',ep=ep,total_tokens_per_ep=8192,sources=ep):.2f}"])
 s=replace(s,'REAL_MOE_TABLE',table(['EP','16 inputs (ms)','128 inputs (ms)','2K inputs (ms)','8K inputs (ms)','8K expert max/mean'],rows))
 rows=[]
 for tp in (4,8,16):
  rows.append([tp,*[f"{maximum(cold,'cold_prefill_cuda_ms',tp=tp,context=L,chunk=8192,cache_dtype='fp8_e4m3')/1000:.3f}" for L in (8192,131072,1048576)]])
 s=replace(s,'COLD_TABLE',table(['TP','Cold 8K (s)','Cold 128K (s)','Cold 1M (s)'],rows))
 rows=[]
 for profile,world,tp in [('B200_180GB_budget',16,8),('B300_270GB_budget',8,8),('B300_288GB_budget',8,4),('B300_288GB_budget',8,8),('B300_270GB_budget',16,1),('B300_288GB_budget',16,1),('GB200_measured',16,4),('GB200_measured',16,8)]:
  row=select(cap,hardware=profile,world=world,tp=tp,nested_cp=1,cache_dtype='fp8_e4m3',context=1048576)[0]
  rows.append([profile.replace('_budget','').replace('_',' '),world,tp,f"{float(row['weight_gib']):.1f}",*[select(cap,hardware=profile,world=world,tp=tp,nested_cp=1,cache_dtype='fp8_e4m3',context=L)[0]['cluster_requests'] for L in (8192,131072,1048576)]])
 s=replace(s,'HW_CAPACITY_TABLE',table(['HBM planning profile','GPUs = EP','Attention TP','Weights (GiB/rank)','8K requests','128K requests','Maximum-length requests'],rows))
 rows=[]
 for r in work:
  L=int(r['context'])
  if L==1024:continue
  cold_work=sum(float(r[k]) for k in ('cold_fixed_pflop','cold_mla_attention_pflop','cold_repeated_expansion_pflop'))
  suffix=sum(float(r[k]) for k in ('suffix_fixed_pflop','suffix_mla_attention_pflop','suffix_expansion_pflop'))
  rows.append([f'{L//1024}K',r['cold_chunks'],f'{cold_work:.2f}',f'{suffix:.2f}',f"{(float(r['decode_fixed_gflop'])+float(r['decode_absorbed_mla_gflop']))/1000:.3f}",f"{float(r['decode_fp8_cache_gib']):.3f}"])
 s=replace(s,'PHASE_WORK_TABLE',table(['Context cap/endpoint','8K cold chunks','Cold work (PFLOP)','Final 8K chunk (PFLOP)','Decode work/token (TFLOP)','FP8 cache/request (GiB)'],rows))
 # Equal global populations, layer-only arithmetic sum; capacity retained in output.
 kda=read('gb200_graph_kda/kda.csv')+read('gb200_real_moe/equal_kda.csv')
 allmla=mla+read('gb200_equal_mla/equal_mla.csv')
 rows=[]
 for B in (8,32):
  for L in (8192,131072):
   for tp in (4,8,16):
    b=B//(16//tp)
    k=maximum(kda,'graph_ms',tp=tp,phase='decode',batch_per_dp=b)
    a=maximum(allmla,'graph_ms',tp=tp,phase='decode',batch_per_dp=b,context=L,cache_dtype='fp8_e4m3')
    count=int(select(cap,hardware='GB200_measured',world=16,tp=tp,nested_cp=1,cache_dtype='fp8_e4m3',context=L)[0]['cluster_requests'])
    rows.append([B,f'{L//1024}K',tp,b,f'{69*k:.2f}',f'{24*a:.2f}',f'{69*k+24*a:.2f}','yes' if B<=count else 'no'])
 s=replace(s,'EQUAL_LOAD_TABLE',table(['Global B','Context','TP','Local B','69 KDA (ms)','24 MLA (ms)','Attention sum (ms)','Pass memory screen'],rows))
 # Complete-prefill component budget uses matched eager collective references.
 comm=read('gb200/collectives.csv')
 rows=[]
 for tp in (4,8,16):
  k=69*maximum(kda,'ms',tp=tp,phase='prefill',chunk=8192)
  a=24*maximum(mla,'eager_ms',tp=tp,phase='prefill',chunk=8192,context=1048576,cache_dtype='fp8_e4m3')
  f=92*maximum(moe,'eager_ms',ep=16,total_tokens_per_ep=8192,sources=tp)
  delta=93*(maximum(comm,'median_ms',size=tp,requested_tokens=8192,op='rs_ag_pair')-maximum(comm,'median_ms',size=tp,requested_tokens=8192,op='all_reduce'))
  rows.append([tp,*[f'{v/1000:.3f}' for v in (k,a,f,delta,k+a+f+delta)]])
 s=replace(s,'COMPLETE_PREFILL_BUDGET',table(['TP / EP16','69 KDA (s)','24 complete MLA (s)','92 complete FFN (s)','Boundary correction (s)','Component estimate (s)'],rows))
 full=json.loads((RESULTS/'gb200_full_serving/result.json').read_text())
 rows=[]
 for v in full.get('workloads',[]):
  if v['status']=='completed':rows.append([v['prompt_tokens'],len(v['output_tokens']),f"{v['ttft_seconds']:.3f}",f"{v['mean_tpot_ms']:.2f}",f"{v['elapsed_seconds']:.3f}"])
 s=replace(s,'FULL_SERVING_TABLE',table(['Input tokens','Output tokens','TTFT (s)','Mean TPOT (ms)','Request wall time (s)'],rows))
 rows=[]
 for tp,name in [(4,'gb200_full_tp4'),(8,'gb200_full_serving'),(16,'gb200_full_tp16')]:
  path=RESULTS/name/'result.json'
  if not path.exists():continue
  v=json.loads(path.read_text());points={w['prompt_tokens']:w for w in v.get('workloads',[]) if w['status']=='completed'}
  if 1048560 not in points:continue
  rows.append([tp,v['config']['gpu_memory_utilization'],v.get('effective_max_model_len'),f"{points[131072]['ttft_seconds']:.2f}" if 131072 in points else '—',f"{points[1048560]['ttft_seconds']:.2f}",f"{points[1048560]['mean_tpot_ms']:.2f}"])
 s=replace(s,'FULL_SERVING_LAYOUT_TABLE',table(['Attention TP / EP16','KV utilization setting','Effective context cap','128K TTFT (s)','Near-cap TTFT (s)','Near-cap mean TPOT (ms)'],rows))
 expert_path=RESULTS/'gb200_expert_tp/expert_tp.csv'
 if expert_path.exists():
  experts=list(csv.DictReader(expert_path.open()))
  if all('routing' in r for r in experts):
   rows=[]
   for route in ('balanced','hot'):
    for tf in (1,2,4):
     rows.append([route,f'{16//tf} × {tf}',*[f"{maximum(experts,'graph_ms',expert_tp=tf,routing=route,global_tokens=n):.3f}" for n in (16,128,2048,8192)],f"{maximum(experts,'relative_l2_vs_ep16',expert_tp=tf,routing=route):.5f}"])
   s=replace(s,'EXPERT_TP_TABLE',table(['Routing','EP × expert TP','16 inputs (ms)','128 inputs (ms)','2K inputs (ms)','8K inputs (ms)','Max relative L2'],rows))
 DOC.write_text(s)

if __name__=='__main__':main()

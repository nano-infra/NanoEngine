#!/usr/bin/env python3
import csv, torch
from pathlib import Path
import torch.nn.functional as F
from dlengine.runtime.kernel.triton.generic.k3_causal_conv1d_prefill import causal_conv1d_fn

D=12288; WIDTH=4

def time(fn,warmup=3,repeats=15):
 for _ in range(warmup): fn()
 torch.cuda.synchronize(); vals=[]; a=torch.cuda.Event(True); b=torch.cuda.Event(True)
 for _ in range(repeats):
  a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b))
 return sorted(vals)[len(vals)//2]

rows=[]
for n in [1024,2048,4096,8192,16384]:
 parts=[torch.zeros(n,D,device='cuda',dtype=torch.bfloat16).T for _ in range(3)]
 weights=[torch.zeros(D,WIDTH,device='cuda',dtype=torch.float32) for _ in range(3)]
 current_x=torch.cat(parts,dim=0).unsqueeze(0)
 current_w=torch.cat(weights,dim=0).unsqueeze(1)
 qstart=torch.tensor([0,n],device='cuda',dtype=torch.int32)
 cache_idx=torch.tensor([0],device='cuda',dtype=torch.int32)
 has_state=torch.tensor([True],device='cuda',dtype=torch.bool)
 states=[torch.zeros(1,D,WIDTH-1,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 def old():
  padded=torch.cat((torch.cat(states,dim=1),current_x),dim=-1)
  return F.silu(F.conv1d(padded.float(),current_w,groups=3*D)).to(torch.bfloat16)
 def new():
  return [causal_conv1d_fn(x,w,None,conv_states=st,query_start_loc=qstart,cache_indices=cache_idx,has_initial_state=has_state,activation='silu',seq_lens_cpu=[n]) for x,w,st in zip(parts,weights,states)]
 old_ms=time(old); new_ms=time(new)
 row={'chunk':n,'current_padded_conv_ms':old_ms,'ragged_triton_ms':new_ms,'speedup':old_ms/new_ms}
 rows.append(row); print(row,flush=True)
out=Path('bench/k3_layer_performance/results/kda_conv_ab.csv')
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

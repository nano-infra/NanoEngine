#!/usr/bin/env python3
"""Single-GB200 MLA layer trace: 16K chunks from empty cache to 1M."""
from pathlib import Path
import csv,time,json,os
import torch, torch.distributed as dist
from types import SimpleNamespace

MODEL=Path('/hgpfs/Kimi-K3'); OUT=Path('bench/k3_partition/results/gb200_single_card_trace');OUT.mkdir(parents=True,exist_ok=True)

def main():
    os.environ.setdefault('MASTER_ADDR','127.0.0.1');os.environ.setdefault('MASTER_PORT','29671')
    dist.init_process_group('nccl',rank=0,world_size=1,device_id=torch.cuda.current_device())
    from bench.k3_partition.extended_kernels import setup
    from examples.kimi_k3_fp8_kv_validation import load_attention,allocate_cache
    from dlengine.runtime.context.batch import set_batch_context,reset_batch_context
    import torch.nn.functional as F
    worker=SimpleNamespace(rank=0,world=1,groups={1:dist.group.WORLD})
    hf,index=setup(worker,1)
    attention,norm=load_attention(MODEL,hf,index,3)
    length,chunk,dtype=1048576,16384,'fp8_e4m3';pages=length//64
    cache=allocate_cache(hf,dtype,pages);attention.attn_fwd.k_cache=cache
    bt=torch.arange(pages-1,-1,-1,device='cuda',dtype=torch.int32).reshape(1,1,pages)
    torch.manual_seed(8654);hidden=F.rms_norm(torch.randn(chunk,hf.hidden_size,device='cuda',dtype=torch.bfloat16),(hf.hidden_size,),norm,hf.rms_norm_eps)
    def step(start,end):
        pos=torch.arange(start,end,device='cuda',dtype=torch.int64);slots=bt.flatten()[pos//64].long()*64+pos%64
        set_batch_context(is_prefill=True,cu_seqlens_q=torch.tensor([0,end-start],device='cuda',dtype=torch.int32),cu_seqlens_k=torch.tensor([0,end],device='cuda',dtype=torch.int32),max_seqlen_q=end-start,max_seqlen_k=end,block_tables=bt,slot_mapping=slots)
        return attention(pos,hidden[:end-start])
    # Compile/warm every context shape once, then clear the cache. The first
    # prefix shape can include multi-second CuTe JIT and must not pollute the
    # marginal context-scaling trace.
    for warm_start in range(0,length,chunk):
        step(warm_start,min(warm_start+chunk,length))
    torch.cuda.synchronize();cache.zero_();torch.cuda.synchronize()
    rows=[];cum=0.;cumwall=0.
    for i,start in enumerate(range(0,length,chunk),1):
        end=min(start+chunk,length);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);t=time.perf_counter();a.record();out=step(start,end);b.record();b.synchronize();ms=a.elapsed_time(b);wall=(time.perf_counter()-t)*1000;cum+=ms;cumwall+=wall
        row=dict(card='GB200',tp=1,layer=3,cache_dtype=dtype,chunk_tokens=end-start,chunk_index=i,visible_context=end,marginal_attention_ms=ms,marginal_wall_ms=wall,cumulative_attention_ms=cum,cumulative_wall_ms=cumwall)
        rows.append(row)
        if i%8==0 or i==1:print('TRACE',row,flush=True)
    assert torch.isfinite(out).all().item()
    with (OUT/'trace.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'metadata.json').write_text(json.dumps(dict(card='GB200',gpu=torch.cuda.get_device_name(),layer=3,cache_dtype=dtype,chunk_tokens=chunk,max_context=length,trace_kind='measured single-card MLA layer from empty cache'),indent=2)+'\n')
    reset_batch_context();dist.destroy_process_group();print('Saved',OUT/'trace.csv',flush=True)
if __name__=='__main__':main()

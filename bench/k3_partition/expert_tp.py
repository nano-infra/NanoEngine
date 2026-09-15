"""Native MXFP4 expert-TP prototype using partitioned MegaMoE intermediates."""
from pathlib import Path
import csv
import re
import torch
import torch.distributed as dist


@torch.inference_mode()
def measure_expert_tp(worker):
    from safetensors import safe_open
    from bench.k3_partition.extended_kernels import MODEL, setup
    from dlengine.runtime.layers.backends.experts.mega_moe import MegaMoEExperts
    from dlengine.runtime.models.quant_config import QuantizationConfig
    from dlengine.runtime.runner.runner_config import set_runner_config
    import torch.distributed._symmetric_memory as symm_mem
    symm_mem.set_backend('NCCL')
    hf,index=setup(worker,1)
    quant=QuantizationConfig(**hf.quantization_config)
    set_runner_config(mega_moe_max_tokens_per_rank=2048)
    rows=[];references={}
    outdir=Path(__file__).parent/'results/gb200_expert_tp_progress';outdir.mkdir(exist_ok=True)
    for tf in (1,2,4):
        ep=16//tf;erank=worker.rank%ep;trank=worker.rank//ep
        if tf>1:
            for e in range(ep):
                group=dist.new_group([e+t*ep for t in range(tf)])
                if e==erank:tpgroup=group
        else:tpgroup=worker.groups[1]
        logical_intermediate=3072//tf
        physical_intermediate=(logical_intermediate+511)//512*512
        with torch.device('cuda'):
            experts=MegaMoEExperts(hidden_size=3584,intermediate_size=physical_intermediate,
                num_experts=896,top_k=16,ep_size=ep,tp_size=1,
                ep_group=worker.groups[ep],quantization_config=quant)
        names={}
        for name,shard in index.items():
            m=re.match(r'language_model.model.layers.1.block_sparse_moe.experts.(\d+).(w[123]).(weight_packed|weight_scale)$',name)
            if m and erank*(896//ep)<=int(m[1])<(erank+1)*(896//ep):
                names.setdefault(shard,[]).append((name,m))
        for shard,entries in names.items():
            with safe_open(MODEL/shard,framework='pt',device='cpu') as f:
                for name,m in entries:
                    tensor=f.get_tensor(name)
                    axis=1 if m[2]=='w2' else 0
                    width=tensor.shape[axis]//tf
                    tensor=tensor.narrow(axis,trank*width,width)
                    if physical_intermediate!=logical_intermediate:
                        shape=list(tensor.shape)
                        shape[axis]=shape[axis]*physical_intermediate//logical_intermediate
                        padded=torch.full(shape,127 if m[3]=='weight_scale' else 0,dtype=tensor.dtype)
                        padded.narrow(axis,0,width).copy_(tensor)
                        tensor=padded
                    experts.load_expert_weight(int(m[1]),m[2],m[3],tensor,ep_rank=erank)
        experts.prepare_mega_weights()
        for total in (16,128,2048,8192):
            for routing in ('balanced', 'hot'):
                n=total//16
                torch.manual_seed(12345+worker.rank)
                x=torch.randn(n,3584,device='cuda',dtype=torch.bfloat16)*.1
                ids=(torch.arange(n*16,device='cuda',dtype=torch.int32).reshape(n,16)+worker.rank*n*16)%896
                if routing == 'hot': ids=torch.arange(16,device='cuda',dtype=torch.int32).expand(n,16).contiguous()
                scores=torch.full((n,16),1/16,device='cuda',dtype=torch.float32)
                gx=torch.empty(tf*n,3584,device='cuda',dtype=x.dtype)
                gi=torch.empty(tf*n,16,device='cuda',dtype=ids.dtype)
                gs=torch.empty(tf*n,16,device='cuda',dtype=scores.dtype)
                final=torch.empty_like(x)
                def complete():
                    if tf==1:return experts(x,ids,scores)
                    dist.all_gather_into_tensor(gx,x,group=tpgroup)
                    dist.all_gather_into_tensor(gi,ids,group=tpgroup)
                    dist.all_gather_into_tensor(gs,scores,group=tpgroup)
                    partial=experts(gx,gi,gs)
                    dist.reduce_scatter_tensor(final,partial,group=tpgroup)
                    return final
                actual=complete();torch.cuda.synchronize()
                if tf==1:references[(total,routing)]=actual.clone()
                expected=references[(total,routing)]
                err=((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)).item()
                assert torch.isfinite(actual).all().item() and err<.06,(tf,total,err)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):complete()
                stream.synchronize();graph=torch.cuda.CUDAGraph();dist.barrier()
                with torch.cuda.graph(graph,stream=stream):output=complete()
                dist.barrier();values=[worker.time(graph.replay,warmup=3,repeats=30) for _ in range(3)]
                row=dict(rank=worker.rank,ep=ep,expert_tp=tf,routing=routing,global_tokens=total,
                    logical_intermediate=logical_intermediate,physical_intermediate=physical_intermediate,original_rows_per_rank=n,rows_after_tp_gather=n*tf,graph_ms=sorted(values)[1],
                    relative_l2_vs_ep16=err,includes_tp_gather_and_reduce_scatter=True)
                rows.append(row)
                with (outdir/f'rank{worker.rank}.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows(rows)
                if worker.rank==0:print('EXPERT_TP',row,flush=True)
                del graph,output,actual,x,ids,scores,gx,gi,gs,final
        del experts
        torch.cuda.empty_cache()
    return rows

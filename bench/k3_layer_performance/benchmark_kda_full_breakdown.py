#!/usr/bin/env python3
"""Production K3 KDA Prefill operator and stage-latency breakdown."""
from __future__ import annotations
import argparse, csv, math, os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist


def median_ms(fn, restore, warmup, repeats):
    for _ in range(warmup):
        restore(); fn()
    torch.cuda.synchronize()
    values=[]
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        restore(); start.record(); fn(); end.record(); end.synchronize(); values.append(start.elapsed_time(end))
    return sorted(values)[len(values)//2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contexts', default='32768,131072,524288,1048576')
    p.add_argument('--chunks', default='1024,2048,4096,8192,16384')
    p.add_argument('--warmup', type=int, default=2); p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results/kda_full_prefill_breakdown.csv')
    a=p.parse_args(); contexts=[int(x) for x in a.contexts.split(',')]; chunks=[int(x) for x in a.chunks.split(',')]
    if not dist.is_initialized():
        dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ.get('LOCAL_RANK','0'))))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    from dlengine.runtime.context.distributed import set_dist_context
    from dlengine.runtime.context.batch import get_batch_context, set_batch_context
    from dlengine.runtime.layers import init_backend
    from dlengine.runtime.layers.backends.delta_net.kda import FlashInferKda
    set_dist_context(rank=0, world_size=1); init_backend()
    cfg=SimpleNamespace(model_type='kimi_k3', hidden_size=7168, v_head_dim=128, rms_norm_eps=1e-6,
        linear_attn_config={'num_heads':96,'head_dim':128,'value_head_dim':128,'short_conv_kernel_size':4,'gate_lower_bound':-5.0})
    old_device=torch.get_default_device(); old_dtype=torch.get_default_dtype()
    torch.set_default_device('cuda'); torch.set_default_dtype(torch.bfloat16)
    layer=FlashInferKda(0,0,cfg).eval()
    for param in layer.parameters():
        if param.device.type == 'cuda': param.data.zero_()
    torch.set_default_device(old_device); torch.set_default_dtype(old_dtype)
    H=96; D=128; conv_dim=3*H*D
    recurrent=torch.zeros(1,1,H,D,D,device='cuda',dtype=torch.bfloat16)
    conv_state=torch.zeros(1,1,conv_dim,4,device='cuda',dtype=torch.bfloat16)
    recurrent_base=recurrent.clone(); conv_base=conv_state.clone(); slots=torch.tensor([0],device='cuda',dtype=torch.int32)
    rows=[]
    for context_len in contexts:
      for chunk in chunks:
        if chunk>context_len: continue
        hidden=torch.zeros(chunk,7168,device='cuda',dtype=torch.bfloat16)
        cu_q=torch.tensor([0,chunk],device='cuda',dtype=torch.int32); cu_k=torch.tensor([0,context_len],device='cuda',dtype=torch.int32)
        set_batch_context(is_prefill=True,cu_seqlens_q=cu_q,cu_seqlens_k=cu_k,max_seqlen_q=chunk,max_seqlen_k=context_len,
            block_tables=torch.zeros(1,1,device='cuda',dtype=torch.int32),gdn_conv_states=conv_state,gdn_recurrent_states=recurrent,gdn_state_slots=slots)
        def restore(): recurrent.copy_(recurrent_base); conv_state.copy_(conv_base)
        full=median_ms(lambda: layer(hidden),restore,a.warmup,a.repeats)
        projected={}
        def projections():
            projected['q']=layer.q_proj(hidden); projected['k']=layer.k_proj(hidden); projected['v']=layer.v_proj(hidden)
            projected['gate']=layer.g_proj(hidden); projected['beta']=layer.b_proj(hidden)
            projected['forget']=layer.f_b_proj(layer.f_a_proj(hidden))
        projections(); torch.cuda.synchronize()
        proj_ms=median_ms(projections,lambda:None,a.warmup,a.repeats)
        conv_out={}
        def causal_conv():
            q, k, v = layer._prefill_causal_conv_qkv(
                projected['q'], projected['k'], projected['v'], get_batch_context()
            )
            conv_out.update(q=q, k=k, v=v)
        conv_ms=median_ms(causal_conv,restore,a.warmup,a.repeats)
        causal_conv(); torch.cuda.synchronize()
        q,k,v=conv_out['q'],conv_out['k'],conv_out['v']
        q=q.view(chunk,H,D); k=k.view(chunk,H,D); v=v.view(chunk,H,D)
        raw_g=projected['forget'].view(chunk,H,D); beta=projected['beta'].view(chunk,H).float().sigmoid()
        core={}
        def recurrence_fn():
            core['x']=layer._chunk_kda(q.unsqueeze(0).contiguous(),k.unsqueeze(0).contiguous(),v.unsqueeze(0).contiguous(),raw_g.unsqueeze(0).contiguous(),beta.unsqueeze(0).contiguous(),initial_state=recurrent[0],initial_state_indices=slots,use_qk_l2norm_in_kernel=True,cu_seqlens=cu_q,A_log=layer.A_log.float().contiguous(),dt_bias=layer.dt_bias.float().contiguous(),lower_bound=layer.lower_bound).squeeze(0)
        rec_ms=median_ms(recurrence_fn,restore,a.warmup,a.repeats); recurrence_fn(); torch.cuda.synchronize()
        normed={}; gate=projected['gate'].view(chunk,H,D)
        norm_ms=median_ms(lambda: normed.update(x=layer.o_norm(core['x'].reshape(chunk,H,D),gate).reshape(chunk,H*D)),lambda:None,a.warmup,a.repeats)
        layer.o_norm(core['x'].reshape(chunk,H,D),gate).reshape(chunk,H*D); torch.cuda.synchronize()
        norm_x=normed['x']
        out_ms=median_ms(lambda: layer.o_proj(norm_x),lambda:None,a.warmup,a.repeats)
        stage_sum=proj_ms+conv_ms+rec_ms+norm_ms+out_ms
        row={'context':context_len,'chunk':chunk,'full_ms':full,'input_projections_ms':proj_ms,'causal_conv_ms':conv_ms,'recurrence_ms':rec_ms,'gated_norm_ms':norm_ms,'output_projection_ms':out_ms,'stage_sum_ms':stage_sum}
        rows.append(row); print(row,flush=True)
        del hidden,q,k,v,raw_g,beta; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    dist.destroy_process_group()
if __name__=='__main__': main()

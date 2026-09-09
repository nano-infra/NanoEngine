"""Full-checkpoint Ray serving smoke and context sweep; records actual outcomes."""
import json
from pathlib import Path
import time
import traceback
import ray


def setup_worker():
    import torch
    import torch.distributed._symmetric_memory as symm_mem
    torch.set_num_threads(2)
    symm_mem.set_backend('NCCL')


def main():
    from dlengine.config import Config
    from dlengine.engine.llm_component import LLM
    from dlengine._rust.proto import SamplingParams
    from dlengine._rust.proto import RequestIn
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp',type=int,choices=[4,8,16],default=8)
    parser.add_argument('--memory-utilization',type=float,default=.88)
    parser.add_argument('--contexts',nargs='+',type=int,default=[5,1024,8192,32768,131072,524288,1048576-16])
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'results/gb200_full_serving')
    args=parser.parse_args()
    outdir=args.output_dir;outdir.mkdir(exist_ok=True)
    result={'model':'/hgpfs/Kimi-K3','status':'starting','workloads':[]}
    def save(): (outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    engine=None
    try:
        ray.init(address='auto',runtime_env={'env_vars':{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},'worker_process_setup_hook':setup_worker})
        cfg=Config(model='/hgpfs/Kimi-K3',attention_tp=args.tp,attention_dp=16//args.tp,ffn_ep=16,
            max_model_len=1048576,max_num_batched_tokens=8192,max_num_seqs=4,
            gpu_memory_utilization=args.memory_utilization,kv_cache_dtype='fp8_e4m3',hardware_backend='blackwell',
            trust_remote_code=True,enforce_eager=False)
        result['config']={k:getattr(cfg,k) for k in ['attention_tp','attention_dp','ffn_ep','max_model_len','max_num_batched_tokens','max_num_seqs','gpu_memory_utilization','kv_cache_dtype','enforce_eager']}
        save();t=time.perf_counter();engine=LLM(cfg)
        result['load_seconds']=time.perf_counter()-t
        result['effective_max_model_len']=engine.config.max_model_len
        result['kv_blocks']=engine.config.num_kvcache_blocks
        result['status']='loaded';save()
        prompt='The capital of France is'
        tokens=engine.tokenizer.encode(prompt)
        for length in args.contexts:
            if length+8>engine.config.max_model_len:
                result['workloads'].append({'prompt_tokens':length,'status':'exceeds effective capacity'});save();continue
            ids=(tokens*((length+len(tokens)-1)//len(tokens)))[:length]
            seqid=100000+length
            engine.add_request_payload(RequestIn(seqid,ids,SamplingParams(max_tokens=8,temperature=0.0,ignore_eos=True),0).to_bytes())
            t=time.perf_counter()
            progress=0;token_ids=[];raw_emissions=[];token_times=[];step_times=[]
            while not engine.is_finished():
                step_start=time.perf_counter();step=engine.step()
                now=time.perf_counter();progress+=step.prefill_tokens
                step_times.append(dict(prefill_tokens=step.prefill_tokens,decode_tokens=step.decode_tokens,wall_ms=(now-step_start)*1000))
                for event in step.outputs:
                    if event['seq_id']==seqid and event['num_tokens']>0:
                        raw_emissions.append(int(event['last_token']))
                        # The current offline helper also collects temporary
                        # logits emitted on intermediate prefill chunks. For
                        # this single-request harness, only the final prefill
                        # chunk and subsequent Decode steps are completions.
                        if progress>=length:
                            token_ids.append(int(event['last_token']));token_times.append(now-t)
            elapsed=time.perf_counter()-t
            selected={'token_ids':token_ids}
            assert len(token_ids)==8,(length,len(token_ids),progress)
            result['latest_step_trace']=step_times
            result['workloads'].append({'prompt_tokens':length,'output_tokens':selected['token_ids'],
                'output_text':engine.tokenizer.decode(selected['token_ids']), 'elapsed_seconds':elapsed,'ttft_seconds':token_times[0],
                'mean_tpot_ms':(token_times[-1]-token_times[0])*1000/(len(token_times)-1),
                'raw_emitted_token_count':len(raw_emissions),'filtered_intermediate_prefill_tokens':len(raw_emissions)-len(token_ids),
                'status':'completed'})
            save();print('FULL_SERVING',result['workloads'][-1],flush=True)
        result['status']='completed';save()
    except Exception as exc:
        result['status']='failed';result['error']=str(exc);result['traceback']=traceback.format_exc();save();raise
    finally:
        if engine is not None: engine.exit()
        ray.shutdown()

if __name__=='__main__':main()

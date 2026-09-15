"""Record installed TRTLLM-GEN MLA head-count support on one Ray GPU."""
import json
from pathlib import Path
import ray

@ray.remote(num_gpus=1, num_cpus=2)
def probe():
    import torch
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as mla
    torch.cuda.set_device(0)
    w = torch.zeros(512 * 2**20, dtype=torch.uint8, device='cuda')
    rows = []
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for heads in (6, 12, 16, 24, 32, 48, 64, 96, 128):
            for lse in (False, True):
                q = torch.zeros(1, 1, heads, 576, dtype=dtype, device='cuda')
                kv = torch.zeros(4, 1, 64, 576, dtype=dtype, device='cuda')
                bt = torch.arange(4, device='cuda', dtype=torch.int32).reshape(1, 4)
                lens = torch.tensor([133], device='cuda', dtype=torch.int32)
                try:
                    y = mla(q, kv, w, 128, 512, 64, bt, lens, 133, backend='trtllm-gen', return_lse=lse)
                    torch.cuda.synchronize()
                    status = 'pass'
                except Exception as exc:
                    status = str(exc)
                rows.append(dict(dtype=str(dtype), heads=heads, return_lse=lse, status=status))
    return rows

if __name__ == '__main__':
    ray.init(address='auto')
    rows = ray.get(probe.remote())
    p = Path(__file__).parent / 'results/gb200/mla_head_support.json'
    p.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows: print(r)
    ray.shutdown()

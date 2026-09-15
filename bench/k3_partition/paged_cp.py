"""Bench-only absorbed paged MLA DCP with explicit Q exchange and LSE merge."""
import torch
import torch.distributed as dist
from bench.k3_partition.cp_merge import ContextMerger


@torch.inference_mode()
def measure_paged_cp(worker):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    import csv
    from pathlib import Path
    outdir = Path(__file__).parent / 'results/gb200_paged_cp_progress'
    outdir.mkdir(exist_ok=True)
    workspace = torch.zeros(512 * 2**20, device='cuda', dtype=torch.uint8)
    rows = []
    for tp, cp in [(16, p) for p in (1, 2, 4, 8, 16)] + [(8, p) for p in (1, 2, 4, 8)] + [(4, p) for p in (1, 2, 4)]:
        group = worker.groups[cp]
        h0, heads = 96 // tp, 96 // tp * cp
        local_rank = dist.get_rank(group)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            def make(batch, length):
                pages = length // cp // 64
                torch.manual_seed(1256 + worker.rank)
                q = (torch.randn(batch, 1, h0, 576, device='cuda') * .1).to(dtype)
                gathered = torch.empty(cp, batch, 1, h0, 576, device='cuda', dtype=dtype)
                cache = (torch.randn(batch * pages, 1, 64, 576, device='cuda') * .1).to(dtype)
                table = torch.arange(batch * pages, device='cuda', dtype=torch.int32).reshape(batch, pages)
                lens = torch.full((batch,), length // cp, device='cuda', dtype=torch.int32)
                merger = ContextMerger(batch, heads, 512, group) if cp > 1 else None

                def exchange():
                    if cp == 1: return q
                    # Byte views allow FP8 communication through NCCL.
                    dist.all_gather_into_tensor(gathered.view(torch.uint8).flatten(), q.view(torch.uint8).flatten(), group=group)
                    return gathered.permute(1, 2, 0, 3, 4).reshape(batch, 1, heads, 576)

                def local(query):
                    if heads in (24, 96): query = torch.nn.functional.pad(query, (0, 0, 0, (32 if heads == 24 else 128) - heads))
                    result = trtllm_batch_decode_with_kv_cache_mla(query=query, kv_cache=cache,
                        workspace_buffer=workspace, qk_nope_head_dim=128, kv_lora_rank=512,
                        qk_rope_head_dim=64, block_tables=table, seq_lens=lens,
                        max_seq_len=length // cp, bmm1_scale=192 ** -.5,
                        bmm2_scale=1.0, backend='trtllm-gen', return_lse=cp > 1)
                    if cp > 1:
                        out, lse = result
                        return out[..., :heads, :].contiguous(), lse[..., :heads].contiguous()
                    return result[..., :heads, :].contiguous()

                def complete():
                    query = exchange()
                    value = local(query)
                    if cp == 1: return value.reshape(batch, h0, 512)
                    output, lse = value
                    merged = merger(output.reshape(batch, heads, 512), lse.reshape(batch, heads).T)
                    return merged[:, local_rank * h0:(local_rank + 1) * h0]
                return q, cache, exchange, local, complete

            # Independent small dense FP32 reference validates paging, query
            # ownership, natural-log LSE and correct context composition.
            q, cache, exchange, local, complete = make(2, cp * 128)
            result = complete()
            kvs = [torch.empty_like(cache) for _ in range(cp)]
            dist.all_gather([x.view(torch.uint8) for x in kvs], cache.view(torch.uint8), group=group)
            fullkv = torch.cat([x.reshape(2, 128, 576).float() for x in kvs], dim=1)
            scores = torch.einsum('bhd,bkd->bhk', q[:, 0].float(), fullkv) * 192 ** -.5
            ref = torch.einsum('bhk,bkv->bhv', scores.softmax(-1), fullkv[..., :512])
            error = (result.float() - ref).abs().max().item()
            torch.testing.assert_close(result.float(), ref, atol=.003, rtol=.05)
            del q, cache, exchange, local, complete, result, kvs, fullkv, scores, ref
            for length in (1024, 8192, 32768, 131072, 524288, 1048576):
                if length // cp < 128: continue
                for batch in (1, 8, 32):
                    q, cache, exchange, local, complete = make(batch, length)
                    complete()
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3): complete()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with torch.cuda.graph(graph, stream=stream): output = complete()
                    dist.barrier()
                    values = [worker.time(graph.replay, warmup=3, repeats=20) for _ in range(3)]
                    assert torch.isfinite(output).all().item()
                    row = dict(rank=worker.rank, projection_tp=tp, nested_cp=cp,
                        effective_head_tp=tp // cp, dp=16 // tp, batch_per_dp=batch,
                        context=length, cache_dtype=str(dtype), graph_ms=sorted(values)[1],
                        local_cache_bytes=cache.untyped_storage().nbytes(),
                        correctness_max_abs=error, includes_query_exchange=True)
                    rows.append(row)
                    with (outdir / f'rank{worker.rank}.csv').open('w') as f:
                        w = csv.DictWriter(f, fieldnames=row); w.writeheader(); w.writerows(rows)
                    if worker.rank == 0: print('PAGED_CP', row, flush=True)
                    del graph, output, q, cache, exchange, local, complete
                    torch.cuda.empty_cache()
    return rows

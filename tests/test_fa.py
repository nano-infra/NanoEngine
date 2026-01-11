import time

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
from triton.testing import do_bench, do_bench_cudagraph

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError:
    flash_mla_with_kvcache, get_mla_metadata = None, None


device = "cuda"
dtype = torch.bfloat16
seqlen_q = 1
nheads_q = 64

use_bench_cudagraph = True

attn_variant = "gqa"
print(f"仅测试: {attn_variant.upper()}")

nheads_kv = 4
headdim = 128
headdim_v = headdim
has_qv = False
page_size = 256

should_run_flashmla = False

torch.manual_seed(0)

batch_size = 1
cache_seqlens = None

print(
    f"\n{attn_variant.upper()}, nheads_q = {nheads_q}, nheads_kv = {nheads_kv}, headdim = {headdim}, headdim_v = {headdim_v}, page_size = {page_size}"
)

for seqlen in [2**i for i in range(10, 20)]:  # 2**10 到 2**19
    cache_seqlens = torch.tensor([seqlen] * batch_size, device=device, dtype=torch.int)
    num_splits = 0

    q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
    try:
        v_cache = torch.randn(
            batch_size, seqlen, nheads_kv, headdim_v, dtype=dtype, device=device
        )
        k_cache = torch.randn(
            batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device
        )

        if page_size is not None:
            assert seqlen % page_size == 0
            k_cache, v_cache = [
                rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
                for x in [k_cache, v_cache]
            ]
            page_table = rearrange(
                torch.arange(
                    batch_size * seqlen // page_size, device=device, dtype=torch.int32
                ),
                "(b s) -> b s",
                s=seqlen // page_size,
            )
        else:
            page_table = None
    except torch.OutOfMemoryError:
        print(f"序列长度 {seqlen} 导致内存不足，跳过...")
        continue

    qv = None  # GQA不使用qv

    fn0 = lambda: flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        num_splits=num_splits,
        qv=qv,
        page_table=page_table,
        causal=True,
    )

    time.sleep(1)

    if not use_bench_cudagraph:
        t0 = do_bench(fn0, warmup=100, rep=200)
    else:
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            t0 = do_bench_cudagraph(fn0, rep=200)

    total_seqlen = (
        seqlen * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
    )
    mem_io = (
        total_seqlen * nheads_kv * (headdim + headdim_v) * 2
        + q.numel() * 2
        + (qv.numel() * 2 if has_qv else 0)
        + q.numel() * headdim_v // headdim * 2
    )
    flops = (
        seqlen_q
        * total_seqlen
        * nheads_q
        * (headdim + headdim_v * (2 if has_qv else 1))
        * 2
    )
    ideal_h100_time_mem = mem_io / 3.35e12 * 1e6
    ideal_h100_time_flop = flops / 989e12 * 1e6
    ideal_h100_time = max(ideal_h100_time_mem, ideal_h100_time_flop)

    print(
        f"Seqlen = {seqlen}, FA3 time{'' if not use_bench_cudagraph else ' w CUDA Graph'}: {t0 * 1e3:.1f} us, "
        f"{mem_io * 1e-9 / (t0 * 1e-3):.0f} GB/s, {flops * 1e-12 / (t0 * 1e-3):.0f} TFLOPS/s"
    )
    print(f"Arithmetic intensity: {flops / mem_io:.1f}")
    print(f"Ideal time: {ideal_h100_time:.0f} us\n")

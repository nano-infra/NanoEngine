#!/usr/bin/env python3
"""Hardware/world-size/context capacity and phase-work model; no timing predictions."""
import csv
import json
from pathlib import Path
from bench.k3_partition.analyze_joint import weights, STATE, GIB, HBM

ROOT = Path(__file__).parent
CONTEXTS = (1024, 8192, 32768, 131072, 524288, 1048576)
# Product labels are decimal planning budgets, NOT measured CUDA-visible bytes.
HARDWARE = {'GB200_measured': HBM, 'B200_180GB_budget': 180 * 10**9,
            'B300_270GB_budget': 270 * 10**9, 'B300_288GB_budget': 288 * 10**9}
CACHE = {'bfloat16': 1152, 'fp8_e4m3': 576}
FIXED_GF = 69 * (0.887160832 + 0.006291456) + 24 * 0.464388096 + 92 * 1.436811264 + 1.453


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def main():
    out = ROOT / 'results/serving_model'
    out.mkdir(exist_ok=True)
    rows = []
    for hardware, hbm in HARDWARE.items():
        for world in (1, 2, 4, 8, 16, 32):
            for tp in (1, 2, 4, 8, 16, 32):
                if tp > world: continue
                for cp in (1, 2, 4, 8):
                    if tp % cp: continue
                    for dtype, row_bytes in CACHE.items():
                        for context in CONTEXTS:
                            w = weights(tp, tp, ep=world)
                            reserve = 20 * GIB
                            cache = 24 * row_bytes * context / cp + STATE / tp
                            slots = max(0, int((hbm - w - reserve) // cache))
                            rows.append(dict(hardware=hardware, hbm_budget_gib=hbm / GIB,
                                world=world, tp=tp, dp=world // tp, ep=world, ffn_tp=1,
                                nested_cp=cp, cache_dtype=dtype, context=context,
                                weight_gib=w / GIB, allowance_gib=20,
                                cache_state_per_request_gib=cache / GIB,
                                minimum_hbm_one_request_gib=(w + reserve + cache) / GIB,
                                requests_per_dp=slots, cluster_requests=slots * (world // tp),
                                resident_tokens=slots * (world // tp) * context,
                                implemented_mesh=cp == 1))
    write_csv(out / 'capacity.csv', rows)
    work = []
    for length in CONTEXTS:
        c = min(length, 8192)
        chunks = (length + c - 1) // c
        cold_pairs = length * (length + 1) / 2
        suffix_pairs = c * (length - c) + c * (c + 1) / 2
        # Current chunked expanded-MLA path restores/re-expands all prefix rows
        # on each chunk. Bounded workspace is not persistent expanded KV reuse.
        expanded_prefix_rows = c * chunks * (chunks - 1) / 2
        work.append(dict(context=length, prefill_chunk=c, cold_chunks=chunks,
            cold_fixed_pflop=FIXED_GF * length / 1e6,
            cold_mla_attention_pflop=24 * 2 * 96 * 320 * cold_pairs / 1e15,
            cold_repeated_expansion_pflop=24 * 2 * 96 * 256 * 512 * expanded_prefix_rows / 1e15,
            suffix_fixed_pflop=FIXED_GF * c / 1e6,
            suffix_mla_attention_pflop=24 * 2 * 96 * 320 * suffix_pairs / 1e15,
            suffix_expansion_pflop=24 * 2 * 96 * 256 * 512 * (length - c) / 1e15,
            decode_fixed_gflop=FIXED_GF,
            decode_absorbed_mla_gflop=24 * 2 * 96 * 1088 * length / 1e9,
            decode_fp8_cache_gib=24 * 576 * length / GIB,
            decode_bf16_cache_gib=24 * 1152 * length / GIB))
    write_csv(out / 'work.csv', work)
    notes = dict(assumptions=['All-resident weights, no PP/offload, TP-sharded embedding/head',
        '20 GiB planning allowance includes communication and live transient allocations',
        'BF16 default K3 cache; explicit fp8_e4m3 uses raw Blackwell 576-byte pages',
        'CP is nested in projection TP and not production K3 runtime support',
        'World32 and both B300/B200 capacity budgets are analytical, not this run hardware',
        'Counts are memory-only balanced upper estimates, not scheduler or SLO limits'],
        fixed_gflops_per_token=FIXED_GF, cache_bytes_per_token_layer=CACHE,
        checkpoint_path='/hgpfs/Kimi-K3')
    (out / 'assumptions.json').write_text(json.dumps(notes, indent=2) + '\n')
    for hardware in HARDWARE:
        print(hardware)
        for world in (8, 16):
            selected = [r for r in rows if r['hardware']==hardware and r['world']==world and r['context']==1048576 and r['nested_cp']==1 and r['cache_dtype']=='fp8_e4m3']
            print([(r['tp'], round(r['weight_gib'], 1), r['cluster_requests']) for r in selected])
    print('Cold/suffix/decode work:', work[-1])


if __name__ == '__main__': main()

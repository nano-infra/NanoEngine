import argparse
import json
from pathlib import Path

import torch
from deep_gemm import ceil_div, get_m_alignment_for_contiguous_layout
from dlengine.runtime.kernel.triton.hopper import block_gemm_fp8


def _allocate_outputs(x: torch.Tensor, group_size: int):
    m, k = x.shape
    aligned_m = ceil_div(m, get_m_alignment_for_contiguous_layout())
    aligned_m *= get_m_alignment_for_contiguous_layout()
    quant = x.new_empty(aligned_m, k, dtype=torch.float8_e4m3fn)
    scales = x.new_empty(k // group_size, aligned_m, dtype=torch.float32).T
    return quant, scales


def _launch(
    x: torch.Tensor,
    quant: torch.Tensor,
    scales: torch.Tensor,
    *,
    packed: bool,
    round_ue8m0: bool,
):
    block_gemm_fp8._USE_PACKED_SMALL_M_QUANT = packed
    return block_gemm_fp8._quant_fp8_launcher(
        x,
        128,
        quant,
        scales,
        round_ue8m0=round_ue8m0,
    )


def _time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def _capture_graph(fn) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def benchmark_case(m: int, k: int, round_ue8m0: bool, iterations: int):
    torch.manual_seed(137 + m + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    legacy_q, legacy_s = _allocate_outputs(x, 128)
    packed_q, packed_s = _allocate_outputs(x, 128)
    _launch(x, legacy_q, legacy_s, packed=False, round_ue8m0=round_ue8m0)
    _launch(x, packed_q, packed_s, packed=True, round_ue8m0=round_ue8m0)
    torch.cuda.synchronize()

    legacy_us = _time_us(
        lambda: _launch(
            x,
            legacy_q,
            legacy_s,
            packed=False,
            round_ue8m0=round_ue8m0,
        ),
        20,
        iterations,
    )
    packed_us = _time_us(
        lambda: _launch(
            x,
            packed_q,
            packed_s,
            packed=True,
            round_ue8m0=round_ue8m0,
        ),
        20,
        iterations,
    )
    legacy_graph = _capture_graph(
        lambda: _launch(
            x,
            legacy_q,
            legacy_s,
            packed=False,
            round_ue8m0=round_ue8m0,
        )
    )
    packed_graph = _capture_graph(
        lambda: _launch(
            x,
            packed_q,
            packed_s,
            packed=True,
            round_ue8m0=round_ue8m0,
        )
    )
    legacy_graph_us = _time_us(legacy_graph.replay, 20, iterations)
    packed_graph_us = _time_us(packed_graph.replay, 20, iterations)
    return {
        "m": m,
        "k": k,
        "round_ue8m0": round_ue8m0,
        "aligned_m": legacy_q.shape[0],
        "quant_bit_exact": bool(torch.equal(legacy_q, packed_q)),
        "scale_bit_exact": bool(torch.equal(legacy_s, packed_s)),
        "legacy_us": legacy_us,
        "packed_us": packed_us,
        "speedup": legacy_us / packed_us,
        "legacy_graph_us": legacy_graph_us,
        "packed_graph_us": packed_graph_us,
        "graph_speedup": legacy_graph_us / packed_graph_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [
        benchmark_case(m, k, round_ue8m0, args.iterations)
        for round_ue8m0 in (False, True)
        for m in (1, 6, 16, 64)
        for k in (2048, 4096, 6144)
    ]
    report = {
        "device": torch.cuda.get_device_name(),
        "iterations": args.iterations,
        "cases": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

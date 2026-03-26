#!/usr/bin/env python3
"""
Multi-sequence-length latency benchmark for Protenix.

Each length is run in an isolated subprocess so GPU memory resets cleanly
between runs. Measures wall-clock time and peak VRAM.

Usage:
    python bench_latency.py
    python bench_latency.py --lengths 100 500 1000 2000
    python bench_latency.py --model protenix_mini_default_v0.5.0 --n_step 50
    python bench_latency.py --gpu 1 --csv results.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# ── Synthetic sequence ────────────────────────────────────────────────
_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_sequence(length: int) -> str:
    return (_AA * (length // len(_AA) + 1))[:length]


def make_input_json(name: str, length: int) -> list[dict]:
    return [
        {
            "name": name,
            "sequences": [
                {"proteinChain": {"sequence": make_sequence(length), "count": 1}}
            ],
        }
    ]


# ── Per-run subprocess wrapper ────────────────────────────────────────
_WRAPPER = textwrap.dedent(
    """
import json, os, sys, time, torch, subprocess, tempfile

args  = json.loads(sys.argv[1])
out_f = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])
torch.cuda.reset_peak_memory_stats()

with tempfile.TemporaryDirectory() as tmp:
    inp = os.path.join(tmp, "input.json")
    out = os.path.join(tmp, "output")
    with open(inp, "w") as fh:
        json.dump(args["input_json"], fh)

    cmd = [
        "protenix", "pred",
        "-i", inp,
        "-o", out,
        "-n", args["model_name"],
        "--use_msa", "false",
        "--use_template", "false",
        "--seeds", str(args["seed"]),
        f"--sample_diffusion.N_sample={args['n_sample']}",
        f"--sample_diffusion.N_step={args['n_step']}",
        f"--model.N_cycle={args['n_cycle']}",
        "--trimul_kernel", args["trimul_kernel"],
        "--triatt_kernel", args["triatt_kernel"],
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

peak_gb = torch.cuda.max_memory_allocated() / 1024**3

result = {
    "returncode": proc.returncode,
    "elapsed_s":  elapsed,
    "peak_vram_gb": peak_gb,
    "stdout": proc.stdout[-2000:],
    "stderr": proc.stderr[-2000:],
}
with open(out_f, "w") as fh:
    json.dump(result, fh)
"""
)


def run_one(
    length: int,
    model_name: str,
    n_sample: int,
    n_step: int,
    n_cycle: int,
    seed: int,
    gpu: int,
    trimul_kernel: str,
    triatt_kernel: str,
) -> dict:
    input_json = make_input_json(f"bench_L{length}", length)
    run_args = {
        "gpu": gpu,
        "model_name": model_name,
        "input_json": input_json,
        "seed": seed,
        "n_sample": n_sample,
        "n_step": n_step,
        "n_cycle": n_cycle,
        "trimul_kernel": trimul_kernel,
        "triatt_kernel": triatt_kernel,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as wf:
        wf.write(_WRAPPER)
        wrapper_path = wf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as rf:
        result_path = rf.name

    try:
        subprocess.run(
            [sys.executable, wrapper_path, json.dumps(run_args), result_path],
            check=False,
        )
        with open(result_path) as fh:
            return json.load(fh)
    finally:
        os.unlink(wrapper_path)
        os.unlink(result_path)


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Protenix latency benchmark")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 1500, 2000, 3000],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--model", default="protenix_mini_default_v0.5.0")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument(
        "--n_step",
        type=int,
        default=50,
        help="Diffusion steps (reduce for faster bench)",
    )
    parser.add_argument(
        "--n_cycle",
        type=int,
        default=2,
        help="Pairformer cycles (reduce for faster bench)",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trimul_kernel", default="cuequivariance")
    parser.add_argument("--triatt_kernel", default="cuequivariance")
    parser.add_argument(
        "--skip_on_oom", action="store_true", help="Stop sweep after first OOM/failure"
    )
    parser.add_argument("--csv", default=None, help="Write results to CSV file")
    args = parser.parse_args()

    rows = []
    header = f"{'Length':>8}  {'Status':>8}  {'Time (s)':>10}  {'VRAM (GB)':>10}"
    print(header)
    print("-" * len(header))

    for length in sorted(args.lengths):
        result = run_one(
            length=length,
            model_name=args.model,
            n_sample=args.n_sample,
            n_step=args.n_step,
            n_cycle=args.n_cycle,
            seed=args.seed,
            gpu=args.gpu,
            trimul_kernel=args.trimul_kernel,
            triatt_kernel=args.triatt_kernel,
        )

        ok = result.get("returncode", 1) == 0
        status = "ok" if ok else "FAIL"
        elapsed = result.get("elapsed_s", float("nan"))
        vram = result.get("peak_vram_gb", float("nan"))

        print(f"{length:>8}  {status:>8}  {elapsed:>10.1f}  {vram:>10.2f}")
        rows.append(
            {
                "length": length,
                "status": status,
                "elapsed_s": elapsed,
                "peak_vram_gb": vram,
                "returncode": result.get("returncode"),
            }
        )

        if args.skip_on_oom and not ok:
            print(f"Stopping sweep after failure at length {length}.")
            if result.get("stderr"):
                print("--- stderr ---")
                print(result["stderr"][-1000:])
            break

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()

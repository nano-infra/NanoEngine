#!/usr/bin/env python3
"""Measure fixed KDA fresh chunks under multiple logical-context labels."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench.kda_eval.benchmark_kda import benchmark_prefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--contexts", default="32768,131072,524288,1048576")
    parser.add_argument("--chunks", default="1024,2048,4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    rows = []
    for context in contexts:
        for chunk in chunks:
            if chunk > context:
                continue
            run = SimpleNamespace(
                heads=96, kdim=128, vdim=128, batch_sizes=[1], lengths=[context],
                hit_rates=[(context - chunk) / context], block_size=64,
                lower_bound=-5.0, warmup=args.warmup, repeats=args.repeats,
                include_state_restore=True,
            )
            rows.extend(benchmark_prefill(run))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "kda_context_chunk_matrix.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "kda_context_chunk_matrix_metadata.json").write_text(
        json.dumps({"gpu": torch.cuda.get_device_name(), "contexts": contexts,
                    "chunks": chunks, "warmup": args.warmup,
                    "repeats": args.repeats}, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()

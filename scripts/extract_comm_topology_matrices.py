#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-DP Q/Res 8x8 communication topology matrices from "
            "NanoDeploy decode-step logs."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A log file, a single rate directory, or a directory containing multiple rate dirs.",
    )
    parser.add_argument(
        "--output-subdir",
        default="comm_topology_matrices",
        help="Subdirectory name to create under each rate directory.",
    )
    return parser.parse_args()


def find_rate_logs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    logs = sorted(input_path.rglob("*.log"))
    rate_logs = []
    for log_path in logs:
        if log_path.parent.name.startswith("dp") and "_r" in log_path.parent.name:
            rate_logs.append(log_path)
    return rate_logs


def zero_matrix(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def add_matrix_inplace(dst: list[list[float]], src: list[list[int | float]]) -> None:
    for i in range(len(dst)):
        for j in range(len(dst[i])):
            dst[i][j] += float(src[i][j])


def div_matrix(src: list[list[float]], denom: float) -> list[list[float]]:
    if denom == 0:
        return [[0.0 for _ in row] for row in src]
    return [[value / denom for value in row] for row in src]


def write_matrix_csv(path: Path, matrix: list[list[int | float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src\\dst"] + [f"sp{j}" for j in range(len(matrix[0]))])
        for i, row in enumerate(matrix):
            writer.writerow([f"sp{i}"] + row)


def sanitize_line(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line)


def parse_decode_payloads(log_path: Path) -> list[dict]:
    payloads = []
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = sanitize_line(raw_line)
        if "llm_engine.py:204" not in line or "'mode': 'decode'" not in line:
            continue
        _, payload_text = line.split(" - ", 1)
        payload = ast.literal_eval(payload_text)
        if "sp_q_matrix" not in payload or "sp_res_matrix" not in payload:
            continue
        payloads.append(payload)
    return payloads


def summarize_log(log_path: Path, output_subdir: str) -> None:
    payloads = parse_decode_payloads(log_path)
    if not payloads:
        return

    rate_dir = log_path.parent
    output_dir = rate_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    q_sample = payloads[0]["sp_q_matrix"]
    res_sample = payloads[0]["sp_res_matrix"]
    dp_count = len(q_sample)
    sp_count = len(q_sample[0])
    decode_iters = len(payloads)

    q_sums = [zero_matrix(sp_count) for _ in range(dp_count)]
    res_sums = [zero_matrix(sp_count) for _ in range(dp_count)]

    for payload in payloads:
        q_matrices = payload["sp_q_matrix"]
        res_matrices = payload["sp_res_matrix"]
        for dp_idx in range(dp_count):
            add_matrix_inplace(q_sums[dp_idx], q_matrices[dp_idx])
            add_matrix_inplace(res_sums[dp_idx], res_matrices[dp_idx])

    q_means = [div_matrix(matrix, decode_iters) for matrix in q_sums]
    res_means = [div_matrix(matrix, decode_iters) for matrix in res_sums]

    global_q_sum = zero_matrix(sp_count)
    global_res_sum = zero_matrix(sp_count)
    for dp_idx in range(dp_count):
        add_matrix_inplace(global_q_sum, q_sums[dp_idx])
        add_matrix_inplace(global_res_sum, res_sums[dp_idx])
    global_q_mean = div_matrix(global_q_sum, dp_count)
    global_res_mean = div_matrix(global_res_sum, dp_count)

    summary = {
        "source_log": str(log_path),
        "rate_dir": str(rate_dir),
        "decode_iters": decode_iters,
        "dp_count": dp_count,
        "sp_count": sp_count,
        "dp_matrices": {},
        "global": {
            "q_sum": global_q_sum,
            "q_mean_per_dp": global_q_mean,
            "res_sum": global_res_sum,
            "res_mean_per_dp": global_res_mean,
        },
    }

    for dp_idx in range(dp_count):
        dp_key = f"dp{dp_idx}"
        summary["dp_matrices"][dp_key] = {
            "q_sum": q_sums[dp_idx],
            "q_mean_per_iter": q_means[dp_idx],
            "res_sum": res_sums[dp_idx],
            "res_mean_per_iter": res_means[dp_idx],
        }
        write_matrix_csv(output_dir / f"{dp_key}_q_sum.csv", q_sums[dp_idx])
        write_matrix_csv(output_dir / f"{dp_key}_q_mean_per_iter.csv", q_means[dp_idx])
        write_matrix_csv(output_dir / f"{dp_key}_res_sum.csv", res_sums[dp_idx])
        write_matrix_csv(output_dir / f"{dp_key}_res_mean_per_iter.csv", res_means[dp_idx])

    write_matrix_csv(output_dir / "global_q_sum.csv", global_q_sum)
    write_matrix_csv(output_dir / "global_q_mean_per_dp.csv", global_q_mean)
    write_matrix_csv(output_dir / "global_res_sum.csv", global_res_sum)
    write_matrix_csv(output_dir / "global_res_mean_per_dp.csv", global_res_mean)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    for log_path in find_rate_logs(args.input_path):
        summarize_log(log_path, args.output_subdir)


if __name__ == "__main__":
    main()

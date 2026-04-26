#!/usr/bin/env python3
"""Run NanoDeploy DSV4 with tensor dumping enabled.

This is a thin wrapper around examples/non_disagg.py.  It sets the debug
environment variables consumed by nanodeploy.models.deepseek_v4.deepseek_v4.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model",
        default="/models_cfs/models--lovedheart--DeepSeek-V4-Flash-FP8-SGlang",
    )
    parser.add_argument("--ray-address", default="10.102.97.179:7078")
    parser.add_argument("--master-address", default="10.102.97.179:6006")
    parser.add_argument("--prompt", default="1+1=?")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--debug-rank", default="all")
    parser.add_argument("--debug-layers", default="0")
    parser.add_argument("--debug-max-tokens", type=int, default=8)
    parser.add_argument("--log-file", default="dsv4_nanodeploy_debug.log")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NANODEPLOY_DSV4_DEBUG_DIR"] = str(Path(args.out_dir).resolve())
    env["NANODEPLOY_DSV4_DEBUG_RANK"] = args.debug_rank
    env["NANODEPLOY_DSV4_DEBUG_LAYERS"] = args.debug_layers
    env["NANODEPLOY_DSV4_DEBUG_MAX_TOKENS"] = str(args.debug_max_tokens)
    env.setdefault("NANODEPLOY_DSV4_DEBUG_PREFILL_ONLY", "1")
    env.setdefault("NANODEPLOY_DSV4_DEBUG_ONCE", "1")
    env.setdefault("NANODEPLOY_DSV4_DEBUG_SKIP_DUMMY", "1")

    cmd = [
        sys.executable,
        str(repo_root / "examples" / "non_disagg.py"),
        "--num_speculative_tokens",
        "0",
        "--trust_remote_code",
        "false",
        "--ray_address",
        args.ray_address,
        "--master_address",
        args.master_address,
        "--gpu_memory_utilization",
        "0.8",
        "--model",
        args.model,
        "--attention_tp",
        "1",
        "--attention_dp",
        "8",
        "--attention_sp",
        "1",
        "--ffn_dp",
        "1",
        "--ffn_ep",
        "8",
        "--ffn_tp",
        "1",
        "--loop_count",
        "1",
        "--kvcache_block_size",
        "64",
        "--max_num_seqs",
        "16",
        "--temperature",
        "0",
        "--prompt",
        args.prompt,
        "--max_tokens",
        str(args.max_tokens),
        "--max_num_batched_tokens",
        "512",
        "--max_model_len",
        "4096",
        "--log_level",
        "INFO",
        "--enforce_eager",
        "true",
    ]

    log_path = Path(args.log_file)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout, end="")
        log.write(proc.stdout)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

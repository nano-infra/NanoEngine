#!/usr/bin/env python3
"""Generate the four Fig. 13 KV-cache distributions for both profilers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "kv-distributions"
LONGS_PER_NODE = (1, 3, 5, 7)
NODE_COUNT = 4
GPUS_PER_NODE = 8
LONG_LENGTH = 512 * 1024
SHORT_LENGTH = 2048
SHORTS_PER_GPU = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify the Fig. 13 NanoDeploy and vLLM inputs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed files instead of rewriting them",
    )
    return parser.parse_args()


def case_name(longs_per_node: int) -> str:
    return f"mix_{longs_per_node}x512k_pernode_plus_64x2048_pergpu"


def interleaved_lengths(longs_per_node: int) -> list[int]:
    total_longs = longs_per_node * NODE_COUNT
    total_shorts = SHORTS_PER_GPU * GPUS_PER_NODE * NODE_COUNT
    total = total_longs + total_shorts
    long_positions = {
        (long_index * total) // total_longs for long_index in range(total_longs)
    }
    return [
        LONG_LENGTH if position in long_positions else SHORT_LENGTH
        for position in range(total)
    ]


def expected_files(output_root: Path) -> dict[Path, object]:
    files: dict[Path, object] = {}
    summary = []
    for longs_per_node in LONGS_PER_NODE:
        name = case_name(longs_per_node)
        lengths = interleaved_lengths(longs_per_node)
        files[output_root / "vllm" / f"{name}.json"] = lengths
        files[
            output_root / "nano" / name / "processed_input_3d.json"
        ] = {"sp_seq_lens": [[lengths]]}
        summary.append(
            {
                "case": name,
                "nodes": NODE_COUNT,
                "gpus_per_node": GPUS_PER_NODE,
                "long_requests_per_node": longs_per_node,
                "long_request_length": LONG_LENGTH,
                "short_requests_per_gpu": SHORTS_PER_GPU,
                "short_request_length": SHORT_LENGTH,
                "total_long_requests": longs_per_node * NODE_COUNT,
                "total_short_requests": SHORTS_PER_GPU * GPUS_PER_NODE * NODE_COUNT,
                "total_requests": len(lengths),
            }
        )
    files[output_root / "summary.json"] = summary
    return files


def check_files(files: dict[Path, object]) -> None:
    errors = []
    for path, expected in files.items():
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if actual != expected:
            errors.append(f"content differs: {path}")
    if errors:
        raise SystemExit("KV-distribution check failed:\n  " + "\n  ".join(errors))
    print(f"verified {len(files)} Fig. 13 KV-distribution files")


def write_files(files: dict[Path, object]) -> None:
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if path.name == "summary.json" else None
        path.write_text(
            json.dumps(payload, indent=indent, separators=None if indent else (",", ":"))
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    files = expected_files(output_root)
    if args.check:
        check_files(files)
    else:
        write_files(files)


if __name__ == "__main__":
    main()

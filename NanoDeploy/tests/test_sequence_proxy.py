"""Sanity checks for C++ Sequence container proxies.

This repo does not require pytest; run with:
  python tests/test_sequence_proxy.py

It exits 0 on success (or if C++ extension is unavailable).
"""

from __future__ import annotations

import sys


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def main() -> int:
    try:
        from nanodeploy._cpp import Sequence  # type: ignore
    except Exception as e:
        return _skip(f"nanodeploy._cpp not importable: {type(e).__name__}: {e}")

    # C++ Sequence signature: (token_ids, temperature, max_tokens, ignore_eos, engine_id, master_sp_rank)
    seq = Sequence([1, 2, 3], 1.0, 16, False, "engine", 0)

    ctx = seq.block_ctx("engine")

    # 1) block_location is a mutable proxy (not a Python list copy)
    ctx.block_location.clear()
    ctx.block_location.append((0, 42))
    assert len(ctx.block_location) == 1
    assert tuple(ctx.block_location[0]) == (0, 42)

    # 2) group_block_table is a mutable proxy with defaultdict(list)-like semantics
    ctx.group_block_table.clear()
    ctx.group_block_table[0].append(7)
    assert list(ctx.group_block_table[0]) == [7]

    # Also verify Sequence.block_table returns a mutable proxy into the same storage
    table = seq.block_table("engine", 0)
    table.append(9)
    assert list(ctx.group_block_table[0]) == [7, 9]

    # 3) block_ctx_map behaves like a mutable mapping proxy (no dict copy)
    # This mirrors Python usage: seq.block_ctx_map[engine_id].dp_idx = selected_dp_idx
    seq.block_ctx_map["engine"].dp_idx = 123
    assert seq.block_ctx("engine").dp_idx == 123

    print("[ok] C++ proxy containers behave as mutable views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
        from nanodeploy._cpp import BlockContextSlot, Sequence  # type: ignore
    except Exception as e:
        return _skip(f"nanodeploy._cpp not importable: {type(e).__name__}: {e}")

    seq = Sequence([1, 2, 3], 1.0, 16, False)
    seq.active("engine", 2, 1)

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)

    # 1) block_location is a mutable proxy (not a Python list copy)
    ctx.block_location.clear()
    ctx.block_location.append((0, 42))
    assert len(ctx.block_location) == 1
    assert tuple(ctx.block_location[0]) == (0, 42)

    # 2) sp_block_table is a mutable proxy with defaultdict(list)-like semantics
    ctx.sp_block_table[0] = []
    ctx.sp_block_table[0].append(7)
    assert list(ctx.sp_block_table[0]) == [7]

    # Also verify Sequence.block_table returns a mutable proxy into the same storage
    table = seq.block_table(BlockContextSlot.ACTIVE, 0)
    table.append(9)
    assert list(ctx.sp_block_table[0]) == [7, 9]

    # 3) scalar fields remain writable through the active BlockContext view.
    ctx.dp_idx = 123
    assert seq.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 123

    # 4) optimistic decode snapshots do not mutate canonical state.
    seq.seq_id = 17
    ctx.master_sp_idx = 0
    ctx.num_dispatched_tokens = [3, 0]
    clone = seq.clone_for_decode_dispatch()
    clone.num_tokens += 1
    clone.block_ctx().num_dispatched_tokens = [4, 0]
    clone.block_table(BlockContextSlot.ACTIVE, 0).append(11)
    assert clone.seq_id == 17
    assert clone.token_ids == [1, 2, 3]
    assert seq.num_tokens == 3
    assert list(seq.block_ctx().num_dispatched_tokens) == [3, 0]
    assert list(seq.block_table(BlockContextSlot.ACTIVE, 0)) == [7, 9]

    print("[ok] C++ proxy containers behave as mutable views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

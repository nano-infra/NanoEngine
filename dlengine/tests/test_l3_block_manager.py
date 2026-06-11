"""Standalone unit test for the L3 (3FS) hooks in the C++ BlockManager.

Pure-logic test (no GPU / model / 3FS needed): exercises offload-on-evict,
load-on-prefix-miss, hash revalidation on reuse, and inert-when-disabled.

Run with:
  python tests/test_l3_block_manager.py
Exits 0 on success (or if the C++ extension is unavailable).
"""

from __future__ import annotations

import sys

BS = 4  # tokens per block


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def _make_seq(token_ids, num_blocks=16):
    """Build an ACTIVE-slot sequence ready for BlockManager.allocate().

    Mirrors what the scheduler does: activate the engine/group then set
    ``num_dispatched_tokens`` (which drives ``num_blocks``) to the prompt len.
    """
    from dlengine._cpp import BlockContextSlot, SamplingParams, Sequence

    tokens = list(token_ids)
    seq = Sequence(tokens, SamplingParams(1.0, 16, False, False))
    seq.active(
        "engine", 1, 1, num_blocks
    )  # engine_id, group_size, attention_dp, blocks
    seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [len(tokens)]
    return seq


def main() -> int:
    try:
        from dlengine._cpp import BlockContextSlot, BlockManager, Sequence
    except Exception as e:  # pragma: no cover
        return _skip(f"dlengine._cpp not importable: {type(e).__name__}: {e}")

    Sequence.set_block_size(BS)

    # ------------------------------------------------------------------ #
    # 1) Disabled by default: no pending work is recorded.
    # ------------------------------------------------------------------ #
    bm = BlockManager(engine_id="engine", group_id=0, num_blocks=16, block_size=BS)
    seq = _make_seq(range(8))  # 2 full blocks
    assert not bm.l3_enabled
    bm.allocate(seq)
    bm.deallocate(seq, BlockContextSlot.ACTIVE)
    assert bm.drain_pending_offloads() == [], "offloads recorded while L3 disabled"
    assert bm.drain_pending_loads() == [], "loads recorded while L3 disabled"
    print("[ok] L3 inert when disabled")

    # ------------------------------------------------------------------ #
    # 2) Offload-on-evict: full blocks queued when ref_count hits 0.
    # ------------------------------------------------------------------ #
    bm = BlockManager(engine_id="engine", group_id=0, num_blocks=16, block_size=BS)
    bm.set_l3_enabled(True)
    seq = _make_seq(range(8))
    hashes = bm.compute_block_hashes(seq)
    assert len(hashes) == 2, f"expected 2 full-block hashes, got {hashes}"
    bm.allocate(seq)
    table = list(seq.block_table(BlockContextSlot.ACTIVE, 0))
    assert len(table) == 2
    bm.deallocate(seq, BlockContextSlot.ACTIVE)
    offloads = bm.drain_pending_offloads()
    assert len(offloads) == 2, f"expected 2 offloads, got {offloads}"
    off_hashes = sorted(h for h, _ in offloads)
    assert off_hashes == sorted(hashes), f"{off_hashes} != {sorted(hashes)}"
    # Draining clears the queue.
    assert bm.drain_pending_offloads() == []
    print("[ok] offload-on-evict queues full evicted blocks")

    # ------------------------------------------------------------------ #
    # 3) Hash revalidation: a block reused for different content is NOT
    #    offloaded with the stale hash.
    # ------------------------------------------------------------------ #
    bm = BlockManager(engine_id="engine", group_id=0, num_blocks=2, block_size=BS)
    bm.set_l3_enabled(True)
    seqA = _make_seq(range(8))  # uses both physical blocks
    bm.allocate(seqA)
    bm.deallocate(seqA, BlockContextSlot.ACTIVE)  # queues 2 offloads
    # Reuse all blocks with different content before draining.
    seqB = _make_seq(range(100, 108))
    bm.allocate(seqB)
    survivors = bm.drain_pending_offloads()
    assert survivors == [], f"stale-hash blocks should be skipped, got {survivors}"
    print("[ok] reused blocks are not offloaded with stale hashes")

    # ------------------------------------------------------------------ #
    # 4) Load-on-prefix-miss: hashes resident only in L3 are treated as a
    #    cached prefix (recompute skipped) and queued for load.
    # ------------------------------------------------------------------ #
    bm = BlockManager(engine_id="engine", group_id=0, num_blocks=16, block_size=BS)
    bm.set_l3_enabled(True)
    seq = _make_seq(range(8))
    hashes = bm.compute_block_hashes(seq)
    bm.set_l3_resident_hashes(hashes)  # pretend both blocks are durable in L3
    n_cached_hint = bm.can_allocate(seq)
    bm.allocate(seq, n_cached_hint)
    loads = bm.drain_pending_loads()
    assert len(loads) == 2, f"expected 2 L3 loads, got {loads}"
    load_hashes = sorted(h for h, _ in loads)
    assert load_hashes == sorted(hashes)
    # Both full blocks counted as cached -> recompute skipped (capped at n-1).
    assert seq.num_cached_tokens == 7, f"num_cached_tokens={seq.num_cached_tokens}"
    # Loaded blocks now resident in the GPU prefix cache.
    table = list(seq.block_table(BlockContextSlot.ACTIVE, 0))
    for h, bid in loads:
        assert bm.is_l3_resident(h)
        assert bid in table
    print("[ok] load-on-miss queues L3 blocks and marks prefix cached")

    # ------------------------------------------------------------------ #
    # 5) GPU hit takes precedence over L3 (no redundant load).
    # ------------------------------------------------------------------ #
    bm = BlockManager(engine_id="engine", group_id=0, num_blocks=16, block_size=BS)
    bm.set_l3_enabled(True)
    seq1 = _make_seq(range(8))
    hashes = bm.compute_block_hashes(seq1)
    bm.set_l3_resident_hashes(hashes)
    bm.allocate(seq1)  # populates GPU cache; seq1 keeps refs
    _ = bm.drain_pending_loads()  # ignore (cold first alloc may L3-hit)
    seq2 = _make_seq(range(8))  # identical prefix -> should hit GPU, not L3
    h2 = bm.can_allocate(seq2)
    bm.allocate(seq2, h2)
    loads2 = bm.drain_pending_loads()
    assert loads2 == [], f"GPU-resident prefix should not queue L3 loads, got {loads2}"
    assert seq2.num_cached_tokens == 7
    print("[ok] GPU prefix hit takes precedence over L3 load")

    print("\nAll L3 BlockManager tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

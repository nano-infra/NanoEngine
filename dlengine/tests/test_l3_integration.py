"""End-to-end L3 round-trip: C++ BlockManager hooks + Hf3fsL3Store on real 3FS.

Simulates the engine's offload-then-load path WITHOUT a model forward or Ray,
exercising the exact (hash, block_id) contract the driver wiring uses:

  Phase 1 (engine A): allocate a prompt, write KV keyed by block hash, evict
                      -> drain_pending_offloads() -> Hf3fsL3Store.store().
  (zero GPU + fresh BlockManager == engine restart / different node)
  Phase 2 (engine B): warm resident set from 3FS -> allocate the SAME prompt
                      -> the prefix is an L3 hit (recompute skipped) ->
                      drain_pending_loads() -> Hf3fsL3Store.load() -> verify
                      the KV bytes are restored bit-exact.

Requires CUDA + a 3FS mount. Skips (0) otherwise.
Run: python tests/test_l3_integration.py --mountpoint /3fs/mnt
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch

BS = 64
LAYERS = 8
PAGES = 8


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def _make_ctx():
    kv = torch.zeros(LAYERS, PAGES + 1, BS, 1, 584, dtype=torch.uint8, device="cuda")
    return SimpleNamespace(
        kv_cache=kv,
        mode="dsv4",
        num_hidden_layers=LAYERS,
        block_size=BS,
        num_local_kv_heads=1,
        head_dim=512,
        dtype=torch.uint8,
        is_fp8_kvcache=False,
        _fp8_head_dim=0,
    )


def _make_seq(tokens):
    from dlengine._cpp import BlockContextSlot, SamplingParams, Sequence

    Sequence.set_block_size(BS)
    seq = Sequence(list(tokens), SamplingParams(1.0, 16, False, False))
    seq.active("engine", 1, 1, PAGES)
    seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [len(tokens)]
    return seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mountpoint", default=os.environ.get("HF3FS_MOUNTPOINT", "/3fs/mnt")
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        return _skip("CUDA not available")
    if not os.path.isdir(os.path.join(args.mountpoint, "3fs-virt")):
        return _skip(f"{args.mountpoint} is not a 3FS mount")
    try:
        import hf3fs_fuse.io  # noqa: F401
        from dlengine._cpp import BlockContextSlot, BlockManager
        from dlengine.context.cache.l3_hf3fs import Hf3fsL3Store
    except Exception as e:
        return _skip(f"deps unavailable: {e}")

    ctx = _make_ctx()
    l3_dir = os.path.join(args.mountpoint, "dlengine_l3_itest")
    store = Hf3fsL3Store(
        ctx, mountpoint=args.mountpoint, l3_dir=l3_dir, staging_blocks=4, rank=0
    )
    tokens = list(range(2 * BS))  # exactly 2 full blocks
    fill = {}  # hash -> byte value written into that block's KV
    try:
        # -------- Phase 1: engine A populates + evicts + offloads ---------- #
        bm_a = BlockManager("engine", 0, PAGES, BS)
        bm_a.set_l3_enabled(True)
        seqA = _make_seq(tokens)
        hashes = bm_a.compute_block_hashes(seqA)
        assert len(hashes) == 2
        bm_a.allocate(seqA, bm_a.can_allocate(seqA))
        table = list(seqA.block_table(BlockContextSlot.ACTIVE, 0))
        # Write a distinct, hash-derived pattern into each block's KV.
        for h, bid in zip(hashes, table):
            v = h & 0xFF
            fill[h] = v
            ctx.kv_cache[:, bid].fill_(v)
        bm_a.deallocate(seqA, BlockContextSlot.ACTIVE)
        offloads = bm_a.drain_pending_offloads()
        assert sorted(h for h, _ in offloads) == sorted(hashes)
        store.store(offloads, skip_existing=False)
        print(f"[ok] phase1: offloaded {len(offloads)} blocks to 3FS")

        # Simulate restart: wipe GPU KV; the data now lives only in 3FS.
        ctx.kv_cache.zero_()

        # -------- Phase 2: engine B warms resident set + L3-loads ---------- #
        bm_b = BlockManager("engine", 0, PAGES, BS)
        bm_b.set_l3_enabled(True)
        resident = store.exists(hashes)  # filesystem warm-up (cross-restart)
        assert sorted(resident) == sorted(hashes), "blocks not durable in 3FS"
        bm_b.mark_l3_resident(resident)

        seqB = _make_seq(tokens)  # identical prompt
        bm_b.allocate(seqB, bm_b.can_allocate(seqB))
        loads = bm_b.drain_pending_loads()
        assert len(loads) == 2, f"expected 2 L3 loads, got {loads}"
        # Whole prompt minus 1 (need >=1 Q token) should be cache-marked.
        assert seqB.num_cached_tokens == 2 * BS - 1, seqB.num_cached_tokens

        store.load(loads)
        torch.cuda.synchronize()

        # Verify each loaded block holds exactly the bytes engine A wrote.
        for h, bid in loads:
            blk = ctx.kv_cache[:, bid]
            if not bool((blk == fill[h]).all()):
                raise AssertionError(f"block {h:#x} not restored bit-exact")
        print(
            f"[ok] phase2: L3-loaded {len(loads)} blocks, KV bit-exact, "
            f"prefix recompute skipped ({seqB.num_cached_tokens} tokens cached)"
        )

        print("\nL3 end-to-end round-trip passed.")
        return 0
    finally:
        for h in fill:
            try:
                os.remove(store._path(h))
            except OSError:
                pass
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

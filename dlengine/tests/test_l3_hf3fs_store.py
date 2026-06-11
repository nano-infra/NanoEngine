"""Standalone correctness + bandwidth test for Hf3fsL3Store against real 3FS.

Requires a CUDA GPU and a mounted 3FS at --mountpoint (default /3fs/mnt).
Run:
  python tests/test_l3_hf3fs_store.py --mountpoint /3fs/mnt
Exits 0 on success, skips (0) if GPU / 3FS / bindings are unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def _make_ctx(mode: str, num_layers: int, num_pages: int, block_size: int):
    dev = "cuda"
    if mode == "dsv4":
        kv = torch.randint(
            0,
            256,
            (num_layers, num_pages + 1, block_size, 1, 584),
            dtype=torch.uint8,
            device=dev,
        )
        return SimpleNamespace(
            kv_cache=kv,
            mode="dsv4",
            num_hidden_layers=num_layers,
            block_size=block_size,
            num_local_kv_heads=1,
            head_dim=512,
            dtype=torch.uint8,
            is_fp8_kvcache=False,
            _fp8_head_dim=0,
        )
    if mode == "gqa":
        kv_heads, head_dim = 8, 128
        kv = torch.randn(
            2,
            num_layers,
            num_pages,
            block_size,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=dev,
        )
        return SimpleNamespace(
            kv_cache=kv,
            mode="gqa",
            num_hidden_layers=num_layers,
            block_size=block_size,
            num_local_kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            is_fp8_kvcache=False,
            _fp8_head_dim=0,
        )
    raise ValueError(mode)


def _run_mode(mode, mountpoint, num_layers, num_pages, block_size, nblocks):
    from dlengine.context.cache.l3_hf3fs import Hf3fsL3Store

    ctx = _make_ctx(mode, num_layers, num_pages, block_size)
    store = Hf3fsL3Store(
        ctx,
        mountpoint=mountpoint,
        l3_dir=os.path.join(mountpoint, "dlengine_l3_test"),
        staging_blocks=4,
        rank=0,
    )
    try:
        # Synthetic hashes; block_ids are the first nblocks pages.
        pairs = [(0x5A5A0000 + i, i) for i in range(nblocks)]

        def _gather_u8(bid):
            return torch.cat(
                [
                    sub.contiguous().view(torch.uint8).reshape(-1).clone()
                    for _off, sub, _n in store._subviews(bid)
                ]
            )

        # Snapshot the original GPU bytes for each block.
        golden = [_gather_u8(bid) for _, bid in pairs]

        assert store.exists([h for h, _ in pairs]) == []
        stored = store.store(pairs, skip_existing=False)
        assert sorted(stored) == sorted(h for h, _ in pairs)
        assert set(store.exists([h for h, _ in pairs])) == {h for h, _ in pairs}

        # Corrupt the GPU blocks, then load them back from 3FS.
        for _, bid in pairs:
            for _off, sub, _n in store._subviews(bid):
                sub.zero_()
        loaded = store.load(pairs)
        assert loaded == nblocks

        # Verify bit-exact restoration.
        for (h, bid), gold in zip(pairs, golden):
            now = _gather_u8(bid)
            if not torch.equal(now, gold):
                raise AssertionError(f"[{mode}] block {h:#x} mismatch after load")

        s = store.stats()
        print(
            f"[ok] {mode}: {nblocks} blocks x {store.block_nbytes/1024:.0f} KiB  "
            f"store={s['store_GiBps']:.2f} GiB/s  load={s['load_GiBps']:.2f} GiB/s  "
            f"(bit-exact)"
        )
        return store.block_nbytes, s
    finally:
        # Cleanup test files.
        for h, _ in [(0x5A5A0000 + i, i) for i in range(nblocks)]:
            try:
                os.remove(store._path(h))
            except OSError:
                pass
        store.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mountpoint", default=os.environ.get("HF3FS_MOUNTPOINT", "/3fs/mnt")
    )
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--pages", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--nblocks", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        return _skip("CUDA not available")
    if not os.path.isdir(os.path.join(args.mountpoint, "3fs-virt")):
        return _skip(f"{args.mountpoint} is not a 3FS mount")
    try:
        import hf3fs_fuse.io  # noqa: F401
    except Exception as e:
        return _skip(f"hf3fs_fuse not importable: {e}")

    for mode in ("dsv4", "gqa"):
        _run_mode(
            mode,
            args.mountpoint,
            args.layers,
            args.pages,
            args.block_size,
            args.nblocks,
        )

    print("\nAll Hf3fsL3Store store/load tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

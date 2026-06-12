"""L3 (3FS) KV-cache tier connector.

Persists evicted KV-cache *blocks* to a 3FS distributed file system keyed by
their streaming block hash, and loads them back on a prefix miss instead of
recomputing prefill. This is the storage L3 tier below the GPU HBM L1 cache
(no persistent CPU L2 — only a transient host staging buffer required by the
USRBIO iovec, mirroring sglang HiCache's HF3FS backend).

Data path (per block):
    store:  GPU kv_cache[..., block_id] --cudaMemcpy--> shm staging
            --USRBIO write--> /<l3_dir>/<hash>.kv
    load:   /<l3_dir>/<hash>.kv --USRBIO read--> shm staging
            --cudaMemcpy--> GPU kv_cache[..., block_id]

One file == one block == all (kv_idx, layer) chunks for that block laid out
contiguously in a fixed order. Store and load must run against an identical
model/cache layout (same key derivation), which the engine guarantees.

This connector lives on the *worker* (it touches the GPU ``kv_cache`` tensor),
driven by the driver-side ``BlockManager`` L3 hooks via ``collective_rpc``.
"""

from __future__ import annotations

import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable

import torch
from dlengine.logging import get_logger

logger = get_logger("dlengine")


class Hf3fsL3Store:
    """USRBIO-backed per-block KV store on a 3FS mount.

    Parameters
    ----------
    cache_ctx:
        Object exposing the KV layout: ``kv_cache`` tensor, ``mode``
        ("gqa"|"mla"|"dsv4"), ``num_hidden_layers``, ``block_size``,
        ``num_local_kv_heads``, ``head_dim``, ``dtype``, ``is_fp8_kvcache``,
        ``_fp8_head_dim``. (A real ``CacheContext`` satisfies this.)
    mountpoint:
        3FS FUSE mountpoint (e.g. ``/3fs/mnt``).
    l3_dir:
        Directory under the mount where per-block files live.
    staging_blocks:
        Number of blocks the host staging buffer / ioring can hold.
    rank:
        Worker rank; files are namespaced per rank so KV-head shards (GQA
        ``attention_tp`` / SP groups) never collide.
    """

    def __init__(
        self,
        cache_ctx,
        mountpoint: str = "/3fs/mnt",
        l3_dir: str | None = None,
        staging_blocks: int = 8,
        rank: int = 0,
    ):
        self.ctx = cache_ctx
        self.mountpoint = mountpoint.rstrip("/")
        self.rank = rank
        self.staging_blocks = max(1, int(staging_blocks))

        if l3_dir is None:
            l3_dir = os.path.join(self.mountpoint, "dlengine_l3")
        # Per-rank subdir: each rank owns a distinct KV shard.
        self.l3_dir = os.path.join(l3_dir, f"rank{rank}")
        os.makedirs(self.l3_dir, exist_ok=True)

        self.block_nbytes = self._compute_block_nbytes()

        # Lazily import the USRBIO bindings so the module imports even on hosts
        # without the 3FS client (the engine only constructs this when enabled).
        from hf3fs_fuse.io import (  # noqa: F401
            deregister_fd,
            make_ioring,
            make_iovec,
            register_fd,
        )

        self._register_fd = register_fd
        self._deregister_fd = deregister_fd

        total = self.block_nbytes * self.staging_blocks
        self._shm = SharedMemory(size=total, create=True)
        self._iov = make_iovec(self._shm, self.mountpoint, 0, -1)
        # CPU uint8 view that shares memory with the iovec staging buffer.
        self._staging = torch.frombuffer(self._shm.buf, dtype=torch.uint8)
        self._wring = make_ioring(self.mountpoint, self.staging_blocks, False, 0)
        self._rring = make_ioring(self.mountpoint, self.staging_blocks, True, 0)

        # Stats (for the benefit estimate / metrics).
        self.bytes_stored = 0
        self.bytes_loaded = 0
        self.n_store = 0
        self.n_load = 0
        self.t_store = 0.0
        self.t_load = 0.0

        logger.info(
            f"[L3] Hf3fsL3Store rank={rank} dir={self.l3_dir} "
            f"block={self.block_nbytes / 1024:.1f} KiB staging={self.staging_blocks}"
        )

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _compute_block_nbytes(self) -> int:
        c = self.ctx
        bs = c.block_size
        if c.mode == "dsv4":
            per_layer = bs * 584  # uint8
            return c.num_hidden_layers * per_layer
        if c.mode == "mla" and getattr(c, "is_fp8_kvcache", False):
            per_layer = bs * c._fp8_head_dim  # fp8 == 1 byte
            return c.num_hidden_layers * per_layer
        # gqa / non-fp8 mla
        kv_count = 2 if c.mode == "gqa" else 1
        per_layer = bs * c.num_local_kv_heads * c.head_dim * c.dtype.itemsize
        return kv_count * c.num_hidden_layers * per_layer

    def _kv_count(self) -> int:
        return 2 if self.ctx.mode == "gqa" else 1

    def _subviews(self, block_id: int):
        """Yield (staging_byte_offset, gpu_subview, nbytes) per kv-index.

        Each subview spans *all layers* for one block, so a block costs at most
        two GPU<->host copies (kv_count). The subview is generally strided
        (block index fixed on an inner dim); store reads it (contiguous copy),
        load scatters into it (``copy_`` handles strided destinations).
        """
        kv = self.ctx.kv_cache
        off = 0
        for kv_idx in range(self._kv_count()):
            if self.ctx.mode == "dsv4":
                # [L, num_pages+1, block_size, 1, 584] -> [L, block_size, 1, 584]
                sub = kv[:, block_id]
            elif self.ctx.mode == "mla":
                # [1, L, blocks, block_size, 1, head_dim] -> [L, block_size, 1, head_dim]
                sub = kv[0, :, block_id]
            else:
                # gqa: [2, L, blocks, block_size, kv_heads, head_dim]
                sub = kv[kv_idx, :, block_id]
            nbytes = sub.numel() * sub.element_size()
            yield off, sub, nbytes
            off += nbytes

    def _path(self, block_hash: int) -> str:
        # int64 -> unsigned 16-hex filename (stable, no sign).
        return os.path.join(self.l3_dir, f"{block_hash & 0xFFFFFFFFFFFFFFFF:016x}.kv")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def exists(self, hashes: Iterable[int]) -> list[int]:
        """Return the subset of ``hashes`` durably present in L3."""
        return [h for h in hashes if os.path.exists(self._path(h))]

    def store(
        self, pairs: list[tuple[int, int]], skip_existing: bool = True
    ) -> list[int]:
        """Persist ``(hash, block_id)`` GPU blocks to 3FS. Returns stored hashes."""
        stored: list[int] = []
        slot0 = 0
        base = slot0 * self.block_nbytes
        for block_hash, block_id in pairs:
            path = self._path(block_hash)
            if skip_existing and os.path.exists(path):
                stored.append(block_hash)
                continue
            t0 = time.monotonic()
            # Gather GPU -> staging slot 0 (contiguous read copy per kv-index).
            for off, sub, nbytes in self._subviews(block_id):
                src = sub.contiguous().view(torch.uint8).reshape(-1)
                self._staging[base + off : base + off + nbytes].copy_(src)
            # USRBIO write staging -> file.
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
            os.ftruncate(fd, self.block_nbytes)
            self._register_fd(fd)
            try:
                buf = self._iov[base : base + self.block_nbytes]
                self._wring.prepare(buf, False, fd, 0, userdata=buf)
                for r in self._wring.submit().wait(min_results=1):
                    if r.result < 0:
                        raise OSError(-r.result, os.strerror(-r.result))
            finally:
                self._deregister_fd(fd)
                os.close(fd)
            self.bytes_stored += self.block_nbytes
            self.t_store += time.monotonic() - t0
            self.n_store += 1
            stored.append(block_hash)
        return stored

    def load(self, pairs: list[tuple[int, int]]) -> int:
        """Load ``(hash, block_id)`` blocks from 3FS into GPU. Returns #loaded."""
        slot0 = 0
        base = slot0 * self.block_nbytes
        loaded = 0
        for block_hash, block_id in pairs:
            path = self._path(block_hash)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[L3] missing block {block_hash:#x} at {path} — would corrupt KV"
                )
            t0 = time.monotonic()
            fd = os.open(path, os.O_RDONLY)
            self._register_fd(fd)
            try:
                buf = self._iov[base : base + self.block_nbytes]
                self._rring.prepare(buf, True, fd, 0, userdata=buf)
                for r in self._rring.submit().wait(min_results=1):
                    if r.result < 0:
                        raise OSError(-r.result, os.strerror(-r.result))
            finally:
                self._deregister_fd(fd)
                os.close(fd)
            # Scatter staging slot 0 -> GPU (write into the strided block view).
            for off, sub, nbytes in self._subviews(block_id):
                raw = self._staging[base + off : base + off + nbytes]
                tmp = raw.to(sub.device).view(sub.dtype).reshape(sub.shape)
                sub.copy_(tmp)
            self.bytes_loaded += self.block_nbytes
            self.t_load += time.monotonic() - t0
            self.n_load += 1
            loaded += 1
        return loaded

    def stats(self) -> dict:
        def bw(b, t):
            return (b / t / (1 << 30)) if t > 0 else 0.0

        return {
            "n_store": self.n_store,
            "n_load": self.n_load,
            "GiB_stored": self.bytes_stored / (1 << 30),
            "GiB_loaded": self.bytes_loaded / (1 << 30),
            "store_GiBps": bw(self.bytes_stored, self.t_store),
            "load_GiBps": bw(self.bytes_loaded, self.t_load),
        }

    def close(self):
        try:
            del self._iov
        except Exception:
            pass
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

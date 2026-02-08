import concurrent.futures
import dataclasses
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import dlslime
import numpy as np
import torch
import torch.distributed as dist
from nanodeploy._cpp import BlockContextSlot
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")

# PeerAgent path: buffer ID for kv_cache registration
_KV_CACHE_BUFFER_ID = "kv_cache"

# Cache TTL for engine_info from NanoCtrl (seconds)
# Engine registration rarely changes, cache forever by default (inf means never expire)
# To refresh, call invalidate_engine_info_cache() or restart the engine
_ENGINE_INFO_CACHE_TTL = float("inf")


@dataclasses.dataclass
class CacheContext:
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_hidden_layers: int
    attention_tp: int
    gpu_memory_utilization: float
    gpu_memory_limit_gb: float | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    mode: Literal["gqa", "mla"] = "gqa"
    num_local_kvcache_blocks = -1
    num_remote_kvcache_blocks: dict[str, int] = None
    kv_cache: torch.Tensor = None
    selected_nic: str | None = None
    endpoints: dict[str, dict[int, Any]] = None  # RDMAEndpoint or RDMALazyPeer

    # used for MLA mode
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    # Control plane: server address and engine ID for centralized connection
    nanoctrl_address: str | None = (
        None  # Control plane server URL (e.g., "http://10.102.97.183:3000")
    )
    engine_id: str | None = None  # Engine ID for agent naming (format: EngineName:rank)
    # If nanoctrl_address is provided, engine_id will be fetched from NanoCtrl instead of config

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def __post_init__(self):

        free, total = torch.cuda.mem_get_info()
        if self.gpu_memory_limit_gb is not None:
            total = min(total, self.gpu_memory_limit_gb * 1024**3)
        used = torch.cuda.mem_get_info()[1] - free  # real used
        memory_stats = torch.cuda.memory_stats()
        peak = memory_stats["allocated_bytes.all.peak"]
        current = memory_stats["allocated_bytes.all.current"]

        if self.mode == "gqa":
            assert self.attention_tp <= self.num_kv_heads
        elif self.mode == "mla":
            assert self.attention_tp == 1
            assert self.block_size == 64, "MLA mode only support block_size=64"
            self.num_kv_heads = 1
            self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        block_bytes = (
            self.num_hidden_layers
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.mode == "gqa":
            block_bytes *= 2

        self.num_local_kvcache_blocks = (
            int(total * self.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        logger.info(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )

        assert self.num_local_kvcache_blocks > 0

        available_nics = dlslime.available_nic()
        selected_nic_idx = dist.get_rank() % len(available_nics)
        self.selected_nic = available_nics[selected_nic_idx]
        assert self.selected_nic

        self.endpoints = {}
        self.num_remote_kvcache_blocks = {}
        self._peer_agent = None
        self._peer_agent_addr: str | None = None
        self._connected_peers: set[str] = set()  # track connected peer addresses
        self._local_mr_handler: int | None = None  # local MR handler for kv_cache
        self._engine_info_cache: tuple[float, dict[str, dict]] | None = (
            None  # (timestamp, engine_id -> engine_info_dict)
        )

    def block_stride(self, block_idx: int):
        return (
            block_idx
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    def local_layer_stride(self, layer_idx: int, block_idx: int):
        return (
            self.block_stride(self.num_local_kvcache_blocks)
        ) * layer_idx + self.block_stride(block_idx)

    def remote_layer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return (
            self.block_stride(self.num_remote_kvcache_blocks[remote_engine_id])
        ) * layer_idx + self.block_stride(block_idx)

    def local_kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int):
        return self.local_layer_stride(
            self.num_hidden_layers, 0
        ) * kv_idx + self.local_layer_stride(layer_idx, block_idx)

    def remote_kv_stride(
        self, kv_idx: int, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return self.remote_layer_stride(
            self.num_hidden_layers, 0, remote_engine_id
        ) * kv_idx + self.remote_layer_stride(layer_idx, block_idx, remote_engine_id)

    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks

        kv_count = 2 if self.mode == "gqa" else 1

        self.kv_cache = torch.empty(
            kv_count,
            self.num_hidden_layers,
            self.num_local_kvcache_blocks,
            self.block_size,
            self.num_local_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        # PeerAgent path: initialize RdmaPeerAgent and register kv_cache buffer
        # Use control plane API if nanoctrl_address is provided
        # If engine_id is not provided, fetch it from NanoCtrl
        if self.nanoctrl_address is not None:
            if self.engine_id is not None:
                start_peer_agent_fn = getattr(dlslime, "start_peer_agent", None)
                if callable(start_peer_agent_fn):
                    rank = dist.get_rank()
                    # Agent alias format: EngineName:rank
                    agent_alias = f"{self.engine_id}:{rank}"

                    # Convert nanoctrl_address to full URL if needed
                    server_url = self.nanoctrl_address
                    if not server_url.startswith(
                        "http://"
                    ) and not server_url.startswith("https://"):
                        server_url = f"http://{server_url}"

                    # Note: Control plane should handle concurrent registrations gracefully

                    try:
                        self._peer_agent = start_peer_agent_fn(
                            alias=agent_alias,
                            server_url=server_url,
                            device=None,  # Auto-select
                            ib_port=1,
                            link_type="RoCE",
                            qp_num=1,
                        )
                        # Store agent alias as peer_agent_addr (for control plane, alias is the identifier)
                        self._peer_agent_addr = agent_alias

                        kv_size = self.kv_cache.numel() * self.kv_cache.itemsize
                        self._local_mr_handler = (
                            self._peer_agent.register_memory_region(
                                _KV_CACHE_BUFFER_ID,
                                self.kv_cache.data_ptr()
                                + int(self.kv_cache.storage_offset()),
                                kv_size,
                            )
                        )
                        logger.info(
                            f"PeerAgent started with alias {agent_alias} on control plane {server_url}, registered kv_cache buffer (handler={self._local_mr_handler})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to start PeerAgent with alias {agent_alias} on control plane {server_url}: {e}"
                        )
                        raise

    def get_peer_agent_addr(self) -> str | None:
        """Return the local peer agent address for this rank."""
        return self._peer_agent_addr

    def invalidate_engine_info_cache(self):
        """Invalidate the engine_info cache to force a refresh on next fetch."""
        self._engine_info_cache = None
        logger.info("Invalidated engine_info cache")

    def _fetch_engine_info_from_nanoctrl(self, engine_ids: set[str]) -> dict[str, dict]:
        """Get engine_info for specified engine_ids (cache + fetch if needed).

        This method handles all caching logic: checks cache, identifies missing IDs,
        fetches only missing ones from NanoCtrl, and updates cache.

        Uses the lightweight /get_engine_info endpoint instead of /list_engines.

        Args:
            engine_ids: Set of engine_ids to get info for.

        Returns:
            dict mapping engine_id to engine_info dict containing:
                - id, role, world_size, num_blocks, host, port, peer_addrs, etc.
        """
        import httpx

        if not engine_ids:
            return {}

        # Check cache and identify missing IDs
        engine_info_map = {}
        missing_ids = engine_ids

        if self._engine_info_cache is not None:
            cached_at, cached = self._engine_info_cache
            if time.time() - cached_at < _ENGINE_INFO_CACHE_TTL:
                # Get cached results
                engine_info_map = {
                    eid: info for eid, info in cached.items() if eid in engine_ids
                }
                missing_ids = engine_ids - cached.keys()

                if not missing_ids:
                    logger.debug(
                        f"All {len(engine_ids)} engines found in cache, no fetch needed"
                    )
                    return engine_info_map
                else:
                    logger.debug(
                        f"Cache hit for {len(engine_info_map)} engines, fetching {len(missing_ids)} missing: {missing_ids}"
                    )

        # Fetch missing engines from NanoCtrl
        if not self.nanoctrl_address:
            logger.warning(
                "nanoctrl_address not configured, returning cached results only"
            )
            return engine_info_map

        fetched_map: dict[str, dict] = {}
        url = f"http://{self.nanoctrl_address}/get_engine_info"

        try:
            with httpx.Client(timeout=5.0) as client:
                for engine_id in missing_ids:
                    try:
                        response = client.post(url, json={"engine_id": engine_id})
                        response.raise_for_status()
                        data = response.json()

                        if data.get("status") == "ok":
                            engine_info = data.get("engine_info", {})
                            if engine_info:
                                fetched_map[engine_id] = engine_info
                        else:
                            logger.warning(
                                f"get_engine_info for {engine_id} returned status: {data.get('status')}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching engine_info for {engine_id}: {e}")
                        continue

            # Update cache with newly fetched data
            if fetched_map:
                now = time.time()
                if self._engine_info_cache is not None:
                    cached_data = self._engine_info_cache[1]
                    cached_data.update(fetched_map)
                    self._engine_info_cache = (now, cached_data)
                else:
                    self._engine_info_cache = (now, fetched_map)

                logger.debug(
                    f"Fetched and cached {len(fetched_map)} engine_info: {list(fetched_map.keys())}"
                )

            # Return combined results
            engine_info_map.update(fetched_map)
            return engine_info_map

        except Exception as e:
            logger.error(f"Error fetching engine_info from NanoCtrl: {e}")
            # Return whatever we have from cache
            return engine_info_map

    def migrate(self, seqs: list[Sequence]):
        """Migrate KV cache blocks from local to remote engine using PeerAgent.

        This method uses lazy connection: it will connect to remote peer
        on-demand based on engine_info fetched from NanoCtrl.

        Engine info is fetched on-demand (only when needed) and cached forever by default.
        Call invalidate_engine_info_cache() to force a refresh.

        Args:
            seqs: List of sequences to migrate
        """
        logger.debug(f"migrate called with {len(seqs)} sequences")

        if self._peer_agent is None:
            logger.error("migrate called but PeerAgent not initialized")
            return

        # First, collect target engine_ids from sequences (truly on-demand)
        target_engine_ids = set()
        for seq in seqs:
            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            engine_id = migrate_ctx.engine_id
            if engine_id:
                target_engine_ids.add(engine_id)

        # Only get engine_info if we have target engines (truly on-demand)
        if not target_engine_ids:
            logger.debug("No target engine_ids found, skipping migration")
            return

        # Get engine_info (handles caching internally)
        engine_info_map = self._fetch_engine_info_from_nanoctrl(target_engine_ids)

        assigns = defaultdict(lambda: defaultdict(list))
        sp_idx = get_dist_context().attn_sp_rank

        # Collect all unique remote peer aliases from engine_info and ensure connections
        # engine_info format: engine_id -> {peer_addrs: [list of peer_agent aliases], ...}
        remote_peers_to_connect: dict[str, str] = {}  # peer_alias -> engine_id
        for seq in seqs:
            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            engine_id = migrate_ctx.engine_id
            engine_info = engine_info_map.get(engine_id, {})
            peer_aliases = engine_info.get("peer_addrs", [])
            if peer_aliases:
                for peer_alias in peer_aliases:
                    if peer_alias and peer_alias not in self._connected_peers:
                        self.num_remote_kvcache_blocks[engine_id] = (
                            migrate_ctx.num_kvcache_blocks
                        )
                        remote_peers_to_connect[peer_alias] = engine_id

        # Declarative: set desired topology and wait for reconciliation
        # TopologyReconciler converges Actual State to Desired State (Symmetric Rendezvous)
        if remote_peers_to_connect:
            new_peers = list(remote_peers_to_connect.keys())
            logger.info(f"Batch connecting to {len(new_peers)} peers: {new_peers}")

            # Merge with existing desired peers (set_desired_topology replaces entire spec)
            all_desired = set(self._connected_peers) | set(new_peers)
            # symmetric=True: NanoCtrl propagates our alias to each target's spec so both sides reconcile
            self._peer_agent.set_desired_topology(
                target_peers=list(all_desired), symmetric=True
            )

            # Wait for new peers to be connected by TopologyReconciler
            self._peer_agent.wait_for_peers(new_peers, timeout_sec=30)

            self._connected_peers.update(new_peers)
            logger.info(f"Batch connection completed for {len(new_peers)} peers")

        # Build assignment list for each remote endpoint
        for seq in seqs:
            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            active_ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
            engine_id = migrate_ctx.engine_id
            engine_info = engine_info_map.get(engine_id, {})
            peer_addrs = engine_info.get("peer_addrs", [])
            if not peer_addrs:
                logger.warning(
                    f"Sequence {seq.seq_id} has no peer_addrs for engine {engine_id}"
                )
                continue

            # Log block_location alignment for debugging
            logger.info(
                f"Sequence {seq.seq_id}: migrate_ctx.block_location={list(migrate_ctx.block_location)}, "
                f"active_ctx.block_location={list(active_ctx.block_location)}, "
                f"length: migrate={len(migrate_ctx.block_location)}, active={len(active_ctx.block_location)}"
            )

            # Ensure block_location lists have the same length
            # block_location is already a mapping from prefill to decode, no need to sort
            if len(migrate_ctx.block_location) != len(active_ctx.block_location):
                logger.error(
                    f"Sequence {seq.seq_id}: block_location length mismatch! "
                    f"migrate={len(migrate_ctx.block_location)}, active={len(active_ctx.block_location)}"
                )
                continue

            for block_pos, (remote_block_idx, source_block_idx) in enumerate(
                zip(
                    migrate_ctx.block_location,
                    active_ctx.block_location,
                )
            ):
                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        # Only process blocks where source_block_idx.first matches this rank's sp_idx
                        # This ensures each block is only processed by the correct rank
                        if source_block_idx.first != sp_idx:
                            # Skip blocks that don't belong to this rank
                            continue

                        # Calculate remote_rank for blocks that belong to this rank
                        remote_rank = (
                            seq.dp_idx(BlockContextSlot.MIGRATE)
                            * migrate_ctx.attention_sp
                            + remote_block_idx.first
                        )

                        # Get remote peer alias for this rank (format: EngineName:rank)
                        if remote_rank < len(peer_addrs):
                            peer_alias = peer_addrs[remote_rank]
                        else:
                            logger.error(
                                f"remote_rank {remote_rank} >= len(peer_addrs) {len(peer_addrs)}"
                            )
                            continue

                        assignment = (
                            peer_alias,
                            kv_idx,
                            layer_idx,
                            remote_block_idx.second,
                            source_block_idx.second,
                        )
                        assigns[engine_id][peer_alias].append(assignment)

        # Execute RDMA reads using PeerAgent control plane API
        # Cache remote MR handlers to avoid re-registering (as in reference example)
        remote_mr_handlers = {}  # (peer_alias, mr_name) -> handler

        for engine_id, peer_assigns in assigns.items():
            for peer_alias, assign_batch in peer_assigns.items():
                # Check if peer is connected
                if peer_alias not in self._connected_peers:
                    logger.error(f"Peer {peer_alias} not connected, skipping")
                    continue

                # Get or cache remote MR handler
                cache_key = (peer_alias, _KV_CACHE_BUFFER_ID)
                if cache_key not in remote_mr_handlers:
                    # Get remote MR info from control plane (as in reference example)
                    remote_mr_info = self._peer_agent.get_mr_info(
                        peer_alias, _KV_CACHE_BUFFER_ID
                    )
                    if remote_mr_info is None:
                        logger.error(f"Failed to get MR info for {peer_alias}")
                        continue

                    # Register remote memory region (as in reference example)
                    remote_mr_handler = self._peer_agent.register_remote_memory_region(
                        peer_alias,
                        _KV_CACHE_BUFFER_ID,
                        remote_mr_info,
                    )
                    remote_mr_handlers[cache_key] = remote_mr_handler
                    logger.info(
                        f"Registered remote MR for {peer_alias}: handler={remote_mr_handler}, "
                        f"local_handler={self._local_mr_handler}, cache_key={cache_key}"
                    )
                else:
                    remote_mr_handler = remote_mr_handlers[cache_key]

                # Get local MR handler (stored during allocate_kvcache)
                if self._local_mr_handler is None:
                    logger.error(
                        f"Local MR handler not available for {_KV_CACHE_BUFFER_ID}"
                    )
                    continue
                local_mr_handler = self._local_mr_handler

                # Get endpoint for this peer (as in reference example)
                endpoint = self._peer_agent.get_endpoint(peer_alias)
                if endpoint is None:
                    logger.error(f"Failed to get endpoint for {peer_alias}")
                    continue

                # Build batch of RDMA read operations
                # endpoint.read accepts list of (local_handler, remote_handler, remote_off, local_off, length)
                rdma_ops: list[tuple] = []
                for op_idx, (
                    _peer_alias,
                    kv_idx,
                    layer_idx,
                    remote_block_idx,
                    source_block_idx,
                ) in enumerate(assign_batch):
                    local_off = self.local_kv_stride(
                        kv_idx, layer_idx, source_block_idx
                    )
                    remote_off = self.remote_kv_stride(
                        kv_idx, layer_idx, remote_block_idx, engine_id
                    )
                    length = self.block_stride(1)

                    if local_mr_handler is None or remote_mr_handler is None:
                        logger.error(
                            f"[Op {op_idx}] Invalid MR handlers: local={local_mr_handler}, remote={remote_mr_handler}"
                        )
                        continue
                    if local_off < 0 or remote_off < 0 or length <= 0:
                        logger.error(
                            f"[Op {op_idx}] Invalid offsets/length: local_off={local_off}, remote_off={remote_off}, length={length}"
                        )
                        continue

                    rdma_ops.append(
                        (
                            local_mr_handler,
                            remote_mr_handler,
                            remote_off,
                            local_off,
                            length,
                        )
                    )

                if not rdma_ops:
                    logger.error(f"No valid RDMA ops for {peer_alias}, skipping")
                    continue

                try:
                    slot = endpoint.read(rdma_ops, None)
                    if slot is None:
                        logger.error("endpoint.read returned None")
                        raise RuntimeError("endpoint.read returned None")
                    slot.wait()
                    logger.info(
                        f"Completed batch RDMA read from {peer_alias} ({len(rdma_ops)} operations)"
                    )
                except Exception as e:
                    logger.error(
                        f"Batch RDMA read FAILED from {peer_alias}: {len(rdma_ops)} ops, error={e}",
                        exc_info=True,
                    )
                    raise


_CACHE_CONTEXT: CacheContext


def get_cache_context():
    return _CACHE_CONTEXT


def set_cache_context(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_hidden_layers: int,
    attention_tp: int,
    gpu_memory_utilization: float,
    gpu_memory_limit_gb: float | None = None,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mode: Literal["gqa", "mla"] = "gqa",
    nanoctrl_address: str | None = None,
    engine_id: str | None = None,
):
    global _CACHE_CONTEXT
    _CACHE_CONTEXT = CacheContext(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        num_hidden_layers=num_hidden_layers,
        attention_tp=attention_tp,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_limit_gb=gpu_memory_limit_gb,
        device=device,
        dtype=dtype,
        mode=mode,
        nanoctrl_address=nanoctrl_address,
        engine_id=engine_id,
    )
    return _CACHE_CONTEXT

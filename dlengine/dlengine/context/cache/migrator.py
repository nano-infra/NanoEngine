"""KV-cache migration / transfer (RDMA + NanoCtrl control plane).

Extracted from ``CacheContext`` and mixed back in. Owns the
PD-disaggregation transfer path: peer-agent memory-region registration,
NanoCtrl engine lookup, peer connection setup, and the batched RDMA reads
that pull KV / GDN / indexer / DSv4 state from a remote prefill engine.

This module is the seam where alternative KV transfer backends (e.g. a
3FS-backed KV store) should plug in.
"""

import time
from collections import defaultdict

import torch
from dlengine.context.distributed import get_dist_context
from dlengine.context.peer_agent import PeerAgentContext
from dlengine.logging import get_logger

logger = get_logger("dlengine")

# PeerAgent path: buffer ID for kv_cache registration
_KV_CACHE_BUFFER_ID = "kv_cache"

# Cache TTL for engine_info from NanoCtrl (seconds); inf = never expire.
_ENGINE_INFO_CACHE_TTL = float("inf")


class KVMigratorMixin:
    def set_peer_agent_context(self, peer_context: PeerAgentContext | None) -> None:
        """Attach the worker-owned PeerAgentContext to cache RDMA users."""
        self.peer_agent_context = peer_context

    def register_peer_agent_memory_regions(self, mode: str = "hybrid") -> None:
        """Register cache-owned RDMA memory regions on the attached PeerAgent.

        Must be called AFTER allocate_kvcache() and allocate_gdn_states() so that
        all tensors exist before registration. In hybrid mode the PeerAgent is
        still alive, but KV cache / GDN MR registration is skipped because
        hybrid mode does not perform P2P KV transfer.
        """
        peer_context = self.peer_agent_context
        if peer_context is None:
            return

        agent_alias = peer_context.alias
        server_url = peer_context.server_url
        peer_agent = peer_context.agent

        try:
            # In hybrid mode we only need the PeerAgent alive (for vision
            # embed RDMA fetch); KV cache / GDN MR registration is not needed.
            if mode == "hybrid":
                logger.info(
                    f"PeerAgent started (hybrid, no KV MR): alias={agent_alias}, "
                    f"server={server_url}"
                )
                return

            # Register KV cache
            kv_size = self.kv_cache.numel() * self.kv_cache.itemsize
            self._local_mr_handler = peer_agent.register_memory_region(
                _KV_CACHE_BUFFER_ID,
                self.kv_cache.data_ptr(),
                int(self.kv_cache.storage_offset()),
                kv_size,
            )
            logger.info(
                f"PeerAgent started: alias={agent_alias}, server={server_url}, "
                f"kv_cache MR handler={self._local_mr_handler}"
            )

            # Register GDN states (if allocated)
            if (
                self.gdn_conv_states is not None
                and self.gdn_recurrent_states is not None
            ):
                conv_size = self.gdn_conv_states.numel() * self.gdn_conv_states.itemsize
                self._local_gdn_conv_mr_handler = peer_agent.register_memory_region(
                    "gdn_conv",
                    self.gdn_conv_states.data_ptr(),
                    int(self.gdn_conv_states.storage_offset()),
                    conv_size,
                )
                recurrent_size = (
                    self.gdn_recurrent_states.numel()
                    * self.gdn_recurrent_states.itemsize
                )
                self._local_gdn_recurrent_mr_handler = (
                    peer_agent.register_memory_region(
                        "gdn_recurrent",
                        self.gdn_recurrent_states.data_ptr(),
                        int(self.gdn_recurrent_states.storage_offset()),
                        recurrent_size,
                    )
                )
                logger.info(
                    f"Registered GDN MRs: conv={self._local_gdn_conv_mr_handler}, "
                    f"recurrent={self._local_gdn_recurrent_mr_handler}"
                )

            # Register IndexerCache (if allocated, V3.2 sparse attention)
            if self.indexer_cache is not None:
                indexer_buf = self.indexer_cache.buffer
                indexer_size = indexer_buf.numel() * indexer_buf.itemsize
                self._local_indexer_mr_handler = peer_agent.register_memory_region(
                    "indexer_cache",
                    indexer_buf.data_ptr(),
                    0,
                    indexer_size,
                )
                logger.info(
                    f"Registered IndexerCache MR: handler={self._local_indexer_mr_handler}"
                )

            # DSv4 (S2.4): register flat per-ratio compressed cache + compressor
            # state buffers — one MR per ratio per kind. Skipped in hybrid mode
            # via the same outer guard that protects KV/GDN registration.
            for ratio, buf in (
                getattr(self, "dsv4_compressed_caches_flat", None) or {}
            ).items():
                handler = peer_agent.register_memory_region(
                    f"dsv4_compressed_r{ratio}",
                    buf.data_ptr(),
                    int(buf.storage_offset()),
                    buf.numel() * buf.itemsize,
                )
                self._local_dsv4_compressed_mr_handlers[ratio] = handler
                logger.info(
                    f"Registered DSv4 compressed cache MR: ratio={ratio}, "
                    f"handler={handler}, size={buf.numel() * buf.itemsize / 1e9:.2f} GB"
                )

            for ratio, buf in (
                getattr(self, "dsv4_compressor_kv_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_kv_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_kv_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            for ratio, buf in (
                getattr(self, "dsv4_compressor_score_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_score_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_score_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            for ratio, buf in (
                getattr(self, "dsv4_compressor_counts_flat", None) or {}
            ).items():
                self._local_dsv4_compressor_counts_mr_handlers[ratio] = (
                    peer_agent.register_memory_region(
                        f"dsv4_compressor_counts_r{ratio}",
                        buf.data_ptr(),
                        int(buf.storage_offset()),
                        buf.numel() * buf.itemsize,
                    )
                )
            if self._local_dsv4_compressor_kv_mr_handlers:
                logger.info(
                    f"Registered DSv4 compressor scratch MRs: "
                    f"kv={self._local_dsv4_compressor_kv_mr_handlers}, "
                    f"score={self._local_dsv4_compressor_score_mr_handlers}, "
                    f"counts={self._local_dsv4_compressor_counts_mr_handlers}"
                )

        except Exception as e:
            logger.error(f"Failed to register PeerAgent memory regions: {e}")
            raise

    def get_peer_agent_addr(self) -> str | None:
        """Return the local peer agent address for this rank."""
        return (
            None if self.peer_agent_context is None else self.peer_agent_context.alias
        )

    def get_peer_agent_context(self) -> PeerAgentContext:
        """Return the attached worker-owned PeerAgentContext."""
        if self.peer_agent_context is None:
            raise RuntimeError(
                "CacheContext PeerAgentContext is not attached. "
                "Was ModelRunner PeerAgentContext initialized?"
            )
        return self.peer_agent_context

    def ensure_peer_agent_connected(self, peer_alias: str) -> None:
        """Ensure the local PeerAgent is connected to ``peer_alias``."""
        self.get_peer_agent_context().ensure_connected(peer_alias)

    def invalidate_engine_info_cache(self):
        """Invalidate the engine_info cache to force a refresh on next fetch."""
        self._engine_info_cache = None
        logger.info("Invalidated engine_info cache")

    def _fetch_engine_info_from_ctrl(self, engine_ids: set[str]) -> dict[str, dict]:
        """Get engine_info for specified engine_ids (cache + fetch if needed).

        This method handles all caching logic: checks cache, identifies missing IDs,
        fetches only missing ones from NanoCtrl, and updates cache.

        Uses the lightweight /get_entity_info endpoint instead of /list_entities.

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
        if not self.ctrl_address:
            logger.warning("ctrl_address not configured, returning cached results only")
            return engine_info_map

        fetched_map: dict[str, dict] = {}
        url = f"{self.ctrl_address}/get_entity_info"
        scope = self.ctrl_scope or ""

        try:
            # trust_env=False: ignore HTTP(S)_PROXY/ALL_PROXY env vars. NanoCtrl
            # is an internal address; routing it through a cluster proxy makes
            # the request hang (httpx defaults to trust_env=True). This matches
            # dlslime's NanoCtrlClient, which also uses trust_env=False.
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                for engine_id in missing_ids:
                    try:
                        request_payload = {
                            "entity_type": "service",
                            "entity_id": engine_id,
                        }
                        if scope:
                            request_payload["scope"] = scope

                        response = client.post(url, json=request_payload)
                        response.raise_for_status()
                        data = response.json()

                        if data.get("status") == "ok":
                            entity_info = data.get("entity_info") or {}
                            engine_info = dict(entity_info.get("metadata") or {})
                            if engine_info:
                                engine_info.setdefault(
                                    "id",
                                    entity_info.get("entity_id", engine_id),
                                )
                                fetched_map[engine_id] = engine_info
                        else:
                            logger.warning(
                                f"get_entity_info for {engine_id} returned status: {data.get('status')}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching entity_info for {engine_id}: {e}")
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

    # ------------------------------------------------------------------
    # Shared migration helpers
    # ------------------------------------------------------------------

    def _remote_global_rank(
        self,
        dp_idx: int,
        sp_idx: int,
        sp_size: int,
        tp_idx: int,
        tp_size: int,
    ) -> int:
        """Map a remote (dp, sp, tp) cell to its global rank in ``peer_addrs``.

        ``peer_addrs`` is ordered by the attention device-mesh global rank,
        i.e. ``dp_idx * (sp_size * tp_size) + sp_idx * tp_size + tp_idx``.
        For attention_tp == 1 this reduces to ``dp_idx * sp_size + sp_idx``,
        preserving the previous (TP=1 / MLA) behavior.
        """
        return (dp_idx * sp_size + sp_idx) * tp_size + tp_idx

    def _ensure_peer_connections(
        self, connection_requests: list[tuple[str, str, int, int, int]]
    ) -> None:
        """Establish connections to remote peers if not already connected.

        Args:
            connection_requests: list of (peer_alias, engine_id, num_kvcache_blocks,
                                          max_num_seqs, gdn_num_slots)
        """
        remote_peers_to_connect: dict[str, str] = {}
        peer_context = self.get_peer_agent_context()
        for (
            peer_alias,
            engine_id,
            num_kvcache_blocks,
            max_num_seqs,
            gdn_num_slots,
        ) in connection_requests:
            if peer_alias and not peer_context.is_connected(peer_alias):
                self.num_remote_kvcache_blocks[engine_id] = num_kvcache_blocks
                self.remote_max_num_seqs[engine_id] = max_num_seqs
                if gdn_num_slots > 0:
                    self.remote_gdn_num_slots[engine_id] = gdn_num_slots
                remote_peers_to_connect[peer_alias] = engine_id

        if not remote_peers_to_connect:
            return

        new_peers = list(remote_peers_to_connect.keys())
        logger.info(f"Batch connecting to {len(new_peers)} peers: {new_peers}")
        connected = peer_context.ensure_many_connected(new_peers)
        logger.info(f"Batch connection completed for {len(connected)} peers")

    def _execute_rdma_reads(
        self,
        assigns: dict[str, dict[str, list[tuple]]],
        gdn_assigns: dict[str, dict[str, list[tuple]]],
        indexer_assigns: dict[str, dict[str, list[tuple]]] | None = None,
        compressed_assigns: dict[str, dict[str, list[tuple]]] | None = None,
        compressor_state_assigns: dict[str, dict[str, list[tuple]]] | None = None,
    ) -> None:
        """Execute batched RDMA reads for KV cache and GDN state migration.

        Args:
            assigns: engine_id -> peer_alias -> list of
                     (peer_alias, kv_idx, layer_idx, remote_block_idx, source_block_idx)
            gdn_assigns: engine_id -> peer_alias -> list of
                         (layer_idx, remote_state_slot, local_state_slot)
            indexer_assigns: engine_id -> peer_alias -> list of
                             (layer_idx, remote_block_idx, source_block_idx)
            compressed_assigns: engine_id -> peer_alias -> list of
                                (ratio, ratio_layer_idx, remote_page_idx, local_page_idx)
            compressor_state_assigns: engine_id -> peer_alias -> list of
                                (ratio, ratio_layer_idx, remote_state_slot, local_state_slot)
        """
        peer_context = self.get_peer_agent_context()
        peer_agent = peer_context.agent
        for engine_id, peer_assigns in assigns.items():
            for peer_alias, assign_batch in peer_assigns.items():
                if not peer_context.is_connected(peer_alias):
                    logger.error(f"Peer {peer_alias} not connected, skipping")
                    continue

                conn = peer_agent.query_connection(peer_alias)
                if conn is None or conn.endpoint is None:
                    logger.error(f"Failed to get endpoint for {peer_alias}")
                    continue
                endpoint = conn.endpoint

                remote_mr_info = peer_agent.get_mr_info(peer_alias, _KV_CACHE_BUFFER_ID)
                if remote_mr_info is None:
                    logger.error(f"Failed to get MR info for {peer_alias}")
                    continue

                remote_mr_handler = peer_agent.get_handle(
                    _KV_CACHE_BUFFER_ID, peer_alias=peer_alias
                )
                logger.debug(
                    f"Remote MR for {peer_alias}: handler={remote_mr_handler}, "
                    f"local_handler={self._local_mr_handler}"
                )

                if self._local_mr_handler is None:
                    logger.error(
                        f"Local MR handler not available for {_KV_CACHE_BUFFER_ID}"
                    )
                    continue
                local_mr_handler = self._local_mr_handler

                # Build KV cache RDMA ops
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

                # Append GDN state RDMA ops
                gdn_batch = gdn_assigns.get(engine_id, {}).get(peer_alias, [])

                if (
                    gdn_batch
                    and self.gdn_conv_states is not None
                    and self.gdn_recurrent_states is not None
                ):
                    # Conv state
                    remote_conv_mr_info = peer_agent.get_mr_info(peer_alias, "gdn_conv")
                    if remote_conv_mr_info:
                        remote_conv_mr = peer_agent.get_handle(
                            "gdn_conv", peer_alias=peer_alias
                        )
                        local_conv_mr = self._local_gdn_conv_mr_handler
                        conv_len = self.gdn_conv_slot_num_bytes()
                        for layer_idx, remote_slot, local_slot in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_conv_mr,
                                    remote_conv_mr,
                                    self.remote_gdn_conv_stride(
                                        layer_idx, remote_slot, engine_id
                                    ),
                                    self.gdn_conv_stride(layer_idx, local_slot),
                                    conv_len,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get gdn_conv MR info for {peer_alias}"
                        )

                    # Recurrent state
                    remote_rec_mr_info = peer_agent.get_mr_info(
                        peer_alias, "gdn_recurrent"
                    )
                    if remote_rec_mr_info:
                        remote_rec_mr = peer_agent.get_handle(
                            "gdn_recurrent", peer_alias=peer_alias
                        )
                        local_rec_mr = self._local_gdn_recurrent_mr_handler
                        rec_len = self.gdn_recurrent_slot_num_bytes()
                        for layer_idx, remote_slot, local_slot in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_rec_mr,
                                    remote_rec_mr,
                                    self.remote_gdn_recurrent_stride(
                                        layer_idx, remote_slot, engine_id
                                    ),
                                    self.gdn_recurrent_stride(layer_idx, local_slot),
                                    rec_len,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get gdn_recurrent MR info for {peer_alias}"
                        )

                # Append DSv4 compressed cache + compressor scratch state RDMA ops.
                comp_batch = (
                    (compressed_assigns or {}).get(engine_id, {}).get(peer_alias, [])
                )
                if comp_batch:
                    # Group by ratio to register the right MR per ratio.
                    by_ratio: dict[int, list[tuple]] = {}
                    for r, rli, rpage, lpage in comp_batch:
                        by_ratio.setdefault(r, []).append((rli, rpage, lpage))
                    for ratio, ops in by_ratio.items():
                        local_handler = self._local_dsv4_compressed_mr_handlers.get(
                            ratio
                        )
                        if local_handler is None:
                            continue
                        remote_info = peer_agent.get_mr_info(
                            peer_alias, f"dsv4_compressed_r{ratio}"
                        )
                        if not remote_info:
                            logger.warning(
                                f"Failed to get DSv4 compressed MR info for {peer_alias}, ratio={ratio}"
                            )
                            continue
                        remote_handler = peer_agent.get_handle(
                            f"dsv4_compressed_r{ratio}", peer_alias=peer_alias
                        )
                        page_bytes = self.compressed_page_bytes(ratio)
                        for rli, rpage, lpage in ops:
                            rdma_ops.append(
                                (
                                    local_handler,
                                    remote_handler,
                                    self.remote_compressed_stride(
                                        ratio, rli, rpage, engine_id
                                    ),
                                    self.local_compressed_stride(ratio, rli, lpage),
                                    page_bytes,
                                )
                            )

                cstate_batch = (
                    (compressor_state_assigns or {})
                    .get(engine_id, {})
                    .get(peer_alias, [])
                )
                if cstate_batch:
                    by_ratio_s: dict[int, list[tuple]] = {}
                    for r, rli, rslot, lslot in cstate_batch:
                        by_ratio_s.setdefault(r, []).append((rli, rslot, lslot))
                    for ratio, ops in by_ratio_s.items():
                        for kind, local_map in (
                            ("kv", self._local_dsv4_compressor_kv_mr_handlers),
                            ("score", self._local_dsv4_compressor_score_mr_handlers),
                            ("counts", self._local_dsv4_compressor_counts_mr_handlers),
                        ):
                            local_handler = local_map.get(ratio)
                            if local_handler is None:
                                continue
                            mr_name = f"dsv4_compressor_{kind}_r{ratio}"
                            remote_info = peer_agent.get_mr_info(peer_alias, mr_name)
                            if not remote_info:
                                logger.warning(
                                    f"Failed to get {mr_name} MR info for {peer_alias}"
                                )
                                continue
                            remote_handler = peer_agent.get_handle(
                                mr_name, peer_alias=peer_alias
                            )
                            row_bytes = self._compressor_state_row_bytes(ratio, kind)
                            for rli, rslot, lslot in ops:
                                rdma_ops.append(
                                    (
                                        local_handler,
                                        remote_handler,
                                        self.remote_compressor_state_stride(
                                            ratio, rli, rslot, kind, engine_id
                                        ),
                                        self.local_compressor_state_stride(
                                            ratio, rli, lslot, kind
                                        ),
                                        row_bytes,
                                    )
                                )

                # Append IndexerCache RDMA ops (V3.2 sparse attention)
                indexer_batch = (
                    (indexer_assigns or {}).get(engine_id, {}).get(peer_alias, [])
                )
                if indexer_batch and self.indexer_cache is not None:
                    remote_indexer_mr_info = peer_agent.get_mr_info(
                        peer_alias, "indexer_cache"
                    )
                    if remote_indexer_mr_info:
                        remote_indexer_mr = peer_agent.get_handle(
                            "indexer_cache", peer_alias=peer_alias
                        )
                        local_indexer_mr = self._local_indexer_mr_handler
                        page_bytes = self.indexer_page_num_bytes()
                        for layer_idx, remote_block, local_block in indexer_batch:
                            rdma_ops.append(
                                (
                                    local_indexer_mr,
                                    remote_indexer_mr,
                                    self.remote_indexer_stride(
                                        layer_idx, remote_block, engine_id
                                    ),
                                    self.local_indexer_stride(layer_idx, local_block),
                                    page_bytes,
                                )
                            )
                    else:
                        logger.warning(
                            f"Failed to get indexer_cache MR info for {peer_alias}"
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
                    # GPUDirect RDMA may bypass CUDA stream ordering.
                    # Synchronize to ensure migrated KV data is visible to subsequent kernels.
                    torch.cuda.synchronize()

                    logger.info(
                        f"Completed batch RDMA read from {peer_alias} ({len(rdma_ops)} operations)"
                    )
                except Exception as e:
                    logger.error(
                        f"Batch RDMA read FAILED from {peer_alias}: {len(rdma_ops)} ops, error={e}",
                        exc_info=True,
                    )
                    raise

    def migrate_from_bytes(self, data: bytes):
        """Migrate KV cache using lean MigrateBatchInput protocol (no Sequence objects)."""
        from dlengine._cpp import parse_migrate_batch

        views = parse_migrate_batch(data)

        if self.peer_agent_context is None:
            logger.error("migrate_from_bytes called but PeerAgent not initialized")
            return

        # Collect target engine_ids
        target_engine_ids = set()
        for v in views:
            if v.migrate_engine_id:
                target_engine_ids.add(v.migrate_engine_id)

        if not target_engine_ids:
            logger.debug("No target engine_ids found, skipping migration")
            return

        engine_info_map = self._fetch_engine_info_from_ctrl(target_engine_ids)

        # Ensure connections
        connection_requests: list[tuple[str, str, int, int]] = []
        for v in views:
            engine_id = v.migrate_engine_id
            engine_info = engine_info_map.get(engine_id, {})
            remote_max_num_seqs = engine_info.get("max_num_seqs", 0)
            remote_gdn_num_slots = engine_info.get("gdn_num_slots", 0)
            # PD + GQA: remember the remote engine's attention_tp so peer
            # selection can address the matching per-rank KV-head shard.
            self.remote_attention_tp[engine_id] = int(
                engine_info.get("attention_tp", 1)
            )
            # The RDMA block-copy migrates whole (layer, block) regions whose
            # byte size depends on num_local_kv_heads. That only lines up when
            # prefill and decode shard KV heads identically; otherwise a decode
            # rank would copy bytes that belong to a different head subset.
            remote_nlkv = int(
                engine_info.get("num_local_kv_heads", self.num_local_kv_heads)
            )
            if remote_nlkv != self.num_local_kv_heads:
                logger.error(
                    f"KV-head shard mismatch for engine {engine_id}: "
                    f"remote num_local_kv_heads={remote_nlkv}, "
                    f"local={self.num_local_kv_heads}. PD KV migration requires "
                    f"matching attention_tp / KV-head sharding between prefill "
                    f"and decode engines."
                )
            # DSv4 (S2.5): record remote pool sizes for stride math.
            remote_dsv4_pools = engine_info.get("dsv4_compressed_pool_pages", {}) or {}
            # Keys may be strings (JSON) — coerce to int.
            self.remote_compressed_pool_pages[engine_id] = {
                int(r): int(p) for r, p in remote_dsv4_pools.items()
            }
            self.remote_dsv4_max_slots[engine_id] = int(
                engine_info.get("dsv4_max_slots", remote_max_num_seqs)
            )
            remote_layers = engine_info.get("dsv4_num_layers_per_ratio", {}) or {}
            self.remote_dsv4_num_layers_per_ratio[engine_id] = {
                int(r): int(n) for r, n in remote_layers.items()
            }
            for peer_alias in engine_info.get("peer_addrs", []):
                connection_requests.append(
                    (
                        peer_alias,
                        engine_id,
                        v.migrate_num_kvcache_blocks,
                        remote_max_num_seqs,
                        remote_gdn_num_slots,
                    )
                )
        self._ensure_peer_connections(connection_requests)

        # Build assignment list
        assigns = defaultdict(lambda: defaultdict(list))
        gdn_assigns = defaultdict(lambda: defaultdict(list))
        indexer_assigns = defaultdict(lambda: defaultdict(list))
        # DSv4 (S2.6): per-ratio compressed pages + compressor scratch state.
        compressed_assigns = defaultdict(lambda: defaultdict(list))
        compressor_state_assigns = defaultdict(lambda: defaultdict(list))
        dist_ctx = get_dist_context()
        sp_idx = dist_ctx.attn_sp_rank
        # Local TP rank. Each decode TP rank owns a distinct KV-head shard and
        # reads it from the prefill rank holding the same shard (same tp_idx).
        tp_idx = dist_ctx.attn_tp_rank

        for v in views:
            engine_id = v.migrate_engine_id
            engine_info = engine_info_map.get(engine_id, {})
            peer_addrs = engine_info.get("peer_addrs", [])
            if not peer_addrs:
                logger.warning(
                    f"Sequence {v.seq_id} has no peer_addrs for engine {engine_id}"
                )
                continue

            if len(v.migrate_block_location) > len(v.active_block_location):
                logger.error(
                    f"Sequence {v.seq_id}: migrate has MORE blocks than active! "
                    f"migrate={len(v.migrate_block_location)}, active={len(v.active_block_location)}"
                )
                continue
            if len(v.migrate_block_location) < len(v.active_block_location):
                # Expected when prompt_tokens % block_size == 0: prefill serializes N blocks
                # for prompt KV, but decode allocates N+1 blocks for (prompt+1) total tokens.
                # zip() below naturally iterates only over the migrate (shorter) side;
                # the extra active block will be filled during the first decode step.
                logger.info(
                    f"Sequence {v.seq_id}: partial migration "
                    f"(migrate={len(v.migrate_block_location)}, active={len(v.active_block_location)})"
                )

            for remote_bl, source_bl in zip(
                v.migrate_block_location, v.active_block_location
            ):
                remote_sp_idx, remote_block_idx = remote_bl
                source_sp_idx, source_block_idx = source_bl

                # Validate block indices
                if (
                    source_block_idx < 0
                    or source_block_idx >= self.num_local_kvcache_blocks
                ):
                    logger.error(
                        f"Sequence {v.seq_id}: source_block_idx {source_block_idx} "
                        f"out of range [0, {self.num_local_kvcache_blocks})"
                    )
                    continue
                remote_max = self.num_remote_kvcache_blocks.get(engine_id, 0)
                if remote_block_idx < 0 or (
                    remote_max > 0 and remote_block_idx >= remote_max
                ):
                    logger.error(
                        f"Sequence {v.seq_id}: remote_block_idx {remote_block_idx} "
                        f"out of range [0, {remote_max})"
                    )
                    continue

                if source_sp_idx != sp_idx:
                    continue

                remote_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    remote_sp_idx,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )

                if remote_rank >= len(peer_addrs):
                    logger.error(
                        f"remote_rank {remote_rank} >= len(peer_addrs) {len(peer_addrs)}"
                    )
                    continue
                peer_alias = peer_addrs[remote_rank]

                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        assigns[engine_id][peer_alias].append(
                            (
                                peer_alias,
                                kv_idx,
                                layer_idx,
                                remote_block_idx,
                                source_block_idx,
                            )
                        )

                # Indexer cache assignments (V3.2 sparse attention)
                if self.indexer_cache is not None:
                    for layer_idx in range(self.num_hidden_layers):
                        indexer_assigns[engine_id][peer_alias].append(
                            (layer_idx, remote_block_idx, source_block_idx)
                        )

            # DSv4 (S2.6): per-ratio compressed cache + compressor state migration.
            # migrate_compressed_block_tables[ratio] = list of remote page IDs.
            # active_compressed_block_tables[ratio]  = list of local page IDs allocated by
            #                                          the decode engine's GroupManager.
            if (
                getattr(self, "dsv4_compressed_caches_flat", None)
                and v.migrate_compressed_block_tables
            ):
                remote_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    v.migrate_group_size - 1,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )
                if 0 <= remote_rank < len(peer_addrs):
                    peer_alias = peer_addrs[remote_rank]
                    for (
                        ratio,
                        remote_pages,
                    ) in v.migrate_compressed_block_tables.items():
                        if ratio not in self.dsv4_compressed_caches_flat:
                            continue  # decode engine doesn't have this ratio (mismatch)
                        local_pages = v.active_compressed_block_tables.get(ratio, [])
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        # Pair-wise remote→local page mapping; for each (rli) layer, the
                        # SAME (remote_page, local_page) pairs apply (the table is per-seq,
                        # not per-layer).
                        for ratio_layer_idx in range(n_layers):
                            for rpage, lpage in zip(remote_pages, local_pages):
                                compressed_assigns[engine_id][peer_alias].append(
                                    (ratio, ratio_layer_idx, rpage, lpage)
                                )

            # DSv4 (S2.6): per-ratio compressor scratch state migration.
            if (
                getattr(self, "dsv4_compressor_kv_flat", None)
                and v.migrate_state_slot >= 0
                and v.active_state_slot >= 0
            ):
                remote_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    v.migrate_group_size - 1,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )
                if 0 <= remote_rank < len(peer_addrs):
                    peer_alias = peer_addrs[remote_rank]
                    for ratio in self.dsv4_compressor_kv_flat.keys():
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        for ratio_layer_idx in range(n_layers):
                            compressor_state_assigns[engine_id][peer_alias].append(
                                (
                                    ratio,
                                    ratio_layer_idx,
                                    v.migrate_state_slot,
                                    v.active_state_slot,
                                )
                            )

            # GDN assignments
            if (
                self.gdn_conv_states is not None
                and self.gdn_recurrent_states is not None
            ):
                remote_state_slot = v.migrate_state_slot
                local_state_slot = v.active_state_slot

                if remote_state_slot >= 0 and local_state_slot >= 0:
                    remote_rank = self._remote_global_rank(
                        v.migrate_dp_idx,
                        v.migrate_group_size - 1,
                        v.migrate_group_size,
                        tp_idx,
                        self.remote_attention_tp.get(engine_id, 1),
                    )
                    if remote_rank < len(peer_addrs):
                        peer_alias = peer_addrs[remote_rank]
                        num_gdn_layers = self.gdn_recurrent_states.shape[0]
                        for layer_idx in range(num_gdn_layers):
                            gdn_assigns[engine_id][peer_alias].append(
                                (
                                    layer_idx,
                                    remote_state_slot,
                                    local_state_slot,
                                )
                            )

        self._execute_rdma_reads(
            assigns,
            gdn_assigns,
            indexer_assigns,
            compressed_assigns=compressed_assigns,
            compressor_state_assigns=compressor_state_assigns,
        )

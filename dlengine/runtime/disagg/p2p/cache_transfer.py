"""KV-cache P2P migration / transfer (RDMA + NanoCtrl control plane).

Owns the PD-disaggregation transfer path: peer-agent memory-region
registration, NanoCtrl engine lookup, peer connection setup, and the batched
RDMA reads that pull KV / GDN / indexer / DSv4 state from a remote prefill
engine. Cache state remains in ``context.cache``; P2P-specific byte-offset
math lives next to the transfer path.

This module is intentionally P2P-specific. Storage-backed transfer backends
(e.g. 3FS) belong under ``dlengine.runtime.disagg.storage``.
"""

import time
from collections import defaultdict

import dlslime
import torch
import torch.distributed as dist

from dlengine.logging import get_logger
from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.context.peer import PeerAgentContext
from dlengine.runtime.disagg.p2p.cache_layout import P2PCacheLayout
from dlengine.runtime.models.pp_utils import (
    pp_global_rank,
    pp_layer_partition,
    pp_stage_of_layer,
)

logger = get_logger("dlengine")

# PeerAgent path: buffer ID for kv_cache registration
_KV_CACHE_BUFFER_ID = "kv_cache"
_HISPARSE_COLD_KV_BUFFER_ID = "hisparse_cold_kv"

# Cache TTL for engine_info from NanoCtrl (seconds); inf = never expire.
_ENGINE_INFO_CACHE_TTL = float("inf")


def _tensor_storage_span_num_bytes(tensor: torch.Tensor) -> int:
    """Bytes addressable from tensor.data_ptr() through its strided view.

    Non-contiguous cache views, notably FP8 MLA KV cache, intentionally keep
    padding rows in the underlying storage. RDMA offsets are computed from
    tensor strides, so the registered MR must cover the full strided span, not
    only tensor.numel().
    """
    if tensor.numel() == 0:
        return 0
    max_element_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if size > 0:
            max_element_offset += (int(size) - 1) * int(stride)
    first_element_offset = int(tensor.storage_offset())
    return (max_element_offset - first_element_offset + 1) * tensor.element_size()


def select_peer_device() -> str:
    available_nics = dlslime.available_nic()
    selected_nic_idx = dist.get_rank() % len(available_nics)
    selected_nic = available_nics[selected_nic_idx]
    assert selected_nic
    logger.info(
        "Selected PeerAgent NIC: rank=%s local_rank=%s available=%s selected=%s",
        dist.get_rank() if dist.is_initialized() else "n/a",
        get_dist_context().local_rank if dist.is_initialized() else "n/a",
        available_nics,
        selected_nic,
    )
    return selected_nic


def initialize_migration_state(context) -> None:
    context.num_remote_kvcache_blocks = {}
    context.remote_max_num_seqs = {}
    context.remote_attention_tp = {}
    context.remote_attention_dp = {}
    context.remote_pp = {}
    context.remote_num_hidden_layers = {}
    context.remote_pp_layer_ranges = {}
    context.remote_pp_cache_layer_indices = {}
    context.remote_pp_dsv4_ratio_layer_indices = {}
    context.remote_gdn_num_slots = {}
    context.remote_compressed_pool_pages = {}
    context.remote_dsv4_max_slots = {}
    context.remote_dsv4_num_layers_per_ratio = {}
    context._local_mr_handler = None
    context._local_hisparse_cold_mr_handler = None
    context._local_indexer_mr_handler = None
    context._local_gdn_conv_mr_handler = None
    context._local_gdn_recurrent_mr_handler = None
    context._local_dsv4_compressed_mr_handlers = {}
    context._local_dsv4_compressor_kv_mr_handlers = {}
    context._local_dsv4_compressor_score_mr_handlers = {}
    context._local_dsv4_compressor_counts_mr_handlers = {}
    context._engine_info_cache = None


class P2PCacheTransfer:
    """P2P cache transfer manager.

    The transfer manager owns remote-engine metadata, PeerAgent state, and local
    MR handlers. It delegates cache tensors and layout helper methods to the
    active CacheContext returned by ``get_cache_context()``.
    """

    def __init__(self) -> None:
        self.peer_agent_context: PeerAgentContext | None = None
        self.layout = P2PCacheLayout(self)
        initialize_migration_state(self)

    @property
    def cache_context(self):
        from dlengine.runtime.context.cache import get_cache_context

        return get_cache_context()

    def __getattr__(self, name):
        cache_context = self.cache_context
        class_attr = getattr(type(cache_context), name, None)
        if hasattr(class_attr, "__get__"):
            return class_attr.__get__(self, type(self))
        return getattr(cache_context, name)

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
            kv_size = _tensor_storage_span_num_bytes(self.kv_cache)
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

            # Decode-only NSA/MLA HiSparse receives the prefill KV directly
            # into its CPU cold tier. The normal device MR remains registered
            # for non-HiSparse migration and for backwards compatibility.
            from dlengine.runtime.context.cache.hisparse import get_hisparse_context

            hisparse_ctx = get_hisparse_context()
            if (
                hisparse_ctx.enabled
                and self.mode == "mla"
                and self.host_kv_cache is not None
            ):
                cold_size = _tensor_storage_span_num_bytes(self.host_kv_cache)
                self._local_hisparse_cold_mr_handler = (
                    peer_agent.register_memory_region(
                        _HISPARSE_COLD_KV_BUFFER_ID,
                        self.host_kv_cache.data_ptr(),
                        int(self.host_kv_cache.storage_offset()),
                        cold_size,
                    )
                )
                logger.info(
                    "Registered HiSparse host cold KV MR: handler=%s, size=%.2f GiB",
                    self._local_hisparse_cold_mr_handler,
                    cold_size / 1024**3,
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
                "P2PCacheTransfer PeerAgentContext is not attached. "
                "Was ModelRunner PeerAgentContext initialized?"
            )
        return self.peer_agent_context

    def ensure_peer_agent_connected(self, peer_alias: str) -> None:
        """Ensure the local PeerAgent is connected to ``peer_alias``."""
        self.get_peer_agent_context().ensure_connected(peer_alias)

    def p2p_disconnect(self, remote_engine_id: str):
        """Forget local connection metadata for a remote engine.

        PeerAgent does not currently expose a required hard-disconnect path for
        this flow, so this method clears transfer-side bookkeeping and cached
        engine metadata. Existing PeerAgent connections may remain reusable.
        """
        self.num_remote_kvcache_blocks.pop(remote_engine_id, None)
        self.remote_max_num_seqs.pop(remote_engine_id, None)
        self.remote_attention_tp.pop(remote_engine_id, None)
        self.remote_attention_dp.pop(remote_engine_id, None)
        self.remote_pp.pop(remote_engine_id, None)
        self.remote_num_hidden_layers.pop(remote_engine_id, None)
        self.remote_pp_layer_ranges.pop(remote_engine_id, None)
        self.remote_pp_cache_layer_indices.pop(remote_engine_id, None)
        self.remote_pp_dsv4_ratio_layer_indices.pop(remote_engine_id, None)
        self.remote_gdn_num_slots.pop(remote_engine_id, None)
        self.remote_compressed_pool_pages.pop(remote_engine_id, None)
        self.remote_dsv4_max_slots.pop(remote_engine_id, None)
        self.remote_dsv4_num_layers_per_ratio.pop(remote_engine_id, None)
        self.invalidate_engine_info_cache()

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
        dp_size: int,
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
        return pp_global_rank(
            0,
            dp_idx,
            sp_idx,
            tp_idx,
            dp_size=dp_size,
            sp_size=sp_size,
            tp_size=tp_size,
        )

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
                     (peer_alias, kv_idx, local_layer_idx, remote_layer_idx,
                      remote_num_layers, remote_block_idx, source_block_idx)
                     where local_layer_idx indexes the local kv_cache tensor,
                     remote_layer_idx indexes the remote (possibly PP-stage
                     sharded) kv_cache tensor and remote_num_layers is that
                     remote tensor's layer count (for stride math)
            gdn_assigns: engine_id -> peer_alias -> list of
                         (local_layer_idx, remote_layer_idx,
                          remote_state_slot, local_state_slot)
            indexer_assigns: engine_id -> peer_alias -> list of
                             (local_layer_idx, remote_layer_idx,
                              remote_block_idx, source_block_idx)
            compressed_assigns: engine_id -> peer_alias -> list of
                                (ratio, local_ratio_layer_idx,
                                 remote_ratio_layer_idx, remote_page_idx,
                                 local_page_idx)
            compressor_state_assigns: engine_id -> peer_alias -> list of
                                (ratio, local_ratio_layer_idx,
                                 remote_ratio_layer_idx, remote_state_slot,
                                 local_state_slot)
        """
        peer_context = self.get_peer_agent_context()
        peer_agent = peer_context.agent
        assignment_maps = (
            assigns,
            gdn_assigns,
            indexer_assigns or {},
            compressed_assigns or {},
            compressor_state_assigns or {},
        )
        engine_ids = {engine_id for mapping in assignment_maps for engine_id in mapping}
        for engine_id in engine_ids:
            peer_aliases = {
                peer_alias
                for mapping in assignment_maps
                for peer_alias in mapping.get(engine_id, {})
            }
            for peer_alias in peer_aliases:
                assign_batch = assigns.get(engine_id, {}).get(peer_alias, [])
                if not peer_context.is_connected(peer_alias):
                    raise RuntimeError(f"Peer {peer_alias} not connected")

                conn = peer_agent.query_connection(peer_alias)
                if conn is None or conn.endpoint is None:
                    raise RuntimeError(f"Failed to get endpoint for {peer_alias}")
                endpoint = conn.endpoint

                remote_mr_handler = None
                local_mr_handler = self._local_mr_handler
                if assign_batch:
                    remote_mr_info = peer_agent.get_mr_info(
                        peer_alias, _KV_CACHE_BUFFER_ID
                    )
                    if remote_mr_info is None:
                        raise RuntimeError(
                            f"Failed to get {_KV_CACHE_BUFFER_ID} MR info for "
                            f"{peer_alias}"
                        )
                    remote_mr_handler = peer_agent.get_handle(
                        _KV_CACHE_BUFFER_ID, peer_alias=peer_alias
                    )
                    if local_mr_handler is None:
                        raise RuntimeError(
                            f"Local MR handler not available for "
                            f"{_KV_CACHE_BUFFER_ID}"
                        )
                    logger.debug(
                        f"Remote MR for {peer_alias}: handler={remote_mr_handler}, "
                        f"local_handler={local_mr_handler}"
                    )
                use_hisparse_cold = self._local_hisparse_cold_mr_handler is not None
                if assign_batch and use_hisparse_cold:
                    local_mr_handler = self._local_hisparse_cold_mr_handler

                # Build KV cache RDMA ops
                rdma_ops: list[tuple] = []
                for op_idx, (
                    _peer_alias,
                    kv_idx,
                    local_layer_idx,
                    remote_layer_idx,
                    remote_num_layers,
                    remote_block_idx,
                    source_block_idx,
                ) in enumerate(assign_batch):
                    if use_hisparse_cold:
                        cold = self.host_kv_cache
                        local_off = (
                            kv_idx * cold.stride(0)
                            + local_layer_idx * cold.stride(1)
                            + source_block_idx * cold.stride(2)
                        ) * cold.element_size()
                    else:
                        local_off = self.layout.local_kv_stride(
                            kv_idx, local_layer_idx, source_block_idx
                        )
                    remote_off = self.layout.remote_kv_stride(
                        kv_idx,
                        remote_layer_idx,
                        remote_block_idx,
                        engine_id,
                        remote_num_layers,
                    )
                    length = self.layout.block_stride(1)

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
                        if local_conv_mr is None:
                            raise RuntimeError("Local gdn_conv MR is not registered")
                        conv_len = self.layout.gdn_conv_slot_num_bytes()
                        for (
                            local_layer_idx,
                            remote_layer_idx,
                            remote_slot,
                            local_slot,
                        ) in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_conv_mr,
                                    remote_conv_mr,
                                    self.layout.remote_gdn_conv_stride(
                                        remote_layer_idx, remote_slot, engine_id
                                    ),
                                    self.layout.gdn_conv_stride(
                                        local_layer_idx, local_slot
                                    ),
                                    conv_len,
                                )
                            )
                    else:
                        raise RuntimeError(
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
                        if local_rec_mr is None:
                            raise RuntimeError(
                                "Local gdn_recurrent MR is not registered"
                            )
                        rec_len = self.layout.gdn_recurrent_slot_num_bytes()
                        for (
                            local_layer_idx,
                            remote_layer_idx,
                            remote_slot,
                            local_slot,
                        ) in gdn_batch:
                            rdma_ops.append(
                                (
                                    local_rec_mr,
                                    remote_rec_mr,
                                    self.layout.remote_gdn_recurrent_stride(
                                        remote_layer_idx, remote_slot, engine_id
                                    ),
                                    self.layout.gdn_recurrent_stride(
                                        local_layer_idx, local_slot
                                    ),
                                    rec_len,
                                )
                            )
                    else:
                        raise RuntimeError(
                            f"Failed to get gdn_recurrent MR info for {peer_alias}"
                        )

                # Append DSv4 compressed cache + compressor scratch state RDMA ops.
                comp_batch = (
                    (compressed_assigns or {}).get(engine_id, {}).get(peer_alias, [])
                )
                if comp_batch:
                    # Group by ratio to register the right MR per ratio.
                    by_ratio: dict[int, list[tuple]] = {}
                    for r, local_rli, remote_rli, rpage, lpage in comp_batch:
                        by_ratio.setdefault(r, []).append(
                            (local_rli, remote_rli, rpage, lpage)
                        )
                    for ratio, ops in by_ratio.items():
                        local_handler = self._local_dsv4_compressed_mr_handlers.get(
                            ratio
                        )
                        if local_handler is None:
                            raise RuntimeError(
                                f"Local DSv4 compressed ratio={ratio} MR is not registered"
                            )
                        remote_info = peer_agent.get_mr_info(
                            peer_alias, f"dsv4_compressed_r{ratio}"
                        )
                        if not remote_info:
                            raise RuntimeError(
                                f"Failed to get DSv4 compressed MR info for {peer_alias}, ratio={ratio}"
                            )
                            continue
                        remote_handler = peer_agent.get_handle(
                            f"dsv4_compressed_r{ratio}", peer_alias=peer_alias
                        )
                        page_bytes = self.layout.compressed_page_bytes(ratio)
                        for local_rli, remote_rli, rpage, lpage in ops:
                            rdma_ops.append(
                                (
                                    local_handler,
                                    remote_handler,
                                    self.layout.remote_compressed_stride(
                                        ratio, remote_rli, rpage, engine_id
                                    ),
                                    self.layout.local_compressed_stride(
                                        ratio, local_rli, lpage
                                    ),
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
                    for r, local_rli, remote_rli, rslot, lslot in cstate_batch:
                        by_ratio_s.setdefault(r, []).append(
                            (local_rli, remote_rli, rslot, lslot)
                        )
                    for ratio, ops in by_ratio_s.items():
                        for kind, local_map in (
                            ("kv", self._local_dsv4_compressor_kv_mr_handlers),
                            ("score", self._local_dsv4_compressor_score_mr_handlers),
                            ("counts", self._local_dsv4_compressor_counts_mr_handlers),
                        ):
                            local_handler = local_map.get(ratio)
                            if local_handler is None:
                                raise RuntimeError(
                                    f"Local {kind} compressor ratio={ratio} MR "
                                    "is not registered"
                                )
                            mr_name = f"dsv4_compressor_{kind}_r{ratio}"
                            remote_info = peer_agent.get_mr_info(peer_alias, mr_name)
                            if not remote_info:
                                raise RuntimeError(
                                    f"Failed to get {mr_name} MR info for {peer_alias}"
                                )
                                continue
                            remote_handler = peer_agent.get_handle(
                                mr_name, peer_alias=peer_alias
                            )
                            row_bytes = self.layout.compressor_state_row_bytes(
                                ratio, kind
                            )
                            for local_rli, remote_rli, rslot, lslot in ops:
                                rdma_ops.append(
                                    (
                                        local_handler,
                                        remote_handler,
                                        self.layout.remote_compressor_state_stride(
                                            ratio,
                                            remote_rli,
                                            rslot,
                                            kind,
                                            engine_id,
                                        ),
                                        self.layout.local_compressor_state_stride(
                                            ratio, local_rli, lslot, kind
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
                        page_bytes = self.layout.indexer_page_num_bytes()
                        for (
                            local_layer_idx,
                            remote_layer_idx,
                            remote_block,
                            local_block,
                        ) in indexer_batch:
                            rdma_ops.append(
                                (
                                    local_indexer_mr,
                                    remote_indexer_mr,
                                    self.layout.remote_indexer_stride(
                                        remote_layer_idx, remote_block, engine_id
                                    ),
                                    self.layout.local_indexer_stride(
                                        local_layer_idx, local_block
                                    ),
                                    page_bytes,
                                )
                            )
                    else:
                        raise RuntimeError(
                            f"Failed to get indexer_cache MR info for {peer_alias}"
                        )

                if not rdma_ops:
                    raise RuntimeError(f"No valid RDMA ops for {peer_alias}")

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
        from dlengine._rust.proto import MigrationIn

        views = MigrationIn.from_bytes(data).sequences

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
        connection_requests: list[tuple[str, str, int, int, int]] = []
        for v in views:
            engine_id = v.migrate_engine_id
            engine_info = engine_info_map.get(engine_id, {})
            remote_arch = engine_info.get("architecture")
            local_arch = getattr(self, "architecture", None)
            if remote_arch and local_arch and remote_arch != local_arch:
                raise RuntimeError(
                    f"PD cache architecture mismatch for {engine_id}: "
                    f"remote={remote_arch}, local={local_arch}"
                )
            remote_block_size = int(
                engine_info.get("kvcache_block_size", self.block_size)
            )
            if remote_block_size != self.block_size:
                raise RuntimeError(
                    f"PD cache block-size mismatch for {engine_id}: "
                    f"remote={remote_block_size}, local={self.block_size}"
                )
            remote_sp = int(engine_info.get("attention_sp", v.migrate_group_size))
            if remote_sp != v.migrate_group_size:
                raise RuntimeError(
                    f"PD attention_sp metadata mismatch for {engine_id}: "
                    f"registered={remote_sp}, migration={v.migrate_group_size}"
                )
            remote_max_num_seqs = engine_info.get("max_num_seqs", 0)
            remote_gdn_num_slots = engine_info.get("gdn_num_slots", 0)
            # PD + GQA: remember the remote engine's attention_tp so peer
            # selection can address the matching per-rank KV-head shard.
            self.remote_attention_tp[engine_id] = int(
                engine_info.get("attention_tp", 1)
            )
            self.remote_attention_dp[engine_id] = int(
                engine_info.get("attention_dp", 1)
            )
            # PP prefill → pp=1 decode: the prefill engine's KV cache is
            # sharded per pipeline stage, so the decode side must route each
            # global layer to the stage that owns it. Validate the supported
            # shape here so failures are loud and early.
            remote_pp = int(engine_info.get("pp", 1))
            self.remote_pp[engine_id] = remote_pp
            remote_layers_total = int(engine_info.get("num_hidden_layers", 0))
            effective_remote_layers = remote_layers_total or self.num_hidden_layers
            self.remote_num_hidden_layers[engine_id] = remote_layers_total
            published_ranges = engine_info.get("pp_layer_ranges") or []
            if published_ranges:
                stage_ranges = [(int(s), int(e)) for s, e in published_ranges]
            else:
                # Older prefill engines don't publish pp_layer_ranges;
                # recompute with the shared even-split policy.
                stage_ranges = pp_layer_partition(effective_remote_layers, remote_pp)
            if (
                len(stage_ranges) != remote_pp
                or not stage_ranges
                or stage_ranges[0][0] != 0
                or stage_ranges[-1][1] != effective_remote_layers
                or any(
                    start >= end or end != stage_ranges[idx + 1][0]
                    for idx, (start, end) in enumerate(stage_ranges[:-1])
                )
                or stage_ranges[-1][0] >= stage_ranges[-1][1]
            ):
                raise RuntimeError(
                    f"Invalid PP layer ranges for {engine_id}: pp={remote_pp}, "
                    f"num_layers={effective_remote_layers}, ranges={stage_ranges}"
                )
            self.remote_pp_layer_ranges[engine_id] = stage_ranges
            published_cache_layers = engine_info.get("pp_cache_layer_indices") or []
            if published_cache_layers:
                cache_layers_by_stage = [
                    [int(layer) for layer in stage_layers]
                    for stage_layers in published_cache_layers
                ]
            else:
                cache_layers_by_stage = [
                    list(range(start, end)) for start, end in stage_ranges
                ]
            self.remote_pp_cache_layer_indices[engine_id] = cache_layers_by_stage
            published_ratio_layers = (
                engine_info.get("pp_dsv4_ratio_layer_indices") or {}
            )
            self.remote_pp_dsv4_ratio_layer_indices[engine_id] = {
                int(ratio): [
                    [int(layer) for layer in stage_layers]
                    for stage_layers in layers_by_stage
                ]
                for ratio, layers_by_stage in published_ratio_layers.items()
            }
            if remote_pp > 1:
                if remote_layers_total <= 0:
                    raise RuntimeError(
                        f"PP prefill engine {engine_id} did not publish "
                        "num_hidden_layers"
                    )
                if get_dist_context().pp_world_size > 1:
                    raise RuntimeError(
                        "PD migration between two PP engines is not supported "
                        "(PP prefill requires a pp=1 decode engine)"
                    )
                if len(cache_layers_by_stage) != remote_pp:
                    raise RuntimeError(
                        f"PP cache-layer metadata mismatch for {engine_id}: "
                        f"pp={remote_pp}, layers={cache_layers_by_stage}"
                    )
                if len(stage_ranges) != remote_pp:
                    raise RuntimeError(
                        f"PP layer-range metadata mismatch for {engine_id}: "
                        f"pp={remote_pp}, pp_layer_ranges={stage_ranges}"
                    )
                flattened_cache_layers = [
                    layer
                    for stage_layers in cache_layers_by_stage
                    for layer in stage_layers
                ]
                if len(flattened_cache_layers) != self.num_hidden_layers:
                    raise RuntimeError(
                        f"PP cache-layer count mismatch for {engine_id}: remote "
                        f"has {len(flattened_cache_layers)} slots, decode has "
                        f"{self.num_hidden_layers}"
                    )
                if len(set(flattened_cache_layers)) != len(flattened_cache_layers):
                    raise RuntimeError(
                        f"PP cache-layer metadata contains duplicates for "
                        f"{engine_id}: {cache_layers_by_stage}"
                    )
                if flattened_cache_layers != sorted(flattened_cache_layers):
                    raise RuntimeError(
                        f"PP cache layers are not in global slot order for "
                        f"{engine_id}: {cache_layers_by_stage}"
                    )
                for stage, layers in enumerate(cache_layers_by_stage):
                    start, end = stage_ranges[stage]
                    if any(layer < start or layer >= end for layer in layers):
                        raise RuntimeError(
                            f"PP cache layer outside stage {stage} range "
                            f"[{start}, {end}) for {engine_id}: {layers}"
                        )
            # The RDMA block-copy migrates whole (layer, block) regions whose
            # byte size depends on num_local_kv_heads. That only lines up when
            # prefill and decode shard KV heads identically; otherwise a decode
            # rank would copy bytes that belong to a different head subset.
            remote_nlkv = int(
                engine_info.get("num_local_kv_heads", self.num_local_kv_heads)
            )
            if remote_nlkv != self.num_local_kv_heads:
                raise RuntimeError(
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
                raise RuntimeError(
                    f"Sequence {v.seq_id} has no peer_addrs for engine {engine_id}"
                )

            # PP prefill layout: the remote per-stage layer ownership routes
            # each global layer's RDMA read to the prefill stage (and thus the
            # peer rank / byte offset) that holds it. For remote_pp == 1 this
            # degenerates to a single stage covering all layers, preserving
            # the previous behavior exactly.
            remote_pp = self.remote_pp.get(engine_id, 1)
            if remote_pp > 1:
                remote_stage_ranges = self.remote_pp_layer_ranges[engine_id]
                remote_cache_layers_by_stage = self.remote_pp_cache_layer_indices[
                    engine_id
                ]
            else:
                # Single stage covering the local KV layer count. Do NOT use
                # the published model-layer ranges here: hybrid-attention
                # models have fewer KV layers than decoder layers, and the
                # remote cache tensor is sized by KV layers.
                remote_stage_ranges = [(0, self.num_hidden_layers)]
                remote_cache_layers_by_stage = [list(range(self.num_hidden_layers))]
            # Workers within one prefill stage: attention_dp * sp * tp.
            remote_inner_world_size = (
                self.remote_attention_dp.get(engine_id, 1)
                * v.migrate_group_size
                * self.remote_attention_tp.get(engine_id, 1)
            )
            remote_dp = self.remote_attention_dp.get(engine_id, 1)
            remote_tp = self.remote_attention_tp.get(engine_id, 1)
            if not 0 <= v.migrate_dp_idx < remote_dp:
                raise RuntimeError(
                    f"remote DP index {v.migrate_dp_idx} outside [0, "
                    f"{remote_dp}) for {engine_id}"
                )
            if not 0 <= tp_idx < remote_tp:
                raise RuntimeError(
                    f"decode TP index {tp_idx} outside remote TP size "
                    f"{remote_tp} for {engine_id}"
                )
            if remote_pp > 1 and remote_pp * remote_inner_world_size != len(peer_addrs):
                raise RuntimeError(
                    f"PP peer layout mismatch for {engine_id}: pp={remote_pp} * "
                    f"inner={remote_inner_world_size} != "
                    f"len(peer_addrs)={len(peer_addrs)}"
                )

            if len(v.migrate_block_location) > len(v.active_block_location):
                raise RuntimeError(
                    f"Sequence {v.seq_id}: migrate has MORE blocks than active! "
                    f"migrate={len(v.migrate_block_location)}, active={len(v.active_block_location)}"
                )
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
                    raise RuntimeError(
                        f"Sequence {v.seq_id}: source_block_idx {source_block_idx} "
                        f"out of range [0, {self.num_local_kvcache_blocks})"
                    )
                remote_max = self.num_remote_kvcache_blocks.get(engine_id, 0)
                if remote_block_idx < 0 or (
                    remote_max > 0 and remote_block_idx >= remote_max
                ):
                    raise RuntimeError(
                        f"Sequence {v.seq_id}: remote_block_idx {remote_block_idx} "
                        f"out of range [0, {remote_max})"
                    )

                if source_sp_idx != sp_idx:
                    continue

                # Rank of the matching (dp, sp, tp) cell within one prefill
                # stage; the stage offset is added per layer below.
                remote_inner_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    remote_dp,
                    remote_sp_idx,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )

                max_remote_rank = (
                    len(remote_stage_ranges) - 1
                ) * remote_inner_world_size + remote_inner_rank
                if max_remote_rank >= len(peer_addrs):
                    raise RuntimeError(
                        f"remote_rank {max_remote_rank} >= len(peer_addrs) "
                        f"{len(peer_addrs)}"
                    )

                cache_planes = 1 if self.mode in ("mla", "dsv4") else 2
                local_cache_layers = [
                    layer
                    for stage_layers in remote_cache_layers_by_stage
                    for layer in stage_layers
                ]
                remote_cache_locations = {
                    layer: (stage, remote_layer_idx)
                    for stage, stage_layers in enumerate(remote_cache_layers_by_stage)
                    for remote_layer_idx, layer in enumerate(stage_layers)
                }
                for kv_idx in range(cache_planes):
                    for local_layer_idx, global_layer_idx in enumerate(
                        local_cache_layers
                    ):
                        stage, remote_layer_idx = remote_cache_locations[
                            global_layer_idx
                        ]
                        peer_alias = peer_addrs[
                            stage * remote_inner_world_size + remote_inner_rank
                        ]
                        assigns[engine_id][peer_alias].append(
                            (
                                peer_alias,
                                kv_idx,
                                local_layer_idx,
                                remote_layer_idx,
                                len(remote_cache_layers_by_stage[stage]),
                                remote_block_idx,
                                source_block_idx,
                            )
                        )

                # Indexer cache has one slot per decoder layer. Route each
                # global layer to its PP stage and stage-local layer index.
                if self.indexer_cache is not None:
                    for layer_idx in range(self.num_hidden_layers):
                        stage = pp_stage_of_layer(layer_idx, remote_stage_ranges)
                        stage_start, _ = remote_stage_ranges[stage]
                        indexer_peer_alias = peer_addrs[
                            stage * remote_inner_world_size + remote_inner_rank
                        ]
                        indexer_assigns[engine_id][indexer_peer_alias].append(
                            (
                                layer_idx,
                                layer_idx - stage_start,
                                remote_block_idx,
                                source_block_idx,
                            )
                        )

            # DSv4 (S2.6): per-ratio compressed cache + compressor state migration.
            # migrate_compressed_block_tables[ratio] = list of remote page IDs.
            # active_compressed_block_tables[ratio]  = list of local page IDs allocated by
            #                                          the decode engine's GroupManager.
            if (
                getattr(self, "dsv4_compressed_caches_flat", None)
                and v.migrate_compressed_block_tables
            ):
                remote_inner_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    remote_dp,
                    v.migrate_group_size - 1,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )
                for ratio, remote_pages in v.migrate_compressed_block_tables.items():
                    if ratio not in self.dsv4_compressed_caches_flat:
                        continue  # decode engine doesn't have this ratio (mismatch)
                    local_pages = v.active_compressed_block_tables.get(ratio, [])
                    if remote_pp > 1:
                        layers_by_stage = self.remote_pp_dsv4_ratio_layer_indices[
                            engine_id
                        ].get(ratio, [])
                        if len(layers_by_stage) != remote_pp:
                            raise RuntimeError(
                                f"Missing DSv4 ratio={ratio} PP layer metadata "
                                f"for {engine_id}: {layers_by_stage}"
                            )
                    else:
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        layers_by_stage = [list(range(n_layers))]
                    local_ratio_layers = [
                        layer
                        for stage_layers in layers_by_stage
                        for layer in stage_layers
                    ]
                    local_ratio_count = self.dsv4_compressed_caches_flat[ratio].shape[0]
                    if len(local_ratio_layers) != local_ratio_count:
                        raise RuntimeError(
                            f"DSv4 ratio={ratio} layer-count mismatch for "
                            f"{engine_id}: remote={len(local_ratio_layers)}, "
                            f"local={local_ratio_count}"
                        )
                    remote_ratio_locations = {
                        layer: (stage, remote_rli)
                        for stage, stage_layers in enumerate(layers_by_stage)
                        for remote_rli, layer in enumerate(stage_layers)
                    }
                    for local_rli, global_layer_idx in enumerate(local_ratio_layers):
                        stage, remote_rli = remote_ratio_locations[global_layer_idx]
                        peer_alias = peer_addrs[
                            stage * remote_inner_world_size + remote_inner_rank
                        ]
                        for rpage, lpage in zip(remote_pages, local_pages):
                            compressed_assigns[engine_id][peer_alias].append(
                                (
                                    ratio,
                                    local_rli,
                                    remote_rli,
                                    rpage,
                                    lpage,
                                )
                            )

            # DSv4 (S2.6): per-ratio compressor scratch state migration.
            if (
                getattr(self, "dsv4_compressor_kv_flat", None)
                and v.migrate_state_slot >= 0
                and v.active_state_slot >= 0
            ):
                remote_inner_rank = self._remote_global_rank(
                    v.migrate_dp_idx,
                    remote_dp,
                    v.migrate_group_size - 1,
                    v.migrate_group_size,
                    tp_idx,
                    self.remote_attention_tp.get(engine_id, 1),
                )
                for ratio in self.dsv4_compressor_kv_flat.keys():
                    if remote_pp > 1:
                        layers_by_stage = self.remote_pp_dsv4_ratio_layer_indices[
                            engine_id
                        ].get(ratio, [])
                    else:
                        n_layers = len(self.dsv4_layers_per_ratio.get(ratio, []))
                        layers_by_stage = [list(range(n_layers))]
                    local_ratio_layers = [
                        layer
                        for stage_layers in layers_by_stage
                        for layer in stage_layers
                    ]
                    local_ratio_count = self.dsv4_compressor_kv_flat[ratio].shape[0]
                    if len(local_ratio_layers) != local_ratio_count:
                        raise RuntimeError(
                            f"DSv4 compressor ratio={ratio} layer-count "
                            f"mismatch for {engine_id}: "
                            f"remote={len(local_ratio_layers)}, "
                            f"local={local_ratio_count}"
                        )
                    remote_ratio_locations = {
                        layer: (stage, remote_rli)
                        for stage, stage_layers in enumerate(layers_by_stage)
                        for remote_rli, layer in enumerate(stage_layers)
                    }
                    for local_rli, global_layer_idx in enumerate(local_ratio_layers):
                        stage, remote_rli = remote_ratio_locations[global_layer_idx]
                        peer_alias = peer_addrs[
                            stage * remote_inner_world_size + remote_inner_rank
                        ]
                        compressor_state_assigns[engine_id][peer_alias].append(
                            (
                                ratio,
                                local_rli,
                                remote_rli,
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
                    remote_inner_rank = self._remote_global_rank(
                        v.migrate_dp_idx,
                        remote_dp,
                        v.migrate_group_size - 1,
                        v.migrate_group_size,
                        tp_idx,
                        self.remote_attention_tp.get(engine_id, 1),
                    )
                    num_gdn_layers = self.gdn_recurrent_states.shape[0]
                    for layer_idx in range(num_gdn_layers):
                        if remote_pp > 1:
                            stage = pp_stage_of_layer(layer_idx, remote_stage_ranges)
                            stage_start, _ = remote_stage_ranges[stage]
                            remote_layer_idx = layer_idx - stage_start
                        else:
                            stage = 0
                            remote_layer_idx = layer_idx
                        peer_alias = peer_addrs[
                            stage * remote_inner_world_size + remote_inner_rank
                        ]
                        gdn_assigns[engine_id][peer_alias].append(
                            (
                                layer_idx,
                                remote_layer_idx,
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


_P2P_CACHE_TRANSFER = P2PCacheTransfer()


def get_p2p_cache_transfer() -> P2PCacheTransfer:
    return _P2P_CACHE_TRANSFER


def reset_p2p_cache_transfer() -> None:
    _P2P_CACHE_TRANSFER.peer_agent_context = None
    initialize_migration_state(_P2P_CACHE_TRANSFER)


# Backward compatibility for old context/cache shims.
KVMigratorMixin = P2PCacheTransfer


__all__ = [
    "KVMigratorMixin",
    "P2PCacheTransfer",
    "get_p2p_cache_transfer",
    "initialize_migration_state",
    "reset_p2p_cache_transfer",
    "select_peer_device",
]

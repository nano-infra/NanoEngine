import atexit
import json
from typing import List, Set, Tuple

import ray
from dlslime.ctrl import NanoCtrlClient

from dlengine.config import Config
from dlengine.engine.llm_engine import LLMEngine
from dlengine.logging import get_logger

logger = get_logger("dlengine")


class LLM(LLMEngine):
    """LLM class with Ray remote execution support."""

    from dlengine.offline.generate import generate

    @classmethod
    def as_remote(cls, config):
        ray_address = getattr(config, "ray_address", "auto")
        ray.init(address=ray_address, ignore_reinit_error=True)

        # The LLMComponent actor is only the engine controller; its workers
        # reserve GPUs later via RayExecutor placement groups. Let Ray place
        # this CPU-only controller; it no longer needs to live on a user-chosen
        # torch distributed master node.
        return ray.remote(num_cpus=1, num_gpus=0)(cls).remote(config)


class LLMComponent(LLM):
    """LLM component with lifecycle management and service discovery integration.

    This class extends LLM with:
    - dlslime-ctrl service registration and heartbeat
    - Peer engine information management for distributed serving
    - Automatic resource cleanup on shutdown
    """

    def __init__(self, config: Config):
        # In PD disagg mode, verify dlslime-ctrl is reachable before heavy model loading.
        if config.ctrl_address:
            NanoCtrlClient(config.ctrl_address, config.ctrl_scope).check_connection()

        super().__init__(config)

        # peer_engine_id -> dict(num_blocks, world_size, peer_addrs, p2p_host, p2p_port) for lazy migration
        self._peer_info: dict[str, dict] = {}
        self.active_p2p_links: Set[str] = set()

        # P2P ZMQ socket for receiving free instructions (set by engine_server.py)
        self.p2p_socket = None
        self.p2p_port = None

        # P2P client connections cache: engine_id -> ZMQ socket
        self._p2p_clients: dict[str, any] = {}
        self._p2p_ctx = None

        # dlslime-ctrl lifecycle client (None when ctrl_address is not configured)
        self._nanoctrl: NanoCtrlClient | None = None

        # Register with dlslime-ctrl if configured.
        # engine_server.py will re-register after binding its own P2P socket
        # (replacing p2p_port), but for direct LLMComponent usage (e.g.
        # deepseek_v3_disagg.py) this is the only registration point.
        if self.config.ctrl_address:
            self._register_with_nanoctrl()

        atexit.register(self.shutdown)

    def _pp_layer_ranges(self) -> list[list[int]]:
        """Authoritative ``[start, end)`` decoder-layer range per pipeline stage.

        Published in engine metadata so a pp=1 decode engine can route each
        global layer's KV read to the prefill stage that owns it, without
        having to replicate architecture-specific split policies (e.g.
        Gemma4 reserves its shared-KV suffix for the last stage).
        """
        from dlengine.models.pp_utils import (
            get_gemma4_shared_kv_source_start,
            pp_layer_partition,
        )

        num_layers = getattr(self.config.hf_config, "num_hidden_layers", 0)
        arch = (getattr(self.config.hf_config, "architectures", None) or [""])[0]
        final_stage_start = None
        if self.config.pp > 1 and arch in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
        ):
            final_stage_start = get_gemma4_shared_kv_source_start(self.config.hf_config)
        return [
            [start, end]
            for start, end in pp_layer_partition(
                num_layers, self.config.pp, final_stage_start
            )
        ]

    def _pp_cache_layer_indices(self) -> list[list[int]]:
        """Global primary-cache layer indices owned by each PP stage."""
        from dlengine.models.pp_utils import (
            cache_layer_indices,
            partition_layer_indices,
        )

        ranges = [tuple(value) for value in self._pp_layer_ranges()]
        return partition_layer_indices(
            cache_layer_indices(
                self.config.hf_config,
                gemma_hisparse_only_full_attention=bool(self.config.enable_hisparse),
            ),
            ranges,
        )

    def _pp_dsv4_ratio_layer_indices(self) -> dict[int, list[list[int]]]:
        """Global DSv4 compressed-layer indices by ratio and PP stage."""
        from dlengine.models.pp_utils import partition_layer_indices

        ranges = [tuple(value) for value in self._pp_layer_ranges()]
        ratios = list(getattr(self.config.hf_config, "compress_ratios", None) or [])
        return {
            int(ratio): partition_layer_indices(
                [idx for idx, value in enumerate(ratios) if value == ratio], ranges
            )
            for ratio in sorted({value for value in ratios if value > 0})
        }

    def get_engine_info(self, status: str = "ready") -> str:
        """Get engine info as JSON string."""
        # Get peer_agent addresses from all workers
        peer_addrs = (
            self.executor.get_peer_agent_addrs()
            if hasattr(self.executor, "get_peer_agent_addrs")
            else []
        )

        from dlengine.utils.network import get_advertise_host

        zmq_host = get_advertise_host(self.config.host)

        engine_info = {
            "id": self.engine_id,
            "role": self.config.mode,
            "rank": 0,
            "world_size": self.config.world_size,
            "num_blocks": self.config.num_kvcache_blocks,
            "host": zmq_host,
            "port": self.config.port,
            "status": status,
            "peer_addrs": peer_addrs,
            "p2p_host": zmq_host,
            "p2p_port": self.p2p_port if self.p2p_port else 0,
            "max_num_seqs": self.config.max_num_seqs,
            # Attention parallel layout. Consumers (decode engines) need these
            # to map a (pp_idx, dp_idx, sp_idx, tp_idx) cell to the right
            # global rank in ``peer_addrs`` during PD KV migration. peer_addrs
            # is ordered by global rank =
            # pp_idx*(dp*sp*tp) + dp_idx*(sp*tp) + sp_idx*tp + tp_idx.
            "attention_dp": self.config.attention_dp,
            "attention_sp": self.config.attention_sp,
            "attention_tp": self.config.attention_tp,
            "pp": self.config.pp,
            "num_hidden_layers": getattr(self.config.hf_config, "num_hidden_layers", 0),
            "pp_layer_ranges": self._pp_layer_ranges(),
            "pp_cache_layer_indices": self._pp_cache_layer_indices(),
            "pp_dsv4_ratio_layer_indices": self._pp_dsv4_ratio_layer_indices(),
            # Per-rank KV-head shard size. The RDMA block-copy migration requires
            # the prefill and decode engines to share the same per-rank KV-head
            # layout (i.e. equal attention_tp for GQA), so the decode side can
            # validate before issuing reads.
            "num_local_kv_heads": (
                getattr(self.config.hf_config, "num_key_value_heads", 1)
                // self.config.attention_tp
            ),
        }

        # DSv4 (S2.5): publish per-ratio compressed pool sizes so peer engines
        # can compute remote strides for RDMA migration.
        is_dsv4 = self.config.hf_config.architectures[0] == "DeepseekV4ForCausalLM"
        if is_dsv4:
            compress_ratios = (
                getattr(self.config.hf_config, "compress_ratios", None) or []
            )
            unique_ratios = sorted({r for r in compress_ratios if r > 0})
            page_size = 2  # matches CacheContext.allocate_dsv4_compressed_caches
            pool_pages = {}
            num_layers_per_ratio = {}
            for ratio in unique_ratios:
                max_compressed = (self.config.max_model_len // ratio + 63) // 64 * 64
                max_blocks = (max_compressed + page_size - 1) // page_size
                worst = self.config.max_num_seqs * max_blocks
                override = 0
                if ratio == 4:
                    override = self.config.dsv4_compressed_pool_pages_ratio4
                elif ratio == 128:
                    override = self.config.dsv4_compressed_pool_pages_ratio128
                pool_pages[ratio] = override if override > 0 else worst
                num_layers_per_ratio[ratio] = sum(
                    1 for r in compress_ratios if r == ratio
                )
            engine_info["dsv4_compressed_pool_pages"] = pool_pages
            engine_info["dsv4_max_slots"] = self.config.max_num_seqs
            engine_info["dsv4_num_layers_per_ratio"] = num_layers_per_ratio

        return json.dumps(engine_info)

    def set_peer_info(self, remote_engine_info: str | bytes) -> None:
        """Store remote engine info including peer_addrs and P2P port for lazy migration.

        This method parses the remote engine's info and stores it so that during
        migration, the endpoints can be embedded in BlockContext.endpoints for
        lazy P2P connection.

        Args:
            remote_engine_info: JSON string from the engine registry.
        """
        info_dict = json.loads(remote_engine_info)

        # Parse JSON format
        remote_engine_id = info_dict.get("id", "")
        num_kv_blocks = info_dict.get("num_blocks", 0)
        world_size = info_dict.get("world_size", 1)
        peer_addrs = info_dict.get("peer_addrs", [])
        p2p_host = info_dict.get("p2p_host", "")
        p2p_port = info_dict.get("p2p_port", 0)

        # Store in engine for use during migration and free instructions
        self._peer_info[remote_engine_id] = {
            "num_blocks": num_kv_blocks,
            "world_size": world_size,
            "peer_addrs": peer_addrs,
            "p2p_host": p2p_host,
            "p2p_port": p2p_port,
        }
        logger.info(
            f"Stored peer info for {remote_engine_id}: {len(peer_addrs)} addresses, P2P={p2p_host}:{p2p_port}"
        )

    def _fetch_peer_info_from_ctrl(self, target_engine_id: str) -> bool:
        """Fetch peer engine info from dlslime-ctrl on-demand."""
        if self._nanoctrl is None:
            logger.error("Cannot fetch peer info: dlslime-ctrl not configured")
            return False

        entity_info = self._nanoctrl.get_entity_info(target_engine_id)
        if entity_info:
            logger.info(f"dlslime-ctrl returned entity_info: {entity_info}")
            engine_info = dict(entity_info.get("metadata") or {})
            engine_info.setdefault("id", entity_info.get("entity_id", target_engine_id))
            self.set_peer_info(json.dumps(engine_info))
            logger.info(f"Fetched peer info for {target_engine_id} from dlslime-ctrl")
            return True

        logger.error(f"Failed to fetch peer info for {target_engine_id}: not found")
        return False

    def send_free_vision_slots(
        self, target_encoder_id: str, slot_indices: List[int]
    ) -> None:
        """Send P2P free instruction for vision embedding slots to remote encoder.

        Args:
            target_encoder_id: Encoder engine ID to send free instruction to
            slot_indices: List of slot indices to free in the encoder's EmbeddingPool
        """
        import zmq

        from dlengine.server.wire import encode_free_vision_slots, encode_packet

        if not slot_indices:
            return

        # Get target encoder P2P address
        if target_encoder_id not in self._peer_info:
            if not self._fetch_peer_info_from_ctrl(target_encoder_id):
                logger.error(
                    f"Cannot send vision free: failed to fetch peer info for {target_encoder_id}"
                )
                return

        peer_info = self._peer_info[target_encoder_id]
        p2p_host = peer_info.get("p2p_host")
        p2p_port = peer_info.get("p2p_port")

        if not p2p_host or not p2p_port:
            logger.error(
                f"Cannot send vision free: encoder {target_encoder_id} has no P2P address"
            )
            return

        # Get or create P2P client socket
        if target_encoder_id not in self._p2p_clients:
            if self._p2p_ctx is None:
                self._p2p_ctx = zmq.Context()

            client_socket = self._p2p_ctx.socket(zmq.DEALER)
            client_socket.set(zmq.LINGER, 0)
            client_socket.set(zmq.SNDTIMEO, 5000)
            endpoint = f"tcp://{p2p_host}:{p2p_port}"
            client_socket.connect(endpoint)
            self._p2p_clients[target_encoder_id] = client_socket
            logger.info(f"Created P2P client connection to encoder at {endpoint}")
        else:
            client_socket = self._p2p_clients[target_encoder_id]

        payload = encode_free_vision_slots(
            target_encoder_id, slot_indices, self.engine_id
        )

        # Send via P2P (Action 4 = FreeVisionSlots)
        packet = encode_packet(action=4, payload=payload)

        try:
            client_socket.send(packet, zmq.NOBLOCK)
            logger.debug(
                f"P2P: Sent vision slot free to encoder {target_encoder_id} "
                f"for {len(slot_indices)} slots: {slot_indices}"
            )
        except zmq.ZMQError as e:
            logger.error(f"P2P: Failed to send vision free instruction: {e}")
            try:
                client_socket.close()
            except Exception:
                pass
            del self._p2p_clients[target_encoder_id]

    def send_free_sequences(self, target_engine_id: str, seq_ids: List[int]) -> None:
        """Send P2P free instruction directly to remote engine (no dlengine-router).

        Args:
            target_engine_id: Engine ID to send free instruction to
            seq_ids: List of sequence IDs to free
        """
        import zmq

        from dlengine.server.wire import encode_free_sequences, encode_packet

        if not seq_ids:
            logger.warning(f"No sequence IDs to free for engine {target_engine_id}")
            return

        # Get target engine P2P address from peer_info (fetch on-demand if not cached)
        if target_engine_id not in self._peer_info:
            logger.info(
                f"Peer info for {target_engine_id} not cached, fetching from dlslime-ctrl..."
            )
            if not self._fetch_peer_info_from_ctrl(target_engine_id):
                logger.error(
                    f"Cannot send free: failed to fetch peer info for {target_engine_id}"
                )
                return

        peer_info = self._peer_info[target_engine_id]
        p2p_host = peer_info.get("p2p_host")
        p2p_port = peer_info.get("p2p_port")

        if not p2p_host or not p2p_port:
            logger.error(
                f"Cannot send free: target engine {target_engine_id} has no P2P address"
            )
            return

        # Get or create P2P client socket
        if target_engine_id not in self._p2p_clients:
            if self._p2p_ctx is None:
                self._p2p_ctx = zmq.Context()

            client_socket = self._p2p_ctx.socket(zmq.DEALER)
            client_socket.set(zmq.LINGER, 0)
            client_socket.set(zmq.SNDTIMEO, 5000)
            endpoint = f"tcp://{p2p_host}:{p2p_port}"
            client_socket.connect(endpoint)
            self._p2p_clients[target_engine_id] = client_socket
            logger.info(f"Created P2P client connection to {endpoint}")
        else:
            client_socket = self._p2p_clients[target_engine_id]

        payload = encode_free_sequences(seq_ids, self.engine_id)

        # Send via P2P (Action 3)
        packet = encode_packet(action=3, payload=payload)

        try:
            client_socket.send(packet, zmq.NOBLOCK)
            logger.info(
                f"P2P: Sent free instruction to {target_engine_id} for {len(seq_ids)} sequences: {seq_ids}"
            )
        except zmq.ZMQError as e:
            logger.error(f"P2P: Failed to send free instruction: {e}")
            # Remove failed connection from cache
            try:
                client_socket.close()
            except Exception:
                pass
            del self._p2p_clients[target_engine_id]

    def shutdown(self):
        """Shutdown the component and cleanup resources."""
        # Stop heartbeat and unregister from dlslime-ctrl
        if hasattr(self, "_nanoctrl") and self._nanoctrl:
            self._nanoctrl.stop()

        # Close P2P client connections
        if hasattr(self, "_p2p_clients"):
            for engine_id, socket in self._p2p_clients.items():
                try:
                    socket.close()
                except Exception:
                    pass
            self._p2p_clients.clear()

        if hasattr(self, "_p2p_ctx") and self._p2p_ctx:
            try:
                self._p2p_ctx.term()
            except Exception:
                pass

        # Call parent exit to cleanup executor
        super().exit()

    def _register_with_nanoctrl(self):
        """Register engine with dlslime-ctrl and start heartbeat.

        Also used as the ``on_not_found`` callback — if dlslime-ctrl restarts and
        loses state, the heartbeat thread re-invokes this to re-register without
        restarting the heartbeat thread itself.
        """
        if not self.config.ctrl_address:
            return

        if self._nanoctrl is None:
            self._nanoctrl = NanoCtrlClient(
                self.config.ctrl_address, self.config.ctrl_scope
            )

        from dlengine.utils.network import get_advertise_host

        zmq_host = get_advertise_host(self.config.host)

        peer_addrs = self.executor.get_peer_agent_addrs()
        # Compute gdn_num_slots to match allocate_gdn_states logic. The active
        # region includes gdn_state_cache_slots parked slots for session-scoped
        # state caching, so active = max_num_seqs + gdn_state_cache_slots:
        # MTP (num_speculative_tokens > 0) → active * 2 + 1 (active + backup + dummy)
        # No MTP → active + 1 (active + dummy)
        max_bs = self.config.max_num_seqs + max(
            0, getattr(self.config, "gdn_state_cache_slots", 0)
        )
        if self.config.num_speculative_tokens > 0:
            gdn_num_slots = max_bs * 2 + 1
        else:
            gdn_num_slots = max_bs + 1
        architecture = (getattr(self.config.hf_config, "architectures", None) or [""])[
            0
        ]
        is_nsa_mla_hisparse = bool(self.config.enable_hisparse) and architecture in (
            "DeepseekV32ForCausalLM",
            "GlmMoeDsaForCausalLM",
        )
        metadata = {
            "role": self.config.mode,
            # Total worker count (all pipeline stages). ``peer_addrs`` has one
            # entry per worker, ordered by global rank:
            # pp_idx * (dp*sp*tp) + dp_idx*(sp*tp) + sp_idx*tp + tp_idx.
            "world_size": self.config.world_size,
            "num_blocks": self.config.num_kvcache_blocks,
            "host": zmq_host,
            "port": self.config.port,
            "peer_addrs": peer_addrs,
            "p2p_host": zmq_host,
            "p2p_port": self.p2p_port if self.p2p_port else 0,
            "max_num_seqs": self.config.max_num_seqs,
            "gdn_num_slots": gdn_num_slots,
            "model_path": self.config.model,  # tokenizer directory = model directory
            # Cache/topology compatibility contract consumed by PD migration.
            # Keep these explicit: equal world_size is insufficient to prove
            # that two DP+EP deployments shard attention/cache identically.
            "architecture": architecture,
            "attention_dp": self.config.attention_dp,
            "attention_sp": self.config.attention_sp,
            "attention_tp": self.config.attention_tp,
            "ffn_ep": self.config.ffn_ep,
            # Pipeline layout. A pp=1 decode engine uses these to map each
            # global layer to the prefill stage that owns it during PD KV
            # migration. pp_layer_ranges is the authoritative [start, end)
            # decoder-layer range per stage (uneven splits included, e.g.
            # Gemma4's reserved final-stage suffix).
            "pp": self.config.pp,
            "num_hidden_layers": getattr(self.config.hf_config, "num_hidden_layers", 0),
            "pp_layer_ranges": self._pp_layer_ranges(),
            "pp_cache_layer_indices": self._pp_cache_layer_indices(),
            "pp_dsv4_ratio_layer_indices": self._pp_dsv4_ratio_layer_indices(),
            "num_local_kv_heads": (
                getattr(self.config.hf_config, "num_key_value_heads", 1)
                // self.config.attention_tp
            ),
            "kvcache_block_size": self.config.kvcache_block_size,
            "num_host_blocks": self.config.num_host_kvcache_blocks,
            "indexer_num_blocks": (
                self.config.num_kvcache_blocks if is_nsa_mla_hisparse else 0
            ),
            "enable_hisparse": bool(self.config.enable_hisparse),
            "hisparse_device_buffer_size": (
                self.config.hisparse_device_buffer_size
                if self.config.enable_hisparse
                else 0
            ),
            "hisparse_cold_tier": ("decode_host" if is_nsa_mla_hisparse else "device"),
            # Gemma4 SWA writes its bounded ring during prefill and therefore
            # supports HiSparse in both phases. NSA/MLA models deliberately
            # keep prefill ordinary and enable HiSparse only on decode.
            "hisparse_phase": (
                "decode_only"
                if is_nsa_mla_hisparse
                else "prefill_and_decode" if self.config.enable_hisparse else "disabled"
            ),
        }

        ok = self._nanoctrl.register(
            self.engine_id,
            kind=self.config.mode,
            endpoint={"host": zmq_host, "port": self.config.port},
            metadata=metadata,
        )
        if ok:
            # start_heartbeat is a no-op if the thread is already running
            # (re-registration path from on_not_found callback)
            self._nanoctrl.start_heartbeat(
                on_not_found=self._register_with_nanoctrl,
                name=f"heartbeat-{self.engine_id}",
            )

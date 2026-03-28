import atexit
import json
from typing import List, Set, Tuple

import ray
from nanoctrl.client import NanoCtrlClient
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanodeploy.config import Config
from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.engine.ray_utils import get_available_nodes_with_master_first
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class LLM(LLMEngine):
    """LLM class with Ray remote execution support."""

    @classmethod
    def as_remote(cls, config):
        ray_address = getattr(config, "ray_address", "127.0.0.1:6379")
        master_address = getattr(config, "master_address", "127.0.0.1:6006")
        ray.init(address=ray_address, ignore_reinit_error=True)

        nodes = get_available_nodes_with_master_first(master_address)
        target_node_id = nodes[0]["NodeID"]

        return (
            ray.remote(num_cpus=1, num_gpus=0)(cls)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=target_node_id, soft=False
                )
            )
            .remote(config)
        )


class LLMComponent(LLM):
    """LLM component with lifecycle management and service discovery integration.

    This class extends LLM with:
    - NanoCtrl service registration and heartbeat
    - Peer engine information management for distributed serving
    - Automatic resource cleanup on shutdown
    """

    def __init__(self, config: Config):
        # In PD disagg mode, verify NanoCtrl is reachable before heavy model loading.
        if config.nanoctrl_address:
            NanoCtrlClient(
                config.nanoctrl_address, config.nanoctrl_scope
            ).check_connection()

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

        # NanoCtrl lifecycle client (None when nanoctrl_address is not configured)
        self._nanoctrl: NanoCtrlClient | None = None

        # Register with NanoCtrl if configured.
        # engine_server.py will re-register after binding its own P2P socket
        # (replacing p2p_port), but for direct LLMComponent usage (e.g.
        # deepseek_v3_disagg.py) this is the only registration point.
        if self.config.nanoctrl_address:
            self._register_with_nanoctrl()

        atexit.register(self.shutdown)

    def get_engine_info(self, status: str = "ready") -> str:
        """Get engine info as JSON string."""
        # Get peer_agent addresses from all workers
        peer_addrs = (
            self.executor.get_peer_agent_addrs()
            if hasattr(self.executor, "get_peer_agent_addrs")
            else []
        )

        # For ZMQ connection: use 127.0.0.1 if host is 0.0.0.0 (localhost mode),
        # otherwise use the specified host IP (distributed mode)
        zmq_host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host

        engine_info = {
            "id": self.engine_id,
            "role": self.config.mode,
            "rank": 0,
            "world_size": self.config.attn_world_size,
            "num_blocks": self.config.num_kvcache_blocks,
            "host": zmq_host,
            "port": self.config.port,
            "status": status,
            "peer_addrs": peer_addrs,
            "p2p_host": zmq_host,
            "p2p_port": self.p2p_port if self.p2p_port else 0,
        }

        return json.dumps(engine_info)

    def set_peer_info(self, remote_engine_info: str | bytes) -> None:
        """Store remote engine info including peer_addrs and P2P port for lazy migration.

        This method parses the remote engine's info and stores it so that during
        migration, the endpoints can be embedded in BlockContext.endpoints for
        lazy P2P connection.

        Args:
            remote_engine_info: JSON string or FlatBuffers bytes (for backward compatibility)
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

    def _fetch_peer_info_from_nanoctrl(self, target_engine_id: str) -> bool:
        """Fetch peer engine info from NanoCtrl on-demand."""
        if self._nanoctrl is None:
            logger.error("Cannot fetch peer info: NanoCtrl not configured")
            return False

        engine_info = self._nanoctrl.get_engine_info(target_engine_id)
        if engine_info:
            logger.info(f"NanoCtrl returned engine_info: {engine_info}")
            self.set_peer_info(json.dumps(engine_info))
            logger.info(f"Fetched peer info for {target_engine_id} from NanoCtrl")
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
        import flatbuffers
        import numpy as np
        import zmq

        from nanodeploy.fbs.FreeVisionSlots import (
            FreeVisionSlotsAddEncoderEngineId,
            FreeVisionSlotsAddSlotIndices,
            FreeVisionSlotsAddSourceEngineId,
            FreeVisionSlotsEnd,
            FreeVisionSlotsStart,
            FreeVisionSlotsStartSlotIndicesVector,
        )
        from nanodeploy.server.zmq_protocol import encode_packet

        if not slot_indices:
            return

        # Get target encoder P2P address
        if target_encoder_id not in self._peer_info:
            if not self._fetch_peer_info_from_nanoctrl(target_encoder_id):
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

        # Build FreeVisionSlots FlatBuffer
        builder = flatbuffers.Builder(256)
        encoder_id_off = builder.CreateString(target_encoder_id)
        source_id_off = builder.CreateString(self.engine_id)

        FreeVisionSlotsStartSlotIndicesVector(builder, len(slot_indices))
        for idx in reversed(slot_indices):
            builder.PrependInt32(idx)
        slot_vec = builder.EndVector()

        FreeVisionSlotsStart(builder)
        FreeVisionSlotsAddEncoderEngineId(builder, encoder_id_off)
        FreeVisionSlotsAddSlotIndices(builder, slot_vec)
        FreeVisionSlotsAddSourceEngineId(builder, source_id_off)
        free_req = FreeVisionSlotsEnd(builder)
        builder.Finish(free_req)

        payload = bytes(builder.Output())

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
        """Send P2P free instruction directly to remote engine (no NanoRoute).

        Args:
            target_engine_id: Engine ID to send free instruction to
            seq_ids: List of sequence IDs to free
        """
        import flatbuffers
        import numpy as np
        import zmq

        from nanodeploy.fbs.FreeSequences import (
            FreeSequencesAddSeqIds,
            FreeSequencesAddSourceEngineId,
            FreeSequencesEnd,
            FreeSequencesStart,
        )
        from nanodeploy.server.zmq_protocol import encode_packet

        if not seq_ids:
            logger.warning(f"No sequence IDs to free for engine {target_engine_id}")
            return

        # Get target engine P2P address from peer_info (fetch on-demand if not cached)
        if target_engine_id not in self._peer_info:
            logger.info(
                f"Peer info for {target_engine_id} not cached, fetching from NanoCtrl..."
            )
            if not self._fetch_peer_info_from_nanoctrl(target_engine_id):
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

        # Build FreeSequences FlatBuffer
        builder = flatbuffers.Builder(256)
        seq_ids_vec = builder.CreateNumpyVector(np.array(seq_ids, dtype=np.uint64))
        source_id_offset = builder.CreateString(self.engine_id)

        FreeSequencesStart(builder)
        FreeSequencesAddSeqIds(builder, seq_ids_vec)
        FreeSequencesAddSourceEngineId(builder, source_id_offset)
        free_req = FreeSequencesEnd(builder)
        builder.Finish(free_req)

        payload = bytes(builder.Output())

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
        # Stop heartbeat and unregister from NanoCtrl
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
        """Register engine with NanoCtrl and start heartbeat.

        Also used as the ``on_not_found`` callback — if NanoCtrl restarts and
        loses state, the heartbeat thread re-invokes this to re-register without
        restarting the heartbeat thread itself.
        """
        if not self.config.nanoctrl_address:
            return

        if self._nanoctrl is None:
            self._nanoctrl = NanoCtrlClient(
                self.config.nanoctrl_address, self.config.nanoctrl_scope
            )

        if self.config.host in ("0.0.0.0", ""):
            from nanodeploy.context.distributed import get_local_ip

            zmq_host = get_local_ip()
        else:
            zmq_host = self.config.host

        peer_addrs = self.get_peer_agent_addrs()
        extra = {
            "role": self.config.mode,
            "world_size": self.config.attn_world_size,
            "num_blocks": self.config.num_kvcache_blocks,
            "host": zmq_host,
            "port": self.config.port,
            "peer_addrs": peer_addrs,
            "p2p_host": zmq_host,
            "p2p_port": self.p2p_port if self.p2p_port else 0,
            "max_num_seqs": self.config.max_num_seqs,
            "model_path": self.config.model,  # tokenizer directory = model directory
        }

        ok = self._nanoctrl.register(self.engine_id, extra)
        if ok:
            # start_heartbeat is a no-op if the thread is already running
            # (re-registration path from on_not_found callback)
            self._nanoctrl.start_heartbeat(
                on_not_found=self._register_with_nanoctrl,
                name=f"heartbeat-{self.engine_id}",
            )

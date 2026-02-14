import atexit
import json
import threading
from typing import List, Optional, Set, Tuple

import httpx
import ray
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
        super().__init__(config)

        # peer_engine_id -> dict(num_blocks, world_size, peer_addrs, p2p_host, p2p_port) for lazy migration
        self._peer_info: dict[str, dict] = {}
        self.active_p2p_links: Set[str] = set()

        # P2P ZMQ socket for receiving free instructions (dynamically assigned port)
        self.p2p_socket = None
        self.p2p_port = None

        # P2P client connections cache: engine_id -> ZMQ socket
        self._p2p_clients: dict[str, any] = {}
        self._p2p_ctx = None

        # Engine lifecycle management
        self._nanoctrl_registered = False
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Note: Registration deferred until P2P socket is bound in engine_server
        # This ensures p2p_port is set before registration

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
        """Fetch peer engine info from NanoCtrl on-demand.

        Args:
            target_engine_id: Engine ID to fetch info for

        Returns:
            True if successfully fetched and cached, False otherwise
        """
        if not self.config.nanoctrl_address:
            logger.error("Cannot fetch peer info: NanoCtrl not configured")
            return False

        try:
            url = f"http://{self.config.nanoctrl_address}/get_engine_info"
            payload = {"engine_id": target_engine_id}
            if self.config.scope:
                payload["scope"] = self.config.scope

            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "ok":
                    engine_info = data.get("engine_info", {})
                    if engine_info:
                        logger.info(f"NanoCtrl returned engine_info: {engine_info}")
                        engine_info_str = json.dumps(engine_info)
                        self.set_peer_info(engine_info_str)
                        logger.info(
                            f"Fetched peer info for {target_engine_id} from NanoCtrl"
                        )
                        return True

                logger.error(
                    f"Failed to fetch peer info for {target_engine_id}: {data.get('message', 'No engine_info in response')}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Error fetching peer info from NanoCtrl for {target_engine_id}: {e}"
            )
            return False

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
        # Stop heartbeat thread
        if hasattr(self, "_heartbeat_stop_event"):
            self._heartbeat_stop_event.set()
        if hasattr(self, "_heartbeat_thread") and self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

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

        # Unregister from NanoCtrl before exiting
        if hasattr(self, "_nanoctrl_registered") and self._nanoctrl_registered:
            self._unregister_from_nanoctrl()

        # Call parent exit to cleanup executor
        super().exit()

    def _register_with_nanoctrl(self):
        """Register engine information with NanoCtrl control plane."""
        if not self.config.nanoctrl_address:
            return

        try:
            # Get peer agent addresses (may be empty initially, but that's ok)
            peer_addrs = self.get_peer_agent_addrs()
            logger.info(
                f"Registering engine {self.engine_id} with {len(peer_addrs)} peer addresses"
            )

            # Prepare registration payload
            # For ZMQ connection: use the actual node IP so that route (which
            # may be on a different node) can reach us.  Only fall back to
            # 127.0.0.1 if we really only bind to loopback.
            if self.config.host in ("0.0.0.0", ""):
                from nanodeploy.context.distributed import get_local_ip

                zmq_host = get_local_ip()
            else:
                zmq_host = self.config.host

            payload = {
                "engine_id": self.engine_id,
                "role": self.config.mode,
                "world_size": self.config.attn_world_size,
                "num_blocks": self.config.num_kvcache_blocks,
                "host": zmq_host,
                "port": self.config.port,
                "peer_addrs": peer_addrs,
                "p2p_host": zmq_host,
                "p2p_port": self.p2p_port if self.p2p_port else 0,
            }
            # Add scope if configured
            if self.config.scope:
                payload["scope"] = self.config.scope

            url = f"http://{self.config.nanoctrl_address}/register_engine"
            logger.info(
                f"Registering engine with NanoCtrl at {url}, payload: {payload}"
            )

            # Use sync client since this is called during init
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    self._nanoctrl_registered = True
                    logger.info(
                        f"Successfully registered engine {payload['engine_id']} with NanoCtrl"
                    )
                    # Start heartbeat thread after successful registration
                    self._start_heartbeat()
                else:
                    logger.error(
                        f"Failed to register engine: {data.get('message', 'Unknown error')}"
                    )
                    logger.error(f"Response data: {data}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error registering engine with NanoCtrl: {e}")
            logger.error(
                f"Response: {e.response.text if e.response else 'No response'}"
            )
        except Exception as e:
            logger.error(f"Error registering engine with NanoCtrl: {e}", exc_info=True)
            # Don't raise - allow engine to continue even if registration fails

    def _unregister_from_nanoctrl(self):
        """Unregister engine from NanoCtrl control plane."""
        if not self.config.nanoctrl_address:
            return

        try:
            payload = {"engine_id": self.engine_id}
            if self.config.scope:
                payload["scope"] = self.config.scope
            url = f"http://{self.config.nanoctrl_address}/unregister_engine"
            logger.info(f"Unregistering engine {self.engine_id} from NanoCtrl")

            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    logger.info(
                        f"Successfully unregistered engine {self.engine_id} from NanoCtrl"
                    )
                else:
                    logger.warning(
                        f"Failed to unregister engine: {data.get('message', 'Unknown error')}"
                    )
        except Exception as e:
            logger.error(f"Error unregistering engine from NanoCtrl: {e}")
            # Don't raise - exit should continue even if unregistration fails

    def _start_heartbeat(self):
        """Start heartbeat thread to keep engine registration alive."""
        if not self.config.nanoctrl_address or not self._nanoctrl_registered:
            return

        # Stop existing heartbeat thread if any
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop_event.clear()

        def heartbeat_loop():
            """Heartbeat loop: send heartbeat every 15 seconds."""
            while not self._heartbeat_stop_event.wait(15.0):
                try:
                    self._heartbeat_to_nanoctrl()
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}", exc_info=True)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop, name=f"heartbeat-{self.engine_id}", daemon=True
        )
        self._heartbeat_thread.start()
        logger.info(
            f"Started heartbeat thread for engine {self.engine_id} (interval: 15s)"
        )

    def _heartbeat_to_nanoctrl(self):
        """Send heartbeat to NanoCtrl to refresh TTL."""
        if not self.config.nanoctrl_address:
            return

        try:
            payload = {"engine_id": self.engine_id}
            if self.config.scope:
                payload["scope"] = self.config.scope
            url = f"http://{self.config.nanoctrl_address}/heartbeat_engine"

            # Use sync client with short timeout
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "ok":
                    logger.debug(f"Heartbeat successful for engine {self.engine_id}")
                elif data.get("status") == "not_found":
                    # Engine not found, re-register
                    logger.warning(
                        f"Engine {self.engine_id} not found in NanoCtrl, re-registering..."
                    )
                    self._nanoctrl_registered = False
                    self._register_with_nanoctrl()
                else:
                    logger.warning(
                        f"Heartbeat failed for engine {self.engine_id}: {data.get('message', 'Unknown error')}"
                    )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in heartbeat: {e}")
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}", exc_info=True)

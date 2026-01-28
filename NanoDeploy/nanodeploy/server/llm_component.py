import json
import threading
import time
from typing import Any, Dict, Set

import etcd3
import flatbuffers
import nanodeploy.fbs.EndpointBytes as EndpointBytes
import nanodeploy.fbs.EndpointInfoList as EndpointInfoList
import nanodeploy.fbs.EngineInfo as EngineInfo
import nanodeploy.fbs.P2PInit as P2PInit
import nanodeploy.fbs.Peer as Peer
import nanodeploy.fbs.RdmaEndpointInfo as RdmaEndpointInfo
from nanodeploy.config import Config
from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class LLMComponent(LLMEngine):
    def __init__(self, config: Config):
        super().__init__(config)

        if config.enable_etcd:
            etcd_host, etcd_port = config.etcd_address.split(":")
            self.etcd = etcd3.client(host=etcd_host, port=int(etcd_port))
            self.cluster_id = config.cluster_id
            self.lease = None
            self.active_p2p_links: Set[str] = set()
            self.handshake_watches: Dict[str, Any] = {}

            self._setup_etcd()
            self.discovery_thread = threading.Thread(
                target=self._watch_loop, daemon=True
            )
            self.discovery_thread.start()

    def get_engine_info(self, status: str = "ready") -> bytes:
        builder = flatbuffers.Builder(1024)

        id_off = builder.CreateString(self.engine_id)
        role_off = builder.CreateString(self.config.mode)
        host_off = builder.CreateString(self.config.host)
        status_off = builder.CreateString(status)

        EngineInfo.Start(builder)
        EngineInfo.AddId(builder, id_off)
        EngineInfo.AddRole(builder, role_off)
        EngineInfo.AddRank(builder, 0)
        EngineInfo.AddWorldSize(builder, self.config.attn_world_size)
        EngineInfo.AddNumBlocks(builder, self.config.num_kvcache_blocks)
        EngineInfo.AddHost(builder, host_off)
        EngineInfo.AddPort(builder, self.config.port)
        EngineInfo.AddStatus(builder, status_off)

        info_off = EngineInfo.End(builder)
        builder.Finish(info_off)

        return bytes(builder.Output())

    def p2p_init(self, remote_engine_info_bytes: bytes) -> bytes:
        info = EngineInfo.EngineInfo.GetRootAsEngineInfo(remote_engine_info_bytes, 0)

        remote_engine_name = info.Id().decode("utf-8") if info.Id() else ""
        num_kv_blocks = info.NumBlocks()
        remote_world_size = info.WorldSize()

        # this is List[List[dict]] (outer: local ranks, inner: remote ranks)
        my_info_list = self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

        peer_t = Peer.PeerT()
        # The Peer packet is sent TO the remote, but it describes ME (the sender).
        # So "remoteId" (from receiver's perspective) should be MY id.
        peer_t.remoteId = self.engine_id

        peer_t.remoteInfo = []
        for local_rank_eps in my_info_list:
            ep_list_t = EndpointInfoList.EndpointInfoListT()
            ep_list_t.endpoints = []
            for ep_info in local_rank_eps:
                # Serialize JSON dict to bytes
                ep_json_str = json.dumps(ep_info)
                ep_bytes = ep_json_str.encode("utf-8")

                # Wrap bytes into EndpointBytesT
                ep_bytes_t = EndpointBytes.EndpointBytesT()
                ep_bytes_t.data = [b for b in ep_bytes]
                ep_list_t.endpoints.append(ep_bytes_t)
            peer_t.remoteInfo.append(ep_list_t)

        builder = flatbuffers.Builder(4096)
        peer_offset = peer_t.Pack(builder)
        builder.Finish(peer_offset)

        return bytes(builder.Output())

    def p2p_connect(self, peer_bytes: bytes):
        # 1. Parse PeerT
        peer_view = Peer.Peer.GetRootAsPeer(peer_bytes, 0)
        peer_t = Peer.PeerT.InitFromObj(peer_view)

        remote_engine_name = peer_t.remoteId
        if isinstance(remote_engine_name, bytes):
            remote_engine_name = remote_engine_name.decode("utf-8")

        # 2. Convert List[EndpointInfoListT] back to List[List[dict]]
        remote_endpoints_info = []
        if peer_t.remoteInfo:
            for ep_list_t in peer_t.remoteInfo:
                local_rank_list = []
                if ep_list_t.endpoints:
                    for ep_bytes_t in ep_list_t.endpoints:
                        # Extract JSON bytes and deserialize
                        if ep_bytes_t.data is not None:
                            json_bytes = bytes(ep_bytes_t.data)
                            json_str = json_bytes.decode("utf-8")
                            ep_info = json.loads(json_str)
                            local_rank_list.append(ep_info)
                        else:
                            # Should not happen, but robust handling
                            local_rank_list.append({})
                remote_endpoints_info.append(local_rank_list)

        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def _setup_etcd(self):
        """Register node in etcd and start keep-alive."""
        logger.info(f"Registering node {self.engine_id} in cluster {self.cluster_id}")
        self.lease = self.etcd.lease(ttl=10)

        val = self.get_engine_info(status="initializing")
        key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{self.engine_id}"
        self.etcd.put(key, val, lease=self.lease)

        # Start keep-alive thread
        def keep_alive():
            logger.info("Starting etcd lease keep-alive loop")
            while hasattr(self, "etcd"):
                try:
                    self.lease.refresh()
                    time.sleep(3)  # Refresh every 3s (TTL is 10s)
                except Exception as e:
                    logger.error(f"Lease refresh failed: {e}")
                    time.sleep(1)

        self.ka_thread = threading.Thread(target=keep_alive, daemon=True)
        self.ka_thread.start()

    def _update_status(self, status: str):
        """Update node status in etcd."""
        logger.info(f"Updating node status to {status}")
        val = self.get_engine_info(status=status)
        key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{self.engine_id}"
        self.etcd.put(key, val, lease=self.lease)

    def _watch_loop(self):
        """Watch for new nodes and handshake events."""
        node_prefix = f"/nanodeploy/mesh/{self.cluster_id}/nodes/"
        logger.info(f"Starting discovery scan on prefix: {node_prefix}")

        # 1. Scan for existing nodes
        try:
            existing_nodes = self.etcd.get_prefix(node_prefix)
            count = 0
            for val, meta in existing_nodes:
                count += 1
                key = meta.key.decode("utf-8")
                peer_id = key.split("/")[-1]
                logger.info(f"Scan found node: {peer_id} (Key: {key})")

                if peer_id == self.engine_id:
                    logger.info("Skipping self in scan.")
                    continue

                logger.info(f"Discovered existing peer: {peer_id}")
                peer_info = EngineInfo.EngineInfo.GetRootAsEngineInfo(val, 0)
                self._on_peer_online(
                    peer_id, peer_info, peer_info_bytes=val, is_initiator=True
                )
            logger.info(f"Scan completed. Found {count} nodes.")
            self._update_status("ready")
        except Exception as e:
            logger.error(f"Discovery scan failed: {e}")
            import traceback

            traceback.print_exc()

        logger.info("Entering event watch loop...")
        events_iterator, cancel = self.etcd.watch_prefix(node_prefix)

        for event in events_iterator:
            key = event.key.decode("utf-8")
            peer_id = key.split("/")[-1]
            if peer_id == self.engine_id:
                continue
            if isinstance(event, etcd3.events.PutEvent):
                peer_info = EngineInfo.EngineInfo.GetRootAsEngineInfo(event.value, 0)
                self._on_peer_online(
                    peer_id, peer_info, peer_info_bytes=event.value, is_initiator=False
                )
            elif isinstance(event, etcd3.events.DeleteEvent):
                self._on_peer_offline(peer_id)

    def _on_peer_online(
        self,
        peer_id: str,
        peer_info,
        peer_info_bytes: bytes,
        is_initiator: bool = False,
    ):
        # Determine paths based on initiator status
        if is_initiator:
            initiator = self.engine_id
            terminator = peer_id
            path_root = (
                f"/nanodeploy/mesh/{self.cluster_id}/handshake/{initiator}/{terminator}"
            )

            logger.info(f"Initiating handshake with peer {peer_id} (I am Initiator)")
            my_init_bytes = self.p2p_init(peer_info_bytes)
            self.etcd.put(f"{path_root}/init", my_init_bytes)
            self._start_handshake_watch(
                peer_id, f"{path_root}/resp", "resp", is_initiator=True
            )

        else:
            initiator = peer_id
            terminator = self.engine_id
            path_root = (
                f"/nanodeploy/mesh/{self.cluster_id}/handshake/{initiator}/{terminator}"
            )

            logger.info(f"Waiting for handshake from peer {peer_id} (I am Responder)")

            self._start_handshake_watch(
                peer_id, f"{path_root}/init", "init", is_initiator=False
            )

    def _start_handshake_watch(
        self, peer_id: str, key: str, expected_step: str, is_initiator: bool
    ):
        if peer_id in self.handshake_watches:
            return

        logger.info(
            f"Starting polling for handshake key: {key} (expecting {expected_step})"
        )

        def poll_loop():
            start_time = time.time()
            while peer_id not in self.active_p2p_links:
                if not hasattr(self, "executor"):
                    return

                try:
                    val, _ = self.etcd.get(key)
                    if val:
                        logger.info(f"Polled handshake key {key} found!")
                        self._handle_handshake_step(
                            peer_id, expected_step, val, is_initiator
                        )
                        return
                    time.sleep(0.5)
                except Exception as e:
                    if not hasattr(self, "executor"):
                        return
                    logger.error(f"Error polling {key}: {e}")
                    time.sleep(1.0)

        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()
        self.handshake_watches[peer_id] = t

    def _handle_handshake_step(
        self, peer_id: str, step: str, payload_bytes: bytes, is_initiator: bool
    ):
        # Safety check
        if not hasattr(self, "executor"):
            return

        # Idempotency Check:
        if peer_id in self.active_p2p_links:
            logger.debug(f"Ignoring handshake step for {peer_id} (already connected)")
            return

        logger.info(f"Processing handshake step '{step}' from {peer_id}")

        if step == "init":
            node_key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{peer_id}"
            val, _ = self.etcd.get(node_key)
            if not val:
                logger.error(
                    f"Cannot find node info for {peer_id} during handshake response"
                )
                return

            # Initialize my endpoints first (creates self.endpoints[peer_id])
            my_resp_bytes = self.p2p_init(val)

            # Then connect to remote endpoints
            self.p2p_connect(payload_bytes)

            initiator = peer_id
            terminator = self.engine_id
            path_root = (
                f"/nanodeploy/mesh/{self.cluster_id}/handshake/{initiator}/{terminator}"
            )

            self.etcd.put(f"{path_root}/resp", my_resp_bytes)
            self.active_p2p_links.add(peer_id)

        elif step == "resp":
            self.p2p_connect(payload_bytes)
            self.active_p2p_links.add(peer_id)

        if peer_id in self.handshake_watches:
            # Clean up watch
            del self.handshake_watches[peer_id]

    def _on_peer_offline(self, peer_id: str):
        if peer_id in self.active_p2p_links:
            logger.info(f"Peer {peer_id} offline. Cleaning up DLSlime link.")
            try:
                self.executor.p2p_disconnect(peer_id)
            except:
                pass
            self.active_p2p_links.remove(peer_id)

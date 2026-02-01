import json
import threading
import time
from typing import List, Set, Tuple

import etcd3
import flatbuffers
import nanodeploy.fbs.EndpointBytes as EndpointBytes
import nanodeploy.fbs.EndpointInfoList as EndpointInfoList
import nanodeploy.fbs.EngineInfo as EngineInfo
import nanodeploy.fbs.Peer as Peer
from nanodeploy.config import Config
from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.llm import LLM
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class LLMComponent(LLM):
    def __init__(self, config: Config):
        super().__init__(config)

        # peer_engine_id -> dict(num_blocks, world_size, peer_addrs) for lazy migration
        self._peer_info: dict[str, dict] = {}

        if config.enable_etcd:
            etcd_host, etcd_port = config.etcd_address.split(":")
            self.etcd = etcd3.Client(host=etcd_host, port=int(etcd_port))
            self.cluster_id = config.cluster_id
            self.lease = None
            self.active_p2p_links: Set[str] = set()

            logger.info(
                f"LLMComponent init: Node={self.engine_id}, Mode={self.config.mode}, Cluster={self.cluster_id}"
            )
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

        # Get peer_agent addresses from all workers
        peer_addrs = (
            self.executor.get_peer_agent_addrs()
            if hasattr(self.executor, "get_peer_agent_addrs")
            else []
        )
        peer_addrs_offs = [builder.CreateString(addr) for addr in peer_addrs if addr]

        # Create peer_addrs vector
        EngineInfo.StartPeerAddrsVector(builder, len(peer_addrs_offs))
        for off in reversed(peer_addrs_offs):
            builder.PrependUOffsetTRelative(off)
        peer_addrs_vec = builder.EndVector()

        EngineInfo.Start(builder)
        EngineInfo.AddId(builder, id_off)
        EngineInfo.AddRole(builder, role_off)
        EngineInfo.AddRank(builder, 0)
        EngineInfo.AddWorldSize(builder, self.config.attn_world_size)
        EngineInfo.AddNumBlocks(builder, self.config.num_kvcache_blocks)
        EngineInfo.AddHost(builder, host_off)
        EngineInfo.AddPort(builder, self.config.port)
        EngineInfo.AddStatus(builder, status_off)
        EngineInfo.AddPeerAddrs(builder, peer_addrs_vec)

        info_off = EngineInfo.End(builder)
        builder.Finish(info_off)

        return bytes(builder.Output())

    def p2p_init(self, remote_engine_info_bytes: bytes) -> bytes:
        """Legacy RPC: init endpoints and return Peer payload. Used by engine_server."""
        info = EngineInfo.EngineInfo.GetRootAsEngineInfo(remote_engine_info_bytes, 0)
        remote_engine_name = info.Id().decode("utf-8") if info.Id() else ""
        num_kv_blocks = info.NumBlocks()
        remote_world_size = info.WorldSize()
        my_info_list = self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )
        peer_t = Peer.PeerT()
        peer_t.remoteId = self.engine_id
        peer_t.remoteInfo = []
        for local_rank_eps in my_info_list:
            ep_list_t = EndpointInfoList.EndpointInfoListT()
            ep_list_t.endpoints = []
            for ep_info in local_rank_eps:
                ep_json_str = json.dumps(ep_info)
                ep_bytes = ep_json_str.encode("utf-8")
                ep_bytes_t = EndpointBytes.EndpointBytesT()
                ep_bytes_t.data = [b for b in ep_bytes]
                ep_list_t.endpoints.append(ep_bytes_t)
            peer_t.remoteInfo.append(ep_list_t)
        builder = flatbuffers.Builder(4096)
        peer_offset = peer_t.Pack(builder)
        builder.Finish(peer_offset)
        return bytes(builder.Output())

    def set_peer_info(self, remote_engine_info_bytes: bytes) -> None:
        """Store remote engine info including peer_addrs for lazy migration.

        This method parses the remote engine's info and stores it so that during
        migration, the endpoints can be embedded in BlockContext.endpoints for
        lazy P2P connection.
        """
        info = EngineInfo.EngineInfo.GetRootAsEngineInfo(remote_engine_info_bytes, 0)
        remote_engine_id = info.Id().decode("utf-8") if info.Id() else ""
        num_kv_blocks = info.NumBlocks()

        # Extract peer_addrs from EngineInfo
        peer_addrs = []
        for i in range(info.PeerAddrsLength()):
            addr = info.PeerAddrs(i)
            if addr:
                peer_addrs.append(
                    addr.decode("utf-8") if isinstance(addr, bytes) else addr
                )

        # Store in engine for use during migration
        self._peer_info[remote_engine_id] = {
            "num_blocks": num_kv_blocks,
            "world_size": info.WorldSize(),
            "peer_addrs": peer_addrs,
        }
        logger.info(
            f"Stored peer info for {remote_engine_id}: {len(peer_addrs)} addresses"
        )

    def p2p_connect(self, peer_bytes: bytes):
        """Legacy RPC: connect using Peer payload. Used by engine_server."""
        peer_view = Peer.Peer.GetRootAsPeer(peer_bytes, 0)
        peer_t = Peer.PeerT.InitFromObj(peer_view)
        remote_engine_name = peer_t.remoteId
        if isinstance(remote_engine_name, bytes):
            remote_engine_name = remote_engine_name.decode("utf-8")
        remote_endpoints_info = []
        if peer_t.remoteInfo:
            for ep_list_t in peer_t.remoteInfo:
                local_rank_list = []
                if ep_list_t.endpoints:
                    for ep_bytes_t in ep_list_t.endpoints:
                        if ep_bytes_t.data is not None:
                            json_bytes = bytes(ep_bytes_t.data)
                            ep_info = json.loads(json_bytes.decode("utf-8"))
                            local_rank_list.append(ep_info)
                        else:
                            local_rank_list.append({})
                remote_endpoints_info.append(local_rank_list)
        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def ensure_p2p_connected(self, peer_id: str) -> None:
        """Ensure P2P link to peer_id is established (lazy connect). Idempotent."""
        if peer_id in self.active_p2p_links:
            return
        if peer_id not in self._peer_info:
            logger.warning(
                f"ensure_p2p_connected: no peer info for {peer_id}, skipping"
            )
            return
        addrs, num_blocks = self._peer_info[peer_id]
        try:
            self.executor.ensure_p2p_connected(peer_id, addrs, num_blocks)
            self.active_p2p_links.add(peer_id)
            logger.info(f"P2P link ensured to {peer_id}")
        except Exception as e:
            logger.error(f"ensure_p2p_connected failed for {peer_id}: {e}")
            raise

    def _setup_etcd(self):
        """Register node in etcd and start keep-alive."""
        logger.info(f"Registering node {self.engine_id} in cluster {self.cluster_id}")
        self.lease = self.etcd.Lease(ttl=10)

        val = self.get_engine_info(status="initializing")
        key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{self.engine_id}"
        # Use lease ID (int) instead of object
        self.etcd.put(key, val, lease=self.lease.ID)

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
        self.etcd.put(key, val, lease=self.lease.ID)

    def _watch_loop(self):
        """Watch for new nodes and handshake events."""
        node_prefix = f"/nanodeploy/mesh/{self.cluster_id}/nodes/"

        while True:
            try:
                logger.info(f"Starting discovery session on prefix: {node_prefix}")

                # 1. Start Watch FIRST to catch everything from this point on
                # etcd3-py watch_create returns the stream response object (iterator) directly.
                events_iterator = self.etcd.watch_create(node_prefix, prefix=True)
                logger.info(f"Watch created on {node_prefix}")

                # 2. Scan for existing nodes (captures anything that happened before watch)
                resp = self.etcd.range(node_prefix, prefix=True)
                kvs = resp.kvs if hasattr(resp, "kvs") else resp

                count = 0
                for kv in kvs:
                    if not hasattr(kv, "key"):
                        continue
                    count += 1
                    key = kv.key.decode("utf-8")
                    val = kv.value
                    peer_id = key.split("/")[-1]
                    if peer_id == self.engine_id:
                        continue

                    logger.info(f"Initial scan discovered peer: {peer_id} (Key: {key})")
                    try:
                        peer_info = EngineInfo.EngineInfo.GetRootAsEngineInfo(val, 0)
                        self._on_peer_online(peer_id, peer_info, peer_info_bytes=val)
                    except Exception as e:
                        logger.error(f"Failed to parse info for peer {peer_id}: {e}")

                logger.info(
                    f"Initial scan completed. Found {count} total nodes (including self)."
                )

                # 3. Update own status to 'ready'
                self._update_status("ready")

                # 4. Consume events from watch
                logger.info("Entering event watch loop processing...")
                for response in events_iterator:
                    # Handle WatchResponse (contains list of events)
                    if hasattr(response, "events"):
                        events = response.events if response.events is not None else []
                    else:
                        events = [response]

                    for event in events:
                        if not hasattr(event, "kv") or not event.kv:
                            continue

                        kv = event.kv
                        key = kv.key.decode("utf-8")
                        peer_id = key.split("/")[-1]
                        if peer_id == self.engine_id:
                            continue

                        # Robust event type detection
                        is_put = False
                        is_delete = False
                        if hasattr(event, "type"):
                            etype = str(event.type).upper()
                            if etype in ("PUT", "0", "PUTEVENT"):
                                is_put = True
                            elif etype in ("DELETE", "1", "DELETEEVENT"):
                                is_delete = True
                        if not is_delete and kv.value:
                            is_put = True

                        if is_put:
                            logger.info(
                                f"Watch discovered peer online/updated: {peer_id}"
                            )
                            try:
                                peer_info = EngineInfo.EngineInfo.GetRootAsEngineInfo(
                                    kv.value, 0
                                )
                                self._on_peer_online(
                                    peer_id, peer_info, peer_info_bytes=kv.value
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to parse peer info from watch: {e}"
                                )
                        elif is_delete:
                            logger.info(f"Watch discovered peer offline: {peer_id}")
                            self._on_peer_offline(peer_id)

                logger.warning(
                    "Watch stream ended unexpectedly, restarting discovery..."
                )
            except Exception as e:
                logger.error(f"Discovery loop encountered error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(5)

    def _on_peer_online(
        self,
        peer_id: str,
        peer_info,
        peer_info_bytes: bytes,
    ):
        # Discovery only: store peer addrs and num_blocks for lazy ensure_p2p_connected
        if peer_id in self._peer_info:
            return

        role = "unknown"
        if hasattr(peer_info, "Role") and peer_info.Role():
            role = peer_info.Role().decode("utf-8")
        logger.info(f"Peer Online: {peer_id} (Role: {role})")

        host = peer_info.Host().decode("utf-8") if peer_info.Host() else "127.0.0.1"
        port = (
            int(peer_info.Port())
            if hasattr(peer_info, "Port") and peer_info.Port() is not None
            else 5000
        )
        world_size = (
            int(peer_info.WorldSize())
            if hasattr(peer_info, "WorldSize") and peer_info.WorldSize() is not None
            else 1
        )
        num_blocks = (
            int(peer_info.NumBlocks())
            if hasattr(peer_info, "NumBlocks") and peer_info.NumBlocks() is not None
            else 0
        )

        addrs = [f"{host}:{port + r}" for r in range(world_size)]
        self._peer_info[peer_id] = (addrs, num_blocks)

    def _on_peer_offline(self, peer_id: str):
        if peer_id in self._peer_info:
            del self._peer_info[peer_id]
        if peer_id in self.active_p2p_links:
            logger.info(f"Peer {peer_id} offline. Cleaning up DLSlime link.")
            try:
                self.executor.p2p_disconnect(peer_id)
            except Exception:
                pass
            self.active_p2p_links.remove(peer_id)

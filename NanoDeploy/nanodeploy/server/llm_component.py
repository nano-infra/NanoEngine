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
            self.etcd = etcd3.Client(host=etcd_host, port=int(etcd_port))
            self.cluster_id = config.cluster_id
            self.lease = None
            self.active_p2p_links: Set[str] = set()
            self.handshake_watches: Dict[str, Any] = {}

            logger.info(f"LLMComponent init: Node={self.engine_id}, Mode={self.config.mode}, Cluster={self.cluster_id}")
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

                logger.info(f"Initial scan completed. Found {count} total nodes (including self).")

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
                            logger.info(f"Watch discovered peer online/updated: {peer_id}")
                            try:
                                peer_info = EngineInfo.EngineInfo.GetRootAsEngineInfo(kv.value, 0)
                                self._on_peer_online(peer_id, peer_info, peer_info_bytes=kv.value)
                            except Exception as e:
                                logger.error(f"Failed to parse peer info from watch: {e}")
                        elif is_delete:
                            logger.info(f"Watch discovered peer offline: {peer_id}")
                            self._on_peer_offline(peer_id)
                
                logger.warning("Watch stream ended unexpectedly, restarting discovery...")
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
        # Prevent multiple handshakes for same peer if already connected or in progress
        if peer_id in self.active_p2p_links:
            return
        if peer_id in self.handshake_watches:
            logger.debug(f"[Handshake] Task for {peer_id} already in progress, skipping redundant trigger")
            return

        # Log Role
        role = "unknown"
        if hasattr(peer_info, "Role") and peer_info.Role():
            role = peer_info.Role().decode("utf-8")

        logger.info(f"Peer Online: {peer_id} (Role: {role})")

        def _handshake_task():
            try:
                # Deterministic Initiator Election
                is_initiator = self.engine_id < peer_id
                
                initiator = self.engine_id if is_initiator else peer_id
                terminator = peer_id if is_initiator else self.engine_id
                path_root = (
                    f"/nanodeploy/mesh/{self.cluster_id}/handshake/{initiator}/{terminator}"
                )

                if is_initiator:
                    logger.info(f"[Handshake] Initiating with {peer_id}. Target init key: {path_root}/init")
                    try:
                        start_init = time.perf_counter()
                        my_init_bytes = self.p2p_init(peer_info_bytes)
                        duration = time.perf_counter() - start_init
                        logger.info(f"[Handshake] p2p_init success for {peer_id} ({len(my_init_bytes)} bytes) in {duration:.3f}s")
                    except Exception as e:
                        logger.error(f"[Handshake] p2p_init FAILED for {peer_id}: {e}")
                        return

                    logger.info(f"[Handshake] Writing /init key to etcd for {peer_id}")
                    self.etcd.put(f"{path_root}/init", my_init_bytes)
                    
                    logger.info(f"[Handshake] Starting poll for /resp from {peer_id}")
                    self._start_handshake_watch(
                        peer_id, f"{path_root}/resp", "resp"
                    )
                else:
                    logger.info(f"[Handshake] Responder role for {peer_id}. Waiting for {path_root}/init")
                    self._start_handshake_watch(
                        peer_id, f"{path_root}/init", "init"
                    )
            except Exception as e:
                logger.error(f"Handshake Error for {peer_id}: {e}")
                import traceback
                traceback.print_exc()

        threading.Thread(target=_handshake_task, daemon=True).start()

    def _start_handshake_watch(
        self, peer_id: str, key: str, expected_step: str
    ):
        if peer_id in self.handshake_watches:
            return
        
        is_initiator = self.engine_id < peer_id
        logger.info(
            f"Starting polling for handshake key: {key} (expecting {expected_step}, I am {'Initiator' if is_initiator else 'Responder'})"
        )

        def poll_loop():
            start_time = time.time()
            while peer_id not in self.active_p2p_links:
                if not hasattr(self, "executor"):
                    return

                try:
                    # etcd3-py uses range, returns RangeResponse
                    resp = self.etcd.range(key)
                    kvs = resp.kvs if hasattr(resp, 'kvs') else resp
                    val = None
                    if kvs:
                        val = kvs[0].value
                    
                    if val:
                        logger.info(f"Polled handshake key {key} found! Payload size: {len(val)}")
                        self._handle_handshake_step(
                            peer_id, expected_step, val
                        )
                        return
                    
                    if time.time() - start_time > 120:
                        logger.warning(f"Handshake poll timeout for {peer_id} (key: {key}). Peer might be dead.")
                        if peer_id in self.handshake_watches:
                            del self.handshake_watches[peer_id]
                        return

                    time.sleep(1.0)
                except Exception as e:
                    if not hasattr(self, "executor"):
                        return
                    logger.error(f"Error polling {key}: {e}")
                    time.sleep(1.0)

        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()
        self.handshake_watches[peer_id] = t

    def _handle_handshake_step(
        self, peer_id: str, step: str, payload_bytes: bytes
    ):
        # Safety check
        if not hasattr(self, "executor"):
            return

        # Idempotency Check:
        if peer_id in self.active_p2p_links:
            logger.debug(f"Ignoring handshake step for {peer_id} (already connected)")
            return

        logger.info(f"Processing handshake step '{step}' from {peer_id} (payload {len(payload_bytes)} bytes)")

        try:
            if step == "init":
                node_key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{peer_id}"
                
                # Get peer info for local p2p_init call
                resp = self.etcd.range(node_key)
                kvs = resp.kvs if hasattr(resp, 'kvs') else resp
                val = None
                if kvs:
                    val = kvs[0].value

                if not val:
                    logger.error(
                        f"Cannot find node info for {peer_id} to generate handshake response"
                    )
                    return

                # Initialize my endpoints and connect to remote ones
                logger.info(f"Handshake step 'init': Calling p2p_init & p2p_connect for {peer_id}")
                
                start_op = time.perf_counter()
                my_resp_bytes = self.p2p_init(val)
                self.p2p_connect(payload_bytes)
                duration = time.perf_counter() - start_op
                logger.info(f"Handshake step 'init': p2p_ops success for {peer_id} in {duration:.3f}s")

                initiator = peer_id
                terminator = self.engine_id
                path_root = (
                    f"/nanodeploy/mesh/{self.cluster_id}/handshake/{initiator}/{terminator}"
                )

                logger.info(f"Handshake step 'init': Writing /resp key for {peer_id}")
                self.etcd.put(f"{path_root}/resp", my_resp_bytes)
                self.active_p2p_links.add(peer_id)
                logger.info(f"P2P Link established with {peer_id} (as Responder)")

            elif step == "resp":
                logger.info(f"Handshake step 'resp': Calling p2p_connect for {peer_id}")
                start_op = time.perf_counter()
                self.p2p_connect(payload_bytes)
                duration = time.perf_counter() - start_op
                logger.info(f"Handshake step 'resp': p2p_connect success for {peer_id} in {duration:.3f}s")
                self.active_p2p_links.add(peer_id)
                logger.info(f"P2P Link established with {peer_id} (as Initiator)")
        except Exception as e:
            logger.error(f"Error handling handshake step '{step}' for {peer_id}: {e}")
            import traceback
            traceback.print_exc()

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

"""
PeerAgent: Control plane client for DLSlime RDMA connection management.

Refactored to use a Declarative Horizontal model (Symmetric Rendezvous):
- NanoCtrl stores "Desired Topology" in Redis (spec:topology:{agent_id})
- PeerAgent runs a TopologyReconciler loop to converge Actual State to Desired State
- QP info exchange via Redis (exchange:{sender}:{receiver})
"""

import json
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

try:
    import redis
    import requests
except ImportError as e:
    raise ImportError(
        "PeerAgent requires 'requests' and 'redis' packages. "
        "Install them with: pip install requests redis"
    ) from e

from dlslime import available_nic, RDMAContext, RDMAEndpoint, RDMAMemoryPool


def create_redis_prefix(server_url: str) -> str:
    """Create Redis key prefix from NanoCtrl server URL for data isolation."""
    # Remove protocol and sanitize (e.g., http://10.102.97.179:3000 -> nano_10_102_97_179_3000)
    sanitized = (
        server_url.replace("http://", "")
        .replace("https://", "")
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )
    return f"nano_{sanitized}"


class TopologyReconciler:
    """
    Background reconciliation loop: converges Actual State to Desired State.
    Implements Symmetric Rendezvous via Redis exchange keys.
    """

    def __init__(
        self,
        agent: "PeerAgent",
        reconcile_interval_sec: float = 1.0,
    ):
        self._agent = agent
        self._interval = reconcile_interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the reconcile loop in a background thread."""

        def run():
            while not self._stop_event.is_set():
                try:
                    self._reconcile_once()
                except Exception as e:
                    print(
                        f"TopologyReconciler {self._agent.alias}: Error in reconcile: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                self._stop_event.wait(self._interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        print(
            f"TopologyReconciler {self._agent.alias}: Started (interval={self._interval}s)"
        )

    def stop(self) -> None:
        """Stop the reconcile loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        print(f"TopologyReconciler {self._agent.alias}: Stopped")

    def _reconcile_once(self) -> None:
        """Single reconciliation pass: diff desired vs actual, act on delta."""
        # 1. Get Desired State from Redis (with scoped prefix)
        spec_key = f"{self._agent.redis_key_prefix}:spec:topology:{self._agent.alias}"
        spec_str = self._agent.redis_client.get(spec_key)
        if spec_str is None:
            return  # No desired topology, nothing to do

        try:
            spec = json.loads(spec_str)
            target_peers: List[str] = spec.get("target_peers", [])
        except (json.JSONDecodeError, TypeError):
            return

        desired = set(target_peers)
        if not desired:
            return

        # 2. Get Actual State (peers we've successfully connected to)
        actual: Set[str] = self._agent.get_connected_peers()

        # 3. Diff
        to_connect = desired - actual

        # 4. Act (Symmetric Rendezvous) - parallel connection attempts
        if not to_connect:
            return
        max_workers = min(32, len(to_connect))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._try_connect_peer, peer): peer
                for peer in to_connect
            }
            for future in as_completed(futures):
                peer = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(
                        f"TopologyReconciler {self._agent.alias}: Failed to connect to {peer}: {e}"
                    )

    def _try_connect_peer(self, peer: str) -> None:
        """
        Attempt Symmetric Rendezvous with peer.
        Idempotent: safe to call multiple times.
        Non-blocking: if peer info not in Redis, skip and retry next loop.
        """
        # A. Idempotent QP/Endpoint creation
        endpoint = self._agent.ensure_local_endpoint_created(peer)
        my_qp_info = endpoint.endpoint_info()

        # B. Publish our info to Redis (exchange:{sender}:{receiver} with scope prefix)
        exchange_key_out = (
            f"{self._agent.redis_key_prefix}:exchange:{self._agent.alias}:{peer}"
        )
        self._agent.redis_client.set(
            exchange_key_out,
            json.dumps(my_qp_info, default=str),
        )

        # C. Try to fetch peer's info (non-blocking, short timeout, with scope prefix)
        exchange_key_in = (
            f"{self._agent.redis_key_prefix}:exchange:{peer}:{self._agent.alias}"
        )
        peer_qp_info_str = self._agent.redis_client.get(exchange_key_in)

        if peer_qp_info_str is None:
            # Peer hasn't published yet; skip, retry next loop
            return

        try:
            peer_qp_info = json.loads(peer_qp_info_str)
        except json.JSONDecodeError:
            return

        # D. Handshake: modify QP to RTR/RTS (via endpoint.connect)
        if self._agent.is_peer_connected(peer):
            return  # Already connected, idempotent
        endpoint.connect(peer_qp_info)
        self._agent.mark_peer_connected(peer)
        print(f"Link Established: {self._agent.alias} <-> {peer}")


class PeerAgent:
    """PeerAgent manages RDMA connections via declarative topology reconciliation."""

    def __init__(
        self,
        alias: str,
        server_url: str = "http://127.0.0.1:3000",
        redis_address: str = "127.0.0.1:6379",
        device: Optional[str] = None,
        ib_port: int = 1,
        link_type: str = "RoCE",
        qp_num: int = 1,
        reconcile_interval_sec: float = 1.0,
    ):
        """
        Initialize a PeerAgent.

        Args:
            alias: Unique name for this agent
            server_url: URL of the control plane server (NanoCtrl)
            redis_address: Redis server address (host:port)
            device: RDMA device name (e.g., "mlx5_0"), if None, auto-select
            ib_port: InfiniBand port number
            link_type: Link type ("RoCE", "InfiniBand", etc.)
            qp_num: Number of queue pairs per endpoint
            reconcile_interval_sec: Topology reconciliation loop interval
        """
        self.alias = alias
        self.server_url = server_url
        self.redis_address = redis_address
        self.device = device
        self.ib_port = ib_port
        self.link_type = link_type
        self.qp_num = qp_num

        # Use empty prefix for now (no scoping)
        self.redis_key_prefix = ""

        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        self.address = local_ip

        # RDMA
        if self.device is None:
            devices = available_nic()
            if not devices:
                raise RuntimeError("No RDMA devices available")
            self.device = devices[0]

        self._rdma_context = RDMAContext()
        self._rdma_context.init(self.device, self.ib_port, self.link_type)
        self._memory_pool = RDMAMemoryPool(self._rdma_context)

        self._endpoints: Dict[str, RDMAEndpoint] = {}
        self._endpoints_lock = (
            threading.Lock()
        )  # Protects _endpoints for concurrent reconcile
        self._connected_peers: Set[str] = set()
        self._connected_peers_lock = threading.Lock()

        # MR cache
        self._mr_info_cache: Dict[tuple, tuple] = {}
        self._mr_info_cache_ttl_secs = 60
        self._mr_info_cache_lock = threading.Lock()

        # Redis
        redis_host, redis_port = redis_address.split(":")
        self.redis_client = redis.Redis(
            host=redis_host, port=int(redis_port), decode_responses=True
        )

        self._stop_event = threading.Event()
        self._shutdown_called = False

        # Event listener for cleanup only (legacy inbox)
        self._event_thread: Optional[threading.Thread] = None

        # Register with control plane
        self._register()

        # Start TopologyReconciler
        self._reconciler = TopologyReconciler(self, reconcile_interval_sec)
        self._reconciler.start()

        # Start cleanup event listener
        self._start_cleanup_listener()

        time.sleep(0.1)

    def _register(self) -> None:
        """Register this agent with the control plane."""
        max_retries = 5
        retry_delay = 1.0

        print(
            f"PeerAgent {self.alias}: Registering with control plane at {self.server_url}"
        )

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/start_peer_agent",
                    json={
                        "alias": self.alias,
                        "device": self.device,
                        "ib_port": self.ib_port,
                        "link_type": self.link_type,
                        "address": self.address,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                if "redis_address" in result:
                    server_redis_address = result["redis_address"]
                    if server_redis_address != self.redis_address:
                        self.redis_address = server_redis_address
                        redis_host, redis_port = self.redis_address.split(":")
                        self.redis_client = redis.Redis(
                            host=redis_host, port=int(redis_port), decode_responses=True
                        )
                print(f"PeerAgent {self.alias} registered at {self.server_url}")
                return
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    print(
                        f"PeerAgent {self.alias} registration failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"PeerAgent {self.alias} registration failed after {max_retries} attempts"
                    )
                    raise

    def _start_cleanup_listener(self) -> None:
        """Listen for cleanup events from peers (NanoCtrl pushes to inbox)."""

        def event_loop():
            inbox_key = f"{self.redis_key_prefix}:inbox:{self.alias}"
            while not self._stop_event.is_set():
                try:
                    result = self.redis_client.blpop(inbox_key, timeout=1)
                    if result:
                        _, event_str = result
                        event = json.loads(event_str)
                        if event.get("type") == "cleanup":
                            peer = event.get("peer")
                            print(f"PeerAgent {self.alias}: Cleanup from peer {peer}")
                            with self._endpoints_lock:
                                if peer in self._endpoints:
                                    with self._connected_peers_lock:
                                        self._connected_peers.discard(peer)
                                    del self._endpoints[peer]
                                    print(
                                        f"PeerAgent {self.alias}: Removed endpoint for {peer}"
                                    )
                except redis.exceptions.ConnectionError:
                    time.sleep(0.1)
                except Exception as e:
                    print(f"PeerAgent {self.alias}: Cleanup listener error: {e}")
                    time.sleep(0.1)

        self._event_thread = threading.Thread(target=event_loop, daemon=True)
        self._event_thread.start()

    def ensure_local_endpoint_created(self, peer_alias: str) -> RDMAEndpoint:
        """
        Idempotent: create endpoint for peer if not exists.
        Returns the endpoint (existing or newly created). Thread-safe.
        """
        with self._endpoints_lock:
            if peer_alias not in self._endpoints:
                endpoint = RDMAEndpoint(
                    pool=self._memory_pool,
                    num_qp=self.qp_num,
                )
                self._endpoints[peer_alias] = endpoint
            return self._endpoints[peer_alias]

    def get_connected_peers(self) -> Set[str]:
        """Return set of peer aliases we've successfully connected to."""
        with self._connected_peers_lock:
            return set(self._connected_peers)

    def is_peer_connected(self, peer_alias: str) -> bool:
        with self._connected_peers_lock:
            return peer_alias in self._connected_peers

    def mark_peer_connected(self, peer_alias: str) -> None:
        with self._connected_peers_lock:
            self._connected_peers.add(peer_alias)

    def set_desired_topology(
        self,
        target_peers: List[str],
        min_bw: Optional[str] = None,
        symmetric: bool = False,
    ) -> None:
        """
        Set desired topology via control plane. NanoCtrl saves to Redis.
        Reconciler will converge to this state.

        Args:
            target_peers: List of peer agent aliases to connect to
            min_bw: Optional min bandwidth hint (e.g. "100Gbps"), reserved
            symmetric: If True, NanoCtrl also merges this agent into each target's spec.
                Required when only one side initiates (e.g. decode -> prefill migration).
        """
        spec: Dict[str, Any] = {"target_peers": target_peers}
        if min_bw is not None:
            spec["min_bw"] = min_bw
        if symmetric:
            spec["symmetric"] = True
        response = requests.post(
            f"{self.server_url}/v1/desired_topology/{self.alias}",
            json=spec,
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "ok":
            raise RuntimeError(f"set_desired_topology failed: {result}")

    def query(self) -> Dict[str, Dict[str, Any]]:
        """Query all registered peer agents."""
        response = requests.post(
            f"{self.server_url}/query",
            json={},
            timeout=5,
        )
        response.raise_for_status()
        agents = response.json()
        return {agent["name"]: agent for agent in agents}

    def register_memory_region(
        self,
        mr_name: str,
        ptr: int,
        length: int,
    ) -> int:
        """Register local memory region and report to control plane."""
        handler = self._memory_pool.register_memory_region(ptr, length, mr_name)
        mr_info = self._memory_pool.mr_info()[mr_name]
        request_data = {
            "agent_name": self.alias,
            "mr_name": mr_name,
            "addr": int(mr_info["addr"]),
            "length": int(mr_info["length"]),
            "rkey": int(mr_info["rkey"]),
            "lkey": 0,
        }
        response = requests.post(
            f"{self.server_url}/register_mr",
            json=request_data,
            timeout=5,
        )
        response.raise_for_status()
        return handler

    def get_mr_info(self, peer_alias: str, mr_name: str) -> Optional[Dict[str, Any]]:
        """Get remote memory region info (cached)."""
        cache_key = (peer_alias, mr_name)
        with self._mr_info_cache_lock:
            if cache_key in self._mr_info_cache:
                cached_at, cached = self._mr_info_cache[cache_key]
                if time.time() - cached_at < self._mr_info_cache_ttl_secs:
                    return cached

        response = requests.post(
            f"{self.server_url}/get_mr_info",
            json={
                "src": self.alias,
                "dst": peer_alias,
                "mr_name": mr_name,
            },
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        mr_info = result.get("mr_info")

        with self._mr_info_cache_lock:
            self._mr_info_cache[cache_key] = (time.time(), mr_info)

        return mr_info

    def register_remote_memory_region(
        self,
        peer_alias: str,
        mr_name: str,
        mr_info: Dict[str, Any],
    ) -> int:
        """Register remote memory region."""
        with self._endpoints_lock:
            if peer_alias not in self._endpoints:
                raise RuntimeError(f"Endpoint for {peer_alias} not initialized")
            endpoint = self._endpoints[peer_alias]
        return endpoint.register_remote_memory_region(mr_name, mr_info)

    def get_endpoint(self, peer_alias: str) -> RDMAEndpoint:
        """Get RDMA endpoint for peer (must be connected)."""
        with self._endpoints_lock:
            if peer_alias not in self._endpoints:
                raise RuntimeError(
                    f"Endpoint for {peer_alias} not found. "
                    "Ensure set_desired_topology([...]) includes this peer and wait for reconciliation."
                )
            return self._endpoints[peer_alias]

    def wait_for_peers(self, peers: List[str], timeout_sec: float = 60.0) -> None:
        """
        Block until all specified peers are connected.
        Useful for tests / sync points after set_desired_topology.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            connected = self.get_connected_peers()
            missing = [p for p in peers if p not in connected]
            if not missing:
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"Timeout waiting for peers {peers}. "
            f"Connected: {self.get_connected_peers()}"
        )

    def shutdown(self) -> None:
        """Shutdown and clean up."""
        if self._shutdown_called:
            return
        self._shutdown_called = True

        print(f"PeerAgent {self.alias}: Shutting down...")

        self._stop_event.set()
        self._reconciler.stop()

        if self._event_thread:
            self._event_thread.join(timeout=1)

        with self._endpoints_lock:
            self._endpoints.clear()
        with self._connected_peers_lock:
            self._connected_peers.clear()

        try:
            response = requests.post(
                f"{self.server_url}/cleanup",
                json={"agent_name": self.alias},
                timeout=5,
            )
            response.raise_for_status()
            print(f"PeerAgent {self.alias}: Cleanup OK")
        except Exception as e:
            print(f"PeerAgent {self.alias}: Cleanup API warning: {e}")

        print(f"PeerAgent {self.alias}: Shutdown complete")

    def __enter__(self) -> "PeerAgent":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.shutdown()
        return False

    def __del__(self) -> None:
        if not self._shutdown_called:
            try:
                self.shutdown()
            except Exception as e:
                print(f"PeerAgent {self.alias}: Warning in __del__: {e}")


def start_peer_agent(
    alias: str,
    server_url: str = "http://127.0.0.1:3000",
    address: Optional[str] = None,
    device: Optional[str] = None,
    ib_port: int = 1,
    link_type: str = "RoCE",
    qp_num: int = 1,
) -> PeerAgent:
    """
    Start a peer agent (convenience function).

    Use set_desired_topology(target_peers=[...]) to declare which peers to connect to.
    """
    redis_address = address if address is not None else "127.0.0.1:6379"
    return PeerAgent(
        alias=alias,
        server_url=server_url,
        redis_address=redis_address,
        device=device,
        ib_port=ib_port,
        link_type=link_type,
        qp_num=qp_num,
    )

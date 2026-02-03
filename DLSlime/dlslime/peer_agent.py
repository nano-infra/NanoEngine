"""
PeerAgent: Control plane client for DLSlime RDMA connection management.
"""

import json
import threading
import time
from typing import Any, Dict, Optional

try:
    import redis
    import requests
except ImportError as e:
    raise ImportError(
        "PeerAgent requires 'requests' and 'redis' packages. "
        "Install them with: pip install requests redis"
    ) from e

from dlslime import available_nic, RDMAContext, RDMAEndpoint, RDMAMemoryPool


class PeerAgent:
    """PeerAgent manages RDMA connections through a centralized control plane."""

    def __init__(
        self,
        alias: str,
        server_url: str = "http://127.0.0.1:3000",
        redis_address: str = "127.0.0.1:6379",
        device: Optional[str] = None,
        ib_port: int = 1,
        link_type: str = "RoCE",
        qp_num: int = 1,
    ):
        """
        Initialize a PeerAgent.

        Args:
            alias: Unique name for this agent
            server_url: URL of the control plane server
            redis_address: Redis server address (host:port)
            device: RDMA device name (e.g., "mlx5_0"), if None, auto-select
            ib_port: InfiniBand port number
            link_type: Link type ("RoCE", "InfiniBand", etc.)
            qp_num: Number of queue pairs
        """
        self.alias = alias
        self.server_url = server_url
        self.redis_address = redis_address
        self.device = device
        self.ib_port = ib_port
        self.link_type = link_type
        self.qp_num = qp_num

        # Get local IP address (simplified)
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        self.address = local_ip

        # Initialize RDMA device
        if self.device is None:
            devices = available_nic()
            if not devices:
                raise RuntimeError("No RDMA devices available")
            self.device = devices[0]

        # Pre-allocate shared MemoryPool (see docs/control_plane/share_memory.md)
        # All endpoints share this pool for local MR registration
        self._rdma_context = RDMAContext()
        self._rdma_context.init(self.device, self.ib_port, self.link_type)
        self._memory_pool = RDMAMemoryPool(self._rdma_context)

        self._endpoints: Dict[str, RDMAEndpoint] = {}  # peer_alias -> endpoint

        # get_mr_info cache: (peer_alias, mr_name) -> (cached_at, mr_info), TTL 60s
        self._mr_info_cache: Dict[tuple, tuple] = {}
        self._mr_info_cache_ttl_secs = 60
        self._mr_info_cache_lock = threading.Lock()

        # Redis connection for event listening
        redis_host, redis_port = redis_address.split(":")
        self.redis_client = redis.Redis(
            host=redis_host, port=int(redis_port), decode_responses=True
        )

        # Start event listener thread
        self._event_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._shutdown_called = False  # Track if shutdown has been called

        # Register with control plane
        self._register()

        # Start event listener
        self._start_event_listener()

        # Give event listener thread time to start
        time.sleep(0.1)

    def _register(self):
        """Register this agent with the control plane."""
        max_retries = 5
        retry_delay = 1.0  # seconds

        print(
            f"PeerAgent {self.alias}: Attempting to register with control plane at {self.server_url}"
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
                    timeout=10,  # Increased timeout
                )
                response.raise_for_status()
                result = response.json()
                # Update redis_address from server response if available
                if "redis_address" in result:
                    server_redis_address = result["redis_address"]
                    if server_redis_address != self.redis_address:
                        # Update redis_address and reconnect if different
                        self.redis_address = server_redis_address
                        redis_host, redis_port = self.redis_address.split(":")
                        self.redis_client = redis.Redis(
                            host=redis_host, port=int(redis_port), decode_responses=True
                        )
                print(
                    f"PeerAgent {self.alias} registered with control plane at {self.server_url}"
                )
                return  # Success, exit retry loop
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"PeerAgent {self.alias} registration failed (attempt {attempt + 1}/{max_retries}): ConnectionError to {self.server_url}: {e}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    print(
                        f"PeerAgent {self.alias} registration failed after {max_retries} attempts: Cannot connect to {self.server_url}: {e}"
                    )
                    raise
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"PeerAgent {self.alias} registration failed (attempt {attempt + 1}/{max_retries}): Timeout connecting to {self.server_url}: {e}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    print(
                        f"PeerAgent {self.alias} registration failed after {max_retries} attempts: Timeout connecting to {self.server_url}: {e}"
                    )
                    raise
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"PeerAgent {self.alias} registration failed (attempt {attempt + 1}/{max_retries}): HTTP error from {self.server_url}: {e} (status: {response.status_code if 'response' in locals() else 'N/A'}). Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    print(
                        f"PeerAgent {self.alias} registration failed after {max_retries} attempts: HTTP error from {self.server_url}: {e}"
                    )
                    raise

    def _start_event_listener(self):
        """Start listening for events from Redis mailbox."""

        def event_loop():
            inbox_key = f"inbox:{self.alias}"
            print(
                f"PeerAgent {self.alias}: Event listener started, listening to {inbox_key}"
            )
            event_count = 0
            while not self._stop_event.is_set():
                try:
                    # Blocking pop from mailbox
                    result = self.redis_client.blpop(inbox_key, timeout=1)
                    if result:
                        _, event_str = result
                        event = json.loads(event_str)
                        event_count += 1
                        print(
                            f"PeerAgent {self.alias}: Received event #{event_count} from {inbox_key}: {event.get('type', 'unknown')} (src={event.get('src', 'N/A')}, dst={event.get('dst', 'N/A')})"
                        )
                        self._handle_event(event)
                    # Log every 10 seconds to show listener is alive
                    elif event_count == 0:
                        # Only log if no events received yet (to avoid spam)
                        pass
                except redis.exceptions.ConnectionError as e:
                    print(
                        f"PeerAgent {self.alias}: Redis connection error in event loop: {e}"
                    )
                    time.sleep(0.1)
                except Exception as e:
                    print(f"PeerAgent {self.alias}: Error in event loop: {e}")
                    import traceback

                    traceback.print_exc()
                    time.sleep(0.1)
            print(
                f"PeerAgent {self.alias}: Event listener stopped (processed {event_count} events)"
            )

        self._event_thread = threading.Thread(target=event_loop, daemon=True)
        self._event_thread.start()

    def _handle_event(self, event: Dict[str, Any]):
        """Handle an event from the control plane."""
        event_type = event.get("type")

        if event_type == "init":
            self._handle_init_event(event)
        elif event_type == "connect":
            self._handle_connect_event(event)
        elif event_type == "cleanup":
            self._handle_cleanup_event(event)
        else:
            print(f"Unknown event type: {event_type}")

    def _handle_init_event(self, event: Dict[str, Any]):
        """Handle init event: create RDMA endpoint."""
        src = event["src"]
        dst = event["dst"]
        qp_num = event.get("qp_num", self.qp_num)

        # Determine which peer we should create endpoint for
        # If we are src, create endpoint for dst; if we are dst, create endpoint for src
        if src == self.alias:
            peer_alias = dst
        elif dst == self.alias:
            peer_alias = src
        else:
            return  # Not for us

        print(
            f"PeerAgent {self.alias}: Received init event, creating endpoint for {peer_alias}"
        )

        # Create endpoint if not exists (use shared MemoryPool)
        if peer_alias not in self._endpoints:
            endpoint = RDMAEndpoint(
                pool=self._memory_pool,
                num_qp=qp_num,
            )
            self._endpoints[peer_alias] = endpoint

        # Send ACK with endpoint info
        endpoint = self._endpoints[peer_alias]
        endpoint_info = endpoint.endpoint_info()

        response = requests.post(
            f"{self.server_url}/ack_init",
            json={
                "src": self.alias,
                "dst": peer_alias,
                "endpoint_info": endpoint_info,
            },
            timeout=5,
        )
        response.raise_for_status()

    def _handle_connect_event(self, event: Dict[str, Any]):
        """Handle connect event: establish RDMA connection."""
        src = event["src"]
        dst = event["dst"]

        # Determine which peer we should connect to
        if src == self.alias:
            peer_alias = dst
        elif dst == self.alias:
            peer_alias = src
        else:
            return  # Not for us

        # Get remote endpoint info from server
        # We need the endpoint info of the remote peer
        response = requests.post(
            f"{self.server_url}/get_endpoint_info",
            json={"src": self.alias, "dst": peer_alias},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            remote_info = data.get("endpoint_info")
        else:
            print(
                f"PeerAgent {self.alias}: Cannot get remote endpoint info for {peer_alias}"
            )
            return

        if remote_info is None:
            print(
                f"PeerAgent {self.alias}: Remote endpoint info is None for {peer_alias}"
            )
            return

        # Connect
        if peer_alias in self._endpoints:
            endpoint = self._endpoints[peer_alias]
            endpoint.connect(remote_info)
            print(f"PeerAgent {self.alias}: Connected to {peer_alias}")
        else:
            print(f"PeerAgent {self.alias}: Endpoint for {peer_alias} not found")
            return

        # Send ACK
        response = requests.post(
            f"{self.server_url}/ack_connect",
            json={
                "src": self.alias,
                "dst": peer_alias,
            },
            timeout=5,
        )
        response.raise_for_status()

    def _handle_cleanup_event(self, event: Dict[str, Any]):
        """Handle cleanup event from peer agent."""
        peer = event.get("peer")
        print(f"PeerAgent {self.alias}: Received cleanup event from {peer}")

        # Remove endpoint for this peer
        if peer in self._endpoints:
            del self._endpoints[peer]
            print(f"PeerAgent {self.alias}: Removed endpoint for {peer}")

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

    def init(self, peer_alias: str, qp_num: Optional[int] = None) -> None:
        """
        Initialize connection with a peer.

        Level-triggered (水平触发): server publishes init event to BOTH sides.
        Endpoint is created in _handle_init_event when event is received,
        not here.

        Args:
            peer_alias: Alias of the peer agent
            qp_num: Number of queue pairs (defaults to self.qp_num)
        """
        if qp_num is None:
            qp_num = self.qp_num

        response = requests.post(
            f"{self.server_url}/init",
            json={
                "src": self.alias,
                "dst": peer_alias,
                "qp_num": qp_num,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "ok":
            raise RuntimeError(f"Init failed: {result.get('message')}")

    def connect(self, peer_alias: str) -> None:
        """
        Connect to a peer (after init).

        Args:
            peer_alias: Alias of the peer agent
        """
        response = requests.post(
            f"{self.server_url}/connect",
            json={
                "src": self.alias,
                "dst": peer_alias,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "ok":
            raise RuntimeError(f"Connect failed: {result.get('message')}")

    def register_memory_region(
        self,
        mr_name: str,
        ptr: int,
        length: int,
    ) -> int:
        """
        Register a local memory region and report to the control plane.

        Uses the pre-allocated shared MemoryPool. Registers our own MR and
        reports it to the control plane so remote peers can connect via
        get_mr_info().

        Args:
            mr_name: Name of the memory region
            ptr: Pointer to memory
            length: Length in bytes

        Returns:
            Memory region handler/key
        """
        # Register on shared MemoryPool (see docs/control_plane/share_memory.md)
        handler = self._memory_pool.register_memory_region(ptr, length, mr_name)

        # Get MR info from pool
        mr_info = self._memory_pool.mr_info()[mr_name]

        # Register with control plane
        # Note: mr_info contains "handle", "addr", "rkey", "length" (no lkey)
        # lkey is local-only and not needed for control plane, but we need to send it
        request_data = {
            "agent_name": self.alias,
            "mr_name": mr_name,
            "addr": int(mr_info["addr"]),  # Ensure it's an integer
            "length": int(mr_info["length"]),  # Ensure it's an integer
            "rkey": int(mr_info["rkey"]),  # Ensure it's an integer
            "lkey": 0,  # lkey is not in mr_info, use 0 as default
        }
        print(f"Registering MR with data: {request_data}")
        response = requests.post(
            f"{self.server_url}/register_mr",
            json=request_data,
            timeout=5,
        )
        if response.status_code != 200:
            print(f"Error registering MR: {response.status_code} - {response.text}")
        response.raise_for_status()

        return handler

    def get_mr_info(self, peer_alias: str, mr_name: str) -> Optional[Dict[str, Any]]:
        """
        Get remote memory region info.
        Cached for 60 seconds to reduce HTTP calls to control plane.

        Args:
            peer_alias: Alias of the peer
            mr_name: Name of the memory region

        Returns:
            MR info dict or None if not found
        """
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
        """
        Register a remote memory region.

        Args:
            peer_alias: Alias of the peer
            mr_name: Name of the memory region
            mr_info: MR info dict from get_mr_info

        Returns:
            Remote MR handler
        """
        if peer_alias not in self._endpoints:
            raise RuntimeError(f"Endpoint for {peer_alias} not initialized")

        endpoint = self._endpoints[peer_alias]
        return endpoint.register_remote_memory_region(mr_name, mr_info)

    def get_endpoint(self, peer_alias: str) -> RDMAEndpoint:
        """Get the RDMA endpoint for a peer."""
        if peer_alias not in self._endpoints:
            raise RuntimeError(f"Endpoint for {peer_alias} not initialized")
        return self._endpoints[peer_alias]

    def shutdown(self):
        """Shutdown the peer agent and clean up Redis data via control plane."""
        # Prevent multiple shutdown calls
        if self._shutdown_called:
            return
        self._shutdown_called = True

        print(f"PeerAgent {self.alias}: Shutting down and cleaning up connections...")

        # Stop event listener
        self._stop_event.set()
        if self._event_thread:
            self._event_thread.join(timeout=1)

        # Clean up all endpoints (disconnect all connections)
        if self._endpoints:
            print(
                f"PeerAgent {self.alias}: Cleaning up {len(self._endpoints)} endpoint(s)..."
            )
            for peer_alias in list(self._endpoints.keys()):
                try:
                    # Note: RDMAEndpoint doesn't expose explicit disconnect() in Python bindings
                    # Clearing the endpoint will trigger cleanup
                    print(f"PeerAgent {self.alias}: Removing endpoint for {peer_alias}")
                except Exception as e:
                    print(
                        f"PeerAgent {self.alias}: Warning: Error cleaning up endpoint for {peer_alias}: {e}"
                    )
            self._endpoints.clear()

        # Call service-side cleanup API (对等清理)
        try:
            response = requests.post(
                f"{self.server_url}/cleanup",
                json={
                    "agent_name": self.alias,
                },
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()
            print(f"PeerAgent {self.alias}: Cleanup response: {result.get('message')}")
        except Exception as e:
            print(f"PeerAgent {self.alias}: Warning: Failed to call cleanup API: {e}")

        print(f"PeerAgent {self.alias}: Shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically shutdown on exit."""
        self.shutdown()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor - automatically shutdown when object is destroyed."""
        # Only shutdown if not already called (safety check)
        if not self._shutdown_called:
            try:
                self.shutdown()
            except Exception as e:
                # Suppress exceptions in destructor to avoid issues during garbage collection
                print(f"PeerAgent {self.alias}: Warning: Error in __del__: {e}")


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

    Args:
        alias: Unique name for this agent
        server_url: URL of the control plane server
        address: Redis address (host:port). If None, will be fetched from server.
        device: RDMA device name
        ib_port: InfiniBand port number
        link_type: Link type
        qp_num: Number of queue pairs

    Returns:
        PeerAgent instance
    """
    # If address is not provided, use a default and let _register() update it from server response
    # We need a valid address format for PeerAgent initialization
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

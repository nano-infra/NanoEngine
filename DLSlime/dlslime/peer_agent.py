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

        # Redis connection for event listening
        redis_host, redis_port = redis_address.split(":")
        self.redis_client = redis.Redis(
            host=redis_host, port=int(redis_port), decode_responses=True
        )

        # Start event listener thread
        self._event_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Register with control plane
        self._register()

        # Start event listener
        self._start_event_listener()

        # Give event listener thread time to start
        time.sleep(0.1)

    def _register(self):
        """Register this agent with the control plane."""
        response = requests.post(
            f"{self.server_url}/start_peer_agent",
            json={
                "alias": self.alias,
                "device": self.device,
                "ib_port": self.ib_port,
                "link_type": self.link_type,
                "address": self.address,
            },
            timeout=5,
        )
        response.raise_for_status()
        print(f"PeerAgent {self.alias} registered with control plane")

    def _start_event_listener(self):
        """Start listening for events from Redis mailbox."""

        def event_loop():
            while not self._stop_event.is_set():
                try:
                    # Blocking pop from mailbox
                    result = self.redis_client.blpop(f"inbox:{self.alias}", timeout=1)
                    if result:
                        _, event_str = result
                        event = json.loads(event_str)
                        self._handle_event(event)
                except redis.exceptions.ConnectionError:
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error in event loop: {e}")
                    time.sleep(0.1)

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
        print(f"PeerAgent {self.alias}: Sent init ACK for {peer_alias}")

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

        print(
            f"PeerAgent {self.alias}: Received connect event, connecting to {peer_alias}"
        )

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

        Args:
            peer_alias: Alias of the peer
            mr_name: Name of the memory region

        Returns:
            MR info dict or None if not found
        """
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
        return result.get("mr_info")

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
        self._stop_event.set()
        if self._event_thread:
            self._event_thread.join(timeout=1)

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
            print(f"Cleanup response: {result.get('message')}")
        except Exception as e:
            print(f"Warning: Failed to call cleanup API: {e}")

        # Note: RDMAEndpoint doesn't expose shutdown() in Python bindings
        # Python's garbage collector will handle cleanup
        self._endpoints.clear()


def start_peer_agent(
    alias: str,
    server_url: str = "http://127.0.0.1:3000",
    address: str = "127.0.0.1:6379",
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
        address: Redis address (host:port)
        device: RDMA device name
        ib_port: InfiniBand port number
        link_type: Link type
        qp_num: Number of queue pairs

    Returns:
        PeerAgent instance
    """
    return PeerAgent(
        alias=alias,
        server_url=server_url,
        redis_address=address,
        device=device,
        ib_port=ib_port,
        link_type=link_type,
        qp_num=qp_num,
    )

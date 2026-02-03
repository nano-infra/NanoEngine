"""
Example: Multiple Peer Agents using Control Plane for centralized connection setup.

8 agents, mesh network (each agent connects to all others). Measures timing.
"""

import contextlib
import threading
import time

import torch
from dlslime import start_peer_agent


# Helper: Time measurement context manager
@contextlib.contextmanager
def time_measure(operation_name):
    """Context manager to measure and print execution time."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[TIME] {operation_name}: {elapsed:.3f}s")


# Helper: Run function in parallel for all agents
def run_parallel(agents, func):
    """Run a function in parallel for all agents."""
    threads = []
    for alias, agent in agents.items():
        t = threading.Thread(target=func, args=(alias, agent))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# Start multiple peer agents
num_agents = 8
verbose = False  # Set True to print each init/connect/read

print("=" * 60)
print(f"Starting {num_agents} peer agents (mesh)...")
print("=" * 60)

# Use ExitStack to manage multiple context managers (auto-cleanup)
with contextlib.ExitStack() as stack:
    agents = {}
    with time_measure("start"):
        for i in range(num_agents):
            alias = f"agent_{i}"
            agent = start_peer_agent(
                alias=alias,
                server_url="http://127.0.0.1:3000",
                device=None,  # Auto-select
                ib_port=1,
                link_type="RoCE",
                qp_num=1,
            )
            agents[alias] = stack.enter_context(agent)  # Auto-cleanup on exit
            if verbose:
                print(f"Started {alias}")

    # Query available peers
    print("\n" + "=" * 60)
    print("Available peers:")
    print("=" * 60)
    for alias, agent in agents.items():
        peers = agent.query()
        if verbose:
            print(f"{alias} sees: {list(peers.keys())}")

    # Initialize connections: create a mesh network (each agent connects to all others)
    print("\n" + "=" * 60)
    print("Initializing connections (mesh network)...")
    print("=" * 60)

    def init_connections(agent_alias, agent):
        """Initialize connections from one agent to all others."""
        for peer_alias in agents.keys():
            if peer_alias != agent_alias:
                try:
                    agent.init(peer_alias, qp_num=1)
                    if verbose:
                        print(f"  {agent_alias} -> {peer_alias}: init OK")
                except Exception as e:
                    print(f"  {agent_alias} -> {peer_alias}: init FAILED - {e}")

    with time_measure("init"):
        run_parallel(agents, init_connections)

    # Brief wait for init events to be processed
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("Connecting...")
    print("=" * 60)

    def connect_to_peers(agent_alias, agent):
        """Connect from one agent to all others."""
        for peer_alias in agents.keys():
            if peer_alias != agent_alias:
                try:
                    agent.connect(peer_alias)
                    if verbose:
                        print(f"  {agent_alias} -> {peer_alias}: connect OK")
                except Exception as e:
                    print(f"  {agent_alias} -> {peer_alias}: connect FAILED - {e}")

    with time_measure("connect"):
        run_parallel(agents, connect_to_peers)

    # Brief wait for connect events
    time.sleep(0.3)

    # Register memory regions for each agent
    print("\n" + "=" * 60)
    print("Registering memory regions...")
    print("=" * 60)

    # Source tensors: each agent's data (never overwritten by reads)
    source_tensors = {}
    source_handlers = {}
    # Receive buffers: per (reader, peer) to avoid overwrite when one agent reads from multiple peers
    recv_buffers = {}  # (reader_alias, peer_alias) -> tensor
    recv_handlers = {}  # (reader_alias, peer_alias) -> handler

    with time_measure("register"):
        for alias, agent in agents.items():
            agent_id = int(alias.split("_")[1])
            tensor = torch.full([32], agent_id, device="cpu", dtype=torch.uint8)
            source_tensors[alias] = tensor

            try:
                handler = agent.register_memory_region(
                    "data",
                    tensor.data_ptr() + int(tensor.storage_offset()),
                    tensor.numel() * tensor.itemsize,
                )
                source_handlers[alias] = handler
                if verbose:
                    print(f"  {alias} registered MR 'data' (self)")
            except Exception as e:
                print(f"  {alias}: register MR FAILED - {e}")

        # Each agent needs a recv buffer per peer (to avoid overwriting when reading from multiple peers)
        for reader_alias, agent in agents.items():
            for peer_alias in agents.keys():
                if peer_alias != reader_alias:
                    recv_tensor = torch.zeros([32], device="cpu", dtype=torch.uint8)
                    recv_buffers[(reader_alias, peer_alias)] = recv_tensor
                    recv_name = f"recv_{reader_alias}_from_{peer_alias}"
                    handler = agent.register_memory_region(
                        recv_name,
                        recv_tensor.data_ptr() + int(recv_tensor.storage_offset()),
                        recv_tensor.numel() * recv_tensor.itemsize,
                    )
                    recv_handlers[(reader_alias, peer_alias)] = handler

    # Perform RDMA operations: each agent reads from all others
    print("\n" + "=" * 60)
    print("Performing RDMA reads...")
    print("=" * 60)

    def perform_reads(agent_alias, agent):
        """Agent reads from all other agents."""
        for peer_alias in agents.keys():
            if peer_alias != agent_alias:
                try:
                    remote_mr_info = agent.get_mr_info(peer_alias, "data")
                    if remote_mr_info is None:
                        print(f"  {agent_alias} -> {peer_alias}: MR info not found")
                        continue

                    remote_handler = agent.register_remote_memory_region(
                        peer_alias,
                        "data",
                        remote_mr_info,
                    )

                    local_handler = recv_handlers.get((agent_alias, peer_alias))
                    if local_handler is None:
                        print(
                            f"  {agent_alias} -> {peer_alias}: local handler not found"
                        )
                        continue

                    endpoint = agent.get_endpoint(peer_alias)
                    slot = endpoint.read(
                        [(local_handler, remote_handler, 0, 0, 8)], None
                    )
                    slot.wait()

                    expected_value = int(peer_alias.split("_")[1])
                    read_value = recv_buffers[(agent_alias, peer_alias)][0].item()

                    if verbose:
                        if read_value == expected_value:
                            print(
                                f"  {agent_alias} <- {peer_alias}: read OK (value={read_value})"
                            )
                        else:
                            print(
                                f"  {agent_alias} <- {peer_alias}: read MISMATCH (got={read_value}, expected={expected_value})"
                            )
                    elif read_value != expected_value:
                        print(
                            f"  {agent_alias} <- {peer_alias}: MISMATCH (got={read_value}, expected={expected_value})"
                        )

                except Exception as e:
                    print(f"  {agent_alias} -> {peer_alias}: read FAILED - {e}")

    with time_measure("read (56 ops)"):
        run_parallel(agents, perform_reads)

    # Cleanup is automatic via ExitStack context manager
    print("\n" + "=" * 60)
    print("Multi-agent control plane example completed!")
    print("=" * 60)
    print("(Cleanup will happen automatically when exiting context)")

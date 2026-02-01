"""
Example: P2P RDMA RC Read using Control Plane for centralized connection setup.

This example demonstrates how to use the control plane to establish RDMA connections
without manual endpoint_info exchange.
"""

import torch
from dlslime import start_peer_agent

# Start two peer agents
# In a real distributed scenario, these would run on different machines
initiator_agent = start_peer_agent(
    alias="initiator",
    server_url="http://127.0.0.1:3000",
    address="127.0.0.1:6379",
    device=None,  # Auto-select
    ib_port=1,
    link_type="RoCE",
    qp_num=1,
)

target_agent = start_peer_agent(
    alias="target",
    server_url="http://127.0.0.1:3000",
    address="127.0.0.1:6379",
    device=None,  # Auto-select (will use same device if only one available)
    ib_port=1,
    link_type="RoCE",
    qp_num=1,
)

# Query available peers
print("Available peers:", initiator_agent.query())

# Initialize connection (both sides)
print("Initializing connection...")
# Both agents need to call init, but we can do it in parallel or with a small delay
import threading


def init_async(agent, peer):
    agent.init(peer, qp_num=1)


t1 = threading.Thread(target=init_async, args=(initiator_agent, "target"))
t2 = threading.Thread(target=init_async, args=(target_agent, "initiator"))
t1.start()
t2.start()
t1.join()
t2.join()

# Connect (both sides)
print("Connecting...")


def connect_async(agent, peer):
    agent.connect(peer)


t1 = threading.Thread(target=connect_async, args=(initiator_agent, "target"))
t2 = threading.Thread(target=connect_async, args=(target_agent, "initiator"))
t1.start()
t2.start()
t1.join()
t2.join()

# Get endpoints
initiator = initiator_agent.get_endpoint("target")
target = target_agent.get_endpoint("initiator")

# Register local memory regions (each agent registers its own MR)
local_tensor = torch.zeros([16], device="cpu", dtype=torch.uint8)
handler = initiator_agent.register_memory_region(
    "kv",
    local_tensor.data_ptr() + int(local_tensor.storage_offset()),
    local_tensor.numel() * local_tensor.itemsize,
)

remote_tensor = torch.ones([16], device="cpu", dtype=torch.uint8)
target_agent.register_memory_region(
    "kv",
    remote_tensor.data_ptr() + int(remote_tensor.storage_offset()),
    remote_tensor.numel() * remote_tensor.itemsize,
)

# Get remote MR info through control plane
print("Getting remote MR info...")
remote_mr_info = initiator_agent.get_mr_info("target", "kv")
assert remote_mr_info is not None, "Failed to get remote MR info"

# Register remote memory region
hremote_on_initiator = initiator_agent.register_remote_memory_region(
    "target",
    "kv",
    remote_mr_info,
)

# Perform RDMA read
print("Performing RDMA read...")
slot = initiator.read([(handler, hremote_on_initiator, 0, 8, 8)], None)
slot.wait()

# Verify results
assert torch.all(local_tensor[:8] == 0)
assert torch.all(local_tensor[8:] == 1)
print("Local tensor after RDMA read:", local_tensor)

# Cleanup
initiator_agent.shutdown()
target_agent.shutdown()
print("Control plane example completed successfully")

# Spoke V2 User Guide: Distributed Resource & Gang Scheduling

**Version**: 0.2.0 (Spoke V2)
**Date**: 2026-01-10

## 1. Overview

Spoke V2 introduces a powerful **Resource-Aware Gang Scheduling** system designed for high-performance AI inference and training. It solves the critical problem of bootstrapping distributed process groups (e.g., NCCL/PyTorch DDP) by decoupling **Resource Allocation** from **Actor Launch**.

### Key Features
- **Resource Awareness**: Hub tracks GPU usage per node.
- **Gang Scheduling**: Allocates multiple actors atomically on the same node (or across nodes) to ensure locality and prevent resource fragmentation.
- **Two-Phase Launch**: 
    1. `gangAllocate()`: Reserve resources and get a topology map.
    2. `launchActor()`: Start processes with precise placement and configuration.
- **GPU Isolation**: Automatically sets `CUDA_VISIBLE_DEVICES` for each actor.
- **Self-Contained**: DLSlime (RDMA) is now vendored, simplifying deployment.

## 2. Quick Start

### 2.1 Start Infrastructure

1. **Start Hub**:
   ```bash
   ./bin/spoke_hub [port]
   ```

2. **Start Agent (Daemon)**:
   Report available resources (e.g., 8 GPUs) to the Hub.
   ```bash
   ./bin/smoke_v7_agent [my_port] [hub_ip] [hub_port] --gpus 8
   ```

### 2.2 Client Code (Engine/Controller)

Use the new `gangAllocate` API to reserve a Gang.

```cpp
#include "spoke/csrc/client.h"

int main() {
    spoke::Client client("127.0.0.1", 8888, true); // Connect to Hub

    // 1. Define Requirement: 1 Node, 8 Actors, 1 GPU each
    spoke::ResourceSpec res;
    res.num_gpus = 1;
    
    // 2. Allocation Phase (Blocking)
    // Request: "Give me 1 node with 8 slots"
    auto fut = client.gangAllocate(1, 8, res);
    auto resp = fut.get();
    
    std::cout << "Allocation Ticket: " << resp.ticket_id << std::endl;

    // 3. Configuration Phase (Local)
    // Now you know exactly where each rank will run.
    // You can generate config files, master addresses, etc. here.

    // 4. Launch Phase
    for (int i = 0; i < resp.num_members; ++i) {
        // Launch rank 'i' using the reserved ticket
        client.launchActor(resp.ticket_id, i, "ModelRunner", "worker_" + std::to_string(i), "--rank " + std::to_string(i));
    }
    
    return 0;
}
```

## 3. Architecture Deep Dive

### 3.1 Allocation Strategy
When `gangAllocate` is called:
1. Hub scans registered nodes for availability (Free GPUs = Total - Used).
2. Hub uses a greedy packing algorithm (currently) to find nodes that fit `actors_per_node`.
3. Hub **commits** the usage immediately and generates a `ticket_id`.
4. Hub returns the `AllocatedSlot` list (Ticket ID + Topology).

### 3.2 GPU Isolation mechanism
When `launchActor` is called:
1. Hub looks up the `AllocatedSlot` for the given `ticket_id` and `rank`.
2. Hub retrieves the assigned physical `gpu_id` (e.g., GPU 3).
3. Hub sends a `kNetLaunch` command to the specific Agent on that node.
4. Agent performs `fork()`.
5. Inside the child process, Agent executes:
   ```cpp
   setenv("CUDA_VISIBLE_DEVICES", "3", 1); // Isolation happens here
   execv(exe_path, args);
   ```
6. The Actor process sees only GPU 0 (which maps to physical GPU 3).

## 4. Build & Integration

### 4.1 Vendored DLSlime
Spoke now includes DLSlime in `third_party/DLSlime`. You do not need to install DLSlime system-wide.
- **CMake**: `target_link_libraries(your_target Spoke::core)` automatically links the internal RDMA transport.

### 4.2 Creating Custom Agents
Inherit from `spoke::Actor` as usual. The resource management is transparent to the Actor class.

```cpp
class MyActor : public spoke::Actor {
    void run() override {
        // Verify isolation
        std::cout << "My Visible Devices: " << getenv("CUDA_VISIBLE_DEVICES") << std::endl;
    }
};
SPOKE_REGISTER_ACTOR("MyActor", MyActor);
```

## 5. API Reference

### `Client::gangAllocate`
```cpp
std::future<AllocateResp> gangAllocate(
    uint32_t num_nodes, 
    uint32_t actors_per_node, 
    const ResourceSpec& res_per_actor, 
    bool strict_pack = true
);
```
- **strict_pack**: If true, ensures `actors_per_node` are on the SAME physical node.

### `Client::launchActor`
```cpp
std::future<bool> launchActor(
    const std::string& ticket_id, 
    uint32_t global_rank, 
    const std::string& type, 
    const std::string& id, 
    const std::string& args_serialized
);
```
- **global_rank**: The index [0..N-1] within the Gang allocation.

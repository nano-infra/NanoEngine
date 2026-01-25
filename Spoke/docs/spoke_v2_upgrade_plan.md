# Spoke V2 Upgrade Plan: Resource Management & Gang Scheduling

**Status**: In Progress (Core features completed)
**Target**: Support NanoDeploy Distributed Inference (8-GPU/Node)

## 1. Motivation

To support high-performance distributed inference (e.g., Qwen3 70B on 8 GPUs), Spoke needs to evolve from a simple "Spawn anywhere" system to a resource-aware cluster manager.
We need to ensure that:

1. **Locality**: All 8 ranks of a model instance run on the same physical node (or strictly defined topology) to maximize NVLink/PCIe bandwidth.
2. **Resource Isolation**: The Hub must track used vs. free GPUs/NICs to prevent oversubscription.
3. **Robustness**: Better lifecycle management for long-running Actors.

## 2. Resource Management (Hub & Agent)

### 2.1 Resource Definitions

Define a standard resource descriptor structure.

```cpp
struct ResourceSpec {
    int num_gpus = 0;       // e.g., 8
    int num_nics = 0;       // e.g., 1 (RoCE/IB)
    uint64_t gpu_mem = 0;   // Required VRAM (optional)
    std::vector<int> specific_gpu_ids; // For specific pinning
};

struct AgentCapacity {
    std::string hostname;
    std::string ip;
    ResourceSpec total;
    ResourceSpec used;
    // Potentially: Topology map (which GPU connects to which NIC)
};
```

### 2.2 Agent Side

- **Startup**: Auto-detect resources using `NVML` (for NVIDIA GPUs) and `ibverbs` (for RDMA NICs), or load from `spoke_agent.yaml`.
- **Registration**: Send `AgentCapacity` to Hub upon connection.
- **Heartbeat**: Periodically report health and synced resource usage (reconciliation).

### 2.3 Hub Side

- **Registry**: Maintain a `Map<AgentID, AgentCapacity>`.
- **Accounting**: When `SpawnActor` is called with resource requirements, atomically decrement available resources. When Actor terminates, increment.

## 3. Gang Scheduling (GangScheduler)

NanoDeploy requires "Gang Scheduling" — either all 8 ranks start successfully on a suitable node, or none start. Partial starts are useless for tensor parallelism.

### 3.1 Spawn API Extension

Extend the `Client::spawn_actor` API to support resource constraints and group IDs.

```cpp
struct SpawnOptions {
    ResourceSpec resources;
    std::string placement_group; // "job_123"
    bool strict_locality = true; // Must be on same node
};

// Example Usage for NanoDeploy
for (int i = 0; i < 8; ++i) {
    client.spawn_actor("ModelRunner", ..., {
        .resources = { .num_gpus = 1 },
        .placement_group = "inference_req_1",
        .strict_locality = true
    });
}
```

### 3.2 Scheduling Logic (Hub)

1. **Group Request**: When the first request for `placement_group` "job_123" arrives:
   - Identify it needs 8 slots (based on hint or implicit config).
   - Scan `AgentRegistry` for *one* Agent with `free_gpus >= 8`.
   - **Lock** those resources for this group.
2. **Placement**: Assign all subsequent spawn requests for this group to that specific Agent.
3. **Fallback**: If no single node has resources, fail fast (or queue).

## 4. Lifecycle Management

### 4.1 Actor Lifecycle

- **Process Supervision**: The Agent uses `fork/exec` (implemented) but needs to monitor the PID.
- **Cleanup**:
  - If Actor process exits (0 or 1), Agent sends `ActorTerminated` to Hub.
  - Hub releases resources.
- **Zombie Cleanup**: Agent should clean up child processes if the Agent itself receives SIGTERM/SIGKILL.

### 4.2 Job/Gang Lifecycle

- If one Actor in a Gang fails (crashes), the entire job is likely invalid (ProcessGroup Hang).
- **Strategy**:
  - Client detects failure (RPC timeout or explicit error).
  - Client calls `StopJob(group_id)` on Hub.
  - Hub tells Agent to `kill -9` all PIDs associated with that group.

## 5. Implementation Roadmap

### Phase 1: Resource Awareness (Completed)

- [x] Define `ResourceSpec` protobuf/struct.
- [x] Update `nanodeploy_agent` to accept `--gpus <n>` flag.
- [x] Update Hub to store and display agent capacities.

### Phase 2: Simple Scheduler (Completed)

- [x] Implement `GangScheduler` logic in Hub (support `gangAllocate` RPC).
- [x] Support `strict_pack` (formerly `strict_locality`) flag.
- [x] Basic matching logic: "Find Agent with N free GPUs".

### Phase 3: Hardware Integration (In Progress)

- [ ] Integrate NVML in Agent to auto-detect GPU count/indices.
- [x] Support `CUDA_VISIBLE_DEVICES` isolation (Agent sets env var for Actor).

### Phase 4: Lifecycle & Robustness (In Progress)

- [x] **Automatic Resource Recovery**: Client auto-releases tickets on destruction.
- [x] **Thread-Safe Destructor**: Fixed deadlock during RDMA thread shutdown.
- [ ] Actor Supervision: Monitor PID and report unexpected exits.
- [ ] StopJob(group_id): Bulk termination of actors in a gang.

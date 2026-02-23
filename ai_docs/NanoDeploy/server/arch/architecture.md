# Rust Server + Router Architecture Design

## 1. Architectural Goals

Build a high-performance, low-latency Rust Gateway for LLM Inference. In **Phase III**, the primary goal is to support **Prefill-Decode (PD) Disaggregation**, allowing the system to scale Prefill (compute-bound) and Decode (memory-bound) resources independently.

## 2. System Overview

The system adopts a **Controller-Worker** architecture:

- **Controller (Rust Server)**: Handles HTTP requests, Tokenization, and Global Scheduling.
- **Workers (Python Engines)**: Execute the actual LLM inference.
  - **Prefill Workers**: Specialized execution of the Prompt Phase.
  - **Decode Workers**: Specialized execution of the Generation Phase (Token-by-Token).

```mermaid
graph TD
    Client([Client]) -->|HTTP/SSE| Server[Rust Server & Router]

    subgraph Control_Plane["Control Plane (Rust Server)"]
        HTTP[Axum Web Layer]
        Tokenizer[Tokenizer Service]
        scheduler[Disaggregated Scheduler]
        IPC[Spoke IPC Client]
    end

    HTTP --> Tokenizer
    Tokenizer --> scheduler
    scheduler --> IPC

    subgraph Data_Plane ["Data Plane (Distributed Engine Cluster)"]
        subgraph Host_A [Host A (Prefill Optimized)]
            P1[Prefill Engine 1]
            P2[Prefill Engine 2]
        end
        subgraph Host_B [Host B (Decode Optimized)]
            D1[Decode Engine 1]
            D2[Decode Engine 2]
        end
    end

    %% Control Signals (TCP/IP)
    IPC -->|Network/TCP| P1
    IPC -->|Network/TCP| P2
    IPC -->|Network/TCP| D1
    IPC -->|Network/TCP| D2

    %% Data Transfer (RDMA/High-Speed TCP)
    P1 -.->|Cross-Host Transfer| D1
    P1 -.->|Cross-Host Transfer| D2
    P2 -.->|Cross-Host Transfer| D1
    P2 -.->|Cross-Host Transfer| D2

    note_p2p["P2P Link (RDMA/TCP) <br/> Engines bind to 0.0.0.0 <br/> Handshake exchanges Public IP"] -.-> P1
```

## 3. Module Breakdown

### 3.1 `router` (The Brain)

The Router is the most critical component for disaggregation.

- **Global Queue**: A central priority queue for all incoming requests.
- **Topology Manager**: Maintains the map of all connected Engines and their roles (`Prefill` vs `Decode`).
- **KV Lifecycle Manager**: Tracks which Engine holds the KV Cache for a given Request ID.
  - *State 1*: `None` (New Request)
  - *State 2*: `OnPrefill(EngineID)`
  - *State 3*: `Migrating(SourceID -> TargetID)`
  - *State 4*: `OnDecode(EngineID)`

### 3.2 `engine_client` (The Limb)

Manages the `Spoke` connection to Python Engines.

- **Multiplexing**: Handles multiple Engine connections concurrently.
- **Handshake logic**: Implements the `P2PInit` + `P2PConnect` sequence during startup to link Prefill and Decode nodes.

## 4. Disaggregated Workflow (The Lifecycle)

### 4.1 Startup Phase (Handshake)

Before serving requests, the Rust Server must mesh the engines.

1. **Connect**: Rust Server connects to all configured Engines.
2. **Identify**: Rust Server queries each Engine for its Role (`mode="prefill"` or `decode`) and Hardware ID/Topology Info.
3. **Mesh**:
   - Rust Server sends `P2PInit` to all engines with the global topology map.
   - Engines allocate P2P buffers/transports.
   - Rust Server sends `P2PConnect` to enable direct links between Prefill and Decode nodes.

### 4.2 Request Phase

1. **Ingest**: Client sends prompt. Tokenizer converts to IDs.
2. **Prefill Schedule**: Scheduler picks a `Prefill Engine` (e.g., Round Robin or Least Load).
3. **Prefill Exec**: Router sends `AddRequest` to the chosen Prefill Engine.
4. **Handoff**:
   - Prefill Engine computes prompt, stores KV Cache.
   - Prefill Engine responds to Router with `PrefillDone` (and `kv_handle` or similar).
5. **Migration Queue**:
   - Router moves request to **PendingDecode Queue**.
   - Scheduler waits for a `Decode Engine` with sufficient KV Block capacity.
6. **Migration & Decode**:
   - Once a slot is available, Router sends `Migrate(req_id, target_node)` to Prefill Engine (or `AddRequest` to Decode Engine).
   - *Refinement*: Based on `pd_disagg.py`, the orchestration seems to be: The "Migrated" sequence is added to Decode Engine.
7. **Decode Exec**: Decode Engine receives the request context, fetches KV Cache via P2P (if not pushed), and begins generation.
8. **Cleanup**: Once Decode stream starts, Router sends `Free` to Prefill Engine to reclaim memory.

## 5. Spoke Protocol Extensions

To support this, we need new Command IDs in the IPC protocol (defined in `proto/sequence.fbs` or similar):

- **P2P_INIT (0x30)**: Payload contains Cluster Metadata.
- **P2P_CONNECT (0x31)**: Payload contains Peer Endpoints.
- **MIGRATE_OUT (0x32)**: Instruct Prefill to send KV to Target.
- **MIGRATE_IN (0x33)**: Instruct Decode to receive KV (if not implicit in AddRequest).
- **FREE_PREFILL (0x34)**: Explicitly free prefill resources.

## 6. Implementation Stages

1. **Multi-Client Support**: Upgrade `EngineManager` to hold a Map of Clients, not just one.
2. **Handshake Implementation**: Implement the startup mesh logic.
3. **Migration Logic**: Implement the "Prefill Done -> Queue for Decode" state transition in Scheduler.

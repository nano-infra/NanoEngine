# NanoInfra Virtual Team Directive: Etcd-Based Automated DLSlime P2P Mesh

**Context (Architecture Correction):**
The CEO (User) has clarified a critical architectural detail: The high-performance data path between Engines is a **DLSlime P2P connection**, not a raw Spoke connection.
We are tasked with creating a new test entry point, **`NanoDeploy/examples/pd_disagg_etcd.py`**, to demonstrate automated discovery and DLSlime link establishment using etcd.

**Core Mechanisms (Auto-Mesh with Crash Safety):**

1. **Lease & KeepAlive:** Upon startup, every Engine must acquire a lease from etcd (TTL=10s) and maintain it via a background keep-alive thread. If the process crashes (e.g., `kill -9`), the node's DLSlime metadata must automatically vanish from etcd within 10 seconds.
2. **Watch & Link (DLSlime):**
   - Engines must actively `Watch` the prefix `/nanodeploy/mesh/...`.
   - **On Peer Online (PUT):** Parse the peer's DLSlime address and trigger `self.engine.p2p_connect(remote_info)` to perform the DLSlime handshake.
   - **On Peer Offline (DELETE):** Trigger `self.engine.p2p_disconnect(remote_id)` to clean up the DLSlime context and free resources.
3. **Barrier:** The new script `pd_disagg_etcd.py` must use a blocking call (`wait_for_mesh`) to ensure the DLSlime topology is fully connected before issuing any inference requests.

______________________________________________________________________

**Role Assignment:**

1. **CTO (Chief Technology Officer):**

   - **Schema Definition:**
     - **Key:** `/nanodeploy/mesh/{cluster_id}/nodes/{engine_id}` (Must be attached to the Lease).
     - **Value (JSON):** Must contain all necessary fields for the DLSlime handshake.
       ```json
       {
         "dlslime_ip": "192.168.1.10",    // DLSlime listening IP
         "dlslime_port": 4000,            // DLSlime listening Port
         "rank": 0,                       // For connection directionality (e.g., Low Rank connects to High Rank)
         "role": "prefill",               // or "decode"
         "kv_cache_meta": { ... }
       }
       ```
   - **Connection Policy:** Define the rule to prevent double connections (e.g., "Only initiate connection if `my_rank < remote_rank`").

2. **DEVELOPER (Senior Engineer):**

   - **Server-Side Refactor (`nanodeploy/server/engine_server.py`):**
     - **Registration:** Integrate the `etcd3` client. On startup, upload DLSlime metadata attached to a Lease and start the KeepAlive thread.
     - **Event Loop:** Implement a background Watch Loop. Handle `PUT` events to initiate DLSlime connections and `DELETE` events to tear them down.
   - **New Script (`examples/pd_disagg_etcd.py`):**
     - Create a clean disaggregation inference script.
     - **Remove** all manual `p2p_init` and `p2p_connect` calls.
     - **Add** the barrier: `ray.get(decode.wait_for_mesh.remote(expected_peers=1))`.

3. **PM (Project Manager):**

   - **Milestone:** "Crash & Recovery Verification" — Verify that if the Prefill node is killed (`kill -9`), the Decode node's logs explicitly show "DLSlime Peer disconnected: <ID>" within the TTL window.

4. **TESTING (QA Engineer):**

   - **Connectivity Test:** After the auto-mesh is established, run a real KV Cache transfer (Prefill -> Decode) to confirm the DLSlime pipe is physically functional, not just logically registered.
   - **Resilience Test:** Simulate a node restart and verify that DLSlime connections can be re-established automatically (Re-handshake).

______________________________________________________________________

**Current Task (Immediate Action):**

1. **CTO:** Please provide the exact JSON Schema definition for the DLSlime handshake data.
2. **DEVELOPER:** Please provide the pseudo-code for `engine_server.py` specifically handling the **DELETE event**, showing how to safely clean up the DLSlime state.

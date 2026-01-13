#include "nanodeploy/worker/deep_ep_ipc.h"
#include "spoke/csrc/client.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace nanodeploy;
using namespace spoke;

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <hub_ip> <hub_port> [num_actors]" << std::endl;
        std::cerr << "  num_actors: Number of actors (2-8, default: 8). DeepEP requires multi-GPU." << std::endl;
        return 1;
    }

    std::string hub_ip   = argv[1];
    int         hub_port = std::atoi(argv[2]);

    // Parse number of actors (default: 8, minimum: 2 for DeepEP)
    uint32_t actors_per_node = 8;
    if (argc >= 4) {
        int num_actors = std::atoi(argv[3]);
        if (num_actors < 2 || num_actors > 8) {
            std::cerr << "Error: num_actors must be 2-8 (DeepEP requires multi-GPU), got " << num_actors << std::endl;
            std::cerr << "Note: Single-card MoE uses local computation, not DeepEP." << std::endl;
            return 1;
        }
        actors_per_node = static_cast<uint32_t>(num_actors);
    }

    std::cout << "=== DeepEP Test: " << actors_per_node << " Actor(s) ===" << std::endl;

    // 1. Connect to Hub
    Client client(hub_ip, hub_port, true);  // true = Hub Mode

    // 2. Allocate Resources
    uint32_t     num_nodes = 1;
    ResourceSpec res_spec;
    res_spec.num_gpus = 1;  // Require 1 GPU per actor

    std::cout << "Requesting Gang Allocation..." << std::endl;
    auto alloc_fut = client.gangAllocate(num_nodes, actors_per_node, res_spec);

    spoke::AllocateResp alloc_resp;
    try {
        alloc_resp = alloc_fut.get();
    }
    catch (const std::exception& e) {
        std::cerr << "Allocation failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Allocation successful. Ticket: " << alloc_resp.ticket_id << std::endl;

    // 3. Launch Actors
    std::vector<std::string>       actor_ids;
    std::vector<std::future<bool>> launch_futs;
    for (int i = 0; i < num_nodes * actors_per_node; ++i) {
        std::string id = "ep_worker_" + std::to_string(i);
        actor_ids.push_back(id);
        // We can pass arguments if needed, but InitReq handles config
        launch_futs.push_back(client.launchActor(alloc_resp.ticket_id, i, "DeepEPTestActor", id, ""));
    }

    for (auto& f : launch_futs) {
        if (!f.get()) {
            std::cerr << "Actor launch failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "All Actors launched." << std::endl;

    // 4. Initialize DeepEP
    // Action IDs must match Spoke Executor registration
    // DeepEPTestActor starts at kUserActionStart + 20
    Action kInit    = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 20);
    Action kGetInfo = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 21);
    Action kSync    = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 22);
    Action kTest    = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 23);

    // IMPORTANT: Init requests MUST be sent in parallel!
    // NVSHMEM initialization is a collective operation - all ranks must enter simultaneously.
    std::cout << "Initializing DeepEP (parallel)..." << std::endl;
    std::vector<std::future<DeepEPInitResp>> init_futs;
    for (int i = 0; i < actor_ids.size(); ++i) {
        DeepEPInitReq req;
        req.rank             = i;
        req.world_size       = actor_ids.size();
        req.low_latency_mode = true;
        req.num_rdma_bytes   = 1024 * 1024 * 1024;  // 1GB
        // Use NVLink buffer only for multi-card; single-card doesn't need it
        req.num_nvl_bytes = (actors_per_node > 1) ? (1024 * 1024 * 1024) : 0;
        init_futs.push_back(client.callRemote<DeepEPInitReq, DeepEPInitResp>(actor_ids[i], kInit, req));
    }

    // Wait for all Init responses
    for (int i = 0; i < actor_ids.size(); ++i) {
        auto resp = init_futs[i].get();
        if (!resp.success) {
            std::cerr << "Init failed on " << actor_ids[i] << ": " << resp.message << std::endl;
            return 1;
        }
    }
    std::cout << "All actors initialized." << std::endl;

    // 5. Get Info (Handles) - send in parallel for consistency
    std::cout << "Gathering IPC Handles (parallel)..." << std::endl;
    std::vector<std::future<DeepEPInfoResp>> info_futs;
    for (int i = 0; i < actor_ids.size(); ++i) {
        DeepEPInfoReq req;
        info_futs.push_back(client.callRemote<DeepEPInfoReq, DeepEPInfoResp>(actor_ids[i], kGetInfo, req));
    }

    DeepEPSyncReq sync_req;
    sync_req.valid_mask.resize(actor_ids.size(), 0);

    // Collect responses
    for (int i = 0; i < actor_ids.size(); ++i) {
        auto resp = info_futs[i].get();

        if (resp.ipc_handle.empty()) {
            std::cerr << "Got empty handle from " << actor_ids[i] << std::endl;
            return 1;
        }

        // Only set handle_size once
        if (i == 0)
            sync_req.handle_size = resp.ipc_handle.size();

        sync_req.all_handles_flat.insert(
            sync_req.all_handles_flat.end(), resp.ipc_handle.begin(), resp.ipc_handle.end());
        sync_req.valid_mask[i] = 1;

        if (!resp.nvshmem_unique_id.empty()) {
            sync_req.nvshmem_unique_id = resp.nvshmem_unique_id;
            std::cout << "Got NVSHMEM ID from rank " << i << std::endl;
        }
    }

    // 6. Sync
    std::cout << "Broadcasting Sync..." << std::endl;
    std::vector<std::future<DeepEPSyncResp>> sync_futs;
    for (int i = 0; i < actor_ids.size(); ++i) {
        sync_futs.push_back(client.callRemote<DeepEPSyncReq, DeepEPSyncResp>(actor_ids[i], kSync, sync_req));
    }

    for (int i = 0; i < actor_ids.size(); ++i) {
        auto resp = sync_futs[i].get();
        if (!resp.success) {
            std::cerr << "Sync failed on " << actor_ids[i] << ": " << resp.message << std::endl;
            return 1;
        }
    }

    // 7. Run Test (send requests in parallel - required because clean_low_latency_buffer has NVSHMEM barrier)
    std::cout << "Running Low Latency Test..." << std::endl;
    DeepEPTestReq test_req;
    // Default params

    // Send all test requests in parallel (required for NVSHMEM collective operations)
    std::vector<std::future<DeepEPTestResp>> test_futs;
    for (int i = 0; i < actor_ids.size(); ++i) {
        test_futs.push_back(client.callRemote<DeepEPTestReq, DeepEPTestResp>(actor_ids[i], kTest, test_req));
    }

    // Collect results
    for (int i = 0; i < actor_ids.size(); ++i) {
        auto resp = test_futs[i].get();
        if (resp.success) {
            std::cout << "Rank " << i << ": Dispatch=" << resp.dispatch_lat_us << "us, Combine=" << resp.combine_lat_us
                      << "us" << std::endl;
        }
        else {
            std::cerr << "Test failed on rank " << i << ": " << resp.message << std::endl;
        }
    }

    // 8. Shutdown Actors
    std::cout << "Shutting down actors..." << std::endl;
    for (const auto& id : actor_ids) {
        try {
            client.stopRemote(id);
        }
        catch (...) {
            // Ignore connection errors during shutdown
        }
    }

    // 9. Release GPU Resources
    std::cout << "Releasing GPU resources..." << std::endl;
    try {
        if (client.gangRelease(alloc_resp.ticket_id).get()) {
            std::cout << "Resources released successfully." << std::endl;
        }
        else {
            std::cerr << "Failed to release resources." << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Release failed: " << e.what() << std::endl;
    }

    std::cout << "Test Complete." << std::endl;
    return 0;
}

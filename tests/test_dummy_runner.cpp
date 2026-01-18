#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nanodeploy/csrc/worker/dummy_runner.h"
#include "nanodeploy/csrc/worker/dummy_runner_ipc.h"  // Includes serialization
#include "spoke/csrc/client.h"

using namespace nanodeploy;

constexpr int kRunAction      = static_cast<int>(spoke::Action::kUserActionStart);
constexpr int kInitDistAction = static_cast<int>(spoke::Action::kUserActionStart) + 1;

int main(int argc, char** argv)
{
    // 1. Initialize Spoke Client
    std::string ip   = "127.0.0.1";
    int         port = 9000;  // Default Daemon port from daemon.cpp

    if (argc > 1)
        ip = argv[1];
    if (argc > 2)
        port = std::atoi(argv[2]);

    std::cout << "[Test] Connecting to Daemon at " << ip << ":" << port << std::endl;
    // auto client = std::make_shared<spoke::Client>(ip, port); // Single client causes HOL blocking in Daemon

    int                      world_size = 8;
    std::vector<std::string> actor_ids;

    // Create a pool of clients, one per actor/rank, to ensure full parallelism
    // The Daemon handles each connection in a separate thread, preventing blocking.
    std::vector<std::shared_ptr<spoke::Client>> clients;
    for (int i = 0; i < world_size; ++i) {
        clients.push_back(std::make_shared<spoke::Client>(ip, port, true));  // Enable RDMA
    }

    // 2. Spawn 8 Actors (Dynamic Spawning)
    // We can use clients[0] for spawning, or round robin. Spawning is fast.
    std::string actor_type = "DummyRunner";
    for (int i = 0; i < world_size; ++i) {
        std::string id = "DummyRunner:" + std::to_string(i);
        actor_ids.push_back(id);

        std::cout << "[Test] Spawning actor " << id << "..." << std::endl;
        clients[i]->spawnRemote(actor_type, id);
    }

    // NOTE: RDMA init moved AFTER distributed group setup to avoid c10d conflicts

    // 3. Initialize Process Group
    std::string master_addr = "127.0.0.1";
    int         master_port = 29500;

    std::vector<std::future<DistResp>> init_futures;
    for (int rank = 0; rank < world_size; ++rank) {
        DistConfig config;
        config.rank        = rank;
        config.world_size  = world_size;
        config.master_addr = master_addr;
        config.master_port = master_port;

        std::cout << "[Test] Initializing Rank " << rank << " (Async)..." << std::endl;

        // Async call using dedicated client
        init_futures.push_back(clients[rank]->callRemote<DistConfig, DistResp>(
            actor_ids[rank], static_cast<spoke::Action>(kInitDistAction), config));
    }

    // Wait for all inits
    for (int rank = 0; rank < world_size; ++rank) {
        auto resp = init_futures[rank].get();
        if (resp.success) {
            std::cout << "[Test] Rank " << rank << " init success: " << resp.message << std::endl;
        }
        else {
            std::cerr << "[Test] Rank " << rank << " init failed: " << resp.message << std::endl;
            return 1;
        }
    }

    std::cout << "[Test] All actors initialized distributed group!" << std::endl;

    // 3.5. Now initialize RDMA (after c10d setup to avoid conflicts)
    std::cout << "[Test] Initializing RDMA connections..." << std::endl;
    for (int i = 0; i < world_size; ++i) {
        if (!clients[i]->initRDMA(actor_ids[i])) {
            std::cerr << "[Test] Failed to init RDMA for " << actor_ids[i] << std::endl;
            return 1;
        }
        std::cout << "[Test] RDMA initialized for " << actor_ids[i] << std::endl;
    }

    // 4. Test Run (mock stateful run)
    // Run a sequence on Rank 0
    std::vector<int> tokens(10, 1);
    auto             seq = std::make_shared<Sequence>(tokens);
    RunReq           req;
    req.push_back(seq);

    std::cout << "[Test] Calling run() on Rank 0..." << std::endl;
    // Use clients[0] for Rank 0
    auto tensor =
        clients[0]->callRemote<RunReq, RunResp>(actor_ids[0], static_cast<spoke::Action>(kRunAction), req).get();

    std::cout << "[Test] Rank 0 returned tensor: " << tensor.sizes() << std::endl;
    assert(tensor.size(0) == 1);
    assert(tensor.size(1) == 16);

    std::cout << "[Test] Distributed Test Passed!" << std::endl;

    // 5. Test AllReduce (Real c10d logic)
    // We must invoke this PARALLELIZABLY because all_reduce is blocking.
    // Code: Launch 8 async calls, then wait.

    std::cout << "[Test] Starting AllReduce (Sum 0..7 should be 28)..." << std::endl;
    std::vector<std::future<RunResp>> futures;
    nanodeploy::RunAllReduceReq       empty_req;

    constexpr int kAllReduceAction = static_cast<int>(spoke::Action::kUserActionStart) + 2;

    for (int i = 0; i < world_size; ++i) {
        // Use client i for actor i
        futures.push_back(clients[i]->callRemote<nanodeploy::RunAllReduceReq, RunResp>(
            actor_ids[i], static_cast<spoke::Action>(kAllReduceAction), empty_req));
    }

    // Verify results
    for (int i = 0; i < world_size; ++i) {
        auto  tensor = futures[i].get();
        float val    = tensor.item<float>();
        std::cout << "[Test] Rank " << i << " all_reduce result: " << val << std::endl;

        // Sum of 0..7 is 28
        assert(std::abs(val - 28.0) < 1e-5);
    }
    std::cout << "[Test] AllReduce Verification Passed!" << std::endl;

    // 5. Cleanup: Stop all actors
    std::cout << "[Test] Stopping all actors..." << std::endl;
    for (int i = 0; i < world_size; ++i) {
        clients[i]->stopRemote(actor_ids[i]);
    }
    std::cout << "[Test] Done." << std::endl;

    return 0;
}

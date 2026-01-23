#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nanodeploy/csrc/worker/model_runner_ipc.h"
#include "spoke/csrc/client.h"

using namespace nanodeploy;

// Action IDs from spoke_executor.cpp
constexpr int kInitAction               = static_cast<int>(spoke::Action::kUserActionStart) + 10;
constexpr int kGetAvailableBlocksAction = static_cast<int>(spoke::Action::kUserActionStart) + 17;
constexpr int kAllocKVBlocksAction      = static_cast<int>(spoke::Action::kUserActionStart) + 18;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_path> [world_size]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    int         world_size  = (argc > 2) ? std::atoi(argv[2]) : 1;

    std::string ip   = "127.0.0.1";
    int         port = 8888;

    std::cout << "[Test] Spawning " << world_size << " ModelRunners..." << std::endl;

    std::vector<std::shared_ptr<spoke::Client>> clients;
    std::vector<std::string>                    actor_ids;
    for (int i = 0; i < world_size; ++i) {
        clients.push_back(std::make_shared<spoke::Client>(ip, port, true));
        std::string id = "ModelRunner:" + std::to_string(i);
        actor_ids.push_back(id);
        clients[i]->spawnRemote("ModelRunner", id);

        // Enable RDMA (simulated or real)
        clients[i]->initRDMA(id);
    }

    // Initialize
    std::vector<std::future<ModelInitResp>> init_futures;
    for (int i = 0; i < world_size; ++i) {
        ModelInitReq req;
        req.config_path = config_path;
        req.rank        = i;
        req.world_size  = world_size;

        init_futures.push_back(clients[i]->callRemote<ModelInitReq, ModelInitResp>(
            actor_ids[i], static_cast<spoke::Action>(kInitAction), req));
    }

    for (auto& f : init_futures) {
        bool success = f.get();
        if (!success) {
            std::cerr << "Failed to init runner!" << std::endl;
            return 1;
        }
    }
    std::cout << "[Test] Initialization Complete." << std::endl;

    // Test 1: Get Available Blocks
    std::cout << "[Test] Querying Available Blocks..." << std::endl;
    std::vector<std::future<GetAvailableKVBlocksResp>> avail_futures;
    for (int i = 0; i < world_size; ++i) {
        GetAvailableKVBlocksReq req;
        req.max_batch_size         = 16;
        req.gpu_memory_utilization = 0.5f;  // Use small ratio for testing

        avail_futures.push_back(clients[i]->callRemote<GetAvailableKVBlocksReq, GetAvailableKVBlocksResp>(
            actor_ids[i], static_cast<spoke::Action>(kGetAvailableBlocksAction), req));
    }

    int min_blocks = 1000000;
    for (int i = 0; i < world_size; ++i) {
        int blocks = avail_futures[i].get();
        std::cout << "[Test] Rank " << i << " Available Blocks: " << blocks << std::endl;
        assert(blocks > 0);
        if (blocks < min_blocks)
            min_blocks = blocks;
    }

    // Test 2: Alloc KV Blocks
    std::cout << "[Test] Allocating " << min_blocks << " Blocks..." << std::endl;
    std::vector<std::future<AllocKVBlocksResp>> alloc_futures;
    for (int i = 0; i < world_size; ++i) {
        AllocKVBlocksReq req;
        req.num_blocks     = min_blocks;
        req.max_batch_size = 16;

        alloc_futures.push_back(clients[i]->callRemote<AllocKVBlocksReq, AllocKVBlocksResp>(
            actor_ids[i], static_cast<spoke::Action>(kAllocKVBlocksAction), req));
    }

    for (int i = 0; i < world_size; ++i) {
        bool success = alloc_futures[i].get();
        std::cout << "[Test] Rank " << i << " Allocation Success: " << success << std::endl;
        assert(success);
    }

    std::cout << "[Test] KV Cache Split Test Complete. Success!" << std::endl;

    // Cleanup
    for (int i = 0; i < world_size; ++i) {
        clients[i]->stopRemote(actor_ids[i]);
    }

    return 0;
}

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nanodeploy/worker/model_runner_ipc.h"
#include "spoke/csrc/client.h"

using namespace nanodeploy;

// Action IDs from spoke_executor.cpp
constexpr int kInitAction = static_cast<int>(spoke::Action::kUserActionStart) + 10;
constexpr int kRunAction  = static_cast<int>(spoke::Action::kUserActionStart) + 11;

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

        // Enable RDMA
        if (!clients[i]->initRDMA(id)) {
            std::cerr << "[Test] Failed to init RDMA for " << id << std::endl;
            return 1;
        }
        std::cout << "[Test] RDMA Initialized for " << id << std::endl;
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

    // Run Forward
    std::vector<std::future<ModelRunResp>> run_futures;
    for (int i = 0; i < world_size; ++i) {
        ModelRunReq req;
        req.is_prefill = true;
        // Mock Sequence
        auto seq = std::make_shared<Sequence>(std::vector<int>{1, 2, 3});
        req.seqs.push_back(seq);

        run_futures.push_back(clients[i]->callRemote<ModelRunReq, ModelRunResp>(
            actor_ids[i], static_cast<spoke::Action>(kRunAction), req));
    }

    for (int i = 0; i < world_size; ++i) {
        auto resp   = run_futures[i].get();
        auto tensor = resp.tensor;
        std::cout << "[Test] Rank " << i << " Output: " << tensor.sizes() << std::endl;
        // Just verify it's not empty
        assert(tensor.numel() > 0);
    }
    std::cout << "[Test] Run Complete. Success!" << std::endl;

    // Cleanup
    for (int i = 0; i < world_size; ++i) {
        clients[i]->stopRemote(actor_ids[i]);
    }

    return 0;
}

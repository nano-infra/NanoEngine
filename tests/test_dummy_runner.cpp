#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "spoke/agent.h"  // This might not be needed anymore, but keeping it as it was in original
#include "spoke/types.h"

// Include the definition of RunReq, RunResp, and serialization specializations
// This is header-only so it's fine to include here for client-side usage
#include "nanodeploy/worker/dummy_runner.h"
#include "nanodeploy/worker/dummy_runner_ipc.h"

#include "spoke/client.h"

using namespace spoke;
using namespace nanodeploy;

int main(int argc, char** argv)
{
    std::string ip   = "127.0.0.1";
    int         port = 9999;  // Default port for spoke daemon

    if (argc > 1)
        ip = argv[1];
    if (argc > 2)
        port = std::stoi(argv[2]);

    std::cout << "[Test] Connecting to Daemon at " << ip << ":" << port << "..." << std::endl;

    try {
        Client client(ip, port);

        std::string actor_id = "test_runner_01";
        std::cout << "[Test] Spawning DummyRunner remote actor (" << actor_id << ")..." << std::endl;
        client.spawnRemote("DummyRunner", actor_id);

        // Give it a moment to spawn
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "[Test] Preparing input data..." << std::endl;
        RunReq req;
        // Add some dummy sequences
        // Sequence(const std::vector<int>& token_ids, double temperature = 1.0, int max_tokens = 256, bool ignore_eos =
        // false);
        req.push_back(std::make_shared<Sequence>(std::vector<int>{1, 2, 3}));
        req.push_back(std::make_shared<Sequence>(std::vector<int>{4, 5, 6}));
        req.push_back(std::make_shared<Sequence>(std::vector<int>{7, 8, 9}));
        req.push_back(std::make_shared<Sequence>(std::vector<int>{10, 11, 12}));  // 4 sequences

        std::cout << "[Test] Calling 'run' method..." << std::endl;
        for (int i = 0; i < 2; ++i) {
            auto future = client.callRemote<RunReq, RunResp>(actor_id, spoke::Action::kUserActionStart, req);

            std::cout << "[Test] Waiting for result..." << std::endl;
            RunResp result = future.get();

            std::cout << "[Test] Result received!" << std::endl;
            std::cout << "       Tensor sizes: " << result.sizes() << std::endl;
            // Validation
            assert(result.size(0) == 4);
            assert(result.size(1) == 16);
            std::cout << "[Test] Verification PASSED!" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Test] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

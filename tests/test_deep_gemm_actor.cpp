#include "nanodeploy/worker/deep_gemm_ipc.h"
#include "spoke/csrc/client.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace nanodeploy;
using namespace spoke;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <hub_ip> <hub_port>" << std::endl;
        return 1;
    }

    std::string hub_ip   = argv[1];
    int         hub_port = std::atoi(argv[2]);

    // 1. Connect to Hub
    Client client(hub_ip, hub_port, true);  // true = Hub Mode

    // 2. Allocate Resources (e.g. 1 Node, 1 Actor)
    // DeepGemm test usually single GPU is enough for unit test, but we can launch multiple if needed.
    // Let's use 1 node, 1 actor for simplicity.
    uint32_t     num_nodes       = 1;
    uint32_t     actors_per_node = 1;
    ResourceSpec res_spec;
    res_spec.num_gpus = 1;  // Require 1 GPU per actor

    std::cout << "Requesting Allocation..." << std::endl;
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
    std::string actor_id   = "gemm_worker_0";
    auto        launch_fut = client.launchActor(alloc_resp.ticket_id, 0, "DeepGemmTestActor", actor_id, "");

    if (!launch_fut.get()) {
        std::cerr << "Actor launch failed!" << std::endl;
        return 1;
    }
    std::cout << "Actor launched." << std::endl;

    // 4. Initialize DeepGemm
    // Action IDs must match Spoke Executor registration
    Action kInit    = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 30);
    Action kRunTest = static_cast<Action>(static_cast<int>(Action::kUserActionStart) + 31);

    std::cout << "Initializing DeepGemm..." << std::endl;
    DeepGemmInitReq init_req;
    init_req.rank       = 0;
    init_req.world_size = 1;

    auto init_resp = client.callRemote<DeepGemmInitReq, DeepGemmInitResp>(actor_id, kInit, init_req).get();
    if (!init_resp.success) {
        std::cerr << "Init failed: " << init_resp.message << std::endl;
        return 1;
    }
    std::cout << "DeepGemm initialized." << std::endl;

    // 5. Run FP8 GEMM Test
    std::cout << "Running FP8 GEMM Test..." << std::endl;
    DeepGemmTestReq test_req_fp8;
    test_req_fp8.mode = (int)DeepGemmTestMode::kFp8Gemm;
    test_req_fp8.m    = 4096;
    test_req_fp8.n    = 4096;
    test_req_fp8.k    = 4096;

    auto test_resp_fp8 = client.callRemote<DeepGemmTestReq, DeepGemmTestResp>(actor_id, kRunTest, test_req_fp8).get();
    if (test_resp_fp8.success) {
        std::cout << "FP8 GEMM Latency: " << test_resp_fp8.lat_us << " us" << std::endl;
    }
    else {
        std::cerr << "FP8 GEMM Failed: " << test_resp_fp8.message << std::endl;
    }

    // 6. Run Masked Group GEMM Test
    std::cout << "Running Masked Group GEMM Test..." << std::endl;
    DeepGemmTestReq test_req_grp;
    test_req_grp.mode       = (int)DeepGemmTestMode::kMaskedGroupGemm;
    test_req_grp.m          = 4096;
    test_req_grp.n          = 4096;
    test_req_grp.k          = 4096;
    test_req_grp.num_groups = 128;

    auto test_resp_grp = client.callRemote<DeepGemmTestReq, DeepGemmTestResp>(actor_id, kRunTest, test_req_grp).get();
    if (test_resp_grp.success) {
        std::cout << "Masked Group GEMM Latency: " << test_resp_grp.lat_us << " us" << std::endl;
    }
    else {
        std::cerr << "Masked Group GEMM Failed: " << test_resp_grp.message << std::endl;
    }

    // 7. Run Grouped BF16 GEMM Test (Large M)
    std::cout << "Running Grouped BF16 GEMM Test (M=128)..." << std::endl;
    DeepGemmTestReq test_req_grouped_bf16;
    test_req_grouped_bf16.mode       = (int)DeepGemmTestMode::kGroupedBf16Gemm;
    test_req_grouped_bf16.m          = 128;
    test_req_grouped_bf16.n          = 4096;
    test_req_grouped_bf16.k          = 4096;
    test_req_grouped_bf16.num_groups = 8;

    auto test_resp_grouped_bf16 =
        client.callRemote<DeepGemmTestReq, DeepGemmTestResp>(actor_id, kRunTest, test_req_grouped_bf16).get();
    if (test_resp_grouped_bf16.success) {
        std::cout << "Grouped BF16 GEMM (M=128) Latency: " << test_resp_grouped_bf16.lat_us << " us" << std::endl;
    }
    else {
        std::cerr << "Grouped BF16 GEMM (M=128) Failed: " << test_resp_grouped_bf16.message << std::endl;
    }

    // 8. Run Grouped BF16 GEMM Test (Small M - simulate single token decode)
    std::cout << "Running Grouped BF16 GEMM Test (M=8)..." << std::endl;
    DeepGemmTestReq test_req_small;
    test_req_small.mode       = (int)DeepGemmTestMode::kGroupedBf16Gemm;
    test_req_small.m          = 8;
    test_req_small.n          = 4096;
    test_req_small.k          = 4096;
    test_req_small.num_groups = 8;

    auto test_resp_small =
        client.callRemote<DeepGemmTestReq, DeepGemmTestResp>(actor_id, kRunTest, test_req_small).get();
    if (test_resp_small.success) {
        std::cout << "Grouped BF16 GEMM (M=8) Latency: " << test_resp_small.lat_us << " us" << std::endl;
    }
    else {
        std::cerr << "Grouped BF16 GEMM (M=8) Failed: " << test_resp_small.message << std::endl;
    }

    // 9. Shutdown Actor
    std::cout << "Shutting down actor..." << std::endl;
    try {
        client.stopRemote(actor_id);
    }
    catch (...) {
    }

    // 10. Release Resources
    std::cout << "Releasing resources..." << std::endl;
    client.gangRelease(alloc_resp.ticket_id).get();

    std::cout << "Test Complete." << std::endl;
    return 0;
}

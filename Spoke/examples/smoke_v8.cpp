// smoke_v8.cpp - Test Per-Actor RDMA Channels
#include "spoke/csrc/client.h"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv)
{
    std::string hub_ip   = "127.0.0.1";
    int         hub_port = 23760;
    if (argc > 1)
        hub_port = atoi(argv[1]);

    std::cout << "[Smoke v8] Starting Per-Actor RDMA Test..." << std::endl;
    spoke::Client client(hub_ip, hub_port, true);  // Hub Mode

    // 1. Allocate 2 slots for 2 independent RDMA actors
    spoke::ResourceSpec res;
    res.num_gpus    = 1;
    auto alloc_resp = client.gangAllocate(1, 2, res, true).get();

    if (strlen(alloc_resp.ticket_id) == 0) {
        std::cerr << "[Smoke v8] Allocation Failed!" << std::endl;
        return 1;
    }

    std::cout << "[Smoke v8] Allocation Success. Ticket: " << alloc_resp.ticket_id << std::endl;

    // 2. Launch 2 actors
    std::string actor0_id = "math_expert_0";
    std::string actor1_id = "math_expert_1";

    client.launchActor(alloc_resp.ticket_id, 0, "RdmaMathActor", actor0_id, "");
    client.launchActor(alloc_resp.ticket_id, 1, "RdmaMathActor", actor1_id, "");

    // Give them a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 3. Initialize RDMA for BOTH actors independently
    std::cout << "[Smoke v8] Establishing RDMA channel for " << actor0_id << "..." << std::endl;
    if (!client.initRDMA(actor0_id)) {
        std::cerr << "[Smoke v8] Failed to init RDMA for " << actor0_id << std::endl;
        return 1;
    }

    std::cout << "[Smoke v8] Establishing RDMA channel for " << actor1_id << "..." << std::endl;
    if (!client.initRDMA(actor1_id)) {
        std::cerr << "[Smoke v8] Failed to init RDMA for " << actor1_id << std::endl;
        return 1;
    }

    // 4. Test parallel RDMA calls
    std::cout << "[Smoke v8] Testing RDMA calls..." << std::endl;

    auto fut0 = client.callRemote<int, int>(actor0_id, (spoke::Action)0x10,
                                            5);  // 5^2 = 25
    auto fut1 = client.callRemote<std::string, std::string>(actor1_id, (spoke::Action)0x11, "HelloRDMA");

    int         res0 = fut0.get();
    std::string res1 = fut1.get();

    std::cout << "[Smoke v8] Result from " << actor0_id << ": " << res0 << " (Expected 25)" << std::endl;
    std::cout << "[Smoke v8] Result from " << actor1_id << ": " << res1 << " (Expected Pong: HelloRDMA)" << std::endl;

    assert(res0 == 25);
    assert(res1 == "Pong: HelloRDMA");

    std::cout << "[Smoke v8] ALL TESTS PASSED!" << std::endl;

    // 5. Release resources
    std::cout << "[Smoke v8] Releasing resources..." << std::endl;
    bool released = client.gangRelease(alloc_resp.ticket_id).get();
    if (released) {
        std::cout << "[Smoke v8] Resources released successfully." << std::endl;
    }
    else {
        std::cerr << "[Smoke v8] Failed to release resources!" << std::endl;
    }

    return 0;
}

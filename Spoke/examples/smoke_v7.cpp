// smoke_v7.cpp - Test Spoke V2 Allocation API

#include "spoke/csrc/client.h"
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv)
{
    std::string hub_ip   = "127.0.0.1";
    int         hub_port = 23760;
    if (argc > 1)
        hub_port = atoi(argv[1]);

    std::cout << "[Smoke v7] Connecting to Hub for Gang Allocation..." << std::endl;
    spoke::Client client(hub_ip, hub_port, true);  // Hub Mode = true

    // 1. Request Allocation
    spoke::ResourceSpec res;
    res.num_gpus = 1;

    // Allocate 1 Node, 2 Actors per node (e.g., simulating 2-GPU job)
    auto alloc_fut  = client.gangAllocate(1, 2, res, true /* strict_pack */);
    auto alloc_resp = alloc_fut.get();

    if (strlen(alloc_resp.ticket_id) == 0) {
        std::cerr << "[Smoke v7] Allocation Failed!" << std::endl;
        return 1;
    }

    std::cout << "[Smoke v7] Allocation Success! Ticket: " << alloc_resp.ticket_id << std::endl;
    std::cout << "  - Members: " << alloc_resp.num_members << std::endl;

    // (In real usage, we would parse variable length details here,
    // but for now we trust the AllocResp structure handling in Client)

    // 2. Launch Actors
    std::cout << "[Smoke v7] Launching Actors..." << std::endl;
    for (uint32_t i = 0; i < alloc_resp.num_members; ++i) {
        // Construct args for each rank
        std::vector<std::string> args;
        args.push_back("--rank");
        args.push_back(std::to_string(i));

        std::string actor_id = "v2_test_" + std::to_string(i);

        // Launch using the ticket
        // Note: For arguments, we just send empty body for now as V2TestActor doesn't parse them yet
        client.launchActor(alloc_resp.ticket_id, i, "V2TestActor", actor_id, "");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "[Smoke v7] All actors launched! Check Agent logs for GPU isolation output." << std::endl;

    return 0;
}

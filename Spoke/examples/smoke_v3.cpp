// examples/smoke_v3.cpp
#include "spoke/csrc/client.h"
#include <iostream>
#include <thread>
#include <vector>

const spoke::Action kAddTen = (spoke::Action)0x11;
const spoke::Action kSquare = (spoke::Action)0x12;

int main(int argc, char** argv)
{
    std::string hub_ip   = "127.0.0.1";
    int         hub_port = 8888;

    std::cout << "[Orchestrator] Connecting to Hub at " << hub_ip << ":" << hub_port << "..." << std::endl;

    spoke::Client client_alpha(hub_ip, hub_port, true);
    spoke::Client client_beta(hub_ip, hub_port, true);

    std::cout << "\n[Orchestrator] Spawning Agent Alpha..." << std::endl;
    client_alpha.spawnRemote("WorkerActor", "agent_alpha");

    std::cout << "[Orchestrator] Spawning Agent Beta..." << std::endl;
    client_beta.spawnRemote("WorkerActor", "agent_beta");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    double initial_value = 50.0;
    std::cout << "\n[Orchestrator] Starting Pipeline: (" << initial_value << " + 10)^2" << std::endl;

    std::cout << "  -> Sending " << initial_value << " to Agent Alpha (AddTen)..." << std::endl;
    auto f1 = client_alpha.callRemote<double, double>("agent_alpha", kAddTen, initial_value);

    double mid_result = f1.get();
    std::cout << "  <- Alpha returned: " << mid_result << std::endl;

    std::cout << "  -> Sending " << mid_result << " to Agent Beta (Square)..." << std::endl;
    auto f2 = client_beta.callRemote<double, double>("agent_beta", kSquare, mid_result);

    double final_result = f2.get();
    std::cout << "  <- Beta returned: " << final_result << std::endl;

    std::cout << "\n[Orchestrator] Starting Parallel Stress Test..." << std::endl;

    auto p1 = client_alpha.callRemote<double, double>("agent_alpha", kAddTen, 100.0);
    auto p2 = client_beta.callRemote<double, double>("agent_beta", kSquare, 5.0);

    std::cout << "  -> Tasks sent to both agents simultaneously." << std::endl;

    double r1 = p1.get();
    double r2 = p2.get();

    std::cout << "  <- Parallel Results: " << r1 << " & " << r2 << std::endl;

    return 0;
}

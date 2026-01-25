#include <iostream>
#include <thread>

#include "spoke/csrc/client.h"

const spoke::Action kAddTen = (spoke::Action)0x11;
const spoke::Action kSquare = (spoke::Action)0x12;

int main(int argc, char** argv)
{
    std::string ip = "127.0.0.1";
    if (argc > 1)
        ip = argv[1];

    std::cout << "[Client] Connecting to Daemon at " << ip << "..." << std::endl;
    spoke::Client client(ip, 9000);

    std::cout << "[Client] Spawning remote actor..." << std::endl;
    client.spawnRemote("WorkerActor", "gpu_node_1");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "[Client] Sending tasks..." << std::endl;

    auto f1 = client.callRemote<double, double>("gpu_node_1", kAddTen, 100.0);
    auto f2 = client.callRemote<double, double>("gpu_node_1", kSquare, 8.0);

    std::cout << "[Client] Waiting results..." << std::endl;

    std::cout << "  100 + 10 = " << f1.get() << std::endl;
    std::cout << "  8 * 8    = " << f2.get() << std::endl;

    return 0;
}

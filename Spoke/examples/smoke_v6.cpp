#include "spoke/csrc/client.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

struct ComplexTask {
    int                 id;
    std::string         name;
    std::vector<double> values;
};

namespace spoke {
template<>
struct Serializer<ComplexTask> {
    static std::string pack(const ComplexTask& t)
    {
        std::stringstream ss;
        ss << t.id << "|" << t.name << "|";
        for (double v : t.values)
            ss << v << ",";
        return ss.str();
    }
    static ComplexTask unpack(const std::string& s)
    {
        return {};  // Stub
    }

    // [New] Zero-Copy Interface
    static size_t size(const ComplexTask& t)
    {
        return pack(t).size();
    }

    static void packTo(const ComplexTask& t, char* buf)
    {
        std::string s = pack(t);
        memcpy(buf, s.data(), s.size());
    }

    static ComplexTask unpackFrom(const char* buf, size_t size)
    {
        return unpack(std::string(buf, size));
    }
};
}  // namespace spoke

const spoke::Action kProcessTask = (spoke::Action)0x60;

int main()
{
    std::string hub_ip   = "127.0.0.1";
    int         hub_port = 8888;

    std::cout << "[Smoke v6] Connecting to Hub..." << std::endl;
    spoke::Client client(hub_ip, hub_port, true);

    std::cout << "[Smoke v6] Spawning AdvancedActor..." << std::endl;
    client.spawnRemote("AdvancedActor", "adv_6");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Explicitly Init RDMA
    std::cout << "[Smoke v6] Initializing RDMA..." << std::endl;
    if (!client.initRDMA("adv_6")) {
        std::cerr << "[Smoke v6] Failed to initialize RDMA! Aborting." << std::endl;
        return 1;
    }
    std::cout << "[Smoke v6] RDMA Initialized." << std::endl;

    ComplexTask task;
    task.id     = 666;
    task.name   = "ZeroCopyCheck";
    task.values = {10.0, 20.0, 30.0};

    std::cout << "[Smoke v6] Calling remote with Complex Object via RDMA..." << std::endl;

    auto fut = client.callRemote<ComplexTask, std::string>("adv_6", kProcessTask, task);

    std::cout << "[Smoke v6] Result: " << fut.get() << std::endl;

    return 0;
}

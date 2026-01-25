#include "spoke/csrc/client.h"
#include <iostream>

struct Vec3 {
    double x, y, z;
};
struct PhysicsReq {
    int  id;
    Vec3 pos;
    Vec3 velocity;
};
struct PhysicsResp {
    int    id;
    Vec3   new_pos;
    double energy;
};

const spoke::Action kUpdatePhysics = (spoke::Action)0x50;

int main()
{
    std::cout << "[V4] Connecting..." << std::endl;
    spoke::Client client("127.0.0.1", 8888);

    std::cout << "[V4] Spawning PhysicsActor..." << std::endl;
    client.spawnRemote("PhysicsActor", "phys_1");

    PhysicsReq req;
    req.id       = 101;
    req.pos      = {0.0, 0.0, 0.0};
    req.velocity = {1.0, 2.0, 3.0};

    std::cout << "[V4] Sending Physics Request (struct)..." << std::endl;

    auto future = client.callRemote<PhysicsReq, PhysicsResp>("phys_1", kUpdatePhysics, req);

    PhysicsResp res = future.get();

    std::cout << "[V4] Result Received:" << std::endl;
    std::cout << "  ID: " << res.id << std::endl;
    std::cout << "  New Pos: (" << res.new_pos.x << ", " << res.new_pos.y << ", " << res.new_pos.z << ")" << std::endl;
    std::cout << "  Energy: " << res.energy << std::endl;

    return 0;
}

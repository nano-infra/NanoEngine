// Path: examples/smoke_v1.cpp
#include "spoke/csrc/actor.h"
#include "spoke/csrc/agent.h"
#include <iostream>

using namespace spoke;

const Action kAddTen = (Action)0x11;
const Action kSquare = (Action)0x12;

class MyActor: public Actor {
public:
    MyActor(const std::string& id, int rx, int tx): Actor(id, rx, tx) {}

    SPOKE_METHOD(MyActor, addTen, kAddTen, double, double)
    {
        return val + 10.0;
    }

    SPOKE_METHOD(MyActor, square, kSquare, double, double)
    {
        return val * val;
    }
};

int main()
{
    Agent agent;

    std::string id = "v1_engine";
    agent.spawnActor<MyActor>(id);

    std::cout << "[Smoke V1] Firing tasks to " << id << "..." << std::endl;

    auto f1 = agent.callRemote<double, double>(id, kAddTen, 5.5);
    auto f2 = agent.callRemote<double, double>(id, kSquare, 4.0);

    std::cout << "[Smoke V1] 5.5 + 10 = " << f1.get() << std::endl;
    std::cout << "[Smoke V1] 4.0^2    = " << f2.get() << std::endl;

    return 0;
}

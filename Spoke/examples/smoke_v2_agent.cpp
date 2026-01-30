// examples/smoke_v2_agent.cpp
#include <cmath>

#include "spoke/csrc/actor.h"
#include "spoke/csrc/serializer.h"

const spoke::Action kAddTen = (spoke::Action)0x11;
const spoke::Action kSquare = (spoke::Action)0x12;

class WorkerActor: public spoke::Actor {
public:
    WorkerActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    SPOKE_METHOD(WorkerActor, addTen, kAddTen, double, double)
    {
        return val + 10.0;
    }

    SPOKE_METHOD(WorkerActor, square, kSquare, double, double)
    {
        return val * val;
    }
};

SPOKE_REGISTER_ACTOR("WorkerActor", WorkerActor);

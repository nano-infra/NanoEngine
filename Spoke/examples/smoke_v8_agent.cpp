// smoke_v8_agent.cpp - Multi-Actor RDMA Test Agent
#include "spoke/csrc/actor.h"
#include <iostream>
#include <string>

// An actor that performs simple arithmetic to verify RDMA data integrity
class RdmaMathActor: public spoke::Actor {
public:
    using spoke::Actor::Actor;

    RdmaMathActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx)
    {
        std::cout << "[RdmaMathActor] " << id << " started and ready for RDMA." << std::endl;
    }

    // SPOKE_METHOD for square operation
    // Explicitly cast the ID to spoke::Action to match the enum class
    SPOKE_METHOD(RdmaMathActor, square, static_cast<spoke::Action>(0x10), int, int)
    {
        std::cout << "[RdmaMathActor] Squaring: " << val << std::endl;
        return val * val;
    }

    // SPOKE_METHOD for identity/ping operation
    SPOKE_METHOD(RdmaMathActor, ping, static_cast<spoke::Action>(0x11), std::string, std::string)
    {
        std::cout << "[RdmaMathActor] Ping received: " << val << std::endl;
        return "Pong: " + val;
    }

    // Removed the non-existent 'registerMethods' override
};

SPOKE_REGISTER_ACTOR("RdmaMathActor", RdmaMathActor);

// smoke_v7_agent.cpp - Dummy Actor for V2 Test

#include "spoke/csrc/actor.h"
#include <iostream>
#include <string>

// Just a dummy actor that prints its environment variables
class V2TestActor: public spoke::Actor {
public:
    using spoke::Actor::Actor;

    V2TestActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx)
    {
        std::cout << "=== V2 Test Actor Started ===" << std::endl;
        const char* rank = getenv("RANK");
        std::cout << "Rank: " << (rank ? rank : "Unset") << std::endl;
        const char* cuda_dev = getenv("CUDA_VISIBLE_DEVICES");
        std::cout << "CUDA_VISIBLE_DEVICES: " << (cuda_dev ? cuda_dev : "Unset") << std::endl;
    }
};

SPOKE_REGISTER_ACTOR("V2TestActor", V2TestActor);

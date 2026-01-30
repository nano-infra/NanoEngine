#include <iostream>

#include "spoke/csrc/actor.h"
#include "spoke/csrc/serializer.h"

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

class PhysicsActor: public spoke::Actor {
public:
    using spoke::Actor::Actor;

    SPOKE_METHOD(PhysicsActor, update, kUpdatePhysics, PhysicsReq, PhysicsResp)
    {
        PhysicsResp resp;
        resp.id        = val.id;
        resp.new_pos.x = val.pos.x + val.velocity.x;
        resp.new_pos.y = val.pos.y + val.velocity.y;
        resp.new_pos.z = val.pos.z + val.velocity.z;

        double v2 = val.velocity.x * val.velocity.x + val.velocity.y * val.velocity.y + val.velocity.z * val.velocity.z;
        resp.energy = 0.5 * v2;

        return resp;
    }
};

SPOKE_REGISTER_ACTOR("PhysicsActor", PhysicsActor);

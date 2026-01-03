#include "spoke/actor.h"

#include "nanodeploy/worker/dummy_runner.h"
#include "nanodeploy/worker/dummy_runner_ipc.h"

using namespace nanodeploy;

// Define Actor Wrapper
class DummyRunnerActor: public spoke::Actor {
public:
    DummyRunnerActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    // Wrapped method exposed via Spoke
    SPOKE_METHOD(DummyRunnerActor, run, spoke::Action::kUserActionStart, RunReq, RunResp)
    {
        return runner_.run(val);
    }

private:
    DummyRunner runner_;
};

// Register Actor Type
SPOKE_REGISTER_ACTOR("DummyRunner", DummyRunnerActor)

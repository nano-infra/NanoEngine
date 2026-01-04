#include "spoke/csrc/actor.h"

#include <memory>
#include <string>

#include "nanodeploy/logging.h"
#include "nanodeploy/worker/dummy_runner.h"
#include "nanodeploy/worker/dummy_runner_ipc.h"

using namespace nanodeploy;

// Define Actor Wrapper
class DummyRunnerActor: public spoke::Actor {
public:
    DummyRunnerActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    // Implement methods directly inside SPOKE_METHOD to avoid declaration conflicts
    // AND to implement the wrapper logic. Use 'val' as input argument name.

    SPOKE_METHOD(DummyRunnerActor, run, spoke::Action::kUserActionStart, nanodeploy::RunReq, nanodeploy::RunResp)
    {
        return runner_.run(val);
    }

    SPOKE_METHOD(DummyRunnerActor,
                 init_distributed,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 1),
                 nanodeploy::DistConfig,
                 nanodeploy::DistResp)
    {
        return runner_.init_distributed(val);
    }

    SPOKE_METHOD(DummyRunnerActor,
                 run_allreduce,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 2),
                 nanodeploy::RunAllReduceReq,
                 nanodeploy::RunResp)
    {
        return runner_.run_allreduce();
    }

private:
    nanodeploy::DummyRunner runner_;
};

// Register Actor Type
SPOKE_REGISTER_ACTOR("DummyRunner", DummyRunnerActor)

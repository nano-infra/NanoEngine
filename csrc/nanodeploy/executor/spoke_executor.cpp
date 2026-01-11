#include "spoke/csrc/actor.h"

#include <memory>
#include <string>

#include "nanodeploy/logging.h"
#include "nanodeploy/worker/deep_ep_runner.h"
#include "nanodeploy/worker/deep_gemm_ipc.h"
#include "nanodeploy/worker/deep_gemm_runner.h"
#include "nanodeploy/worker/dummy_runner.h"
#include "nanodeploy/worker/dummy_runner_ipc.h"
#include "nanodeploy/worker/model_runner.h"
#include "nanodeploy/worker/model_runner_ipc.h"

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

// Define ModelRunner Actor
class ModelRunnerActor: public spoke::Actor {
public:
    ModelRunnerActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    SPOKE_METHOD(ModelRunnerActor,
                 init,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 10),
                 nanodeploy::ModelInitReq,
                 nanodeploy::ModelInitResp)
    {
        std::cout << "[Executor] Received Init Request. Calling runner_.init()..." << std::endl;

        nanodeploy::DistributedConfig dconf;
        dconf.global_rank = val.rank;
        dconf.world_size  = val.world_size;
        dconf.tp_degree   = val.tp_degree;
        dconf.pp_degree   = val.pp_degree;
        dconf.dp_degree   = val.dp_degree;

        runner_.init(val.config_path, dconf);
        std::cout << "[Executor] runner_.init() returned. Sending response..." << std::endl;
        return true;
    }

    SPOKE_METHOD(ModelRunnerActor,
                 run,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 11),
                 nanodeploy::ModelRunReq,
                 nanodeploy::ModelRunResp)
    {
        return {runner_.run(val)};
    }

private:
    nanodeploy::ModelRunner runner_;
};

// Define DeepEPTest Actor
class DeepEPTestActor: public spoke::Actor {
public:
    DeepEPTestActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    SPOKE_METHOD(DeepEPTestActor,
                 init,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 20),
                 nanodeploy::DeepEPInitReq,
                 nanodeploy::DeepEPInitResp)
    {
        return runner_.init(val);
    }

    SPOKE_METHOD(DeepEPTestActor,
                 get_info,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 21),
                 nanodeploy::DeepEPInfoReq,
                 nanodeploy::DeepEPInfoResp)
    {
        return runner_.get_info();
    }

    SPOKE_METHOD(DeepEPTestActor,
                 sync,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 22),
                 nanodeploy::DeepEPSyncReq,
                 nanodeploy::DeepEPSyncResp)
    {
        return runner_.sync(val);
    }

    SPOKE_METHOD(DeepEPTestActor,
                 run_test,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 23),
                 nanodeploy::DeepEPTestReq,
                 nanodeploy::DeepEPTestResp)
    {
        return runner_.run_test(val);
    }

private:
    nanodeploy::DeepEPRunner runner_;
};

// Define DeepGemmTest Actor
class DeepGemmTestActor: public spoke::Actor {
public:
    DeepGemmTestActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx) {}

    SPOKE_METHOD(DeepGemmTestActor,
                 init,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 30),
                 nanodeploy::DeepGemmInitReq,
                 nanodeploy::DeepGemmInitResp)
    {
        return runner_.init(val);
    }

    SPOKE_METHOD(DeepGemmTestActor,
                 run_test,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 31),
                 nanodeploy::DeepGemmTestReq,
                 nanodeploy::DeepGemmTestResp)
    {
        return runner_.run_test(val);
    }

private:
    nanodeploy::DeepGemmRunner runner_;
};

// Register Actor Type
SPOKE_REGISTER_ACTOR("DummyRunner", DummyRunnerActor)
SPOKE_REGISTER_ACTOR("ModelRunner", ModelRunnerActor)
SPOKE_REGISTER_ACTOR("DeepEPTestActor", DeepEPTestActor)
SPOKE_REGISTER_ACTOR("DeepGemmTestActor", DeepGemmTestActor)

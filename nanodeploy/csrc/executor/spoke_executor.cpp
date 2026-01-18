#include "spoke/csrc/actor.h"

#include <memory>
#include <string>

#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/engine/engine_actor.h"
#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/worker/dummy_runner.h"
#include "nanodeploy/csrc/worker/dummy_runner_ipc.h"
#include "nanodeploy/csrc/worker/model_runner_ipc.h"

#include "nanodeploy/csrc/worker/model_runner.h"

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
        dconf.pp_degree   = val.pp_degree;

        dconf.attention_tp      = val.attention_tp;
        dconf.attention_dp      = val.attention_dp;
        dconf.attention_sp      = val.attention_sp;
        dconf.ffn_tp            = val.ffn_tp;
        dconf.ffn_dp            = val.ffn_dp;
        dconf.ffn_ep            = val.ffn_ep;
        dconf.enable_cuda_graph = val.enable_cuda_graph;

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
        NANODEPLOY_LOG_DEBUG("[Executor] Received Run Request");
        return {runner_.run(val)};
    }

    // DeepEP sync: get local info
    SPOKE_METHOD(ModelRunnerActor,
                 getDeepEpInfo,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 12),
                 int,  // dummy request type
                 nanodeploy::DeepEpInfoResp)
    {
        (void)val;  // unused
        NANODEPLOY_LOG_DEBUG("[Executor] Received GetDeepEpInfo Request");
        return runner_.getDeepEpInfo();
    }

    // DeepEP sync: sync all handles
    SPOKE_METHOD(ModelRunnerActor,
                 syncDeepEp,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 13),
                 nanodeploy::DeepEpSyncReq,
                 bool)
    {
        NANODEPLOY_LOG_DEBUG("[Executor] Received SyncDeepEp Request");
        return runner_.syncDeepEp(val);
    }

    // Init KV Cache
    SPOKE_METHOD(ModelRunnerActor,
                 init_kv_cache,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 14),
                 nanodeploy::KvCacheInitReq,
                 nanodeploy::KvCacheInitResp)
    {
        NANODEPLOY_LOG_INFO("[Executor] Received InitKvCache Request");
        return runner_.init_kv_cache(val);
    }

    SPOKE_METHOD(ModelRunnerActor,
                 capture_decode_graphs,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 15),
                 nanodeploy::GraphCaptureReq,
                 nanodeploy::GraphCaptureResp)
    {
        NANODEPLOY_LOG_INFO("[Executor] Received CaptureDecodeGraphs Request");
        return runner_.capture_decode_graphs(val);
    }

    // Warmup MoE (standalone)
    SPOKE_METHOD(ModelRunnerActor,
                 warmup_moe,
                 static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 16),
                 int,  // placeholder input, not used
                 bool)
    {
        (void)val;  // unused
        NANODEPLOY_LOG_INFO("[Executor] Received WarmupMoe Request");
        return runner_.warmup_moe();
    }

private:
    nanodeploy::ModelRunner runner_;
};

// Register Actor Type
SPOKE_REGISTER_ACTOR("DummyRunner", DummyRunnerActor)
SPOKE_REGISTER_ACTOR("ModelRunner", ModelRunnerActor)

// EngineActor is in nanodeploy namespace, create alias for macro
using EngineActorImpl = nanodeploy::EngineActor;
SPOKE_REGISTER_ACTOR("EngineActor", EngineActorImpl)

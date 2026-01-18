#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {

void DistributedContext::init(const DistributedConfig& config)
{
    if (initialized_) {
        NANODEPLOY_LOG_WARN("DistributedContext already initialized. Overwriting config.");
    }

    config_      = config;
    initialized_ = true;

    NANODEPLOY_LOG_INFO("DistributedContext Initialized: Rank=", config.global_rank, " WorldSize=", config.world_size);
    NANODEPLOY_LOG_INFO(
        "  Attention: TP=", config.attention_tp, " DP=", config.attention_dp, " SP=", config.attention_sp);
    NANODEPLOY_LOG_INFO("  FFN: TP=", config.ffn_tp, " DP=", config.ffn_dp, " EP=", config.ffn_ep);
}

int DistributedContext::pp_rank() const
{
    // PP is outermost dimension
    int attention_group_size = attention_tp() * attention_dp();
    return config_.global_rank / attention_group_size;
}

int DistributedContext::attention_tp_rank() const
{
    // TP is innermost dimension within attention group
    return config_.global_rank % attention_tp();
}

int DistributedContext::attention_dp_rank() const
{
    // DP is next dimension after TP
    return (config_.global_rank / attention_tp()) % attention_dp();
}

int DistributedContext::ffn_tp_rank() const
{
    // For FFN, TP rank calculation (assuming same layout)
    return config_.global_rank % ffn_tp();
}

int DistributedContext::ffn_dp_rank() const
{
    return (config_.global_rank / ffn_tp()) % ffn_dp();
}

int DistributedContext::ffn_ep_rank() const
{
    // EP rank: global_rank modulo expert parallelism degree
    return config_.global_rank % ffn_ep();
}

}  // namespace nanodeploy

#include "distributed.h"
#include "nanodeploy/logging.h"
#include <stdexcept>

namespace nanodeploy {

DistContext& get_dist_context()
{
    return DistContext::instance();
}

void DistContext::init(const DistributedConfig& config)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (initialized_) {
        // Allow re-init if config is same? Or warn.
        NANODEPLOY_LOG_WARN("DistContext already initialized. Overwriting config.");
    }

    config_      = config;
    initialized_ = true;

    NANODEPLOY_LOG_INFO("DistContext Initialized: Rank=", config.global_rank);
    NANODEPLOY_LOG_INFO("WorldSize=", config.world_size, " TP=", config.tp_degree);

    // Basic validation
    int total = config.tp_degree * config.pp_degree * config.dp_degree;  // Simplified (ignoring EP for now)
    if (total != config.world_size) {
        // Soft warning for now as world_size might include other auxiliary processes or logic might vary
        NANODEPLOY_LOG_WARN("DistContext Config: TP*PP*DP (", total, ") != WorldSize (", config.world_size, ")");
    }
}

// Assuming layout: [PP, DP, TP] where TP is innermost (contiguous ranks)
// global_rank = pp_idx * (dp * tp) + dp_idx * tp + tp_idx

int DistContext::tp_rank() const
{
    // TP is the finest granularity
    return config_.global_rank % config_.tp_degree;
}

int DistContext::dp_rank() const
{
    // Remove TP component
    int rank_div_tp = config_.global_rank / config_.tp_degree;
    return rank_div_tp % config_.dp_degree;
}

int DistContext::pp_rank() const
{
    // Remove TP and DP components
    int rank_div_tp_dp = config_.global_rank / (config_.tp_degree * config_.dp_degree);
    return rank_div_tp_dp % config_.pp_degree;
}

int DistContext::ep_rank() const
{
    // EP usually partitions the model at the level of TP groups (experts are not usually TP sliced in basic EP)
    // So all TP ranks in a group belong to the same EP rank.
    return (config_.global_rank / config_.tp_degree) % config_.ep_degree;
}

}  // namespace nanodeploy
